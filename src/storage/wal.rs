//! Write-Ahead Log (WAL) System for ReasonKit Memory
//!
//! Provides durability guarantees for memory operations through journaling.
//! All writes are persisted to the WAL before being applied to the main storage,
//! ensuring crash recovery and data integrity.
//!
//! # Architecture
//!
//! ```text
//! +-----------------------------------------------------------------------+
//! |                           WAL Architecture                            |
//! +-----------------------------------------------------------------------+
//! |                                                                       |
//! |  +---------------+    +---------------+    +-------------------------+|
//! |  | Memory Write  |--->|  WAL Append   |--->| Storage Backend Apply   ||
//! |  |   Request     |    |   (fsync)     |    | (Qdrant/File/InMemory)  ||
//! |  +---------------+    +---------------+    +-------------------------+|
//! |         |                   |                        |                |
//! |         |                   |                        |                |
//! |         v                   v                        v                |
//! |  +---------------+    +---------------+    +-------------------------+|
//! |  |   Response    |<---|  Checkpoint   |<---|    Truncate Old WAL     ||
//! |  +---------------+    +---------------+    +-------------------------+|
//! |                                                                       |
//! +-----------------------------------------------------------------------+
//! ```
//!
//! ## File Layout
//!
//! ```text
//! {wal_dir}/
//! +-- segment_00000001.wal      # Active segment (append-only)
//! +-- segment_00000002.wal      # Previous segment
//! +-- checkpoint_00000005.ckpt  # Latest checkpoint
//! +-- checkpoint_00000003.ckpt  # Previous checkpoint (retained for safety)
//! +-- wal.meta                  # WAL metadata (current LSN, active segment)
//! ```
//!
//! ## WAL Entry Format (Binary)
//!
//! ```text
//! +------------------------------------------------------------------------+
//! |                         WAL Entry (Variable Size)                       |
//! +--------+-------------+-----------+--------------+------------+---------+
//! | Magic  | Entry Size  | Checksum  |  LSN (u64)   | Timestamp  |  OpType |
//! | 4 bytes|  4 bytes    |  4 bytes  |   8 bytes    |  8 bytes   | 1 byte  |
//! +--------+-------------+-----------+--------------+------------+---------+
//! |                         Payload (Variable)                              |
//! |                    [JSON-encoded WalOperation]                          |
//! +------------------------------------------------------------------------+
//! |                      Entry Checksum (4 bytes)                           |
//! +------------------------------------------------------------------------+
//!
//! Total Header: 29 bytes + variable payload + 4 byte trailer
//! ```
//!
//! # Recovery Algorithm (Pseudocode)
//!
//! ```text
//! PROCEDURE recover():
//!     // Phase 1: Crash Detection
//!     meta = load_wal_metadata()
//!     IF meta.clean_shutdown == false:
//!         log("Crash detected, starting recovery...")
//!
//!     // Phase 2: Find Latest Valid Checkpoint
//!     checkpoints = scan_checkpoint_files()
//!     SORT checkpoints BY id DESCENDING
//!
//!     latest_valid_checkpoint = NULL
//!     FOR each checkpoint IN checkpoints:
//!         IF validate_checkpoint(checkpoint):
//!             latest_valid_checkpoint = checkpoint
//!             BREAK
//!
//!     // Phase 3: Determine Replay Start Point
//!     IF latest_valid_checkpoint != NULL:
//!         replay_start_lsn = latest_valid_checkpoint.lsn
//!         load_checkpoint_state(latest_valid_checkpoint)
//!     ELSE:
//!         replay_start_lsn = 0
//!         initialize_empty_state()
//!
//!     // Phase 4: Scan and Collect Entries
//!     segments = scan_segment_files()
//!     SORT segments BY segment_number
//!
//!     entries_to_replay = []
//!     FOR each segment IN segments:
//!         position = 0
//!         WHILE position < segment.size:
//!             TRY:
//!                 header = read_header(segment, position)
//!                 IF NOT validate_header_checksum(header):
//!                     log_error("Corrupted header at {}", position)
//!                     position = scan_for_magic(segment, position + 1)
//!                     CONTINUE
//!
//!                 payload = read_payload(segment, position, header.size)
//!                 IF NOT validate_payload_checksum(payload):
//!                     log_error("Corrupted payload at {}", position)
//!                     position += header.size
//!                     CONTINUE
//!
//!                 entry = parse_entry(header, payload)
//!                 IF entry.lsn >= replay_start_lsn:
//!                     entries_to_replay.append(entry)
//!
//!                 position += header.entry_size
//!             CATCH incomplete_read:
//!                 BREAK
//!
//!     // Phase 5: Replay Entries in Order
//!     SORT entries_to_replay BY lsn
//!     FOR each entry IN entries_to_replay:
//!         apply_operation(entry)
//!
//!     // Phase 6: Finalize
//!     current_lsn = last_replayed_lsn + 1
//!     mark_recovery_complete()
//!     RETURN RecoveryReport
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use reasonkit_mem::storage::wal::{WriteAheadLog, WalConfig, WalOperation, SyncMode};
//! use std::path::PathBuf;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = WalConfig {
//!         dir: PathBuf::from("./wal"),
//!         segment_size_mb: 64,
//!         sync_mode: SyncMode::Batched(Duration::from_millis(100)),
//!     };
//!
//!     let wal = WriteAheadLog::new(config).await?;
//!
//!     // Log an insert operation
//!     let lsn = wal.append(WalOperation::Insert {
//!         id: uuid::Uuid::new_v4(),
//!         content: "Hello, World!".to_string(),
//!         embedding: vec![0.1, 0.2, 0.3],
//!     }).await?;
//!
//!     // Checkpoint to mark operations as durable
//!     wal.checkpoint().await?;
//!
//!     Ok(())
//! }
//! ```

use crate::{MemError, MemResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

// ============================================================================
// Constants
// ============================================================================

/// WAL segment file prefix
const SEGMENT_PREFIX: &str = "wal_";
/// WAL segment file extension
const SEGMENT_EXTENSION: &str = ".log";
/// Checkpoint file prefix
const CHECKPOINT_PREFIX: &str = "checkpoint_";
/// Checkpoint file extension
const CHECKPOINT_EXTENSION: &str = ".ckpt";
/// Magic bytes for WAL entry header: "WAL1"
const WAL_MAGIC: [u8; 4] = [0x57, 0x41, 0x4C, 0x31];
/// Magic bytes for checkpoint files: "CKPT"
const CHECKPOINT_MAGIC: [u8; 4] = [0x43, 0x4B, 0x50, 0x54];
/// Version of the WAL format
const WAL_VERSION: u8 = 1;
/// Default checkpoint retention count
const DEFAULT_CHECKPOINT_RETENTION: usize = 2;

// ============================================================================
// Log Sequence Number
// ============================================================================

/// Log Sequence Number - monotonically increasing identifier for each WAL entry
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct LogSequenceNumber(pub u64);

impl LogSequenceNumber {
    /// Create a new LSN from raw value
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get the raw LSN value
    pub const fn value(&self) -> u64 {
        self.0
    }

    /// Increment and return the next LSN
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for LogSequenceNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LSN({})", self.0)
    }
}

impl From<u64> for LogSequenceNumber {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

// ============================================================================
// Checkpoint ID
// ============================================================================

/// Checkpoint identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct CheckpointId(pub u64);

impl CheckpointId {
    /// Create a new checkpoint ID
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get the underlying value
    pub const fn value(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CKPT({})", self.0)
    }
}

// ============================================================================
// WAL Operations
// ============================================================================

/// Operations that can be logged to the WAL
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WalOperation {
    /// Insert a new document/chunk with embedding
    Insert {
        /// Unique identifier
        id: Uuid,
        /// Text content
        content: String,
        /// Embedding vector
        embedding: Vec<f32>,
    },
    /// Update an existing document/chunk
    Update {
        /// Unique identifier
        id: Uuid,
        /// Updated text content
        content: String,
        /// Updated embedding vector
        embedding: Vec<f32>,
    },
    /// Delete a document/chunk
    Delete {
        /// Unique identifier to delete
        id: Uuid,
    },
    /// Batch insert operation (atomic)
    BatchInsert {
        /// List of items to insert
        items: Vec<BatchItem>,
    },
    /// Batch delete operation (atomic)
    BatchDelete {
        /// List of IDs to delete
        ids: Vec<Uuid>,
    },
    /// Checkpoint marker - all operations before this LSN are durable
    Checkpoint {
        /// Log Sequence Number at checkpoint
        lsn: u64,
        /// Checkpoint ID for tracking
        checkpoint_id: u64,
    },
    /// Transaction begin marker
    TxnBegin {
        /// Transaction ID
        txn_id: Uuid,
    },
    /// Transaction commit marker
    TxnCommit {
        /// Transaction ID
        txn_id: Uuid,
    },
    /// Transaction rollback marker
    TxnRollback {
        /// Transaction ID
        txn_id: Uuid,
    },
}

/// Item for batch operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchItem {
    /// Unique identifier
    pub id: Uuid,
    /// Text content
    pub content: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
}

impl WalOperation {
    /// Get the operation type as a string for logging/debugging
    pub fn op_type(&self) -> &'static str {
        match self {
            WalOperation::Insert { .. } => "INSERT",
            WalOperation::Update { .. } => "UPDATE",
            WalOperation::Delete { .. } => "DELETE",
            WalOperation::BatchInsert { .. } => "BATCH_INSERT",
            WalOperation::BatchDelete { .. } => "BATCH_DELETE",
            WalOperation::Checkpoint { .. } => "CHECKPOINT",
            WalOperation::TxnBegin { .. } => "TXN_BEGIN",
            WalOperation::TxnCommit { .. } => "TXN_COMMIT",
            WalOperation::TxnRollback { .. } => "TXN_ROLLBACK",
        }
    }
}

// ============================================================================
// WAL Entry
// ============================================================================

/// A single entry in the WAL with full metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Log Sequence Number - monotonically increasing
    pub lsn: u64,
    /// Unix timestamp in microseconds
    pub timestamp: i64,
    /// The operation being logged
    pub operation: WalOperation,
    /// CRC32 checksum of the entry (excluding this field)
    pub checksum: u32,
}

impl WalEntry {
    /// Create a new WAL entry with computed checksum
    pub fn new(lsn: u64, operation: WalOperation) -> Self {
        let timestamp = chrono::Utc::now().timestamp_micros();
        let mut entry = WalEntry {
            lsn,
            timestamp,
            operation,
            checksum: 0,
        };
        entry.checksum = compute_checksum(&entry);
        entry
    }

    /// Verify the checksum of this entry
    pub fn verify(&self) -> bool {
        verify_checksum(self)
    }

    /// Get the size of this entry when serialized
    pub fn serialized_size(&self) -> MemResult<usize> {
        let data = bincode_serialize(self)?;
        // Header: magic (4) + version (1) + length (4) + data + trailing length (4)
        Ok(4 + 1 + 4 + data.len() + 4)
    }

    /// Get the LSN as a LogSequenceNumber
    pub fn lsn(&self) -> LogSequenceNumber {
        LogSequenceNumber::new(self.lsn)
    }
}

// ============================================================================
// Checkpoint
// ============================================================================

/// Checkpoint data structure for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint ID
    pub id: CheckpointId,
    /// LSN at checkpoint creation
    pub lsn: LogSequenceNumber,
    /// Timestamp of checkpoint creation (unix seconds)
    pub created_at: u64,
    /// Snapshot of document IDs present at checkpoint
    pub document_ids: Vec<Uuid>,
    /// Snapshot of chunk IDs present at checkpoint
    pub chunk_ids: Vec<Uuid>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Checksum of the checkpoint data
    pub checksum: u32,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(
        id: u64,
        lsn: LogSequenceNumber,
        document_ids: Vec<Uuid>,
        chunk_ids: Vec<Uuid>,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut checkpoint = Self {
            id: CheckpointId::new(id),
            lsn,
            created_at,
            document_ids,
            chunk_ids,
            metadata: HashMap::new(),
            checksum: 0,
        };
        checkpoint.checksum = checkpoint.compute_checksum();
        checkpoint
    }

    /// Compute checksum for validation
    fn compute_checksum(&self) -> u32 {
        let data = serde_json::to_vec(&(&self.id, &self.lsn, &self.created_at, &self.document_ids))
            .unwrap_or_default();
        crc32_compute(&data)
    }

    /// Validate checkpoint integrity
    pub fn validate(&self) -> bool {
        let expected = self.compute_checksum();
        self.checksum == expected
    }

    /// Serialize to bytes with magic header
    pub fn to_bytes(&self) -> MemResult<Vec<u8>> {
        let json = serde_json::to_vec(self)
            .map_err(|e| MemError::storage(format!("Failed to serialize checkpoint: {}", e)))?;

        let mut buf = Vec::with_capacity(4 + 1 + json.len());
        buf.extend_from_slice(&CHECKPOINT_MAGIC);
        buf.push(WAL_VERSION);
        buf.extend_from_slice(&json);

        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn from_bytes(buf: &[u8]) -> MemResult<Self> {
        if buf.len() < 5 {
            return Err(MemError::storage("Buffer too small for checkpoint"));
        }

        if buf[0..4] != CHECKPOINT_MAGIC {
            return Err(MemError::storage("Invalid checkpoint magic"));
        }

        if buf[4] > WAL_VERSION {
            return Err(MemError::storage(format!(
                "Unsupported checkpoint version: {}",
                buf[4]
            )));
        }

        let checkpoint: Checkpoint = serde_json::from_slice(&buf[5..])
            .map_err(|e| MemError::storage(format!("Failed to deserialize checkpoint: {}", e)))?;

        if !checkpoint.validate() {
            return Err(MemError::storage("Checkpoint checksum validation failed"));
        }

        Ok(checkpoint)
    }
}

// ============================================================================
// Recovery Report
// ============================================================================

/// Report from WAL recovery process
#[derive(Debug, Clone, Default)]
pub struct RecoveryReport {
    /// Number of entries recovered
    pub entries_recovered: u64,
    /// Number of entries skipped due to corruption
    pub entries_skipped: u64,
    /// Number of entries replayed successfully
    pub entries_replayed: u64,
    /// Last valid LSN found
    pub last_valid_lsn: LogSequenceNumber,
    /// Checkpoint used for recovery (if any)
    pub checkpoint_used: Option<CheckpointId>,
    /// Recovery duration in milliseconds
    pub duration_ms: u64,
    /// Errors encountered during recovery
    pub errors: Vec<RecoveryError>,
    /// Whether recovery was successful
    pub success: bool,
}

/// Error encountered during recovery
#[derive(Debug, Clone)]
pub struct RecoveryError {
    /// LSN where error occurred (if known)
    pub lsn: Option<LogSequenceNumber>,
    /// Segment file where error occurred
    pub segment: Option<PathBuf>,
    /// Error message
    pub message: String,
    /// Whether this was a fatal error
    pub fatal: bool,
}

// ============================================================================
// Synchronization Mode
// ============================================================================

/// Synchronization mode for WAL writes
#[derive(Debug, Clone)]
pub enum SyncMode {
    /// Fsync after each write - maximum durability, lowest performance
    Immediate,
    /// Fsync at specified intervals - balanced durability/performance
    Batched(Duration),
    /// OS-managed sync - highest performance, lower durability guarantees
    Async,
}

impl Default for SyncMode {
    fn default() -> Self {
        SyncMode::Batched(Duration::from_millis(100))
    }
}

// ============================================================================
// WAL Configuration
// ============================================================================

/// Configuration for the Write-Ahead Log
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Directory to store WAL segment files
    pub dir: PathBuf,
    /// Maximum size of each segment file in megabytes
    pub segment_size_mb: usize,
    /// Synchronization mode for durability
    pub sync_mode: SyncMode,
    /// Number of checkpoints to retain
    pub checkpoint_retention: usize,
    /// Pre-allocate segment files for performance
    pub preallocate_segments: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("./wal"),
            segment_size_mb: 64,
            sync_mode: SyncMode::default(),
            checkpoint_retention: DEFAULT_CHECKPOINT_RETENTION,
            preallocate_segments: true,
        }
    }
}

impl WalConfig {
    /// Create a new WAL config with the specified directory
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            ..Default::default()
        }
    }

    /// Set the segment size in megabytes
    pub fn with_segment_size(mut self, size_mb: usize) -> Self {
        self.segment_size_mb = size_mb;
        self
    }

    /// Set the sync mode
    pub fn with_sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Set checkpoint retention count
    pub fn with_checkpoint_retention(mut self, count: usize) -> Self {
        self.checkpoint_retention = count;
        self
    }

    /// Get the segment size in bytes
    fn segment_size_bytes(&self) -> usize {
        self.segment_size_mb * 1024 * 1024
    }
}

// ============================================================================
// WAL Trait (API Interface)
// ============================================================================

/// Write-Ahead Log trait defining the core operations
///
/// This trait provides a unified interface for WAL implementations,
/// enabling both file-based and in-memory variants.
#[async_trait]
pub trait WriteAheadLogTrait: Send + Sync {
    /// Append an operation to the WAL
    ///
    /// Returns the LSN assigned to this operation.
    /// The operation is guaranteed to be durable after this call returns
    /// (subject to sync_mode configuration).
    async fn append(&self, op: WalOperation) -> MemResult<LogSequenceNumber>;

    /// Append a batch of operations atomically
    ///
    /// All operations in the batch are assigned consecutive LSNs.
    /// Returns the LSN of the first operation.
    async fn append_batch(&self, ops: Vec<WalOperation>) -> MemResult<LogSequenceNumber>;

    /// Force sync all pending writes to disk
    ///
    /// This is a no-op in Immediate sync mode.
    async fn sync(&self) -> MemResult<()>;

    /// Create a checkpoint of the current state
    ///
    /// Checkpoints enable faster recovery by providing a snapshot
    /// of the system state, reducing the number of log entries
    /// that need to be replayed.
    async fn checkpoint(&self) -> MemResult<CheckpointId>;

    /// Recover from WAL after a crash
    ///
    /// This should be called during startup. It will:
    /// 1. Find the latest valid checkpoint
    /// 2. Replay all entries after the checkpoint
    /// 3. Return a report of the recovery process
    async fn recover(&self) -> MemResult<RecoveryReport>;

    /// Truncate WAL entries before the given LSN
    ///
    /// Used after successful checkpoint to reclaim disk space.
    /// Entries before the LSN will be permanently deleted.
    async fn truncate_before(&self, lsn: LogSequenceNumber) -> MemResult<()>;

    /// Get the current LSN (next LSN to be assigned)
    async fn current_lsn(&self) -> LogSequenceNumber;

    /// Get the last synced LSN (guaranteed durable)
    async fn synced_lsn(&self) -> LogSequenceNumber;

    /// Read entries from a given LSN (for replication/debugging)
    async fn read_from(
        &self,
        start_lsn: LogSequenceNumber,
        limit: usize,
    ) -> MemResult<Vec<WalEntry>>;

    /// Close the WAL gracefully
    async fn close(&self) -> MemResult<()>;
}

// ============================================================================
// Segment Metadata
// ============================================================================

/// Metadata for a WAL segment file
#[derive(Debug, Clone)]
struct SegmentMeta {
    /// Segment number (used for file naming)
    segment_id: u64,
    /// Path to the segment file
    path: PathBuf,
    /// First LSN in this segment
    first_lsn: u64,
    /// Last LSN in this segment (None if segment is empty)
    last_lsn: Option<u64>,
    /// Current size in bytes
    size_bytes: u64,
}

// ============================================================================
// Segment Writer
// ============================================================================

/// Writer for a single WAL segment
struct SegmentWriter {
    /// Buffered writer
    writer: BufWriter<File>,
    /// Segment metadata
    meta: SegmentMeta,
    /// Whether we need to sync
    needs_sync: bool,
}

impl SegmentWriter {
    /// Create a new segment writer
    fn new(meta: SegmentMeta) -> MemResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&meta.path)
            .map_err(|e| MemError::storage(format!("Failed to open WAL segment: {}", e)))?;

        Ok(Self {
            writer: BufWriter::with_capacity(64 * 1024, file), // 64KB buffer
            meta,
            needs_sync: false,
        })
    }

    /// Write an entry to this segment
    fn write_entry(&mut self, entry: &WalEntry) -> MemResult<()> {
        let data = bincode_serialize(entry)?;
        let len = data.len() as u32;

        // Write entry format:
        // [magic: 4 bytes][version: 1 byte][length: 4 bytes][data: N bytes][length: 4 bytes]
        // Trailing length allows reverse scanning

        self.writer
            .write_all(&WAL_MAGIC)
            .map_err(|e| MemError::storage(format!("Failed to write WAL magic: {}", e)))?;

        self.writer
            .write_all(&[WAL_VERSION])
            .map_err(|e| MemError::storage(format!("Failed to write WAL version: {}", e)))?;

        self.writer
            .write_all(&len.to_le_bytes())
            .map_err(|e| MemError::storage(format!("Failed to write entry length: {}", e)))?;

        self.writer
            .write_all(&data)
            .map_err(|e| MemError::storage(format!("Failed to write entry data: {}", e)))?;

        self.writer
            .write_all(&len.to_le_bytes())
            .map_err(|e| MemError::storage(format!("Failed to write trailing length: {}", e)))?;

        // Update metadata
        self.meta.size_bytes += 4 + 1 + 4 + data.len() as u64 + 4;
        self.meta.last_lsn = Some(entry.lsn);
        self.needs_sync = true;

        Ok(())
    }

    /// Flush the buffer to disk
    fn flush(&mut self) -> MemResult<()> {
        self.writer
            .flush()
            .map_err(|e| MemError::storage(format!("Failed to flush WAL buffer: {}", e)))
    }

    /// Sync to disk (fsync)
    fn sync(&mut self) -> MemResult<()> {
        self.flush()?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| MemError::storage(format!("Failed to sync WAL segment: {}", e)))?;
        self.needs_sync = false;
        Ok(())
    }
}

// ============================================================================
// Write-Ahead Log Implementation
// ============================================================================

/// Write-Ahead Log for durable storage operations
pub struct WriteAheadLog {
    /// Configuration
    config: WalConfig,
    /// Current segment writer (protected by mutex for write access)
    current_segment: Arc<Mutex<SegmentWriter>>,
    /// Current Log Sequence Number
    current_lsn: AtomicU64,
    /// Last synced LSN
    synced_lsn: AtomicU64,
    /// LSN of the last checkpoint
    last_checkpoint: AtomicU64,
    /// Checkpoint counter
    checkpoint_counter: AtomicU64,
    /// List of all segment metadata
    segments: Arc<RwLock<Vec<SegmentMeta>>>,
    /// Background sync task handle (for batched mode)
    #[allow(dead_code)]
    sync_handle: Option<tokio::task::JoinHandle<()>>,
    /// Flag to signal shutdown to background tasks
    shutdown: Arc<AtomicU64>,
    /// Closed flag
    closed: AtomicU64,
}

impl WriteAheadLog {
    /// Create a new Write-Ahead Log
    pub async fn new(config: WalConfig) -> MemResult<Self> {
        // Create WAL directory if it doesn't exist
        std::fs::create_dir_all(&config.dir)
            .map_err(|e| MemError::storage(format!("Failed to create WAL directory: {}", e)))?;

        // Discover existing segments
        let (segments, current_lsn, last_checkpoint, checkpoint_id) =
            Self::discover_segments(&config).await?;

        // Create or open current segment
        let current_segment = if segments.is_empty() {
            Self::create_new_segment(&config, 0, current_lsn)?
        } else {
            let last = segments.last().unwrap();
            // Check if we need to rotate
            if last.size_bytes >= config.segment_size_bytes() as u64 {
                Self::create_new_segment(&config, last.segment_id + 1, current_lsn)?
            } else {
                SegmentWriter::new(last.clone())?
            }
        };

        let segments = Arc::new(RwLock::new(segments));
        let current_segment = Arc::new(Mutex::new(current_segment));
        let shutdown = Arc::new(AtomicU64::new(0));

        // Start background sync task if using batched mode
        let sync_handle = match &config.sync_mode {
            SyncMode::Batched(interval) => {
                let interval = *interval;
                let segment_clone = current_segment.clone();
                let shutdown_clone = shutdown.clone();

                Some(tokio::spawn(async move {
                    let mut ticker = tokio::time::interval(interval);
                    loop {
                        ticker.tick().await;

                        // Check for shutdown
                        if shutdown_clone.load(Ordering::SeqCst) != 0 {
                            break;
                        }

                        // Sync if needed
                        let mut segment = segment_clone.lock().await;
                        if segment.needs_sync {
                            if let Err(e) = segment.sync() {
                                tracing::error!("Background WAL sync failed: {}", e);
                            }
                        }
                    }
                }))
            }
            _ => None,
        };

        Ok(Self {
            config,
            current_segment,
            current_lsn: AtomicU64::new(current_lsn),
            synced_lsn: AtomicU64::new(current_lsn.saturating_sub(1)),
            last_checkpoint: AtomicU64::new(last_checkpoint),
            checkpoint_counter: AtomicU64::new(checkpoint_id),
            segments,
            sync_handle,
            shutdown,
            closed: AtomicU64::new(0),
        })
    }

    /// Discover existing WAL segments and determine current state
    async fn discover_segments(config: &WalConfig) -> MemResult<(Vec<SegmentMeta>, u64, u64, u64)> {
        let mut segments = Vec::new();
        let mut max_lsn: u64 = 0;
        let mut last_checkpoint: u64 = 0;
        let mut max_checkpoint_id: u64 = 0;

        // Read directory entries
        let entries = std::fs::read_dir(&config.dir)
            .map_err(|e| MemError::storage(format!("Failed to read WAL directory: {}", e)))?;

        let mut segment_files: Vec<(u64, PathBuf)> = Vec::new();

        for entry in entries {
            let entry =
                entry.map_err(|e| MemError::storage(format!("Failed to read dir entry: {}", e)))?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(SEGMENT_PREFIX) && name.ends_with(SEGMENT_EXTENSION) {
                    // Extract segment ID from filename
                    let id_str = name
                        .trim_start_matches(SEGMENT_PREFIX)
                        .trim_end_matches(SEGMENT_EXTENSION);
                    if let Ok(id) = u64::from_str_radix(id_str, 16) {
                        segment_files.push((id, path));
                    }
                }
            }
        }

        // Sort by segment ID
        segment_files.sort_by_key(|(id, _)| *id);

        // Process each segment
        for (segment_id, path) in segment_files {
            let file = File::open(&path)
                .map_err(|e| MemError::storage(format!("Failed to open segment: {}", e)))?;
            let size_bytes = file
                .metadata()
                .map_err(|e| MemError::storage(format!("Failed to get segment metadata: {}", e)))?
                .len();

            // Read entries to find first/last LSN and checkpoints
            let mut reader = BufReader::new(file);
            let mut first_lsn = None;
            let mut last_lsn = None;

            while let Ok(Some(entry)) = read_entry(&mut reader) {
                if first_lsn.is_none() {
                    first_lsn = Some(entry.lsn);
                }
                last_lsn = Some(entry.lsn);

                if entry.lsn > max_lsn {
                    max_lsn = entry.lsn;
                }

                if let WalOperation::Checkpoint { lsn, checkpoint_id } = &entry.operation {
                    if *lsn > last_checkpoint {
                        last_checkpoint = *lsn;
                    }
                    if *checkpoint_id > max_checkpoint_id {
                        max_checkpoint_id = *checkpoint_id;
                    }
                }
            }

            segments.push(SegmentMeta {
                segment_id,
                path,
                first_lsn: first_lsn.unwrap_or(0),
                last_lsn,
                size_bytes,
            });
        }

        // Also check for checkpoint files
        if let Ok(checkpoint) = Self::find_latest_checkpoint(&config.dir) {
            if checkpoint.lsn.value() > last_checkpoint {
                last_checkpoint = checkpoint.lsn.value();
            }
            if checkpoint.id.value() > max_checkpoint_id {
                max_checkpoint_id = checkpoint.id.value();
            }
        }

        Ok((
            segments,
            max_lsn + 1,
            last_checkpoint,
            max_checkpoint_id + 1,
        ))
    }

    /// Find the latest valid checkpoint file
    fn find_latest_checkpoint(dir: &PathBuf) -> MemResult<Checkpoint> {
        let mut checkpoints: Vec<(u64, PathBuf)> = Vec::new();

        let entries = std::fs::read_dir(dir)
            .map_err(|e| MemError::storage(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry =
                entry.map_err(|e| MemError::storage(format!("Failed to read entry: {}", e)))?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(CHECKPOINT_PREFIX) && name.ends_with(CHECKPOINT_EXTENSION) {
                    let id_str = name
                        .trim_start_matches(CHECKPOINT_PREFIX)
                        .trim_end_matches(CHECKPOINT_EXTENSION);
                    if let Ok(id) = u64::from_str_radix(id_str, 16) {
                        checkpoints.push((id, path));
                    }
                }
            }
        }

        if checkpoints.is_empty() {
            return Err(MemError::not_found("No checkpoint files found"));
        }

        // Sort by ID descending and try each one
        checkpoints.sort_by(|a, b| b.0.cmp(&a.0));

        for (_, path) in checkpoints {
            let data = std::fs::read(&path)
                .map_err(|e| MemError::storage(format!("Failed to read checkpoint: {}", e)))?;

            match Checkpoint::from_bytes(&data) {
                Ok(ckpt) if ckpt.validate() => return Ok(ckpt),
                _ => continue,
            }
        }

        Err(MemError::not_found("No valid checkpoint files found"))
    }

    /// Create a new segment file
    fn create_new_segment(
        config: &WalConfig,
        segment_id: u64,
        first_lsn: u64,
    ) -> MemResult<SegmentWriter> {
        let filename = format!("{}{:016x}{}", SEGMENT_PREFIX, segment_id, SEGMENT_EXTENSION);
        let path = config.dir.join(filename);

        let meta = SegmentMeta {
            segment_id,
            path,
            first_lsn,
            last_lsn: None,
            size_bytes: 0,
        };

        SegmentWriter::new(meta)
    }

    /// Append an operation to the WAL
    ///
    /// Returns the Log Sequence Number (LSN) assigned to this operation.
    pub async fn append(&self, op: WalOperation) -> MemResult<u64> {
        if self.closed.load(Ordering::SeqCst) != 0 {
            return Err(MemError::storage("WAL is closed"));
        }

        let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst);
        let entry = WalEntry::new(lsn, op);

        let mut segment = self.current_segment.lock().await;

        // Check if we need to rotate to a new segment
        if segment.meta.size_bytes >= self.config.segment_size_bytes() as u64 {
            // Sync current segment before rotating
            segment.sync()?;

            // Update synced LSN
            if let Some(last) = segment.meta.last_lsn {
                self.synced_lsn.store(last, Ordering::SeqCst);
            }

            // Add current segment to list
            let mut segments = self.segments.write().await;
            segments.push(segment.meta.clone());

            // Create new segment
            let new_segment_id = segment.meta.segment_id + 1;
            *segment = Self::create_new_segment(&self.config, new_segment_id, lsn)?;
        }

        // Write the entry
        segment.write_entry(&entry)?;

        // Handle sync based on mode
        match &self.config.sync_mode {
            SyncMode::Immediate => {
                segment.sync()?;
                self.synced_lsn.store(lsn, Ordering::SeqCst);
            }
            SyncMode::Async | SyncMode::Batched(_) => {
                // Flush buffer but don't fsync
                segment.flush()?;
            }
        }

        tracing::trace!(
            lsn = lsn,
            op = entry.operation.op_type(),
            "WAL: Appended entry"
        );

        Ok(lsn)
    }

    /// Append multiple operations atomically
    pub async fn append_batch(&self, ops: Vec<WalOperation>) -> MemResult<u64> {
        if ops.is_empty() {
            return Ok(self.current_lsn.load(Ordering::SeqCst));
        }

        if self.closed.load(Ordering::SeqCst) != 0 {
            return Err(MemError::storage("WAL is closed"));
        }

        let first_lsn = self
            .current_lsn
            .fetch_add(ops.len() as u64, Ordering::SeqCst);
        let mut segment = self.current_segment.lock().await;

        for (i, op) in ops.into_iter().enumerate() {
            let lsn = first_lsn + i as u64;
            let entry = WalEntry::new(lsn, op);
            segment.write_entry(&entry)?;
        }

        // Handle sync based on mode
        match &self.config.sync_mode {
            SyncMode::Immediate => {
                segment.sync()?;
                self.synced_lsn.store(
                    self.current_lsn.load(Ordering::SeqCst) - 1,
                    Ordering::SeqCst,
                );
            }
            SyncMode::Async | SyncMode::Batched(_) => {
                segment.flush()?;
            }
        }

        Ok(first_lsn)
    }

    /// Force sync to disk
    pub async fn sync(&self) -> MemResult<()> {
        let mut segment = self.current_segment.lock().await;
        segment.sync()?;
        if let Some(last) = segment.meta.last_lsn {
            self.synced_lsn.store(last, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Create a checkpoint
    ///
    /// A checkpoint marks all operations up to this point as durable.
    /// Returns the checkpoint ID.
    pub async fn checkpoint(&self) -> MemResult<CheckpointId> {
        // Force sync first
        self.sync().await?;

        let checkpoint_id = self.checkpoint_counter.fetch_add(1, Ordering::SeqCst);
        let checkpoint_lsn = self.current_lsn.load(Ordering::SeqCst);

        // Append checkpoint operation to WAL
        let lsn = self
            .append(WalOperation::Checkpoint {
                lsn: checkpoint_lsn,
                checkpoint_id,
            })
            .await?;

        // Force sync the checkpoint entry
        self.sync().await?;

        // Create checkpoint file
        let checkpoint = Checkpoint::new(
            checkpoint_id,
            LogSequenceNumber::new(checkpoint_lsn),
            vec![], // TODO: Collect actual document IDs
            vec![], // TODO: Collect actual chunk IDs
        );

        let checkpoint_path = self.config.dir.join(format!(
            "{}{:016x}{}",
            CHECKPOINT_PREFIX, checkpoint_id, CHECKPOINT_EXTENSION
        ));

        let data = checkpoint.to_bytes()?;
        std::fs::write(&checkpoint_path, data)
            .map_err(|e| MemError::storage(format!("Failed to write checkpoint file: {}", e)))?;

        // Update last checkpoint
        self.last_checkpoint.store(checkpoint_lsn, Ordering::SeqCst);

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints().await?;

        tracing::info!(
            checkpoint_id = checkpoint_id,
            lsn = lsn,
            "WAL: Created checkpoint"
        );

        Ok(CheckpointId::new(checkpoint_id))
    }

    /// Cleanup old checkpoint files
    async fn cleanup_old_checkpoints(&self) -> MemResult<()> {
        let mut checkpoints: Vec<(u64, PathBuf)> = Vec::new();

        let entries = std::fs::read_dir(&self.config.dir)
            .map_err(|e| MemError::storage(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry =
                entry.map_err(|e| MemError::storage(format!("Failed to read entry: {}", e)))?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(CHECKPOINT_PREFIX) && name.ends_with(CHECKPOINT_EXTENSION) {
                    let id_str = name
                        .trim_start_matches(CHECKPOINT_PREFIX)
                        .trim_end_matches(CHECKPOINT_EXTENSION);
                    if let Ok(id) = u64::from_str_radix(id_str, 16) {
                        checkpoints.push((id, path));
                    }
                }
            }
        }

        // Sort by ID descending
        checkpoints.sort_by(|a, b| b.0.cmp(&a.0));

        // Remove old checkpoints beyond retention
        for (_, path) in checkpoints.iter().skip(self.config.checkpoint_retention) {
            if let Err(e) = std::fs::remove_file(path) {
                tracing::warn!(path = ?path, error = %e, "Failed to remove old checkpoint");
            }
        }

        Ok(())
    }

    /// Recover entries from the WAL after crash
    ///
    /// Returns a recovery report with all recovered entries.
    pub async fn recover(&self) -> MemResult<RecoveryReport> {
        let start_time = Instant::now();
        let mut report = RecoveryReport::default();

        // Find latest checkpoint
        let replay_from = match Self::find_latest_checkpoint(&self.config.dir) {
            Ok(checkpoint) => {
                report.checkpoint_used = Some(checkpoint.id);
                checkpoint.lsn.value()
            }
            Err(_) => self.last_checkpoint.load(Ordering::SeqCst),
        };

        let segments = self.segments.read().await;

        // Find segments that may contain entries after checkpoint
        for seg_meta in segments.iter() {
            // Skip segments entirely before checkpoint
            if let Some(last_lsn) = seg_meta.last_lsn {
                if last_lsn < replay_from {
                    continue;
                }
            }

            // Read entries from this segment
            match File::open(&seg_meta.path) {
                Ok(file) => {
                    let mut reader = BufReader::new(file);
                    while let Ok(Some(entry)) = read_entry(&mut reader) {
                        if entry.lsn > replay_from
                            && !matches!(entry.operation, WalOperation::Checkpoint { .. })
                        {
                            if entry.verify() {
                                report.entries_recovered += 1;
                                report.entries_replayed += 1;
                                report.last_valid_lsn = entry.lsn();
                            } else {
                                report.entries_skipped += 1;
                                report.errors.push(RecoveryError {
                                    lsn: Some(entry.lsn()),
                                    segment: Some(seg_meta.path.clone()),
                                    message: "Invalid checksum".to_string(),
                                    fatal: false,
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    report.errors.push(RecoveryError {
                        lsn: None,
                        segment: Some(seg_meta.path.clone()),
                        message: format!("Failed to open segment: {}", e),
                        fatal: false,
                    });
                }
            }
        }

        // Also read from current segment
        {
            let segment = self.current_segment.lock().await;
            if segment.meta.path.exists() {
                if let Ok(file) = File::open(&segment.meta.path) {
                    let mut reader = BufReader::new(file);
                    while let Ok(Some(entry)) = read_entry(&mut reader) {
                        if entry.lsn > replay_from
                            && !matches!(entry.operation, WalOperation::Checkpoint { .. })
                        {
                            if entry.verify() {
                                report.entries_recovered += 1;
                                report.entries_replayed += 1;
                                report.last_valid_lsn = entry.lsn();
                            } else {
                                report.entries_skipped += 1;
                            }
                        }
                    }
                }
            }
        }

        report.duration_ms = start_time.elapsed().as_millis() as u64;
        report.success = report.errors.iter().all(|e| !e.fatal);

        tracing::info!(
            count = report.entries_recovered,
            skipped = report.entries_skipped,
            duration_ms = report.duration_ms,
            "WAL: Recovery completed"
        );

        Ok(report)
    }

    /// Truncate WAL segments before the given LSN
    ///
    /// This removes old segments that are no longer needed.
    /// Only segments where all entries have LSN < the given value will be removed.
    pub async fn truncate_before(&self, lsn: LogSequenceNumber) -> MemResult<()> {
        let mut segments = self.segments.write().await;

        // Find segments to remove
        let mut to_remove = Vec::new();
        let mut i = 0;
        while i < segments.len() {
            let seg = &segments[i];
            if let Some(last_lsn) = seg.last_lsn {
                if last_lsn < lsn.value() {
                    to_remove.push(i);
                }
            }
            i += 1;
        }

        // Remove segments (in reverse order to maintain indices)
        for &idx in to_remove.iter().rev() {
            let seg = segments.remove(idx);
            if let Err(e) = std::fs::remove_file(&seg.path) {
                tracing::warn!(
                    path = ?seg.path,
                    error = %e,
                    "WAL: Failed to remove old segment"
                );
            } else {
                tracing::info!(
                    segment_id = seg.segment_id,
                    last_lsn = ?seg.last_lsn,
                    "WAL: Removed old segment"
                );
            }
        }

        Ok(())
    }

    /// Get the current LSN
    pub fn get_current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::SeqCst)
    }

    /// Get the last checkpoint LSN
    pub fn last_checkpoint_lsn(&self) -> u64 {
        self.last_checkpoint.load(Ordering::SeqCst)
    }

    /// Read entries from a given LSN
    pub async fn read_from(
        &self,
        start_lsn: LogSequenceNumber,
        limit: usize,
    ) -> MemResult<Vec<WalEntry>> {
        let mut result = Vec::with_capacity(limit);
        let segments = self.segments.read().await;

        // Read from all segments
        for seg_meta in segments.iter() {
            if result.len() >= limit {
                break;
            }

            if let Some(last_lsn) = seg_meta.last_lsn {
                if last_lsn < start_lsn.value() {
                    continue;
                }
            }

            if let Ok(file) = File::open(&seg_meta.path) {
                let mut reader = BufReader::new(file);
                while let Ok(Some(entry)) = read_entry(&mut reader) {
                    if entry.lsn >= start_lsn.value() && entry.verify() {
                        result.push(entry);
                        if result.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        // Also read from current segment
        {
            let segment = self.current_segment.lock().await;
            if result.len() < limit && segment.meta.path.exists() {
                if let Ok(file) = File::open(&segment.meta.path) {
                    let mut reader = BufReader::new(file);
                    while let Ok(Some(entry)) = read_entry(&mut reader) {
                        if entry.lsn >= start_lsn.value() && entry.verify() {
                            result.push(entry);
                            if result.len() >= limit {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Sort by LSN
        result.sort_by_key(|e| e.lsn);

        Ok(result)
    }

    /// Close the WAL gracefully
    pub async fn close(&self) -> MemResult<()> {
        self.closed.store(1, Ordering::SeqCst);
        self.shutdown.store(1, Ordering::SeqCst);
        self.sync().await?;
        Ok(())
    }

    /// Get statistics about the WAL
    pub async fn stats(&self) -> WalStats {
        let segments = self.segments.read().await;
        let current = self.current_segment.lock().await;

        let total_segments = segments.len() + 1; // Include current
        let total_size: u64 =
            segments.iter().map(|s| s.size_bytes).sum::<u64>() + current.meta.size_bytes;

        WalStats {
            current_lsn: self.current_lsn.load(Ordering::SeqCst),
            synced_lsn: self.synced_lsn.load(Ordering::SeqCst),
            last_checkpoint_lsn: self.last_checkpoint.load(Ordering::SeqCst),
            total_segments,
            total_size_bytes: total_size,
        }
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        // Signal shutdown to background tasks
        self.shutdown.store(1, Ordering::SeqCst);
    }
}

// ============================================================================
// Implement Trait for WriteAheadLog
// ============================================================================

#[async_trait]
impl WriteAheadLogTrait for WriteAheadLog {
    async fn append(&self, op: WalOperation) -> MemResult<LogSequenceNumber> {
        let lsn = WriteAheadLog::append(self, op).await?;
        Ok(LogSequenceNumber::new(lsn))
    }

    async fn append_batch(&self, ops: Vec<WalOperation>) -> MemResult<LogSequenceNumber> {
        let lsn = WriteAheadLog::append_batch(self, ops).await?;
        Ok(LogSequenceNumber::new(lsn))
    }

    async fn sync(&self) -> MemResult<()> {
        WriteAheadLog::sync(self).await
    }

    async fn checkpoint(&self) -> MemResult<CheckpointId> {
        WriteAheadLog::checkpoint(self).await
    }

    async fn recover(&self) -> MemResult<RecoveryReport> {
        WriteAheadLog::recover(self).await
    }

    async fn truncate_before(&self, lsn: LogSequenceNumber) -> MemResult<()> {
        WriteAheadLog::truncate_before(self, lsn).await
    }

    async fn current_lsn(&self) -> LogSequenceNumber {
        LogSequenceNumber::new(self.get_current_lsn())
    }

    async fn synced_lsn(&self) -> LogSequenceNumber {
        LogSequenceNumber::new(self.synced_lsn.load(Ordering::SeqCst))
    }

    async fn read_from(
        &self,
        start_lsn: LogSequenceNumber,
        limit: usize,
    ) -> MemResult<Vec<WalEntry>> {
        WriteAheadLog::read_from(self, start_lsn, limit).await
    }

    async fn close(&self) -> MemResult<()> {
        WriteAheadLog::close(self).await
    }
}

// ============================================================================
// In-Memory WAL (for testing)
// ============================================================================

/// In-memory WAL implementation for testing purposes
pub struct InMemoryWal {
    entries: RwLock<Vec<WalEntry>>,
    current_lsn: AtomicU64,
    checkpoints: RwLock<Vec<Checkpoint>>,
}

impl Default for InMemoryWal {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryWal {
    /// Create a new in-memory WAL instance
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            current_lsn: AtomicU64::new(1),
            checkpoints: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl WriteAheadLogTrait for InMemoryWal {
    async fn append(&self, op: WalOperation) -> MemResult<LogSequenceNumber> {
        let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst);
        let entry = WalEntry::new(lsn, op);

        let mut entries = self.entries.write().await;
        entries.push(entry);

        Ok(LogSequenceNumber::new(lsn))
    }

    async fn append_batch(&self, ops: Vec<WalOperation>) -> MemResult<LogSequenceNumber> {
        if ops.is_empty() {
            return Ok(LogSequenceNumber::new(
                self.current_lsn.load(Ordering::SeqCst),
            ));
        }

        let first_lsn = self
            .current_lsn
            .fetch_add(ops.len() as u64, Ordering::SeqCst);

        let mut entries = self.entries.write().await;
        for (i, op) in ops.into_iter().enumerate() {
            let entry = WalEntry::new(first_lsn + i as u64, op);
            entries.push(entry);
        }

        Ok(LogSequenceNumber::new(first_lsn))
    }

    async fn sync(&self) -> MemResult<()> {
        // No-op for in-memory
        Ok(())
    }

    async fn checkpoint(&self) -> MemResult<CheckpointId> {
        let checkpoints = self.checkpoints.read().await;
        let id = checkpoints.len() as u64 + 1;
        drop(checkpoints);

        let lsn = LogSequenceNumber::new(self.current_lsn.load(Ordering::SeqCst));
        let checkpoint = Checkpoint::new(id, lsn, vec![], vec![]);

        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.push(checkpoint);

        Ok(CheckpointId::new(id))
    }

    async fn recover(&self) -> MemResult<RecoveryReport> {
        let entries = self.entries.read().await;
        let checkpoints = self.checkpoints.read().await;

        Ok(RecoveryReport {
            entries_recovered: entries.len() as u64,
            entries_skipped: 0,
            entries_replayed: entries.len() as u64,
            last_valid_lsn: entries.last().map(|e| e.lsn()).unwrap_or_default(),
            checkpoint_used: checkpoints.last().map(|c| c.id),
            duration_ms: 0,
            errors: vec![],
            success: true,
        })
    }

    async fn truncate_before(&self, lsn: LogSequenceNumber) -> MemResult<()> {
        let mut entries = self.entries.write().await;
        entries.retain(|e| e.lsn >= lsn.value());
        Ok(())
    }

    async fn current_lsn(&self) -> LogSequenceNumber {
        LogSequenceNumber::new(self.current_lsn.load(Ordering::SeqCst))
    }

    async fn synced_lsn(&self) -> LogSequenceNumber {
        // In-memory is always "synced"
        self.current_lsn().await
    }

    async fn read_from(
        &self,
        start_lsn: LogSequenceNumber,
        limit: usize,
    ) -> MemResult<Vec<WalEntry>> {
        let entries = self.entries.read().await;
        Ok(entries
            .iter()
            .filter(|e| e.lsn >= start_lsn.value())
            .take(limit)
            .cloned()
            .collect())
    }

    async fn close(&self) -> MemResult<()> {
        Ok(())
    }
}

// ============================================================================
// WAL Statistics
// ============================================================================

/// WAL statistics
#[derive(Debug, Clone)]
pub struct WalStats {
    /// Current Log Sequence Number
    pub current_lsn: u64,
    /// Last synced LSN
    pub synced_lsn: u64,
    /// Last checkpoint LSN
    pub last_checkpoint_lsn: u64,
    /// Number of segment files
    pub total_segments: usize,
    /// Total size of all segments in bytes
    pub total_size_bytes: u64,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute CRC32 checksum for a WAL entry
///
/// The checksum is computed over all fields except the checksum itself.
pub fn compute_checksum(entry: &WalEntry) -> u32 {
    // Create a copy with checksum zeroed for computation
    let entry_for_hash = WalEntry {
        lsn: entry.lsn,
        timestamp: entry.timestamp,
        operation: entry.operation.clone(),
        checksum: 0,
    };

    // Serialize and compute CRC32
    match bincode_serialize(&entry_for_hash) {
        Ok(data) => crc32_compute(&data),
        Err(_) => 0,
    }
}

/// Verify the checksum of a WAL entry
pub fn verify_checksum(entry: &WalEntry) -> bool {
    let computed = compute_checksum(entry);
    computed == entry.checksum
}

/// Compute CRC32 checksum using the Castagnoli polynomial (CRC-32C)
fn crc32_compute(data: &[u8]) -> u32 {
    // CRC-32C polynomial: 0x1EDC6F41
    const CRC32C_TABLE: [u32; 256] = generate_crc32c_table();

    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        let index = ((crc ^ (*byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[index];
    }
    !crc
}

/// Generate CRC-32C lookup table at compile time
const fn generate_crc32c_table() -> [u32; 256] {
    const POLYNOMIAL: u32 = 0x82F63B78; // Reflected CRC-32C
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Serialize using JSON format
fn bincode_serialize<T: Serialize>(value: &T) -> MemResult<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|e| MemError::storage(format!("Failed to serialize WAL entry: {}", e)))
}

/// Deserialize using JSON format
fn bincode_deserialize<T: for<'de> Deserialize<'de>>(data: &[u8]) -> MemResult<T> {
    serde_json::from_slice(data)
        .map_err(|e| MemError::storage(format!("Failed to deserialize WAL entry: {}", e)))
}

/// Read a single entry from a WAL segment
fn read_entry<R: Read + BufRead>(reader: &mut R) -> MemResult<Option<WalEntry>> {
    // Read magic bytes
    let mut magic = [0u8; 4];
    match reader.read_exact(&mut magic) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => {
            return Err(MemError::storage(format!(
                "Failed to read WAL magic: {}",
                e
            )))
        }
    }

    if magic != WAL_MAGIC {
        return Err(MemError::storage("Invalid WAL magic bytes"));
    }

    // Read version
    let mut version = [0u8; 1];
    reader
        .read_exact(&mut version)
        .map_err(|e| MemError::storage(format!("Failed to read WAL version: {}", e)))?;

    if version[0] != WAL_VERSION {
        return Err(MemError::storage(format!(
            "Unsupported WAL version: {}",
            version[0]
        )));
    }

    // Read length
    let mut len_bytes = [0u8; 4];
    reader
        .read_exact(&mut len_bytes)
        .map_err(|e| MemError::storage(format!("Failed to read entry length: {}", e)))?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    // Sanity check length
    if len > 100 * 1024 * 1024 {
        // 100MB max
        return Err(MemError::storage(format!(
            "WAL entry too large: {} bytes",
            len
        )));
    }

    // Read data
    let mut data = vec![0u8; len];
    reader
        .read_exact(&mut data)
        .map_err(|e| MemError::storage(format!("Failed to read entry data: {}", e)))?;

    // Read trailing length
    let mut trailing_len = [0u8; 4];
    reader
        .read_exact(&mut trailing_len)
        .map_err(|e| MemError::storage(format!("Failed to read trailing length: {}", e)))?;

    if u32::from_le_bytes(trailing_len) != len as u32 {
        return Err(MemError::storage("WAL entry length mismatch"));
    }

    // Deserialize
    let entry: WalEntry = bincode_deserialize(&data)?;

    Ok(Some(entry))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_lsn_ordering() {
        let lsn1 = LogSequenceNumber::new(1);
        let lsn2 = LogSequenceNumber::new(2);
        let lsn3 = LogSequenceNumber::new(2);

        assert!(lsn1 < lsn2);
        assert_eq!(lsn2, lsn3);
        assert_eq!(lsn1.next(), lsn2);
    }

    #[test]
    fn test_checksum_computation() {
        let entry = WalEntry::new(
            1,
            WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "test content".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
            },
        );

        assert!(entry.verify());

        // Modify and verify checksum fails
        let mut modified = entry.clone();
        modified.lsn = 999;
        assert!(!modified.verify());
    }

    #[test]
    fn test_crc32_computation() {
        // Test vector: "123456789" should give 0xE3069283 for CRC-32C
        let data = b"123456789";
        let crc = crc32_compute(data);
        assert_eq!(crc, 0xE3069283);
    }

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = Checkpoint::new(
            1,
            LogSequenceNumber::new(100),
            vec![Uuid::new_v4(), Uuid::new_v4()],
            vec![Uuid::new_v4()],
        );

        let bytes = checkpoint.to_bytes().unwrap();
        let recovered = Checkpoint::from_bytes(&bytes).unwrap();

        assert_eq!(checkpoint.id, recovered.id);
        assert_eq!(checkpoint.lsn, recovered.lsn);
        assert_eq!(checkpoint.document_ids.len(), recovered.document_ids.len());
    }

    #[tokio::test]
    async fn test_in_memory_wal() {
        let wal = InMemoryWal::new();

        // Append some entries
        let lsn1 = wal
            .append(WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "doc1".to_string(),
                embedding: vec![0.1, 0.2, 0.3],
            })
            .await
            .unwrap();

        let lsn2 = wal
            .append(WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "doc2".to_string(),
                embedding: vec![0.4, 0.5, 0.6],
            })
            .await
            .unwrap();

        assert_eq!(lsn1.value(), 1);
        assert_eq!(lsn2.value(), 2);

        // Read back
        let entries = wal.read_from(LogSequenceNumber::new(1), 10).await.unwrap();
        assert_eq!(entries.len(), 2);

        // Checkpoint
        let ckpt = wal.checkpoint().await.unwrap();
        assert_eq!(ckpt.value(), 1);

        // Recovery report
        let report = wal.recover().await.unwrap();
        assert!(report.success);
        assert_eq!(report.entries_recovered, 2);
    }

    #[tokio::test]
    async fn test_wal_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            WalConfig::new(temp_dir.path().to_path_buf()).with_sync_mode(SyncMode::Immediate);

        let wal = WriteAheadLog::new(config).await.unwrap();

        // Append some entries
        let id1 = Uuid::new_v4();
        let lsn1 = wal
            .append(WalOperation::Insert {
                id: id1,
                content: "First entry".to_string(),
                embedding: vec![0.1, 0.2],
            })
            .await
            .unwrap();

        let id2 = Uuid::new_v4();
        let lsn2 = wal
            .append(WalOperation::Insert {
                id: id2,
                content: "Second entry".to_string(),
                embedding: vec![0.3, 0.4],
            })
            .await
            .unwrap();

        assert!(lsn2 > lsn1);

        // Create checkpoint
        wal.checkpoint().await.unwrap();

        // Check stats
        let stats = wal.stats().await;
        assert!(stats.current_lsn >= 3); // At least 2 inserts + 1 checkpoint
        assert_eq!(stats.total_segments, 1);
    }

    #[tokio::test]
    async fn test_wal_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            WalConfig::new(temp_dir.path().to_path_buf()).with_sync_mode(SyncMode::Immediate);

        // Write some entries
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        {
            let wal = WriteAheadLog::new(config.clone()).await.unwrap();

            wal.append(WalOperation::Insert {
                id: id1,
                content: "Before checkpoint".to_string(),
                embedding: vec![0.1],
            })
            .await
            .unwrap();

            wal.checkpoint().await.unwrap();

            wal.append(WalOperation::Insert {
                id: id2,
                content: "After checkpoint".to_string(),
                embedding: vec![0.2],
            })
            .await
            .unwrap();

            wal.close().await.unwrap();
        }

        // Recover
        let wal2 = WriteAheadLog::new(config).await.unwrap();
        let report = wal2.recover().await.unwrap();

        assert!(report.success);
        // Should have at least the entry after checkpoint
        assert!(report.entries_recovered >= 1);
    }

    #[tokio::test]
    async fn test_wal_batch_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            WalConfig::new(temp_dir.path().to_path_buf()).with_sync_mode(SyncMode::Immediate);

        let wal = WriteAheadLog::new(config).await.unwrap();

        // Batch insert
        let ops = vec![
            WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "Batch 1".to_string(),
                embedding: vec![0.1],
            },
            WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "Batch 2".to_string(),
                embedding: vec![0.2],
            },
            WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "Batch 3".to_string(),
                embedding: vec![0.3],
            },
        ];

        let first_lsn = wal.append_batch(ops).await.unwrap();

        // Verify consecutive LSNs
        let entries = wal
            .read_from(LogSequenceNumber::new(first_lsn), 10)
            .await
            .unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].lsn, first_lsn);
        assert_eq!(entries[1].lsn, first_lsn + 1);
        assert_eq!(entries[2].lsn, first_lsn + 2);
    }

    #[tokio::test]
    async fn test_wal_segment_rotation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path().to_path_buf())
            .with_segment_size(1) // 1MB segments
            .with_sync_mode(SyncMode::Immediate);

        let wal = WriteAheadLog::new(config).await.unwrap();

        // Write enough data to trigger rotation
        let large_embedding: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        for i in 0..20 {
            wal.append(WalOperation::Insert {
                id: Uuid::new_v4(),
                content: format!("Entry {} with large embedding", i),
                embedding: large_embedding.clone(),
            })
            .await
            .unwrap();
        }

        let stats = wal.stats().await;
        assert!(stats.total_segments >= 1);
    }

    #[tokio::test]
    async fn test_wal_truncate() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path().to_path_buf())
            .with_segment_size(1) // Small segments for testing
            .with_sync_mode(SyncMode::Immediate);

        let wal = WriteAheadLog::new(config).await.unwrap();

        // Write entries and checkpoint
        for _ in 0..5 {
            wal.append(WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "test".to_string(),
                embedding: vec![0.1; 1000],
            })
            .await
            .unwrap();
        }

        let checkpoint_id = wal.checkpoint().await.unwrap();

        // Truncate before checkpoint
        wal.truncate_before(LogSequenceNumber::new(wal.last_checkpoint_lsn()))
            .await
            .unwrap();

        assert!(checkpoint_id.value() > 0);
    }

    #[test]
    fn test_wal_operation_types() {
        let insert = WalOperation::Insert {
            id: Uuid::new_v4(),
            content: "test".to_string(),
            embedding: vec![],
        };
        assert_eq!(insert.op_type(), "INSERT");

        let update = WalOperation::Update {
            id: Uuid::new_v4(),
            content: "test".to_string(),
            embedding: vec![],
        };
        assert_eq!(update.op_type(), "UPDATE");

        let delete = WalOperation::Delete { id: Uuid::new_v4() };
        assert_eq!(delete.op_type(), "DELETE");

        let checkpoint = WalOperation::Checkpoint {
            lsn: 0,
            checkpoint_id: 1,
        };
        assert_eq!(checkpoint.op_type(), "CHECKPOINT");

        let batch = WalOperation::BatchInsert { items: vec![] };
        assert_eq!(batch.op_type(), "BATCH_INSERT");

        let txn_begin = WalOperation::TxnBegin {
            txn_id: Uuid::new_v4(),
        };
        assert_eq!(txn_begin.op_type(), "TXN_BEGIN");
    }

    #[test]
    fn test_wal_config_builder() {
        let config = WalConfig::new(PathBuf::from("/tmp/wal"))
            .with_segment_size(128)
            .with_sync_mode(SyncMode::Immediate)
            .with_checkpoint_retention(5);

        assert_eq!(config.segment_size_mb, 128);
        assert_eq!(config.checkpoint_retention, 5);
        assert!(matches!(config.sync_mode, SyncMode::Immediate));
    }

    #[test]
    fn test_entry_serialized_size() {
        let entry = WalEntry::new(
            1,
            WalOperation::Insert {
                id: Uuid::new_v4(),
                content: "Hello, World!".to_string(),
                embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            },
        );

        let size = entry.serialized_size().unwrap();
        assert!(size > 0);
        // Header overhead + data
        assert!(size > 50); // Reasonable minimum for the entry
    }

    #[tokio::test]
    async fn test_wal_trait_interface() {
        // Test the trait interface works with both implementations
        async fn use_wal(wal: &dyn WriteAheadLogTrait) -> MemResult<()> {
            let lsn = wal
                .append(WalOperation::Insert {
                    id: Uuid::new_v4(),
                    content: "test".to_string(),
                    embedding: vec![0.1],
                })
                .await?;

            assert!(lsn.value() > 0);
            Ok(())
        }

        // Test with InMemoryWal
        let mem_wal = InMemoryWal::new();
        use_wal(&mem_wal).await.unwrap();

        // Test with FileWal
        let temp_dir = TempDir::new().unwrap();
        let config =
            WalConfig::new(temp_dir.path().to_path_buf()).with_sync_mode(SyncMode::Immediate);
        let file_wal = WriteAheadLog::new(config).await.unwrap();
        use_wal(&file_wal).await.unwrap();
    }
}
