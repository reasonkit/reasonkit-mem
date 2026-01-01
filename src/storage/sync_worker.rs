//! Background Sync Worker for Hot-to-Cold Memory Migration
//!
//! This module implements a background worker that periodically moves data from
//! hot (fast, in-memory) storage to cold (persistent, disk-based) storage.
//!
//! ## Architecture
//!
//! ```text
//! +-------------+     SyncWorker     +-------------+
//! |  HotMemory  | -----------------> | ColdMemory  |
//! | (In-memory) |     (periodic)     |  (On-disk)  |
//! +-------------+                    +-------------+
//!       |                                   ^
//!       |           WriteAheadLog           |
//!       +-----------> (WAL) ----------------+
//!                  (durability)
//! ```
//!
//! ## Features
//!
//! - Periodic background synchronization using tokio
//! - Age-based migration from hot to cold storage
//! - WAL checkpointing for crash recovery
//! - Graceful shutdown with completion signaling
//! - Configurable sync intervals and batch sizes
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit_mem::storage::sync_worker::{
//!     spawn_sync_worker, SyncWorkerConfig, HotMemory, ColdMemory, WriteAheadLog,
//! };
//! use std::sync::Arc;
//! use std::time::Duration;
//!
//! let hot = Arc::new(HotMemory::new(config));
//! let cold = Arc::new(ColdMemory::new(path).await?);
//! let wal = Arc::new(WriteAheadLog::new(wal_path).await?);
//!
//! let config = SyncWorkerConfig {
//!     sync_interval: Duration::from_secs(60),
//!     hot_to_cold_age: Duration::from_secs(300),
//!     batch_size: 100,
//! };
//!
//! let (tx, handle) = spawn_sync_worker(hot, cold, wal, config);
//!
//! // Later, graceful shutdown:
//! let (done_tx, done_rx) = tokio::sync::oneshot::channel();
//! tx.send(SyncCommand::Shutdown(done_tx)).await?;
//! done_rx.await?;
//! handle.await?;
//! ```

use crate::{MemError, MemResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ============================================================================
// Commands and Configuration
// ============================================================================

/// Commands that can be sent to the sync worker
#[derive(Debug)]
pub enum SyncCommand {
    /// Trigger an immediate sync from hot to cold
    Sync,
    /// Trigger a WAL checkpoint
    Checkpoint,
    /// Gracefully shutdown the worker (sends completion signal)
    Shutdown(oneshot::Sender<()>),
}

/// Configuration for the sync worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncWorkerConfig {
    /// Interval between automatic sync operations
    pub sync_interval: Duration,
    /// Minimum age for entries to be moved from hot to cold
    pub hot_to_cold_age: Duration,
    /// Maximum number of entries to process in a single batch
    pub batch_size: usize,
}

impl Default for SyncWorkerConfig {
    fn default() -> Self {
        Self {
            sync_interval: Duration::from_secs(60),
            hot_to_cold_age: Duration::from_secs(300), // 5 minutes
            batch_size: 100,
        }
    }
}

// ============================================================================
// Memory Entry Types
// ============================================================================

/// A memory entry stored in hot or cold storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier for this entry
    pub id: Uuid,
    /// The actual data (embeddings, metadata, etc.)
    pub data: MemoryData,
    /// When this entry was created
    pub created_at: DateTime<Utc>,
    /// When this entry was last accessed
    pub last_accessed: DateTime<Utc>,
    /// Access count for LRU/LFU eviction
    pub access_count: u64,
    /// Optional TTL for automatic expiration
    pub ttl: Option<Duration>,
}

/// Types of data that can be stored in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryData {
    /// Dense embedding vector
    Embedding(Vec<f32>),
    /// Document chunk with text
    Chunk {
        text: String,
        document_id: Uuid,
        position: usize,
    },
    /// Metadata blob
    Metadata(HashMap<String, serde_json::Value>),
    /// RAPTOR tree node
    RaptorNode {
        level: usize,
        summary: String,
        children: Vec<Uuid>,
    },
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(id: Uuid, data: MemoryData) -> Self {
        let now = Utc::now();
        Self {
            id,
            data,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ttl: None,
        }
    }

    /// Create an embedding entry
    pub fn embedding(id: Uuid, vector: Vec<f32>) -> Self {
        Self::new(id, MemoryData::Embedding(vector))
    }

    /// Create a chunk entry
    pub fn chunk(id: Uuid, text: String, document_id: Uuid, position: usize) -> Self {
        Self::new(
            id,
            MemoryData::Chunk {
                text,
                document_id,
                position,
            },
        )
    }

    /// Check if this entry has expired based on its TTL
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let age = Utc::now()
                .signed_duration_since(self.created_at)
                .to_std()
                .unwrap_or(Duration::ZERO);
            age > ttl
        } else {
            false
        }
    }

    /// Check if this entry is old enough to be moved to cold storage
    pub fn is_cold_eligible(&self, threshold: Duration) -> bool {
        let age = Utc::now()
            .signed_duration_since(self.last_accessed)
            .to_std()
            .unwrap_or(Duration::ZERO);
        age > threshold
    }

    /// Record an access to this entry
    pub fn record_access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }

    /// Get the serialized size estimate in bytes
    pub fn estimated_size(&self) -> usize {
        match &self.data {
            MemoryData::Embedding(v) => v.len() * 4 + 64, // floats + overhead
            MemoryData::Chunk { text, .. } => text.len() + 128,
            MemoryData::Metadata(m) => m.len() * 64, // rough estimate
            MemoryData::RaptorNode {
                summary, children, ..
            } => summary.len() + children.len() * 16 + 32,
        }
    }
}

// ============================================================================
// Hot Memory (In-Memory Cache)
// ============================================================================

/// Configuration for hot memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMemoryConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Enable LRU eviction
    pub enable_lru_eviction: bool,
}

impl Default for HotMemoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            enable_lru_eviction: true,
        }
    }
}

/// Hot memory storage (fast, in-memory)
///
/// Uses DashMap for concurrent access with minimal locking.
pub struct HotMemory {
    /// The actual storage
    entries: DashMap<Uuid, MemoryEntry>,
    /// Configuration
    config: HotMemoryConfig,
    /// Current memory usage estimate
    memory_usage: AtomicU64,
    /// Statistics
    stats: Arc<RwLock<HotMemoryStats>>,
}

/// Statistics for hot memory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HotMemoryStats {
    /// Total entries stored
    pub entry_count: usize,
    /// Total memory usage in bytes
    pub memory_bytes: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Entries evicted due to capacity
    pub evictions: u64,
}

impl HotMemory {
    /// Create a new hot memory instance
    pub fn new(config: HotMemoryConfig) -> Self {
        Self {
            entries: DashMap::with_capacity(config.max_entries),
            config,
            memory_usage: AtomicU64::new(0),
            stats: Arc::new(RwLock::new(HotMemoryStats::default())),
        }
    }

    /// Insert an entry into hot memory
    pub async fn insert(&self, entry: MemoryEntry) -> MemResult<()> {
        let size = entry.estimated_size() as u64;
        let id = entry.id;

        // Check capacity and evict if necessary
        if self.config.enable_lru_eviction {
            while self.entries.len() >= self.config.max_entries
                || self.memory_usage.load(Ordering::Relaxed) + size
                    > self.config.max_memory_bytes as u64
            {
                if !self.evict_lru().await {
                    break;
                }
            }
        }

        // Remove old entry if exists
        if let Some(old) = self.entries.remove(&id) {
            self.memory_usage
                .fetch_sub(old.1.estimated_size() as u64, Ordering::Relaxed);
        }

        // Insert new entry
        self.entries.insert(id, entry);
        self.memory_usage.fetch_add(size, Ordering::Relaxed);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.entry_count = self.entries.len();
            stats.memory_bytes = self.memory_usage.load(Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get an entry from hot memory (updates access time)
    pub async fn get(&self, id: &Uuid) -> Option<MemoryEntry> {
        if let Some(mut entry) = self.entries.get_mut(id) {
            entry.record_access();
            let mut stats = self.stats.write().await;
            stats.hits += 1;
            Some(entry.clone())
        } else {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
            None
        }
    }

    /// Get an entry without updating access time (for sync operations)
    pub fn get_readonly(&self, id: &Uuid) -> Option<MemoryEntry> {
        self.entries.get(id).map(|e| e.clone())
    }

    /// Remove an entry from hot memory
    pub async fn remove(&self, id: &Uuid) -> Option<MemoryEntry> {
        if let Some((_, entry)) = self.entries.remove(id) {
            self.memory_usage
                .fetch_sub(entry.estimated_size() as u64, Ordering::Relaxed);
            let mut stats = self.stats.write().await;
            stats.entry_count = self.entries.len();
            stats.memory_bytes = self.memory_usage.load(Ordering::Relaxed);
            Some(entry)
        } else {
            None
        }
    }

    /// Get entries eligible for cold storage migration
    pub fn get_cold_eligible(&self, threshold: Duration, limit: usize) -> Vec<MemoryEntry> {
        let mut eligible = Vec::new();

        for entry_ref in self.entries.iter() {
            if entry_ref.is_cold_eligible(threshold) && !entry_ref.is_expired() {
                eligible.push(entry_ref.clone());
                if eligible.len() >= limit {
                    break;
                }
            }
        }

        // Sort by last_accessed (oldest first)
        eligible.sort_by(|a, b| a.last_accessed.cmp(&b.last_accessed));
        eligible
    }

    /// Evict the least recently used entry
    async fn evict_lru(&self) -> bool {
        // Find LRU entry
        let lru_id = self
            .entries
            .iter()
            .min_by(|a, b| a.last_accessed.cmp(&b.last_accessed))
            .map(|e| *e.key());

        if let Some(id) = lru_id {
            self.remove(&id).await;
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Get current statistics
    pub async fn stats(&self) -> HotMemoryStats {
        self.stats.read().await.clone()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================================
// Cold Memory (Persistent Disk Storage)
// ============================================================================

/// Cold memory storage (persistent, disk-based)
///
/// Uses sled for embedded key-value storage with crash recovery.
pub struct ColdMemory {
    /// Sled database
    db: sled::Db,
    /// Path to the database
    path: PathBuf,
    /// Statistics
    stats: Arc<RwLock<ColdMemoryStats>>,
}

/// Statistics for cold memory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColdMemoryStats {
    /// Total entries stored
    pub entry_count: usize,
    /// Total size on disk in bytes
    pub disk_bytes: u64,
    /// Reads performed
    pub reads: u64,
    /// Writes performed
    pub writes: u64,
}

impl ColdMemory {
    /// Create or open cold memory storage
    pub async fn new(path: PathBuf) -> MemResult<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                MemError::storage(format!("Failed to create cold memory directory: {}", e))
            })?;
        }

        // Open sled database
        let db = sled::open(&path)
            .map_err(|e| MemError::storage(format!("Failed to open cold memory: {}", e)))?;

        let stats = Arc::new(RwLock::new(ColdMemoryStats {
            entry_count: db.len(),
            ..Default::default()
        }));

        Ok(Self { db, path, stats })
    }

    /// Store an entry in cold memory
    pub async fn store(&self, entry: &MemoryEntry) -> MemResult<()> {
        let key = entry.id.as_bytes().to_vec();
        let value = bincode::serialize(entry)
            .map_err(|e| MemError::storage(format!("Failed to serialize entry: {}", e)))?;

        self.db
            .insert(key, value)
            .map_err(|e| MemError::storage(format!("Failed to store in cold memory: {}", e)))?;

        let mut stats = self.stats.write().await;
        stats.writes += 1;
        stats.entry_count = self.db.len();

        Ok(())
    }

    /// Store multiple entries in a batch
    pub async fn store_batch(&self, entries: &[MemoryEntry]) -> MemResult<usize> {
        let mut batch = sled::Batch::default();
        let mut stored = 0;

        for entry in entries {
            let key = entry.id.as_bytes().to_vec();
            match bincode::serialize(entry) {
                Ok(value) => {
                    batch.insert(key, value);
                    stored += 1;
                }
                Err(e) => {
                    warn!("Failed to serialize entry {}: {}", entry.id, e);
                }
            }
        }

        self.db
            .apply_batch(batch)
            .map_err(|e| MemError::storage(format!("Failed to apply batch: {}", e)))?;

        let mut stats = self.stats.write().await;
        stats.writes += stored as u64;
        stats.entry_count = self.db.len();

        Ok(stored)
    }

    /// Retrieve an entry from cold memory
    pub async fn get(&self, id: &Uuid) -> MemResult<Option<MemoryEntry>> {
        let key = id.as_bytes().to_vec();

        let result = self
            .db
            .get(key)
            .map_err(|e| MemError::storage(format!("Failed to read from cold memory: {}", e)))?;

        let mut stats = self.stats.write().await;
        stats.reads += 1;

        if let Some(value) = result {
            let entry: MemoryEntry = bincode::deserialize(&value)
                .map_err(|e| MemError::storage(format!("Failed to deserialize entry: {}", e)))?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    /// Remove an entry from cold memory
    pub async fn remove(&self, id: &Uuid) -> MemResult<Option<MemoryEntry>> {
        let key = id.as_bytes().to_vec();

        let result = self
            .db
            .remove(key)
            .map_err(|e| MemError::storage(format!("Failed to remove from cold memory: {}", e)))?;

        if let Some(value) = result {
            let entry: MemoryEntry = bincode::deserialize(&value)
                .map_err(|e| MemError::storage(format!("Failed to deserialize entry: {}", e)))?;
            let mut stats = self.stats.write().await;
            stats.entry_count = self.db.len();
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    /// Flush all pending writes to disk
    pub async fn flush(&self) -> MemResult<()> {
        self.db
            .flush_async()
            .await
            .map_err(|e| MemError::storage(format!("Failed to flush cold memory: {}", e)))?;
        Ok(())
    }

    /// Get disk usage estimate
    pub async fn disk_usage(&self) -> MemResult<u64> {
        let size = self
            .db
            .size_on_disk()
            .map_err(|e| MemError::storage(format!("Failed to get disk size: {}", e)))?;
        Ok(size)
    }

    /// Get current statistics
    pub async fn stats(&self) -> ColdMemoryStats {
        let mut stats = self.stats.read().await.clone();
        stats.disk_bytes = self.disk_usage().await.unwrap_or(0);
        stats
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.db.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.db.is_empty()
    }
}

// ============================================================================
// Write-Ahead Log (WAL)
// ============================================================================

/// WAL entry type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    /// Insert or update an entry
    Insert(MemoryEntry),
    /// Remove an entry
    Remove(Uuid),
    /// Batch insert
    BatchInsert(Vec<MemoryEntry>),
    /// Checkpoint marker
    Checkpoint { sequence: u64 },
}

/// Write-ahead log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Sequence number
    pub sequence: u64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Operation
    pub operation: WalOperation,
    /// CRC32 checksum for integrity
    pub checksum: u32,
}

impl WalEntry {
    /// Create a new WAL entry
    pub fn new(sequence: u64, operation: WalOperation) -> Self {
        let entry = Self {
            sequence,
            timestamp: Utc::now(),
            operation,
            checksum: 0, // Calculated on serialize
        };
        entry.with_checksum()
    }

    /// Calculate and set the checksum
    fn with_checksum(mut self) -> Self {
        // Serialize without checksum for calculation
        let checksum_data = bincode::serialize(&(&self.sequence, &self.timestamp, &self.operation))
            .unwrap_or_default();
        self.checksum = crc32fast::hash(&checksum_data);
        self
    }

    /// Verify the checksum
    pub fn verify_checksum(&self) -> bool {
        let checksum_data = bincode::serialize(&(&self.sequence, &self.timestamp, &self.operation))
            .unwrap_or_default();
        let calculated = crc32fast::hash(&checksum_data);
        calculated == self.checksum
    }
}

/// Write-ahead log for durability
pub struct WriteAheadLog {
    /// Path to WAL file
    path: PathBuf,
    /// Current sequence number
    sequence: AtomicU64,
    /// Last checkpoint sequence
    last_checkpoint: AtomicU64,
    /// Write lock
    write_lock: RwLock<()>,
}

impl WriteAheadLog {
    /// Create or open a WAL
    pub async fn new(path: PathBuf) -> MemResult<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| MemError::storage(format!("Failed to create WAL directory: {}", e)))?;
        }

        // Get the last sequence number from existing WAL
        let (sequence, last_checkpoint) = Self::recover_sequence(&path).await.unwrap_or((0, 0));

        Ok(Self {
            path,
            sequence: AtomicU64::new(sequence),
            last_checkpoint: AtomicU64::new(last_checkpoint),
            write_lock: RwLock::new(()),
        })
    }

    /// Recover sequence number from existing WAL
    async fn recover_sequence(path: &PathBuf) -> MemResult<(u64, u64)> {
        if !path.exists() {
            return Ok((0, 0));
        }

        let content = tokio::fs::read(path)
            .await
            .map_err(|e| MemError::storage(format!("Failed to read WAL for recovery: {}", e)))?;

        let mut max_sequence = 0u64;
        let mut last_checkpoint = 0u64;
        let mut offset = 0;

        while offset < content.len() {
            // Try to read length prefix (u32)
            if offset + 4 > content.len() {
                break;
            }
            let len = u32::from_le_bytes(content[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + len > content.len() {
                break;
            }

            // Try to deserialize entry
            if let Ok(entry) = bincode::deserialize::<WalEntry>(&content[offset..offset + len]) {
                if entry.verify_checksum() {
                    max_sequence = max_sequence.max(entry.sequence);
                    if let WalOperation::Checkpoint { sequence } = entry.operation {
                        last_checkpoint = sequence;
                    }
                }
            }
            offset += len;
        }

        Ok((max_sequence, last_checkpoint))
    }

    /// Append an operation to the WAL
    pub async fn append(&self, operation: WalOperation) -> MemResult<u64> {
        let _lock = self.write_lock.write().await;
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let entry = WalEntry::new(sequence, operation);

        let data = bincode::serialize(&entry)
            .map_err(|e| MemError::storage(format!("WAL serialize: {}", e)))?;

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .map_err(|e| MemError::storage(format!("WAL open: {}", e)))?;

        // Write length prefix + data
        let len = (data.len() as u32).to_le_bytes();
        file.write_all(&len)
            .await
            .map_err(|e| MemError::storage(format!("WAL write: {}", e)))?;
        file.write_all(&data)
            .await
            .map_err(|e| MemError::storage(format!("WAL write: {}", e)))?;
        file.sync_all()
            .await
            .map_err(|e| MemError::storage(format!("WAL sync: {}", e)))?;

        Ok(sequence)
    }

    /// Write a checkpoint marker and truncate old entries
    pub async fn checkpoint(&self) -> MemResult<u64> {
        let sequence = self.sequence.load(Ordering::SeqCst);

        // Write checkpoint marker
        self.append(WalOperation::Checkpoint { sequence }).await?;

        // Update last checkpoint
        self.last_checkpoint.store(sequence, Ordering::SeqCst);

        // Truncate old entries (create new file with only checkpoint)
        let checkpoint_entry = WalEntry::new(sequence, WalOperation::Checkpoint { sequence });
        let data = bincode::serialize(&checkpoint_entry)
            .map_err(|e| MemError::storage(format!("Checkpoint serialize: {}", e)))?;

        let len = (data.len() as u32).to_le_bytes();
        let mut content = Vec::with_capacity(4 + data.len());
        content.extend_from_slice(&len);
        content.extend_from_slice(&data);

        tokio::fs::write(&self.path, content)
            .await
            .map_err(|e| MemError::storage(format!("Checkpoint write: {}", e)))?;

        info!(sequence, "WAL checkpoint completed");
        Ok(sequence)
    }

    /// Get entries since the last checkpoint for recovery
    pub async fn get_entries_since_checkpoint(&self) -> MemResult<Vec<WalEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let content = tokio::fs::read(&self.path)
            .await
            .map_err(|e| MemError::storage(format!("WAL read: {}", e)))?;

        let last_checkpoint = self.last_checkpoint.load(Ordering::SeqCst);
        let mut entries = Vec::new();
        let mut offset = 0;

        while offset < content.len() {
            if offset + 4 > content.len() {
                break;
            }
            let len = u32::from_le_bytes(content[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + len > content.len() {
                break;
            }

            if let Ok(entry) = bincode::deserialize::<WalEntry>(&content[offset..offset + len]) {
                if entry.verify_checksum() && entry.sequence > last_checkpoint {
                    entries.push(entry);
                }
            }
            offset += len;
        }

        entries.sort_by_key(|e| e.sequence);
        Ok(entries)
    }

    /// Get current sequence number
    pub fn current_sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }

    /// Get last checkpoint sequence
    pub fn last_checkpoint_sequence(&self) -> u64 {
        self.last_checkpoint.load(Ordering::SeqCst)
    }
}

// ============================================================================
// Sync Worker
// ============================================================================

/// Statistics from a sync operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    /// Number of entries moved from hot to cold
    pub entries_synced: usize,
    /// Number of entries skipped (too young)
    pub entries_skipped: usize,
    /// Number of errors encountered
    pub errors: usize,
    /// Total bytes synced
    pub bytes_synced: u64,
    /// Duration of the sync operation
    pub duration: Duration,
    /// Timestamp of the sync
    pub timestamp: DateTime<Utc>,
}

/// Background sync worker for hot-to-cold migration
pub struct SyncWorker {
    /// Hot memory reference
    hot: Arc<HotMemory>,
    /// Cold memory reference
    cold: Arc<ColdMemory>,
    /// Write-ahead log reference
    wal: Arc<WriteAheadLog>,
    /// Worker configuration
    config: SyncWorkerConfig,
    /// Command receiver
    command_rx: mpsc::Receiver<SyncCommand>,
}

impl SyncWorker {
    /// Create a new sync worker
    ///
    /// Returns the worker and a command sender for controlling it.
    pub fn new(
        hot: Arc<HotMemory>,
        cold: Arc<ColdMemory>,
        wal: Arc<WriteAheadLog>,
        config: SyncWorkerConfig,
    ) -> (Self, mpsc::Sender<SyncCommand>) {
        let (command_tx, command_rx) = mpsc::channel(16);

        let worker = Self {
            hot,
            cold,
            wal,
            config,
            command_rx,
        };

        (worker, command_tx)
    }

    /// Run the sync worker
    ///
    /// This is the main loop that:
    /// - Listens for commands
    /// - Periodically syncs hot to cold
    /// - Handles graceful shutdown
    pub async fn run(mut self) {
        info!(
            sync_interval_secs = self.config.sync_interval.as_secs(),
            hot_to_cold_age_secs = self.config.hot_to_cold_age.as_secs(),
            batch_size = self.config.batch_size,
            "Sync worker started"
        );

        let mut interval = tokio::time::interval(self.config.sync_interval);
        // Don't immediately tick
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Handle incoming commands
                Some(cmd) = self.command_rx.recv() => {
                    match cmd {
                        SyncCommand::Sync => {
                            debug!("Received manual sync command");
                            match self.sync_hot_to_cold().await {
                                Ok(stats) => {
                                    info!(
                                        entries_synced = stats.entries_synced,
                                        bytes_synced = stats.bytes_synced,
                                        duration_ms = stats.duration.as_millis() as u64,
                                        "Manual sync completed"
                                    );
                                }
                                Err(e) => {
                                    error!(error = %e, "Manual sync failed");
                                }
                            }
                        }
                        SyncCommand::Checkpoint => {
                            debug!("Received checkpoint command");
                            match self.checkpoint().await {
                                Ok(seq) => {
                                    info!(sequence = seq, "Checkpoint completed");
                                }
                                Err(e) => {
                                    error!(error = %e, "Checkpoint failed");
                                }
                            }
                        }
                        SyncCommand::Shutdown(done) => {
                            info!("Received shutdown command, performing final sync");

                            // Final sync before shutdown
                            if let Err(e) = self.sync_hot_to_cold().await {
                                warn!(error = %e, "Final sync failed during shutdown");
                            }

                            // Final checkpoint
                            if let Err(e) = self.checkpoint().await {
                                warn!(error = %e, "Final checkpoint failed during shutdown");
                            }

                            // Flush cold storage
                            if let Err(e) = self.cold.flush().await {
                                warn!(error = %e, "Cold storage flush failed during shutdown");
                            }

                            info!("Sync worker shutdown complete");
                            let _ = done.send(());
                            return;
                        }
                    }
                }

                // Periodic sync
                _ = interval.tick() => {
                    debug!("Periodic sync triggered");
                    match self.sync_hot_to_cold().await {
                        Ok(stats) => {
                            if stats.entries_synced > 0 {
                                info!(
                                    entries_synced = stats.entries_synced,
                                    bytes_synced = stats.bytes_synced,
                                    duration_ms = stats.duration.as_millis() as u64,
                                    "Periodic sync completed"
                                );
                            } else {
                                debug!("Periodic sync: no entries eligible for migration");
                            }
                        }
                        Err(e) => {
                            error!(error = %e, "Periodic sync failed");
                        }
                    }
                }
            }
        }
    }

    /// Sync entries from hot to cold memory
    pub async fn sync_hot_to_cold(&self) -> MemResult<SyncStats> {
        let start = std::time::Instant::now();
        let mut stats = SyncStats {
            timestamp: Utc::now(),
            ..Default::default()
        };

        // Get entries eligible for cold storage
        let eligible = self
            .hot
            .get_cold_eligible(self.config.hot_to_cold_age, self.config.batch_size);

        if eligible.is_empty() {
            stats.duration = start.elapsed();
            return Ok(stats);
        }

        debug!(
            eligible_count = eligible.len(),
            "Found entries eligible for cold migration"
        );

        // Write to WAL first (durability)
        for entry in &eligible {
            if let Err(e) = self.wal.append(WalOperation::Insert(entry.clone())).await {
                warn!(id = %entry.id, error = %e, "Failed to write to WAL");
                stats.errors += 1;
                continue;
            }
        }

        // Store in cold memory
        match self.cold.store_batch(&eligible).await {
            Ok(stored) => {
                stats.entries_synced = stored;
                stats.bytes_synced = eligible.iter().map(|e| e.estimated_size() as u64).sum();

                // Remove from hot memory after successful cold storage
                for entry in &eligible {
                    if let Err(e) = async {
                        self.hot.remove(&entry.id).await;
                        Ok::<_, MemError>(())
                    }
                    .await
                    {
                        warn!(id = %entry.id, error = %e, "Failed to remove from hot memory");
                    }
                }
            }
            Err(e) => {
                error!(error = %e, "Failed to store batch in cold memory");
                stats.errors += eligible.len();
            }
        }

        stats.entries_skipped = self.hot.len();
        stats.duration = start.elapsed();

        Ok(stats)
    }

    /// Checkpoint the WAL
    pub async fn checkpoint(&self) -> MemResult<u64> {
        // Flush cold storage first
        self.cold.flush().await?;

        // Then checkpoint the WAL
        self.wal.checkpoint().await
    }
}

// ============================================================================
// Spawn Helper
// ============================================================================

/// Spawn the sync worker as a background task
///
/// Returns a command sender and the task join handle.
///
/// # Example
///
/// ```rust,ignore
/// let (tx, handle) = spawn_sync_worker(hot, cold, wal, config);
///
/// // Trigger manual sync
/// tx.send(SyncCommand::Sync).await?;
///
/// // Graceful shutdown
/// let (done_tx, done_rx) = tokio::sync::oneshot::channel();
/// tx.send(SyncCommand::Shutdown(done_tx)).await?;
/// done_rx.await?;
/// handle.await?;
/// ```
pub fn spawn_sync_worker(
    hot: Arc<HotMemory>,
    cold: Arc<ColdMemory>,
    wal: Arc<WriteAheadLog>,
    config: SyncWorkerConfig,
) -> (mpsc::Sender<SyncCommand>, tokio::task::JoinHandle<()>) {
    let (worker, command_tx) = SyncWorker::new(hot, cold, wal, config);
    let handle = tokio::spawn(worker.run());
    (command_tx, handle)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_memory_entry_creation() {
        let id = Uuid::new_v4();
        let entry = MemoryEntry::embedding(id, vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.id, id);
        assert_eq!(entry.access_count, 0);
        assert!(!entry.is_expired());
    }

    #[tokio::test]
    async fn test_memory_entry_cold_eligibility() {
        let id = Uuid::new_v4();
        let mut entry = MemoryEntry::embedding(id, vec![1.0, 2.0, 3.0]);

        // Fresh entry should not be cold eligible
        assert!(!entry.is_cold_eligible(Duration::from_secs(1)));

        // Simulate old access time
        entry.last_accessed = Utc::now() - chrono::Duration::seconds(10);
        assert!(entry.is_cold_eligible(Duration::from_secs(5)));
    }

    #[tokio::test]
    async fn test_hot_memory_insert_get() {
        let config = HotMemoryConfig::default();
        let hot = HotMemory::new(config);

        let id = Uuid::new_v4();
        let entry = MemoryEntry::embedding(id, vec![1.0, 2.0, 3.0]);

        hot.insert(entry.clone()).await.unwrap();
        assert_eq!(hot.len(), 1);

        let retrieved = hot.get(&id).await.unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.access_count, 1); // Access was recorded
    }

    #[tokio::test]
    async fn test_hot_memory_lru_eviction() {
        let config = HotMemoryConfig {
            max_entries: 2,
            max_memory_bytes: 1024 * 1024,
            enable_lru_eviction: true,
        };
        let hot = HotMemory::new(config);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        hot.insert(MemoryEntry::embedding(id1, vec![1.0]))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        hot.insert(MemoryEntry::embedding(id2, vec![2.0]))
            .await
            .unwrap();

        // Access id1 to make it more recent
        hot.get(&id1).await;
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Insert id3, should evict id2 (least recently accessed)
        hot.insert(MemoryEntry::embedding(id3, vec![3.0]))
            .await
            .unwrap();

        assert_eq!(hot.len(), 2);
        assert!(hot.get(&id1).await.is_some());
        assert!(hot.get(&id2).await.is_none()); // Evicted
        assert!(hot.get(&id3).await.is_some());
    }

    #[tokio::test]
    async fn test_cold_memory_store_get() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("cold.db");

        let cold = ColdMemory::new(path).await.unwrap();

        let id = Uuid::new_v4();
        let entry = MemoryEntry::embedding(id, vec![1.0, 2.0, 3.0]);

        cold.store(&entry).await.unwrap();
        assert_eq!(cold.len(), 1);

        let retrieved = cold.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.id, id);
    }

    #[tokio::test]
    async fn test_cold_memory_batch_store() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("cold.db");

        let cold = ColdMemory::new(path).await.unwrap();

        let entries: Vec<_> = (0..10)
            .map(|i| MemoryEntry::embedding(Uuid::new_v4(), vec![i as f32]))
            .collect();

        let stored = cold.store_batch(&entries).await.unwrap();
        assert_eq!(stored, 10);
        assert_eq!(cold.len(), 10);
    }

    #[tokio::test]
    async fn test_wal_append_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.wal");

        let wal = WriteAheadLog::new(path).await.unwrap();

        // Append some operations
        let id = Uuid::new_v4();
        let entry = MemoryEntry::embedding(id, vec![1.0, 2.0]);

        let seq1 = wal
            .append(WalOperation::Insert(entry.clone()))
            .await
            .unwrap();
        let seq2 = wal.append(WalOperation::Remove(id)).await.unwrap();

        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);
        assert_eq!(wal.current_sequence(), 2);

        // Checkpoint
        let checkpoint_seq = wal.checkpoint().await.unwrap();
        assert_eq!(checkpoint_seq, 2);
    }

    #[tokio::test]
    async fn test_wal_entry_checksum() {
        let entry = WalEntry::new(1, WalOperation::Remove(Uuid::new_v4()));
        assert!(entry.verify_checksum());

        // Corrupt the entry
        let mut corrupted = entry.clone();
        corrupted.sequence = 999;
        assert!(!corrupted.verify_checksum());
    }

    #[tokio::test]
    async fn test_sync_worker_creation() {
        let temp_dir = TempDir::new().unwrap();

        let hot = Arc::new(HotMemory::new(HotMemoryConfig::default()));
        let cold = Arc::new(
            ColdMemory::new(temp_dir.path().join("cold.db"))
                .await
                .unwrap(),
        );
        let wal = Arc::new(
            WriteAheadLog::new(temp_dir.path().join("test.wal"))
                .await
                .unwrap(),
        );
        let config = SyncWorkerConfig::default();

        let (worker, tx) = SyncWorker::new(hot, cold, wal, config);

        // Worker should be created
        assert!(tx.capacity() > 0);

        // Verify we can send commands
        let (done_tx, done_rx) = oneshot::channel();

        // Spawn the worker
        let handle = tokio::spawn(worker.run());

        // Send shutdown
        tx.send(SyncCommand::Shutdown(done_tx)).await.unwrap();

        // Wait for completion
        tokio::time::timeout(Duration::from_secs(5), done_rx)
            .await
            .expect("Shutdown timed out")
            .expect("Shutdown channel closed");

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_sync_worker_sync_operation() {
        let temp_dir = TempDir::new().unwrap();

        let hot = Arc::new(HotMemory::new(HotMemoryConfig::default()));
        let cold = Arc::new(
            ColdMemory::new(temp_dir.path().join("cold.db"))
                .await
                .unwrap(),
        );
        let wal = Arc::new(
            WriteAheadLog::new(temp_dir.path().join("test.wal"))
                .await
                .unwrap(),
        );

        // Insert entries with old access times
        for i in 0..5 {
            let mut entry = MemoryEntry::embedding(Uuid::new_v4(), vec![i as f32]);
            entry.last_accessed = Utc::now() - chrono::Duration::seconds(600);
            hot.insert(entry).await.unwrap();
        }

        assert_eq!(hot.len(), 5);
        assert_eq!(cold.len(), 0);

        let config = SyncWorkerConfig {
            sync_interval: Duration::from_secs(1),
            hot_to_cold_age: Duration::from_secs(1), // 1 second threshold
            batch_size: 10,
        };

        let (worker, tx) = SyncWorker::new(
            Arc::clone(&hot),
            Arc::clone(&cold),
            Arc::clone(&wal),
            config,
        );

        let handle = tokio::spawn(worker.run());

        // Trigger sync
        tx.send(SyncCommand::Sync).await.unwrap();

        // Give it a moment to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Graceful shutdown
        let (done_tx, done_rx) = oneshot::channel();
        tx.send(SyncCommand::Shutdown(done_tx)).await.unwrap();
        done_rx.await.unwrap();
        handle.await.unwrap();

        // Entries should have moved from hot to cold
        assert_eq!(hot.len(), 0);
        assert_eq!(cold.len(), 5);
    }

    #[tokio::test]
    async fn test_spawn_sync_worker() {
        let temp_dir = TempDir::new().unwrap();

        let hot = Arc::new(HotMemory::new(HotMemoryConfig::default()));
        let cold = Arc::new(
            ColdMemory::new(temp_dir.path().join("cold.db"))
                .await
                .unwrap(),
        );
        let wal = Arc::new(
            WriteAheadLog::new(temp_dir.path().join("test.wal"))
                .await
                .unwrap(),
        );
        let config = SyncWorkerConfig::default();

        let (tx, handle) = spawn_sync_worker(hot, cold, wal, config);

        // Graceful shutdown
        let (done_tx, done_rx) = oneshot::channel();
        tx.send(SyncCommand::Shutdown(done_tx)).await.unwrap();
        done_rx.await.unwrap();
        handle.await.unwrap();
    }
}
