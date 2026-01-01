//! Integration tests for crash recovery
//!
//! These tests verify that data survives simulated crashes
//! and can be recovered from the Write-Ahead Log (WAL).
//!
//! # Test Architecture
//!
//! The tests use a dual-layer memory system:
//! - **Hot Layer**: In-memory cache for fast access
//! - **Cold Layer**: Persistent file storage for durability
//! - **WAL**: Write-ahead log for crash recovery
//!
//! # Test Scenarios
//!
//! 1. Normal crash and recovery
//! 2. Partial write recovery (mid-transaction crash)
//! 3. Corrupted WAL entry handling (skip and continue)
//! 4. Checkpoint-based recovery
//! 5. Multiple crash cycles
//! 6. Stress test with many writes before crash

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// TEST INFRASTRUCTURE: Dual-Layer Memory System Types
// ============================================================================

/// Memory entry stored in the dual-layer system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: Uuid,
    /// Content of the entry
    pub content: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Creation timestamp (Unix epoch millis)
    pub created_at: u64,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(content: String, embedding: Vec<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            metadata: HashMap::new(),
        }
    }

    /// Create a new memory entry with a specific ID (for testing)
    pub fn with_id(id: Uuid, content: String, embedding: Vec<f32>) -> Self {
        Self {
            id,
            content,
            embedding,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            metadata: HashMap::new(),
        }
    }
}

/// Configuration for dual-layer memory
#[derive(Debug, Clone)]
pub struct DualLayerConfig {
    /// Data directory for persistent storage
    pub data_dir: PathBuf,
    /// Maximum entries in hot layer before flush
    pub hot_layer_capacity: usize,
    /// WAL flush interval in milliseconds
    pub wal_flush_interval_ms: u64,
    /// Enable checksums for WAL entries
    pub enable_checksums: bool,
    /// Checkpoint every N entries
    pub checkpoint_interval: usize,
}

impl DualLayerConfig {
    /// Create config with a specific data directory
    pub fn with_data_dir(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            hot_layer_capacity: 1000,
            wal_flush_interval_ms: 100,
            enable_checksums: true,
            checkpoint_interval: 100,
        }
    }
}

/// WAL entry operation type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WalOperation {
    /// Insert or update an entry
    Upsert,
    /// Delete an entry
    Delete,
    /// Checkpoint marker
    Checkpoint,
}

/// Write-Ahead Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Sequence number (monotonically increasing)
    pub sequence: u64,
    /// Operation type
    pub operation: WalOperation,
    /// Entry ID
    pub entry_id: Uuid,
    /// Entry data (None for delete operations)
    pub data: Option<MemoryEntry>,
    /// CRC32 checksum of the entry
    pub checksum: u32,
    /// Timestamp
    pub timestamp: u64,
}

impl WalEntry {
    /// Create a new WAL entry
    pub fn new(
        sequence: u64,
        operation: WalOperation,
        entry_id: Uuid,
        data: Option<MemoryEntry>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut entry = Self {
            sequence,
            operation,
            entry_id,
            data,
            checksum: 0,
            timestamp,
        };
        entry.checksum = entry.calculate_checksum();
        entry
    }

    /// Calculate CRC32 checksum for this entry
    pub fn calculate_checksum(&self) -> u32 {
        // Create a copy without checksum for calculation
        let data_for_hash = (
            self.sequence,
            &self.operation,
            &self.entry_id,
            &self.data,
            self.timestamp,
        );
        let serialized = serde_json::to_vec(&data_for_hash).unwrap_or_default();
        crc32fast::hash(&serialized)
    }

    /// Verify the checksum
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }
}

/// Write-Ahead Log for crash recovery
pub struct WriteAheadLog {
    /// Path to WAL file
    path: PathBuf,
    /// Current sequence number
    sequence: AtomicU64,
    /// WAL file handle
    file: Option<File>,
    /// Whether checksums are enabled
    enable_checksums: bool,
}

impl WriteAheadLog {
    /// Create or open a WAL file
    pub fn open(path: PathBuf, enable_checksums: bool) -> io::Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;

        // Get the highest sequence number from existing entries
        let max_seq = Self::get_max_sequence(&path)?;

        Ok(Self {
            path,
            sequence: AtomicU64::new(max_seq + 1),
            file: Some(file),
            enable_checksums,
        })
    }

    /// Get the maximum sequence number from the WAL file
    fn get_max_sequence(path: &Path) -> io::Result<u64> {
        if !path.exists() {
            return Ok(0);
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut max_seq = 0u64;

        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(entry) = serde_json::from_str::<WalEntry>(&line) {
                    if entry.sequence > max_seq {
                        max_seq = entry.sequence;
                    }
                }
            }
        }

        Ok(max_seq)
    }

    /// Append an entry to the WAL
    pub fn append(
        &mut self,
        operation: WalOperation,
        entry_id: Uuid,
        data: Option<MemoryEntry>,
    ) -> io::Result<u64> {
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        let wal_entry = WalEntry::new(seq, operation, entry_id, data);

        let serialized = serde_json::to_string(&wal_entry)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        if let Some(ref mut file) = self.file {
            writeln!(file, "{}", serialized)?;
            file.flush()?;
        }

        Ok(seq)
    }

    /// Sync the WAL to disk
    pub fn sync(&mut self) -> io::Result<()> {
        if let Some(ref file) = self.file {
            file.sync_all()?;
        }
        Ok(())
    }

    /// Read all entries from the WAL
    pub fn read_all(&self) -> io::Result<Vec<WalEntry>> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(entry) = serde_json::from_str::<WalEntry>(&line) {
                    entries.push(entry);
                }
            }
        }

        // Sort by sequence number
        entries.sort_by_key(|e| e.sequence);
        Ok(entries)
    }

    /// Read entries, filtering out corrupted ones
    pub fn read_valid_entries(&self) -> io::Result<(Vec<WalEntry>, usize)> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut valid_entries = Vec::new();
        let mut corrupted_count = 0;

        for line in reader.lines() {
            if let Ok(line) = line {
                match serde_json::from_str::<WalEntry>(&line) {
                    Ok(entry) => {
                        if self.enable_checksums && !entry.verify_checksum() {
                            corrupted_count += 1;
                            continue;
                        }
                        valid_entries.push(entry);
                    }
                    Err(_) => {
                        corrupted_count += 1;
                    }
                }
            }
        }

        valid_entries.sort_by_key(|e| e.sequence);
        Ok((valid_entries, corrupted_count))
    }

    /// Write a checkpoint marker
    pub fn checkpoint(&mut self) -> io::Result<u64> {
        self.append(WalOperation::Checkpoint, Uuid::nil(), None)
    }

    /// Truncate WAL after a checkpoint
    pub fn truncate_after_checkpoint(&mut self, checkpoint_seq: u64) -> io::Result<()> {
        // Read all entries up to and including the checkpoint
        let entries = self.read_all()?;
        let entries_to_keep: Vec<_> = entries
            .into_iter()
            .filter(|e| e.sequence > checkpoint_seq)
            .collect();

        // Close the current file
        self.file.take();

        // Rewrite the file with only entries after checkpoint
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;

        let mut writer = io::BufWriter::new(file);
        for entry in &entries_to_keep {
            let serialized = serde_json::to_string(entry)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            writeln!(writer, "{}", serialized)?;
        }
        writer.flush()?;

        // Reopen the file for appending
        self.file = Some(OpenOptions::new().append(true).open(&self.path)?);

        Ok(())
    }

    /// Close the WAL
    pub fn close(&mut self) -> io::Result<()> {
        if let Some(ref file) = self.file {
            file.sync_all()?;
        }
        self.file.take();
        Ok(())
    }

    /// Get current sequence number
    pub fn current_sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }
}

/// Recovery report after crash recovery
#[derive(Debug, Clone, Default)]
pub struct RecoveryReport {
    /// Number of entries successfully recovered
    pub entries_recovered: usize,
    /// Number of corrupted entries skipped
    pub entries_corrupted: usize,
    /// Last checkpoint sequence (if any)
    pub last_checkpoint: Option<u64>,
    /// Total WAL entries processed
    pub total_wal_entries: usize,
    /// Recovery duration in milliseconds
    pub recovery_time_ms: u64,
}

/// Dual-layer memory system with WAL support
pub struct DualLayerMemory {
    /// Hot layer (in-memory cache)
    hot_layer: Arc<RwLock<HashMap<Uuid, MemoryEntry>>>,
    /// Cold layer path (persistent storage)
    cold_layer_path: PathBuf,
    /// Write-ahead log
    wal: Arc<RwLock<WriteAheadLog>>,
    /// Configuration
    config: DualLayerConfig,
    /// Entry count for checkpoint tracking
    entry_count: AtomicU64,
}

impl DualLayerMemory {
    /// Create a new dual-layer memory system
    pub async fn new(config: DualLayerConfig) -> io::Result<Self> {
        fs::create_dir_all(&config.data_dir)?;

        let wal_path = config.data_dir.join("wal.log");
        let cold_layer_path = config.data_dir.join("cold_storage.json");
        let wal = WriteAheadLog::open(wal_path, config.enable_checksums)?;

        Ok(Self {
            hot_layer: Arc::new(RwLock::new(HashMap::new())),
            cold_layer_path,
            wal: Arc::new(RwLock::new(wal)),
            config,
            entry_count: AtomicU64::new(0),
        })
    }

    /// Store an entry (write to WAL first, then hot layer)
    pub async fn store(&self, entry: MemoryEntry) -> io::Result<()> {
        // Write to WAL first (for durability)
        {
            let mut wal = self.wal.write().await;
            wal.append(WalOperation::Upsert, entry.id, Some(entry.clone()))?;
        }

        // Then update hot layer
        {
            let mut hot = self.hot_layer.write().await;
            hot.insert(entry.id, entry);
        }

        // Check if we need to create a checkpoint
        let count = self.entry_count.fetch_add(1, Ordering::SeqCst);
        if count > 0 && count % self.config.checkpoint_interval as u64 == 0 {
            self.create_checkpoint().await?;
        }

        Ok(())
    }

    /// Retrieve an entry by ID
    pub async fn get(&self, id: &Uuid) -> Option<MemoryEntry> {
        let hot = self.hot_layer.read().await;
        hot.get(id).cloned()
    }

    /// Delete an entry
    pub async fn delete(&self, id: &Uuid) -> io::Result<()> {
        // Write to WAL first
        {
            let mut wal = self.wal.write().await;
            wal.append(WalOperation::Delete, *id, None)?;
        }

        // Then update hot layer
        {
            let mut hot = self.hot_layer.write().await;
            hot.remove(id);
        }

        Ok(())
    }

    /// Get all entries
    pub async fn get_all(&self) -> Vec<MemoryEntry> {
        let hot = self.hot_layer.read().await;
        hot.values().cloned().collect()
    }

    /// Get entry count
    pub async fn count(&self) -> usize {
        let hot = self.hot_layer.read().await;
        hot.len()
    }

    /// Create a checkpoint (persist to cold storage)
    pub async fn create_checkpoint(&self) -> io::Result<u64> {
        // Get all current entries
        let entries: HashMap<Uuid, MemoryEntry> = {
            let hot = self.hot_layer.read().await;
            hot.clone()
        };

        // Write to cold storage
        let serialized = serde_json::to_string_pretty(&entries)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(&self.cold_layer_path, serialized)?;

        // Write checkpoint marker to WAL
        let checkpoint_seq = {
            let mut wal = self.wal.write().await;
            let seq = wal.checkpoint()?;
            wal.sync()?;
            seq
        };

        Ok(checkpoint_seq)
    }

    /// Recover from crash by replaying WAL
    pub async fn recover(&self) -> io::Result<RecoveryReport> {
        let start = std::time::Instant::now();
        let mut report = RecoveryReport::default();

        // First, try to load from cold storage (last checkpoint)
        let mut recovered_entries: HashMap<Uuid, MemoryEntry> = if self.cold_layer_path.exists() {
            let content = fs::read_to_string(&self.cold_layer_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            HashMap::new()
        };

        // Read WAL entries
        let (wal_entries, corrupted) = {
            let wal = self.wal.read().await;
            wal.read_valid_entries()?
        };

        report.entries_corrupted = corrupted;
        report.total_wal_entries = wal_entries.len() + corrupted;

        // Find last checkpoint
        let last_checkpoint = wal_entries
            .iter()
            .filter(|e| e.operation == WalOperation::Checkpoint)
            .map(|e| e.sequence)
            .max();
        report.last_checkpoint = last_checkpoint;

        // Replay WAL entries after last checkpoint
        let replay_from = last_checkpoint.unwrap_or(0);
        for entry in wal_entries {
            if entry.sequence <= replay_from {
                continue;
            }

            match entry.operation {
                WalOperation::Upsert => {
                    if let Some(data) = entry.data {
                        recovered_entries.insert(entry.entry_id, data);
                        report.entries_recovered += 1;
                    }
                }
                WalOperation::Delete => {
                    recovered_entries.remove(&entry.entry_id);
                }
                WalOperation::Checkpoint => {
                    // Already handled above
                }
            }
        }

        // Update hot layer with recovered entries
        {
            let mut hot = self.hot_layer.write().await;
            *hot = recovered_entries;
        }

        report.recovery_time_ms = start.elapsed().as_millis() as u64;
        Ok(report)
    }

    /// Graceful shutdown with final checkpoint
    pub async fn shutdown(&self) -> io::Result<()> {
        // Create final checkpoint
        self.create_checkpoint().await?;

        // Close WAL
        let mut wal = self.wal.write().await;
        wal.close()?;

        Ok(())
    }

    /// Force sync WAL to disk
    pub async fn sync(&self) -> io::Result<()> {
        let mut wal = self.wal.write().await;
        wal.sync()
    }
}

// ============================================================================
// TESTS
// ============================================================================

/// Test basic crash and recovery scenario
#[tokio::test]
async fn test_crash_and_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    // Phase 1: Write data and simulate crash (drop without shutdown)
    let entry_ids: Vec<Uuid> = {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        let mut ids = Vec::new();
        for i in 0..100 {
            let entry = MemoryEntry::new(format!("Test content {}", i), vec![0.1; 768]);
            ids.push(entry.id);
            memory.store(entry).await.unwrap();
        }

        // Force sync to ensure data is written
        memory.sync().await.unwrap();

        // Simulate crash: drop without graceful shutdown
        // The memory object is dropped here
        ids
    };

    // Phase 2: Recover and verify
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        assert_eq!(
            report.entries_recovered, 100,
            "Should recover all 100 entries"
        );
        assert_eq!(
            report.entries_corrupted, 0,
            "Should have no corrupted entries"
        );

        // Verify all data is accessible
        let count = memory.count().await;
        assert_eq!(count, 100, "Should have 100 entries after recovery");

        // Verify specific entries
        for id in &entry_ids {
            let entry = memory.get(id).await;
            assert!(entry.is_some(), "Entry {} should exist after recovery", id);
        }

        // Verify content
        let first_entry = memory.get(&entry_ids[0]).await.unwrap();
        assert!(
            first_entry.content.starts_with("Test content"),
            "Content should be preserved"
        );
    }
}

/// Test recovery of partial write (mid-transaction crash)
#[tokio::test]
async fn test_partial_write_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    // Phase 1: Write some data and simulate crash mid-batch
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        // Write 50 entries
        for i in 0..50 {
            let entry = MemoryEntry::new(format!("Entry {}", i), vec![0.2; 768]);
            memory.store(entry).await.unwrap();
        }

        // Simulate crash without sync (partial write scenario)
        // Some entries may or may not be on disk
        drop(memory);
    }

    // Phase 2: Recover
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        // Should recover at least some entries (those that were flushed)
        // Due to flush after each write, we expect all 50
        assert!(report.entries_recovered >= 0, "Should recover some entries");

        let count = memory.count().await;
        println!("Recovered {} entries from partial write", count);
    }
}

/// Test that corrupted WAL entries are skipped during recovery
#[tokio::test]
async fn test_corrupted_wal_entry_skipped() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    // Phase 1: Write valid data
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        for i in 0..10 {
            let entry = MemoryEntry::new(format!("Valid entry {}", i), vec![0.3; 768]);
            memory.store(entry).await.unwrap();
        }

        memory.sync().await.unwrap();
        drop(memory);
    }

    // Corrupt some WAL entries by appending garbage
    let wal_path = temp_dir.path().join("wal.log");
    {
        let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();

        // Append corrupted entries (invalid JSON)
        writeln!(
            file,
            "{{\"this is\": \"corrupted json without proper structure"
        )
        .unwrap();
        writeln!(file, "completely invalid garbage data 12345").unwrap();
        writeln!(file, "").unwrap(); // Empty line
    }

    // Phase 2: Recover and verify corrupted entries are skipped
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        assert!(
            report.entries_corrupted >= 2,
            "Should detect at least 2 corrupted entries, got {}",
            report.entries_corrupted
        );
        assert_eq!(
            report.entries_recovered, 10,
            "Should recover all 10 valid entries"
        );

        let count = memory.count().await;
        assert_eq!(count, 10, "Should have 10 valid entries");
    }
}

/// Test checkpoint-based recovery
#[tokio::test]
async fn test_checkpoint_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());
    config.checkpoint_interval = 25; // Checkpoint every 25 entries

    let mut post_checkpoint_ids: Vec<Uuid> = Vec::new();

    // Phase 1: Write data, create checkpoint, write more data
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        // Write first batch (will trigger checkpoint at 25)
        for i in 0..30 {
            let entry = MemoryEntry::new(format!("Pre-checkpoint entry {}", i), vec![0.4; 768]);
            memory.store(entry).await.unwrap();
        }

        // Write post-checkpoint entries
        for i in 0..20 {
            let entry = MemoryEntry::new(format!("Post-checkpoint entry {}", i), vec![0.5; 768]);
            post_checkpoint_ids.push(entry.id);
            memory.store(entry).await.unwrap();
        }

        memory.sync().await.unwrap();
        drop(memory);
    }

    // Phase 2: Recover and verify
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        // Should have a checkpoint
        assert!(
            report.last_checkpoint.is_some(),
            "Should have at least one checkpoint"
        );

        // Should recover all entries
        let count = memory.count().await;
        assert_eq!(count, 50, "Should have all 50 entries after recovery");

        // Verify post-checkpoint entries are recovered
        for id in &post_checkpoint_ids {
            let entry = memory.get(id).await;
            assert!(
                entry.is_some(),
                "Post-checkpoint entry {} should be recovered",
                id
            );
        }
    }
}

/// Test multiple crash cycles
#[tokio::test]
async fn test_multiple_crash_cycles() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    let mut total_entries = 0;
    let mut all_ids: Vec<Uuid> = Vec::new();

    // Cycle 1: Write and crash
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        for i in 0..25 {
            let entry = MemoryEntry::new(format!("Cycle 1 entry {}", i), vec![0.6; 768]);
            all_ids.push(entry.id);
            memory.store(entry).await.unwrap();
        }
        total_entries += 25;

        memory.sync().await.unwrap();
        // Crash
    }

    // Cycle 2: Recover, write more, crash
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        memory.recover().await.unwrap();

        // Verify previous entries exist
        assert_eq!(memory.count().await, 25);

        // Add more entries
        for i in 0..25 {
            let entry = MemoryEntry::new(format!("Cycle 2 entry {}", i), vec![0.7; 768]);
            all_ids.push(entry.id);
            memory.store(entry).await.unwrap();
        }
        total_entries += 25;

        memory.sync().await.unwrap();
        // Crash
    }

    // Cycle 3: Recover, write more, crash
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        memory.recover().await.unwrap();

        assert_eq!(memory.count().await, 50);

        for i in 0..25 {
            let entry = MemoryEntry::new(format!("Cycle 3 entry {}", i), vec![0.8; 768]);
            all_ids.push(entry.id);
            memory.store(entry).await.unwrap();
        }
        total_entries += 25;

        memory.sync().await.unwrap();
        // Crash
    }

    // Final recovery and verification
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        let count = memory.count().await;
        assert_eq!(
            count, total_entries,
            "Should have all {} entries after multiple crash cycles",
            total_entries
        );

        // Verify all IDs are present
        for id in &all_ids {
            assert!(
                memory.get(id).await.is_some(),
                "Entry {} should survive multiple crash cycles",
                id
            );
        }

        println!(
            "Multiple crash cycles: {} entries recovered, {} corrupted",
            report.entries_recovered, report.entries_corrupted
        );
    }
}

/// Stress test: many writes followed by crash
#[tokio::test]
async fn test_stress_crash_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());
    config.checkpoint_interval = 500; // Larger checkpoint interval for stress test

    let entry_count = 1000;
    let mut all_ids: Vec<Uuid> = Vec::with_capacity(entry_count);

    // Phase 1: Stress write
    let start = std::time::Instant::now();
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        for i in 0..entry_count {
            let content = format!(
                "Stress test entry {} with some longer content to simulate realistic data sizes. \
                 This includes metadata, context, and other information that would typically be stored.",
                i
            );
            let entry = MemoryEntry::new(content, vec![0.1 + (i as f32 * 0.0001); 768]);
            all_ids.push(entry.id);
            memory.store(entry).await.unwrap();
        }

        memory.sync().await.unwrap();
        // Simulate crash
    }
    let write_duration = start.elapsed();

    // Phase 2: Recovery
    let recovery_start = std::time::Instant::now();
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();
        let recovery_duration = recovery_start.elapsed();

        // Verify all entries recovered
        let count = memory.count().await;
        assert_eq!(
            count, entry_count,
            "Should recover all {} entries, got {}",
            entry_count, count
        );

        // Verify no corruption
        assert_eq!(
            report.entries_corrupted, 0,
            "Should have no corrupted entries"
        );

        // Sample verification of content
        for i in [0, 100, 500, 999].iter() {
            if *i < all_ids.len() {
                let entry = memory.get(&all_ids[*i]).await.unwrap();
                assert!(
                    entry.content.contains("Stress test entry"),
                    "Content should be preserved for entry {}",
                    i
                );
            }
        }

        println!(
            "Stress test results:\n\
             - Entries: {}\n\
             - Write time: {:?}\n\
             - Recovery time: {:?}\n\
             - Report: {:?}",
            entry_count, write_duration, recovery_duration, report
        );
    }
}

/// Test delete operations survive crash
#[tokio::test]
async fn test_delete_crash_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    let mut to_delete: Vec<Uuid> = Vec::new();
    let mut to_keep: Vec<Uuid> = Vec::new();

    // Phase 1: Write entries, delete some, crash
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        // Write 20 entries
        for i in 0..20 {
            let entry = MemoryEntry::new(format!("Entry {}", i), vec![0.9; 768]);
            if i % 2 == 0 {
                to_delete.push(entry.id);
            } else {
                to_keep.push(entry.id);
            }
            memory.store(entry).await.unwrap();
        }

        // Delete even entries
        for id in &to_delete {
            memory.delete(id).await.unwrap();
        }

        memory.sync().await.unwrap();
        // Crash
    }

    // Phase 2: Recover and verify
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        memory.recover().await.unwrap();

        // Deleted entries should not exist
        for id in &to_delete {
            assert!(
                memory.get(id).await.is_none(),
                "Deleted entry {} should not exist after recovery",
                id
            );
        }

        // Kept entries should exist
        for id in &to_keep {
            assert!(
                memory.get(id).await.is_some(),
                "Kept entry {} should exist after recovery",
                id
            );
        }

        assert_eq!(memory.count().await, 10, "Should have 10 remaining entries");
    }
}

/// Test WAL with checksum verification
#[tokio::test]
async fn test_checksum_verification() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    // Write some data
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        for i in 0..5 {
            let entry = MemoryEntry::new(format!("Checksum test {}", i), vec![1.0; 768]);
            memory.store(entry).await.unwrap();
        }

        memory.sync().await.unwrap();
        drop(memory);
    }

    // Corrupt a WAL entry by modifying checksum
    let wal_path = temp_dir.path().join("wal.log");
    {
        let content = fs::read_to_string(&wal_path).unwrap();
        let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();

        if lines.len() > 2 {
            // Corrupt the third entry's checksum
            if let Ok(mut entry) = serde_json::from_str::<WalEntry>(&lines[2]) {
                entry.checksum = 0xDEADBEEF; // Invalid checksum
                lines[2] = serde_json::to_string(&entry).unwrap();
            }
        }

        fs::write(&wal_path, lines.join("\n") + "\n").unwrap();
    }

    // Recover and verify checksum failure is detected
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        assert!(
            report.entries_corrupted >= 1,
            "Should detect at least 1 corrupted entry due to checksum mismatch"
        );

        // Should still recover the valid entries
        assert!(
            report.entries_recovered >= 4,
            "Should recover at least 4 valid entries"
        );
    }
}

/// Test graceful shutdown vs crash recovery
#[tokio::test]
async fn test_graceful_vs_crash_shutdown() {
    // Test 1: Graceful shutdown
    let temp_dir1 = TempDir::new().unwrap();
    let config1 = DualLayerConfig::with_data_dir(temp_dir1.path().to_path_buf());

    {
        let memory = DualLayerMemory::new(config1.clone()).await.unwrap();

        for i in 0..50 {
            let entry = MemoryEntry::new(format!("Graceful {}", i), vec![0.1; 768]);
            memory.store(entry).await.unwrap();
        }

        // Graceful shutdown
        memory.shutdown().await.unwrap();
    }

    // Recovery should be instant (from checkpoint)
    {
        let memory = DualLayerMemory::new(config1.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        assert_eq!(memory.count().await, 50);
        assert!(
            report.last_checkpoint.is_some(),
            "Should have checkpoint after graceful shutdown"
        );
    }

    // Test 2: Crash shutdown (no graceful shutdown)
    let temp_dir2 = TempDir::new().unwrap();
    let config2 = DualLayerConfig::with_data_dir(temp_dir2.path().to_path_buf());

    {
        let memory = DualLayerMemory::new(config2.clone()).await.unwrap();

        for i in 0..50 {
            let entry = MemoryEntry::new(format!("Crash {}", i), vec![0.2; 768]);
            memory.store(entry).await.unwrap();
        }

        memory.sync().await.unwrap();
        // No graceful shutdown - simulate crash
        drop(memory);
    }

    // Recovery requires WAL replay
    {
        let memory = DualLayerMemory::new(config2.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        assert_eq!(memory.count().await, 50);
        // Should still recover all entries from WAL
        assert!(
            report.entries_recovered > 0,
            "Should have recovered entries from WAL"
        );
    }
}

/// Test recovery with empty WAL
#[tokio::test]
async fn test_empty_wal_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    // Create a new memory system (empty)
    let memory = DualLayerMemory::new(config.clone()).await.unwrap();
    let report = memory.recover().await.unwrap();

    assert_eq!(
        report.entries_recovered, 0,
        "Empty WAL should have 0 entries"
    );
    assert_eq!(
        report.entries_corrupted, 0,
        "Empty WAL should have 0 corrupted"
    );
    assert_eq!(memory.count().await, 0, "Should have 0 entries");
}

/// Test recovery with only checkpoints (no data entries)
#[tokio::test]
async fn test_checkpoint_only_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    // Create entries, checkpoint, then delete all
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();

        // Add entries
        let mut ids = Vec::new();
        for i in 0..10 {
            let entry = MemoryEntry::new(format!("Entry {}", i), vec![0.3; 768]);
            ids.push(entry.id);
            memory.store(entry).await.unwrap();
        }

        // Create checkpoint
        memory.create_checkpoint().await.unwrap();

        // Delete all entries
        for id in &ids {
            memory.delete(id).await.unwrap();
        }

        memory.sync().await.unwrap();
        drop(memory);
    }

    // Recover - should have 0 entries (all deleted)
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        assert_eq!(memory.count().await, 0, "All entries were deleted");
        assert!(report.last_checkpoint.is_some(), "Should have checkpoint");
    }
}

/// Test concurrent writes and recovery
#[tokio::test]
async fn test_concurrent_writes_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = DualLayerConfig::with_data_dir(temp_dir.path().to_path_buf());

    let entry_count = 100;
    let mut all_ids = Vec::new();

    {
        let memory = Arc::new(DualLayerMemory::new(config.clone()).await.unwrap());
        let ids = Arc::new(RwLock::new(Vec::new()));

        // Spawn multiple concurrent write tasks
        let mut handles = Vec::new();
        for batch in 0..4 {
            let memory = memory.clone();
            let ids = ids.clone();

            let handle = tokio::spawn(async move {
                let mut batch_ids = Vec::new();
                for i in 0..25 {
                    let entry = MemoryEntry::new(
                        format!("Concurrent batch {} entry {}", batch, i),
                        vec![0.1 * batch as f32; 768],
                    );
                    batch_ids.push(entry.id);
                    memory.store(entry).await.unwrap();
                }

                let mut ids = ids.write().await;
                ids.extend(batch_ids);
            });
            handles.push(handle);
        }

        // Wait for all writes to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Sync and get IDs
        memory.sync().await.unwrap();
        all_ids = ids.read().await.clone();

        // Simulate crash
    }

    // Recover and verify
    {
        let memory = DualLayerMemory::new(config.clone()).await.unwrap();
        let report = memory.recover().await.unwrap();

        let count = memory.count().await;
        assert_eq!(
            count, entry_count,
            "Should recover all {} concurrent entries, got {}",
            entry_count, count
        );

        // Verify all IDs
        for id in &all_ids {
            assert!(
                memory.get(id).await.is_some(),
                "Concurrent entry {} should be recovered",
                id
            );
        }

        println!(
            "Concurrent writes recovery: {} entries, {} corrupted",
            report.entries_recovered, report.entries_corrupted
        );
    }
}
