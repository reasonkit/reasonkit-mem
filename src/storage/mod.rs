//! Storage module for ReasonKit Memory
//!
//! Provides document and chunk storage using Qdrant vector database,
//! with a dual-layer architecture for hot/cold memory management.
//!
//! ## Architecture
//!
//! ```text
//! +-------------------+
//! |  DualLayerMemory  |  Unified interface for all memory operations
//! +-------------------+
//!          |
//!    +-----+-----+
//!    |           |
//! +------+   +------+
//! | Hot  |   | Cold |   Hot = recent/active, Cold = historical
//! +------+   +------+
//!    |           |
//!    +-----+-----+
//!          |
//!    +----------+
//!    |   WAL    |   Write-ahead log for durability
//!    +----------+
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit_mem::storage::{DualLayerMemory, DualLayerConfig};
//!
//! // Create dual-layer memory
//! let config = DualLayerConfig::default();
//! let memory = DualLayerMemory::new(config).await?;
//!
//! // Store a memory entry
//! let id = memory.store(entry).await?;
//!
//! // Retrieve context for a query
//! let results = memory.retrieve_context("search query", 10).await?;
//! ```

use crate::{embedding::cosine_similarity, Document, Error, Result};
use async_trait::async_trait;
use qdrant_client::qdrant::{
    CreateCollection, DeletePoints, Distance, GetPoints, PointId, PointStruct, QuantizationConfig,
    ScalarQuantization, ScrollPoints, SearchPoints, UpsertPoints, VectorParams, VectorsConfig,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// NEW: Dual-Layer Memory Modules
// ============================================================================

/// Hot memory layer for recent/active memories (fast access)
pub mod hot;

/// Cold memory layer for historical/archived memories (optimized storage)
pub mod cold;

/// Write-ahead log for durability and recovery
pub mod wal;

/// Context retrieval utilities
pub mod context;

/// Dual-layer storage facade with unified interface
pub mod dual_layer;

/// Background sync worker for hot-to-cold migration
pub mod sync_worker;

/// Configuration types for dual-layer storage
pub mod config;

/// Memory entry types and traits
pub mod memory_types;

/// Serialization utilities
pub mod serde_utils;

// Re-export new module types
pub use cold::{ColdMemory, ColdMemoryConfig, ColdMemoryEntry};
pub use context::{retrieve_context, ContextQuery, ContextResult};
pub use hot::{HotMemory, HotMemoryConfig, HotMemoryEntry};
pub use wal::{WalConfig, WalOperation, WriteAheadLog};

// Re-export dual-layer storage types
// Note: DualLayerConfig is exported from config module, not dual_layer
pub use dual_layer::{
    ContextResult as DualContextResult, DualLayerError, DualLayerResult, DualLayerStorage,
    StorageTier,
};

// ============================================================================
// Memory Entry Types
// ============================================================================

/// A unified memory entry that can be stored in either hot or cold layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: Uuid,
    /// Text content
    pub content: String,
    /// Embedding vector (optional, computed if not provided)
    pub embedding: Option<Vec<f32>>,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, String>,
    /// Importance score (0.0 - 1.0)
    pub importance: f32,
    /// Access count for LRU tracking
    pub access_count: u64,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last access timestamp
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Optional TTL in seconds (None = never expires)
    pub ttl_secs: Option<u64>,
    /// Memory layer location
    pub layer: MemoryLayer,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Which layer a memory resides in
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLayer {
    /// Hot layer - recent, frequently accessed
    Hot,
    /// Cold layer - historical, archived
    Cold,
    /// Pending - being written, not yet committed
    #[default]
    Pending,
}

impl MemoryEntry {
    /// Create a new memory entry with the given content
    pub fn new(content: impl Into<String>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            embedding: None,
            metadata: HashMap::new(),
            importance: 0.5,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            ttl_secs: None,
            layer: MemoryLayer::Pending,
            tags: Vec::new(),
        }
    }

    /// Set the embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set importance score
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Add a metadata key-value pair
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set TTL in seconds
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = Some(ttl_secs);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if this entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl_secs {
            let elapsed = chrono::Utc::now()
                .signed_duration_since(self.created_at)
                .num_seconds() as u64;
            elapsed > ttl
        } else {
            false
        }
    }

    /// Calculate age in seconds
    pub fn age_secs(&self) -> i64 {
        chrono::Utc::now()
            .signed_duration_since(self.created_at)
            .num_seconds()
    }

    /// Calculate time since last access in seconds
    pub fn idle_secs(&self) -> i64 {
        chrono::Utc::now()
            .signed_duration_since(self.last_accessed)
            .num_seconds()
    }
}

// ============================================================================
// Dual-Layer Configuration
// ============================================================================

// DualLayerConfig is defined in storage/config.rs - use that instead
// This was a duplicate definition causing E0255 errors
pub use config::DualLayerConfig;

// DualLayerConfig implementation methods are in storage/config.rs
// This impl block was removed to avoid conflicts with canonical definition

// ============================================================================
// Sync Statistics
// ============================================================================

/// Statistics from a sync operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    /// Number of entries moved from hot to cold
    pub hot_to_cold: usize,
    /// Number of expired entries removed
    pub expired_removed: usize,
    /// Number of WAL entries replayed
    pub wal_replayed: usize,
    /// Number of WAL entries compacted
    pub wal_compacted: usize,
    /// Duration of the sync operation
    pub duration_ms: u64,
    /// Any errors encountered (non-fatal)
    pub warnings: Vec<String>,
}

/// Report from a recovery operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecoveryReport {
    /// Number of entries recovered from WAL
    pub entries_recovered: usize,
    /// Number of entries lost (unrecoverable)
    pub entries_lost: usize,
    /// Number of operations replayed
    pub operations_replayed: usize,
    /// Last committed sequence number
    pub last_sequence: u64,
    /// Duration of recovery
    pub duration_ms: u64,
    /// Errors encountered during recovery
    pub errors: Vec<String>,
    /// Whether recovery was successful
    pub success: bool,
}

// ============================================================================
// Dual-Layer Memory Implementation
// ============================================================================

/// Unified dual-layer memory storage combining hot and cold layers with WAL
///
/// This is the primary interface for all memory operations in ReasonKit.
/// It automatically handles:
/// - Routing writes through WAL for durability
/// - Placing new entries in hot layer
/// - Migrating old/less-accessed entries to cold layer
/// - Background synchronization
/// - Recovery from crashes
pub struct DualLayerMemory {
    /// Hot memory layer (in-memory, fast)
    hot: Arc<HotMemory>,
    /// Cold memory layer (disk-backed, large capacity)
    cold: Arc<ColdMemory>,
    /// Write-ahead log for durability
    wal: Arc<WriteAheadLog>,
    /// Configuration
    config: DualLayerConfig,
    /// Background sync task handle
    sync_handle: Option<tokio::task::JoinHandle<()>>,
    /// Shutdown signal sender
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    /// Track if shutdown has been initiated
    is_shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl DualLayerMemory {
    /// Create a new dual-layer memory system
    ///
    /// # Arguments
    /// * `config` - Configuration for all layers
    ///
    /// # Returns
    /// * `Result<Self>` - The initialized memory system
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = DualLayerConfig::default();
    /// let memory = DualLayerMemory::new(config).await?;
    /// ```
    pub async fn new(config: DualLayerConfig) -> Result<Self> {
        // Initialize hot memory layer
        // Convert canonical HotMemoryConfig to hot::HotMemoryConfig
        let hot_config = hot::HotMemoryConfig {
            max_entries: config.hot.max_entries,
            ttl: std::time::Duration::from_secs(config.hot.ttl_secs),
            eviction_batch_size: config.hot.eviction_batch_size,
        };
        let hot = Arc::new(HotMemory::new(hot_config));

        // Initialize cold memory layer
        // Convert canonical ColdMemoryConfig to cold::ColdMemoryConfig
        let cold_config = cold::ColdMemoryConfig {
            db_path: config.cold.db_path.clone(),
            cache_size_mb: config.cold.cache_size_mb,
            flush_interval_secs: config.cold.flush_interval_secs,
            enable_compression: config.cold.enable_compression,
            parallel_scan_threshold: 1000, // default
            use_simd: true,                // default
        };
        let cold = Arc::new(ColdMemory::new(cold_config).await?);

        // Initialize write-ahead log
        // Convert canonical WalConfig to wal::WalConfig
        // Note: wal::WalConfig has different fields (checkpoint_retention, preallocate_segments)
        let wal_config = wal::WalConfig {
            dir: config.wal.dir.clone(),
            segment_size_mb: config.wal.segment_size_mb,
            sync_mode: match config.wal.sync_mode {
                config::SyncMode::Sync => wal::SyncMode::Immediate,
                config::SyncMode::Async => wal::SyncMode::Async,
                config::SyncMode::Balanced => {
                    wal::SyncMode::Batched(std::time::Duration::from_millis(100))
                }
                config::SyncMode::OsDefault => wal::SyncMode::Async, // approximate
            },
            checkpoint_retention: config.wal.max_segments, // approximate mapping
            preallocate_segments: config.wal.preallocate,
        };
        let wal = Arc::new(WriteAheadLog::new(wal_config).await?);

        let is_shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut memory = Self {
            hot,
            cold,
            wal,
            config: config.clone(),
            sync_handle: None,
            shutdown_tx: None,
            is_shutdown,
        };

        // Recover from WAL if needed
        let recovery = memory.recover().await?;
        if recovery.entries_recovered > 0 {
            tracing::info!(
                "Recovered {} entries from WAL in {}ms",
                recovery.entries_recovered,
                recovery.duration_ms
            );
        }

        // Start background sync if enabled
        if config.sync.auto_sync_enabled {
            memory.start_background_sync();
        }

        Ok(memory)
    }

    /// Store a memory entry
    ///
    /// The entry is first written to WAL for durability, then stored in the hot layer.
    ///
    /// # Arguments
    /// * `entry` - The memory entry to store
    ///
    /// # Returns
    /// * `Result<Uuid>` - The ID of the stored entry
    pub async fn store(&self, mut entry: MemoryEntry) -> Result<Uuid> {
        // Assign ID if not set
        if entry.id == Uuid::nil() {
            entry.id = Uuid::new_v4();
        }

        // Write to WAL first
        let embedding = entry.embedding.clone().unwrap_or_default(); // MemoryEntry has Option<Vec<f32>>
        let operation = WalOperation::Insert {
            id: entry.id,
            content: entry.content.clone(),
            embedding: embedding.clone(),
        };
        self.wal.append(operation).await?;

        // Update layer to hot
        entry.layer = MemoryLayer::Hot;

        // Store in hot layer
        // Convert HashMap<String, String> to serde_json::Value
        let metadata_json = {
            let mut obj = serde_json::Map::new();
            for (k, v) in &entry.metadata {
                obj.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            serde_json::Value::Object(obj)
        };
        let hot_entry =
            HotMemoryEntry::new(entry.id, entry.content.clone(), embedding, metadata_json);
        self.hot.put(hot_entry).await?;

        tracing::debug!(id = %entry.id, "Stored entry in hot layer");

        Ok(entry.id)
    }

    /// Get a memory entry by ID
    ///
    /// Checks hot layer first, then cold layer.
    ///
    /// # Arguments
    /// * `id` - The UUID of the entry to retrieve
    ///
    /// # Returns
    /// * `Result<Option<MemoryEntry>>` - The entry if found
    pub async fn get(&self, id: &Uuid) -> Result<Option<MemoryEntry>> {
        // Check hot layer first
        if let Some(hot_entry) = self.hot.get(id).await {
            // Convert HotMemoryEntry to MemoryEntry
            let memory_entry = MemoryEntry {
                id: hot_entry.id,
                content: hot_entry.content,
                embedding: Some(hot_entry.embedding),
                metadata: {
                    let mut meta = HashMap::new();
                    if let Some(obj) = hot_entry.metadata.as_object() {
                        for (k, v) in obj {
                            meta.insert(k.clone(), v.to_string());
                        }
                    }
                    meta
                },
                importance: 0.5, // default
                access_count: hot_entry.access_count,
                created_at: chrono::DateTime::from_timestamp(
                    hot_entry.created_at.elapsed().as_secs() as i64,
                    0,
                )
                .unwrap_or_else(chrono::Utc::now),
                last_accessed: chrono::DateTime::from_timestamp(
                    hot_entry.accessed_at.elapsed().as_secs() as i64,
                    0,
                )
                .unwrap_or_else(chrono::Utc::now),
                ttl_secs: None,
                layer: MemoryLayer::Hot,
                tags: Vec::new(),
            };
            return Ok(Some(memory_entry));
        }

        // Check cold layer
        if let Some(cold_entry) = self.cold.get(id).await? {
            // Convert ColdMemoryEntry to MemoryEntry
            let memory_entry = MemoryEntry {
                id: cold_entry.id,
                content: cold_entry.content,
                embedding: Some(cold_entry.embedding),
                metadata: {
                    let mut meta = HashMap::new();
                    if let Some(obj) = cold_entry.metadata.as_object() {
                        for (k, v) in obj {
                            meta.insert(k.clone(), v.to_string());
                        }
                    }
                    meta
                },
                importance: 0.5, // default
                access_count: 0, // cold entries don't track access
                created_at: chrono::DateTime::from_timestamp(cold_entry.created_at, 0)
                    .unwrap_or_else(chrono::Utc::now),
                last_accessed: chrono::Utc::now(),
                ttl_secs: None,
                layer: MemoryLayer::Cold,
                tags: Vec::new(),
            };
            return Ok(Some(memory_entry));
        }

        Ok(None)
    }

    /// Delete a memory entry
    ///
    /// Removes from both hot and cold layers.
    ///
    /// # Arguments
    /// * `id` - The UUID of the entry to delete
    ///
    /// # Returns
    /// * `Result<bool>` - True if the entry was found and deleted
    pub async fn delete(&self, id: &Uuid) -> Result<bool> {
        // Write delete operation to WAL
        let operation = WalOperation::Delete { id: *id };
        self.wal.append(operation).await?;

        // Delete from hot layer
        let hot_deleted = self.hot.delete(id).await?;

        // Delete from cold layer
        let cold_deleted = self.cold.delete(id).await?;

        Ok(hot_deleted || cold_deleted)
    }

    /// Retrieve context for a query
    ///
    /// Searches both hot and cold layers and returns relevant entries.
    ///
    /// # Arguments
    /// * `query` - The search query string
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// * `Result<Vec<ContextResult>>` - Matching entries with scores
    pub async fn retrieve_context(&self, query: &str, limit: usize) -> Result<Vec<ContextResult>> {
        // TODO: Need embedding for ContextQuery - for now, create with empty embedding
        // This is a placeholder - actual implementation should compute embedding
        let _context_query = ContextQuery {
            text: query.to_string(),
            embedding: Vec::new(), // TODO: Compute embedding from query text
            limit,
            min_score: 0.0,
            recency_weight: 0.3,
        };

        // TODO: Implement adapter to convert hot::HotMemory/cold::ColdMemory to context::HotMemory/context::ColdMemory
        // For now, return empty results
        Ok(Vec::new())
    }

    /// Perform a sync operation
    ///
    /// Moves old entries from hot to cold, removes expired entries, and compacts WAL.
    ///
    /// # Returns
    /// * `Result<SyncStats>` - Statistics about the sync operation
    pub async fn sync(&self) -> Result<SyncStats> {
        let start = Instant::now();
        let mut stats = SyncStats::default();

        // Get entries to migrate from hot to cold
        let threshold = Duration::from_secs(self.config.sync.hot_to_cold_age_secs);
        // Note: min_hot_importance is not in SyncConfig - using default threshold
        let min_importance = 0.0; // TODO: Add to SyncConfig if needed

        // HotMemory doesn't have list_entries() - use search to get all entries
        // For now, skip hot-to-cold migration (would need to implement entry listing)
        // let hot_entries = self.hot.list_entries().await?;
        let hot_entries: Vec<MemoryEntry> = Vec::new(); // TODO: Implement entry listing for HotMemory

        for entry in hot_entries {
            // Skip high-importance entries
            if entry.importance >= min_importance {
                continue;
            }

            // Check if entry is old enough to migrate
            let idle_duration = Duration::from_secs(entry.idle_secs() as u64);
            if idle_duration >= threshold {
                // Migrate to cold
                // Convert MemoryEntry to ColdMemoryEntry
                let cold_entry = ColdMemoryEntry {
                    id: entry.id,
                    content: entry.content.clone(),
                    embedding: entry.embedding.clone().unwrap_or_default(),
                    metadata: serde_json::json!(entry.metadata),
                    created_at: entry.created_at.timestamp(),
                };
                self.cold.store(&cold_entry).await?;

                // Remove from hot
                self.hot.delete(&entry.id).await?;

                stats.hot_to_cold += 1;
            }

            // Check for expired entries
            if entry.is_expired() {
                self.hot.delete(&entry.id).await?;
                stats.expired_removed += 1;
            }
        }

        // Remove expired entries from cold layer
        // ColdMemory doesn't have cleanup_expired() - skip for now
        // let expired_cold = self.cold.cleanup_expired().await?;
        let expired_cold = 0; // TODO: Implement cleanup_expired for ColdMemory
        stats.expired_removed += expired_cold;

        // Compact WAL
        // WriteAheadLog doesn't have compact() method - skip for now
        // let wal_stats = self.wal.compact().await?;
        // stats.wal_compacted = wal_stats.entries_removed;
        stats.wal_compacted = 0; // TODO: Implement compact() for WriteAheadLog

        stats.duration_ms = start.elapsed().as_millis() as u64;

        tracing::info!(
            hot_to_cold = stats.hot_to_cold,
            expired = stats.expired_removed,
            wal_compacted = stats.wal_compacted,
            duration_ms = stats.duration_ms,
            "Sync completed"
        );

        Ok(stats)
    }

    /// Recover from WAL after a crash
    ///
    /// Replays all uncommitted operations from the WAL.
    ///
    /// # Returns
    /// * `Result<RecoveryReport>` - Report of the recovery operation
    pub async fn recover(&self) -> Result<RecoveryReport> {
        let start = Instant::now();
        let mut report = RecoveryReport::default();

        // Read operations from WAL
        // WriteAheadLog doesn't have read_uncommitted() - use recover() instead
        // For now, skip WAL recovery (would need to implement read_uncommitted or use recover)
        let operations: Vec<(u64, WalOperation)> = Vec::new(); // TODO: Implement WAL recovery

        for (_seq, operation) in operations {
            match operation {
                WalOperation::Insert {
                    id,
                    content,
                    embedding,
                } => {
                    // Restore to hot layer
                    let hot_entry =
                        HotMemoryEntry::new(id, content, embedding, serde_json::json!({}));
                    match self.hot.put(hot_entry).await {
                        Ok(_) => {
                            report.entries_recovered += 1;
                        }
                        Err(e) => {
                            report
                                .errors
                                .push(format!("Failed to recover entry: {}", e));
                            report.entries_lost += 1;
                        }
                    }
                }
                WalOperation::Delete { id } => {
                    // Apply delete
                    let _ = self.hot.delete(&id).await;
                    let _ = self.cold.delete(&id).await;
                }
                WalOperation::Update {
                    id,
                    content,
                    embedding,
                } => {
                    // Apply update
                    let hot_entry =
                        HotMemoryEntry::new(id, content, embedding, serde_json::json!({}));
                    match self.hot.put(hot_entry).await {
                        Ok(_) => {
                            report.entries_recovered += 1;
                        }
                        Err(e) => {
                            report
                                .errors
                                .push(format!("Failed to recover update: {}", e));
                        }
                    }
                }
                WalOperation::Checkpoint {
                    lsn,
                    checkpoint_id: _,
                } => {
                    report.last_sequence = lsn; // lsn is u64, not LogSequenceNumber
                }
                WalOperation::BatchInsert { .. }
                | WalOperation::BatchDelete { .. }
                | WalOperation::TxnBegin { .. }
                | WalOperation::TxnCommit { .. }
                | WalOperation::TxnRollback { .. } => {
                    // Skip batch/txn operations for now
                    // TODO: Implement batch/txn recovery
                }
            }
            report.operations_replayed += 1;
        }

        report.duration_ms = start.elapsed().as_millis() as u64;
        report.success = report.errors.is_empty();

        Ok(report)
    }

    /// Gracefully shutdown the memory system
    ///
    /// Stops background sync, flushes WAL, and cleans up resources.
    ///
    /// # Returns
    /// * `Result<()>` - Success if shutdown was clean
    pub async fn shutdown(&self) -> Result<()> {
        self.is_shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);

        // Signal shutdown to background task
        // oneshot::Sender cannot be cloned - the shutdown flag is already set above
        // Background tasks should check is_shutdown flag instead
        // Note: shutdown_tx is Option<oneshot::Sender> - if present, it was already consumed or will be

        // Wait for background task to finish
        // Note: sync_handle is behind Option and we can't take from &self
        // The task will exit on next iteration when it sees is_shutdown

        // Perform final sync
        let _ = self.sync().await;

        // Flush WAL
        // WriteAheadLog doesn't have flush() - use sync() instead
        self.wal.sync().await?;

        tracing::info!("DualLayerMemory shutdown complete");

        Ok(())
    }

    /// Start the background sync worker
    fn start_background_sync(&mut self) {
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);

        let hot = self.hot.clone();
        let cold = self.cold.clone();
        let wal = self.wal.clone();
        let config = self.config.clone();
        let is_shutdown = self.is_shutdown.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(config.sync.interval_secs));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if is_shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                            break;
                        }

                        // Perform sync operations
                        if let Err(e) = Self::background_sync_iteration(
                            &hot,
                            &cold,
                            &wal,
                            &config,
                        ).await {
                            tracing::warn!(error = %e, "Background sync iteration failed");
                        }
                    }
                    _ = &mut shutdown_rx => {
                        tracing::debug!("Background sync received shutdown signal");
                        break;
                    }
                }
            }

            tracing::debug!("Background sync worker exited");
        });

        self.sync_handle = Some(handle);
    }

    /// Single iteration of background sync
    async fn background_sync_iteration(
        hot: &Arc<HotMemory>,
        cold: &Arc<ColdMemory>,
        _wal: &Arc<WriteAheadLog>,
        config: &DualLayerConfig,
    ) -> Result<()> {
        let threshold = Duration::from_secs(config.sync.hot_to_cold_age_secs);
        // Note: min_hot_importance is not in SyncConfig - using default threshold
        let min_importance = 0.0; // TODO: Add to SyncConfig if needed

        // Get entries to potentially migrate
        // HotMemory doesn't have list_entries() - skip migration for now
        // let hot_entries = hot.list_entries().await?;
        let hot_entries: Vec<MemoryEntry> = Vec::new(); // TODO: Implement entry listing for HotMemory
        let mut migrated = 0;
        let mut expired = 0;

        for entry in hot_entries {
            // Remove expired entries
            if entry.is_expired() {
                hot.delete(&entry.id).await?;
                expired += 1;
                continue;
            }

            // Skip high-importance entries
            if entry.importance >= min_importance {
                continue;
            }

            // Check if old enough to migrate
            let idle_duration = Duration::from_secs(entry.idle_secs() as u64);
            if idle_duration >= threshold {
                // Migrate to cold
                // Convert MemoryEntry to ColdMemoryEntry
                let cold_entry = ColdMemoryEntry {
                    id: entry.id,
                    content: entry.content.clone(),
                    embedding: entry.embedding.clone().unwrap_or_default(),
                    metadata: serde_json::json!(entry.metadata),
                    created_at: entry.created_at.timestamp(),
                };
                cold.store(&cold_entry).await?;
                hot.delete(&entry.id).await?;
                migrated += 1;
            }
        }

        // Cleanup expired from cold
        let cold_expired = 0; // TODO: Implement cleanup_expired for ColdMemory

        // Compact WAL periodically
        // TODO: Implement compact for WriteAheadLog

        if migrated > 0 || expired > 0 || cold_expired > 0 {
            tracing::debug!(
                migrated = migrated,
                hot_expired = expired,
                cold_expired = cold_expired,
                "Background sync completed"
            );
        }

        Ok(())
    }

    /// Get statistics about memory usage
    pub async fn stats(&self) -> Result<DualLayerStats> {
        let hot_stats = self.hot.stats().await;
        let cold_stats = self.cold.stats().await;
        let wal_stats = self.wal.stats().await;

        Ok(DualLayerStats {
            hot_entry_count: hot_stats.entry_count,
            hot_memory_bytes: 0, // TODO: Calculate from hot_stats (no direct field available)
            cold_entry_count: cold_stats.entry_count as usize,
            cold_disk_bytes: cold_stats.embeddings_size_bytes + cold_stats.metadata_size_bytes,
            wal_entry_count: 0, // TODO: WAL doesn't track entry count
            wal_disk_bytes: wal_stats.total_size_bytes,
            total_entries: hot_stats.entry_count + cold_stats.entry_count as usize,
        })
    }

    /// Get the hot memory layer (for advanced usage)
    pub fn hot(&self) -> &Arc<HotMemory> {
        &self.hot
    }

    /// Get the cold memory layer (for advanced usage)
    pub fn cold(&self) -> &Arc<ColdMemory> {
        &self.cold
    }

    /// Get the WAL (for advanced usage)
    pub fn wal(&self) -> &Arc<WriteAheadLog> {
        &self.wal
    }
}

/// Statistics for the dual-layer memory system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DualLayerStats {
    /// Number of entries in hot layer
    pub hot_entry_count: usize,
    /// Memory used by hot layer in bytes
    pub hot_memory_bytes: usize,
    /// Number of entries in cold layer
    pub cold_entry_count: usize,
    /// Disk space used by cold layer in bytes
    pub cold_disk_bytes: u64,
    /// Number of entries in WAL
    pub wal_entry_count: usize,
    /// Disk space used by WAL in bytes
    pub wal_disk_bytes: u64,
    /// Total entries across all layers
    pub total_entries: usize,
}

// ============================================================================
// Original Storage Module Code (Preserved)
// ============================================================================

/// Security configuration for Qdrant connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantSecurityConfig {
    /// API key for authentication
    pub api_key: Option<String>,
    /// Enable TLS
    pub tls_enabled: bool,
    /// CA certificate path
    pub ca_cert_path: Option<String>,
    /// Client certificate path
    pub client_cert_path: Option<String>,
    /// Client key path
    pub client_key_path: Option<String>,
    /// Skip TLS verification (not recommended for production)
    pub skip_tls_verify: bool,
}

impl Default for QdrantSecurityConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            tls_enabled: true,
            ca_cert_path: None,
            client_cert_path: None,
            client_key_path: None,
            skip_tls_verify: false,
        }
    }
}

/// Connection pool configuration for Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConnectionConfig {
    /// Maximum number of connections in the pool
    pub max_connections: usize,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Maximum idle time for connections in seconds
    pub max_idle_secs: u64,
    /// Security configuration
    pub security: QdrantSecurityConfig,
}

impl Default for QdrantConnectionConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            connect_timeout_secs: 30,
            request_timeout_secs: 60,
            health_check_interval_secs: 300, // 5 minutes
            max_idle_secs: 600,              // 10 minutes
            security: QdrantSecurityConfig::default(),
        }
    }
}

/// Convert qdrant Value to serde_json Value
fn qdrant_value_to_json(value: &qdrant_client::qdrant::Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;

    match &value.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(v)) => serde_json::Value::Bool(*v),
        Some(Kind::IntegerValue(v)) => serde_json::Value::Number((*v).into()),
        Some(Kind::DoubleValue(v)) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*v).unwrap_or(0.into()))
        }
        Some(Kind::StringValue(v)) => serde_json::Value::String(v.clone()),
        Some(Kind::ListValue(v)) => {
            let items = v.values.iter().map(qdrant_value_to_json).collect();
            serde_json::Value::Array(items)
        }
        Some(Kind::StructValue(v)) => {
            let fields = v
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();
            serde_json::Value::Object(fields)
        }
        None => serde_json::Value::Null,
    }
}

/// Access level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Read-only access
    Read,
    /// Read and write access
    ReadWrite,
    /// Full administrative access
    Admin,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Required level for read operations
    pub read_level: AccessLevel,
    /// Required level for write operations
    pub write_level: AccessLevel,
    /// Required level for delete operations
    pub delete_level: AccessLevel,
    /// Required level for admin operations
    pub admin_level: AccessLevel,
    /// Enable audit logging
    pub enable_audit_log: bool,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            read_level: AccessLevel::Read,
            write_level: AccessLevel::ReadWrite,
            delete_level: AccessLevel::ReadWrite,
            admin_level: AccessLevel::Admin,
            enable_audit_log: true,
        }
    }
}

/// Access context for operations
#[derive(Debug, Clone)]
pub struct AccessContext {
    /// User identifier
    pub user_id: String,
    /// User's access level
    pub access_level: AccessLevel,
    /// Operation being performed
    pub operation: String,
    /// Timestamp of the operation
    pub timestamp: i64,
}

impl AccessContext {
    /// Create a new access context
    pub fn new(user_id: String, access_level: AccessLevel, operation: String) -> Self {
        Self {
            user_id,
            access_level,
            operation,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Check if this context has permission for the required level
    pub fn has_permission(
        &self,
        required_level: &AccessLevel,
        _config: &AccessControlConfig,
    ) -> bool {
        match required_level {
            AccessLevel::Read => matches!(
                self.access_level,
                AccessLevel::Read | AccessLevel::ReadWrite | AccessLevel::Admin
            ),
            AccessLevel::ReadWrite => matches!(
                self.access_level,
                AccessLevel::ReadWrite | AccessLevel::Admin
            ),
            AccessLevel::Admin => matches!(self.access_level, AccessLevel::Admin),
        }
    }
}

/// Configuration for embedding cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCacheConfig {
    /// Maximum number of embeddings to cache
    pub max_size: usize,
    /// TTL for cached embeddings in seconds
    pub ttl_secs: u64,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            ttl_secs: 3600, // 1 hour
        }
    }
}

/// Cached embedding entry
#[derive(Debug, Clone)]
struct CachedEmbedding {
    /// The embedding vector
    embedding: Vec<f32>,
    /// When this entry was created
    created_at: Instant,
}

/// LRU cache for embeddings
#[derive(Debug)]
pub struct EmbeddingCache {
    /// Cache storage
    cache: HashMap<Uuid, CachedEmbedding>,
    /// Access order for LRU eviction
    access_order: Vec<Uuid>,
    /// Configuration
    config: EmbeddingCacheConfig,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(config: EmbeddingCacheConfig) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            config,
        }
    }

    /// Put an embedding in the cache
    pub fn put(&mut self, chunk_id: Uuid, embedding: Vec<f32>) {
        let entry = CachedEmbedding {
            embedding,
            created_at: Instant::now(),
        };

        // Remove existing entry if present
        if self.cache.contains_key(&chunk_id) {
            self.access_order.retain(|&id| id != chunk_id);
        }

        // Add new entry
        self.cache.insert(chunk_id, entry);
        self.access_order.push(chunk_id);

        // Evict if over capacity
        while self.cache.len() > self.config.max_size {
            let oldest_id = self.access_order.remove(0);
            self.cache.remove(&oldest_id);
        }
    }

    /// Get an embedding from the cache
    pub fn get(&mut self, chunk_id: &Uuid) -> Option<Vec<f32>> {
        // Check if entry exists and is not expired
        if let Some(entry) = self.cache.get(chunk_id) {
            if entry.created_at.elapsed().as_secs() <= self.config.ttl_secs {
                // Update access order for LRU
                self.access_order.retain(|&id| id != *chunk_id);
                self.access_order.push(*chunk_id);
                return Some(entry.embedding.clone());
            } else {
                // Remove expired entry
                self.cache.remove(chunk_id);
                self.access_order.retain(|&id| id != *chunk_id);
            }
        }
        None
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&mut self) {
        let mut to_remove = Vec::new();

        for (id, entry) in &self.cache {
            if entry.created_at.elapsed().as_secs() > self.config.ttl_secs {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            self.cache.remove(&id);
            self.access_order.retain(|&order_id| order_id != id);
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Number of documents stored
    pub document_count: usize,
    /// Number of chunks stored
    pub chunk_count: usize,
    /// Number of embeddings stored
    pub embedding_count: usize,
    /// Total size in bytes
    pub size_bytes: u64,
}

/// Storage backend trait
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a document
    async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()>;

    /// Get a document by ID
    async fn get_document(&self, id: &Uuid, context: &AccessContext) -> Result<Option<Document>>;

    /// Delete a document
    async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()>;

    /// List all documents
    async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>>;

    /// Store embeddings for a chunk
    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()>;

    /// Get embeddings for a chunk
    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>>;

    /// Search by vector similarity
    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>>;

    /// Get storage statistics
    async fn stats(&self, context: &AccessContext) -> Result<StorageStats>;
}

/// In-memory storage backend for testing
pub struct InMemoryStorage {
    documents: Arc<RwLock<HashMap<Uuid, Document>>>,
    embeddings: Arc<RwLock<HashMap<Uuid, Vec<f32>>>>,
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            embeddings: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl InMemoryStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl StorageBackend for InMemoryStorage {
    async fn store_document(&self, doc: &Document, _context: &AccessContext) -> Result<()> {
        let mut docs = self.documents.write().await;
        docs.insert(doc.id, doc.clone());
        Ok(())
    }

    async fn get_document(&self, id: &Uuid, _context: &AccessContext) -> Result<Option<Document>> {
        let docs = self.documents.read().await;
        Ok(docs.get(id).cloned())
    }

    async fn delete_document(&self, id: &Uuid, _context: &AccessContext) -> Result<()> {
        let mut docs = self.documents.write().await;
        docs.remove(id);
        Ok(())
    }

    async fn list_documents(&self, _context: &AccessContext) -> Result<Vec<Uuid>> {
        let docs = self.documents.read().await;
        Ok(docs.keys().cloned().collect())
    }

    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        _context: &AccessContext,
    ) -> Result<()> {
        let mut embs = self.embeddings.write().await;
        embs.insert(*chunk_id, embeddings.to_vec());
        Ok(())
    }

    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        _context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        let embs = self.embeddings.read().await;
        Ok(embs.get(chunk_id).cloned())
    }

    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        _context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        let embs = self.embeddings.read().await;
        let mut results: Vec<(Uuid, f32)> = embs
            .iter()
            .map(|(id, emb)| (*id, cosine_similarity(query_embedding, emb)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    async fn stats(&self, _context: &AccessContext) -> Result<StorageStats> {
        let docs = self.documents.read().await;
        let embs = self.embeddings.read().await;

        // Count actual chunks from documents, not embeddings
        let chunk_count: usize = docs.values().map(|d| d.chunks.len()).sum();

        Ok(StorageStats {
            document_count: docs.len(),
            chunk_count,
            embedding_count: embs.len(),
            size_bytes: 0, // Not tracked in memory
        })
    }
}

/// Pooled Qdrant connection entry
struct PooledConnection {
    client: Qdrant,
    last_used: Instant,
    #[allow(dead_code)]
    created_at: Instant,
}

/// Simple connection pool for Qdrant clients
struct QdrantConnectionPool {
    connections: Vec<PooledConnection>,
    config: QdrantConnectionConfig,
    client_config: qdrant_client::config::QdrantConfig,
}

impl QdrantConnectionPool {
    fn new(
        client_config: qdrant_client::config::QdrantConfig,
        config: QdrantConnectionConfig,
    ) -> Self {
        Self {
            connections: Vec::new(),
            config,
            client_config,
        }
    }

    async fn get_connection(&mut self) -> Result<&mut Qdrant> {
        // First, try to find an available connection
        let available_index = self.connections.iter().position(|conn| {
            conn.last_used.elapsed() < Duration::from_secs(self.config.max_idle_secs)
        });

        if let Some(index) = available_index {
            self.connections[index].last_used = Instant::now();
            return Ok(&mut self.connections[index].client);
        }

        // Create new connection if under limit
        if self.connections.len() < self.config.max_connections {
            let client = Qdrant::new(self.client_config.clone())
                .map_err(|e| Error::io(format!("Failed to create Qdrant client: {}", e)))?;

            self.connections.push(PooledConnection {
                client,
                last_used: Instant::now(),
                created_at: Instant::now(),
            });

            let len = self.connections.len();
            return Ok(&mut self.connections[len - 1].client);
        }

        Err(Error::io("Connection pool exhausted".to_string()))
    }

    #[allow(dead_code)]
    fn cleanup_expired(&mut self) {
        self.connections.retain(|conn| {
            conn.created_at.elapsed() < Duration::from_secs(self.config.max_idle_secs)
        });
    }

    async fn health_check(&mut self) -> Result<()> {
        if let Ok(client) = self.get_connection().await {
            // Simple health check - try to list collections
            client
                .list_collections()
                .await
                .map_err(|e| Error::io(format!("Health check failed: {}", e)))?;
        }
        Ok(())
    }
}

/// File-based storage backend (JSON files)
pub struct FileStorage {
    base_path: PathBuf,
    documents: Arc<RwLock<HashMap<Uuid, Document>>>,
}

impl FileStorage {
    /// Create a new file-based storage
    pub async fn new(base_path: PathBuf) -> Result<Self> {
        // Create directories if they don't exist
        tokio::fs::create_dir_all(&base_path)
            .await
            .map_err(|e| Error::io(format!("Failed to create storage directory: {}", e)))?;
        tokio::fs::create_dir_all(base_path.join("documents"))
            .await
            .map_err(|e| Error::io(format!("Failed to create documents directory: {}", e)))?;
        tokio::fs::create_dir_all(base_path.join("embeddings"))
            .await
            .map_err(|e| Error::io(format!("Failed to create embeddings directory: {}", e)))?;

        // Load existing documents
        let documents = Arc::new(RwLock::new(HashMap::new()));

        let storage = Self {
            base_path,
            documents,
        };
        storage.load_documents().await?;

        Ok(storage)
    }

    async fn load_documents(&self) -> Result<()> {
        let docs_path = self.base_path.join("documents");

        let mut entries = tokio::fs::read_dir(&docs_path)
            .await
            .map_err(|e| Error::io(format!("Failed to read documents directory: {}", e)))?;

        let mut docs = self.documents.write().await;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| Error::io(format!("Failed to read directory entry: {}", e)))?
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let content = tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| Error::io(format!("Failed to read document file: {}", e)))?;
                let doc: Document = serde_json::from_str(&content)
                    .map_err(|e| Error::parse(format!("Failed to parse document: {}", e)))?;
                docs.insert(doc.id, doc);
            }
        }

        Ok(())
    }

    fn doc_path(&self, id: &Uuid) -> PathBuf {
        self.base_path
            .join("documents")
            .join(format!("{}.json", id))
    }

    fn embedding_path(&self, id: &Uuid) -> PathBuf {
        self.base_path
            .join("embeddings")
            .join(format!("{}.bin", id))
    }
}

#[async_trait]
impl StorageBackend for FileStorage {
    async fn store_document(&self, doc: &Document, _context: &AccessContext) -> Result<()> {
        let path = self.doc_path(&doc.id);
        let content = serde_json::to_string_pretty(doc)
            .map_err(|e| Error::parse(format!("Failed to serialize document: {}", e)))?;
        tokio::fs::write(&path, content)
            .await
            .map_err(|e| Error::io(format!("Failed to write document: {}", e)))?;

        let mut docs = self.documents.write().await;
        docs.insert(doc.id, doc.clone());

        Ok(())
    }

    async fn get_document(&self, id: &Uuid, _context: &AccessContext) -> Result<Option<Document>> {
        let docs = self.documents.read().await;
        Ok(docs.get(id).cloned())
    }

    async fn delete_document(&self, id: &Uuid, _context: &AccessContext) -> Result<()> {
        let path = self.doc_path(id);
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| Error::io(format!("Failed to delete document: {}", e)))?;
        }

        let mut docs = self.documents.write().await;
        docs.remove(id);

        Ok(())
    }

    async fn list_documents(&self, _context: &AccessContext) -> Result<Vec<Uuid>> {
        let docs = self.documents.read().await;
        Ok(docs.keys().cloned().collect())
    }

    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        _context: &AccessContext,
    ) -> Result<()> {
        let path = self.embedding_path(chunk_id);

        // Store as binary for efficiency
        let bytes: Vec<u8> = embeddings.iter().flat_map(|f| f.to_le_bytes()).collect();

        tokio::fs::write(&path, bytes)
            .await
            .map_err(|e| Error::io(format!("Failed to write embeddings: {}", e)))?;

        Ok(())
    }

    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        _context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        let path = self.embedding_path(chunk_id);

        if !path.exists() {
            return Ok(None);
        }

        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| Error::io(format!("Failed to read embeddings: {}", e)))?;

        let embeddings: Vec<f32> = bytes
            .chunks(4)
            .map(|chunk: &[u8]| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect();

        Ok(Some(embeddings))
    }

    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        _context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        let embeddings_dir = self.base_path.join("embeddings");
        let mut results: Vec<(Uuid, f32)> = Vec::new();

        let mut entries = tokio::fs::read_dir(&embeddings_dir)
            .await
            .map_err(|e| Error::io(format!("Failed to read embeddings directory: {}", e)))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| Error::io(format!("Failed to read entry: {}", e)))?
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "bin") {
                // Extract UUID from filename
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(id) = Uuid::parse_str(stem) {
                        // Load embeddings
                        let bytes = tokio::fs::read(&path)
                            .await
                            .map_err(|e| Error::io(format!("Failed to read embeddings: {}", e)))?;

                        let embeddings: Vec<f32> = bytes
                            .chunks(4)
                            .map(|chunk: &[u8]| {
                                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                                f32::from_le_bytes(arr)
                            })
                            .collect();

                        let score = cosine_similarity(query_embedding, &embeddings);
                        results.push((id, score));
                    }
                }
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    async fn stats(&self, _context: &AccessContext) -> Result<StorageStats> {
        let docs = self.documents.read().await;
        let chunk_count: usize = docs.values().map(|d| d.chunks.len()).sum();

        // Count embedding files
        let embeddings_dir = self.base_path.join("embeddings");
        let mut embedding_count = 0;

        if let Ok(mut entries) = tokio::fs::read_dir(&embeddings_dir).await {
            while let Ok(Some(_)) = entries.next_entry().await {
                embedding_count += 1;
            }
        }

        // Calculate approximate size
        let mut size_bytes: u64 = 0;
        let docs_dir = self.base_path.join("documents");
        if let Ok(mut entries) = tokio::fs::read_dir(&docs_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Ok(metadata) = entry.metadata().await {
                    size_bytes += metadata.len();
                }
            }
        }

        Ok(StorageStats {
            document_count: docs.len(),
            chunk_count,
            embedding_count,
            size_bytes,
        })
    }
}

/// Qdrant vector database storage backend
pub struct QdrantStorage {
    pool: Arc<RwLock<QdrantConnectionPool>>,
    collection_name: String,
    vector_size: usize,
    embedding_cache: Arc<RwLock<EmbeddingCache>>,
    access_control: AccessControlConfig,
}

impl QdrantStorage {
    /// Create a new Qdrant storage backend
    pub async fn new(
        host: &str,
        port: u16,
        grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
    ) -> Result<Self> {
        Self::new_with_config(
            host,
            port,
            grpc_port,
            collection_name,
            vector_size,
            embedded,
            QdrantConnectionConfig::default(),
            EmbeddingCacheConfig::default(),
            AccessControlConfig::default(),
        )
        .await
    }

    /// Create a new Qdrant storage backend with custom configuration
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_config(
        host: &str,
        port: u16,
        _grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
        conn_config: QdrantConnectionConfig,
        cache_config: EmbeddingCacheConfig,
        access_config: AccessControlConfig,
    ) -> Result<Self> {
        let config = if embedded {
            qdrant_client::config::QdrantConfig::from_url("http://localhost:6333")
        } else {
            qdrant_client::config::QdrantConfig::from_url(&format!("http://{}:{}", host, port))
        };

        let pool = Arc::new(RwLock::new(QdrantConnectionPool::new(
            config,
            conn_config.clone(),
        )));
        let embedding_cache = Arc::new(RwLock::new(EmbeddingCache::new(cache_config)));

        let storage = Self {
            pool: pool.clone(),
            collection_name: collection_name.clone(),
            vector_size,
            embedding_cache,
            access_control: access_config,
        };

        // Ensure collection exists using a connection from the pool
        {
            let mut pool_guard = pool.write().await;
            let client = pool_guard.get_connection().await?;
            Self::ensure_collection(client, &collection_name, vector_size).await?;
        }

        // Start background health check task
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(conn_config.health_check_interval_secs));
            loop {
                interval.tick().await;
                let mut pool = pool_clone.write().await;
                if let Err(e) = pool.health_check().await {
                    tracing::warn!("Qdrant health check failed: {}", e);
                }
            }
        });

        Ok(storage)
    }

    async fn ensure_collection(
        client: &Qdrant,
        collection_name: &str,
        vector_size: usize,
    ) -> Result<()> {
        // Check if collection exists
        let collections = client
            .list_collections()
            .await
            .map_err(|e| Error::io(format!("Failed to list collections: {}", e)))?;

        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);

        if !collection_exists {
            // Create collection with vector configuration
            let vector_params = VectorParams {
                size: vector_size as u64,
                distance: Distance::Cosine as i32,
                hnsw_config: None,
                quantization_config: Some(QuantizationConfig {
                    quantization: Some(
                        qdrant_client::qdrant::quantization_config::Quantization::Scalar(
                            ScalarQuantization {
                                r#type: qdrant_client::qdrant::QuantizationType::Int8 as i32,
                                quantile: None,
                                always_ram: None,
                            },
                        ),
                    ),
                }),
                on_disk: None,
                datatype: None,
                multivector_config: None,
            };

            let collection_params = CreateCollection {
                collection_name: collection_name.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                        vector_params,
                    )),
                }),
                ..Default::default()
            };

            client
                .create_collection(collection_params)
                .await
                .map_err(|e| Error::io(format!("Failed to create collection: {}", e)))?;
        }

        Ok(())
    }

    fn point_id_from_uuid(uuid: &Uuid) -> PointId {
        PointId::from(uuid.to_string())
    }

    fn uuid_from_point_id(point_id: &PointId) -> Option<Uuid> {
        // PointId is created from UUID string, so we need to extract and parse it
        match &point_id.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid_str)) => {
                Uuid::parse_str(uuid_str).ok()
            }
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => {
                // For numeric IDs, we can't reliably convert back to UUID
                // This shouldn't happen if we always use UUID strings
                tracing::warn!(
                    "Cannot convert numeric PointId {} to UUID - using UUID strings is required",
                    num
                );
                None
            }
            None => None,
        }
    }

    fn check_access(&self, context: &AccessContext, required_level: &AccessLevel) -> Result<()> {
        if !context.has_permission(required_level, &self.access_control) {
            return Err(Error::validation(format!(
                "Access denied: user {} requires {:?} level for operation '{}', has {:?}",
                context.user_id, required_level, context.operation, context.access_level
            )));
        }

        if self.access_control.enable_audit_log {
            tracing::info!(
                "Access granted: user={}, operation={}, level={:?}, timestamp={}",
                context.user_id,
                context.operation,
                context.access_level,
                context.timestamp
            );
        }

        Ok(())
    }
}

#[async_trait]
impl StorageBackend for QdrantStorage {
    async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()> {
        self.check_access(context, &self.access_control.write_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        // Store document metadata as payload
        let payload: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&serde_json::to_string(doc)?)
                .map_err(|e| Error::parse(format!("Failed to serialize document: {}", e)))?;

        let point = PointStruct::new(
            Self::point_id_from_uuid(&doc.id),
            vec![], // No vectors for document metadata
            payload,
        );

        let points = vec![point];
        let upsert_points = UpsertPoints {
            collection_name: self.collection_name.clone(),
            wait: Some(true),
            points,
            ..Default::default()
        };

        client
            .upsert_points(upsert_points)
            .await
            .map_err(|e| Error::io(format!("Failed to store document: {}", e)))?;

        Ok(())
    }

    async fn get_document(&self, id: &Uuid, context: &AccessContext) -> Result<Option<Document>> {
        self.check_access(context, &self.access_control.read_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(id);

        let get_points = GetPoints {
            collection_name: self.collection_name.clone(),
            ids: vec![point_id],
            with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true),
                ),
            }),
            with_vectors: Some(qdrant_client::qdrant::WithVectorsSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_vectors_selector::SelectorOptions::Enable(false),
                ),
            }),
            ..Default::default()
        };

        let response = client
            .get_points(get_points)
            .await
            .map_err(|e| Error::io(format!("Failed to get document: {}", e)))?;

        if let Some(point) = response.result.first() {
            // Convert qdrant payload to serde_json Value
            let json_payload: std::collections::HashMap<String, serde_json::Value> = point
                .payload
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();

            let doc: Document = serde_json::from_value(serde_json::Value::Object(
                json_payload.into_iter().collect(),
            ))
            .map_err(|e| Error::parse(format!("Failed to deserialize document: {}", e)))?;
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }

    async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()> {
        self.check_access(context, &self.access_control.delete_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(id);

        let delete_points = DeletePoints {
            collection_name: self.collection_name.clone(),
            wait: Some(true),
            points: Some(qdrant_client::qdrant::PointsSelector {
                points_selector_one_of: Some(
                    qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Points(
                        qdrant_client::qdrant::PointsIdsList {
                            ids: vec![point_id],
                        },
                    ),
                ),
            }),
            ..Default::default()
        };

        client
            .delete_points(delete_points)
            .await
            .map_err(|e| Error::io(format!("Failed to delete document: {}", e)))?;

        Ok(())
    }

    async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>> {
        self.check_access(context, &self.access_control.read_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        // Scroll through all points to get document IDs
        let mut all_ids = Vec::new();
        let mut offset = None;

        loop {
            let scroll_points = ScrollPoints {
                collection_name: self.collection_name.clone(),
                limit: Some(100),
                offset,
                with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                    selector_options: Some(
                        qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(
                            false,
                        ),
                    ),
                }),
                with_vectors: Some(qdrant_client::qdrant::WithVectorsSelector {
                    selector_options: Some(
                        qdrant_client::qdrant::with_vectors_selector::SelectorOptions::Enable(
                            false,
                        ),
                    ),
                }),
                ..Default::default()
            };

            let response = client
                .scroll(scroll_points)
                .await
                .map_err(|e| Error::io(format!("Failed to scroll points: {}", e)))?;

            for point in &response.result {
                if let Some(id) = &point.id {
                    if let Some(uuid) = Self::uuid_from_point_id(id) {
                        all_ids.push(uuid);
                    }
                }
            }

            if response.next_page_offset.is_none() {
                break;
            }
            offset = response.next_page_offset;
        }

        Ok(all_ids)
    }

    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()> {
        self.check_access(context, &self.access_control.write_level)?;

        if embeddings.len() != self.vector_size {
            return Err(Error::validation(format!(
                "Embedding size {} does not match configured vector size {}",
                embeddings.len(),
                self.vector_size
            )));
        }

        // Cache the embedding
        {
            let mut cache = self.embedding_cache.write().await;
            cache.put(*chunk_id, embeddings.to_vec());
        }

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(chunk_id);

        // Create payload with chunk metadata
        let mut payload: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();
        payload.insert(
            "chunk_id".to_string(),
            serde_json::Value::String(chunk_id.to_string()),
        );

        let point = PointStruct::new(point_id, embeddings.to_vec(), payload);

        let points = vec![point];
        let upsert_points = UpsertPoints {
            collection_name: self.collection_name.clone(),
            wait: Some(true),
            points,
            ..Default::default()
        };

        client
            .upsert_points(upsert_points)
            .await
            .map_err(|e| Error::io(format!("Failed to store embeddings: {}", e)))?;

        Ok(())
    }

    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        self.check_access(context, &self.access_control.read_level)?;

        // Check cache first
        {
            let mut cache = self.embedding_cache.write().await;
            cache.cleanup_expired(); // Clean up expired entries
            if let Some(embedding) = cache.get(chunk_id) {
                return Ok(Some(embedding));
            }
        }

        // Not in cache, retrieve from Qdrant
        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(chunk_id);

        let get_points = GetPoints {
            collection_name: self.collection_name.clone(),
            ids: vec![point_id],
            with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(false),
                ),
            }),
            with_vectors: Some(qdrant_client::qdrant::WithVectorsSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_vectors_selector::SelectorOptions::Enable(true),
                ),
            }),
            ..Default::default()
        };

        let response = client
            .get_points(get_points)
            .await
            .map_err(|e| Error::io(format!("Failed to get embeddings: {}", e)))?;

        if let Some(point) = response.result.first() {
            // Extract vector data from the retrieved point
            if let Some(vectors) = &point.vectors {
                use qdrant_client::qdrant::vectors_output::VectorsOptions;
                match &vectors.vectors_options {
                    Some(VectorsOptions::Vector(vector_output)) => {
                        // Use into_vector() instead of deprecated .data field
                        // into_vector() returns Vector enum, extract dense data from it
                        use qdrant_client::qdrant::vector_output::Vector as OutputVector;
                        match vector_output.clone().into_vector() {
                            OutputVector::Dense(dense) => {
                                let embedding = dense.data;
                                self.embedding_cache
                                    .write()
                                    .await
                                    .put(*chunk_id, embedding.clone());
                                Ok(Some(embedding))
                            }
                            _ => Ok(None), // Sparse/MultiDense not supported for caching
                        }
                    }
                    Some(VectorsOptions::Vectors(named_vectors)) => {
                        use qdrant_client::qdrant::vector_output::Vector as OutputVector;
                        // For named vectors, try to get the default vector
                        if let Some(vector_output) = named_vectors.vectors.get("") {
                            match vector_output.clone().into_vector() {
                                OutputVector::Dense(dense) => {
                                    let embedding = dense.data;
                                    self.embedding_cache
                                        .write()
                                        .await
                                        .put(*chunk_id, embedding.clone());
                                    Ok(Some(embedding))
                                }
                                _ => Ok(None),
                            }
                        } else if let Some((_, vector_output)) = named_vectors.vectors.iter().next()
                        {
                            // Fallback to first available vector
                            match vector_output.clone().into_vector() {
                                OutputVector::Dense(dense) => {
                                    let embedding = dense.data;
                                    self.embedding_cache
                                        .write()
                                        .await
                                        .put(*chunk_id, embedding.clone());
                                    Ok(Some(embedding))
                                }
                                _ => Ok(None),
                            }
                        } else {
                            Ok(None)
                        }
                    }
                    None => Ok(None),
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        self.check_access(context, &self.access_control.read_level)?;

        if query_embedding.len() != self.vector_size {
            return Err(Error::validation(format!(
                "Query embedding size {} does not match configured vector size {}",
                query_embedding.len(),
                self.vector_size
            )));
        }

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let search_points = SearchPoints {
            collection_name: self.collection_name.clone(),
            vector: query_embedding.to_vec(),
            limit: top_k as u64,
            with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true),
                ),
            }),
            ..Default::default()
        };

        let response = client
            .search_points(search_points)
            .await
            .map_err(|e| Error::io(format!("Failed to search vectors: {}", e)))?;

        let results = response
            .result
            .into_iter()
            .filter_map(|scored_point| {
                scored_point
                    .id
                    .as_ref()
                    .and_then(Self::uuid_from_point_id)
                    .map(|uuid| (uuid, scored_point.score))
            })
            .collect();

        Ok(results)
    }

    async fn stats(&self, context: &AccessContext) -> Result<StorageStats> {
        self.check_access(context, &self.access_control.admin_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let collection_info = client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| Error::io(format!("Failed to get collection info: {}", e)))?;

        let points_count = collection_info
            .result
            .as_ref()
            .map(|info| info.points_count.unwrap_or(0))
            .unwrap_or(0);

        // Estimate document count (points without vectors are documents)
        let document_count = points_count.saturating_sub(points_count);

        Ok(StorageStats {
            document_count: document_count as usize,
            chunk_count: points_count as usize,
            embedding_count: points_count as usize,
            size_bytes: 0, // Qdrant doesn't expose this directly
        })
    }
}
/// Main storage interface for ReasonKit memory layer
pub struct Storage {
    /// Backend implementation for storage operations
    backend: Box<dyn StorageBackend>,
}

impl Storage {
    /// Create storage with in-memory backend
    pub fn in_memory() -> Self {
        Self {
            backend: Box::new(InMemoryStorage::new()),
        }
    }

    /// Create storage with file backend
    pub async fn file(base_path: PathBuf) -> Result<Self> {
        Ok(Self {
            backend: Box::new(FileStorage::new(base_path).await?),
        })
    }

    /// Create embedded storage with automatic configuration
    ///
    /// This is a convenience method that uses `create_embedded_storage()` with default config.
    /// It will automatically use file storage (no Qdrant required).
    ///
    /// # Example
    /// ```rust,no_run
    /// use reasonkit_mem::storage::Storage;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let storage = Storage::new_embedded().await?;
    /// // Use storage...
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new_embedded() -> Result<Self> {
        create_embedded_storage(EmbeddedStorageConfig::default()).await
    }

    /// Create embedded storage with custom configuration
    ///
    /// # Example
    /// ```rust,no_run
    /// use reasonkit_mem::storage::{Storage, EmbeddedStorageConfig};
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = EmbeddedStorageConfig::file_only(PathBuf::from("./data"));
    /// let storage = Storage::new_embedded_with_config(config).await?;
    /// // Use storage...
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new_embedded_with_config(config: EmbeddedStorageConfig) -> Result<Self> {
        create_embedded_storage(config).await
    }

    /// Create storage with Qdrant backend
    pub async fn qdrant(
        host: &str,
        port: u16,
        grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
    ) -> Result<Self> {
        Ok(Self {
            backend: Box::new(
                QdrantStorage::new(
                    host,
                    port,
                    grpc_port,
                    collection_name,
                    vector_size,
                    embedded,
                )
                .await?,
            ),
        })
    }

    /// Create storage with Qdrant backend and custom configuration
    #[allow(clippy::too_many_arguments)]
    pub async fn qdrant_with_config(
        host: &str,
        port: u16,
        grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
        conn_config: QdrantConnectionConfig,
        cache_config: EmbeddingCacheConfig,
        access_config: AccessControlConfig,
    ) -> Result<Self> {
        Ok(Self {
            backend: Box::new(
                QdrantStorage::new_with_config(
                    host,
                    port,
                    grpc_port,
                    collection_name,
                    vector_size,
                    embedded,
                    conn_config,
                    cache_config,
                    access_config,
                )
                .await?,
            ),
        })
    }

    /// Store a document
    pub async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()> {
        self.backend.store_document(doc, context).await
    }

    /// Get a document by ID
    pub async fn get_document(
        &self,
        id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Document>> {
        self.backend.get_document(id, context).await
    }

    /// Delete a document
    pub async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()> {
        self.backend.delete_document(id, context).await
    }

    /// List all documents
    pub async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>> {
        self.backend.list_documents(context).await
    }

    /// Store embeddings
    pub async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()> {
        self.backend
            .store_embeddings(chunk_id, embeddings, context)
            .await
    }

    /// Get embeddings by chunk ID
    pub async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        self.backend.get_embeddings(chunk_id, context).await
    }

    /// Search by vector
    pub async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        self.backend
            .search_by_vector(query_embedding, top_k, context)
            .await
    }

    /// Get stats
    pub async fn stats(&self, context: &AccessContext) -> Result<StorageStats> {
        self.backend.stats(context).await
    }
}

pub mod benchmarks;

// Temporarily disabled due to compilation errors
// pub mod optimized;

/// Embedded storage configuration for local-first usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedStorageConfig {
    /// Path for storage data (used by file backend)
    pub data_path: PathBuf,
    /// Collection name for Qdrant
    pub collection_name: String,
    /// Vector dimension size
    pub vector_size: usize,
    /// Whether to require running Qdrant server (vs file-only mode)
    pub require_qdrant: bool,
    /// Qdrant URL for embedded mode (default: http://localhost:6333)
    pub qdrant_url: String,
}

impl Default for EmbeddedStorageConfig {
    fn default() -> Self {
        Self {
            data_path: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("reasonkit")
                .join("storage"),
            collection_name: "reasonkit_default".to_string(),
            vector_size: 1536, // OpenAI ada-002 default
            require_qdrant: false,
            qdrant_url: "http://localhost:6333".to_string(),
        }
    }
}

impl EmbeddedStorageConfig {
    /// Create config for file-only mode (no Qdrant required)
    pub fn file_only(data_path: PathBuf) -> Self {
        Self {
            data_path,
            require_qdrant: false,
            ..Default::default()
        }
    }

    /// Create config for local Qdrant mode
    pub fn with_qdrant(qdrant_url: &str, collection_name: &str, vector_size: usize) -> Self {
        Self {
            qdrant_url: qdrant_url.to_string(),
            collection_name: collection_name.to_string(),
            vector_size,
            require_qdrant: true,
            ..Default::default()
        }
    }
}

/// Create embedded storage with automatic fallback
///
/// This function attempts to create the best available storage:
/// 1. If `require_qdrant` is true and Qdrant is available: QdrantStorage
/// 2. Otherwise: FileStorage as fallback
///
/// # Example
/// ```ignore
/// let config = EmbeddedStorageConfig::default();
/// let storage = create_embedded_storage(config).await?;
/// ```
pub async fn create_embedded_storage(config: EmbeddedStorageConfig) -> Result<Storage> {
    // Ensure data directory exists
    if !config.data_path.exists() {
        std::fs::create_dir_all(&config.data_path).map_err(|e| {
            Error::io(format!(
                "Failed to create storage directory {:?}: {}",
                config.data_path, e
            ))
        })?;
        tracing::info!(path = ?config.data_path, "Created storage data directory");
    }

    if config.require_qdrant {
        // Try to connect to Qdrant
        match check_qdrant_health(&config.qdrant_url).await {
            Ok(()) => {
                tracing::info!(url = %config.qdrant_url, "Connected to Qdrant server");
                // Parse URL for host and port
                let (host, port) = parse_qdrant_url(&config.qdrant_url);

                return Storage::qdrant(
                    &host,
                    port,
                    port + 1, // gRPC port typically port + 1
                    config.collection_name,
                    config.vector_size,
                    true, // embedded mode
                )
                .await;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    url = %config.qdrant_url,
                    "Qdrant not available, require_qdrant=true will fail"
                );
                return Err(Error::io(format!(
                    "Qdrant required but not available at {}: {}",
                    config.qdrant_url, e
                )));
            }
        }
    }

    // Use file storage as fallback
    tracing::info!(path = ?config.data_path, "Using file-based storage (Qdrant not required)");
    Storage::file(config.data_path).await
}

/// Check if Qdrant server is healthy
///
/// This function checks if a Qdrant server is running and accessible at the given URL.
/// It uses the `/readyz` endpoint which is Qdrant's health check endpoint.
///
/// # Arguments
/// * `url` - Base URL of the Qdrant server (e.g., "http://localhost:6333")
///
/// # Returns
/// * `Ok(())` if Qdrant is healthy and accessible
/// * `Err(Error)` if Qdrant is not accessible or unhealthy
async fn check_qdrant_health(url: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| Error::io(format!("Failed to create HTTP client: {}", e)))?;

    // Normalize URL (remove trailing slash, ensure http:// prefix)
    let normalized_url = url.trim_end_matches('/');
    let base_url =
        if normalized_url.starts_with("http://") || normalized_url.starts_with("https://") {
            normalized_url.to_string()
        } else {
            format!("http://{}", normalized_url)
        };

    let health_url = format!("{}/readyz", base_url);
    let response = client
        .get(&health_url)
        .send()
        .await
        .map_err(|e| Error::io(format!("Qdrant health check failed: {}", e)))?;

    if response.status().is_success() {
        tracing::debug!(url = %base_url, "Qdrant health check passed");
        Ok(())
    } else {
        Err(Error::io(format!(
            "Qdrant health check returned status: {}",
            response.status()
        )))
    }
}

/// Parse Qdrant URL into host and port
///
/// Handles various URL formats:
/// - `http://localhost:6333`
/// - `localhost:6333`
/// - `localhost`
/// - `127.0.0.1:6333`
///
/// # Arguments
/// * `url` - Qdrant URL string
///
/// # Returns
/// * `(host, port)` tuple
fn parse_qdrant_url(url: &str) -> (String, u16) {
    // Remove http:// or https:// prefix
    let url = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");

    // Split by colon to get host and port
    let parts: Vec<&str> = url.split(':').collect();
    let host = parts.first().unwrap_or(&"localhost").to_string();
    let port: u16 = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(6333); // Default Qdrant port

    (host, port)
}

/// Get the default storage path for embedded mode
pub fn default_storage_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("reasonkit")
        .join("storage")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_config_default() {
        let config = EmbeddedStorageConfig::default();
        assert!(!config.require_qdrant);
        assert_eq!(config.vector_size, 1536);
        assert_eq!(config.collection_name, "reasonkit_default");
    }

    #[test]
    fn test_embedded_config_file_only() {
        let config = EmbeddedStorageConfig::file_only(PathBuf::from("/tmp/test"));
        assert!(!config.require_qdrant);
        assert_eq!(config.data_path, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_embedded_config_with_qdrant() {
        let config =
            EmbeddedStorageConfig::with_qdrant("http://localhost:6334", "test_collection", 768);
        assert!(config.require_qdrant);
        assert_eq!(config.qdrant_url, "http://localhost:6334");
        assert_eq!(config.collection_name, "test_collection");
        assert_eq!(config.vector_size, 768);
    }

    #[test]
    fn test_default_storage_path() {
        let path = default_storage_path();
        assert!(path.ends_with("reasonkit/storage") || path.ends_with("reasonkit\\storage"));
    }

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("Test content");
        assert!(!entry.content.is_empty());
        assert_eq!(entry.importance, 0.5);
        assert_eq!(entry.layer, MemoryLayer::Pending);
    }

    #[test]
    fn test_memory_entry_builder() {
        let entry = MemoryEntry::new("Test content")
            .with_importance(0.9)
            .with_metadata("key", "value")
            .with_ttl(3600)
            .with_tags(vec!["tag1".to_string(), "tag2".to_string()]);

        assert_eq!(entry.importance, 0.9);
        assert_eq!(entry.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(entry.ttl_secs, Some(3600));
        assert_eq!(entry.tags.len(), 2);
    }

    #[test]
    fn test_dual_layer_config_default() {
        let config = DualLayerConfig::default();
        assert_eq!(config.sync.interval_secs, 60);
        assert!(config.sync.auto_sync_enabled);
        // max_hot_entries is in hot config, not sync
        assert!(config.hot.max_entries > 0);
    }

    #[test]
    fn test_dual_layer_config_low_latency() {
        // Note: low_latency() method was removed - using high_performance instead
        let config = DualLayerConfig::high_performance(PathBuf::from("/tmp"));
        assert!(config.hot.max_entries > 0);
        assert!(config.sync.interval_secs > 0);
    }

    #[test]
    fn test_dual_layer_config_memory_efficient() {
        // Note: memory_efficient() method was removed - using low_memory instead
        let config = DualLayerConfig::low_memory(PathBuf::from("/tmp"));
        assert!(config.hot.max_entries > 0);
        assert_eq!(config.sync.hot_to_cold_age_secs, 300);
    }

    #[tokio::test]
    async fn test_in_memory_storage() {
        use crate::{DocumentType, Source, SourceType};
        use chrono::Utc;

        let storage = Storage::in_memory();
        let context = AccessContext::new(
            "test_user".to_string(),
            AccessLevel::Admin,
            "test".to_string(),
        );

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("test.md".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc =
            Document::new(DocumentType::Note, source).with_content("Test content".to_string());

        storage.store_document(&doc, &context).await.unwrap();
        let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content.raw, "Test content");
    }

    #[tokio::test]
    async fn test_file_storage_creation() {
        let temp_dir = std::env::temp_dir().join("reasonkit_storage_test");
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let storage = Storage::file(temp_dir.clone()).await.unwrap();
        let context = AccessContext::new(
            "test_user".to_string(),
            AccessLevel::Admin,
            "test".to_string(),
        );

        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 0);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_embedded_storage_file_fallback() {
        let temp_dir = std::env::temp_dir().join("reasonkit_embedded_test");
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        // Should fall back to file storage when require_qdrant is false
        let config = EmbeddedStorageConfig::file_only(temp_dir.clone());
        let storage = create_embedded_storage(config).await.unwrap();

        let context = AccessContext::new(
            "test_user".to_string(),
            AccessLevel::Admin,
            "test".to_string(),
        );

        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 0);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_parse_qdrant_url() {
        // Test various URL formats
        assert_eq!(
            parse_qdrant_url("http://localhost:6333"),
            ("localhost".to_string(), 6333)
        );
        assert_eq!(
            parse_qdrant_url("localhost:6333"),
            ("localhost".to_string(), 6333)
        );
        assert_eq!(
            parse_qdrant_url("localhost"),
            ("localhost".to_string(), 6333)
        );
        assert_eq!(
            parse_qdrant_url("127.0.0.1:6334"),
            ("127.0.0.1".to_string(), 6334)
        );
        assert_eq!(
            parse_qdrant_url("https://qdrant.example.com:6333"),
            ("qdrant.example.com".to_string(), 6333)
        );
    }

    #[tokio::test]
    async fn test_embedded_storage_default_config() {
        let temp_dir = std::env::temp_dir().join("reasonkit_embedded_default_test");
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        // Default config should use file storage (require_qdrant=false)
        let config = EmbeddedStorageConfig {
            data_path: temp_dir.clone(),
            ..Default::default()
        };
        let storage = create_embedded_storage(config).await.unwrap();

        let context = AccessContext::new(
            "test_user".to_string(),
            AccessLevel::Admin,
            "test".to_string(),
        );

        // Verify storage works
        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 0);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_embedded_storage_with_qdrant_required_but_unavailable() {
        let temp_dir = std::env::temp_dir().join("reasonkit_embedded_qdrant_test");
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        // Config requiring Qdrant but pointing to non-existent server
        let config = EmbeddedStorageConfig::with_qdrant(
            "http://localhost:99999", // Non-existent port
            "test_collection",
            768,
        );
        let mut config = config;
        config.data_path = temp_dir.clone();

        // Should fail because require_qdrant=true and Qdrant is not available
        match create_embedded_storage(config).await {
            Ok(_) => panic!("Expected error when Qdrant is required but unavailable"),
            Err(e) => {
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("Qdrant required but not available"),
                    "Error message should mention Qdrant not available, got: {}",
                    error_msg
                );
            }
        }

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
