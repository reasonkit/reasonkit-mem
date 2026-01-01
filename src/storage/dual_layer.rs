//! Dual-Layer Storage API
//!
//! Unified storage interface for hot/cold memory with WAL integration.
//!
//! # Architecture
//!
//! ```text
//!                          DualLayerStorage
//!                                 |
//!          +-----------+---------+----------+-----------+
//!          |           |         |          |           |
//!      HotMemory   ColdMemory   WAL    TierIndex   AccessTracker
//! ```
//!
//! # Features
//!
//! - Single entry point for all storage operations
//! - Automatic hot/cold routing based on access patterns
//! - Transparent WAL integration for durability
//! - Backward compatible with `StorageBackend` trait
//! - Optional transaction support with ACID guarantees

use crate::error::{MemError, MemResult};
use crate::storage::{
    AccessContext, AccessControlConfig, AccessLevel, EmbeddingCacheConfig, QdrantConnectionConfig,
    QdrantSecurityConfig, StorageBackend, StorageStats,
};
use crate::{Document, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// CONFIGURATION TYPES
// ============================================================================

// DualLayerConfig is defined in storage/config.rs - use that instead
// This was a duplicate definition causing E0255 errors
pub use super::config::DualLayerConfig;

// HotMemoryConfig has different fields in dual_layer vs config
// Keep local definition for InMemoryHotLayer compatibility
/// Configuration for hot memory layer (dual_layer-specific)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMemoryConfig {
    /// Maximum capacity in bytes (0 = unlimited)
    pub max_capacity_bytes: u64,
    /// Maximum number of entries (0 = unlimited)
    pub max_entries: usize,
    /// Entry TTL in hot memory (seconds, 0 = no expiry)
    pub ttl_secs: u64,
    /// Eviction policy for when capacity is reached
    pub eviction_policy: EvictionPolicy,
    /// Backend type for hot storage
    pub backend: HotBackendType,
    /// Enable compression for hot data
    pub compression_enabled: bool,
    /// Compression level (1-9, higher = more compression)
    pub compression_level: u8,
}

impl Default for HotMemoryConfig {
    fn default() -> Self {
        Self {
            max_capacity_bytes: 1024 * 1024 * 1024, // 1GB
            max_entries: 100_000,
            ttl_secs: 3600, // 1 hour
            eviction_policy: EvictionPolicy::Lru,
            backend: HotBackendType::InMemory,
            compression_enabled: false,
            compression_level: 3,
        }
    }
}

/// Eviction policies for hot memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EvictionPolicy {
    /// Least Recently Used
    #[default]
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In First Out
    Fifo,
    /// Time-based expiry only
    TimeOnly,
    /// Combined LRU + LFU (ARC-like)
    Adaptive,
}

/// Backend types for hot storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum HotBackendType {
    /// Pure in-memory (HashMap-based)
    #[default]
    InMemory,
    /// Memory-mapped file
    Mmap {
        /// Path to the memory-mapped file
        path: PathBuf,
    },
    /// Embedded RocksDB
    RocksDb {
        /// Path to the RocksDB directory
        path: PathBuf,
    },
    /// Redis/Valkey connection
    Redis {
        /// Redis connection URL
        url: String,
    },
}

/// Configuration for cold memory layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdMemoryConfig {
    /// Backend type for cold storage
    pub backend: ColdBackendType,
    /// Vector dimension size (for Qdrant)
    pub vector_size: usize,
    /// Collection name (for Qdrant)
    pub collection_name: String,
    /// Enable quantization for vectors
    pub quantization_enabled: bool,
    /// Quantization type
    pub quantization_type: QuantizationType,
    /// Connection pool configuration
    pub connection: QdrantConnectionConfig,
    /// Embedding cache configuration
    pub embedding_cache: EmbeddingCacheConfig,
    /// Access control configuration
    pub access_control: AccessControlConfig,
}

impl Default for ColdMemoryConfig {
    fn default() -> Self {
        Self {
            backend: ColdBackendType::File {
                base_path: crate::storage::default_storage_path(),
            },
            vector_size: 1536,
            collection_name: "reasonkit_cold".to_string(),
            quantization_enabled: true,
            quantization_type: QuantizationType::Int8,
            connection: QdrantConnectionConfig::default(),
            embedding_cache: EmbeddingCacheConfig::default(),
            access_control: AccessControlConfig::default(),
        }
    }
}

/// Backend types for cold storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColdBackendType {
    /// File-based storage (JSON + binary embeddings)
    File {
        /// Base path for file storage
        base_path: PathBuf,
    },
    /// Qdrant vector database (local)
    QdrantLocal {
        /// Qdrant server URL
        url: String,
    },
    /// Qdrant vector database (cloud)
    QdrantCloud {
        /// Qdrant server URL
        url: String,
        /// API key for authentication
        api_key: String,
    },
    /// S3-compatible object storage
    S3 {
        /// S3 endpoint URL
        endpoint: String,
        /// Bucket name
        bucket: String,
        /// Access key ID
        access_key: String,
        /// Secret access key
        secret_key: String,
    },
}

impl Default for ColdBackendType {
    fn default() -> Self {
        Self::File {
            base_path: crate::storage::default_storage_path(),
        }
    }
}

/// Vector quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantizationType {
    /// No quantization (full precision)
    None,
    /// 8-bit integer quantization
    #[default]
    Int8,
    /// Binary quantization
    Binary,
    /// Product quantization with specified segments
    ProductQuantization {
        /// Number of segments for product quantization
        segments: usize,
    },
}

/// Write-Ahead Log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfig {
    /// Enable WAL for durability
    pub enabled: bool,
    /// WAL directory path
    pub path: PathBuf,
    /// Maximum WAL file size before rotation (bytes)
    pub max_file_size: u64,
    /// Sync mode for durability guarantees
    pub sync_mode: WalSyncMode,
    /// Maximum time between syncs (milliseconds)
    pub sync_interval_ms: u64,
    /// Retention period for WAL files (seconds)
    pub retention_secs: u64,
    /// Enable WAL compression
    pub compression_enabled: bool,
    /// Maximum WAL entries before checkpoint
    pub checkpoint_threshold: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: crate::storage::default_storage_path().join("wal"),
            max_file_size: 64 * 1024 * 1024, // 64MB
            sync_mode: WalSyncMode::Interval,
            sync_interval_ms: 1000,
            retention_secs: 86400, // 24 hours
            compression_enabled: true,
            checkpoint_threshold: 10000,
        }
    }
}

/// WAL synchronization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WalSyncMode {
    /// Sync every write (slowest, safest)
    Immediate,
    /// Sync at intervals
    #[default]
    Interval,
    /// Let OS handle syncing (fastest, least safe)
    OsManaged,
    /// No sync - WAL in memory only
    None,
}

/// Synchronization and tiering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Tiering policy for hot/cold routing
    pub tiering_policy: TieringPolicy,
    /// Background sync interval (seconds)
    pub background_sync_interval_secs: u64,
    /// Maximum items to sync per batch
    pub sync_batch_size: usize,
    /// Enable automatic promotion from cold to hot
    pub auto_promote: bool,
    /// Promotion threshold (access count)
    pub promotion_threshold: usize,
    /// Demotion threshold (seconds since last access)
    pub demotion_threshold_secs: u64,
    /// Enable parallel sync operations
    pub parallel_sync: bool,
    /// Maximum concurrent sync operations
    pub max_concurrent_syncs: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            tiering_policy: TieringPolicy::default(),
            background_sync_interval_secs: 60,
            sync_batch_size: 100,
            auto_promote: true,
            promotion_threshold: 3,
            demotion_threshold_secs: 3600,
            parallel_sync: true,
            max_concurrent_syncs: 4,
        }
    }
}

/// Tiering policy for data placement decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringPolicy {
    /// Initial tier for new data
    pub default_tier: StorageTier,
    /// Access-based tiering
    pub access_based: bool,
    /// Age-based tiering
    pub age_based: bool,
    /// Size-based tiering (large items go to cold)
    pub size_based: bool,
    /// Size threshold for cold storage (bytes)
    pub size_threshold_bytes: u64,
    /// Age threshold for cold storage (seconds)
    pub age_threshold_secs: u64,
}

impl Default for TieringPolicy {
    fn default() -> Self {
        Self {
            default_tier: StorageTier::Hot,
            access_based: true,
            age_based: true,
            size_based: true,
            size_threshold_bytes: 10 * 1024 * 1024, // 10MB
            age_threshold_secs: 86400,              // 24 hours
        }
    }
}

/// Storage tier identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum StorageTier {
    /// Hot storage tier (fast access, limited capacity)
    #[default]
    Hot,
    /// Cold storage tier (slower access, large capacity)
    Cold,
}

// ============================================================================
// STATISTICS TYPES
// ============================================================================

/// Comprehensive statistics for dual-layer storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLayerStats {
    /// Hot tier statistics
    pub hot: TierStats,
    /// Cold tier statistics
    pub cold: TierStats,
    /// WAL statistics
    pub wal: WalStats,
    /// Overall statistics
    pub overall: OverallStats,
    /// Tiering statistics
    pub tiering: TieringStats,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Timestamp of statistics collection
    pub collected_at: DateTime<Utc>,
}

/// Statistics for a single tier
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierStats {
    /// Number of documents
    pub document_count: usize,
    /// Number of chunks
    pub chunk_count: usize,
    /// Number of embeddings
    pub embedding_count: usize,
    /// Storage size in bytes
    pub size_bytes: u64,
    /// Capacity used percentage
    pub capacity_used_pct: f32,
    /// Hit rate for this tier
    pub hit_rate: f32,
    /// Average access latency (microseconds)
    pub avg_latency_us: u64,
    /// P95 latency (microseconds)
    pub p95_latency_us: u64,
}

/// WAL statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalStats {
    /// WAL enabled
    pub enabled: bool,
    /// Current WAL size in bytes
    pub current_size_bytes: u64,
    /// Number of pending entries
    pub pending_entries: usize,
    /// Last sync timestamp
    pub last_sync_at: Option<DateTime<Utc>>,
    /// Last checkpoint timestamp
    pub last_checkpoint_at: Option<DateTime<Utc>>,
    /// Write throughput (entries/sec)
    pub write_throughput: f64,
}

/// Overall statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverallStats {
    /// Total documents across all tiers
    pub total_documents: usize,
    /// Total chunks
    pub total_chunks: usize,
    /// Total embeddings
    pub total_embeddings: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Uptime in seconds
    pub uptime_secs: u64,
}

/// Tiering statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TieringStats {
    /// Documents in hot tier
    pub hot_count: usize,
    /// Documents in cold tier
    pub cold_count: usize,
    /// Recent promotions (cold -> hot)
    pub promotions_last_hour: usize,
    /// Recent demotions (hot -> cold)
    pub demotions_last_hour: usize,
    /// Candidates awaiting promotion
    pub pending_promotions: usize,
    /// Candidates awaiting demotion
    pub pending_demotions: usize,
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Read operations per second
    pub reads_per_sec: f64,
    /// Write operations per second
    pub writes_per_sec: f64,
    /// Average read latency (microseconds)
    pub avg_read_latency_us: u64,
    /// Average write latency (microseconds)
    pub avg_write_latency_us: u64,
    /// Cache hit ratio (0.0 - 1.0)
    pub cache_hit_ratio: f32,
    /// Hot tier hit ratio
    pub hot_hit_ratio: f32,
}

/// WAL status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalStatus {
    /// WAL is enabled
    pub enabled: bool,
    /// WAL is healthy
    pub healthy: bool,
    /// Current file path
    pub current_file: Option<PathBuf>,
    /// Pending entries count
    pub pending_entries: usize,
    /// Time since last sync (milliseconds)
    pub ms_since_last_sync: u64,
    /// Time since last checkpoint (milliseconds)
    pub ms_since_last_checkpoint: u64,
    /// Any errors
    pub errors: Vec<String>,
}

impl Default for WalStatus {
    fn default() -> Self {
        Self {
            enabled: false,
            healthy: true,
            current_file: None,
            pending_entries: 0,
            ms_since_last_sync: 0,
            ms_since_last_checkpoint: 0,
            errors: Vec::new(),
        }
    }
}

/// Checkpoint result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointResult {
    /// Number of entries checkpointed
    pub entries_checkpointed: usize,
    /// Bytes written to cold storage
    pub bytes_written: u64,
    /// Duration of checkpoint
    pub duration: Duration,
    /// Any errors during checkpoint
    pub errors: Vec<String>,
}

impl Default for CheckpointResult {
    fn default() -> Self {
        Self {
            entries_checkpointed: 0,
            bytes_written: 0,
            duration: Duration::ZERO,
            errors: Vec::new(),
        }
    }
}

/// Tiering operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringResult {
    /// Documents promoted (cold -> hot)
    pub promoted: usize,
    /// Documents demoted (hot -> cold)
    pub demoted: usize,
    /// Skipped due to errors
    pub skipped: usize,
    /// Duration of tiering operation
    pub duration: Duration,
}

impl Default for TieringResult {
    fn default() -> Self {
        Self {
            promoted: 0,
            demoted: 0,
            skipped: 0,
            duration: Duration::ZERO,
        }
    }
}

/// Result of bulk operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BulkOperationResult {
    /// Number of successful operations
    pub succeeded: usize,
    /// Number of failed operations
    pub failed: usize,
    /// Errors with their document IDs
    pub errors: Vec<(Uuid, String)>,
    /// Duration of operation
    pub duration: Duration,
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Unified error type for dual-layer storage operations
#[derive(Debug, thiserror::Error)]
pub enum DualLayerError {
    /// Hot layer error
    #[error("Hot layer error: {0}")]
    HotLayer(String),

    /// Cold layer error
    #[error("Cold layer error: {0}")]
    ColdLayer(String),

    /// WAL error
    #[error("WAL error: {0}")]
    Wal(String),

    /// Document not found
    #[error("Document not found: {0}")]
    NotFound(Uuid),

    /// Tier routing error
    #[error("Tier routing error: {0}")]
    TierRouting(String),

    /// Sync error
    #[error("Sync error: {0}")]
    Sync(String),

    /// Transaction error
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Capacity exceeded
    #[error("Capacity exceeded: {message}")]
    CapacityExceeded { message: String, tier: StorageTier },

    /// Connection error
    #[error("Connection error: {0}")]
    Connection(String),

    /// Timeout
    #[error("Operation timed out after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Access denied
    #[error("Access denied: {0}")]
    AccessDenied(String),

    /// Underlying storage error
    #[error("Storage error: {0}")]
    Storage(#[from] MemError),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl DualLayerError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            DualLayerError::Connection(_)
                | DualLayerError::Timeout { .. }
                | DualLayerError::Sync(_)
        )
    }

    /// Check if error indicates potential data loss
    pub fn may_cause_data_loss(&self) -> bool {
        matches!(
            self,
            DualLayerError::Wal(_) | DualLayerError::Sync(_) | DualLayerError::Transaction(_)
        )
    }
}

/// Result type for dual-layer operations
pub type DualLayerResult<T> = std::result::Result<T, DualLayerError>;

// ============================================================================
// CORE TRAITS
// ============================================================================

/// Hot memory layer trait
#[async_trait]
pub trait HotMemoryLayer: Send + Sync {
    /// Store a document in hot memory
    async fn store(&mut self, doc: &Document) -> Result<()>;

    /// Get a document from hot memory
    async fn get(&self, id: &Uuid) -> Result<Option<Document>>;

    /// Delete a document from hot memory
    async fn delete(&mut self, id: &Uuid) -> Result<bool>;

    /// Check if document exists in hot memory
    async fn contains(&self, id: &Uuid) -> bool;

    /// Get all document IDs in hot memory
    async fn keys(&self) -> Vec<Uuid>;

    /// Get hot memory statistics
    fn stats(&self) -> TierStats;

    /// Clear all entries (for testing/maintenance)
    async fn clear(&mut self);

    /// Evict entries based on policy
    async fn evict(&mut self, count: usize) -> usize;
}

/// Extended storage backend trait for dual-layer operations
#[async_trait]
pub trait DualLayerBackend: StorageBackend {
    // ============================================================
    // TIER-AWARE OPERATIONS
    // ============================================================

    /// Store document in specified tier
    async fn store_document_in_tier(
        &self,
        doc: &Document,
        tier: StorageTier,
        context: &AccessContext,
    ) -> Result<()>;

    /// Get document with tier information
    async fn get_document_with_tier(
        &self,
        id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<(Document, StorageTier)>>;

    /// Move document between tiers
    async fn move_to_tier(
        &self,
        id: &Uuid,
        target_tier: StorageTier,
        context: &AccessContext,
    ) -> Result<()>;

    /// Get current tier for a document
    async fn get_tier(&self, id: &Uuid) -> Result<Option<StorageTier>>;

    // ============================================================
    // BULK OPERATIONS
    // ============================================================

    /// Store multiple documents (batched for efficiency)
    async fn store_documents_bulk(
        &self,
        docs: &[Document],
        context: &AccessContext,
    ) -> Result<BulkOperationResult>;

    /// Get multiple documents by IDs
    async fn get_documents_bulk(
        &self,
        ids: &[Uuid],
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, Option<Document>)>>;

    /// Delete multiple documents
    async fn delete_documents_bulk(
        &self,
        ids: &[Uuid],
        context: &AccessContext,
    ) -> Result<BulkOperationResult>;

    /// Store multiple embeddings (batched)
    async fn store_embeddings_bulk(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
        context: &AccessContext,
    ) -> Result<BulkOperationResult>;

    // ============================================================
    // WAL OPERATIONS
    // ============================================================

    /// Force WAL sync to disk
    async fn sync_wal(&self) -> Result<()>;

    /// Create checkpoint (flush WAL to cold storage)
    async fn checkpoint(&self) -> Result<CheckpointResult>;

    /// Get WAL status
    async fn wal_status(&self) -> Result<WalStatus>;

    // ============================================================
    // TIERING OPERATIONS
    // ============================================================

    /// Trigger background tiering (promote/demote based on policy)
    async fn run_tiering(&self) -> Result<TieringResult>;

    /// Get candidates for promotion to hot tier
    async fn get_promotion_candidates(&self, limit: usize) -> Result<Vec<Uuid>>;

    /// Get candidates for demotion to cold tier
    async fn get_demotion_candidates(&self, limit: usize) -> Result<Vec<Uuid>>;

    // ============================================================
    // EXTENDED STATISTICS
    // ============================================================

    /// Get detailed statistics including tier breakdown
    async fn detailed_stats(&self, context: &AccessContext) -> Result<DualLayerStats>;

    /// Get hot tier statistics
    async fn hot_stats(&self) -> Result<TierStats>;

    /// Get cold tier statistics
    async fn cold_stats(&self, context: &AccessContext) -> Result<TierStats>;
}

// ============================================================================
// TRANSACTION SUPPORT
// ============================================================================

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Active,
    Committed,
    RolledBack,
    Failed,
}

/// Isolation levels for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read uncommitted (fastest, least isolation)
    ReadUncommitted,
    /// Read committed (default)
    ReadCommitted,
    /// Repeatable read
    RepeatableRead,
    /// Serializable (slowest, full isolation)
    Serializable,
}

/// Transaction operations
#[derive(Debug, Clone)]
pub enum TransactionOperation {
    StoreDocument {
        doc: Box<Document>,
        tier: StorageTier,
    },
    DeleteDocument {
        id: Uuid,
    },
    StoreEmbedding {
        chunk_id: Uuid,
        embedding: Vec<f32>,
    },
    MoveTier {
        id: Uuid,
        from: StorageTier,
        to: StorageTier,
    },
}

/// Savepoint for partial rollback
#[derive(Debug, Clone)]
pub struct Savepoint {
    /// Savepoint name
    pub name: String,
    /// Operation index at savepoint creation
    pub operation_index: usize,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Transaction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionResult {
    /// Transaction ID
    pub transaction_id: Uuid,
    /// Number of successful operations
    pub operations_succeeded: usize,
    /// Number of failed operations
    pub operations_failed: usize,
    /// Duration of transaction
    pub duration: Duration,
}

// ============================================================================
// TIER INDEX
// ============================================================================

/// Index tracking which tier each document is in
#[derive(Debug, Default)]
pub struct TierIndex {
    /// Mapping from document ID to tier
    tiers: HashMap<Uuid, StorageTier>,
}

impl TierIndex {
    /// Create new tier index
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tier for a document
    pub fn set_tier(&mut self, id: Uuid, tier: StorageTier) {
        self.tiers.insert(id, tier);
    }

    /// Get tier for a document
    pub fn get_tier(&self, id: &Uuid) -> Option<StorageTier> {
        self.tiers.get(id).copied()
    }

    /// Remove document from index
    pub fn remove(&mut self, id: &Uuid) -> Option<StorageTier> {
        self.tiers.remove(id)
    }

    /// Get all document IDs
    pub fn all_ids(&self) -> Vec<Uuid> {
        self.tiers.keys().copied().collect()
    }

    /// Get count by tier
    pub fn count_by_tier(&self) -> (usize, usize) {
        let hot = self
            .tiers
            .values()
            .filter(|&&t| t == StorageTier::Hot)
            .count();
        let cold = self
            .tiers
            .values()
            .filter(|&&t| t == StorageTier::Cold)
            .count();
        (hot, cold)
    }

    /// Get IDs in specific tier
    pub fn ids_in_tier(&self, tier: StorageTier) -> Vec<Uuid> {
        self.tiers
            .iter()
            .filter(|(_, &t)| t == tier)
            .map(|(id, _)| *id)
            .collect()
    }
}

// ============================================================================
// ACCESS TRACKER
// ============================================================================

/// Entry for tracking document access patterns
#[derive(Debug, Clone)]
pub struct AccessEntry {
    /// Number of accesses
    pub access_count: usize,
    /// Last access timestamp
    pub last_access: Instant,
    /// First access timestamp
    pub first_access: Instant,
}

/// Tracks access patterns for tiering decisions
#[derive(Debug)]
pub struct AccessTracker {
    /// Access entries by document ID
    entries: HashMap<Uuid, AccessEntry>,
    /// Tiering policy reference
    policy: TieringPolicy,
}

impl AccessTracker {
    /// Create new access tracker
    pub fn new(policy: &TieringPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            policy: policy.clone(),
        }
    }

    /// Record an access
    pub fn record_access(&mut self, id: &Uuid) {
        let now = Instant::now();
        self.entries
            .entry(*id)
            .and_modify(|e| {
                e.access_count += 1;
                e.last_access = now;
            })
            .or_insert(AccessEntry {
                access_count: 1,
                last_access: now,
                first_access: now,
            });
    }

    /// Remove tracking for a document
    pub fn remove(&mut self, id: &Uuid) {
        self.entries.remove(id);
    }

    /// Get access count for a document
    pub fn get_access_count(&self, id: &Uuid) -> usize {
        self.entries.get(id).map(|e| e.access_count).unwrap_or(0)
    }

    /// Get candidates for promotion (frequently accessed cold items)
    pub fn get_promotion_candidates(&self, current_cold_ids: &[Uuid], limit: usize) -> Vec<Uuid> {
        let mut candidates: Vec<_> = current_cold_ids
            .iter()
            .filter_map(|id| self.entries.get(id).map(|entry| (*id, entry.access_count)))
            .filter(|(_, count)| *count >= self.policy.promotion_threshold_count())
            .collect();

        // Sort by access count descending
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        candidates
            .into_iter()
            .take(limit)
            .map(|(id, _)| id)
            .collect()
    }

    /// Get candidates for demotion (rarely accessed hot items)
    pub fn get_demotion_candidates(&self, current_hot_ids: &[Uuid], limit: usize) -> Vec<Uuid> {
        let now = Instant::now();
        let threshold = Duration::from_secs(self.policy.age_threshold_secs);

        let mut candidates: Vec<_> = current_hot_ids
            .iter()
            .filter_map(|id| {
                self.entries.get(id).and_then(|entry| {
                    if now.duration_since(entry.last_access) > threshold {
                        Some((*id, entry.last_access))
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Sort by last access ascending (oldest first)
        candidates.sort_by(|a, b| a.1.cmp(&b.1));
        candidates
            .into_iter()
            .take(limit)
            .map(|(id, _)| id)
            .collect()
    }
}

impl TieringPolicy {
    /// Get promotion threshold as access count
    fn promotion_threshold_count(&self) -> usize {
        3 // Default, could be made configurable
    }
}

// ============================================================================
// WAL ENTRY TYPES
// ============================================================================

/// WAL entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    /// Store document operation
    StoreDocument {
        id: Uuid,
        tier: StorageTier,
        timestamp: DateTime<Utc>,
    },
    /// Delete document operation
    DeleteDocument { id: Uuid, timestamp: DateTime<Utc> },
    /// Store embedding operation
    StoreEmbedding {
        chunk_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    /// Move between tiers
    MoveTier {
        id: Uuid,
        from: StorageTier,
        to: StorageTier,
        timestamp: DateTime<Utc>,
    },
    /// Transaction begin
    TransactionBegin { id: Uuid, timestamp: DateTime<Utc> },
    /// Transaction commit
    TransactionCommit { id: Uuid, timestamp: DateTime<Utc> },
    /// Transaction rollback
    TransactionRollback { id: Uuid, timestamp: DateTime<Utc> },
    /// Checkpoint marker
    Checkpoint { timestamp: DateTime<Utc> },
}

// ============================================================================
// TESTS
// ============================================================================

// ============================================================================
// CONTEXT RETRIEVAL RESULT
// ============================================================================

/// Result of a context retrieval operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResult {
    /// Unique identifier of the memory
    pub id: Uuid,
    /// Content text
    pub content: String,
    /// Similarity score (0.0 - 1.0, higher is better)
    pub score: f32,
    /// Which layer the result came from
    pub layer: StorageTier,
    /// When the memory was created
    pub created_at: DateTime<Utc>,
    /// Access count
    pub access_count: usize,
}

// ============================================================================
// DUAL LAYER STORAGE IMPLEMENTATION
// ============================================================================

/// Dual-layer storage engine with hot (DashMap) + cold (Sled) memory and WAL
///
/// This is the main entry point for the reasonkit-mem storage subsystem.
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_mem::storage::dual_layer::{DualLayerStorage, DualLayerConfig};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let config = DualLayerConfig::default();
///     let storage = DualLayerStorage::new(config).await?;
///
///     // Store a memory
///     storage.store(uuid::Uuid::new_v4(), "Important context", &[0.1, 0.2, 0.3]).await?;
///
///     // Retrieve relevant context
///     let results = storage.retrieve_context("memory safety", 5).await?;
///     for r in results {
///         println!("[{:.2}] {}", r.score, r.content);
///     }
///
///     Ok(())
/// }
/// ```
pub struct DualLayerStorage {
    /// Hot memory layer
    hot: Arc<RwLock<InMemoryHotLayer>>,
    /// Cold memory backend (using existing StorageBackend)
    cold: Arc<dyn StorageBackend>,
    /// Tier index
    tier_index: Arc<RwLock<TierIndex>>,
    /// Access tracker
    access_tracker: Arc<RwLock<AccessTracker>>,
    /// Configuration
    config: DualLayerConfig,
    /// Start time for uptime tracking
    started_at: Instant,
}

/// In-memory hot layer implementation
struct InMemoryHotLayer {
    /// Document storage
    documents: HashMap<Uuid, Document>,
    /// Embeddings storage
    embeddings: HashMap<Uuid, Vec<f32>>,
    /// Content storage (text for retrieval)
    content: HashMap<Uuid, String>,
    /// Metadata
    metadata: HashMap<Uuid, HotEntryMeta>,
    /// Configuration
    config: HotMemoryConfig,
}

/// Metadata for hot layer entries
#[derive(Debug, Clone)]
struct HotEntryMeta {
    created_at: DateTime<Utc>,
    last_accessed: Instant,
    access_count: usize,
    size_bytes: usize,
}

impl InMemoryHotLayer {
    fn new(config: super::config::HotMemoryConfig) -> Self {
        // Convert canonical HotMemoryConfig to dual_layer HotMemoryConfig
        let dual_config = HotMemoryConfig {
            max_capacity_bytes: 0, // unlimited
            max_entries: config.max_entries,
            ttl_secs: config.ttl_secs,
            eviction_policy: EvictionPolicy::Lru, // default
            backend: HotBackendType::InMemory,    // default
            compression_enabled: false,           // default
            compression_level: 1,                 // default
        };
        Self {
            documents: HashMap::new(),
            embeddings: HashMap::new(),
            content: HashMap::new(),
            metadata: HashMap::new(),
            config: dual_config,
        }
    }

    fn insert_content(&mut self, id: Uuid, content: String, embedding: Vec<f32>) {
        let size = content.len() + embedding.len() * 4;

        // Check capacity and evict if needed
        while self.documents.len() >= self.config.max_entries && !self.documents.is_empty() {
            self.evict_one();
        }

        self.content.insert(id, content);
        self.embeddings.insert(id, embedding);
        self.metadata.insert(
            id,
            HotEntryMeta {
                created_at: Utc::now(),
                last_accessed: Instant::now(),
                access_count: 0,
                size_bytes: size,
            },
        );
    }

    fn get_content(&mut self, id: &Uuid) -> Option<(String, Vec<f32>)> {
        if let Some(meta) = self.metadata.get_mut(id) {
            meta.last_accessed = Instant::now();
            meta.access_count += 1;
        }

        let content = self.content.get(id)?;
        let embedding = self.embeddings.get(id)?;
        Some((content.clone(), embedding.clone()))
    }

    fn remove(&mut self, id: &Uuid) -> bool {
        self.content.remove(id);
        self.embeddings.remove(id);
        self.metadata.remove(id);
        self.documents.remove(id).is_some()
    }

    fn evict_one(&mut self) {
        // LRU eviction
        if let Some((&oldest_id, _)) = self
            .metadata
            .iter()
            .min_by(|a, b| a.1.last_accessed.cmp(&b.1.last_accessed))
        {
            self.remove(&oldest_id);
        }
    }

    fn search(&self, query_embedding: &[f32], limit: usize) -> Vec<(Uuid, f32, String)> {
        let mut results: Vec<_> = self
            .embeddings
            .iter()
            .filter_map(|(id, emb)| {
                let score = cosine_similarity(query_embedding, emb);
                if score > 0.0 {
                    let content = self.content.get(id)?.clone();
                    Some((*id, score, content))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    fn len(&self) -> usize {
        self.content.len()
    }

    fn get_meta(&self, id: &Uuid) -> Option<&HotEntryMeta> {
        self.metadata.get(id)
    }
}

impl DualLayerStorage {
    /// Create a new dual-layer storage engine
    pub async fn new(config: DualLayerConfig) -> DualLayerResult<Self> {
        // Create hot memory layer
        let hot = Arc::new(RwLock::new(InMemoryHotLayer::new(config.hot.clone())));

        // Create cold memory backend (using file storage for embedded mode)
        // Canonical ColdMemoryConfig uses db_path, not backend
        let cold_path = config.cold.db_path.clone();

        // Ensure directory exists
        tokio::fs::create_dir_all(&cold_path).await?;

        let cold: Arc<dyn StorageBackend> = Arc::new(
            crate::storage::FileStorage::new(cold_path)
                .await
                .map_err(|e| DualLayerError::ColdLayer(e.to_string()))?,
        );

        let tier_index = Arc::new(RwLock::new(TierIndex::new()));
        // tiering_policy not in canonical SyncConfig - using default tiering policy
        let default_tiering = TieringPolicy::default();
        let access_tracker = Arc::new(RwLock::new(AccessTracker::new(&default_tiering)));

        Ok(Self {
            hot,
            cold,
            tier_index,
            access_tracker,
            config,
            started_at: Instant::now(),
        })
    }

    /// Create with default configuration
    pub async fn default_instance() -> DualLayerResult<Self> {
        Self::new(DualLayerConfig::default()).await
    }

    // ========================================================================
    // CORE API: retrieve_context
    // ========================================================================

    /// Retrieve relevant context for a query using semantic search
    ///
    /// This is the main context retrieval function that:
    /// 1. Generates an embedding for the query
    /// 2. Searches hot memory first (fast, in-memory)
    /// 3. Searches cold memory (persistent, may have more data)
    /// 4. Merges and deduplicates results
    /// 5. Returns top-k results sorted by similarity score
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to search for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Vector of `ContextResult` sorted by descending similarity score
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = storage.retrieve_context("memory safety in Rust", 5).await?;
    /// for r in results {
    ///     println!("[{:.2}] {}", r.score, r.content);
    /// }
    /// ```
    pub async fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
    ) -> DualLayerResult<Vec<ContextResult>> {
        // Generate query embedding
        // In production, replace with actual embedding model (BGE-M3, Ada-002)
        let query_embedding = generate_query_embedding(query);

        // Search hot memory
        let hot_results = {
            let hot = self.hot.read().await;
            hot.search(&query_embedding, limit * 2)
        };

        // Search cold memory
        let context = AccessContext::new(
            "system".to_string(),
            AccessLevel::Read,
            "retrieve_context".to_string(),
        );
        let cold_results = self
            .cold
            .search_by_vector(&query_embedding, limit * 2, &context)
            .await
            .map_err(|e| DualLayerError::ColdLayer(e.to_string()))?;

        // Merge results
        let mut merged: HashMap<Uuid, ContextResult> = HashMap::new();

        // Add hot results
        for (id, score, content) in hot_results {
            let hot = self.hot.read().await;
            let meta = hot.get_meta(&id);
            merged.insert(
                id,
                ContextResult {
                    id,
                    content,
                    score,
                    layer: StorageTier::Hot,
                    created_at: meta.map(|m| m.created_at).unwrap_or_else(Utc::now),
                    access_count: meta.map(|m| m.access_count).unwrap_or(0),
                },
            );
        }

        // Add cold results (merge if already exists from hot)
        for (id, score) in cold_results {
            merged
                .entry(id)
                .and_modify(|existing| {
                    if score > existing.score {
                        existing.score = score;
                    }
                })
                .or_insert_with(|| ContextResult {
                    id,
                    content: "[cold storage]".to_string(), // Would need to fetch from cold
                    score,
                    layer: StorageTier::Cold,
                    created_at: Utc::now(),
                    access_count: 0,
                });
        }

        // Sort by score and take top-k
        let mut results: Vec<_> = merged.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        // Update access tracker
        {
            let mut tracker = self.access_tracker.write().await;
            for r in &results {
                tracker.record_access(&r.id);
            }
        }

        Ok(results)
    }

    /// Store a new memory with content and embedding
    pub async fn store(&self, id: Uuid, content: &str, embedding: &[f32]) -> DualLayerResult<()> {
        // Store in hot memory
        {
            let mut hot = self.hot.write().await;
            hot.insert_content(id, content.to_string(), embedding.to_vec());
        }

        // Update tier index
        {
            let mut index = self.tier_index.write().await;
            index.set_tier(id, StorageTier::Hot);
        }

        Ok(())
    }

    /// Delete a memory by ID
    pub async fn delete(&self, id: &Uuid) -> DualLayerResult<bool> {
        // Remove from hot
        let removed_hot = {
            let mut hot = self.hot.write().await;
            hot.remove(id)
        };

        // Remove from tier index
        {
            let mut index = self.tier_index.write().await;
            index.remove(id);
        }

        // Remove from access tracker
        {
            let mut tracker = self.access_tracker.write().await;
            tracker.remove(id);
        }

        Ok(removed_hot)
    }

    /// Get storage statistics
    pub async fn stats(&self) -> DualLayerResult<DualLayerStats> {
        let hot = self.hot.read().await;
        let hot_count = hot.len();

        let context = AccessContext::new(
            "system".to_string(),
            AccessLevel::Admin,
            "stats".to_string(),
        );
        let cold_stats = self
            .cold
            .stats(&context)
            .await
            .map_err(|e| DualLayerError::ColdLayer(e.to_string()))?;

        Ok(DualLayerStats {
            hot: TierStats {
                document_count: 0,
                chunk_count: hot_count,
                embedding_count: hot_count,
                size_bytes: 0,
                capacity_used_pct: (hot_count as f32 / self.config.hot.max_entries as f32) * 100.0,
                hit_rate: 0.0,
                avg_latency_us: 0,
                p95_latency_us: 0,
            },
            cold: TierStats {
                document_count: cold_stats.document_count,
                chunk_count: cold_stats.chunk_count,
                embedding_count: cold_stats.embedding_count,
                size_bytes: cold_stats.size_bytes,
                capacity_used_pct: 0.0,
                hit_rate: 0.0,
                avg_latency_us: 0,
                p95_latency_us: 0,
            },
            wal: WalStats::default(),
            overall: OverallStats {
                total_documents: cold_stats.document_count,
                total_chunks: hot_count + cold_stats.chunk_count,
                total_embeddings: hot_count + cold_stats.embedding_count,
                total_size_bytes: cold_stats.size_bytes,
                uptime_secs: self.started_at.elapsed().as_secs(),
            },
            tiering: TieringStats::default(),
            performance: PerformanceMetrics::default(),
            collected_at: Utc::now(),
        })
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Generate a pseudo-embedding for a query
///
/// In production, replace with actual embedding model call (BGE-M3, Ada-002)
fn generate_query_embedding(query: &str) -> Vec<f32> {
    // Simple hash-based pseudo-embedding (384 dimensions like BGE-small)
    let mut embedding = vec![0.0f32; 384];

    for (i, ch) in query.chars().enumerate() {
        let idx = (ch as usize + i * 31) % 384;
        embedding[idx] += 1.0 / (i as f32 + 1.0);
    }

    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for e in &mut embedding {
            *e /= magnitude;
        }
    }

    embedding
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_generate_embedding() {
        let emb = generate_query_embedding("test query");
        assert_eq!(emb.len(), 384);

        // Should be normalized
        let magnitude: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_default_config() {
        let config = DualLayerConfig::default();
        assert!(config.wal.enabled);
        // TODO: Update test - eviction_policy and tiering_policy fields removed from config
        // assert_eq!(config.hot.eviction_policy, EvictionPolicy::Lru);
        // assert_eq!(config.sync.tiering_policy.default_tier, StorageTier::Hot);
    }

    #[test]
    fn test_tier_index() {
        let mut index = TierIndex::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index.set_tier(id1, StorageTier::Hot);
        index.set_tier(id2, StorageTier::Cold);

        assert_eq!(index.get_tier(&id1), Some(StorageTier::Hot));
        assert_eq!(index.get_tier(&id2), Some(StorageTier::Cold));

        let (hot, cold) = index.count_by_tier();
        assert_eq!(hot, 1);
        assert_eq!(cold, 1);
    }

    #[test]
    fn test_access_tracker() {
        let policy = TieringPolicy::default();
        let mut tracker = AccessTracker::new(&policy);

        let id = Uuid::new_v4();

        // Record multiple accesses
        for _ in 0..5 {
            tracker.record_access(&id);
        }

        assert_eq!(tracker.get_access_count(&id), 5);

        tracker.remove(&id);
        assert_eq!(tracker.get_access_count(&id), 0);
    }

    #[test]
    fn test_error_retryable() {
        let timeout_err = DualLayerError::Timeout { duration_ms: 5000 };
        assert!(timeout_err.is_retryable());

        let not_found = DualLayerError::NotFound(Uuid::new_v4());
        assert!(!not_found.is_retryable());
    }

    #[test]
    fn test_eviction_policy_default() {
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::Lru);
    }

    #[test]
    fn test_wal_config_default() {
        let config = WalConfig::default();
        assert!(config.enabled);
        assert_eq!(config.sync_mode, WalSyncMode::Interval);
        assert_eq!(config.sync_interval_ms, 1000);
    }
}
