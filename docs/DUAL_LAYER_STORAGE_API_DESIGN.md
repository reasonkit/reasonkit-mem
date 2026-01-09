# Dual-Layer Storage API Design

> **ReasonKit Memory Infrastructure**
> Unified Storage Interface for Hot/Cold Memory with WAL Integration

**Version:** 1.0.0
**Status:** Design Document
**Author:** API Architect
**Date:** 2026-01-01

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Configuration](#configuration)
4. [Core Traits and Interfaces](#core-traits-and-interfaces)
5. [Error Handling Strategy](#error-handling-strategy)
6. [Operations API](#operations-api)
7. [Transaction Support](#transaction-support)
8. [Metrics and Statistics](#metrics-and-statistics)
9. [Migration Path](#migration-path)
10. [Usage Examples](#usage-examples)
11. [Performance Considerations](#performance-considerations)

---

## Executive Summary

This document specifies the unified API for ReasonKit Memory's dual-layer storage system. The design provides:

- **Single Entry Point**: One interface for all storage operations
- **Automatic Hot/Cold Routing**: Intelligent data placement based on access patterns
- **Transparent WAL Integration**: Durability guarantees without API complexity
- **Backward Compatibility**: Full compatibility with existing `StorageBackend` trait
- **Transaction Support**: Optional ACID guarantees for critical operations

---

## Architecture Overview

```
                              +---------------------------+
                              |    DualLayerStorage       |
                              |  (Unified Entry Point)    |
                              +-------------+-------------+
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
           +--------v--------+    +---------v---------+   +---------v--------+
           |   HotMemory     |    |    ColdMemory     |   |       WAL        |
           | (In-Memory/SSD) |    | (Qdrant/File)     |   | (Write-Ahead Log)|
           +--------+--------+    +---------+---------+   +---------+--------+
                    |                       |                       |
                    +-----------+-----------+-----------+-----------+
                                |                       |
                      +---------v---------+   +---------v---------+
                      |  TieringPolicy    |   |   SyncManager     |
                      | (Hot/Cold Router) |   | (Consistency)     |
                      +-------------------+   +-------------------+
```

### Layer Responsibilities

| Layer              | Purpose                  | Characteristics                      |
| ------------------ | ------------------------ | ------------------------------------ |
| **Hot Memory**     | Frequently accessed data | Low latency (<5ms), limited capacity |
| **Cold Memory**    | Archival storage         | Higher latency, unlimited capacity   |
| **WAL**            | Durability               | Append-only, sequential writes       |
| **Tiering Policy** | Data placement           | Access frequency, age-based routing  |
| **Sync Manager**   | Consistency              | WAL replay, background sync          |

---

## Configuration

### DualLayerConfig

```rust
/// Complete configuration for dual-layer storage system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLayerConfig {
    /// Hot memory layer configuration
    pub hot: HotMemoryConfig,
    /// Cold memory layer configuration
    pub cold: ColdMemoryConfig,
    /// Write-ahead log configuration
    pub wal: WalConfig,
    /// Synchronization and tiering configuration
    pub sync: SyncConfig,
}

impl Default for DualLayerConfig {
    fn default() -> Self {
        Self {
            hot: HotMemoryConfig::default(),
            cold: ColdMemoryConfig::default(),
            wal: WalConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}
```

### HotMemoryConfig

```rust
/// Configuration for hot memory layer
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotBackendType {
    /// Pure in-memory (HashMap-based)
    InMemory,
    /// Memory-mapped file
    Mmap { path: PathBuf },
    /// Embedded RocksDB
    RocksDb { path: PathBuf },
    /// Redis/Valkey connection
    Redis { url: String },
}
```

### ColdMemoryConfig

```rust
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
                base_path: default_storage_path(),
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
    File { base_path: PathBuf },
    /// Qdrant vector database (local)
    QdrantLocal { url: String },
    /// Qdrant vector database (cloud)
    QdrantCloud {
        url: String,
        api_key: String,
    },
    /// S3-compatible object storage
    ObjectStorage {
        endpoint: String,
        bucket: String,
        access_key: String,
        secret_key: String,
    },
}

/// Vector quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    None,
    Int8,
    Binary,
    ProductQuantization { segments: usize },
}
```

### WalConfig

```rust
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
            path: default_storage_path().join("wal"),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalSyncMode {
    /// Sync every write (slowest, safest)
    Immediate,
    /// Sync at intervals
    Interval,
    /// Let OS handle syncing (fastest, least safe)
    OsManaged,
    /// No sync - WAL in memory only
    None,
}
```

### SyncConfig

```rust
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
            age_threshold_secs: 86400, // 24 hours
        }
    }
}

/// Storage tier identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    Hot,
    Cold,
}
```

---

## Core Traits and Interfaces

### DualLayerBackend Trait

The primary trait that extends `StorageBackend` with dual-layer capabilities:

```rust
use async_trait::async_trait;
use uuid::Uuid;
use crate::{Document, Result};

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
```

### DualLayerStorage Implementation

```rust
/// Unified dual-layer storage implementation
pub struct DualLayerStorage {
    /// Configuration
    config: DualLayerConfig,
    /// Hot memory layer
    hot: Arc<RwLock<Box<dyn HotMemoryLayer>>>,
    /// Cold memory layer
    cold: Arc<Box<dyn StorageBackend>>,
    /// Write-ahead log
    wal: Arc<RwLock<Option<WriteAheadLog>>>,
    /// Tier routing metadata
    tier_index: Arc<RwLock<TierIndex>>,
    /// Access tracking for tiering decisions
    access_tracker: Arc<RwLock<AccessTracker>>,
    /// Background sync handle
    sync_handle: Option<JoinHandle<()>>,
}

impl DualLayerStorage {
    /// Create new dual-layer storage with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(DualLayerConfig::default()).await
    }

    /// Create with custom configuration
    pub async fn with_config(config: DualLayerConfig) -> Result<Self> {
        // Initialize hot layer
        let hot = Self::create_hot_layer(&config.hot).await?;

        // Initialize cold layer
        let cold = Self::create_cold_layer(&config.cold).await?;

        // Initialize WAL if enabled
        let wal = if config.wal.enabled {
            Some(WriteAheadLog::new(&config.wal).await?)
        } else {
            None
        };

        // Initialize tier index
        let tier_index = TierIndex::new();

        // Initialize access tracker
        let access_tracker = AccessTracker::new(&config.sync.tiering_policy);

        let storage = Self {
            config: config.clone(),
            hot: Arc::new(RwLock::new(hot)),
            cold: Arc::new(cold),
            wal: Arc::new(RwLock::new(wal)),
            tier_index: Arc::new(RwLock::new(tier_index)),
            access_tracker: Arc::new(RwLock::new(access_tracker)),
            sync_handle: None,
        };

        // Start background sync if configured
        if config.sync.background_sync_interval_secs > 0 {
            storage.start_background_sync();
        }

        Ok(storage)
    }

    /// Create storage in embedded mode (file-based, no external dependencies)
    pub async fn embedded(data_path: PathBuf) -> Result<Self> {
        let config = DualLayerConfig {
            hot: HotMemoryConfig {
                backend: HotBackendType::InMemory,
                ..Default::default()
            },
            cold: ColdMemoryConfig {
                backend: ColdBackendType::File { base_path: data_path.clone() },
                ..Default::default()
            },
            wal: WalConfig {
                enabled: true,
                path: data_path.join("wal"),
                ..Default::default()
            },
            sync: SyncConfig::default(),
        };

        Self::with_config(config).await
    }

    /// Create storage with Qdrant backend
    pub async fn with_qdrant(qdrant_url: &str, collection_name: &str) -> Result<Self> {
        let config = DualLayerConfig {
            cold: ColdMemoryConfig {
                backend: ColdBackendType::QdrantLocal {
                    url: qdrant_url.to_string(),
                },
                collection_name: collection_name.to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        Self::with_config(config).await
    }

    // ============================================================
    // INTERNAL HELPERS
    // ============================================================

    /// Determine tier for new data based on policy
    fn determine_tier(&self, doc: &Document) -> StorageTier {
        let policy = &self.config.sync.tiering_policy;

        // Size-based check
        if policy.size_based {
            let size = doc.content.raw.len() as u64;
            if size > policy.size_threshold_bytes {
                return StorageTier::Cold;
            }
        }

        policy.default_tier
    }

    /// Record access for tiering decisions
    async fn record_access(&self, id: &Uuid) {
        let mut tracker = self.access_tracker.write().await;
        tracker.record_access(id);
    }

    /// Start background sync task
    fn start_background_sync(&mut self) {
        let hot = self.hot.clone();
        let cold = self.cold.clone();
        let wal = self.wal.clone();
        let tier_index = self.tier_index.clone();
        let access_tracker = self.access_tracker.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let interval = Duration::from_secs(config.sync.background_sync_interval_secs);
            let mut ticker = tokio::time::interval(interval);

            loop {
                ticker.tick().await;

                // Sync WAL to cold storage
                if let Some(ref wal) = *wal.read().await {
                    let _ = wal.checkpoint().await;
                }

                // Run tiering
                let _ = Self::run_tiering_internal(
                    &hot,
                    &cold,
                    &tier_index,
                    &access_tracker,
                    &config.sync,
                ).await;
            }
        });

        self.sync_handle = Some(handle);
    }
}
```

---

## Error Handling Strategy

### Error Types

```rust
/// Unified error type for dual-layer storage operations
#[derive(Debug, thiserror::Error)]
pub enum DualLayerError {
    // ============================================================
    // LAYER-SPECIFIC ERRORS
    // ============================================================

    /// Hot layer error
    #[error("Hot layer error: {0}")]
    HotLayer(String),

    /// Cold layer error
    #[error("Cold layer error: {0}")]
    ColdLayer(String),

    /// WAL error
    #[error("WAL error: {0}")]
    Wal(String),

    // ============================================================
    // OPERATION ERRORS
    // ============================================================

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

    // ============================================================
    // RESOURCE ERRORS
    // ============================================================

    /// Capacity exceeded
    #[error("Capacity exceeded: {message}")]
    CapacityExceeded { message: String, tier: StorageTier },

    /// Connection error
    #[error("Connection error: {0}")]
    Connection(String),

    /// Timeout
    #[error("Operation timed out after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    // ============================================================
    // VALIDATION ERRORS
    // ============================================================

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Access denied
    #[error("Access denied: {0}")]
    AccessDenied(String),

    // ============================================================
    // WRAPPER ERRORS
    // ============================================================

    /// Underlying storage error
    #[error("Storage error: {0}")]
    Storage(#[from] crate::MemError),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for dual-layer operations
pub type DualLayerResult<T> = std::result::Result<T, DualLayerError>;

/// Error context for better diagnostics
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// Affected tier
    pub tier: Option<StorageTier>,
    /// Affected document ID
    pub document_id: Option<Uuid>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl DualLayerError {
    /// Create error with context
    pub fn with_context(self, ctx: ErrorContext) -> ContextualError {
        ContextualError {
            error: self,
            context: ctx,
        }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            DualLayerError::Connection(_)
                | DualLayerError::Timeout { .. }
                | DualLayerError::Sync(_)
        )
    }

    /// Check if error indicates data loss
    pub fn may_cause_data_loss(&self) -> bool {
        matches!(
            self,
            DualLayerError::Wal(_)
                | DualLayerError::Sync(_)
                | DualLayerError::Transaction(_)
        )
    }
}

/// Error with additional context
#[derive(Debug)]
pub struct ContextualError {
    pub error: DualLayerError,
    pub context: ErrorContext,
}
```

### Error Recovery Strategies

```rust
/// Recovery strategies for different error types
pub struct RecoveryStrategy;

impl RecoveryStrategy {
    /// Retry with exponential backoff
    pub async fn retry_with_backoff<F, T, E>(
        operation: F,
        max_retries: usize,
        base_delay_ms: u64,
    ) -> Result<T, E>
    where
        F: Fn() -> Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
        E: std::fmt::Debug,
    {
        let mut delay = Duration::from_millis(base_delay_ms);

        for attempt in 0..max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt + 1 == max_retries {
                        return Err(e);
                    }
                    tracing::warn!(
                        "Operation failed (attempt {}/{}): {:?}",
                        attempt + 1,
                        max_retries,
                        e
                    );
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
            }
        }

        unreachable!()
    }

    /// Fallback to cold storage on hot layer failure
    pub async fn fallback_to_cold<T>(
        hot_operation: impl Future<Output = Result<T, DualLayerError>>,
        cold_operation: impl Future<Output = Result<T, DualLayerError>>,
    ) -> Result<T, DualLayerError> {
        match hot_operation.await {
            Ok(result) => Ok(result),
            Err(DualLayerError::HotLayer(_)) => {
                tracing::warn!("Hot layer failed, falling back to cold storage");
                cold_operation.await
            }
            Err(e) => Err(e),
        }
    }
}
```

---

## Operations API

### CRUD Operations

```rust
#[async_trait]
impl StorageBackend for DualLayerStorage {
    /// Store a document (auto-routes to appropriate tier)
    async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()> {
        // Determine tier
        let tier = self.determine_tier(doc);

        // Write to WAL first (if enabled)
        if let Some(ref wal) = *self.wal.read().await {
            wal.append(WalEntry::StoreDocument {
                id: doc.id,
                tier,
                timestamp: Utc::now(),
            }).await?;
        }

        // Write to appropriate tier
        match tier {
            StorageTier::Hot => {
                let mut hot = self.hot.write().await;
                hot.store(doc).await?;
            }
            StorageTier::Cold => {
                self.cold.store_document(doc, context).await?;
            }
        }

        // Update tier index
        let mut index = self.tier_index.write().await;
        index.set_tier(doc.id, tier);

        // Record access
        self.record_access(&doc.id).await;

        Ok(())
    }

    /// Get a document by ID (checks both tiers)
    async fn get_document(&self, id: &Uuid, context: &AccessContext) -> Result<Option<Document>> {
        // Check tier index first
        let tier = {
            let index = self.tier_index.read().await;
            index.get_tier(id)
        };

        // Record access for tiering decisions
        self.record_access(id).await;

        match tier {
            Some(StorageTier::Hot) => {
                let hot = self.hot.read().await;
                if let Some(doc) = hot.get(id).await? {
                    return Ok(Some(doc));
                }
                // Fallback to cold if not in hot (may have been evicted)
                self.cold.get_document(id, context).await
            }
            Some(StorageTier::Cold) => {
                self.cold.get_document(id, context).await
            }
            None => {
                // Unknown tier - check both
                let hot = self.hot.read().await;
                if let Some(doc) = hot.get(id).await? {
                    return Ok(Some(doc));
                }
                self.cold.get_document(id, context).await
            }
        }
    }

    /// Delete a document from all tiers
    async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()> {
        // Write to WAL
        if let Some(ref wal) = *self.wal.read().await {
            wal.append(WalEntry::DeleteDocument {
                id: *id,
                timestamp: Utc::now(),
            }).await?;
        }

        // Delete from hot
        {
            let mut hot = self.hot.write().await;
            let _ = hot.delete(id).await;
        }

        // Delete from cold
        self.cold.delete_document(id, context).await?;

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

        Ok(())
    }

    /// List all document IDs
    async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>> {
        let tier_index = self.tier_index.read().await;
        Ok(tier_index.all_ids())
    }

    // ... remaining StorageBackend methods implemented similarly
}
```

### Bulk Operations

```rust
impl DualLayerStorage {
    /// Store multiple documents efficiently
    pub async fn store_documents_bulk(
        &self,
        docs: &[Document],
        context: &AccessContext,
    ) -> Result<BulkOperationResult> {
        let mut result = BulkOperationResult::default();
        let batch_size = self.config.sync.sync_batch_size;

        // Group by tier
        let mut hot_docs = Vec::new();
        let mut cold_docs = Vec::new();

        for doc in docs {
            match self.determine_tier(doc) {
                StorageTier::Hot => hot_docs.push(doc),
                StorageTier::Cold => cold_docs.push(doc),
            }
        }

        // Write hot documents in batches
        for batch in hot_docs.chunks(batch_size) {
            let mut hot = self.hot.write().await;
            for doc in batch {
                match hot.store(doc).await {
                    Ok(_) => result.succeeded += 1,
                    Err(e) => {
                        result.failed += 1;
                        result.errors.push((doc.id, e.to_string()));
                    }
                }
            }
        }

        // Write cold documents in batches
        for batch in cold_docs.chunks(batch_size) {
            for doc in batch {
                match self.cold.store_document(doc, context).await {
                    Ok(_) => result.succeeded += 1,
                    Err(e) => {
                        result.failed += 1;
                        result.errors.push((doc.id, e.to_string()));
                    }
                }
            }
        }

        // Update tier index
        {
            let mut index = self.tier_index.write().await;
            for doc in &hot_docs {
                index.set_tier(doc.id, StorageTier::Hot);
            }
            for doc in &cold_docs {
                index.set_tier(doc.id, StorageTier::Cold);
            }
        }

        Ok(result)
    }

    /// Get multiple documents by ID
    pub async fn get_documents_bulk(
        &self,
        ids: &[Uuid],
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, Option<Document>)>> {
        let mut results = Vec::with_capacity(ids.len());

        // Group by tier for efficient access
        let tier_index = self.tier_index.read().await;
        let mut hot_ids = Vec::new();
        let mut cold_ids = Vec::new();
        let mut unknown_ids = Vec::new();

        for id in ids {
            match tier_index.get_tier(id) {
                Some(StorageTier::Hot) => hot_ids.push(*id),
                Some(StorageTier::Cold) => cold_ids.push(*id),
                None => unknown_ids.push(*id),
            }
        }
        drop(tier_index);

        // Fetch from hot tier
        {
            let hot = self.hot.read().await;
            for id in hot_ids {
                let doc = hot.get(&id).await.ok().flatten();
                results.push((id, doc));
            }
        }

        // Fetch from cold tier
        for id in cold_ids {
            let doc = self.cold.get_document(&id, context).await.ok().flatten();
            results.push((id, doc));
        }

        // Check both tiers for unknown
        for id in unknown_ids {
            let doc = self.get_document(&id, context).await.ok().flatten();
            results.push((id, doc));
        }

        Ok(results)
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
```

---

## Transaction Support

### Transaction API

```rust
/// Transaction for ACID guarantees on storage operations
pub struct StorageTransaction {
    /// Transaction ID
    id: Uuid,
    /// Storage reference
    storage: Arc<DualLayerStorage>,
    /// Pending operations
    operations: Vec<TransactionOperation>,
    /// Transaction state
    state: TransactionState,
    /// Savepoints
    savepoints: Vec<Savepoint>,
    /// Isolation level
    isolation: IsolationLevel,
}

/// Transaction operations
#[derive(Debug, Clone)]
pub enum TransactionOperation {
    StoreDocument { doc: Document, tier: StorageTier },
    DeleteDocument { id: Uuid },
    StoreEmbedding { chunk_id: Uuid, embedding: Vec<f32> },
    MoveTier { id: Uuid, from: StorageTier, to: StorageTier },
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Active,
    Committed,
    RolledBack,
    Failed,
}

/// Isolation levels
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

/// Savepoint for partial rollback
#[derive(Debug, Clone)]
pub struct Savepoint {
    name: String,
    operation_index: usize,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl StorageTransaction {
    /// Begin a new transaction
    pub async fn begin(storage: Arc<DualLayerStorage>) -> Result<Self> {
        Self::begin_with_isolation(storage, IsolationLevel::ReadCommitted).await
    }

    /// Begin with specific isolation level
    pub async fn begin_with_isolation(
        storage: Arc<DualLayerStorage>,
        isolation: IsolationLevel,
    ) -> Result<Self> {
        let tx = Self {
            id: Uuid::new_v4(),
            storage,
            operations: Vec::new(),
            state: TransactionState::Active,
            savepoints: Vec::new(),
            isolation,
        };

        tracing::debug!("Transaction {} started with {:?} isolation", tx.id, isolation);
        Ok(tx)
    }

    /// Store a document within transaction
    pub async fn store_document(&mut self, doc: Document) -> Result<()> {
        self.ensure_active()?;

        let tier = self.storage.determine_tier(&doc);
        self.operations.push(TransactionOperation::StoreDocument { doc, tier });

        Ok(())
    }

    /// Delete a document within transaction
    pub async fn delete_document(&mut self, id: Uuid) -> Result<()> {
        self.ensure_active()?;

        self.operations.push(TransactionOperation::DeleteDocument { id });

        Ok(())
    }

    /// Create savepoint for partial rollback
    pub fn savepoint(&mut self, name: &str) -> Result<()> {
        self.ensure_active()?;

        self.savepoints.push(Savepoint {
            name: name.to_string(),
            operation_index: self.operations.len(),
            timestamp: Utc::now(),
        });

        Ok(())
    }

    /// Rollback to savepoint
    pub fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        self.ensure_active()?;

        if let Some(sp) = self.savepoints.iter().rev().find(|s| s.name == name) {
            self.operations.truncate(sp.operation_index);
            // Remove savepoints after this one
            self.savepoints.retain(|s| s.operation_index <= sp.operation_index);
            Ok(())
        } else {
            Err(MemError::validation(format!("Savepoint '{}' not found", name)))
        }
    }

    /// Commit transaction
    pub async fn commit(mut self) -> Result<TransactionResult> {
        self.ensure_active()?;

        let start = Instant::now();
        let context = AccessContext::new(
            "transaction".to_string(),
            AccessLevel::Admin,
            format!("tx_{}", self.id),
        );

        // Write to WAL first
        if let Some(ref wal) = *self.storage.wal.read().await {
            wal.append(WalEntry::TransactionBegin {
                id: self.id,
                timestamp: Utc::now(),
            }).await?;

            for op in &self.operations {
                match op {
                    TransactionOperation::StoreDocument { doc, tier } => {
                        wal.append(WalEntry::StoreDocument {
                            id: doc.id,
                            tier: *tier,
                            timestamp: Utc::now(),
                        }).await?;
                    }
                    TransactionOperation::DeleteDocument { id } => {
                        wal.append(WalEntry::DeleteDocument {
                            id: *id,
                            timestamp: Utc::now(),
                        }).await?;
                    }
                    _ => {}
                }
            }

            wal.append(WalEntry::TransactionCommit {
                id: self.id,
                timestamp: Utc::now(),
            }).await?;
        }

        // Apply operations
        let mut succeeded = 0;
        let mut failed = 0;

        for op in self.operations {
            match op {
                TransactionOperation::StoreDocument { doc, tier } => {
                    let result = match tier {
                        StorageTier::Hot => {
                            let mut hot = self.storage.hot.write().await;
                            hot.store(&doc).await
                        }
                        StorageTier::Cold => {
                            self.storage.cold.store_document(&doc, &context).await
                        }
                    };

                    match result {
                        Ok(_) => {
                            succeeded += 1;
                            let mut index = self.storage.tier_index.write().await;
                            index.set_tier(doc.id, tier);
                        }
                        Err(_) => failed += 1,
                    }
                }
                TransactionOperation::DeleteDocument { id } => {
                    match self.storage.delete_document(&id, &context).await {
                        Ok(_) => succeeded += 1,
                        Err(_) => failed += 1,
                    }
                }
                _ => {}
            }
        }

        self.state = TransactionState::Committed;

        Ok(TransactionResult {
            transaction_id: self.id,
            operations_succeeded: succeeded,
            operations_failed: failed,
            duration: start.elapsed(),
        })
    }

    /// Rollback transaction
    pub async fn rollback(mut self) -> Result<()> {
        self.ensure_active()?;

        // Write rollback to WAL
        if let Some(ref wal) = *self.storage.wal.read().await {
            wal.append(WalEntry::TransactionRollback {
                id: self.id,
                timestamp: Utc::now(),
            }).await?;
        }

        self.operations.clear();
        self.state = TransactionState::RolledBack;

        tracing::debug!("Transaction {} rolled back", self.id);
        Ok(())
    }

    fn ensure_active(&self) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(MemError::validation(format!(
                "Transaction {} is not active (state: {:?})",
                self.id, self.state
            )));
        }
        Ok(())
    }
}

/// Transaction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionResult {
    pub transaction_id: Uuid,
    pub operations_succeeded: usize,
    pub operations_failed: usize,
    pub duration: Duration,
}
```

---

## Metrics and Statistics

### Statistics Types

```rust
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
    pub collected_at: chrono::DateTime<chrono::Utc>,
}

/// Statistics for a single tier
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalStats {
    /// WAL enabled
    pub enabled: bool,
    /// Current WAL size in bytes
    pub current_size_bytes: u64,
    /// Number of pending entries
    pub pending_entries: usize,
    /// Last sync timestamp
    pub last_sync_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Last checkpoint timestamp
    pub last_checkpoint_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Write throughput (entries/sec)
    pub write_throughput: f64,
}

/// Overall statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
```

---

## Migration Path

### From Existing StorageBackend

```rust
/// Migration utilities for transitioning to DualLayerStorage
pub struct StorageMigration;

impl StorageMigration {
    /// Migrate from existing StorageBackend to DualLayerStorage
    pub async fn from_storage_backend<S: StorageBackend>(
        source: S,
        target_config: DualLayerConfig,
    ) -> Result<DualLayerStorage> {
        // Create target storage
        let target = DualLayerStorage::with_config(target_config).await?;

        // Create admin context
        let context = AccessContext::new(
            "migration".to_string(),
            AccessLevel::Admin,
            "migrate".to_string(),
        );

        // List all documents from source
        let doc_ids = source.list_documents(&context).await?;

        tracing::info!("Migrating {} documents...", doc_ids.len());

        let mut migrated = 0;
        let mut errors = 0;

        for doc_id in doc_ids {
            match source.get_document(&doc_id, &context).await {
                Ok(Some(doc)) => {
                    match target.store_document(&doc, &context).await {
                        Ok(_) => migrated += 1,
                        Err(e) => {
                            tracing::error!("Failed to migrate document {}: {}", doc_id, e);
                            errors += 1;
                        }
                    }
                }
                Ok(None) => {
                    tracing::warn!("Document {} not found during migration", doc_id);
                }
                Err(e) => {
                    tracing::error!("Failed to read document {}: {}", doc_id, e);
                    errors += 1;
                }
            }

            if migrated % 100 == 0 {
                tracing::info!("Progress: {}/{} documents migrated", migrated, doc_ids.len());
            }
        }

        tracing::info!(
            "Migration complete: {} migrated, {} errors",
            migrated,
            errors
        );

        Ok(target)
    }

    /// Create DualLayerStorage wrapper around existing backend
    /// (Immediate compatibility, no data migration)
    pub async fn wrap_as_cold_tier<S: StorageBackend + 'static>(
        existing: S,
        hot_config: HotMemoryConfig,
        wal_config: WalConfig,
        sync_config: SyncConfig,
    ) -> Result<DualLayerStorage> {
        // Create hot layer
        let hot = DualLayerStorage::create_hot_layer(&hot_config).await?;

        // Use existing as cold layer
        let cold: Box<dyn StorageBackend> = Box::new(existing);

        // Initialize WAL
        let wal = if wal_config.enabled {
            Some(WriteAheadLog::new(&wal_config).await?)
        } else {
            None
        };

        Ok(DualLayerStorage {
            config: DualLayerConfig {
                hot: hot_config,
                cold: ColdMemoryConfig::default(), // Won't be used
                wal: wal_config,
                sync: sync_config,
            },
            hot: Arc::new(RwLock::new(hot)),
            cold: Arc::new(cold),
            wal: Arc::new(RwLock::new(wal)),
            tier_index: Arc::new(RwLock::new(TierIndex::new())),
            access_tracker: Arc::new(RwLock::new(AccessTracker::new(
                &sync_config.tiering_policy,
            ))),
            sync_handle: None,
        })
    }
}

/// Backward compatibility adapter
/// Allows DualLayerStorage to be used where StorageBackend is expected
impl AsRef<dyn StorageBackend> for DualLayerStorage {
    fn as_ref(&self) -> &(dyn StorageBackend + 'static) {
        self
    }
}
```

---

## Usage Examples

### Basic Usage

```rust
use reasonkit_mem::storage::{
    DualLayerStorage, DualLayerConfig, StorageTier,
    AccessContext, AccessLevel,
};
use reasonkit_mem::{Document, DocumentType, Source, SourceType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create storage with default configuration
    let storage = DualLayerStorage::new().await?;

    // Or create embedded storage (file-based, no external dependencies)
    let storage = DualLayerStorage::embedded("./data".into()).await?;

    // Create access context
    let context = AccessContext::new(
        "user123".to_string(),
        AccessLevel::ReadWrite,
        "store_document".to_string(),
    );

    // Create a document
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some("example.md".to_string()),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: chrono::Utc::now(),
        version: None,
    };

    let doc = Document::new(DocumentType::Note, source)
        .with_content("This is an example document.".to_string());

    // Store document (auto-routes to hot tier for small documents)
    storage.store_document(&doc, &context).await?;

    // Retrieve document
    let retrieved = storage.get_document(&doc.id, &context).await?;
    assert!(retrieved.is_some());

    // Check which tier the document is in
    let tier = storage.get_tier(&doc.id).await?;
    println!("Document is in {:?} tier", tier);

    // Force move to cold tier
    storage.move_to_tier(&doc.id, StorageTier::Cold, &context).await?;

    // Get statistics
    let stats = storage.detailed_stats(&context).await?;
    println!("Hot tier: {} docs, Cold tier: {} docs",
        stats.hot.document_count,
        stats.cold.document_count);

    Ok(())
}
```

### Bulk Operations

```rust
use reasonkit_mem::storage::DualLayerStorage;

async fn bulk_example(storage: &DualLayerStorage) -> anyhow::Result<()> {
    let context = AccessContext::new(
        "bulk_user".to_string(),
        AccessLevel::Admin,
        "bulk_insert".to_string(),
    );

    // Generate many documents
    let documents: Vec<Document> = (0..1000)
        .map(|i| {
            let source = Source {
                source_type: SourceType::Local,
                url: None,
                path: Some(format!("doc_{}.md", i)),
                arxiv_id: None,
                github_repo: None,
                retrieved_at: chrono::Utc::now(),
                version: None,
            };
            Document::new(DocumentType::Note, source)
                .with_content(format!("Document content {}", i))
        })
        .collect();

    // Bulk insert
    let result = storage.store_documents_bulk(&documents, &context).await?;
    println!(
        "Bulk insert: {} succeeded, {} failed in {:?}",
        result.succeeded,
        result.failed,
        result.duration
    );

    // Bulk get
    let ids: Vec<Uuid> = documents.iter().map(|d| d.id).collect();
    let retrieved = storage.get_documents_bulk(&ids, &context).await?;

    let found_count = retrieved.iter().filter(|(_, doc)| doc.is_some()).count();
    println!("Found {}/{} documents", found_count, ids.len());

    Ok(())
}
```

### Transactions

```rust
use reasonkit_mem::storage::{DualLayerStorage, StorageTransaction, IsolationLevel};

async fn transaction_example(storage: Arc<DualLayerStorage>) -> anyhow::Result<()> {
    // Begin transaction
    let mut tx = StorageTransaction::begin(storage.clone()).await?;

    // Create savepoint
    tx.savepoint("before_changes")?;

    // Perform operations
    let doc1 = create_document("doc1");
    tx.store_document(doc1).await?;

    let doc2 = create_document("doc2");
    tx.store_document(doc2).await?;

    // Rollback to savepoint if needed
    // tx.rollback_to_savepoint("before_changes")?;

    // Commit transaction
    let result = tx.commit().await?;
    println!(
        "Transaction {} committed: {} ops in {:?}",
        result.transaction_id,
        result.operations_succeeded,
        result.duration
    );

    Ok(())
}
```

### Custom Configuration

```rust
use reasonkit_mem::storage::*;
use std::path::PathBuf;

async fn custom_config_example() -> anyhow::Result<()> {
    let config = DualLayerConfig {
        hot: HotMemoryConfig {
            max_capacity_bytes: 512 * 1024 * 1024, // 512MB
            max_entries: 50_000,
            ttl_secs: 7200, // 2 hours
            eviction_policy: EvictionPolicy::Adaptive,
            backend: HotBackendType::InMemory,
            compression_enabled: true,
            compression_level: 5,
        },
        cold: ColdMemoryConfig {
            backend: ColdBackendType::QdrantLocal {
                url: "http://localhost:6333".to_string(),
            },
            collection_name: "my_collection".to_string(),
            vector_size: 768,
            quantization_enabled: true,
            quantization_type: QuantizationType::Int8,
            connection: QdrantConnectionConfig {
                max_connections: 20,
                connect_timeout_secs: 30,
                ..Default::default()
            },
            embedding_cache: EmbeddingCacheConfig {
                max_size: 20000,
                ttl_secs: 7200,
            },
            access_control: AccessControlConfig::default(),
        },
        wal: WalConfig {
            enabled: true,
            path: PathBuf::from("./wal"),
            max_file_size: 128 * 1024 * 1024, // 128MB
            sync_mode: WalSyncMode::Interval,
            sync_interval_ms: 500,
            retention_secs: 172800, // 48 hours
            compression_enabled: true,
            checkpoint_threshold: 5000,
        },
        sync: SyncConfig {
            tiering_policy: TieringPolicy {
                default_tier: StorageTier::Hot,
                access_based: true,
                age_based: true,
                size_based: true,
                size_threshold_bytes: 5 * 1024 * 1024, // 5MB
                age_threshold_secs: 7200, // 2 hours
            },
            background_sync_interval_secs: 30,
            sync_batch_size: 200,
            auto_promote: true,
            promotion_threshold: 5,
            demotion_threshold_secs: 3600,
            parallel_sync: true,
            max_concurrent_syncs: 8,
        },
    };

    let storage = DualLayerStorage::with_config(config).await?;

    Ok(())
}
```

---

## Performance Considerations

### Latency Targets

| Operation        | Hot Tier Target | Cold Tier Target |
| ---------------- | --------------- | ---------------- |
| Single read      | < 1ms           | < 10ms           |
| Single write     | < 2ms           | < 20ms           |
| Bulk read (100)  | < 10ms          | < 100ms          |
| Bulk write (100) | < 20ms          | < 200ms          |
| Vector search    | < 5ms           | < 50ms           |

### Memory Budget Guidelines

| Deployment Size | Hot Tier Memory | Recommended Entries |
| --------------- | --------------- | ------------------- |
| Small (dev)     | 256MB           | 25,000              |
| Medium          | 1GB             | 100,000             |
| Large           | 4GB             | 400,000             |
| Enterprise      | 16GB+           | 1,500,000+          |

### Tiering Policy Tuning

```rust
/// Aggressive tiering for memory-constrained environments
let aggressive_policy = TieringPolicy {
    default_tier: StorageTier::Cold, // New data goes to cold
    access_based: true,
    age_based: true,
    size_based: true,
    size_threshold_bytes: 1024 * 1024, // 1MB
    age_threshold_secs: 300, // 5 minutes
};

/// Conservative tiering for performance-critical workloads
let conservative_policy = TieringPolicy {
    default_tier: StorageTier::Hot, // New data stays hot
    access_based: true,
    age_based: true,
    size_based: false, // Don't demote large items automatically
    size_threshold_bytes: 100 * 1024 * 1024, // 100MB
    age_threshold_secs: 86400, // 24 hours
};
```

### WAL Tuning

```rust
/// High-durability configuration
let durable_wal = WalConfig {
    sync_mode: WalSyncMode::Immediate,
    sync_interval_ms: 0, // Not used with Immediate
    checkpoint_threshold: 1000,
    ..Default::default()
};

/// High-throughput configuration
let fast_wal = WalConfig {
    sync_mode: WalSyncMode::Interval,
    sync_interval_ms: 5000, // 5 second batches
    checkpoint_threshold: 50000,
    ..Default::default()
};
```

---

## File Locations

| Artifact        | Path                                    |
| --------------- | --------------------------------------- |
| Design Document | `docs/DUAL_LAYER_STORAGE_API_DESIGN.md` |
| Implementation  | `src/storage/dual_layer.rs`             |
| Tests           | `tests/dual_layer_tests.rs`             |
| Benchmarks      | `benches/dual_layer_bench.rs`           |

---

## Revision History

| Version | Date       | Author        | Changes                 |
| ------- | ---------- | ------------- | ----------------------- |
| 1.0.0   | 2026-01-01 | API Architect | Initial design document |

---

_This design follows ReasonKit's Rust-first philosophy and is compatible with the existing `StorageBackend` trait while providing enhanced capabilities for dual-layer memory management._
