# Dual-Layer Storage Architecture for ReasonKit-Mem

**Version:** 1.0.0
**Date:** 2026-01-01
**Author:** Database Architect
**Status:** Design Document

---

## Executive Summary

This document specifies the complete architecture for a dual-layer memory system in `reasonkit-mem`. The design implements a Hot/Cold storage pattern optimized for AI agent session memory, featuring:

- **Hot Layer**: DashMap-based concurrent cache with TTL and LRU eviction
- **Cold Layer**: Embedded vector store + WAL-backed persistent storage
- **Background Sync Worker**: Async transitions between layers

**Key Design Goals:**

1. Sub-millisecond hot memory access (<1ms p99)
2. Zero external dependencies for embedded mode
3. Crash-safe with WAL recovery
4. Memory-bounded with configurable limits
5. Portable across Debian 13 and other Linux distributions

---

## Architecture Overview

```
+==============================================================================+
|                           REASONKIT-MEM DUAL-LAYER STORAGE                   |
+==============================================================================+

                              +-----------------------+
                              |     Application       |
                              |  (HybridRetriever,    |
                              |   RAG Pipeline)       |
                              +-----------+-----------+
                                          |
                                          v
+==============================================================================+
|                              UNIFIED STORAGE API                             |
|                           (StorageBackend trait)                             |
+==============================================================================+
                    |                               |
          +---------+                               +---------+
          |                                                   |
          v                                                   v
+-------------------+                               +-------------------+
|   HOT MEMORY      |                               |   COLD STORAGE    |
|   LAYER (L1)      |                               |   LAYER (L2)      |
+-------------------+                               +-------------------+
|                   |         Promotion             |                   |
|  +-----------+    |  <------------------------    |  +-------------+  |
|  | DashMap   |    |                               |  | Embedded    |  |
|  | Session   |    |         Demotion              |  | Vector DB   |  |
|  | Store     |    |  ------------------------->   |  | (LanceDB/   |  |
|  +-----------+    |                               |  |  Pure Rust) |  |
|  | TTL Index |    |                               |  +-------------+  |
|  +-----------+    |                               |  | Sled KV     |  |
|  | LRU Queue |    |       Background Sync         |  | (Metadata)  |  |
|  +-----------+    |  <========================>   |  +-------------+  |
|  | Memory    |    |                               |  | Write-Ahead |  |
|  | Metrics   |    |                               |  | Log (WAL)   |  |
|  +-----------+    |                               |  +-------------+  |
+-------------------+                               +-------------------+
         |                                                    |
         |                                                    |
         +--------------------+  +----------------------------+
                              |  |
                              v  v
                    +-------------------+
                    |  SYNC WORKER      |
                    |  (Background)     |
                    +-------------------+
                    | - Promotion logic |
                    | - Demotion logic  |
                    | - TTL cleanup     |
                    | - WAL compaction  |
                    +-------------------+
```

---

## Layer 1: Hot Memory Layer

### Purpose

The Hot Memory Layer serves as an ultra-fast cache for active session context. It stores:

- Current conversation embeddings
- Recently accessed chunks
- Session-specific metadata
- Active RAPTOR tree nodes

### Technology Choice: DashMap

**Why DashMap over alternatives:**

| Solution             | Pros                                          | Cons                            | Decision     |
| -------------------- | --------------------------------------------- | ------------------------------- | ------------ |
| **DashMap**          | Lock-free reads, concurrent writes, pure Rust | Slightly higher memory overhead | **SELECTED** |
| std HashMap + RwLock | Simple, stdlib                                | Global lock contention          | Rejected     |
| Crossbeam SkipMap    | Ordered keys                                  | Higher complexity               | Rejected     |
| Papaya HashMap       | Very fast                                     | Less battle-tested              | Alternative  |

### Data Flow Diagram

```
+==============================================================================+
|                         HOT MEMORY DATA FLOW                                 |
+==============================================================================+

   Write Path                                      Read Path
   ----------                                      ---------

   [1] Store Request                              [1] Get Request
         |                                              |
         v                                              v
   +------------+                                +-------------+
   | Validate   |                                | Check Hot   |
   | Memory     |                                | Cache       |
   | Budget     |                                +------+------+
   +-----+------+                                       |
         |                                        +-----+-----+
         v                                        |           |
   +------------+     [Memory Full?]        [Hit] |      [Miss]
   | LRU Check  +---> Evict Oldest ----+          v           |
   +-----+------+                      |    +---------+       |
         |                             |    | Return  |       |
         v                             v    | Fast    |       |
   +------------+              +----------+ +---------+       |
   | DashMap    |              | Write to |                   v
   | Insert     |              | Cold WAL |           +-------------+
   +-----+------+              +----------+           | Promote     |
         |                                            | from Cold   |
         v                                            +------+------+
   +------------+                                            |
   | Update TTL |                                            v
   | & LRU      |                                     +-------------+
   +-----+------+                                     | Insert Hot  |
         |                                            | + Return    |
         v                                            +-------------+
   +------------+
   | Emit Write |
   | to WAL     |
   +------------+

                    BACKGROUND OPERATIONS
                    ---------------------

   [Interval: 100ms]                    [Interval: 5s]
   +----------------+                   +------------------+
   | TTL Expiration |                   | Memory Pressure  |
   | Check & Evict  |                   | Assessment       |
   +-------+--------+                   +--------+---------+
           |                                     |
           v                                     v
   +----------------+                   +------------------+
   | Mark Expired   |                   | If > 80% limit:  |
   | Entries for    |                   | Trigger LRU      |
   | Cold Promotion |                   | Eviction to Cold |
   +----------------+                   +------------------+
```

### Type Definitions

```rust
//! Hot Memory Layer Types for reasonkit-mem
//!
//! Provides ultra-fast, thread-safe session memory with automatic expiration.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Configuration for the Hot Memory Layer
#[derive(Debug, Clone)]
pub struct HotLayerConfig {
    /// Maximum memory budget in bytes (default: 256 MB)
    pub max_memory_bytes: usize,

    /// Default TTL for entries (default: 30 minutes)
    pub default_ttl: Duration,

    /// Maximum number of entries regardless of memory (default: 100,000)
    pub max_entries: usize,

    /// LRU eviction batch size (default: 1000)
    pub eviction_batch_size: usize,

    /// TTL check interval (default: 100ms)
    pub ttl_check_interval: Duration,

    /// Memory pressure threshold (0.0-1.0, default: 0.8)
    pub memory_pressure_threshold: f32,

    /// Enable write-through to cold storage
    pub write_through: bool,

    /// Session isolation (entries keyed by session)
    pub session_isolation: bool,
}

impl Default for HotLayerConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            default_ttl: Duration::from_secs(30 * 60), // 30 minutes
            max_entries: 100_000,
            eviction_batch_size: 1000,
            ttl_check_interval: Duration::from_millis(100),
            memory_pressure_threshold: 0.8,
            write_through: true,
            session_isolation: true,
        }
    }
}

/// Entry type enumeration for hot storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HotEntryType {
    /// Dense embedding vector
    Embedding,
    /// Text chunk content
    Chunk,
    /// Document metadata
    DocumentMeta,
    /// RAPTOR tree node
    RaptorNode,
    /// Session context
    SessionContext,
    /// Query cache
    QueryCache,
}

/// A single entry in the hot memory layer
#[derive(Debug, Clone)]
pub struct HotEntry {
    /// Unique entry ID (typically chunk UUID)
    pub id: Uuid,

    /// Entry type
    pub entry_type: HotEntryType,

    /// Optional session ID for isolation
    pub session_id: Option<Uuid>,

    /// Dense embedding vector (if applicable)
    pub embedding: Option<Vec<f32>>,

    /// Text content (for chunks/context)
    pub text: Option<String>,

    /// Serialized metadata (JSON bytes)
    pub metadata: Option<Vec<u8>>,

    /// Parent document ID
    pub document_id: Option<Uuid>,

    /// Creation timestamp
    pub created_at: Instant,

    /// Last access timestamp
    pub last_accessed: Instant,

    /// Expiration timestamp
    pub expires_at: Instant,

    /// Access count for frequency tracking
    pub access_count: u64,

    /// Approximate size in bytes
    pub size_bytes: usize,

    /// Source layer (for tracking promotions)
    pub source: EntrySource,
}

/// Source of an entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntrySource {
    /// Newly inserted
    Fresh,
    /// Promoted from cold storage
    Promoted,
    /// Loaded from WAL during recovery
    Recovered,
}

impl HotEntry {
    /// Calculate approximate memory size of this entry
    pub fn calculate_size(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();

        if let Some(ref emb) = self.embedding {
            size += emb.len() * std::mem::size_of::<f32>();
        }
        if let Some(ref text) = self.text {
            size += text.len();
        }
        if let Some(ref meta) = self.metadata {
            size += meta.len();
        }

        size
    }

    /// Check if entry has expired
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Refresh expiration based on access
    pub fn refresh_ttl(&mut self, ttl: Duration) {
        self.last_accessed = Instant::now();
        self.expires_at = Instant::now() + ttl;
        self.access_count += 1;
    }
}

/// Composite key for session-isolated entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HotKey {
    /// Entry UUID
    pub id: Uuid,
    /// Optional session isolation
    pub session_id: Option<Uuid>,
}

impl HotKey {
    pub fn new(id: Uuid, session_id: Option<Uuid>) -> Self {
        Self { id, session_id }
    }

    pub fn global(id: Uuid) -> Self {
        Self { id, session_id: None }
    }
}

/// LRU tracking node for eviction ordering
#[derive(Debug, Clone)]
pub struct LruNode {
    pub key: HotKey,
    pub last_accessed: Instant,
    pub access_count: u64,
}

/// Real-time metrics for the hot layer
#[derive(Debug, Default)]
pub struct HotLayerMetrics {
    /// Current number of entries
    pub entry_count: AtomicUsize,
    /// Current memory usage in bytes
    pub memory_bytes: AtomicUsize,
    /// Total cache hits
    pub hits: AtomicU64,
    /// Total cache misses
    pub misses: AtomicU64,
    /// Total evictions
    pub evictions: AtomicU64,
    /// Total promotions from cold
    pub promotions: AtomicU64,
    /// Total demotions to cold
    pub demotions: AtomicU64,
    /// TTL expirations
    pub expirations: AtomicU64,
}

impl HotLayerMetrics {
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }

    pub fn memory_utilization(&self, max_bytes: usize) -> f64 {
        let used = self.memory_bytes.load(Ordering::Relaxed) as f64;
        let max = max_bytes as f64;
        if max > 0.0 { used / max } else { 0.0 }
    }
}

/// Commands sent to the background sync worker
#[derive(Debug)]
pub enum SyncCommand {
    /// Demote an entry to cold storage
    Demote(HotEntry),
    /// Batch demote multiple entries
    BatchDemote(Vec<HotEntry>),
    /// Request promotion of an entry
    PromoteRequest(Uuid),
    /// Flush all pending writes
    Flush,
    /// Compact the WAL
    CompactWal,
    /// Graceful shutdown
    Shutdown,
}

/// Events emitted by the hot layer for observability
#[derive(Debug, Clone)]
pub enum HotLayerEvent {
    /// Entry inserted
    Inserted { key: HotKey, size_bytes: usize },
    /// Entry accessed (hit)
    Accessed { key: HotKey },
    /// Entry evicted (LRU or TTL)
    Evicted { key: HotKey, reason: EvictionReason },
    /// Memory pressure triggered
    MemoryPressure { current_bytes: usize, threshold_bytes: usize },
    /// Promotion completed
    Promoted { key: HotKey, latency_us: u64 },
}

/// Reason for eviction
#[derive(Debug, Clone, Copy)]
pub enum EvictionReason {
    TTLExpired,
    LRUEviction,
    MemoryPressure,
    ManualDelete,
}

/// The Hot Memory Layer implementation
pub struct HotMemoryLayer {
    /// Primary storage: concurrent hashmap
    store: DashMap<HotKey, HotEntry>,

    /// Configuration
    config: HotLayerConfig,

    /// Real-time metrics
    metrics: Arc<HotLayerMetrics>,

    /// Channel to sync worker
    sync_tx: mpsc::UnboundedSender<SyncCommand>,

    /// Event channel for observability
    event_tx: Option<mpsc::UnboundedSender<HotLayerEvent>>,
}
```

---

## Layer 2: Cold Storage Layer

### Purpose

The Cold Storage Layer provides durable, persistent storage for:

- Historical embeddings and chunks
- Complete document archives
- RAPTOR tree structures
- Full-text search indices

### Technology Evaluation

| Solution                    | Vector Search | KV Store   | WAL  | Pure Rust  | Memory | Decision     |
| --------------------------- | ------------- | ---------- | ---- | ---------- | ------ | ------------ |
| **LanceDB**                 | Native        | Integrated | Yes  | No (Arrow) | Medium | **PRIMARY**  |
| **Qdrant Embedded**         | Native        | Basic      | Yes  | No (C++)   | High   | **FALLBACK** |
| **Pure Rust (Sled + HNSW)** | HNSW lib      | Sled       | Sled | Yes        | Low    | **PORTABLE** |
| RocksDB + Faiss             | Faiss         | RocksDB    | Yes  | No         | High   | Rejected     |

**Recommendation: Tiered approach**

1. LanceDB for production (best vector perf)
2. Pure Rust (Sled + HNSW) for embedded/portable
3. Qdrant embedded as optional feature

### Data Flow Diagram

```
+==============================================================================+
|                         COLD STORAGE DATA FLOW                               |
+==============================================================================+

   Write Path (from Hot Layer)                    Read Path (Promotion)
   ---------------------------                    ---------------------

   [1] Demote Command                            [1] Promotion Request
         |                                              |
         v                                              v
   +----------------+                            +----------------+
   | Serialize      |                            | Check Vector   |
   | Entry to       |                            | Index          |
   | Storage Format |                            +-------+--------+
   +-------+--------+                                    |
           |                                             v
           v                                      +----------------+
   +----------------+                            | Load Embedding |
   | Write to WAL   |                            | from Lance/    |
   | (Append-only)  |                            | Vector Store   |
   +-------+--------+                            +-------+--------+
           |                                             |
           v                                             v
   +----------------+                            +----------------+
   | Update Vector  |                            | Load Metadata  |
   | Index (Lance/  |                            | from Sled KV   |
   | HNSW)          |                            +-------+--------+
   +-------+--------+                                    |
           |                                             v
           v                                      +----------------+
   +----------------+                            | Reconstruct    |
   | Update Sled    |                            | HotEntry       |
   | Metadata KV    |                            +-------+--------+
   +-------+--------+                                    |
           |                                             v
           v                                      +----------------+
   +----------------+                            | Return to      |
   | Update BM25    |                            | Hot Layer      |
   | Index (Tantivy)|                            +----------------+
   +----------------+


                    WAL OPERATIONS
                    --------------

   +-------------------+     +-------------------+     +-------------------+
   |  WAL Append       |     |  WAL Checkpoint   |     |  WAL Recovery     |
   +-------------------+     +-------------------+     +-------------------+
   |                   |     |                   |     |                   |
   | For each write:   |     | Periodic (5min):  |     | On startup:       |
   | 1. Serialize op   |     | 1. Flush WAL      |     | 1. Read WAL       |
   | 2. CRC32 checksum |     | 2. Apply to store |     | 2. Verify CRCs    |
   | 3. Append to file |     | 3. Truncate WAL   |     | 3. Replay ops     |
   | 4. fsync()        |     | 4. Update marker  |     | 4. Mark recovered |
   |                   |     |                   |     |                   |
   +-------------------+     +-------------------+     +-------------------+


                    VECTOR INDEX STRUCTURE
                    ----------------------

   +==============================================================================+
   |                            VECTOR INDEX (HNSW)                               |
   +==============================================================================+
   |                                                                              |
   |   Level 3:    [N1] -------- [N7] -------- [N15]                             |
   |                 |             |              |                               |
   |   Level 2:  [N1]--[N3]--[N7]--[N9]--[N15]                                   |
   |               |     |     |     |      |                                    |
   |   Level 1:  [N1][N2][N3][N5][N7][N8][N9][N12][N15][N18]                     |
   |               |  |  |  |  |  |  |   |    |    |                            |
   |   Level 0:  [N1][N2][N3][N4][N5][N6][N7][N8][N9]...[Nn] (All nodes)        |
   |                                                                              |
   |   Parameters:                                                                |
   |   - ef_construction: 128 (build quality)                                    |
   |   - M: 16 (connections per node)                                            |
   |   - ef_search: 64 (query quality)                                           |
   +==============================================================================+
```

### Type Definitions

```rust
//! Cold Storage Layer Types for reasonkit-mem
//!
//! Provides durable, persistent storage with vector search capabilities.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

/// Configuration for the Cold Storage Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdLayerConfig {
    /// Base directory for all cold storage files
    pub data_dir: PathBuf,

    /// Vector store backend selection
    pub vector_backend: VectorBackend,

    /// Embedding dimension (must match hot layer)
    pub embedding_dimension: usize,

    /// HNSW index parameters (for pure Rust mode)
    pub hnsw_config: HnswConfig,

    /// Sled configuration
    pub sled_config: SledConfig,

    /// WAL configuration
    pub wal_config: WalConfig,

    /// BM25/Tantivy configuration
    pub bm25_config: Bm25Config,

    /// Enable mmap for large indices
    pub use_mmap: bool,

    /// Compression for stored data
    pub compression: CompressionType,
}

impl Default for ColdLayerConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./reasonkit_data"),
            vector_backend: VectorBackend::PureRust,
            embedding_dimension: 1536,
            hnsw_config: HnswConfig::default(),
            sled_config: SledConfig::default(),
            wal_config: WalConfig::default(),
            bm25_config: Bm25Config::default(),
            use_mmap: true,
            compression: CompressionType::Lz4,
        }
    }
}

/// Vector store backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorBackend {
    /// Pure Rust HNSW implementation (portable, no dependencies)
    PureRust,
    /// LanceDB embedded (best performance, requires Arrow)
    LanceDb,
    /// Qdrant embedded (feature-rich, higher memory)
    QdrantEmbedded,
}

/// HNSW index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of connections per node at each layer
    pub m: usize,
    /// Number of connections at layer 0
    pub m0: usize,
    /// ef parameter during construction
    pub ef_construction: usize,
    /// ef parameter during search
    pub ef_search: usize,
    /// Maximum elements in the index
    pub max_elements: usize,
    /// Enable index compaction
    pub enable_compaction: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 128,
            ef_search: 64,
            max_elements: 10_000_000,
            enable_compaction: true,
        }
    }
}

/// Sled embedded database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SledConfig {
    /// Cache size in bytes
    pub cache_capacity_bytes: u64,
    /// Flush interval
    pub flush_every_ms: Option<u64>,
    /// Use compression
    pub use_compression: bool,
    /// Mode: HighThroughput or LowSpace
    pub mode: SledMode,
}

impl Default for SledConfig {
    fn default() -> Self {
        Self {
            cache_capacity_bytes: 128 * 1024 * 1024, // 128 MB
            flush_every_ms: Some(1000),
            use_compression: true,
            mode: SledMode::HighThroughput,
        }
    }
}

/// Sled operation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SledMode {
    HighThroughput,
    LowSpace,
}

/// Write-Ahead Log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfig {
    /// WAL file size before rotation (bytes)
    pub max_segment_size: u64,
    /// Sync mode for durability
    pub sync_mode: WalSyncMode,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
    /// Maximum WAL size before forced checkpoint
    pub max_wal_size: u64,
    /// Enable CRC32 checksums
    pub enable_checksums: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_segment_size: 64 * 1024 * 1024, // 64 MB
            sync_mode: WalSyncMode::EverySecond,
            checkpoint_interval: Duration::from_secs(300), // 5 minutes
            max_wal_size: 512 * 1024 * 1024, // 512 MB
            enable_checksums: true,
        }
    }
}

/// WAL sync mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalSyncMode {
    /// Sync after every write (slowest, safest)
    EveryWrite,
    /// Sync every second
    EverySecond,
    /// Sync only on checkpoint
    OnCheckpoint,
    /// No sync (fastest, risk of data loss)
    NoSync,
}

/// BM25/Tantivy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Config {
    /// Heap size for indexing (bytes)
    pub index_heap_bytes: usize,
    /// Number of indexing threads
    pub num_threads: usize,
    /// BM25 k1 parameter
    pub bm25_k1: f32,
    /// BM25 b parameter
    pub bm25_b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            index_heap_bytes: 50 * 1024 * 1024, // 50 MB
            num_threads: 4,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

/// Compression type for stored data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Lz4,
    Zstd,
    Snappy,
}

/// WAL operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    /// Insert or update an entry
    Upsert(ColdEntry),
    /// Delete an entry
    Delete { id: Uuid },
    /// Batch upsert
    BatchUpsert(Vec<ColdEntry>),
    /// Checkpoint marker
    Checkpoint { timestamp: u64 },
}

/// Cold storage entry (serialized format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdEntry {
    /// Entry UUID
    pub id: Uuid,

    /// Entry type
    pub entry_type: ColdEntryType,

    /// Parent document ID
    pub document_id: Option<Uuid>,

    /// Session ID (if session-isolated)
    pub session_id: Option<Uuid>,

    /// Dense embedding vector
    pub embedding: Vec<f32>,

    /// Text content
    pub text: String,

    /// Serialized metadata (JSON)
    pub metadata: Vec<u8>,

    /// Original creation timestamp (Unix seconds)
    pub created_at: u64,

    /// Last modification timestamp
    pub updated_at: u64,

    /// Access statistics
    pub access_stats: AccessStats,
}

/// Entry types in cold storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColdEntryType {
    Embedding,
    Chunk,
    DocumentMeta,
    RaptorNode,
    RaptorSummary,
}

/// Access statistics for entries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessStats {
    /// Total access count
    pub access_count: u64,
    /// Last access timestamp
    pub last_accessed: Option<u64>,
    /// Number of promotions to hot layer
    pub promotion_count: u32,
}

/// WAL segment file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalSegmentHeader {
    /// Magic bytes: "RKMW" (ReasonKit Memory WAL)
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Segment number
    pub segment_id: u64,
    /// Previous segment CRC (for chain verification)
    pub prev_crc: u32,
    /// Timestamp of segment creation
    pub created_at: u64,
}

/// Single WAL record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    /// Record sequence number
    pub seq: u64,
    /// Record type
    pub op: WalOperation,
    /// CRC32 checksum
    pub crc: u32,
    /// Record size in bytes
    pub size: u32,
}

/// Cold storage query result
#[derive(Debug, Clone)]
pub struct ColdSearchResult {
    /// Entry ID
    pub id: Uuid,
    /// Similarity score
    pub score: f32,
    /// Entry type
    pub entry_type: ColdEntryType,
    /// Distance from query (for HNSW)
    pub distance: f32,
}

/// Cold storage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColdStorageStats {
    /// Total entries
    pub total_entries: u64,
    /// Total embeddings
    pub embedding_count: u64,
    /// Total chunks
    pub chunk_count: u64,
    /// Storage size in bytes
    pub storage_bytes: u64,
    /// Vector index size
    pub vector_index_bytes: u64,
    /// BM25 index size
    pub bm25_index_bytes: u64,
    /// WAL size
    pub wal_bytes: u64,
    /// Last checkpoint time
    pub last_checkpoint: Option<u64>,
}

/// The Cold Storage Layer implementation trait
#[async_trait::async_trait]
pub trait ColdStorageBackend: Send + Sync {
    /// Store an entry
    async fn store(&self, entry: ColdEntry) -> crate::Result<()>;

    /// Store multiple entries
    async fn store_batch(&self, entries: Vec<ColdEntry>) -> crate::Result<()>;

    /// Get an entry by ID
    async fn get(&self, id: &Uuid) -> crate::Result<Option<ColdEntry>>;

    /// Delete an entry
    async fn delete(&self, id: &Uuid) -> crate::Result<()>;

    /// Vector similarity search
    async fn search_vector(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<ColdSearchResult>>;

    /// BM25 text search
    async fn search_text(
        &self,
        query: &str,
        top_k: usize,
    ) -> crate::Result<Vec<ColdSearchResult>>;

    /// Hybrid search (vector + text)
    async fn search_hybrid(
        &self,
        query_vector: &[f32],
        query_text: &str,
        top_k: usize,
        alpha: f32,
    ) -> crate::Result<Vec<ColdSearchResult>>;

    /// Get storage statistics
    fn stats(&self) -> ColdStorageStats;

    /// Force WAL checkpoint
    async fn checkpoint(&self) -> crate::Result<()>;

    /// Compact storage (merge segments, rebuild indices)
    async fn compact(&self) -> crate::Result<()>;
}
```

---

## Layer Transition Logic

### Promotion Criteria (Cold -> Hot)

```
+==============================================================================+
|                         PROMOTION DECISION TREE                              |
+==============================================================================+

   [Access Request for ID]
            |
            v
   +------------------+
   | Is ID in Hot?    |
   +--------+---------+
            |
      +-----+-----+
      |           |
     [Yes]       [No]
      |           |
      v           v
   [Return    +------------------+
    Direct]   | Is ID in Cold?   |
              +--------+---------+
                       |
                 +-----+-----+
                 |           |
                [Yes]       [No]
                 |           |
                 v           v
        +------------------+ [Return
        | Check Promotion  |  None]
        | Criteria         |
        +--------+---------+
                 |
    +------------+------------+
    |            |            |
   [A]          [B]          [C]
    |            |            |
    v            v            v
 [Recent     [Frequent    [Related
  Access]     Access]      Context]
    |            |            |
    +-----+------+-----+------+
          |            |
         [OR]         [OR]
          |            |
          v            v
   +------------------+   +------------------+
   | YES: Promote to  |   | NO: Return from  |
   | Hot Layer        |   | Cold (no cache)  |
   +------------------+   +------------------+

PROMOTION CRITERIA:
-------------------
A. Recent Access: accessed within last 5 minutes
B. Frequent Access: access_count > 3 in last hour
C. Related Context:
   - Same session_id as current
   - Same document_id as recently accessed
   - Neighbor in RAPTOR tree of hot node
```

### Demotion Criteria (Hot -> Cold)

```
+==============================================================================+
|                         DEMOTION DECISION TREE                               |
+==============================================================================+

   [Background Timer: Every 100ms]
            |
            v
   +------------------+
   | Check TTL Queue  |
   +--------+---------+
            |
            v
   +------------------+     +------------------+
   | Expired Entries? | --> | Demote All       |
   +--------+---------+     | Expired to Cold  |
            |               +------------------+
           [No]
            |
            v
   +------------------+
   | Memory > 80%?    |
   +--------+---------+
            |
      +-----+-----+
      |           |
     [Yes]       [No]
      |           |
      v           v
   +------------------+   [Sleep until
   | LRU Eviction     |    next check]
   +--------+---------+
            |
            v
   +------------------+
   | Select oldest N  |
   | entries by LRU   |
   +--------+---------+
            |
            v
   +------------------+
   | Batch demote to  |
   | Cold Storage     |
   +------------------+

DEMOTION PRIORITIES (Evict First):
-----------------------------------
1. TTL Expired entries (immediate)
2. Entries with access_count == 1 (single-access)
3. Entries from closed sessions
4. Oldest by last_accessed timestamp (LRU)
5. Largest entries by size_bytes (size pressure)

DEMOTION EXCLUSIONS (Never Evict):
----------------------------------
- Entries accessed in last 10 seconds
- Entries with pinned flag
- Active session context entries
```

### Background Sync Worker

```rust
//! Sync Worker Types for reasonkit-mem
//!
//! Background worker handling layer transitions and maintenance.

use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

/// Sync worker configuration
#[derive(Debug, Clone)]
pub struct SyncWorkerConfig {
    /// Batch size for demotion operations
    pub demotion_batch_size: usize,

    /// Maximum pending demotions before backpressure
    pub max_pending_demotions: usize,

    /// Promotion concurrency limit
    pub promotion_concurrency: usize,

    /// WAL compaction threshold (bytes)
    pub wal_compaction_threshold: u64,

    /// Idle timeout before worker sleeps
    pub idle_timeout: Duration,

    /// Metrics emission interval
    pub metrics_interval: Duration,
}

impl Default for SyncWorkerConfig {
    fn default() -> Self {
        Self {
            demotion_batch_size: 100,
            max_pending_demotions: 10_000,
            promotion_concurrency: 4,
            wal_compaction_threshold: 256 * 1024 * 1024, // 256 MB
            idle_timeout: Duration::from_millis(100),
            metrics_interval: Duration::from_secs(5),
        }
    }
}

/// Sync worker state
pub struct SyncWorker {
    /// Configuration
    config: SyncWorkerConfig,

    /// Command receiver from hot layer
    command_rx: mpsc::UnboundedReceiver<SyncCommand>,

    /// Reference to cold storage
    cold_storage: Arc<dyn ColdStorageBackend>,

    /// Promotion response channel
    promotion_tx: mpsc::Sender<PromotionResult>,

    /// Shutdown signal
    shutdown_rx: oneshot::Receiver<()>,

    /// Pending demotions buffer
    pending_demotions: Vec<HotEntry>,

    /// Worker metrics
    metrics: SyncWorkerMetrics,
}

/// Result of a promotion request
pub struct PromotionResult {
    /// Requested ID
    pub id: Uuid,
    /// Success or failure
    pub result: crate::Result<Option<HotEntry>>,
    /// Latency in microseconds
    pub latency_us: u64,
}

/// Sync worker metrics
#[derive(Debug, Default)]
pub struct SyncWorkerMetrics {
    /// Total demotions processed
    pub demotions_processed: AtomicU64,
    /// Total promotions processed
    pub promotions_processed: AtomicU64,
    /// WAL bytes written
    pub wal_bytes_written: AtomicU64,
    /// Compactions performed
    pub compactions: AtomicU64,
    /// Current queue depth
    pub queue_depth: AtomicUsize,
}

impl SyncWorker {
    /// Run the sync worker loop
    pub async fn run(mut self) {
        let mut batch_timer = tokio::time::interval(Duration::from_millis(50));
        let mut compaction_timer = tokio::time::interval(Duration::from_secs(60));

        loop {
            tokio::select! {
                // Handle incoming commands
                Some(cmd) = self.command_rx.recv() => {
                    match cmd {
                        SyncCommand::Demote(entry) => {
                            self.pending_demotions.push(entry);
                            if self.pending_demotions.len() >= self.config.demotion_batch_size {
                                self.flush_demotions().await;
                            }
                        }
                        SyncCommand::BatchDemote(entries) => {
                            self.pending_demotions.extend(entries);
                            self.flush_demotions().await;
                        }
                        SyncCommand::PromoteRequest(id) => {
                            self.handle_promotion(id).await;
                        }
                        SyncCommand::Flush => {
                            self.flush_demotions().await;
                        }
                        SyncCommand::CompactWal => {
                            self.compact_wal().await;
                        }
                        SyncCommand::Shutdown => {
                            self.flush_demotions().await;
                            self.compact_wal().await;
                            return;
                        }
                    }
                }

                // Batch timer for pending demotions
                _ = batch_timer.tick() => {
                    if !self.pending_demotions.is_empty() {
                        self.flush_demotions().await;
                    }
                }

                // Periodic compaction check
                _ = compaction_timer.tick() => {
                    self.check_compaction().await;
                }

                // Shutdown signal
                _ = &mut self.shutdown_rx => {
                    self.flush_demotions().await;
                    return;
                }
            }
        }
    }

    async fn flush_demotions(&mut self) {
        if self.pending_demotions.is_empty() {
            return;
        }

        let entries: Vec<ColdEntry> = self.pending_demotions
            .drain(..)
            .map(|hot| hot.into())
            .collect();

        let count = entries.len();

        if let Err(e) = self.cold_storage.store_batch(entries).await {
            tracing::error!("Failed to demote batch: {}", e);
        } else {
            self.metrics.demotions_processed
                .fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    async fn handle_promotion(&self, id: Uuid) {
        let start = std::time::Instant::now();

        let result = self.cold_storage.get(&id).await
            .map(|opt| opt.map(|cold| cold.into()));

        let latency_us = start.elapsed().as_micros() as u64;

        let _ = self.promotion_tx.send(PromotionResult {
            id,
            result,
            latency_us,
        }).await;

        self.metrics.promotions_processed.fetch_add(1, Ordering::Relaxed);
    }

    async fn check_compaction(&self) {
        let stats = self.cold_storage.stats();
        if stats.wal_bytes >= self.config.wal_compaction_threshold {
            self.compact_wal().await;
        }
    }

    async fn compact_wal(&self) {
        if let Err(e) = self.cold_storage.compact().await {
            tracing::error!("WAL compaction failed: {}", e);
        } else {
            self.metrics.compactions.fetch_add(1, Ordering::Relaxed);
        }
    }
}
```

---

## Configuration Schema

```rust
//! Unified Configuration for Dual-Layer Storage

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Complete storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLayerStorageConfig {
    /// Hot layer configuration
    pub hot: HotLayerConfig,

    /// Cold layer configuration
    pub cold: ColdLayerConfig,

    /// Sync worker configuration
    pub sync: SyncWorkerConfig,

    /// Global settings
    pub global: GlobalStorageConfig,
}

/// Global storage settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStorageConfig {
    /// Enable telemetry/metrics
    pub enable_metrics: bool,

    /// Metrics export interval
    pub metrics_interval: Duration,

    /// Enable debug logging
    pub debug_logging: bool,

    /// Recovery mode on startup
    pub recovery_mode: RecoveryMode,

    /// Read-only mode
    pub read_only: bool,
}

/// Recovery mode options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryMode {
    /// Automatic recovery from WAL
    Automatic,
    /// Skip recovery, start fresh
    Skip,
    /// Fail if recovery needed
    Strict,
}

impl Default for DualLayerStorageConfig {
    fn default() -> Self {
        Self {
            hot: HotLayerConfig::default(),
            cold: ColdLayerConfig::default(),
            sync: SyncWorkerConfig::default(),
            global: GlobalStorageConfig {
                enable_metrics: true,
                metrics_interval: Duration::from_secs(10),
                debug_logging: false,
                recovery_mode: RecoveryMode::Automatic,
                read_only: false,
            },
        }
    }
}

impl DualLayerStorageConfig {
    /// Load from TOML file
    pub fn from_file(path: &PathBuf) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content)
            .map_err(|e| crate::MemError::Config(e.to_string()))
    }

    /// Preset: Development (fast, less durable)
    pub fn development() -> Self {
        Self {
            hot: HotLayerConfig {
                max_memory_bytes: 64 * 1024 * 1024, // 64 MB
                default_ttl: Duration::from_secs(60 * 5), // 5 minutes
                ..Default::default()
            },
            cold: ColdLayerConfig {
                vector_backend: VectorBackend::PureRust,
                wal_config: WalConfig {
                    sync_mode: WalSyncMode::NoSync,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Preset: Production (balanced durability and performance)
    pub fn production() -> Self {
        Self {
            hot: HotLayerConfig {
                max_memory_bytes: 512 * 1024 * 1024, // 512 MB
                write_through: true,
                ..Default::default()
            },
            cold: ColdLayerConfig {
                vector_backend: VectorBackend::LanceDb,
                wal_config: WalConfig {
                    sync_mode: WalSyncMode::EverySecond,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Preset: Embedded (minimal dependencies)
    pub fn embedded() -> Self {
        Self {
            cold: ColdLayerConfig {
                vector_backend: VectorBackend::PureRust,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Preset: High Performance (maximum speed)
    pub fn high_performance() -> Self {
        Self {
            hot: HotLayerConfig {
                max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
                ttl_check_interval: Duration::from_millis(50),
                ..Default::default()
            },
            cold: ColdLayerConfig {
                vector_backend: VectorBackend::LanceDb,
                wal_config: WalConfig {
                    sync_mode: WalSyncMode::OnCheckpoint,
                    ..Default::default()
                },
                use_mmap: true,
                ..Default::default()
            },
            sync: SyncWorkerConfig {
                demotion_batch_size: 500,
                promotion_concurrency: 8,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}
```

### Configuration File Example (TOML)

```toml
# reasonkit-mem storage configuration
# Place at: ~/.config/reasonkit/storage.toml

[hot]
max_memory_bytes = 268435456  # 256 MB
default_ttl_secs = 1800       # 30 minutes
max_entries = 100000
eviction_batch_size = 1000
ttl_check_interval_ms = 100
memory_pressure_threshold = 0.8
write_through = true
session_isolation = true

[cold]
data_dir = "/var/lib/reasonkit/storage"
vector_backend = "PureRust"  # or "LanceDb", "QdrantEmbedded"
embedding_dimension = 1536
use_mmap = true
compression = "Lz4"

[cold.hnsw]
m = 16
m0 = 32
ef_construction = 128
ef_search = 64
max_elements = 10000000
enable_compaction = true

[cold.sled]
cache_capacity_bytes = 134217728  # 128 MB
flush_every_ms = 1000
use_compression = true
mode = "HighThroughput"

[cold.wal]
max_segment_size = 67108864   # 64 MB
sync_mode = "EverySecond"
checkpoint_interval_secs = 300
max_wal_size = 536870912      # 512 MB
enable_checksums = true

[cold.bm25]
index_heap_bytes = 52428800   # 50 MB
num_threads = 4
bm25_k1 = 1.2
bm25_b = 0.75

[sync]
demotion_batch_size = 100
max_pending_demotions = 10000
promotion_concurrency = 4
wal_compaction_threshold = 268435456  # 256 MB
idle_timeout_ms = 100
metrics_interval_secs = 5

[global]
enable_metrics = true
metrics_interval_secs = 10
debug_logging = false
recovery_mode = "Automatic"
read_only = false
```

---

## Performance Estimates

### Hot Layer Performance

| Operation                     | Target Latency | Expected Throughput |
| ----------------------------- | -------------- | ------------------- |
| Get (hit)                     | < 100 us       | 1M+ ops/sec         |
| Get (miss)                    | < 200 us       | 500K+ ops/sec       |
| Put                           | < 200 us       | 500K+ ops/sec       |
| TTL check (per 1K entries)    | < 1 ms         | N/A                 |
| LRU eviction (per 1K entries) | < 5 ms         | N/A                 |

### Cold Layer Performance

| Operation                           | Target Latency | Expected Throughput |
| ----------------------------------- | -------------- | ------------------- |
| Get by ID                           | < 1 ms         | 100K+ ops/sec       |
| Vector search (top-10, 1M vectors)  | < 10 ms        | 1K+ qps             |
| Vector search (top-100, 1M vectors) | < 50 ms        | 200+ qps            |
| BM25 search (top-10)                | < 5 ms         | 2K+ qps             |
| Hybrid search                       | < 20 ms        | 500+ qps            |
| WAL append                          | < 100 us       | 100K+ ops/sec       |
| Batch insert (100 entries)          | < 50 ms        | 2K entries/sec      |

### Layer Transition Performance

| Operation                    | Target Latency | Notes                    |
| ---------------------------- | -------------- | ------------------------ |
| Promotion (Cold -> Hot)      | < 2 ms         | Includes deserialization |
| Demotion (Hot -> Cold)       | < 1 ms         | Async, non-blocking      |
| Batch demotion (100 entries) | < 10 ms        | Background worker        |
| WAL checkpoint               | < 500 ms       | Every 5 minutes          |
| Full compaction              | < 30 sec       | Rare, incremental        |

### Memory Footprint

| Component               | Size                | Notes                      |
| ----------------------- | ------------------- | -------------------------- |
| Hot Layer (per entry)   | ~2 KB + embedding   | Embedding = dim \* 4 bytes |
| DashMap overhead        | ~64 bytes/entry     | Bucket + metadata          |
| Cold entry (on disk)    | ~1.5 KB + embedding | Compressed                 |
| HNSW index (per vector) | ~200 bytes          | M=16, 3 layers             |
| Tantivy index           | ~0.5x text size     | Compressed inverted index  |

### Scaling Characteristics

```
+==============================================================================+
|                         SCALING CHARACTERISTICS                              |
+==============================================================================+

   Hot Layer Scaling (Memory-Bound)
   --------------------------------

   Entries:     10K    100K    1M     10M
   Memory:     40MB   400MB   4GB    40GB
   Get p99:   0.1ms   0.1ms  0.2ms   0.5ms

   Note: Linear memory growth, near-constant access time


   Cold Layer Vector Search (HNSW)
   --------------------------------

   Vectors:     10K   100K    1M     10M    100M
   Index:      2MB    20MB  200MB    2GB    20GB
   Search:     1ms    2ms    5ms   15ms    50ms

   Note: Logarithmic search time growth


   Cold Layer Text Search (Tantivy)
   ---------------------------------

   Docs:       10K   100K    1M     10M
   Index:      5MB    50MB  500MB    5GB
   Search:     1ms    2ms    3ms    10ms

   Note: Near-constant search time with proper indexing
```

---

## Directory Structure

```
reasonkit_data/
├── hot/
│   └── metrics/                 # Hot layer telemetry
│       └── metrics.json
├── cold/
│   ├── vectors/                 # Vector index files
│   │   ├── hnsw.idx             # HNSW graph
│   │   ├── vectors.bin          # Raw vectors (mmap)
│   │   └── metadata.json        # Index metadata
│   ├── kv/                      # Sled KV store
│   │   ├── conf
│   │   ├── db
│   │   └── snap.*
│   ├── bm25/                    # Tantivy index
│   │   ├── meta.json
│   │   └── segments/
│   └── wal/                     # Write-ahead log
│       ├── segment_0001.wal
│       ├── segment_0002.wal
│       └── checkpoint.json
├── config/
│   └── storage.toml             # Configuration
└── backup/                      # Backup snapshots
    └── 2026-01-01_120000/
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

1. Implement `HotMemoryLayer` with DashMap
2. Implement TTL and LRU eviction
3. Add metrics and observability
4. Unit tests for hot layer

### Phase 2: Cold Storage (Week 3-4)

1. Implement WAL with append-only log
2. Implement Sled KV backend
3. Implement pure-Rust HNSW index
4. Integrate with existing Tantivy BM25

### Phase 3: Sync Worker (Week 5)

1. Implement background sync worker
2. Implement promotion/demotion logic
3. Add WAL compaction
4. Integration tests

### Phase 4: Integration (Week 6)

1. Integrate with `Storage` trait
2. Update `HybridRetriever` to use dual-layer
3. Add configuration presets
4. Performance benchmarks

### Phase 5: Optional Backends (Week 7-8)

1. LanceDB integration (feature flag)
2. Qdrant embedded integration (feature flag)
3. Cross-backend tests
4. Documentation

---

## Appendix: Pure Rust HNSW Implementation

For the portable embedded mode, we implement HNSW in pure Rust:

```rust
//! Pure Rust HNSW implementation
//! Based on: Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor search"

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use uuid::Uuid;

/// HNSW graph node
#[derive(Debug, Clone)]
pub struct HnswNode {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub connections: Vec<Vec<Uuid>>, // connections[layer] = neighbors
    pub max_layer: usize,
}

/// Distance-ID pair for priority queue
#[derive(Debug, Clone)]
struct DistanceId {
    distance: f32,
    id: Uuid,
}

impl Eq for DistanceId {}
impl PartialEq for DistanceId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Ord for DistanceId {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DistanceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Pure Rust HNSW index
pub struct HnswIndex {
    nodes: HashMap<Uuid, HnswNode>,
    entry_point: Option<Uuid>,
    config: HnswConfig,
    max_layer: usize,
    rng: rand::rngs::StdRng,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        use rand::SeedableRng;
        Self {
            nodes: HashMap::new(),
            entry_point: None,
            config,
            max_layer: 0,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, id: Uuid, vector: Vec<f32>) -> crate::Result<()> {
        use rand::Rng;

        // Calculate random layer
        let ml = 1.0 / (self.config.m as f32).ln();
        let layer = (-self.rng.gen::<f32>().ln() * ml).floor() as usize;

        let node = HnswNode {
            id,
            vector: vector.clone(),
            connections: vec![Vec::new(); layer + 1],
            max_layer: layer,
        };

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = layer;
            self.nodes.insert(id, node);
            return Ok(());
        }

        let entry = self.entry_point.unwrap();

        // Greedy search from top to node's layer
        let mut curr = entry;
        for lc in (layer + 1..=self.max_layer).rev() {
            curr = self.search_layer_single(&vector, curr, lc);
        }

        // Search and connect at each layer
        let mut ep_set = HashSet::new();
        ep_set.insert(curr);

        for lc in (0..=layer.min(self.max_layer)).rev() {
            let W = self.search_layer(&vector, &ep_set, self.config.ef_construction, lc);
            let neighbors = self.select_neighbors(&vector, &W, self.config.m);

            // Add bidirectional connections
            let mut node_connections = self.nodes.get(&id)
                .map(|n| n.connections.clone())
                .unwrap_or_else(|| vec![Vec::new(); layer + 1]);

            if lc < node_connections.len() {
                node_connections[lc] = neighbors.iter().map(|d| d.id).collect();
            }

            // Update node
            if let Some(n) = self.nodes.get_mut(&id) {
                n.connections = node_connections.clone();
            }

            // Add reverse connections
            for neighbor in &neighbors {
                if let Some(n) = self.nodes.get_mut(&neighbor.id) {
                    if lc < n.connections.len() && !n.connections[lc].contains(&id) {
                        n.connections[lc].push(id);
                        // Prune if over limit
                        if n.connections[lc].len() > self.config.m {
                            let n_vec = n.vector.clone();
                            let conn: Vec<_> = n.connections[lc].iter()
                                .filter_map(|&cid| self.nodes.get(&cid).map(|cn|
                                    DistanceId { distance: cosine_distance(&n_vec, &cn.vector), id: cid }
                                ))
                                .collect();
                            let pruned = self.select_neighbors(&n_vec, &conn, self.config.m);
                            n.connections[lc] = pruned.iter().map(|d| d.id).collect();
                        }
                    }
                }
            }

            ep_set = neighbors.iter().map(|d| d.id).collect();
        }

        // Insert node
        self.nodes.insert(id, HnswNode {
            id,
            vector,
            connections: self.nodes.get(&id)
                .map(|n| n.connections.clone())
                .unwrap_or_else(|| vec![Vec::new(); layer + 1]),
            max_layer: layer,
        });

        // Update entry point if needed
        if layer > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = layer;
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(Uuid, f32)> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let entry = self.entry_point.unwrap();
        let mut curr = entry;

        // Traverse from top to layer 1
        for lc in (1..=self.max_layer).rev() {
            curr = self.search_layer_single(query, curr, lc);
        }

        // Search layer 0 with ef_search
        let mut ep_set = HashSet::new();
        ep_set.insert(curr);
        let results = self.search_layer(query, &ep_set, self.config.ef_search, 0);

        results.into_iter()
            .take(k)
            .map(|d| (d.id, 1.0 - d.distance)) // Convert distance to similarity
            .collect()
    }

    fn search_layer_single(&self, query: &[f32], entry: Uuid, layer: usize) -> Uuid {
        let mut curr = entry;
        let mut curr_dist = self.nodes.get(&entry)
            .map(|n| cosine_distance(query, &n.vector))
            .unwrap_or(f32::MAX);

        loop {
            let mut changed = false;

            if let Some(node) = self.nodes.get(&curr) {
                if layer < node.connections.len() {
                    for &neighbor_id in &node.connections[layer] {
                        if let Some(neighbor) = self.nodes.get(&neighbor_id) {
                            let dist = cosine_distance(query, &neighbor.vector);
                            if dist < curr_dist {
                                curr_dist = dist;
                                curr = neighbor_id;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        curr
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &HashSet<Uuid>,
        ef: usize,
        layer: usize,
    ) -> Vec<DistanceId> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        for &ep in entry_points {
            visited.insert(ep);
            if let Some(node) = self.nodes.get(&ep) {
                let dist = cosine_distance(query, &node.vector);
                candidates.push(DistanceId { distance: dist, id: ep });
                results.push(DistanceId { distance: -dist, id: ep }); // Max-heap for worst
            }
        }

        while let Some(curr) = candidates.pop() {
            let worst = results.peek().map(|d| -d.distance).unwrap_or(f32::MAX);
            if curr.distance > worst {
                break;
            }

            if let Some(node) = self.nodes.get(&curr.id) {
                if layer < node.connections.len() {
                    for &neighbor_id in &node.connections[layer] {
                        if visited.insert(neighbor_id) {
                            if let Some(neighbor) = self.nodes.get(&neighbor_id) {
                                let dist = cosine_distance(query, &neighbor.vector);
                                if dist < worst || results.len() < ef {
                                    candidates.push(DistanceId { distance: dist, id: neighbor_id });
                                    results.push(DistanceId { distance: -dist, id: neighbor_id });
                                    if results.len() > ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        results.into_iter()
            .map(|d| DistanceId { distance: -d.distance, id: d.id })
            .collect()
    }

    fn select_neighbors(
        &self,
        query: &[f32],
        candidates: &[DistanceId],
        m: usize,
    ) -> Vec<DistanceId> {
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        sorted.truncate(m);
        sorted
    }
}

/// Cosine distance (1 - similarity)
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}
```

---

## References

1. **DashMap**: <https://docs.rs/dashmap>
2. **Sled**: <https://docs.rs/sled>
3. **LanceDB**: <https://lancedb.com/>
4. **HNSW Paper**: Malkov & Yashunin (2018) - arXiv:1603.09320
5. **Tantivy**: <https://docs.rs/tantivy>
6. **WAL Design**: SQLite WAL documentation
7. **Qdrant Embedded**: <https://qdrant.tech/documentation/>

---

_Document generated for ReasonKit-mem v0.2.0_
_Architecture designed for Debian 13+ compatibility_
