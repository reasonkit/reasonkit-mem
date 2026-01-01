# ReasonKit Memory Storage Architecture

> Technical documentation for the ReasonKit-mem storage engine.
> "Long-term memory, retrieval, and hybrid search infrastructure for ReasonKit."

**Version:** 1.0.0
**Last Updated:** 2025-01-01
**Maintainer:** ReasonKit Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Data Flow](#3-data-flow)
4. [Memory Layout](#4-memory-layout)
5. [Crash Recovery](#5-crash-recovery)
6. [Performance Characteristics](#6-performance-characteristics)
7. [Configuration Guide](#7-configuration-guide)

---

## 1. Overview

### 1.1 Dual-Layer Memory Architecture

ReasonKit Memory (ReasonKit-mem) implements a **dual-layer memory architecture** designed for high-performance retrieval-augmented generation (RAG) workloads. The system separates concerns between:

1. **Hot Layer (Fast Access)**: In-memory structures for frequently accessed data
2. **Cold Layer (Persistent)**: Disk-backed storage for durability and large datasets

```
            +-----------------+
            |   Application   |
            +-----------------+
                    |
            +-------v-------+
            |    Storage    |  <-- Unified API
            +-------+-------+
                    |
        +-----------+-----------+
        |                       |
+-------v-------+       +-------v-------+
|   Hot Memory  |       |  Cold Memory  |
|   (DashMap/   |       |   (Qdrant/    |
|    HashMap)   |       |    File)      |
+---------------+       +---------------+
```

### 1.2 Design Goals

| Goal                      | Implementation                    | Rationale                       |
| ------------------------- | --------------------------------- | ------------------------------- |
| **Sub-millisecond reads** | In-memory cache with LRU eviction | RAG queries require low latency |
| **Durability**            | WAL + Qdrant persistence          | No data loss on crash           |
| **Scalability**           | Qdrant vector DB backend          | Billions of embeddings          |
| **Zero-config start**     | File-based fallback               | Development simplicity          |
| **Hybrid search**         | Dense (Qdrant) + Sparse (Tantivy) | Optimal recall/precision        |

### 1.3 Design Constraints

- **Rust-first**: All core components implemented in Rust for memory safety and performance
- **Async-native**: Tokio-based async I/O throughout
- **Embedded-friendly**: Works without external services via file storage fallback
- **GDPR-compliant**: Data residency and access control built-in

### 1.4 Module Organization

```
reasonkit-mem/
+-- src/
    +-- storage/       # Storage backends (Qdrant, File, InMemory)
    +-- embedding/     # Dense vector embeddings (OpenAI, BGE-M3)
    +-- retrieval/     # Hybrid search, fusion, reranking
    +-- raptor/        # RAPTOR hierarchical tree structure
    +-- indexing/      # BM25/Tantivy sparse indexing
    +-- rag/           # RAG pipeline orchestration
```

---

## 2. Architecture Diagram

### 2.1 High-Level System Architecture

```
+=========================================================================+
|                           ReasonKit Memory                               |
+=========================================================================+
|                                                                          |
|   +------------------------------------------------------------------+  |
|   |                      HybridRetriever                              |  |
|   |  +------------+  +-------------+  +--------------+  +----------+  |  |
|   |  | Dense      |  | Sparse      |  | RAPTOR       |  | Reranker |  |  |
|   |  | (Qdrant)   |  | (Tantivy)   |  | Tree         |  | (Cross)  |  |  |
|   |  +-----+------+  +------+------+  +------+-------+  +-----+----+  |  |
|   |        |                |                |                |       |  |
|   |        +--------+-------+--------+-------+                |       |  |
|   |                 |                                         |       |  |
|   |          +------v------+                           +------v------+|  |
|   |          | RRF Fusion  |                           | CrossEncoder||  |
|   |          +-------------+                           +-------------+|  |
|   +------------------------------------------------------------------+  |
|                                    |                                     |
|   +--------------------------------v---------------------------------+   |
|   |                          Storage Layer                           |   |
|   +------------------------------------------------------------------+   |
|   |                                                                  |   |
|   |   +-------------------+   +-------------------+   +------------+ |   |
|   |   |    HotMemory      |   |    ColdMemory     |   | QueryCache | |   |
|   |   | +---------------+ |   | +---------------+ |   | +--------+ | |   |
|   |   | | EmbeddingCache| |   | | QdrantStorage | |   | | LRU    | | |   |
|   |   | | (HashMap)     | |   | | (Qdrant API)  | |   | | Cache  | | |   |
|   |   | +---------------+ |   | +---------------+ |   | +--------+ | |   |
|   |   | +---------------+ |   | +---------------+ |   +------------+ |   |
|   |   | | DocumentCache | |   | | FileStorage   | |                  |   |
|   |   | | (RwLock Map)  | |   | | (JSON/Binary) | |                  |   |
|   |   | +---------------+ |   | +---------------+ |                  |   |
|   |   +-------------------+   +-------------------+                  |   |
|   |                                    |                             |   |
|   |   +--------------------------------v----------------------------+|   |
|   |   |                    ConnectionPool                           ||   |
|   |   |  +----------------+  +----------------+  +----------------+  ||   |
|   |   |  | QdrantClient 1 |  | QdrantClient 2 |  | QdrantClient N |  ||   |
|   |   |  +----------------+  +----------------+  +----------------+  ||   |
|   |   +-----------------------------------------------------------------+|
|   +------------------------------------------------------------------+   |
|                                                                          |
|   +------------------------------------------------------------------+   |
|   |                       Indexing Layer                              |  |
|   |  +-------------------+    +-------------------+                   |  |
|   |  | BM25Index         |    | IndexManager      |                   |  |
|   |  | (Tantivy)         |    | (Multi-index)     |                   |  |
|   |  +-------------------+    +-------------------+                   |  |
|   +------------------------------------------------------------------+   |
|                                                                          |
+=========================================================================+
```

### 2.2 Storage Backend Selection

```
                    +------------------+
                    | Storage::new()   |
                    +--------+---------+
                             |
               +-------------+-------------+
               |                           |
        +------v------+             +------v------+
        | require_    |             | require_    |
        | qdrant=true |             | qdrant=false|
        +------+------+             +------+------+
               |                           |
        +------v------+             +------v------+
        | Check Qdrant|             | Use File    |
        | Health      |             | Storage     |
        +------+------+             +-------------+
               |
       +-------+-------+
       |               |
  +----v----+    +-----v-----+
  | Healthy |    | Unhealthy |
  +---------+    +-----------+
       |               |
  +----v----+    +-----v-----+
  | Qdrant  |    | Error:    |
  | Storage |    | Required  |
  +---------+    +-----------+
```

---

## 3. Data Flow

### 3.1 Write Path

```
Document/Embedding Write
         |
         v
+--------+--------+
| Validate Input  |
| (size, format)  |
+---------+-------+
          |
          v
+------------------+
| Access Control   |
| Check Permission |
+--------+---------+
         |
         v
+-----------------+        +------------------+
| Update Hot      |------->| EmbeddingCache   |
| Memory Cache    |        | (LRU, TTL)       |
+-----------------+        +------------------+
         |
         v
+-----------------+        +------------------+
| Persist to      |------->| Qdrant Upsert    |
| Cold Storage    |        | or File Write    |
+-----------------+        +------------------+
         |
         v
+-----------------+
| Index in        |
| Tantivy (BM25)  |
+-----------------+
         |
         v
+--------+--------+
| Return Success  |
+-----------------+
```

**Write Path Details:**

1. **Validation**: Check embedding dimension matches configured vector size
2. **Access Control**: Verify user has `ReadWrite` or `Admin` level
3. **Hot Cache Update**: Store in `EmbeddingCache` with TTL (default: 1 hour)
4. **Cold Persistence**:
   - Qdrant: `UpsertPoints` with wait=true for durability
   - File: Atomic write to `{chunk_id}.bin` / `{doc_id}.json`
5. **BM25 Indexing**: Add document chunks to Tantivy index

### 3.2 Read Path (Hot First, Then Cold)

```
Query Request
     |
     v
+----+----+
| Generate|
| Query   |
| Embed   |
+---------+
     |
     v
+----+-----+              +------------------+
| Check    |--HIT------->| Return Cached    |
| Query    |              | Results          |
| Cache    |              +------------------+
+----+-----+
     |
    MISS
     |
     v
+----+-----+              +------------------+
| Check    |--HIT------->| Return from      |
| Embedding|              | EmbeddingCache   |
| Cache    |              +------------------+
+----+-----+
     |
    MISS
     |
     v
+----+-----+
| Query    |
| Cold     |
| Storage  |
+----+-----+
     |
     v
+---------+    +-----------+    +---------+
| Qdrant  |    | File      |    | BM25    |
| Search  |    | Scan      |    | Search  |
+---------+    +-----------+    +---------+
     |              |               |
     +------+-------+---------------+
            |
            v
     +------+------+
     | RRF Fusion  |
     +------+------+
            |
            v
     +------+------+
     | Update      |
     | Caches      |
     +-------------+
            |
            v
     +------+------+
     | Return      |
     | Results     |
     +-------------+
```

**Read Path Latency Targets:**

| Cache Level     | Target Latency | Hit Rate (typical) |
| --------------- | -------------- | ------------------ |
| Query Cache     | < 0.1ms        | 20-40%             |
| Embedding Cache | < 0.5ms        | 60-80%             |
| Qdrant (HNSW)   | 1-5ms          | N/A                |
| File Scan       | 10-100ms       | N/A                |

### 3.3 Sync Path (Hot to Cold Migration)

The system uses a **write-through caching strategy** where writes go to both hot and cold storage immediately. However, cache eviction follows LRU with TTL:

```
+-------------------+
| Cache Entry       |
| created_at: T0    |
| ttl: 3600s        |
+-------------------+
        |
        | (access)
        v
+-------------------+
| Update            |
| last_accessed     |
| access_count++    |
+-------------------+
        |
        | (eviction check)
        v
+-------------------+
| If elapsed > TTL  |----> Remove from cache
| OR cache_size >   |
|    max_size       |
+-------------------+
        |
        | (LRU eviction)
        v
+-------------------+
| Remove oldest     |
| accessed entry    |
+-------------------+
```

---

## 4. Memory Layout

### 4.1 Document Entry Structure

```rust
pub struct Document {
    pub id: Uuid,                    // 16 bytes
    pub doc_type: DocumentType,      // 1 byte (enum)
    pub source: Source,              // ~200 bytes (variable)
    pub content: DocumentContent,    // Variable (raw text)
    pub metadata: Metadata,          // ~500 bytes (variable)
    pub processing: ProcessingStatus,// ~100 bytes
    pub chunks: Vec<Chunk>,          // N * ~300 bytes each
    pub created_at: DateTime<Utc>,   // 12 bytes
    pub updated_at: Option<DateTime>,// 12 bytes optional
}
```

**Memory Estimation per Document:**

- Base overhead: ~900 bytes
- Per chunk: ~300 bytes + text length + embedding_id
- Typical document (10 chunks, 500 words each): ~50KB

### 4.2 Chunk Entry Structure

```rust
pub struct Chunk {
    pub id: Uuid,                    // 16 bytes
    pub text: String,                // Variable
    pub index: usize,                // 8 bytes
    pub start_char: usize,           // 8 bytes
    pub end_char: usize,             // 8 bytes
    pub token_count: Option<usize>,  // 8 bytes optional
    pub section: Option<String>,     // Variable optional
    pub page: Option<usize>,         // 8 bytes optional
    pub embedding_ids: EmbeddingIds, // ~100 bytes
}
```

### 4.3 Embedding Storage Format

**In-Memory (EmbeddingCache):**

```rust
HashMap<Uuid, CachedEmbedding>

struct CachedEmbedding {
    embedding: Vec<f32>,      // vector_size * 4 bytes
    created_at: Instant,      // 8 bytes
}
```

**On-Disk (File Storage):**

```
File: embeddings/{chunk_id}.bin
Format: Raw little-endian f32 array
Size: vector_size * 4 bytes (e.g., 1536 * 4 = 6144 bytes for OpenAI)
```

**In Qdrant:**

```
Point {
    id: UUID string,
    vector: [f32; vector_size],
    payload: {
        "chunk_id": "uuid-string"
    }
}
```

### 4.4 Index Structures

**Tantivy BM25 Index Schema:**

```
Fields:
  - doc_id: STRING | STORED
  - chunk_id: STRING | STORED
  - text: TEXT | STORED
  - section: TEXT | STORED
```

**Qdrant Collection Config:**

```rust
VectorParams {
    size: vector_size,           // e.g., 1536
    distance: Distance::Cosine,
    quantization_config: ScalarQuantization {
        type: Int8,              // 4x compression
    },
}
```

---

## 5. Crash Recovery

### 5.1 Recovery Architecture

ReasonKit-mem relies on the underlying storage backends for durability:

```
                    +------------------+
                    | Application      |
                    | Crash/Restart    |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                                 |
   +--------v--------+               +--------v--------+
   | In-Memory       |               | Persistent      |
   | State Lost      |               | State Intact    |
   +--------+--------+               +--------+--------+
            |                                 |
            |                        +--------v--------+
            |                        | Qdrant/File     |
            |                        | Recovery        |
            |                        +--------+--------+
            |                                 |
   +--------v--------+               +--------v--------+
   | Rebuild Caches  |<--------------| Load from       |
   | On Demand       |               | Cold Storage    |
   +-----------------+               +-----------------+
```

### 5.2 Qdrant Durability

Qdrant provides durability through:

- **WAL (Write-Ahead Log)**: All writes are logged before acknowledgment
- **Snapshots**: Periodic point-in-time snapshots
- **Replication**: Optional multi-node replication

**Recovery Behavior:**

- On Qdrant restart, WAL replay restores uncommitted changes
- `wait: true` on upserts ensures write is durable before return

### 5.3 File Storage Durability

For file-based storage:

- **Atomic Writes**: Files written to temp location, then renamed
- **JSON Documents**: Human-readable, easy to inspect/recover
- **Binary Embeddings**: Compact format, checksummed

**File Layout:**

```
{data_path}/
+-- documents/
|   +-- {uuid1}.json
|   +-- {uuid2}.json
+-- embeddings/
    +-- {chunk_uuid1}.bin
    +-- {chunk_uuid2}.bin
```

### 5.4 Cache Reconstruction

After restart, caches are **lazily rebuilt** on demand:

```rust
// On cache miss, load from cold storage and cache
async fn get_embeddings(&self, chunk_id: &Uuid) -> Result<Option<Vec<f32>>> {
    // Check cache first
    if let Some(cached) = self.embedding_cache.get(chunk_id) {
        return Ok(Some(cached));
    }

    // Cache miss - load from Qdrant/File
    let embedding = self.cold_storage.get_embedding(chunk_id).await?;

    // Populate cache for next access
    if let Some(ref emb) = embedding {
        self.embedding_cache.put(*chunk_id, emb.clone());
    }

    Ok(embedding)
}
```

### 5.5 Checkpoint Strategy

| Component      | Strategy        | Frequency                     |
| -------------- | --------------- | ----------------------------- |
| EmbeddingCache | Write-through   | Every write                   |
| QueryCache     | Volatile        | N/A (rebuilt on demand)       |
| Tantivy Index  | Commit on batch | After each document batch     |
| Qdrant         | WAL + Snapshot  | Configurable (default: async) |
| File Storage   | Immediate sync  | Every write                   |

---

## 6. Performance Characteristics

### 6.1 Expected Latencies

| Operation                    | P50 Latency | P95 Latency | P99 Latency |
| ---------------------------- | ----------- | ----------- | ----------- |
| Cache Hit (Embedding)        | 0.05ms      | 0.1ms       | 0.2ms       |
| Cache Hit (Query)            | 0.02ms      | 0.05ms      | 0.1ms       |
| Qdrant Search (10K vectors)  | 1ms         | 3ms         | 5ms         |
| Qdrant Search (1M vectors)   | 2ms         | 5ms         | 10ms        |
| Qdrant Search (100M vectors) | 5ms         | 15ms        | 30ms        |
| BM25 Search (Tantivy)        | 0.5ms       | 2ms         | 5ms         |
| Hybrid Search + Fusion       | 3ms         | 8ms         | 15ms        |
| Hybrid + Rerank (top-10)     | 10ms        | 25ms        | 50ms        |
| File Storage Read            | 5ms         | 20ms        | 50ms        |
| Document Store               | 2ms         | 10ms        | 25ms        |
| Embedding Store              | 1ms         | 5ms         | 10ms        |

### 6.2 Memory Usage

**Base Memory Footprint:**

```
Component                    | Memory Usage
-----------------------------|------------------
EmbeddingCache (10K entries) | ~60MB (1536-dim)
EmbeddingCache (100K entries)| ~600MB (1536-dim)
QueryCache (1K entries)      | ~5MB
Tantivy Index (100K chunks)  | ~50MB RAM
Connection Pool (10 conn)    | ~10MB
```

**Per-Entry Memory:**

```
Embedding (1536-dim): 6.14 KB (1536 * 4 bytes + overhead)
Embedding (768-dim):  3.07 KB
Embedding (384-dim):  1.54 KB
Document Metadata:    ~1-2 KB
Chunk (avg 500 words): ~4 KB
```

### 6.3 Disk I/O Patterns

**Write Pattern:**

- Sequential writes for bulk ingestion
- Random writes for single document updates
- Qdrant: Append-only WAL, periodic segment compaction

**Read Pattern:**

- Random reads for document/embedding retrieval
- Sequential scans for BM25 full-text search
- Qdrant: HNSW graph traversal (random access)

**Recommended Disk Configuration:**

- SSD required for production workloads
- NVMe preferred for Qdrant segment storage
- Separate volumes for WAL and data segments

### 6.4 Scalability Characteristics

```
                         +------------------------+
Vectors/Latency Graph:   |                    /   |
                         |                  /     |
 Search Latency (ms)     |                /       |
     100 |               |              /         |
      50 |               |            /           |
      25 |               |          /   HNSW      |
      10 |               |        /______________/|
       5 |               |      /                 |
       2 |               |    /                   |
       1 |_______________| __/                    |
         10K  100K  1M  10M  100M  1B            |
                 Vector Count                     |
                         +------------------------+
```

| Scale  | Vectors | RAM (Qdrant) | Search Latency |
| ------ | ------- | ------------ | -------------- |
| Dev    | 10K     | 100MB        | < 2ms          |
| Small  | 100K    | 1GB          | < 5ms          |
| Medium | 1M      | 8GB          | < 10ms         |
| Large  | 10M     | 64GB         | < 20ms         |
| XLarge | 100M    | 512GB        | < 50ms         |

---

## 7. Configuration Guide

### 7.1 Key Parameters

#### Storage Configuration

```rust
pub struct EmbeddedStorageConfig {
    /// Path for storage data
    pub data_path: PathBuf,           // Default: ~/.local/share/reasonkit/storage

    /// Collection name for Qdrant
    pub collection_name: String,       // Default: "reasonkit_default"

    /// Vector dimension size
    pub vector_size: usize,            // Default: 1536 (OpenAI ada-002)

    /// Whether to require Qdrant server
    pub require_qdrant: bool,          // Default: false (file fallback)

    /// Qdrant server URL
    pub qdrant_url: String,            // Default: "http://localhost:6333"
}
```

#### Connection Pool Configuration

```rust
pub struct QdrantConnectionConfig {
    /// Maximum connections in pool
    pub max_connections: usize,        // Default: 10

    /// Connection timeout (seconds)
    pub connect_timeout_secs: u64,     // Default: 30

    /// Request timeout (seconds)
    pub request_timeout_secs: u64,     // Default: 60

    /// Health check interval (seconds)
    pub health_check_interval_secs: u64, // Default: 300 (5 min)

    /// Max idle time (seconds)
    pub max_idle_secs: u64,            // Default: 600 (10 min)
}
```

#### Cache Configuration

```rust
pub struct EmbeddingCacheConfig {
    /// Maximum embeddings to cache
    pub max_size: usize,               // Default: 10000

    /// Cache TTL (seconds)
    pub ttl_secs: u64,                 // Default: 3600 (1 hour)
}

pub struct QueryCacheConfig {
    /// Maximum cached queries
    pub max_cache_entries: usize,      // Default: 1000

    /// Query cache TTL (seconds)
    pub ttl_secs: u64,                 // Default: 300 (5 min)

    /// Enable caching
    pub enabled: bool,                 // Default: true
}
```

#### Batch Configuration

```rust
pub struct BatchConfig {
    /// Max batch size for upserts
    pub max_batch_size: usize,         // Default: 100

    /// Batch timeout (ms)
    pub batch_timeout_ms: u64,         // Default: 1000

    /// Enable parallel batching
    pub parallel_batching: bool,       // Default: true
}
```

### 7.2 Tuning Recommendations

#### Development Environment

```rust
let config = EmbeddedStorageConfig {
    data_path: PathBuf::from("./data"),
    require_qdrant: false,  // Use file storage
    vector_size: 384,       // Smaller model (E5-small)
    ..Default::default()
};
```

#### Production Environment (Small Scale)

```rust
let config = EmbeddedStorageConfig::with_qdrant(
    "http://localhost:6333",
    "production",
    1536,  // OpenAI embeddings
);

let conn_config = QdrantConnectionConfig {
    max_connections: 20,
    connect_timeout_secs: 10,
    request_timeout_secs: 30,
    ..Default::default()
};

let cache_config = EmbeddingCacheConfig {
    max_size: 50000,      // 50K embeddings in cache
    ttl_secs: 7200,       // 2 hour TTL
};
```

#### Production Environment (Large Scale)

```rust
let config = EmbeddedStorageConfig::with_qdrant(
    "qdrant-cluster.internal:6333",
    "production_v2",
    1536,
);

let conn_config = QdrantConnectionConfig {
    max_connections: 50,           // Higher connection pool
    connect_timeout_secs: 5,       // Faster timeout
    request_timeout_secs: 15,      // Lower latency budget
    health_check_interval_secs: 60, // More frequent health checks
    ..Default::default()
};

let cache_config = EmbeddingCacheConfig {
    max_size: 500000,     // 500K embeddings (~3GB RAM for 1536-dim)
    ttl_secs: 3600,       // 1 hour TTL
};

let query_cache = QueryCacheConfig {
    max_cache_entries: 10000,  // More query caching
    ttl_secs: 120,             // Shorter TTL for freshness
    enabled: true,
};
```

### 7.3 Memory Budget Guidelines

| Available RAM | max_size (cache) | Expected Vectors |
| ------------- | ---------------- | ---------------- |
| 1 GB          | 10,000           | 100K             |
| 4 GB          | 50,000           | 500K             |
| 8 GB          | 100,000          | 1M               |
| 32 GB         | 500,000          | 5M               |
| 128 GB        | 2,000,000        | 20M              |

**Formula:**

```
cache_memory_mb = max_size * vector_dim * 4 / 1_000_000
max_size = available_mb * 1_000_000 / (vector_dim * 4)
```

### 7.4 Environment Variables

```bash
# Qdrant configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key

# Storage paths
REASONKIT_DATA_PATH=~/.local/share/reasonkit

# Embedding API
OPENAI_API_KEY=sk-...

# Performance tuning
REASONKIT_CACHE_SIZE=10000
REASONKIT_CACHE_TTL_SECS=3600
REASONKIT_CONNECTION_POOL_SIZE=10
```

---

## Appendix A: Glossary

| Term             | Definition                                                    |
| ---------------- | ------------------------------------------------------------- |
| **Hot Memory**   | In-memory cache for fast access to frequently used data       |
| **Cold Storage** | Persistent disk-backed storage (Qdrant or files)              |
| **HNSW**         | Hierarchical Navigable Small World - Qdrant's ANN algorithm   |
| **RRF**          | Reciprocal Rank Fusion - algorithm to combine ranked lists    |
| **WAL**          | Write-Ahead Log - durability mechanism for crash recovery     |
| **RAPTOR**       | Recursive Abstractive Processing for Tree-Organized Retrieval |

## Appendix B: Related Documentation

- [Embedded Mode Guide](./EMBEDDED_MODE_GUIDE.md) - Zero-config development setup
- [Hybrid Search Guide](./HYBRID_SEARCH_GUIDE.md) - Dense + sparse retrieval
- [Embedding Pipeline Guide](./EMBEDDING_PIPELINE_GUIDE.md) - Embedding configuration
- [RAPTOR Tree Guide](./RAPTOR_TREE_GUIDE.md) - Hierarchical retrieval

## Appendix C: Research References

1. **RAPTOR**: Sarthi et al. 2024 - "Recursive Abstractive Processing for Tree-Organized Retrieval"
2. **RRF Fusion**: Cormack et al. 2009 - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
3. **Cross-Encoder Reranking**: Nogueira et al. 2020 - "Document Ranking with a Pretrained Sequence-to-Sequence Model" (arXiv:2010.06467)
4. **HNSW**: Malkov & Yashunin 2016 - "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"

---

*Document maintained by ReasonKit Team. For questions, see [CONTRIBUTING.md](../CONTRIBUTING.md).*
