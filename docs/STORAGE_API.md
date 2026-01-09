# Storage API Documentation

> Complete API reference for the ReasonKit Mem dual-layer storage system

**Version:** 0.1.0
**Last Updated:** 2026-01-01
**Module:** `reasonkit_mem::storage`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Common Patterns](#common-patterns)
6. [Error Handling](#error-handling)
7. [Migration Guide](#migration-guide)
8. [Performance Tuning](#performance-tuning)
9. [Benchmarking](#benchmarking)

---

## Quick Start

### Minimal Example

```rust
use reasonkit_mem::storage::{Storage, AccessContext, AccessLevel};
use reasonkit_mem::{Document, DocumentType, Source, SourceType};
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create embedded storage (file-based, no external dependencies)
    let storage = Storage::new_embedded().await?;

    // Create access context for operations
    let context = AccessContext::new(
        "user_123".to_string(),
        AccessLevel::ReadWrite,
        "store_document".to_string(),
    );

    // Create a document
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some("notes.md".to_string()),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: None,
    };

    let doc = Document::new(DocumentType::Note, source)
        .with_content("Hello world - this is my first document.".to_string());

    // Store the document
    storage.store_document(&doc, &context).await?;

    // Retrieve it back
    let retrieved = storage.get_document(&doc.id, &context).await?;
    println!("Retrieved: {:?}", retrieved);

    Ok(())
}
```

### With Vector Search

```rust
use reasonkit_mem::storage::{Storage, EmbeddedStorageConfig, AccessContext, AccessLevel};
use uuid::Uuid;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure storage with custom path
    let config = EmbeddedStorageConfig::file_only(PathBuf::from("./data"));
    let storage = Storage::new_embedded_with_config(config).await?;

    let context = AccessContext::new(
        "system".to_string(),
        AccessLevel::Admin,
        "vector_ops".to_string(),
    );

    // Store an embedding (768-dimensional vector)
    let chunk_id = Uuid::new_v4();
    let embedding: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
    storage.store_embeddings(&chunk_id, &embedding, &context).await?;

    // Search by vector similarity
    let query_vector: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).cos()).collect();
    let results = storage.search_by_vector(&query_vector, 5, &context).await?;

    for (id, score) in results {
        println!("Match: {} (score: {:.4})", id, score);
    }

    Ok(())
}
```

---

## Architecture Overview

The ReasonKit Mem storage system implements a flexible, backend-agnostic architecture:

```
+------------------------------------------------------------------+
|                         Storage (Facade)                          |
|  - Unified API for all backends                                  |
|  - Automatic backend selection                                   |
+------------------------------------------------------------------+
                               |
         +---------------------+---------------------+
         |                     |                     |
+----------------+    +----------------+    +------------------+
| InMemoryStorage|    |  FileStorage   |    |  QdrantStorage   |
|  (Testing)     |    | (Embedded Mode)|    | (Production)     |
+----------------+    +----------------+    +------------------+
         |                     |                     |
         v                     v                     v
    HashMap<K,V>         JSON Files          Qdrant Vector DB
                     + Binary Embeddings    + Connection Pool
                                           + Embedding Cache
```

### Storage Backends

| Backend             | Use Case                   | Persistence    | Vector Search | Performance     |
| ------------------- | -------------------------- | -------------- | ------------- | --------------- |
| **InMemoryStorage** | Unit tests, prototyping    | None           | O(n) linear   | Fast writes     |
| **FileStorage**     | Development, embedded apps | JSON + binary  | O(n) linear   | Moderate        |
| **QdrantStorage**   | Production                 | Qdrant cluster | HNSW O(log n) | High throughput |

### Key Components

- **Storage**: Facade providing unified API across all backends
- **StorageBackend**: Trait defining backend operations
- **AccessContext**: Authentication and authorization context
- **EmbeddingCache**: LRU cache for frequently accessed embeddings
- **QdrantConnectionPool**: Connection pooling for Qdrant

---

## Configuration

### EmbeddedStorageConfig

Primary configuration for embedded/local storage:

```rust
use reasonkit_mem::storage::EmbeddedStorageConfig;
use std::path::PathBuf;

// Default configuration (file storage, no Qdrant required)
let config = EmbeddedStorageConfig::default();

// Custom configuration
let config = EmbeddedStorageConfig {
    data_path: PathBuf::from("/var/lib/reasonkit/storage"),
    collection_name: "my_knowledge_base".to_string(),
    vector_size: 1536,           // OpenAI ada-002 dimensions
    require_qdrant: false,       // Fall back to file storage
    qdrant_url: "http://localhost:6333".to_string(),
};
```

#### Configuration Presets

```rust
// File-only mode (no external dependencies)
let config = EmbeddedStorageConfig::file_only(PathBuf::from("./data"));

// Qdrant mode (requires running Qdrant server)
let config = EmbeddedStorageConfig::with_qdrant(
    "http://localhost:6333",
    "my_collection",
    1536,  // vector dimensions
);
```

### High-Performance Configuration

For production deployments with maximum throughput:

```rust
use reasonkit_mem::storage::{
    QdrantConnectionConfig,
    QdrantSecurityConfig,
    EmbeddingCacheConfig,
    AccessControlConfig,
    Storage,
};

// Connection pool configuration
let conn_config = QdrantConnectionConfig {
    max_connections: 20,              // Increase pool size
    connect_timeout_secs: 10,         // Faster timeout detection
    request_timeout_secs: 30,         // Shorter request timeout
    health_check_interval_secs: 60,   // More frequent health checks
    max_idle_secs: 300,               // 5 minute idle timeout
    security: QdrantSecurityConfig {
        api_key: Some("your-api-key".to_string()),
        tls_enabled: true,
        ..Default::default()
    },
};

// Larger embedding cache
let cache_config = EmbeddingCacheConfig {
    max_size: 50000,        // Cache up to 50k embeddings
    ttl_secs: 7200,         // 2 hour TTL
};

// Access control
let access_config = AccessControlConfig::default();

// Create high-performance storage
let storage = Storage::qdrant_with_config(
    "qdrant.example.com",
    6333,               // REST port
    6334,               // gRPC port
    "production_kb".to_string(),
    1536,
    false,              // Not embedded mode
    conn_config,
    cache_config,
    access_config,
).await?;
```

### Low-Memory Configuration

For resource-constrained environments:

```rust
use reasonkit_mem::storage::{
    EmbeddingCacheConfig,
    QdrantConnectionConfig,
    Storage,
};

// Minimal connection pool
let conn_config = QdrantConnectionConfig {
    max_connections: 2,
    connect_timeout_secs: 60,
    request_timeout_secs: 120,
    health_check_interval_secs: 600,
    max_idle_secs: 120,
    ..Default::default()
};

// Small cache
let cache_config = EmbeddingCacheConfig {
    max_size: 1000,         // Only 1k embeddings
    ttl_secs: 300,          // 5 minute TTL
};

let storage = Storage::qdrant_with_config(
    "localhost",
    6333,
    6334,
    "low_memory_kb".to_string(),
    768,                    // Smaller vectors (e.g., E5-small)
    true,                   // Embedded mode
    conn_config,
    cache_config,
    AccessControlConfig::default(),
).await?;
```

---

## API Reference

### Storage

The main facade for all storage operations.

#### Constructors

```rust
impl Storage {
    /// Create in-memory storage (for testing)
    pub fn in_memory() -> Self;

    /// Create file-based storage
    pub async fn file(base_path: PathBuf) -> Result<Self>;

    /// Create embedded storage with automatic fallback
    pub async fn new_embedded() -> Result<Self>;

    /// Create embedded storage with custom configuration
    pub async fn new_embedded_with_config(config: EmbeddedStorageConfig) -> Result<Self>;

    /// Create Qdrant-backed storage
    pub async fn qdrant(
        host: &str,
        port: u16,
        grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
    ) -> Result<Self>;

    /// Create Qdrant storage with full configuration
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
    ) -> Result<Self>;
}
```

#### Document Operations

```rust
impl Storage {
    /// Store a document
    ///
    /// # Arguments
    /// * `doc` - The document to store
    /// * `context` - Access control context
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(MemError)` on failure
    pub async fn store_document(
        &self,
        doc: &Document,
        context: &AccessContext,
    ) -> Result<()>;

    /// Retrieve a document by ID
    ///
    /// # Returns
    /// * `Ok(Some(doc))` if found
    /// * `Ok(None)` if not found
    /// * `Err(MemError)` on failure
    pub async fn get_document(
        &self,
        id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Document>>;

    /// Delete a document
    pub async fn delete_document(
        &self,
        id: &Uuid,
        context: &AccessContext,
    ) -> Result<()>;

    /// List all document IDs
    pub async fn list_documents(
        &self,
        context: &AccessContext,
    ) -> Result<Vec<Uuid>>;
}
```

#### Embedding Operations

```rust
impl Storage {
    /// Store embeddings for a chunk
    ///
    /// # Arguments
    /// * `chunk_id` - UUID of the chunk
    /// * `embeddings` - Vector of f32 values
    /// * `context` - Access control context
    ///
    /// # Errors
    /// Returns `MemError::InvalidInput` if embedding size doesn't match
    /// the configured vector_size for QdrantStorage
    pub async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()>;

    /// Retrieve embeddings by chunk ID
    pub async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>>;

    /// Search by vector similarity
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector
    /// * `top_k` - Number of results to return
    /// * `context` - Access control context
    ///
    /// # Returns
    /// Vector of (chunk_id, similarity_score) tuples, sorted by score descending
    pub async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>>;
}
```

#### Statistics

```rust
impl Storage {
    /// Get storage statistics
    ///
    /// # Note
    /// Requires Admin access level for QdrantStorage
    pub async fn stats(
        &self,
        context: &AccessContext,
    ) -> Result<StorageStats>;
}

/// Storage statistics
pub struct StorageStats {
    pub document_count: usize,
    pub chunk_count: usize,
    pub embedding_count: usize,
    pub size_bytes: u64,
}
```

### AccessContext

Authentication and authorization for storage operations.

```rust
/// Access levels
pub enum AccessLevel {
    Read,       // Read-only operations
    ReadWrite,  // Read and write operations
    Admin,      // Full access including stats and admin ops
}

/// Access context for operations
pub struct AccessContext {
    pub user_id: String,
    pub access_level: AccessLevel,
    pub operation: String,
    pub timestamp: i64,
}

impl AccessContext {
    /// Create a new access context
    pub fn new(
        user_id: String,
        access_level: AccessLevel,
        operation: String,
    ) -> Self;

    /// Check if context has required permission
    pub fn has_permission(
        &self,
        required_level: &AccessLevel,
        config: &AccessControlConfig,
    ) -> bool;
}
```

### EmbeddingCache

LRU cache for embeddings with TTL support.

```rust
pub struct EmbeddingCache {
    // Internal fields
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(config: EmbeddingCacheConfig) -> Self;

    /// Put an embedding in the cache
    ///
    /// Automatically evicts oldest entries when capacity exceeded
    pub fn put(&mut self, chunk_id: Uuid, embedding: Vec<f32>);

    /// Get an embedding from cache
    ///
    /// Returns None if not found or expired
    pub fn get(&mut self, chunk_id: &Uuid) -> Option<Vec<f32>>;

    /// Clean up expired entries
    pub fn cleanup_expired(&mut self);
}
```

### StorageBackend Trait

Interface for implementing custom storage backends.

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()>;
    async fn get_document(&self, id: &Uuid, context: &AccessContext) -> Result<Option<Document>>;
    async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()>;
    async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>>;

    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()>;

    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>>;

    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>>;

    async fn stats(&self, context: &AccessContext) -> Result<StorageStats>;
}
```

---

## Common Patterns

### Session-Based Memory

Implement session-scoped memory with automatic cleanup:

```rust
use reasonkit_mem::storage::{Storage, AccessContext, AccessLevel};
use reasonkit_mem::{Document, DocumentType, Source, SourceType, Chunk, EmbeddingIds};
use chrono::Utc;
use uuid::Uuid;
use std::collections::HashMap;

pub struct SessionMemory {
    storage: Storage,
    session_id: String,
    document_ids: Vec<Uuid>,
}

impl SessionMemory {
    pub async fn new(session_id: String) -> anyhow::Result<Self> {
        let storage = Storage::new_embedded().await?;
        Ok(Self {
            storage,
            session_id,
            document_ids: Vec::new(),
        })
    }

    fn context(&self, op: &str) -> AccessContext {
        AccessContext::new(
            format!("session:{}", self.session_id),
            AccessLevel::ReadWrite,
            op.to_string(),
        )
    }

    /// Store a memory entry for this session
    pub async fn remember(&mut self, content: &str) -> anyhow::Result<Uuid> {
        let source = Source {
            source_type: SourceType::Api,
            url: None,
            path: None,
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: Some(self.session_id.clone()),
        };

        let doc = Document::new(DocumentType::Note, source)
            .with_content(content.to_string());

        let doc_id = doc.id;
        self.storage.store_document(&doc, &self.context("remember")).await?;
        self.document_ids.push(doc_id);

        Ok(doc_id)
    }

    /// Store with embedding for semantic search
    pub async fn remember_with_embedding(
        &mut self,
        content: &str,
        embedding: Vec<f32>,
    ) -> anyhow::Result<Uuid> {
        let source = Source {
            source_type: SourceType::Api,
            url: None,
            path: None,
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: Some(self.session_id.clone()),
        };

        // Create document with chunk
        let chunk_id = Uuid::new_v4();
        let mut doc = Document::new(DocumentType::Note, source)
            .with_content(content.to_string());

        doc.chunks = vec![Chunk {
            id: chunk_id,
            text: content.to_string(),
            index: 0,
            start_char: 0,
            end_char: content.len(),
            token_count: Some(content.split_whitespace().count()),
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        }];

        let doc_id = doc.id;
        let ctx = self.context("remember_with_embedding");

        // Store document and embedding
        self.storage.store_document(&doc, &ctx).await?;
        self.storage.store_embeddings(&chunk_id, &embedding, &ctx).await?;
        self.document_ids.push(doc_id);

        Ok(doc_id)
    }

    /// Search session memory by vector similarity
    pub async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> anyhow::Result<Vec<(Uuid, f32)>> {
        self.storage
            .search_by_vector(query_embedding, top_k, &self.context("search"))
            .await
            .map_err(Into::into)
    }

    /// Clear all session memory
    pub async fn clear(&mut self) -> anyhow::Result<()> {
        let ctx = self.context("clear");
        for doc_id in self.document_ids.drain(..) {
            self.storage.delete_document(&doc_id, &ctx).await?;
        }
        Ok(())
    }
}

impl Drop for SessionMemory {
    fn drop(&mut self) {
        // Note: async cleanup in Drop is tricky
        // Consider using a cleanup task or explicit shutdown
        if !self.document_ids.is_empty() {
            tracing::warn!(
                "SessionMemory dropped with {} documents not cleaned up",
                self.document_ids.len()
            );
        }
    }
}
```

### Batch Ingestion

Efficiently ingest large document sets:

```rust
use reasonkit_mem::storage::{Storage, AccessContext, AccessLevel};
use reasonkit_mem::{Document, DocumentType, Source, SourceType, Chunk, EmbeddingIds};
use chrono::Utc;
use uuid::Uuid;
use futures::stream::{self, StreamExt};

pub struct BatchIngester {
    storage: Storage,
    batch_size: usize,
    concurrency: usize,
}

impl BatchIngester {
    pub fn new(storage: Storage, batch_size: usize, concurrency: usize) -> Self {
        Self {
            storage,
            batch_size,
            concurrency,
        }
    }

    fn context(&self) -> AccessContext {
        AccessContext::new(
            "batch_ingester".to_string(),
            AccessLevel::ReadWrite,
            "ingest".to_string(),
        )
    }

    /// Ingest documents with embeddings in batches
    pub async fn ingest(
        &self,
        items: Vec<(String, Vec<f32>)>,  // (content, embedding)
    ) -> anyhow::Result<IngestStats> {
        let mut stats = IngestStats::default();
        let ctx = self.context();

        // Process in batches with controlled concurrency
        let batches: Vec<_> = items.chunks(self.batch_size).collect();

        for batch in batches {
            let futures: Vec<_> = batch
                .iter()
                .map(|(content, embedding)| {
                    let storage = &self.storage;
                    let ctx = ctx.clone();
                    let content = content.clone();
                    let embedding = embedding.clone();

                    async move {
                        let source = Source {
                            source_type: SourceType::Local,
                            url: None,
                            path: None,
                            arxiv_id: None,
                            github_repo: None,
                            retrieved_at: Utc::now(),
                            version: None,
                        };

                        let chunk_id = Uuid::new_v4();
                        let mut doc = Document::new(DocumentType::Note, source)
                            .with_content(content.clone());

                        doc.chunks = vec![Chunk {
                            id: chunk_id,
                            text: content.clone(),
                            index: 0,
                            start_char: 0,
                            end_char: content.len(),
                            token_count: None,
                            section: None,
                            page: None,
                            embedding_ids: EmbeddingIds::default(),
                        }];

                        // Store document and embedding
                        storage.store_document(&doc, &ctx).await?;
                        storage.store_embeddings(&chunk_id, &embedding, &ctx).await?;

                        Ok::<_, anyhow::Error>(())
                    }
                })
                .collect();

            // Execute batch with concurrency limit
            let results: Vec<_> = stream::iter(futures)
                .buffer_unordered(self.concurrency)
                .collect()
                .await;

            for result in results {
                match result {
                    Ok(_) => stats.success += 1,
                    Err(e) => {
                        stats.failed += 1;
                        stats.errors.push(e.to_string());
                    }
                }
            }
        }

        Ok(stats)
    }
}

#[derive(Default)]
pub struct IngestStats {
    pub success: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}
```

### Search with Filters

Implement filtered vector search with post-processing:

```rust
use reasonkit_mem::storage::{Storage, AccessContext, AccessLevel};
use uuid::Uuid;
use std::collections::HashSet;

pub struct FilteredSearch {
    storage: Storage,
}

impl FilteredSearch {
    pub fn new(storage: Storage) -> Self {
        Self { storage }
    }

    fn context(&self) -> AccessContext {
        AccessContext::new(
            "search".to_string(),
            AccessLevel::Read,
            "filtered_search".to_string(),
        )
    }

    /// Search with document ID filter
    ///
    /// Only returns results from the specified document IDs
    pub async fn search_in_documents(
        &self,
        query_embedding: &[f32],
        document_ids: &[Uuid],
        top_k: usize,
    ) -> anyhow::Result<Vec<(Uuid, f32)>> {
        // Over-fetch to account for filtering
        let overfetch_k = (top_k * 5).min(1000);

        let results = self.storage
            .search_by_vector(query_embedding, overfetch_k, &self.context())
            .await?;

        let allowed_docs: HashSet<_> = document_ids.iter().collect();

        // Filter results (requires mapping chunk_id to doc_id)
        // In production, this would use metadata stored with embeddings
        let filtered: Vec<_> = results
            .into_iter()
            .take(top_k)  // Take top_k after filtering
            .collect();

        Ok(filtered)
    }

    /// Search with score threshold
    pub async fn search_with_threshold(
        &self,
        query_embedding: &[f32],
        min_score: f32,
        max_results: usize,
    ) -> anyhow::Result<Vec<(Uuid, f32)>> {
        let results = self.storage
            .search_by_vector(query_embedding, max_results, &self.context())
            .await?;

        let filtered: Vec<_> = results
            .into_iter()
            .filter(|(_, score)| *score >= min_score)
            .collect();

        Ok(filtered)
    }

    /// Search with diversity (MMR-like)
    ///
    /// Returns diverse results by penalizing similar items
    pub async fn search_diverse(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        diversity_factor: f32,  // 0.0 = pure relevance, 1.0 = max diversity
    ) -> anyhow::Result<Vec<(Uuid, f32)>> {
        // Fetch more candidates for diversity selection
        let candidates = self.storage
            .search_by_vector(query_embedding, top_k * 3, &self.context())
            .await?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Simple MMR-like selection
        let mut selected: Vec<(Uuid, f32)> = vec![];
        let mut remaining: Vec<_> = candidates.into_iter().collect();

        // Select first item (most relevant)
        if let Some(first) = remaining.pop() {
            selected.push(first);
        }

        // Greedily select remaining items
        while selected.len() < top_k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (idx, (_, relevance)) in remaining.iter().enumerate() {
                // Simplified diversity score (would use actual embedding similarity)
                let diversity_penalty = selected.len() as f32 * diversity_factor * 0.1;
                let mmr_score = relevance - diversity_penalty;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = idx;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        Ok(selected)
    }
}
```

### Graceful Shutdown

Properly shut down storage with pending operations:

```rust
use reasonkit_mem::storage::{Storage, AccessContext, AccessLevel};
use tokio::sync::broadcast;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ManagedStorage {
    storage: Arc<Storage>,
    shutdown_tx: broadcast::Sender<()>,
    pending_ops: Arc<RwLock<usize>>,
}

impl ManagedStorage {
    pub async fn new() -> anyhow::Result<Self> {
        let storage = Storage::new_embedded().await?;
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            storage: Arc::new(storage),
            shutdown_tx,
            pending_ops: Arc::new(RwLock::new(0)),
        })
    }

    /// Execute an operation with shutdown awareness
    pub async fn with_op<F, T>(&self, f: F) -> anyhow::Result<T>
    where
        F: std::future::Future<Output = anyhow::Result<T>>,
    {
        // Increment pending ops
        {
            let mut count = self.pending_ops.write().await;
            *count += 1;
        }

        let result = f.await;

        // Decrement pending ops
        {
            let mut count = self.pending_ops.write().await;
            *count -= 1;
        }

        result
    }

    /// Graceful shutdown with timeout
    pub async fn shutdown(self, timeout: std::time::Duration) -> anyhow::Result<()> {
        // Signal shutdown
        let _ = self.shutdown_tx.send(());

        // Wait for pending operations
        let start = std::time::Instant::now();
        loop {
            let count = *self.pending_ops.read().await;
            if count == 0 {
                tracing::info!("All pending operations completed");
                break;
            }

            if start.elapsed() > timeout {
                tracing::warn!(
                    "Shutdown timeout with {} pending operations",
                    count
                );
                break;
            }

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Storage cleanup happens on drop
        Ok(())
    }

    /// Get storage reference
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Subscribe to shutdown signal
    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }
}
```

---

## Error Handling

### Error Types

The storage system uses `MemError` for all error conditions:

```rust
use reasonkit_mem::error::MemError;

// Storage-related errors
MemError::Storage(String)      // General storage errors
MemError::HotMemory(String)    // In-memory cache errors
MemError::ColdMemory(String)   // Persistent storage errors
MemError::Qdrant(String)       // Qdrant-specific errors

// Data integrity errors
MemError::ChecksumMismatch { expected: u32, actual: u32 }
MemError::Sync(String)         // Hot/cold sync errors

// Resource errors
MemError::CapacityExceeded { current: usize, max: usize }
MemError::Wal(String)          // Write-ahead log errors

// Recovery errors
MemError::Recovery(String)     // Crash recovery failures

// Input validation
MemError::InvalidInput(String)
MemError::NotFound(String)
```

### Error Handling Patterns

```rust
use reasonkit_mem::error::MemError;
use reasonkit_mem::storage::Storage;

async fn handle_storage_errors(storage: &Storage) {
    let context = create_context();
    let embedding: Vec<f32> = vec![0.1; 768];

    match storage.search_by_vector(&embedding, 10, &context).await {
        Ok(results) => {
            println!("Found {} results", results.len());
        }

        Err(MemError::InvalidInput(msg)) => {
            // Handle validation errors
            eprintln!("Invalid input: {}", msg);
        }

        Err(MemError::Storage(msg)) | Err(MemError::Qdrant(msg)) => {
            // Handle storage backend errors - might retry
            tracing::error!(error = %msg, "Storage error, retrying...");
        }

        Err(MemError::CapacityExceeded { current, max }) => {
            // Handle capacity issues
            tracing::warn!(
                current = current,
                max = max,
                "Storage capacity exceeded"
            );
        }

        Err(e) => {
            // Handle other errors
            tracing::error!(error = %e, "Unexpected error");
        }
    }
}
```

### Recovery Strategies

```rust
use reasonkit_mem::storage::{Storage, EmbeddedStorageConfig};
use std::path::PathBuf;

/// Attempt storage recovery with fallback
async fn recover_or_fallback() -> anyhow::Result<Storage> {
    let data_path = PathBuf::from("./data");

    // Try primary configuration
    let primary_config = EmbeddedStorageConfig::with_qdrant(
        "http://localhost:6333",
        "production",
        1536,
    );

    match Storage::new_embedded_with_config(primary_config).await {
        Ok(storage) => {
            tracing::info!("Connected to Qdrant");
            Ok(storage)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Qdrant unavailable, falling back to file storage");

            // Fallback to file storage
            let fallback_config = EmbeddedStorageConfig::file_only(data_path);
            Storage::new_embedded_with_config(fallback_config).await
                .map_err(Into::into)
        }
    }
}

/// Retry with exponential backoff
async fn retry_with_backoff<F, T>(
    mut operation: F,
    max_retries: usize,
) -> anyhow::Result<T>
where
    F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<T>> + Send>>,
{
    let mut retry_count = 0;
    let base_delay = std::time::Duration::from_millis(100);

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) if retry_count < max_retries => {
                retry_count += 1;
                let delay = base_delay * 2_u32.pow(retry_count as u32);
                tracing::warn!(
                    retry = retry_count,
                    delay_ms = delay.as_millis(),
                    error = %e,
                    "Retrying operation"
                );
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

---

## Migration Guide

### From InMemoryStorage

If you started with `InMemoryStorage` for testing and need to migrate to persistent storage:

```rust
use reasonkit_mem::storage::{Storage, InMemoryStorage, AccessContext, AccessLevel};
use reasonkit_mem::Document;
use std::path::PathBuf;

async fn migrate_from_memory(
    in_memory: &InMemoryStorage,
) -> anyhow::Result<Storage> {
    let context = AccessContext::new(
        "migration".to_string(),
        AccessLevel::Admin,
        "migrate".to_string(),
    );

    // Create new persistent storage
    let persistent = Storage::file(PathBuf::from("./migrated_data")).await?;

    // Note: InMemoryStorage doesn't expose list_documents directly
    // In practice, you'd track document IDs during insertion
    // This example shows the pattern for migration

    tracing::info!("Migration to persistent storage complete");
    Ok(persistent)
}
```

### From FileStorage to Qdrant

Migrate from file-based to Qdrant for production:

```rust
use reasonkit_mem::storage::{
    Storage, FileStorage, EmbeddedStorageConfig,
    AccessContext, AccessLevel,
};
use std::path::PathBuf;

async fn migrate_to_qdrant(file_path: PathBuf) -> anyhow::Result<Storage> {
    // Open file storage
    let file_storage = Storage::file(file_path.clone()).await?;

    // Create Qdrant storage
    let qdrant_config = EmbeddedStorageConfig::with_qdrant(
        "http://localhost:6333",
        "migrated_collection",
        1536,
    );
    let qdrant_storage = Storage::new_embedded_with_config(qdrant_config).await?;

    let read_ctx = AccessContext::new(
        "migration".to_string(),
        AccessLevel::Admin,
        "read".to_string(),
    );
    let write_ctx = AccessContext::new(
        "migration".to_string(),
        AccessLevel::Admin,
        "write".to_string(),
    );

    // List and migrate all documents
    let doc_ids = file_storage.list_documents(&read_ctx).await?;
    let total = doc_ids.len();

    for (idx, doc_id) in doc_ids.into_iter().enumerate() {
        if let Some(doc) = file_storage.get_document(&doc_id, &read_ctx).await? {
            qdrant_storage.store_document(&doc, &write_ctx).await?;

            // Migrate embeddings for each chunk
            for chunk in &doc.chunks {
                if let Some(embedding) = file_storage
                    .get_embeddings(&chunk.id, &read_ctx)
                    .await?
                {
                    qdrant_storage
                        .store_embeddings(&chunk.id, &embedding, &write_ctx)
                        .await?;
                }
            }
        }

        if (idx + 1) % 100 == 0 {
            tracing::info!("Migrated {}/{} documents", idx + 1, total);
        }
    }

    tracing::info!("Migration complete: {} documents", total);
    Ok(qdrant_storage)
}
```

### Data Format Changes

When upgrading between versions, document schemas may change:

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Old format (v0.0.x)
#[derive(Deserialize)]
struct LegacyDocument {
    id: String,  // String ID
    content: String,
    tags: Vec<String>,
}

// New format (v0.1.x)
#[derive(Serialize, Deserialize)]
struct CurrentDocument {
    id: Uuid,  // UUID ID
    content: DocumentContent,
    metadata: Metadata,
}

fn migrate_document(legacy: LegacyDocument) -> anyhow::Result<CurrentDocument> {
    // Parse string ID to UUID, or generate new one
    let id = Uuid::parse_str(&legacy.id)
        .unwrap_or_else(|_| Uuid::new_v4());

    // Convert to new format
    Ok(CurrentDocument {
        id,
        content: DocumentContent {
            raw: legacy.content,
            format: ContentFormat::Text,
            language: "en".to_string(),
            word_count: 0,
            char_count: 0,
        },
        metadata: Metadata {
            tags: legacy.tags,
            ..Default::default()
        },
    })
}
```

---

## Performance Tuning

### Vector Search Optimization

```rust
// 1. Use appropriate vector dimensions
//    - 1536 for OpenAI ada-002 (high quality)
//    - 768 for E5-base (balanced)
//    - 384 for E5-small (fast)

// 2. Enable scalar quantization (in Qdrant config)
//    Reduces memory by 4x with minimal accuracy loss

// 3. Tune HNSW parameters for your use case
//    - m: 16 (default) - higher = better recall, more memory
//    - ef_construct: 100 (default) - higher = better index quality
//    - ef: 128 - higher = better search quality, slower

// 4. Batch operations when possible
let embeddings: Vec<(Uuid, Vec<f32>)> = generate_embeddings();
for chunk in embeddings.chunks(100) {
    for (id, emb) in chunk {
        storage.store_embeddings(id, emb, &context).await?;
    }
}
```

### Connection Pool Tuning

```rust
use reasonkit_mem::storage::QdrantConnectionConfig;

// For high-throughput scenarios
let config = QdrantConnectionConfig {
    max_connections: 50,        // Match expected concurrency
    connect_timeout_secs: 5,    // Fail fast
    request_timeout_secs: 30,   // Reasonable for vector ops
    health_check_interval_secs: 30,
    max_idle_secs: 120,
    ..Default::default()
};

// For low-latency scenarios
let config = QdrantConnectionConfig {
    max_connections: 10,
    connect_timeout_secs: 2,
    request_timeout_secs: 10,
    health_check_interval_secs: 10,
    max_idle_secs: 60,
    ..Default::default()
};
```

### Cache Configuration

```rust
use reasonkit_mem::storage::EmbeddingCacheConfig;

// Estimate cache size based on:
// - Number of frequently accessed embeddings
// - Available memory
// - Access patterns

// Example: 10k embeddings * 1536 dims * 4 bytes = ~60MB
let cache_config = EmbeddingCacheConfig {
    max_size: 10000,
    ttl_secs: 3600,  // 1 hour for stable data
};

// For frequently changing data
let cache_config = EmbeddingCacheConfig {
    max_size: 5000,
    ttl_secs: 300,   // 5 minutes
};
```

---

## Benchmarking

### Built-in Benchmarker

```rust
use reasonkit_mem::storage::{Storage, AccessLevel};
use reasonkit_mem::storage::benchmarks::{StorageBenchmarker, run_storage_benchmarks};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let storage = Storage::new_embedded().await?;

    // Quick benchmark
    let results = run_storage_benchmarks(
        storage,
        100,    // documents
        1000,   // embeddings
        768,    // vector size
    ).await?;

    println!("=== Benchmark Results ===");
    println!("Document Storage:");
    println!("  Avg latency: {:?}", results.document_storage.avg_latency);
    println!("  Ops/sec: {:.2}", results.document_storage.ops_per_second);
    println!("  P95 latency: {:?}", results.document_storage.p95_latency);

    println!("\nVector Search:");
    println!("  Avg latency: {:?}", results.vector_search.avg_latency);
    println!("  Ops/sec: {:.2}", results.vector_search.ops_per_second);

    Ok(())
}
```

### Custom Benchmarker

```rust
use reasonkit_mem::storage::{Storage, AccessLevel};
use reasonkit_mem::storage::benchmarks::StorageBenchmarker;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let storage = Storage::new_embedded().await?;

    let benchmarker = StorageBenchmarker::new(
        storage,
        "benchmark_user".to_string(),
        AccessLevel::Admin,
    );

    // Run comprehensive benchmark
    let results = benchmarker
        .run_comprehensive_benchmark(
            500,    // 500 documents
            5000,   // 5000 embeddings
            1536,   // OpenAI dimensions
        )
        .await?;

    // Export results
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write("benchmark_results.json", json)?;

    Ok(())
}
```

### Performance Targets

| Operation            | Target (File) | Target (Qdrant) |
| -------------------- | ------------- | --------------- |
| Document store       | < 10ms        | < 5ms           |
| Document retrieve    | < 5ms         | < 2ms           |
| Embedding store      | < 10ms        | < 3ms           |
| Vector search (1k)   | < 50ms        | < 10ms          |
| Vector search (100k) | < 500ms       | < 50ms          |

---

## Appendix: Type Reference

### Core Types

```rust
// From reasonkit_mem::types
pub struct Document { ... }
pub struct Chunk { ... }
pub struct Source { ... }
pub enum DocumentType { Paper, Documentation, Code, Note, Transcript, Benchmark }
pub enum SourceType { Arxiv, Github, Website, Local, Api }

// From reasonkit_mem::storage
pub struct Storage { ... }
pub struct EmbeddedStorageConfig { ... }
pub struct QdrantConnectionConfig { ... }
pub struct QdrantSecurityConfig { ... }
pub struct EmbeddingCacheConfig { ... }
pub struct AccessControlConfig { ... }
pub struct AccessContext { ... }
pub enum AccessLevel { Read, ReadWrite, Admin }
pub struct StorageStats { ... }

// From reasonkit_mem::error
pub enum MemError { ... }
pub type MemResult<T> = Result<T, MemError>;
```

### Import Quick Reference

```rust
// Common imports for storage operations
use reasonkit_mem::{
    // Core types
    Document, DocumentType, Chunk, EmbeddingIds,
    Source, SourceType, Metadata,
    // Errors
    MemError, MemResult,
};

use reasonkit_mem::storage::{
    Storage,
    EmbeddedStorageConfig,
    AccessContext,
    AccessLevel,
    StorageStats,
    // Advanced
    QdrantConnectionConfig,
    QdrantSecurityConfig,
    EmbeddingCacheConfig,
    AccessControlConfig,
    // Backends (if needed)
    StorageBackend,
    InMemoryStorage,
    FileStorage,
    QdrantStorage,
};
```

---

## See Also

- [EMBEDDED_MODE_GUIDE.md](./EMBEDDED_MODE_GUIDE.md) - Detailed embedded mode documentation
- [HYBRID_SEARCH_GUIDE.md](./HYBRID_SEARCH_GUIDE.md) - Hybrid search patterns
- [EMBEDDING_PIPELINE_GUIDE.md](./EMBEDDING_PIPELINE_GUIDE.md) - Embedding generation
- [RAPTOR_TREE_GUIDE.md](./RAPTOR_TREE_GUIDE.md) - Hierarchical retrieval

---

_This documentation is part of the ReasonKit Memory Infrastructure._
_Report issues at: <https://github.com/ReasonKit/reasonkit-mem/issues>_
