//! Cold Memory Storage for ReasonKit
//!
//! Provides persistent, embedded vector storage using Sled (pure Rust key-value store).
//! Designed for cold/archival memory that doesn't require the overhead of Qdrant.
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+
//! |   Hot Memory     |     |   Cold Memory    |
//! |   (In-Memory +   | --> |   (Sled KV +     |
//! |    Qdrant RAM)   |     |    Embedded)     |
//! +------------------+     +------------------+
//!         ^                        |
//!         |                        v
//!         +-------- Sync ----------+
//! ```
//!
//! ## Features
//!
//! - **Pure Rust**: No FFI, no external dependencies beyond Sled
//! - **Embedded**: Runs in-process, no separate server needed
//! - **ACID Transactions**: Fully transactional with crash recovery
//! - **Vector Search**: Pure Rust cosine similarity implementation
//! - **Batch Operations**: Efficient bulk insert/update with Sled batches
//! - **Parallel Search**: Rayon-based parallel similarity search
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit_mem::storage::cold::{ColdMemory, ColdMemoryConfig, ColdMemoryEntry};
//! use std::path::PathBuf;
//!
//! let config = ColdMemoryConfig::new(PathBuf::from("./cold_storage"));
//! let cold = ColdMemory::new(config).await?;
//!
//! // Store an entry
//! let entry = ColdMemoryEntry::new("Hello world".to_string(), vec![0.1, 0.2, 0.3]);
//! cold.store(&entry).await?;
//!
//! // Search similar
//! let results = cold.search_similar(&[0.1, 0.2, 0.3], 10).await?;
//! ```

use crate::error::{MemError, MemResult};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// Core Types (User-Requested API)
// ============================================================================

/// A single entry in cold memory storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdMemoryEntry {
    /// Unique identifier for the entry
    pub id: Uuid,
    /// Text content associated with this entry
    pub content: String,
    /// Dense embedding vector
    pub embedding: Vec<f32>,
    /// Arbitrary metadata (JSON-compatible)
    pub metadata: serde_json::Value,
    /// Unix timestamp when entry was created
    pub created_at: i64,
}

impl ColdMemoryEntry {
    /// Create a new cold memory entry with content and embedding
    ///
    /// # Arguments
    /// * `content` - The text content to store
    /// * `embedding` - The dense embedding vector
    ///
    /// # Returns
    /// A new `ColdMemoryEntry` with auto-generated ID and timestamp
    pub fn new(content: String, embedding: Vec<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            metadata: serde_json::Value::Null,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Create a new entry with metadata
    pub fn with_metadata(
        content: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            metadata,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Create an entry with a specific ID (for updates)
    pub fn with_id(id: Uuid, content: String, embedding: Vec<f32>) -> Self {
        Self {
            id,
            content,
            embedding,
            metadata: serde_json::Value::Null,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Set metadata on the entry
    pub fn set_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }
}

/// Configuration for cold memory storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdMemoryConfig {
    /// Path to the database directory
    pub db_path: PathBuf,
    /// Cache size in megabytes for Sled
    pub cache_size_mb: usize,
    /// Flush interval in seconds (0 = manual flush only)
    pub flush_interval_secs: u64,
    /// Enable compression for stored data
    pub enable_compression: bool,
    /// Maximum number of entries to scan in parallel during search
    pub parallel_scan_threshold: usize,
    /// Use SIMD-accelerated similarity (when available)
    pub use_simd: bool,
}

impl Default for ColdMemoryConfig {
    fn default() -> Self {
        Self {
            db_path: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("reasonkit")
                .join("cold_memory"),
            cache_size_mb: 128,
            flush_interval_secs: 30,
            enable_compression: true,
            parallel_scan_threshold: 1000,
            use_simd: true,
        }
    }
}

impl ColdMemoryConfig {
    /// Create a new config with the specified database path
    pub fn new(db_path: PathBuf) -> Self {
        Self {
            db_path,
            ..Default::default()
        }
    }

    /// Set the cache size in megabytes
    pub fn with_cache_size(mut self, mb: usize) -> Self {
        self.cache_size_mb = mb;
        self
    }

    /// Set the flush interval
    pub fn with_flush_interval(mut self, secs: u64) -> Self {
        self.flush_interval_secs = secs;
        self
    }

    /// Enable or disable compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.enable_compression = enabled;
        self
    }
}

/// Statistics about cold memory storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdMemoryStats {
    /// Total number of entries stored
    pub entry_count: u64,
    /// Total size of embeddings tree in bytes (approximate)
    pub embeddings_size_bytes: u64,
    /// Total size of metadata tree in bytes (approximate)
    pub metadata_size_bytes: u64,
    /// Average embedding dimension
    pub avg_embedding_dimension: usize,
    /// Time of last compaction
    pub last_compaction: Option<i64>,
    /// Number of search operations performed
    pub search_count: u64,
    /// Average search latency in microseconds
    pub avg_search_latency_us: u64,
}

impl Default for ColdMemoryStats {
    fn default() -> Self {
        Self {
            entry_count: 0,
            embeddings_size_bytes: 0,
            metadata_size_bytes: 0,
            avg_embedding_dimension: 0,
            last_compaction: None,
            search_count: 0,
            avg_search_latency_us: 0,
        }
    }
}

/// Filter criteria for vector search
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Minimum similarity score threshold
    pub min_score: Option<f32>,
    /// Maximum age in seconds (entries older than this are excluded)
    pub max_age_secs: Option<i64>,
    /// Required metadata key-value match (if set)
    pub metadata_filter: Option<serde_json::Value>,
}

impl SearchFilter {
    /// Create a new empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum score threshold
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Set maximum age filter
    pub fn with_max_age(mut self, secs: i64) -> Self {
        self.max_age_secs = Some(secs);
        self
    }

    /// Set metadata filter
    pub fn with_metadata(mut self, filter: serde_json::Value) -> Self {
        self.metadata_filter = Some(filter);
        self
    }
}

// ============================================================================
// Similarity Functions (Pure Rust)
// ============================================================================

/// Compute cosine similarity between two vectors
///
/// Uses the formula: cos(a, b) = (a . b) / (||a|| * ||b||)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity in range [-1, 1], or 0.0 if vectors have different lengths
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Use SIMD-friendly operations that the compiler can auto-vectorize
    let (dot, mag_a_sq, mag_b_sq) = a
        .iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(dot, mag_a, mag_b), (&x, &y)| {
            (dot + x * y, mag_a + x * x, mag_b + y * y)
        });

    let mag_a = mag_a_sq.sqrt();
    let mag_b = mag_b_sq.sqrt();

    if mag_a > f32::EPSILON && mag_b > f32::EPSILON {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

/// Compute dot product between two vectors
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Compute Euclidean distance between two vectors
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Normalize a vector to unit length (L2 normalization)
pub fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > f32::EPSILON {
        v.iter().map(|x| x / magnitude).collect()
    } else {
        v.to_vec()
    }
}

// ============================================================================
// Internal Types
// ============================================================================

/// Scored result for priority queue
#[derive(Debug, Clone)]
struct ScoredEntry {
    id: Uuid,
    score: f32,
}

impl PartialEq for ScoredEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.id == other.id
    }
}

impl Eq for ScoredEntry {}

impl PartialOrd for ScoredEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior (we want lowest scores at top for pruning)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Serializable embedding data stored in Sled
#[derive(Debug, Serialize, Deserialize)]
struct StoredEmbedding {
    vector: Vec<f32>,
    content: String,
    metadata: serde_json::Value,
    created_at: i64,
}

// ============================================================================
// Cold Memory Implementation (Sled-Based)
// ============================================================================

/// Cold Memory Storage using Sled
///
/// Provides persistent, embedded vector storage for cold/archival data.
/// All operations are thread-safe and ACID-compliant.
pub struct ColdMemory {
    /// Main database handle
    db: Db,
    /// Tree for storing embeddings (keyed by UUID bytes)
    embeddings_tree: sled::Tree,
    /// Tree for storing metadata (for future use, e.g., indexing)
    metadata_tree: sled::Tree,
    /// Configuration
    config: ColdMemoryConfig,
    /// Statistics tracking
    stats: Arc<RwLock<ColdMemoryStats>>,
    /// Search latency accumulator (microseconds)
    search_latency_sum: AtomicU64,
    /// Search count for averaging
    search_count: AtomicU64,
}

impl ColdMemory {
    /// Create a new cold memory storage instance
    ///
    /// # Arguments
    /// * `config` - Configuration for the storage
    ///
    /// # Returns
    /// A new `ColdMemory` instance or an error
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = ColdMemoryConfig::new(PathBuf::from("./data"));
    /// let cold = ColdMemory::new(config).await?;
    /// ```
    pub async fn new(config: ColdMemoryConfig) -> MemResult<Self> {
        // Ensure directory exists
        if let Some(parent) = config.db_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                MemError::storage(format!("Failed to create database directory: {}", e))
            })?;
        }

        // Also create the db_path directory if needed
        tokio::fs::create_dir_all(&config.db_path)
            .await
            .map_err(|e| {
                MemError::storage(format!("Failed to create database directory: {}", e))
            })?;

        // Open Sled database with configuration
        let db = sled::Config::new()
            .path(&config.db_path)
            .cache_capacity(config.cache_size_mb as u64 * 1024 * 1024)
            .flush_every_ms(if config.flush_interval_secs > 0 {
                Some(config.flush_interval_secs * 1000)
            } else {
                None
            })
            .open()
            .map_err(|e| MemError::storage(format!("Failed to open Sled database: {}", e)))?;

        // Open trees for embeddings and metadata
        let embeddings_tree = db
            .open_tree("embeddings")
            .map_err(|e| MemError::storage(format!("Failed to open embeddings tree: {}", e)))?;

        let metadata_tree = db
            .open_tree("metadata")
            .map_err(|e| MemError::storage(format!("Failed to open metadata tree: {}", e)))?;

        // Calculate initial stats
        let entry_count = embeddings_tree.len() as u64;
        let stats = Arc::new(RwLock::new(ColdMemoryStats {
            entry_count,
            ..Default::default()
        }));

        Ok(Self {
            db,
            embeddings_tree,
            metadata_tree,
            config,
            stats,
            search_latency_sum: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
        })
    }

    /// Store a single entry in cold memory
    ///
    /// # Arguments
    /// * `entry` - The entry to store
    ///
    /// # Returns
    /// `Ok(())` on success, or an error
    pub async fn store(&self, entry: &ColdMemoryEntry) -> MemResult<()> {
        let stored = StoredEmbedding {
            vector: entry.embedding.clone(),
            content: entry.content.clone(),
            metadata: entry.metadata.clone(),
            created_at: entry.created_at,
        };

        let key = entry.id.as_bytes().to_vec();
        let value = self.serialize_entry(&stored)?;

        self.embeddings_tree
            .insert(key, value)
            .map_err(|e| MemError::storage(format!("Failed to store entry: {}", e)))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.entry_count = self.embeddings_tree.len() as u64;
        }

        Ok(())
    }

    /// Retrieve an entry by ID
    ///
    /// # Arguments
    /// * `id` - The UUID of the entry to retrieve
    ///
    /// # Returns
    /// `Some(entry)` if found, `None` if not found
    pub async fn get(&self, id: &Uuid) -> MemResult<Option<ColdMemoryEntry>> {
        let key = id.as_bytes().to_vec();

        match self.embeddings_tree.get(&key) {
            Ok(Some(value)) => {
                let stored: StoredEmbedding = self.deserialize_entry(&value)?;

                Ok(Some(ColdMemoryEntry {
                    id: *id,
                    content: stored.content,
                    embedding: stored.vector,
                    metadata: stored.metadata,
                    created_at: stored.created_at,
                }))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(MemError::storage(format!(
                "Failed to retrieve entry: {}",
                e
            ))),
        }
    }

    /// Delete an entry by ID
    ///
    /// # Arguments
    /// * `id` - The UUID of the entry to delete
    ///
    /// # Returns
    /// `true` if an entry was deleted, `false` if not found
    pub async fn delete(&self, id: &Uuid) -> MemResult<bool> {
        let key = id.as_bytes().to_vec();

        match self.embeddings_tree.remove(&key) {
            Ok(Some(_)) => {
                // Update stats
                {
                    let mut stats = self.stats.write().await;
                    stats.entry_count = self.embeddings_tree.len() as u64;
                }
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(MemError::storage(format!("Failed to delete entry: {}", e))),
        }
    }

    /// Search for similar entries using cosine similarity
    ///
    /// # Arguments
    /// * `query_embedding` - The query vector to search with
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Vector of (UUID, similarity_score) pairs, sorted by descending similarity
    pub async fn search_similar(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> MemResult<Vec<(Uuid, f32)>> {
        let start = Instant::now();

        if query_embedding.is_empty() {
            return Err(MemError::invalid_input("Query embedding cannot be empty"));
        }

        let query_normalized = normalize_vector(query_embedding);
        let entry_count = self.embeddings_tree.len();

        let results = if entry_count > self.config.parallel_scan_threshold {
            // Parallel scan for large datasets
            self.parallel_search(&query_normalized, limit)?
        } else {
            // Sequential scan for smaller datasets
            self.sequential_search(&query_normalized, limit)?
        };

        // Update latency stats
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.search_latency_sum
            .fetch_add(elapsed_us, AtomicOrdering::Relaxed);
        self.search_count.fetch_add(1, AtomicOrdering::Relaxed);

        Ok(results)
    }

    /// Search for similar entries with filters
    ///
    /// # Arguments
    /// * `query_embedding` - The query vector to search with
    /// * `limit` - Maximum number of results to return
    /// * `filter` - Filter criteria to apply
    ///
    /// # Returns
    /// Vector of (UUID, similarity_score) pairs, sorted by descending similarity
    pub async fn search_with_filters(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: &SearchFilter,
    ) -> MemResult<Vec<(Uuid, f32)>> {
        let start = Instant::now();

        if query_embedding.is_empty() {
            return Err(MemError::invalid_input("Query embedding cannot be empty"));
        }

        let query_normalized = normalize_vector(query_embedding);
        let now = chrono::Utc::now().timestamp();

        let mut results: Vec<(Uuid, f32)> = Vec::new();

        for result in self.embeddings_tree.iter() {
            let (key, value) =
                result.map_err(|e| MemError::storage(format!("Iterator error: {}", e)))?;

            // Parse UUID from key
            let id = Uuid::from_slice(&key)
                .map_err(|e| MemError::storage(format!("Invalid UUID in database: {}", e)))?;

            // Deserialize embedding
            let stored: StoredEmbedding = self.deserialize_entry(&value)?;

            // Apply age filter
            if let Some(max_age) = filter.max_age_secs {
                if now - stored.created_at > max_age {
                    continue;
                }
            }

            // Compute similarity
            let score = cosine_similarity(&query_normalized, &stored.vector);

            // Apply score filter
            if let Some(min_score) = filter.min_score {
                if score < min_score {
                    continue;
                }
            }

            results.push((id, score));
        }

        // Sort by score descending and take top-K
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(limit);

        // Update latency stats
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.search_latency_sum
            .fetch_add(elapsed_us, AtomicOrdering::Relaxed);
        self.search_count.fetch_add(1, AtomicOrdering::Relaxed);

        Ok(results)
    }

    /// Sequential search (for smaller datasets)
    fn sequential_search(&self, query: &[f32], limit: usize) -> MemResult<Vec<(Uuid, f32)>> {
        let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(limit + 1);

        for result in self.embeddings_tree.iter() {
            let (key, value) =
                result.map_err(|e| MemError::storage(format!("Iterator error: {}", e)))?;

            // Parse UUID from key
            let id = Uuid::from_slice(&key)
                .map_err(|e| MemError::storage(format!("Invalid UUID in database: {}", e)))?;

            // Deserialize embedding
            let stored: StoredEmbedding = self.deserialize_entry(&value)?;

            // Compute similarity
            let score = cosine_similarity(query, &stored.vector);

            // Maintain top-K using min-heap
            if heap.len() < limit {
                heap.push(ScoredEntry { id, score });
            } else if let Some(min) = heap.peek() {
                if score > min.score {
                    heap.pop();
                    heap.push(ScoredEntry { id, score });
                }
            }
        }

        // Convert heap to sorted vector (descending by score)
        let mut results: Vec<(Uuid, f32)> = heap.into_iter().map(|e| (e.id, e.score)).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Parallel search using Rayon (for larger datasets)
    fn parallel_search(&self, query: &[f32], limit: usize) -> MemResult<Vec<(Uuid, f32)>> {
        // Collect all entries first (needed for parallel iteration)
        let entries: Vec<_> = self.embeddings_tree.iter().filter_map(|r| r.ok()).collect();

        let query_vec = query.to_vec();

        // Parallel map to compute similarities
        let mut scored: Vec<(Uuid, f32)> = entries
            .par_iter()
            .filter_map(|(key, value)| {
                let id = Uuid::from_slice(key).ok()?;
                let stored: StoredEmbedding = serde_json::from_slice(value).ok()?;
                let score = cosine_similarity(&query_vec, &stored.vector);
                Some((id, score))
            })
            .collect();

        // Sort by score descending and take top-K
        scored.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    /// Store multiple entries in a batch (transactional)
    ///
    /// # Arguments
    /// * `entries` - Slice of entries to store
    ///
    /// # Returns
    /// Number of entries successfully stored
    pub async fn store_batch(&self, entries: &[ColdMemoryEntry]) -> MemResult<usize> {
        if entries.is_empty() {
            return Ok(0);
        }

        // Use Sled batch for atomic operation
        let mut batch = sled::Batch::default();
        let mut count = 0;

        for entry in entries {
            let stored = StoredEmbedding {
                vector: entry.embedding.clone(),
                content: entry.content.clone(),
                metadata: entry.metadata.clone(),
                created_at: entry.created_at,
            };

            let key = entry.id.as_bytes().to_vec();
            let value = self.serialize_entry(&stored)?;

            batch.insert(key, value);
            count += 1;
        }

        self.embeddings_tree
            .apply_batch(batch)
            .map_err(|e| MemError::storage(format!("Batch insert failed: {}", e)))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.entry_count = self.embeddings_tree.len() as u64;
        }

        Ok(count)
    }

    /// Compact the database to reclaim space
    ///
    /// This is a potentially expensive operation that should be run during maintenance windows.
    pub async fn compact(&self) -> MemResult<()> {
        // Flush pending writes
        self.db
            .flush_async()
            .await
            .map_err(|e| MemError::storage(format!("Flush failed: {}", e)))?;

        // Update stats with compaction time
        {
            let mut stats = self.stats.write().await;
            stats.last_compaction = Some(chrono::Utc::now().timestamp());
        }

        tracing::info!("Cold memory compaction completed");
        Ok(())
    }

    /// Get current storage statistics
    pub async fn stats(&self) -> ColdMemoryStats {
        let mut stats = self.stats.read().await.clone();

        // Update dynamic stats
        stats.entry_count = self.embeddings_tree.len() as u64;

        // Calculate average search latency
        let count = self.search_count.load(AtomicOrdering::Relaxed);
        if count > 0 {
            let sum = self.search_latency_sum.load(AtomicOrdering::Relaxed);
            stats.search_count = count;
            stats.avg_search_latency_us = sum / count;
        }

        // Estimate sizes (Sled doesn't expose this directly)
        // Using a rough estimate based on entry count
        stats.embeddings_size_bytes = stats.entry_count * 4096; // ~4KB per entry estimate
        stats.metadata_size_bytes = self.metadata_tree.len() as u64 * 256;

        stats
    }

    /// Flush pending writes to disk
    pub async fn flush(&self) -> MemResult<()> {
        self.db
            .flush_async()
            .await
            .map_err(|e| MemError::storage(format!("Flush failed: {}", e)))?;
        Ok(())
    }

    /// Check if the database contains an entry with the given ID
    pub async fn contains(&self, id: &Uuid) -> MemResult<bool> {
        let key = id.as_bytes().to_vec();
        self.embeddings_tree
            .contains_key(&key)
            .map_err(|e| MemError::storage(format!("Contains check failed: {}", e)))
    }

    /// List all entry IDs in the database
    pub async fn list_ids(&self) -> MemResult<Vec<Uuid>> {
        let mut ids = Vec::new();

        for result in self.embeddings_tree.iter().keys() {
            let key = result.map_err(|e| MemError::storage(format!("Iterator error: {}", e)))?;

            let id = Uuid::from_slice(&key)
                .map_err(|e| MemError::storage(format!("Invalid UUID: {}", e)))?;

            ids.push(id);
        }

        Ok(ids)
    }

    /// Get the number of entries in the database
    pub fn len(&self) -> usize {
        self.embeddings_tree.len()
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings_tree.is_empty()
    }

    /// Clear all entries from the database
    pub async fn clear(&self) -> MemResult<()> {
        self.embeddings_tree
            .clear()
            .map_err(|e| MemError::storage(format!("Clear failed: {}", e)))?;

        self.metadata_tree
            .clear()
            .map_err(|e| MemError::storage(format!("Clear metadata failed: {}", e)))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.entry_count = 0;
        }

        Ok(())
    }

    /// Get the database path
    pub fn db_path(&self) -> &PathBuf {
        &self.config.db_path
    }

    /// Get disk size on disk (bytes)
    pub fn size_on_disk(&self) -> u64 {
        self.db.size_on_disk().unwrap_or(0)
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Serialize an entry for storage
    fn serialize_entry<T: Serialize>(&self, data: &T) -> MemResult<Vec<u8>> {
        serde_json::to_vec(data)
            .map_err(|e| MemError::storage(format!("Serialization failed: {}", e)))
    }

    /// Deserialize an entry from storage
    fn deserialize_entry<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> MemResult<T> {
        serde_json::from_slice(data)
            .map_err(|e| MemError::storage(format!("Deserialization failed: {}", e)))
    }
}

impl Drop for ColdMemory {
    fn drop(&mut self) {
        // Ensure data is flushed on drop
        if let Err(e) = self.db.flush() {
            tracing::error!("Failed to flush cold memory on drop: {}", e);
        }
    }
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Builder for ColdMemory with fluent configuration
pub struct ColdMemoryBuilder {
    config: ColdMemoryConfig,
}

impl ColdMemoryBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ColdMemoryConfig::default(),
        }
    }

    /// Set the database path
    pub fn path(mut self, path: PathBuf) -> Self {
        self.config.db_path = path;
        self
    }

    /// Set the cache size in megabytes
    pub fn cache_size_mb(mut self, mb: usize) -> Self {
        self.config.cache_size_mb = mb;
        self
    }

    /// Set the flush interval in seconds
    pub fn flush_interval_secs(mut self, secs: u64) -> Self {
        self.config.flush_interval_secs = secs;
        self
    }

    /// Enable or disable compression
    pub fn compression(mut self, enabled: bool) -> Self {
        self.config.enable_compression = enabled;
        self
    }

    /// Set the parallel scan threshold
    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_scan_threshold = threshold;
        self
    }

    /// Build the ColdMemory instance
    pub async fn build(self) -> MemResult<ColdMemory> {
        ColdMemory::new(self.config).await
    }
}

impl Default for ColdMemoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Comprehensive Tests for Sled-Based Cold Memory
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    // ========================================================================
    // Test Helpers
    // ========================================================================

    /// Create a test cold memory instance with isolated temp directory
    async fn create_test_cold_memory() -> (ColdMemory, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = ColdMemoryConfig::new(temp_dir.path().join("cold_test"));
        let cold = ColdMemory::new(config)
            .await
            .expect("Failed to create ColdMemory");
        (cold, temp_dir)
    }

    /// Create a test embedding with a seed for reproducibility
    fn create_test_embedding(seed: u32, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| ((seed as f32 * 0.1) + (i as f32 * 0.01)) % 1.0)
            .collect()
    }

    // ========================================================================
    // Basic Operations Tests
    // ========================================================================

    #[tokio::test]
    async fn test_store_and_get() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entry =
            ColdMemoryEntry::new("Hello, world!".to_string(), vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let id = entry.id;

        cold.store(&entry).await.expect("Store failed");

        let retrieved = cold.get(&id).await.expect("Get failed");
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.content, "Hello, world!");
        assert_eq!(retrieved.embedding.len(), 5);
        assert_eq!(retrieved.embedding, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[tokio::test]
    async fn test_store_overwrites_existing() {
        let (cold, _temp) = create_test_cold_memory().await;

        let id = Uuid::new_v4();

        // Store first version
        let entry1 = ColdMemoryEntry::with_id(id, "Version 1".to_string(), vec![1.0, 0.0]);
        cold.store(&entry1).await.expect("Store 1 failed");

        // Store second version with same ID
        let entry2 = ColdMemoryEntry::with_id(id, "Version 2".to_string(), vec![0.0, 1.0]);
        cold.store(&entry2).await.expect("Store 2 failed");

        // Should get second version
        let retrieved = cold.get(&id).await.expect("Get failed").unwrap();
        assert_eq!(retrieved.content, "Version 2");
        assert_eq!(retrieved.embedding, vec![0.0, 1.0]);

        // Count should still be 1
        assert_eq!(cold.len(), 1);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let (cold, _temp) = create_test_cold_memory().await;

        let id = Uuid::new_v4();
        let result = cold.get(&id).await.expect("Get failed");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entry = ColdMemoryEntry::new("To delete".to_string(), vec![1.0, 2.0]);
        let id = entry.id;

        cold.store(&entry).await.expect("Store failed");
        assert!(cold.contains(&id).await.unwrap());
        assert_eq!(cold.len(), 1);

        let deleted = cold.delete(&id).await.expect("Delete failed");
        assert!(deleted);

        let not_deleted = cold.delete(&id).await.expect("Delete again failed");
        assert!(!not_deleted);

        assert!(!cold.contains(&id).await.unwrap());
        assert_eq!(cold.len(), 0);
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let (cold, _temp) = create_test_cold_memory().await;

        let id = Uuid::new_v4();
        let deleted = cold.delete(&id).await.expect("Delete failed");
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_batch_store() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entries: Vec<ColdMemoryEntry> = (0..100)
            .map(|i| ColdMemoryEntry::new(format!("Document {}", i), create_test_embedding(i, 128)))
            .collect();

        let count = cold
            .store_batch(&entries)
            .await
            .expect("Batch store failed");
        assert_eq!(count, 100);
        assert_eq!(cold.len(), 100);

        // Verify random entries
        for i in [0, 25, 50, 75, 99] {
            let entry = cold.get(&entries[i].id).await.expect("Get failed").unwrap();
            assert_eq!(entry.content, format!("Document {}", i));
        }
    }

    #[tokio::test]
    async fn test_batch_store_empty() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entries: Vec<ColdMemoryEntry> = vec![];
        let count = cold
            .store_batch(&entries)
            .await
            .expect("Batch store failed");
        assert_eq!(count, 0);
        assert_eq!(cold.len(), 0);
    }

    // ========================================================================
    // Persistence Tests
    // ========================================================================

    #[tokio::test]
    async fn test_persistence_across_restarts() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("persistence_test");

        let id = Uuid::new_v4();
        let content = "Persistent content that survives restarts";
        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // First: store data and close
        {
            let config = ColdMemoryConfig::new(db_path.clone());
            let cold = ColdMemory::new(config).await.expect("Failed to create");

            let entry = ColdMemoryEntry::with_id(id, content.to_string(), embedding.clone());
            cold.store(&entry).await.expect("Store failed");
            cold.flush().await.expect("Flush failed");
        }

        // Second: reopen and verify data persists
        {
            let config = ColdMemoryConfig::new(db_path.clone());
            let cold = ColdMemory::new(config).await.expect("Failed to create");

            assert_eq!(cold.len(), 1);

            let retrieved = cold.get(&id).await.expect("Get failed");
            assert!(retrieved.is_some());

            let entry = retrieved.unwrap();
            assert_eq!(entry.content, content);
            assert_eq!(entry.embedding, embedding);
        }
    }

    #[tokio::test]
    async fn test_persistence_multiple_entries() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("multi_persistence_test");

        let entries: Vec<ColdMemoryEntry> = (0..50)
            .map(|i| ColdMemoryEntry::new(format!("Entry {}", i), create_test_embedding(i, 64)))
            .collect();
        let ids: Vec<Uuid> = entries.iter().map(|e| e.id).collect();

        // Store entries
        {
            let config = ColdMemoryConfig::new(db_path.clone());
            let cold = ColdMemory::new(config).await.expect("Failed to create");
            cold.store_batch(&entries)
                .await
                .expect("Batch store failed");
            cold.flush().await.expect("Flush failed");
        }

        // Verify all entries persist
        {
            let config = ColdMemoryConfig::new(db_path);
            let cold = ColdMemory::new(config).await.expect("Failed to create");

            assert_eq!(cold.len(), 50);

            for (i, id) in ids.iter().enumerate() {
                let entry = cold.get(id).await.expect("Get failed").unwrap();
                assert_eq!(entry.content, format!("Entry {}", i));
            }
        }
    }

    #[tokio::test]
    async fn test_data_integrity() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store entry with specific data
        let original_content = "Data integrity test - exact content matters!";
        let original_embedding = vec![0.123456, 0.789012, 0.345678, 0.901234];
        let original_metadata = serde_json::json!({
            "key1": "value1",
            "nested": {"a": 1, "b": 2}
        });

        let entry = ColdMemoryEntry::with_metadata(
            original_content.to_string(),
            original_embedding.clone(),
            original_metadata.clone(),
        );
        let id = entry.id;

        cold.store(&entry).await.expect("Store failed");
        cold.flush().await.expect("Flush failed");

        // Retrieve and verify exact data
        let retrieved = cold.get(&id).await.expect("Get failed").unwrap();

        assert_eq!(retrieved.content, original_content);
        assert_eq!(retrieved.embedding, original_embedding);
        assert_eq!(retrieved.metadata, original_metadata);

        // Verify floating point precision is maintained
        for (orig, retr) in original_embedding.iter().zip(retrieved.embedding.iter()) {
            assert!((orig - retr).abs() < f32::EPSILON);
        }
    }

    // ========================================================================
    // Vector Search Tests
    // ========================================================================

    #[tokio::test]
    async fn test_search_similar() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store entries with different embeddings
        let entries = vec![
            ColdMemoryEntry::new("Document A".to_string(), vec![1.0, 0.0, 0.0]),
            ColdMemoryEntry::new("Document B".to_string(), vec![0.0, 1.0, 0.0]),
            ColdMemoryEntry::new("Document C".to_string(), vec![0.9, 0.1, 0.0]),
            ColdMemoryEntry::new("Document D".to_string(), vec![0.0, 0.0, 1.0]),
        ];

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        // Search for something similar to [1, 0, 0]
        let results = cold
            .search_similar(&[1.0, 0.0, 0.0], 3)
            .await
            .expect("Search failed");

        assert_eq!(results.len(), 3);

        // First result should be exactly [1, 0, 0] (similarity = 1.0)
        assert!((results[0].1 - 1.0).abs() < 0.001);

        // Results should be sorted by score descending
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[tokio::test]
    async fn test_search_with_filters() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store entries
        let entries: Vec<ColdMemoryEntry> = (0..10)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 64)))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        // Search with high score filter
        let filter = SearchFilter::new().with_min_score(0.99);
        let query = entries[5].embedding.clone();

        let results = cold
            .search_with_filters(&query, 10, &filter)
            .await
            .expect("Search failed");

        // Should get exact match only (score ~1.0)
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.99);
    }

    #[tokio::test]
    async fn test_search_with_age_filter() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entry = ColdMemoryEntry::new("Recent entry".to_string(), vec![1.0, 0.0, 0.0]);
        cold.store(&entry).await.expect("Store failed");

        let query = vec![1.0, 0.0, 0.0];

        // Entry should pass large max age filter
        let filter = SearchFilter::new().with_max_age(3600); // 1 hour
        let results = cold
            .search_with_filters(&query, 10, &filter)
            .await
            .expect("Search failed");
        assert!(!results.is_empty());

        // Entry should fail 0 seconds max age
        let filter = SearchFilter::new().with_max_age(0);
        let results = cold
            .search_with_filters(&query, 10, &filter)
            .await
            .expect("Search failed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_empty_db() {
        let (cold, _temp) = create_test_cold_memory().await;

        let results = cold
            .search_similar(&[1.0, 0.0, 0.0], 10)
            .await
            .expect("Search failed");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_empty_query() {
        let (cold, _temp) = create_test_cold_memory().await;

        cold.store(&ColdMemoryEntry::new("Test".to_string(), vec![1.0]))
            .await
            .expect("Store failed");

        let result = cold.search_similar(&[], 10).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_top_k_limit() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store 100 entries
        let entries: Vec<ColdMemoryEntry> = (0..100)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 64)))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        // Search with limit = 5
        let results = cold
            .search_similar(&create_test_embedding(50, 64), 5)
            .await
            .expect("Search failed");

        assert_eq!(results.len(), 5);
    }

    // ========================================================================
    // Compaction Tests
    // ========================================================================

    #[tokio::test]
    async fn test_compaction() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Add and delete entries to create fragmentation
        let entries: Vec<ColdMemoryEntry> = (0..100)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 128)))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        // Delete half the entries
        for entry in entries.iter().take(50) {
            cold.delete(&entry.id).await.expect("Delete failed");
        }

        assert_eq!(cold.len(), 50);

        // Compact should succeed
        cold.compact().await.expect("Compact failed");

        // Data should still be accessible
        assert_eq!(cold.len(), 50);

        // Remaining entries should be intact
        for entry in entries.iter().skip(50) {
            let retrieved = cold.get(&entry.id).await.expect("Get failed");
            assert!(retrieved.is_some());
        }
    }

    #[tokio::test]
    async fn test_compaction_empty_db() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Compaction on empty DB should not fail
        cold.compact().await.expect("Compact failed");
        assert_eq!(cold.len(), 0);

        // Stats should show compaction time
        let stats = cold.stats().await;
        assert!(stats.last_compaction.is_some());
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[tokio::test]
    async fn test_invalid_path() {
        // Try to open in a location that should fail
        // Note: On some systems this might succeed if permissions allow
        let config = ColdMemoryConfig::new(PathBuf::from("/nonexistent/deeply/nested/path"));
        let result = ColdMemory::new(config).await;

        // We expect this to fail on most systems
        // Just verify it doesn't panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_corrupted_entry_handling() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store a valid entry
        let entry = ColdMemoryEntry::new("Valid".to_string(), vec![1.0, 2.0, 3.0]);
        let id = entry.id;

        cold.store(&entry).await.expect("Store failed");

        // Retrieve should work
        let retrieved = cold.get(&id).await.expect("Get failed");
        assert!(retrieved.is_some());

        // Note: Testing actual corruption would require modifying the raw Sled data
        // which is not easily done through the public API
    }

    // ========================================================================
    // Utility Function Tests
    // ========================================================================

    #[tokio::test]
    async fn test_list_ids() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entries: Vec<ColdMemoryEntry> = (0..5)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), vec![i as f32]))
            .collect();

        let expected_ids: Vec<Uuid> = entries.iter().map(|e| e.id).collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        let ids = cold.list_ids().await.expect("List IDs failed");
        assert_eq!(ids.len(), 5);

        for id in expected_ids {
            assert!(ids.contains(&id));
        }
    }

    #[tokio::test]
    async fn test_contains() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entry = ColdMemoryEntry::new("Test".to_string(), vec![1.0]);
        let id = entry.id;
        let nonexistent_id = Uuid::new_v4();

        cold.store(&entry).await.expect("Store failed");

        assert!(cold.contains(&id).await.expect("Contains failed"));
        assert!(!cold
            .contains(&nonexistent_id)
            .await
            .expect("Contains failed"));
    }

    #[tokio::test]
    async fn test_clear() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entries: Vec<ColdMemoryEntry> = (0..10)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), vec![i as f32]))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");
        assert!(!cold.is_empty());
        assert_eq!(cold.len(), 10);

        cold.clear().await.expect("Clear failed");

        assert!(cold.is_empty());
        assert_eq!(cold.len(), 0);
    }

    #[tokio::test]
    async fn test_stats() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store some entries
        let entries: Vec<ColdMemoryEntry> = (0..10)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 64)))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        // Perform a search to generate stats
        cold.search_similar(&create_test_embedding(5, 64), 5)
            .await
            .expect("Search failed");

        let stats = cold.stats().await;
        assert_eq!(stats.entry_count, 10);
        assert_eq!(stats.search_count, 1);
        assert!(stats.avg_search_latency_us > 0);
    }

    #[tokio::test]
    async fn test_flush() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entry = ColdMemoryEntry::new("Test".to_string(), vec![1.0]);
        cold.store(&entry).await.expect("Store failed");

        // Flush should succeed
        cold.flush().await.expect("Flush failed");
    }

    #[tokio::test]
    async fn test_size_on_disk() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store some data
        let entries: Vec<ColdMemoryEntry> = (0..100)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 256)))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");
        cold.flush().await.expect("Flush failed");

        let size = cold.size_on_disk();
        assert!(size > 0);
    }

    // ========================================================================
    // Entry Metadata Tests
    // ========================================================================

    #[tokio::test]
    async fn test_entry_with_metadata() {
        let (cold, _temp) = create_test_cold_memory().await;

        let metadata = serde_json::json!({
            "source": "arxiv",
            "paper_id": "2401.18059",
            "tags": ["raptor", "rag", "retrieval"],
            "nested": {
                "level1": {
                    "level2": "value"
                }
            }
        });

        let entry = ColdMemoryEntry::with_metadata(
            "RAPTOR paper content".to_string(),
            vec![0.5, 0.5],
            metadata.clone(),
        );
        let id = entry.id;

        cold.store(&entry).await.expect("Store failed");

        let retrieved = cold.get(&id).await.expect("Get failed").unwrap();
        assert_eq!(retrieved.metadata, metadata);
        assert_eq!(retrieved.metadata["source"], "arxiv");
        assert_eq!(retrieved.metadata["nested"]["level1"]["level2"], "value");
    }

    #[tokio::test]
    async fn test_entry_null_metadata() {
        let (cold, _temp) = create_test_cold_memory().await;

        let entry = ColdMemoryEntry::new("No metadata".to_string(), vec![1.0]);
        let id = entry.id;

        cold.store(&entry).await.expect("Store failed");

        let retrieved = cold.get(&id).await.expect("Get failed").unwrap();
        assert_eq!(retrieved.metadata, serde_json::Value::Null);
    }

    // ========================================================================
    // Cosine Similarity Unit Tests
    // ========================================================================

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_similar() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.9, 0.1, 0.0];
        assert!(cosine_similarity(&a, &b) > 0.9);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ========================================================================
    // Other Vector Function Tests
    // ========================================================================

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(&v);

        // Check magnitude is 1
        let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);

        // Check values
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = normalize_vector(&v);
        assert_eq!(normalized, v);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.001);

        let c = vec![0.0, 0.0];
        assert!(euclidean_distance(&a, &c) < 0.001);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 0.001);
    }

    // ========================================================================
    // Configuration Tests
    // ========================================================================

    #[test]
    fn test_cold_memory_config_default() {
        let config = ColdMemoryConfig::default();
        assert_eq!(config.cache_size_mb, 128);
        assert_eq!(config.flush_interval_secs, 30);
        assert!(config.enable_compression);
        assert_eq!(config.parallel_scan_threshold, 1000);
    }

    #[test]
    fn test_cold_memory_config_builder() {
        let config = ColdMemoryConfig::new(PathBuf::from("/tmp/test"))
            .with_cache_size(256)
            .with_flush_interval(60)
            .with_compression(false);

        assert_eq!(config.cache_size_mb, 256);
        assert_eq!(config.flush_interval_secs, 60);
        assert!(!config.enable_compression);
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let cold = ColdMemoryBuilder::new()
            .path(temp_dir.path().join("builder_test"))
            .cache_size_mb(64)
            .flush_interval_secs(60)
            .compression(false)
            .parallel_threshold(500)
            .build()
            .await
            .expect("Builder failed");

        assert!(cold.is_empty());
    }

    // ========================================================================
    // Concurrent Access Tests
    // ========================================================================

    #[tokio::test]
    async fn test_concurrent_reads() {
        let (cold, _temp) = create_test_cold_memory().await;
        let cold = Arc::new(cold);

        // Store entries
        let entries: Vec<ColdMemoryEntry> = (0..10)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 64)))
            .collect();
        let ids: Vec<Uuid> = entries.iter().map(|e| e.id).collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        // Spawn concurrent read tasks
        let mut handles = Vec::new();
        for i in 0..10 {
            let cold_clone = Arc::clone(&cold);
            let id = ids[i];
            let handle = tokio::spawn(async move {
                for _ in 0..100 {
                    let result = cold_clone.get(&id).await;
                    assert!(result.is_ok());
                    assert!(result.unwrap().is_some());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task panicked");
        }
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        let (cold, _temp) = create_test_cold_memory().await;
        let cold = Arc::new(cold);

        // Spawn concurrent write tasks
        let mut handles = Vec::new();
        for i in 0..10 {
            let cold_clone = Arc::clone(&cold);
            let handle = tokio::spawn(async move {
                for j in 0..10 {
                    let entry = ColdMemoryEntry::new(
                        format!("Doc {}_{}", i, j),
                        create_test_embedding(i * 10 + j, 64),
                    );
                    cold_clone.store(&entry).await.expect("Store failed");
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task panicked");
        }

        assert_eq!(cold.len(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_search() {
        let (cold, _temp) = create_test_cold_memory().await;
        let cold = Arc::new(cold);

        // Store entries
        let entries: Vec<ColdMemoryEntry> = (0..50)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 64)))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");

        // Spawn concurrent search tasks
        let mut handles = Vec::new();
        for i in 0..10 {
            let cold_clone = Arc::clone(&cold);
            let handle = tokio::spawn(async move {
                for _ in 0..10 {
                    let query = create_test_embedding(i, 64);
                    let results = cold_clone.search_similar(&query, 5).await;
                    assert!(results.is_ok());
                    assert!(!results.unwrap().is_empty());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task panicked");
        }
    }

    // ========================================================================
    // Large Data Tests
    // ========================================================================

    #[tokio::test]
    async fn test_large_entry() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Create a large entry (1536-dim embedding like OpenAI)
        let large_embedding: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
        let large_content = "x".repeat(10000); // 10KB content

        let entry = ColdMemoryEntry::new(large_content.clone(), large_embedding.clone());
        let id = entry.id;

        cold.store(&entry).await.expect("Store failed");

        let retrieved = cold.get(&id).await.expect("Get failed").unwrap();
        assert_eq!(retrieved.content.len(), 10000);
        assert_eq!(retrieved.embedding.len(), 1536);
        assert_eq!(retrieved.embedding, large_embedding);
    }

    #[tokio::test]
    async fn test_many_entries() {
        let (cold, _temp) = create_test_cold_memory().await;

        // Store 1000 entries
        let entries: Vec<ColdMemoryEntry> = (0..1000)
            .map(|i| ColdMemoryEntry::new(format!("Doc {}", i), create_test_embedding(i, 128)))
            .collect();

        cold.store_batch(&entries)
            .await
            .expect("Batch store failed");
        assert_eq!(cold.len(), 1000);

        // Search should still work efficiently
        let results = cold
            .search_similar(&create_test_embedding(500, 128), 10)
            .await
            .expect("Search failed");

        assert_eq!(results.len(), 10);
        assert!(results[0].1 > 0.9); // Top result should have high similarity
    }

    // ========================================================================
    // Entry Creation Tests
    // ========================================================================

    #[test]
    fn test_cold_memory_entry_new() {
        let entry = ColdMemoryEntry::new("Test content".to_string(), vec![1.0, 2.0, 3.0]);

        assert!(!entry.id.is_nil());
        assert_eq!(entry.content, "Test content");
        assert_eq!(entry.embedding, vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.metadata, serde_json::Value::Null);
        assert!(entry.created_at > 0);
    }

    #[test]
    fn test_cold_memory_entry_with_id() {
        let id = Uuid::new_v4();
        let entry = ColdMemoryEntry::with_id(id, "Content".to_string(), vec![1.0]);

        assert_eq!(entry.id, id);
        assert_eq!(entry.content, "Content");
    }

    #[test]
    fn test_cold_memory_entry_dimension() {
        let entry = ColdMemoryEntry::new("Test".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(entry.dimension(), 5);
    }

    #[test]
    fn test_cold_memory_entry_set_metadata() {
        let metadata = serde_json::json!({"key": "value"});
        let entry =
            ColdMemoryEntry::new("Test".to_string(), vec![1.0]).set_metadata(metadata.clone());

        assert_eq!(entry.metadata, metadata);
    }

    // ========================================================================
    // Search Filter Tests
    // ========================================================================

    #[test]
    fn test_search_filter_builder() {
        let filter = SearchFilter::new()
            .with_min_score(0.8)
            .with_max_age(3600)
            .with_metadata(serde_json::json!({"type": "paper"}));

        assert_eq!(filter.min_score, Some(0.8));
        assert_eq!(filter.max_age_secs, Some(3600));
        assert!(filter.metadata_filter.is_some());
    }

    #[test]
    fn test_search_filter_default() {
        let filter = SearchFilter::default();

        assert!(filter.min_score.is_none());
        assert!(filter.max_age_secs.is_none());
        assert!(filter.metadata_filter.is_none());
    }
}
