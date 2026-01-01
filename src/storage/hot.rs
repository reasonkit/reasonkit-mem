//! Hot Memory Layer - High-performance in-memory cache with DashMap
//!
//! This module provides a thread-safe, concurrent hot memory layer for frequently
//! accessed entries. It uses DashMap for lock-free concurrent access and implements
//! TTL-based expiration with LRU eviction when capacity is exceeded.
//!
//! ## Features
//!
//! - **Thread-safe**: All operations are safe for concurrent access
//! - **TTL Expiration**: Entries automatically expire after configured duration
//! - **LRU Eviction**: Least recently used entries are evicted when at capacity
//! - **Vector Search**: Efficient cosine similarity search for embeddings
//! - **Statistics Tracking**: Real-time hit/miss rates and access patterns
//!
//! ## Example
//!
//! ```rust,ignore
//! use reasonkit_mem::storage::hot::{HotMemory, HotMemoryConfig, HotMemoryEntry};
//! use std::time::Duration;
//! use uuid::Uuid;
//!
//! let config = HotMemoryConfig {
//!     max_entries: 10_000,
//!     ttl: Duration::from_secs(3600),
//!     eviction_batch_size: 100,
//! };
//!
//! let hot_memory = HotMemory::new(config);
//!
//! // Store an entry
//! let entry = HotMemoryEntry::new(
//!     Uuid::new_v4(),
//!     "Sample content".to_string(),
//!     vec![0.1, 0.2, 0.3],
//!     serde_json::json!({"key": "value"}),
//! );
//! hot_memory.put(entry).await?;
//!
//! // Search for similar entries
//! let results = hot_memory.search_similar(&[0.1, 0.2, 0.3], 10).await;
//! ```

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::{MemError, MemResult};

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

/// Entry stored in hot memory
#[derive(Debug, Clone)]
pub struct HotMemoryEntry {
    /// Unique identifier for this entry
    pub id: Uuid,
    /// Text content of the entry
    pub content: String,
    /// Dense embedding vector for similarity search
    pub embedding: Vec<f32>,
    /// Arbitrary metadata associated with the entry
    pub metadata: serde_json::Value,
    /// When this entry was created
    pub created_at: Instant,
    /// When this entry was last accessed
    pub accessed_at: Instant,
    /// Number of times this entry has been accessed
    pub access_count: u64,
}

impl HotMemoryEntry {
    /// Create a new hot memory entry
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier
    /// * `content` - Text content
    /// * `embedding` - Dense vector embedding
    /// * `metadata` - Arbitrary JSON metadata
    pub fn new(
        id: Uuid,
        content: String,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            content,
            embedding,
            metadata,
            created_at: now,
            accessed_at: now,
            access_count: 0,
        }
    }

    /// Create an entry with a specific ID (useful for testing)
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }

    /// Check if this entry has expired based on TTL
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Update access timestamp and increment counter
    pub fn touch(&mut self) {
        self.accessed_at = Instant::now();
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Get the age of this entry since creation
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get the time since last access
    pub fn idle_time(&self) -> Duration {
        self.accessed_at.elapsed()
    }

    /// Estimate memory usage in bytes
    pub fn estimated_size(&self) -> usize {
        // UUID: 16 bytes
        // Content: string length + overhead
        // Embedding: f32 * count + vec overhead
        // Metadata: rough estimate from JSON
        // Timestamps and counters: ~32 bytes
        16 + self.content.len()
            + 24
            + (self.embedding.len() * 4)
            + 24
            + self.metadata.to_string().len()
            + 32
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for hot memory layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMemoryConfig {
    /// Maximum number of entries to store
    pub max_entries: usize,
    /// Time-to-live for entries
    #[serde(with = "duration_serde")]
    pub ttl: Duration,
    /// Number of entries to evict in a single batch
    pub eviction_batch_size: usize,
}

impl Default for HotMemoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl: Duration::from_secs(3600), // 1 hour
            eviction_batch_size: 100,
        }
    }
}

impl HotMemoryConfig {
    /// Create a config for high-throughput scenarios
    pub fn high_throughput() -> Self {
        Self {
            max_entries: 100_000,
            ttl: Duration::from_secs(1800), // 30 minutes
            eviction_batch_size: 500,
        }
    }

    /// Create a config for low-memory scenarios
    pub fn low_memory() -> Self {
        Self {
            max_entries: 1_000,
            ttl: Duration::from_secs(300), // 5 minutes
            eviction_batch_size: 50,
        }
    }

    /// Create a config with custom TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Create a config with custom max entries
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Create a config with custom eviction batch size
    pub fn with_eviction_batch_size(mut self, eviction_batch_size: usize) -> Self {
        self.eviction_batch_size = eviction_batch_size;
        self
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Statistics for hot memory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMemoryStats {
    /// Current number of entries
    pub entry_count: usize,
    /// Maximum capacity
    pub max_entries: usize,
    /// Total number of get operations
    pub total_gets: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Total number of put operations
    pub total_puts: u64,
    /// Total number of delete operations
    pub total_deletes: u64,
    /// Total number of evictions
    pub total_evictions: u64,
    /// Total number of expired entries removed
    pub total_expirations: u64,
    /// Hit rate as a percentage (0.0 - 1.0)
    pub hit_rate: f64,
    /// Average entry age in seconds
    pub avg_entry_age_secs: f64,
    /// Memory usage estimate in bytes
    pub estimated_memory_bytes: u64,
}

impl HotMemoryStats {
    /// Calculate hit rate
    pub fn calculate_hit_rate(&self) -> f64 {
        if self.total_gets == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_gets as f64
        }
    }
}

// ============================================================================
// INTERNAL ENTRY WRAPPER
// ============================================================================

/// Internal entry wrapper with LRU tracking
#[derive(Debug)]
struct InternalEntry {
    entry: HotMemoryEntry,
    /// Monotonically increasing counter for LRU ordering
    lru_counter: u64,
}

// ============================================================================
// HOT MEMORY IMPLEMENTATION
// ============================================================================

/// High-performance hot memory layer with concurrent access
///
/// Uses DashMap for lock-free concurrent reads and writes.
/// Implements TTL-based expiration and LRU eviction.
///
/// # Thread Safety
///
/// DashMap provides thread safety through sharding:
/// - The map is divided into N shards (default: 2 * num_cpus)
/// - Each shard has its own RwLock
/// - Read operations acquire a read lock on the relevant shard
/// - Write operations acquire a write lock on the relevant shard
/// - Different shards can be accessed concurrently
///
/// This means:
/// - Multiple readers can access different shards simultaneously
/// - Multiple readers can access the same shard simultaneously
/// - Writers block other writers to the same shard
/// - Writers block readers to the same shard
pub struct HotMemory {
    /// Concurrent hash map for entry storage (keyed by UUID)
    entries: DashMap<Uuid, InternalEntry>,
    /// Configuration
    config: HotMemoryConfig,
    /// Global LRU counter (monotonically increasing)
    lru_counter: AtomicU64,
    /// Statistics counters
    stats_gets: AtomicU64,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
    stats_puts: AtomicU64,
    stats_deletes: AtomicU64,
    stats_evictions: AtomicU64,
    stats_expirations: AtomicU64,
}

impl HotMemory {
    /// Create a new hot memory instance with the given configuration
    pub fn new(config: HotMemoryConfig) -> Self {
        Self {
            entries: DashMap::with_capacity(config.max_entries),
            config,
            lru_counter: AtomicU64::new(0),
            stats_gets: AtomicU64::new(0),
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
            stats_puts: AtomicU64::new(0),
            stats_deletes: AtomicU64::new(0),
            stats_evictions: AtomicU64::new(0),
            stats_expirations: AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(HotMemoryConfig::default())
    }

    /// Get an entry by ID
    ///
    /// Updates the access timestamp and LRU counter on hit.
    /// Returns None if the entry doesn't exist or has expired.
    pub async fn get(&self, id: &Uuid) -> Option<HotMemoryEntry> {
        self.stats_gets.fetch_add(1, Ordering::Relaxed);

        if let Some(mut entry_ref) = self.entries.get_mut(id) {
            // Check TTL expiration
            if entry_ref.entry.is_expired(self.config.ttl) {
                drop(entry_ref);
                self.entries.remove(id);
                self.stats_expirations.fetch_add(1, Ordering::Relaxed);
                self.stats_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update LRU counter and access time
            entry_ref.lru_counter = self.next_lru_counter();
            entry_ref.entry.touch();

            self.stats_hits.fetch_add(1, Ordering::Relaxed);
            Some(entry_ref.entry.clone())
        } else {
            self.stats_misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Put an entry into hot memory
    ///
    /// If capacity is exceeded, performs LRU eviction.
    pub async fn put(&self, entry: HotMemoryEntry) -> MemResult<()> {
        self.stats_puts.fetch_add(1, Ordering::Relaxed);

        // Check if we need to evict
        if self.entries.len() >= self.config.max_entries {
            self.evict_lru().await;
        }

        let internal = InternalEntry {
            entry,
            lru_counter: self.next_lru_counter(),
        };

        let id = internal.entry.id;
        self.entries.insert(id, internal);

        Ok(())
    }

    /// Put multiple entries in batch
    ///
    /// More efficient than multiple individual puts.
    pub async fn put_batch(&self, entries: Vec<HotMemoryEntry>) -> MemResult<usize> {
        let mut inserted = 0;

        for entry in entries {
            if self.entries.len() >= self.config.max_entries {
                self.evict_lru().await;
            }

            let internal = InternalEntry {
                entry,
                lru_counter: self.next_lru_counter(),
            };

            let id = internal.entry.id;
            self.entries.insert(id, internal);
            inserted += 1;
            self.stats_puts.fetch_add(1, Ordering::Relaxed);
        }

        Ok(inserted)
    }

    /// Delete an entry by ID
    ///
    /// Returns true if the entry existed and was removed.
    pub async fn delete(&self, id: &Uuid) -> MemResult<bool> {
        let removed = self.entries.remove(id).is_some();
        if removed {
            self.stats_deletes.fetch_add(1, Ordering::Relaxed);
        }
        Ok(removed)
    }

    /// Delete multiple entries by ID
    pub async fn delete_batch(&self, ids: &[Uuid]) -> MemResult<usize> {
        let mut deleted = 0;
        for id in ids {
            if self.entries.remove(id).is_some() {
                deleted += 1;
                self.stats_deletes.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(deleted)
    }

    /// Search for entries similar to the query embedding
    ///
    /// Uses cosine similarity for ranking.
    /// Returns entries sorted by similarity score (highest first).
    pub async fn search_similar(&self, query_embedding: &[f32], limit: usize) -> Vec<(Uuid, f32)> {
        let query_norm = vector_norm(query_embedding);
        if query_norm < f32::EPSILON {
            return Vec::new();
        }

        let mut results: Vec<(Uuid, f32)> = self
            .entries
            .iter()
            .filter(|entry| !entry.entry.is_expired(self.config.ttl))
            .map(|entry| {
                let score = cosine_similarity(query_embedding, &entry.entry.embedding, query_norm);
                (entry.entry.id, score)
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit results
        results.truncate(limit);

        results
    }

    /// Search with minimum score threshold
    pub async fn search_similar_with_threshold(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Vec<(Uuid, f32)> {
        let query_norm = vector_norm(query_embedding);
        if query_norm < f32::EPSILON {
            return Vec::new();
        }

        let mut results: Vec<(Uuid, f32)> = self
            .entries
            .iter()
            .filter(|entry| !entry.entry.is_expired(self.config.ttl))
            .filter_map(|entry| {
                let score = cosine_similarity(query_embedding, &entry.entry.embedding, query_norm);
                if score >= min_score {
                    Some((entry.entry.id, score))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Evict expired entries
    ///
    /// Returns the number of entries evicted.
    pub async fn evict_expired(&self) -> usize {
        let ttl = self.config.ttl;
        let expired_ids: Vec<Uuid> = self
            .entries
            .iter()
            .filter(|entry| entry.entry.is_expired(ttl))
            .map(|entry| entry.entry.id)
            .collect();

        let count = expired_ids.len();

        for id in expired_ids {
            self.entries.remove(&id);
        }

        self.stats_expirations
            .fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Evict least recently used entries
    ///
    /// Evicts `eviction_batch_size` entries with the lowest LRU counters.
    async fn evict_lru(&self) {
        let batch_size = self.config.eviction_batch_size;

        // Collect all entries with their LRU counters
        let mut entries: Vec<(Uuid, u64)> = self
            .entries
            .iter()
            .map(|entry| (entry.entry.id, entry.lru_counter))
            .collect();

        // Sort by LRU counter (oldest first)
        entries.sort_by_key(|e| e.1);

        // Evict the oldest entries
        let evict_count = entries.len().min(batch_size);
        for (id, _) in entries.into_iter().take(evict_count) {
            self.entries.remove(&id);
        }

        self.stats_evictions
            .fetch_add(evict_count as u64, Ordering::Relaxed);
    }

    /// Force eviction of a specific number of entries
    pub async fn force_evict(&self, count: usize) -> usize {
        let mut entries: Vec<(Uuid, u64)> = self
            .entries
            .iter()
            .map(|entry| (entry.entry.id, entry.lru_counter))
            .collect();

        entries.sort_by_key(|e| e.1);

        let evict_count = entries.len().min(count);
        for (id, _) in entries.into_iter().take(evict_count) {
            self.entries.remove(&id);
        }

        self.stats_evictions
            .fetch_add(evict_count as u64, Ordering::Relaxed);
        evict_count
    }

    /// Get statistics about the hot memory layer
    pub async fn stats(&self) -> HotMemoryStats {
        let entry_count = self.entries.len();
        let total_gets = self.stats_gets.load(Ordering::Relaxed);
        let cache_hits = self.stats_hits.load(Ordering::Relaxed);
        let cache_misses = self.stats_misses.load(Ordering::Relaxed);
        let total_puts = self.stats_puts.load(Ordering::Relaxed);
        let total_deletes = self.stats_deletes.load(Ordering::Relaxed);
        let total_evictions = self.stats_evictions.load(Ordering::Relaxed);
        let total_expirations = self.stats_expirations.load(Ordering::Relaxed);

        let hit_rate = if total_gets > 0 {
            cache_hits as f64 / total_gets as f64
        } else {
            0.0
        };

        // Calculate average entry age
        let total_age: f64 = self
            .entries
            .iter()
            .map(|e| e.entry.age().as_secs_f64())
            .sum();
        let avg_entry_age_secs = if entry_count > 0 {
            total_age / entry_count as f64
        } else {
            0.0
        };

        // Estimate memory usage
        let estimated_memory_bytes = self.estimate_memory_usage();

        HotMemoryStats {
            entry_count,
            max_entries: self.config.max_entries,
            total_gets,
            cache_hits,
            cache_misses,
            total_puts,
            total_deletes,
            total_evictions,
            total_expirations,
            hit_rate,
            avg_entry_age_secs,
            estimated_memory_bytes,
        }
    }

    /// Check if the cache contains an entry with the given ID
    pub fn contains(&self, id: &Uuid) -> bool {
        if let Some(entry) = self.entries.get(id) {
            !entry.entry.is_expired(self.config.ttl)
        } else {
            false
        }
    }

    /// Get the current number of entries (including potentially expired)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub async fn clear(&self) {
        self.entries.clear();
    }

    /// Get all entry IDs (non-expired only)
    pub fn entry_ids(&self) -> Vec<Uuid> {
        let ttl = self.config.ttl;
        self.entries
            .iter()
            .filter(|e| !e.entry.is_expired(ttl))
            .map(|e| e.entry.id)
            .collect()
    }

    /// Get all entries (non-expired only)
    pub fn all_entries(&self) -> Vec<HotMemoryEntry> {
        let ttl = self.config.ttl;
        self.entries
            .iter()
            .filter(|e| !e.entry.is_expired(ttl))
            .map(|e| e.entry.clone())
            .collect()
    }

    /// Get the configuration
    pub fn config(&self) -> &HotMemoryConfig {
        &self.config
    }

    /// Generate the next LRU counter value
    fn next_lru_counter(&self) -> u64 {
        self.lru_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self) -> u64 {
        let mut total: u64 = 0;

        for entry_ref in self.entries.iter() {
            total += entry_ref.entry.estimated_size() as u64;

            // InternalEntry overhead: ~16 bytes for lru_counter
            total += 16;
        }

        // DashMap overhead estimate: ~64 bytes per entry
        total += (self.entries.len() * 64) as u64;

        total
    }

    /// Get entry by ID without updating access stats (peek operation)
    pub fn peek(&self, id: &Uuid) -> Option<HotMemoryEntry> {
        self.entries.get(id).and_then(|entry| {
            if entry.entry.is_expired(self.config.ttl) {
                None
            } else {
                Some(entry.entry.clone())
            }
        })
    }

    /// Update an existing entry's metadata
    pub async fn update_metadata(&self, id: &Uuid, metadata: serde_json::Value) -> MemResult<bool> {
        if let Some(mut entry_ref) = self.entries.get_mut(id) {
            entry_ref.entry.metadata = metadata;
            entry_ref.entry.touch();
            entry_ref.lru_counter = self.next_lru_counter();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get entries matching a predicate
    pub fn find<F>(&self, predicate: F) -> Vec<HotMemoryEntry>
    where
        F: Fn(&HotMemoryEntry) -> bool,
    {
        let ttl = self.config.ttl;
        self.entries
            .iter()
            .filter(|e| !e.entry.is_expired(ttl) && predicate(&e.entry))
            .map(|e| e.entry.clone())
            .collect()
    }

    /// Get the oldest entries by creation time
    pub fn oldest(&self, count: usize) -> Vec<HotMemoryEntry> {
        let mut entries: Vec<_> = self.all_entries();
        entries.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        entries.truncate(count);
        entries
    }

    /// Get the most recently accessed entries
    pub fn most_recent(&self, count: usize) -> Vec<HotMemoryEntry> {
        let mut entries: Vec<_> = self.all_entries();
        entries.sort_by(|a, b| b.accessed_at.cmp(&a.accessed_at));
        entries.truncate(count);
        entries
    }

    /// Get the most frequently accessed entries
    pub fn most_accessed(&self, count: usize) -> Vec<HotMemoryEntry> {
        let mut entries: Vec<_> = self.all_entries();
        entries.sort_by(|a, b| b.access_count.cmp(&a.access_count));
        entries.truncate(count);
        entries
    }
}

impl Default for HotMemory {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// HotMemory is Send + Sync because DashMap is thread-safe
// and all atomic counters use atomic operations
// HotMemory is automatically Send + Sync due to DashMap and Atomics
// unsafe impl Send for HotMemory {}
// unsafe impl Sync for HotMemory {}

// ============================================================================
// VECTOR OPERATIONS
// ============================================================================

/// Calculate the L2 norm of a vector
#[inline]
fn vector_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Calculate cosine similarity between two vectors
///
/// Optimized version that takes pre-computed query norm.
#[inline]
fn cosine_similarity(query: &[f32], target: &[f32], query_norm: f32) -> f32 {
    if query.len() != target.len() || query.is_empty() {
        return 0.0;
    }

    let target_norm = vector_norm(target);
    if target_norm < f32::EPSILON {
        return 0.0;
    }

    let dot_product: f32 = query.iter().zip(target.iter()).map(|(a, b)| a * b).sum();

    dot_product / (query_norm * target_norm)
}

/// Calculate dot product between two vectors
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalize a vector to unit length
#[inline]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = vector_norm(v);
    if norm < f32::EPSILON {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

// ============================================================================
// SERDE HELPERS
// ============================================================================

/// Serde helper for Duration
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ============================================================================
    // Helper Functions
    // ============================================================================

    fn make_entry(content: &str, embedding: Vec<f32>) -> HotMemoryEntry {
        HotMemoryEntry::new(
            Uuid::new_v4(),
            content.to_string(),
            embedding,
            serde_json::json!({"test": true}),
        )
    }

    fn make_normalized_embedding(dim: usize, seed: f32) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect();
        normalize(&v)
    }

    // ============================================================================
    // Basic Operations Tests
    // ============================================================================

    #[tokio::test]
    async fn test_put_and_get() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("test content", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        let retrieved = hot_memory.get(&id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "test content");
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let hot_memory = HotMemory::with_defaults();
        let id = Uuid::new_v4();

        let retrieved = hot_memory.get(&id).await;
        assert!(retrieved.is_none());

        // Verify miss is recorded
        let stats = hot_memory.stats().await;
        assert_eq!(stats.cache_misses, 1);
    }

    #[tokio::test]
    async fn test_delete() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("to delete", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();
        assert!(hot_memory.contains(&id));

        let deleted = hot_memory.delete(&id).await.unwrap();
        assert!(deleted);
        assert!(!hot_memory.contains(&id));

        // Double delete should return false
        let deleted_again = hot_memory.delete(&id).await.unwrap();
        assert!(!deleted_again);

        let stats = hot_memory.stats().await;
        assert_eq!(stats.total_deletes, 1);
    }

    #[tokio::test]
    async fn test_put_update_existing() {
        let hot_memory = HotMemory::with_defaults();

        let id = Uuid::new_v4();
        let entry1 = HotMemoryEntry::new(
            id,
            "original".to_string(),
            vec![0.1, 0.2, 0.3],
            serde_json::json!({"version": 1}),
        );

        hot_memory.put(entry1).await.unwrap();
        assert_eq!(hot_memory.len(), 1);

        let entry2 = HotMemoryEntry::new(
            id,
            "updated".to_string(),
            vec![0.4, 0.5, 0.6],
            serde_json::json!({"version": 2}),
        );

        hot_memory.put(entry2).await.unwrap();
        assert_eq!(hot_memory.len(), 1); // Still only one entry

        let retrieved = hot_memory.get(&id).await.unwrap();
        assert_eq!(retrieved.content, "updated");
    }

    #[tokio::test]
    async fn test_contains() {
        let hot_memory = HotMemory::with_defaults();
        let id = Uuid::new_v4();

        assert!(!hot_memory.contains(&id));

        let entry = HotMemoryEntry::new(
            id,
            "test".to_string(),
            vec![0.1, 0.2, 0.3],
            serde_json::json!({}),
        );
        hot_memory.put(entry).await.unwrap();
        assert!(hot_memory.contains(&id));

        hot_memory.delete(&id).await.unwrap();
        assert!(!hot_memory.contains(&id));
    }

    #[tokio::test]
    async fn test_clear() {
        let hot_memory = HotMemory::with_defaults();

        for i in 0..10 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            hot_memory.put(entry).await.unwrap();
        }

        assert_eq!(hot_memory.len(), 10);

        hot_memory.clear().await;
        assert!(hot_memory.is_empty());
    }

    // ============================================================================
    // Concurrency Tests
    // ============================================================================

    #[tokio::test]
    async fn test_concurrent_reads() {
        let hot_memory = Arc::new(HotMemory::with_defaults());

        // Add one entry that will be read concurrently
        let id = Uuid::new_v4();
        let entry = HotMemoryEntry::new(
            id,
            "shared content".to_string(),
            vec![0.1, 0.2, 0.3],
            serde_json::json!({}),
        );
        hot_memory.put(entry).await.unwrap();

        let mut handles = vec![];

        // Spawn 10 tasks, each doing 100 concurrent reads
        for _ in 0..10 {
            let hm = Arc::clone(&hot_memory);
            let handle = tokio::spawn(async move {
                for _ in 0..100 {
                    let result = hm.get(&id).await;
                    assert!(result.is_some());
                    assert_eq!(result.unwrap().content, "shared content");
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all hits were recorded
        let stats = hot_memory.stats().await;
        assert_eq!(stats.cache_hits, 1000);
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        let hot_memory = Arc::new(HotMemory::new(HotMemoryConfig {
            max_entries: 10000,
            ..Default::default()
        }));

        let mut handles = vec![];

        // Spawn 10 tasks, each writing 100 entries
        for t in 0..10 {
            let hm = Arc::clone(&hot_memory);
            let handle = tokio::spawn(async move {
                for i in 0..100 {
                    let entry = make_entry(
                        &format!("entry {}:{}", t, i),
                        vec![t as f32 / 10.0, i as f32 / 100.0, 0.1],
                    );
                    hm.put(entry).await.unwrap();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(hot_memory.len(), 1000);
        let stats = hot_memory.stats().await;
        assert_eq!(stats.total_puts, 1000);
    }

    #[tokio::test]
    async fn test_read_while_write() {
        let hot_memory = Arc::new(HotMemory::new(HotMemoryConfig {
            max_entries: 10000,
            ..Default::default()
        }));

        // Pre-populate with entries
        let mut known_ids = Vec::new();
        for i in 0..100 {
            let entry = make_entry(&format!("initial {}", i), vec![0.1, 0.2, 0.3]);
            known_ids.push(entry.id);
            hot_memory.put(entry).await.unwrap();
        }

        let hm_write = Arc::clone(&hot_memory);
        let hm_read = Arc::clone(&hot_memory);
        let read_ids = known_ids.clone();

        // Writer task
        let writer = tokio::spawn(async move {
            for i in 0..500 {
                let entry = make_entry(&format!("new {}", i), vec![0.4, 0.5, 0.6]);
                hm_write.put(entry).await.unwrap();
            }
        });

        // Reader task
        let reader = tokio::spawn(async move {
            let mut successful_reads = 0;
            for _ in 0..10 {
                for id in &read_ids {
                    if hm_read.get(id).await.is_some() {
                        successful_reads += 1;
                    }
                }
            }
            successful_reads
        });

        writer.await.unwrap();
        let reads = reader.await.unwrap();

        // Should have successfully read many of the pre-populated entries
        assert!(reads > 0);
        assert_eq!(hot_memory.len(), 600); // 100 initial + 500 new
    }

    #[tokio::test]
    async fn test_concurrent_mixed_operations() {
        let hot_memory = Arc::new(HotMemory::new(HotMemoryConfig {
            max_entries: 1000,
            eviction_batch_size: 50,
            ..Default::default()
        }));

        let shared_ids: Arc<tokio::sync::Mutex<Vec<Uuid>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::new()));

        let mut handles = vec![];

        // Writers
        for t in 0..5 {
            let hm = Arc::clone(&hot_memory);
            let ids = Arc::clone(&shared_ids);
            handles.push(tokio::spawn(async move {
                for i in 0..50 {
                    let entry = make_entry(
                        &format!("entry {}:{}", t, i),
                        vec![t as f32 / 10.0, i as f32 / 100.0],
                    );
                    let id = entry.id;
                    hm.put(entry).await.unwrap();
                    ids.lock().await.push(id);
                }
            }));
        }

        // Readers
        for _ in 0..5 {
            let hm = Arc::clone(&hot_memory);
            let ids = Arc::clone(&shared_ids);
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    let ids_lock = ids.lock().await;
                    if !ids_lock.is_empty() {
                        let idx = (std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .subsec_nanos() as usize)
                            % ids_lock.len();
                        let id = ids_lock[idx];
                        drop(ids_lock);
                        let _ = hm.get(&id).await;
                    }
                    tokio::time::sleep(Duration::from_micros(10)).await;
                }
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Verify consistency
        let ids = shared_ids.lock().await;
        assert_eq!(ids.len(), 250); // 5 writers * 50 entries each
    }

    // ============================================================================
    // TTL and Expiration Tests
    // ============================================================================

    #[tokio::test]
    async fn test_ttl_expiration() {
        let config = HotMemoryConfig {
            ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let hot_memory = HotMemory::new(config);

        let entry = make_entry("will expire", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        // Entry should be retrievable immediately
        assert!(hot_memory.get(&id).await.is_some());

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Entry should be expired now (get will return None and remove it)
        assert!(hot_memory.get(&id).await.is_none());

        // Verify expiration was recorded
        let stats = hot_memory.stats().await;
        assert_eq!(stats.total_expirations, 1);
    }

    #[tokio::test]
    async fn test_no_ttl_expiration_with_long_ttl() {
        let config = HotMemoryConfig {
            ttl: Duration::from_secs(3600), // 1 hour
            ..Default::default()
        };
        let hot_memory = HotMemory::new(config);

        let entry = make_entry("persistent", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        // Wait a bit (but far less than TTL)
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Entry should still be available
        assert!(hot_memory.get(&id).await.is_some());
    }

    #[tokio::test]
    async fn test_evict_expired() {
        let config = HotMemoryConfig {
            ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let hot_memory = HotMemory::new(config);

        // Add some entries
        for i in 0..10 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            hot_memory.put(entry).await.unwrap();
        }

        assert_eq!(hot_memory.len(), 10);

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Evict expired entries
        let evicted = hot_memory.evict_expired().await;
        assert_eq!(evicted, 10);
        assert_eq!(hot_memory.len(), 0);
    }

    #[tokio::test]
    async fn test_mixed_expiration() {
        let config = HotMemoryConfig {
            ttl: Duration::from_millis(100),
            ..Default::default()
        };
        let hot_memory = HotMemory::new(config);

        // Add first batch
        let mut first_ids = Vec::new();
        for i in 0..5 {
            let entry = make_entry(&format!("first {}", i), vec![0.1, 0.2, 0.3]);
            first_ids.push(entry.id);
            hot_memory.put(entry).await.unwrap();
        }

        // Wait for partial expiration
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Add second batch (these should not expire yet)
        let mut second_ids = Vec::new();
        for i in 0..5 {
            let entry = make_entry(&format!("second {}", i), vec![0.4, 0.5, 0.6]);
            second_ids.push(entry.id);
            hot_memory.put(entry).await.unwrap();
        }

        // Wait for first batch to fully expire
        tokio::time::sleep(Duration::from_millis(50)).await;

        // First batch should be expired
        for id in &first_ids {
            assert!(hot_memory.get(id).await.is_none());
        }

        // Second batch should still be valid
        for id in &second_ids {
            assert!(hot_memory.get(id).await.is_some());
        }
    }

    // ============================================================================
    // LRU Eviction Tests
    // ============================================================================

    #[tokio::test]
    async fn test_lru_eviction() {
        let config = HotMemoryConfig {
            max_entries: 5,
            eviction_batch_size: 2,
            ..Default::default()
        };
        let hot_memory = HotMemory::new(config);

        // Add 5 entries (at capacity)
        let mut ids = Vec::new();
        for i in 0..5 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            ids.push(entry.id);
            hot_memory.put(entry).await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await; // Ensure LRU ordering
        }

        assert_eq!(hot_memory.len(), 5);

        // Access some entries to update their LRU status
        hot_memory.get(&ids[3]).await;
        hot_memory.get(&ids[4]).await;

        // Add one more - should trigger eviction
        let new_entry = make_entry("new entry", vec![0.1, 0.2, 0.3]);
        hot_memory.put(new_entry).await.unwrap();

        // Should have evicted 2 entries and added 1
        assert!(hot_memory.len() <= 5);

        // Evictions should be recorded
        let stats = hot_memory.stats().await;
        assert!(stats.total_evictions > 0);
    }

    #[tokio::test]
    async fn test_eviction_at_capacity() {
        let config = HotMemoryConfig {
            max_entries: 10,
            eviction_batch_size: 3,
            ..Default::default()
        };
        let hot_memory = HotMemory::new(config);

        // Fill to capacity
        for i in 0..10 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            hot_memory.put(entry).await.unwrap();
        }
        assert_eq!(hot_memory.len(), 10);

        // Add more entries - should trigger eviction
        for i in 0..5 {
            let entry = make_entry(&format!("new {}", i), vec![0.4, 0.5, 0.6]);
            hot_memory.put(entry).await.unwrap();
        }

        // Should never exceed max_entries
        assert!(hot_memory.len() <= 10);

        // Evictions should have occurred
        let stats = hot_memory.stats().await;
        assert!(stats.total_evictions >= 5);
    }

    #[tokio::test]
    async fn test_force_evict() {
        let hot_memory = HotMemory::with_defaults();

        // Add entries
        for i in 0..50 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            hot_memory.put(entry).await.unwrap();
        }
        assert_eq!(hot_memory.len(), 50);

        // Force evict 20 entries
        let evicted = hot_memory.force_evict(20).await;
        assert_eq!(evicted, 20);
        assert_eq!(hot_memory.len(), 30);
    }

    // ============================================================================
    // Similarity Search Tests
    // ============================================================================

    #[tokio::test]
    async fn test_search_similar_basic() {
        let hot_memory = HotMemory::with_defaults();

        // Add entries with different embeddings
        let entry1 = make_entry("similar to query", vec![0.9, 0.1, 0.0]);
        let entry2 = make_entry("somewhat similar", vec![0.5, 0.5, 0.0]);
        let entry3 = make_entry("very different", vec![0.0, 0.0, 1.0]);

        hot_memory.put(entry1.clone()).await.unwrap();
        hot_memory.put(entry2.clone()).await.unwrap();
        hot_memory.put(entry3.clone()).await.unwrap();

        // Query similar to entry1
        let query = vec![1.0, 0.0, 0.0];
        let results = hot_memory.search_similar(&query, 10).await;

        assert_eq!(results.len(), 3);
        // entry1 should be most similar
        assert_eq!(results[0].0, entry1.id);
        // entry3 should be least similar
        assert_eq!(results[2].0, entry3.id);
        // Results should be sorted by similarity
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[tokio::test]
    async fn test_search_similar_empty() {
        let hot_memory = HotMemory::with_defaults();

        let query = vec![1.0, 0.0, 0.0];
        let results = hot_memory.search_similar(&query, 10).await;

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_similar_top_k() {
        let hot_memory = HotMemory::with_defaults();

        // Add 100 entries
        for i in 0..100 {
            let embedding = make_normalized_embedding(128, i as f32);
            let entry = make_entry(&format!("entry {}", i), embedding);
            hot_memory.put(entry).await.unwrap();
        }

        let query = make_normalized_embedding(128, 50.0);

        // Request top 5
        let results = hot_memory.search_similar(&query, 5).await;
        assert_eq!(results.len(), 5);

        // Request more than available
        let results = hot_memory.search_similar(&query, 1000).await;
        assert_eq!(results.len(), 100);
    }

    #[tokio::test]
    async fn test_search_with_threshold() {
        let hot_memory = HotMemory::with_defaults();

        let entry1 = make_entry("very similar", vec![1.0, 0.0, 0.0]);
        let entry2 = make_entry("not similar", vec![0.0, 1.0, 0.0]);

        hot_memory.put(entry1.clone()).await.unwrap();
        hot_memory.put(entry2.clone()).await.unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = hot_memory
            .search_similar_with_threshold(&query, 10, 0.9)
            .await;

        // Only entry1 should match the threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, entry1.id);
    }

    #[tokio::test]
    async fn test_search_similar_identical_vectors() {
        let hot_memory = HotMemory::with_defaults();

        let embedding = vec![0.6, 0.8, 0.0]; // Already normalized
        let entry = make_entry("identical", embedding.clone());
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        // Search with identical vector
        let results = hot_memory.search_similar(&embedding, 1).await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        // Similarity should be very close to 1.0
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_search_similar_empty_query() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        hot_memory.put(entry).await.unwrap();

        // Empty query should return empty results
        let results = hot_memory.search_similar(&[], 10).await;
        assert!(results.is_empty());

        // Zero vector query should return empty results
        let results = hot_memory.search_similar(&[0.0, 0.0, 0.0], 10).await;
        assert!(results.is_empty());
    }

    // ============================================================================
    // Statistics Tests
    // ============================================================================

    #[tokio::test]
    async fn test_stats_tracking() {
        let hot_memory = HotMemory::with_defaults();

        // Initial stats should be zero
        let initial_stats = hot_memory.stats().await;
        assert_eq!(initial_stats.cache_hits, 0);
        assert_eq!(initial_stats.cache_misses, 0);
        assert_eq!(initial_stats.total_puts, 0);

        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();
        hot_memory.get(&id).await; // Hit
        hot_memory.get(&Uuid::new_v4()).await; // Miss
        hot_memory.delete(&id).await.unwrap();

        let stats = hot_memory.stats().await;
        assert_eq!(stats.total_puts, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.total_deletes, 1);
    }

    #[tokio::test]
    async fn test_hit_miss_ratio() {
        let hot_memory = HotMemory::with_defaults();

        // No operations - ratio should be 0
        let stats = hot_memory.stats().await;
        assert_eq!(stats.hit_rate, 0.0);

        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        // 3 hits, 1 miss = 0.75 ratio
        hot_memory.get(&id).await;
        hot_memory.get(&id).await;
        hot_memory.get(&id).await;
        hot_memory.get(&Uuid::new_v4()).await; // Miss

        let stats = hot_memory.stats().await;
        assert!((stats.hit_rate - 0.75).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_stats() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();
        hot_memory.get(&id).await;
        hot_memory.get(&id).await;
        hot_memory.get(&Uuid::new_v4()).await; // Miss

        let stats = hot_memory.stats().await;
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.total_puts, 1);
        assert_eq!(stats.total_gets, 3);
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.hit_rate - 0.6666).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_stats_memory_estimation() {
        let hot_memory = HotMemory::with_defaults();

        // Add entries with known sizes
        for i in 0..10 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1; 100]); // 100 floats
            hot_memory.put(entry).await.unwrap();
        }

        let stats = hot_memory.stats().await;
        assert!(stats.estimated_memory_bytes > 0);
        // Each entry should contribute at least the embedding size
        assert!(stats.estimated_memory_bytes > 10 * 100 * 4); // 10 entries * 100 floats * 4 bytes
    }

    // ============================================================================
    // Batch Operations Tests
    // ============================================================================

    #[tokio::test]
    async fn test_batch_operations() {
        let hot_memory = HotMemory::with_defaults();

        let entries: Vec<HotMemoryEntry> = (0..10)
            .map(|i| make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]))
            .collect();

        let ids: Vec<Uuid> = entries.iter().map(|e| e.id).collect();

        let inserted = hot_memory.put_batch(entries).await.unwrap();
        assert_eq!(inserted, 10);
        assert_eq!(hot_memory.len(), 10);

        let deleted = hot_memory.delete_batch(&ids[0..5]).await.unwrap();
        assert_eq!(deleted, 5);
        assert_eq!(hot_memory.len(), 5);
    }

    // ============================================================================
    // Access Pattern Tests
    // ============================================================================

    #[tokio::test]
    async fn test_access_count_updates() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        // Access multiple times
        for _ in 0..5 {
            hot_memory.get(&id).await;
        }

        let retrieved = hot_memory.get(&id).await.unwrap();
        assert_eq!(retrieved.access_count, 6); // 5 previous + 1 current
    }

    #[tokio::test]
    async fn test_peek_does_not_update_stats() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        // Peek should not update access count
        let _ = hot_memory.peek(&id);
        let _ = hot_memory.peek(&id);

        let retrieved = hot_memory.get(&id).await.unwrap();
        assert_eq!(retrieved.access_count, 1); // Only the get() should count
    }

    #[tokio::test]
    async fn test_update_metadata() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        let new_metadata = serde_json::json!({"updated": true, "version": 2});
        let updated = hot_memory
            .update_metadata(&id, new_metadata.clone())
            .await
            .unwrap();
        assert!(updated);

        let retrieved = hot_memory.get(&id).await.unwrap();
        assert_eq!(retrieved.metadata, new_metadata);
    }

    #[tokio::test]
    async fn test_most_recent_and_oldest() {
        let hot_memory = HotMemory::with_defaults();

        // Add entries with small delays to ensure ordering
        for i in 0..5 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            hot_memory.put(entry).await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let oldest = hot_memory.oldest(2);
        assert_eq!(oldest.len(), 2);
        assert!(oldest[0].content.contains("0"));
        assert!(oldest[1].content.contains("1"));

        // Access entries in reverse order to update accessed_at
        let ids = hot_memory.entry_ids();
        for id in ids.iter().rev() {
            hot_memory.get(id).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let most_recent = hot_memory.most_recent(2);
        assert_eq!(most_recent.len(), 2);
    }

    #[tokio::test]
    async fn test_most_accessed() {
        let hot_memory = HotMemory::with_defaults();

        let entry1 = make_entry("rarely accessed", vec![0.1, 0.2, 0.3]);
        let entry2 = make_entry("frequently accessed", vec![0.4, 0.5, 0.6]);
        let id2 = entry2.id;

        hot_memory.put(entry1).await.unwrap();
        hot_memory.put(entry2).await.unwrap();

        // Access entry2 multiple times
        for _ in 0..10 {
            hot_memory.get(&id2).await;
        }

        let most_accessed = hot_memory.most_accessed(1);
        assert_eq!(most_accessed.len(), 1);
        assert_eq!(most_accessed[0].content, "frequently accessed");
    }

    // ============================================================================
    // Find/Filter Tests
    // ============================================================================

    #[tokio::test]
    async fn test_find() {
        let hot_memory = HotMemory::with_defaults();

        for i in 0..10 {
            let mut entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            entry.metadata = serde_json::json!({"index": i, "even": i % 2 == 0});
            hot_memory.put(entry).await.unwrap();
        }

        let even_entries = hot_memory.find(|e| {
            e.metadata
                .get("even")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        });

        assert_eq!(even_entries.len(), 5);
    }

    // ============================================================================
    // Vector Operations Tests
    // ============================================================================

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];

        let norm_a = vector_norm(&a);

        // Same vector = 1.0
        assert!((cosine_similarity(&a, &b, norm_a) - 1.0).abs() < 0.0001);

        // Orthogonal vectors = 0.0
        assert!((cosine_similarity(&a, &c, norm_a)).abs() < 0.0001);

        // Opposite vectors = -1.0
        assert!((cosine_similarity(&a, &d, norm_a) + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let norm_a = vector_norm(&a);

        // Different lengths should return 0.0
        assert_eq!(cosine_similarity(&a, &b, norm_a), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let norm_a = vector_norm(&a);

        // Zero vector should return 0.0
        assert_eq!(cosine_similarity(&a, &b, norm_a), 0.0);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = normalize(&v);
        assert!((normalized[0] - 0.6).abs() < 0.0001);
        assert!((normalized[1] - 0.8).abs() < 0.0001);

        // Check unit length
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = normalize(&v);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 0.0001); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(dot_product(&a, &b), 0.0);
    }

    // ============================================================================
    // Configuration Tests
    // ============================================================================

    #[test]
    fn test_config_builders() {
        let high_throughput = HotMemoryConfig::high_throughput();
        assert_eq!(high_throughput.max_entries, 100_000);

        let low_memory = HotMemoryConfig::low_memory();
        assert_eq!(low_memory.max_entries, 1_000);

        let custom = HotMemoryConfig::default()
            .with_max_entries(5000)
            .with_ttl(Duration::from_secs(600));
        assert_eq!(custom.max_entries, 5000);
        assert_eq!(custom.ttl, Duration::from_secs(600));
    }

    #[test]
    fn test_config_with_eviction_batch_size() {
        let config = HotMemoryConfig::default().with_eviction_batch_size(200);
        assert_eq!(config.eviction_batch_size, 200);
    }

    // ============================================================================
    // Entry Tests
    // ============================================================================

    #[test]
    fn test_entry_expiration_check() {
        let entry = make_entry("test", vec![0.1, 0.2, 0.3]);

        // Entry should not be expired immediately with 1 hour TTL
        assert!(!entry.is_expired(Duration::from_secs(3600)));

        // Entry should be expired with 0 TTL
        assert!(entry.is_expired(Duration::from_secs(0)));
    }

    #[test]
    fn test_entry_estimated_size() {
        let entry = HotMemoryEntry::new(
            Uuid::new_v4(),
            "test content".to_string(),
            vec![0.0; 100],
            serde_json::json!({"key": "value"}),
        );

        let size = entry.estimated_size();
        assert!(size > 0);
        // Should include at least the embedding size
        assert!(size >= 100 * 4);
    }

    #[test]
    fn test_entry_touch() {
        let mut entry = make_entry("test", vec![0.1, 0.2, 0.3]);
        let initial_access_count = entry.access_count;

        entry.touch();
        assert_eq!(entry.access_count, initial_access_count + 1);

        entry.touch();
        assert_eq!(entry.access_count, initial_access_count + 2);
    }

    // ============================================================================
    // Edge Cases and Stress Tests
    // ============================================================================

    #[tokio::test]
    async fn test_large_embeddings() {
        let hot_memory = HotMemory::with_defaults();

        // Large embedding (4096 dimensions like some LLM embeddings)
        let large_embedding = vec![0.1; 4096];
        let entry = make_entry("large embedding", large_embedding.clone());
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        let retrieved = hot_memory.get(&id).await.unwrap();
        assert_eq!(retrieved.embedding.len(), 4096);
        assert_eq!(retrieved.embedding, large_embedding);
    }

    #[tokio::test]
    async fn test_many_entries() {
        let hot_memory = HotMemory::new(HotMemoryConfig {
            max_entries: 10000,
            ..Default::default()
        });

        // Add 1000 entries
        let mut ids = Vec::new();
        for i in 0..1000 {
            let entry = make_entry(&format!("entry {}", i), vec![i as f32 / 1000.0; 64]);
            ids.push(entry.id);
            hot_memory.put(entry).await.unwrap();
        }

        assert_eq!(hot_memory.len(), 1000);

        // Verify all entries are accessible
        for id in &ids {
            assert!(hot_memory.get(id).await.is_some());
        }
    }

    #[tokio::test]
    async fn test_single_dimension_embedding() {
        let hot_memory = HotMemory::with_defaults();

        let entry = make_entry("single dim", vec![1.0]);
        let id = entry.id;

        hot_memory.put(entry).await.unwrap();

        let query = vec![1.0];
        let results = hot_memory.search_similar(&query, 1).await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let hot_memory = Arc::new(HotMemory::with_defaults());
        let mut handles = vec![];

        // Spawn multiple tasks that read and write concurrently
        for i in 0..10 {
            let hm = Arc::clone(&hot_memory);
            let handle = tokio::spawn(async move {
                for j in 0..100 {
                    let entry = make_entry(
                        &format!("entry {}:{}", i, j),
                        vec![i as f32 / 10.0, j as f32 / 100.0, 0.1],
                    );
                    let id = entry.id;
                    hm.put(entry).await.unwrap();
                    hm.get(&id).await;
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let stats = hot_memory.stats().await;
        assert!(stats.entry_count > 0);
        assert_eq!(stats.total_puts, 1000);
    }

    #[tokio::test]
    async fn test_all_entries() {
        let hot_memory = HotMemory::with_defaults();

        for i in 0..5 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            hot_memory.put(entry).await.unwrap();
        }

        let all = hot_memory.all_entries();
        assert_eq!(all.len(), 5);
    }

    #[tokio::test]
    async fn test_entry_ids() {
        let hot_memory = HotMemory::with_defaults();

        let mut expected_ids = Vec::new();
        for i in 0..5 {
            let entry = make_entry(&format!("entry {}", i), vec![0.1, 0.2, 0.3]);
            expected_ids.push(entry.id);
            hot_memory.put(entry).await.unwrap();
        }

        let ids = hot_memory.entry_ids();
        assert_eq!(ids.len(), 5);

        for id in &expected_ids {
            assert!(ids.contains(id));
        }
    }
}
