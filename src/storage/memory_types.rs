//! Dual-layer memory system types and traits
//!
//! This module defines the core types and traits for ReasonKit's dual-layer
//! memory architecture:
//!
//! - **Hot Memory**: Fast, in-memory storage for frequently accessed entries
//! - **Cold Memory**: Persistent storage for long-term memory retention
//!
//! The system provides automatic tiering, crash recovery, and efficient
//! synchronization between layers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{MemError, MemResult};

// ============================================================================
// Core Memory Entry Types
// ============================================================================

/// A memory entry that can be stored in hot or cold memory
///
/// Each entry contains the content, its embedding vector for semantic search,
/// and metadata for tracking access patterns and provenance.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::storage::memory_types::{MemoryEntry, MemoryMetadata, MemoryLayer};
/// use uuid::Uuid;
///
/// let entry = MemoryEntry {
///     id: Uuid::new_v4(),
///     content: "Important meeting notes about Q4 planning".to_string(),
///     embedding: vec![0.1, 0.2, 0.3], // Normally 384-1536 dimensions
///     metadata: MemoryMetadata::default(),
///     created_at: chrono::Utc::now().timestamp(),
///     updated_at: chrono::Utc::now().timestamp(),
///     access_count: 0,
///     layer: MemoryLayer::Hot,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier for this memory entry
    pub id: Uuid,
    /// The actual content/text of the memory
    pub content: String,
    /// Dense embedding vector for semantic search
    pub embedding: Vec<f32>,
    /// Rich metadata for filtering and provenance
    pub metadata: MemoryMetadata,
    /// Unix timestamp when the entry was created
    pub created_at: i64,
    /// Unix timestamp when the entry was last modified
    pub updated_at: i64,
    /// Number of times this entry has been accessed
    pub access_count: u64,
    /// Current storage layer (hot, cold, or both during sync)
    pub layer: MemoryLayer,
}

impl MemoryEntry {
    /// Create a new memory entry with the given content and embedding
    pub fn new(content: String, embedding: Vec<f32>) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            metadata: MemoryMetadata::default(),
            created_at: now,
            updated_at: now,
            access_count: 0,
            layer: MemoryLayer::Hot,
        }
    }

    /// Create a new entry with a specific ID (for recovery/migration)
    pub fn with_id(id: Uuid, content: String, embedding: Vec<f32>) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id,
            content,
            embedding,
            metadata: MemoryMetadata::default(),
            created_at: now,
            updated_at: now,
            access_count: 0,
            layer: MemoryLayer::Hot,
        }
    }

    /// Add metadata to the entry
    pub fn with_metadata(mut self, metadata: MemoryMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the source for this entry
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.metadata.source = Some(source.into());
        self
    }

    /// Set the session ID for this entry
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.metadata.session_id = Some(session_id.into());
        self
    }

    /// Add tags to the entry
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.metadata.tags = tags;
        self
    }

    /// Record an access to this entry (increments counter and updates timestamp)
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Calculate the "heat" score for tiering decisions
    ///
    /// Higher scores indicate the entry should remain in hot memory.
    /// The algorithm considers:
    /// - Recency (more recent = hotter)
    /// - Frequency (more accesses = hotter)
    /// - Age (newer entries get a boost)
    pub fn heat_score(&self) -> f64 {
        let now = chrono::Utc::now().timestamp();
        let age_hours = ((now - self.created_at) as f64) / 3600.0;
        let last_access_hours = ((now - self.updated_at) as f64) / 3600.0;

        // Frequency component: log scale to prevent runaway scores
        let freq_score = (self.access_count as f64 + 1.0).ln();

        // Recency component: exponential decay
        let recency_score = (-last_access_hours / 24.0).exp(); // Decay over ~1 day

        // Age penalty: very old entries get lower scores
        let age_penalty = if age_hours > 168.0 {
            // Older than 1 week
            0.5
        } else {
            1.0
        };

        (freq_score * 0.4 + recency_score * 0.6) * age_penalty
    }

    /// Check if this entry is stale and should be moved to cold storage
    ///
    /// An entry is considered stale if:
    /// - It hasn't been accessed in the specified hours
    /// - Its heat score is below the threshold
    pub fn is_stale(&self, stale_hours: i64, heat_threshold: f64) -> bool {
        let now = chrono::Utc::now().timestamp();
        let hours_since_access = (now - self.updated_at) / 3600;

        hours_since_access > stale_hours || self.heat_score() < heat_threshold
    }
}

/// Metadata associated with a memory entry
///
/// Provides provenance tracking, tagging, and custom data storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMetadata {
    /// Source of the memory (e.g., "user_input", "document:xyz", "conversation:123")
    pub source: Option<String>,
    /// Session ID for grouping related memories
    pub session_id: Option<String>,
    /// User-defined tags for filtering and organization
    pub tags: Vec<String>,
    /// Custom metadata as JSON (flexible extension point)
    pub custom: serde_json::Value,
}

impl MemoryMetadata {
    /// Create metadata with a source
    pub fn with_source(source: impl Into<String>) -> Self {
        Self {
            source: Some(source.into()),
            ..Default::default()
        }
    }

    /// Create metadata with a session ID
    pub fn with_session(session_id: impl Into<String>) -> Self {
        Self {
            session_id: Some(session_id.into()),
            ..Default::default()
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    /// Set custom metadata
    pub fn set_custom<T: Serialize>(&mut self, key: &str, value: T) -> MemResult<()> {
        let obj = self.custom.as_object_mut();
        if let Some(map) = obj {
            map.insert(key.to_string(), serde_json::to_value(value)?);
        } else {
            let mut map = serde_json::Map::new();
            map.insert(key.to_string(), serde_json::to_value(value)?);
            self.custom = serde_json::Value::Object(map);
        }
        Ok(())
    }

    /// Get custom metadata
    pub fn get_custom<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.custom
            .as_object()
            .and_then(|map| map.get(key))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

/// Which memory layer an entry resides in
///
/// - `Hot`: In-memory, fast access, limited capacity
/// - `Cold`: Persistent storage, unlimited capacity, slower access
/// - `Both`: Entry exists in both layers (during sync operations)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MemoryLayer {
    /// Hot memory (in-memory, fast)
    #[default]
    Hot,
    /// Cold memory (persistent, slow)
    Cold,
    /// Both layers (during synchronization)
    Both,
}

impl std::fmt::Display for MemoryLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLayer::Hot => write!(f, "hot"),
            MemoryLayer::Cold => write!(f, "cold"),
            MemoryLayer::Both => write!(f, "both"),
        }
    }
}

// ============================================================================
// Statistics Types
// ============================================================================

/// Statistics for hot memory (in-memory cache)
///
/// Used for monitoring performance and making tiering decisions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HotMemoryStats {
    /// Number of entries currently in hot memory
    pub entry_count: usize,
    /// Total size of all entries in bytes (approximate)
    pub total_size_bytes: usize,
    /// Number of cache hits
    pub hit_count: u64,
    /// Number of cache misses (required cold lookup)
    pub miss_count: u64,
    /// Number of entries evicted to cold storage
    pub eviction_count: u64,
}

impl HotMemoryStats {
    /// Calculate the cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.miss_count += 1;
    }

    /// Record an eviction
    pub fn record_eviction(&mut self) {
        self.eviction_count += 1;
    }

    /// Reset hit/miss counters (for periodic stats collection)
    pub fn reset_counters(&mut self) {
        self.hit_count = 0;
        self.miss_count = 0;
    }
}

/// Statistics for cold memory (persistent storage)
///
/// Tracks storage usage and maintenance operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColdMemoryStats {
    /// Number of entries in cold storage
    pub entry_count: usize,
    /// Total database file size in bytes
    pub db_size_bytes: u64,
    /// Total index size in bytes (for vector and text indices)
    pub index_size_bytes: u64,
    /// Number of compaction operations performed
    pub compaction_count: u64,
}

impl ColdMemoryStats {
    /// Calculate the total storage size
    pub fn total_size_bytes(&self) -> u64 {
        self.db_size_bytes + self.index_size_bytes
    }

    /// Record a compaction operation
    pub fn record_compaction(&mut self) {
        self.compaction_count += 1;
    }

    /// Estimate average entry size
    pub fn avg_entry_size(&self) -> usize {
        if self.entry_count == 0 {
            0
        } else {
            (self.db_size_bytes as usize) / self.entry_count
        }
    }
}

/// Statistics from a sync operation between hot and cold memory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    /// Number of entries synchronized
    pub entries_synced: usize,
    /// Total bytes written during sync
    pub bytes_written: usize,
    /// Duration of the sync operation in milliseconds
    pub duration_ms: u64,
}

impl SyncStats {
    /// Create new sync stats
    pub fn new(entries_synced: usize, bytes_written: usize, duration_ms: u64) -> Self {
        Self {
            entries_synced,
            bytes_written,
            duration_ms,
        }
    }

    /// Calculate throughput in entries per second
    pub fn throughput_entries_per_sec(&self) -> f64 {
        if self.duration_ms == 0 {
            0.0
        } else {
            (self.entries_synced as f64) / (self.duration_ms as f64 / 1000.0)
        }
    }

    /// Calculate throughput in bytes per second
    pub fn throughput_bytes_per_sec(&self) -> f64 {
        if self.duration_ms == 0 {
            0.0
        } else {
            (self.bytes_written as f64) / (self.duration_ms as f64 / 1000.0)
        }
    }
}

/// Report from crash recovery process
///
/// Generated after recovering from an unexpected shutdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecoveryReport {
    /// Number of entries successfully recovered
    pub entries_recovered: usize,
    /// Number of corrupted entries that could not be recovered
    pub entries_corrupted: usize,
    /// Last successful checkpoint LSN (Log Sequence Number)
    pub last_checkpoint_lsn: u64,
    /// Time taken for recovery in milliseconds
    pub recovery_duration_ms: u64,
}

impl RecoveryReport {
    /// Check if the recovery was successful (no corruption)
    pub fn is_clean(&self) -> bool {
        self.entries_corrupted == 0
    }

    /// Calculate the recovery rate (0.0 to 1.0)
    pub fn recovery_rate(&self) -> f64 {
        let total = self.entries_recovered + self.entries_corrupted;
        if total == 0 {
            1.0
        } else {
            self.entries_recovered as f64 / total as f64
        }
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        if self.is_clean() {
            format!(
                "Clean recovery: {} entries restored in {}ms (LSN: {})",
                self.entries_recovered, self.recovery_duration_ms, self.last_checkpoint_lsn
            )
        } else {
            format!(
                "Partial recovery: {}/{} entries restored ({} corrupted) in {}ms (LSN: {})",
                self.entries_recovered,
                self.entries_recovered + self.entries_corrupted,
                self.entries_corrupted,
                self.recovery_duration_ms,
                self.last_checkpoint_lsn
            )
        }
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for hot memory layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMemoryConfig {
    /// Maximum number of entries to keep in hot memory
    pub max_entries: usize,
    /// Maximum total size in bytes
    pub max_size_bytes: usize,
    /// Threshold for considering an entry stale (hours since last access)
    pub stale_threshold_hours: i64,
    /// Heat score threshold for eviction (entries below this are candidates)
    pub heat_threshold: f64,
    /// How often to run eviction checks (in seconds)
    pub eviction_interval_secs: u64,
}

impl Default for HotMemoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_size_bytes: 256 * 1024 * 1024, // 256 MB
            stale_threshold_hours: 24,
            heat_threshold: 0.1,
            eviction_interval_secs: 60,
        }
    }
}

/// Configuration for cold memory layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdMemoryConfig {
    /// Path to the database file
    pub db_path: std::path::PathBuf,
    /// Whether to enable WAL (Write-Ahead Logging)
    pub enable_wal: bool,
    /// Sync mode (0 = OFF, 1 = NORMAL, 2 = FULL)
    pub sync_mode: u8,
    /// How often to run compaction (in seconds, 0 = disabled)
    pub compaction_interval_secs: u64,
    /// Whether to enable compression
    pub enable_compression: bool,
}

impl Default for ColdMemoryConfig {
    fn default() -> Self {
        Self {
            db_path: std::path::PathBuf::from("./data/memory.db"),
            enable_wal: true,
            sync_mode: 1, // NORMAL
            compaction_interval_secs: 3600,
            enable_compression: true,
        }
    }
}

// DualLayerConfig is defined in storage/config.rs - use that instead
// This was a duplicate definition causing E0255 errors
// Re-export from config module
pub use super::config::DualLayerConfig;

// ============================================================================
// Backend Trait
// ============================================================================

/// Trait for memory storage backends
///
/// Implementations must be thread-safe (Send + Sync) and support async operations.
/// This trait is implemented by both hot memory (in-memory) and cold memory (persistent)
/// backends.
///
/// # Example Implementation
///
/// ```rust,ignore
/// use async_trait::async_trait;
/// use reasonkit_mem::storage::memory_types::{MemoryBackend, MemoryEntry};
/// use uuid::Uuid;
///
/// struct MyBackend { /* ... */ }
///
/// #[async_trait]
/// impl MemoryBackend for MyBackend {
///     async fn store(&self, entry: &MemoryEntry) -> MemResult<()> {
///         // Store implementation
///         Ok(())
///     }
///
///     async fn get(&self, id: &Uuid) -> MemResult<Option<MemoryEntry>> {
///         // Get implementation
///         Ok(None)
///     }
///
///     async fn delete(&self, id: &Uuid) -> MemResult<bool> {
///         // Delete implementation
///         Ok(false)
///     }
///
///     async fn search_similar(&self, embedding: &[f32], limit: usize) -> MemResult<Vec<(Uuid, f32)>> {
///         // Search implementation
///         Ok(vec![])
///     }
/// }
/// ```
#[async_trait]
pub trait MemoryBackend: Send + Sync {
    /// Store a memory entry
    ///
    /// If an entry with the same ID already exists, it should be overwritten.
    async fn store(&self, entry: &MemoryEntry) -> MemResult<()>;

    /// Retrieve a memory entry by ID
    ///
    /// Returns `Ok(None)` if the entry does not exist.
    async fn get(&self, id: &Uuid) -> MemResult<Option<MemoryEntry>>;

    /// Delete a memory entry by ID
    ///
    /// Returns `true` if the entry was deleted, `false` if it didn't exist.
    async fn delete(&self, id: &Uuid) -> MemResult<bool>;

    /// Search for similar entries using vector similarity
    ///
    /// Returns a list of (entry_id, similarity_score) pairs, sorted by score descending.
    /// Scores are typically cosine similarity values in the range [-1, 1].
    async fn search_similar(&self, embedding: &[f32], limit: usize) -> MemResult<Vec<(Uuid, f32)>>;
}

/// Extended trait for hot memory backends with eviction support
#[async_trait]
pub trait HotMemoryBackend: MemoryBackend {
    /// Get current statistics
    async fn stats(&self) -> MemResult<HotMemoryStats>;

    /// Evict entries to make room for new ones
    ///
    /// Returns the IDs and entries of evicted items (to be moved to cold storage).
    async fn evict(&self, count: usize) -> MemResult<Vec<MemoryEntry>>;

    /// Get all entries (for sync to cold storage)
    async fn all_entries(&self) -> MemResult<Vec<MemoryEntry>>;

    /// Clear all entries
    async fn clear(&self) -> MemResult<()>;
}

/// Extended trait for cold memory backends with persistence features
#[async_trait]
pub trait ColdMemoryBackend: MemoryBackend {
    /// Get current statistics
    async fn stats(&self) -> MemResult<ColdMemoryStats>;

    /// Store multiple entries in a batch (for efficient sync)
    async fn store_batch(&self, entries: &[MemoryEntry]) -> MemResult<SyncStats>;

    /// Run compaction to reclaim space
    async fn compact(&self) -> MemResult<()>;

    /// Create a checkpoint for recovery
    async fn checkpoint(&self) -> MemResult<u64>;

    /// Recover from the last checkpoint
    async fn recover(&self) -> MemResult<RecoveryReport>;

    /// Search with additional filters
    async fn search_filtered(
        &self,
        embedding: &[f32],
        limit: usize,
        filter: &MemoryFilter,
    ) -> MemResult<Vec<(Uuid, f32)>>;
}

/// Filter criteria for memory searches
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFilter {
    /// Filter by source prefix
    pub source_prefix: Option<String>,
    /// Filter by session ID
    pub session_id: Option<String>,
    /// Filter by tags (entry must have ALL specified tags)
    pub tags: Vec<String>,
    /// Filter by creation time (after this timestamp)
    pub created_after: Option<i64>,
    /// Filter by creation time (before this timestamp)
    pub created_before: Option<i64>,
    /// Minimum access count
    pub min_access_count: Option<u64>,
}

impl MemoryFilter {
    /// Create an empty filter (matches everything)
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by source prefix
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source_prefix = Some(source.into());
        self
    }

    /// Filter by session ID
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Filter by tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Filter by creation time range
    pub fn with_time_range(mut self, after: Option<i64>, before: Option<i64>) -> Self {
        self.created_after = after;
        self.created_before = before;
        self
    }

    /// Filter by minimum access count
    pub fn with_min_access(mut self, count: u64) -> Self {
        self.min_access_count = Some(count);
        self
    }

    /// Check if an entry matches this filter
    pub fn matches(&self, entry: &MemoryEntry) -> bool {
        // Check source prefix
        if let Some(ref prefix) = self.source_prefix {
            if let Some(ref source) = entry.metadata.source {
                if !source.starts_with(prefix) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check session ID
        if let Some(ref session_id) = self.session_id {
            if entry.metadata.session_id.as_ref() != Some(session_id) {
                return false;
            }
        }

        // Check tags (all must be present)
        for tag in &self.tags {
            if !entry.metadata.tags.contains(tag) {
                return false;
            }
        }

        // Check creation time
        if let Some(after) = self.created_after {
            if entry.created_at < after {
                return false;
            }
        }
        if let Some(before) = self.created_before {
            if entry.created_at > before {
                return false;
            }
        }

        // Check access count
        if let Some(min_count) = self.min_access_count {
            if entry.access_count < min_count {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("Test content".to_string(), vec![0.1, 0.2, 0.3]);

        assert!(!entry.id.is_nil());
        assert_eq!(entry.content, "Test content");
        assert_eq!(entry.embedding.len(), 3);
        assert_eq!(entry.access_count, 0);
        assert_eq!(entry.layer, MemoryLayer::Hot);
    }

    #[test]
    fn test_memory_entry_with_metadata() {
        let entry = MemoryEntry::new("Test".to_string(), vec![])
            .with_source("test_source")
            .with_session("session_123")
            .with_tags(vec!["important".to_string(), "work".to_string()]);

        assert_eq!(entry.metadata.source, Some("test_source".to_string()));
        assert_eq!(entry.metadata.session_id, Some("session_123".to_string()));
        assert_eq!(entry.metadata.tags.len(), 2);
    }

    #[test]
    fn test_memory_entry_access_tracking() {
        let mut entry = MemoryEntry::new("Test".to_string(), vec![]);
        let initial_updated = entry.updated_at;

        entry.record_access();

        assert_eq!(entry.access_count, 1);
        assert!(entry.updated_at >= initial_updated);
    }

    #[test]
    fn test_heat_score() {
        let entry = MemoryEntry::new("Test".to_string(), vec![]);
        let score = entry.heat_score();

        // New entry should have a reasonable heat score
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_memory_layer_display() {
        assert_eq!(format!("{}", MemoryLayer::Hot), "hot");
        assert_eq!(format!("{}", MemoryLayer::Cold), "cold");
        assert_eq!(format!("{}", MemoryLayer::Both), "both");
    }

    #[test]
    fn test_hot_memory_stats_hit_rate() {
        let mut stats = HotMemoryStats::default();

        stats.record_hit();
        stats.record_hit();
        stats.record_miss();

        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_sync_stats_throughput() {
        let stats = SyncStats::new(100, 1024 * 1024, 1000); // 100 entries, 1MB, 1 second

        assert!((stats.throughput_entries_per_sec() - 100.0).abs() < 0.01);
        assert!((stats.throughput_bytes_per_sec() - 1024.0 * 1024.0).abs() < 1.0);
    }

    #[test]
    fn test_recovery_report() {
        let report = RecoveryReport {
            entries_recovered: 100,
            entries_corrupted: 0,
            last_checkpoint_lsn: 12345,
            recovery_duration_ms: 500,
        };

        assert!(report.is_clean());
        assert!((report.recovery_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_filter_matches() {
        let entry = MemoryEntry::new("Test".to_string(), vec![])
            .with_source("doc:abc")
            .with_session("sess_1")
            .with_tags(vec!["work".to_string()]);

        // Matching filter
        let filter = MemoryFilter::new().with_source("doc:").with_tag("work");
        assert!(filter.matches(&entry));

        // Non-matching filter (wrong source)
        let filter2 = MemoryFilter::new().with_source("user:");
        assert!(!filter2.matches(&entry));

        // Non-matching filter (missing tag)
        let filter3 = MemoryFilter::new().with_tag("personal");
        assert!(!filter3.matches(&entry));
    }

    #[test]
    fn test_metadata_custom_fields() {
        let mut metadata = MemoryMetadata::default();
        metadata.set_custom("priority", 5i32).unwrap();
        metadata.set_custom("verified", true).unwrap();

        assert_eq!(metadata.get_custom::<i32>("priority"), Some(5));
        assert_eq!(metadata.get_custom::<bool>("verified"), Some(true));
        assert_eq!(metadata.get_custom::<String>("missing"), None);
    }

    #[test]
    fn test_default_configs() {
        let hot_config = HotMemoryConfig::default();
        assert!(hot_config.max_entries > 0);
        assert!(hot_config.max_size_bytes > 0);

        let cold_config = ColdMemoryConfig::default();
        assert!(cold_config.enable_wal);

        let dual_config = DualLayerConfig::default();
        assert!(dual_config.sync.interval_secs > 0);
        // TODO: Update test - auto_promote field removed from DualLayerConfig
        // assert!(dual_config.auto_promote);
    }

    #[test]
    fn test_memory_entry_serialization() {
        let entry =
            MemoryEntry::new("Test content".to_string(), vec![0.1, 0.2, 0.3]).with_source("test");

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: MemoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, entry.id);
        assert_eq!(parsed.content, entry.content);
        assert_eq!(parsed.embedding, entry.embedding);
        assert_eq!(parsed.metadata.source, entry.metadata.source);
    }
}
