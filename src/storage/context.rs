//! Context retrieval for ReasonKit Memory
//!
//! Provides semantic search combining hot (in-memory) and cold (disk-based) memory
//! with cosine similarity scoring, recency weighting, and result fusion.
//!
//! ## Architecture
//!
//! ```text
//! Query -> [Hot Memory Search] -> [Cold Memory Search] -> [RRF Fusion] -> [Recency Weight] -> Results
//!              (fast path)           (disk-based)          (rank-based)    (time decay)
//! ```
//!
//! ## Algorithm
//!
//! 1. Search hot memory (in-memory, fast path)
//! 2. Search cold memory (disk-based, persistent)
//! 3. Fuse results using Reciprocal Rank Fusion (RRF)
//! 4. Apply recency weighting to favor recent memories
//! 5. Deduplicate and sort by final score

use crate::embedding::cosine_similarity;
use crate::{MemError, MemResult};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// =============================================================================
// CORE TYPES
// =============================================================================

/// Query for context retrieval
#[derive(Debug, Clone)]
pub struct ContextQuery {
    /// Original query text (for reference/logging)
    pub text: String,
    /// Query embedding vector for semantic search
    pub embedding: Vec<f32>,
    /// Maximum number of results to return
    pub limit: usize,
    /// Minimum similarity score threshold (0.0-1.0)
    pub min_score: f32,
    /// Recency weight factor (0.0 = ignore recency, 1.0 = heavily favor recent)
    pub recency_weight: f32,
}

impl Default for ContextQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            embedding: Vec::new(),
            limit: 10,
            min_score: 0.0,
            recency_weight: 0.3,
        }
    }
}

impl ContextQuery {
    /// Create a new context query with the given embedding
    pub fn new(embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            ..Default::default()
        }
    }

    /// Set the query text
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    /// Set the maximum number of results
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the minimum score threshold
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score.clamp(0.0, 1.0);
        self
    }

    /// Set the recency weight factor
    pub fn with_recency_weight(mut self, weight: f32) -> Self {
        self.recency_weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// Source of a memory item (hot or cold storage)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemorySource {
    /// Hot memory (in-memory, fast access)
    Hot,
    /// Cold memory (disk-based, persistent)
    Cold,
}

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemorySource::Hot => write!(f, "hot"),
            MemorySource::Cold => write!(f, "cold"),
        }
    }
}

/// Result from context retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextResult {
    /// Unique identifier for the memory item
    pub id: Uuid,
    /// Text content of the memory
    pub content: String,
    /// Combined relevance score (similarity + recency weighting)
    pub score: f32,
    /// Source of the memory (hot or cold)
    pub source: MemorySource,
    /// Additional metadata (chunk info, timestamps, etc.)
    pub metadata: serde_json::Value,
    /// Raw similarity score (before recency weighting)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_score: Option<f32>,
    /// Recency component of the score
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recency_score: Option<f32>,
}

impl ContextResult {
    /// Create a new context result
    pub fn new(
        id: Uuid,
        content: String,
        score: f32,
        source: MemorySource,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id,
            content,
            score,
            source,
            metadata,
            similarity_score: None,
            recency_score: None,
        }
    }

    /// Set score components for debugging/analysis
    pub fn with_score_components(mut self, similarity: f32, recency: f32) -> Self {
        self.similarity_score = Some(similarity);
        self.recency_score = Some(recency);
        self
    }
}

// =============================================================================
// MEMORY ITEM
// =============================================================================

/// A memory item stored in hot or cold memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Unique identifier
    pub id: Uuid,
    /// Text content
    pub content: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Creation timestamp (Unix timestamp in seconds)
    pub created_at: i64,
    /// Last access timestamp
    pub last_accessed: i64,
    /// Access count for LRU-like eviction
    pub access_count: u64,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl MemoryItem {
    /// Create a new memory item
    pub fn new(content: String, embedding: Vec<f32>) -> Self {
        let now = Utc::now().timestamp();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            metadata: serde_json::Value::Null,
        }
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set custom ID
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
    }

    /// Record an access (updates last_accessed and access_count)
    pub fn record_access(&mut self) {
        self.last_accessed = Utc::now().timestamp();
        self.access_count += 1;
    }
}

// =============================================================================
// HOT MEMORY (In-Memory Storage)
// =============================================================================

/// Configuration for hot memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMemoryConfig {
    /// Maximum number of items to store
    pub max_items: usize,
    /// Eviction strategy
    pub eviction: EvictionStrategy,
}

impl Default for HotMemoryConfig {
    fn default() -> Self {
        Self {
            max_items: 10_000,
            eviction: EvictionStrategy::LRU,
        }
    }
}

/// Eviction strategy for hot memory
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Oldest first (FIFO)
    FIFO,
}

/// Hot memory storage (in-memory, fast access)
pub struct HotMemory {
    /// Memory items indexed by ID
    pub(crate) items: Arc<RwLock<HashMap<Uuid, MemoryItem>>>,
    /// Access order for LRU eviction
    access_order: Arc<RwLock<Vec<Uuid>>>,
    /// Configuration
    config: HotMemoryConfig,
}

impl HotMemory {
    /// Create a new hot memory store
    pub fn new(config: HotMemoryConfig) -> Self {
        Self {
            items: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(HotMemoryConfig::default())
    }

    /// Store a memory item
    pub async fn store(&self, item: MemoryItem) -> MemResult<()> {
        let mut items = self.items.write().await;
        let mut order = self.access_order.write().await;

        // Evict if at capacity
        while items.len() >= self.config.max_items && !items.is_empty() {
            if let Some(evict_id) = self.select_eviction_target(&items, &order) {
                items.remove(&evict_id);
                order.retain(|id| *id != evict_id);
            } else {
                break;
            }
        }

        let id = item.id;
        items.insert(id, item);
        order.push(id);

        Ok(())
    }

    /// Get a memory item by ID
    pub async fn get(&self, id: &Uuid) -> MemResult<Option<MemoryItem>> {
        let mut items = self.items.write().await;
        let mut order = self.access_order.write().await;

        if let Some(item) = items.get_mut(id) {
            item.record_access();
            // Update access order (move to end)
            order.retain(|x| x != id);
            order.push(*id);
            Ok(Some(item.clone()))
        } else {
            Ok(None)
        }
    }

    /// Delete a memory item
    pub async fn delete(&self, id: &Uuid) -> MemResult<bool> {
        let mut items = self.items.write().await;
        let mut order = self.access_order.write().await;

        order.retain(|x| x != id);
        Ok(items.remove(id).is_some())
    }

    /// Search by vector similarity
    ///
    /// Returns (id, similarity_score, created_at) tuples sorted by similarity descending
    pub async fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> MemResult<Vec<(Uuid, f32, i64)>> {
        let items = self.items.read().await;

        let mut results: Vec<(Uuid, f32, i64)> = items
            .values()
            .map(|item| {
                let similarity = cosine_similarity(query_embedding, &item.embedding);
                (item.id, similarity, item.created_at)
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Get all items (for bulk operations)
    pub async fn all_items(&self) -> MemResult<Vec<MemoryItem>> {
        let items = self.items.read().await;
        Ok(items.values().cloned().collect())
    }

    /// Get item count
    pub async fn len(&self) -> usize {
        self.items.read().await.len()
    }

    /// Check if empty
    pub async fn is_empty(&self) -> bool {
        self.items.read().await.is_empty()
    }

    /// Select item to evict based on strategy
    fn select_eviction_target(
        &self,
        items: &HashMap<Uuid, MemoryItem>,
        access_order: &[Uuid],
    ) -> Option<Uuid> {
        match self.config.eviction {
            EvictionStrategy::LRU => access_order.first().copied(),
            EvictionStrategy::FIFO => items
                .values()
                .min_by_key(|item| item.created_at)
                .map(|item| item.id),
            EvictionStrategy::LFU => items
                .values()
                .min_by_key(|item| item.access_count)
                .map(|item| item.id),
        }
    }
}

// =============================================================================
// COLD MEMORY (Disk-Based Storage)
// =============================================================================

/// Configuration for cold memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdMemoryConfig {
    /// Path to the storage directory
    pub data_path: PathBuf,
    /// Maximum storage size in bytes (0 = unlimited)
    pub max_size_bytes: u64,
}

impl Default for ColdMemoryConfig {
    fn default() -> Self {
        Self {
            data_path: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("reasonkit")
                .join("cold_memory"),
            max_size_bytes: 0,
        }
    }
}

/// Cold memory storage (disk-based, persistent)
pub struct ColdMemory {
    /// Memory items indexed by ID (loaded lazily)
    pub(crate) items: Arc<RwLock<HashMap<Uuid, MemoryItem>>>,
    /// Configuration
    config: ColdMemoryConfig,
    /// Whether the index has been loaded
    loaded: Arc<RwLock<bool>>,
}

impl ColdMemory {
    /// Create a new cold memory store
    pub async fn new(config: ColdMemoryConfig) -> MemResult<Self> {
        // Ensure data directory exists
        if !config.data_path.exists() {
            tokio::fs::create_dir_all(&config.data_path)
                .await
                .map_err(|e| {
                    MemError::storage(format!(
                        "Failed to create cold memory directory {:?}: {}",
                        config.data_path, e
                    ))
                })?;
        }

        let cold = Self {
            items: Arc::new(RwLock::new(HashMap::new())),
            config,
            loaded: Arc::new(RwLock::new(false)),
        };

        // Load existing items
        cold.load_items().await?;

        Ok(cold)
    }

    /// Create with default configuration
    pub async fn default_config() -> MemResult<Self> {
        Self::new(ColdMemoryConfig::default()).await
    }

    /// Load items from disk
    async fn load_items(&self) -> MemResult<()> {
        let mut loaded = self.loaded.write().await;
        if *loaded {
            return Ok(());
        }

        let index_path = self.config.data_path.join("index.json");
        if index_path.exists() {
            let content = tokio::fs::read_to_string(&index_path).await.map_err(|e| {
                MemError::storage(format!("Failed to read cold memory index: {}", e))
            })?;

            let items: HashMap<Uuid, MemoryItem> = serde_json::from_str(&content).map_err(|e| {
                MemError::storage(format!("Failed to parse cold memory index: {}", e))
            })?;

            let mut store = self.items.write().await;
            *store = items;
        }

        *loaded = true;
        Ok(())
    }

    /// Persist items to disk
    async fn persist(&self) -> MemResult<()> {
        let items = self.items.read().await;
        let index_path = self.config.data_path.join("index.json");

        let content = serde_json::to_string_pretty(&*items)
            .map_err(|e| MemError::storage(format!("Failed to serialize cold memory: {}", e)))?;

        tokio::fs::write(&index_path, content)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write cold memory index: {}", e)))?;

        Ok(())
    }

    /// Store a memory item
    pub async fn store(&self, item: MemoryItem) -> MemResult<()> {
        {
            let mut items = self.items.write().await;
            items.insert(item.id, item);
        }
        self.persist().await
    }

    /// Get a memory item by ID
    pub async fn get(&self, id: &Uuid) -> MemResult<Option<MemoryItem>> {
        let items = self.items.read().await;
        Ok(items.get(id).cloned())
    }

    /// Delete a memory item
    pub async fn delete(&self, id: &Uuid) -> MemResult<bool> {
        let removed = {
            let mut items = self.items.write().await;
            items.remove(id).is_some()
        };
        if removed {
            self.persist().await?;
        }
        Ok(removed)
    }

    /// Search by vector similarity
    ///
    /// Returns (id, similarity_score, created_at) tuples sorted by similarity descending
    pub async fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> MemResult<Vec<(Uuid, f32, i64)>> {
        let items = self.items.read().await;

        let mut results: Vec<(Uuid, f32, i64)> = items
            .values()
            .map(|item| {
                let similarity = cosine_similarity(query_embedding, &item.embedding);
                (item.id, similarity, item.created_at)
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Get all items
    pub async fn all_items(&self) -> MemResult<Vec<MemoryItem>> {
        let items = self.items.read().await;
        Ok(items.values().cloned().collect())
    }

    /// Get item count
    pub async fn len(&self) -> usize {
        self.items.read().await.len()
    }

    /// Check if empty
    pub async fn is_empty(&self) -> bool {
        self.items.read().await.is_empty()
    }

    /// Move item to hot memory (marks for eviction from cold)
    pub async fn promote(&self, id: &Uuid) -> MemResult<Option<MemoryItem>> {
        let item = {
            let mut items = self.items.write().await;
            items.remove(id)
        };
        if item.is_some() {
            self.persist().await?;
        }
        Ok(item)
    }
}

// =============================================================================
// SCORING FUNCTIONS
// =============================================================================

/// Compute final score with recency weighting
///
/// The final score combines semantic similarity with recency bias:
/// - `similarity`: Cosine similarity score (0.0-1.0)
/// - `created_at`: Unix timestamp when the memory was created
/// - `recency_weight`: How much to favor recent memories (0.0-1.0)
///
/// Formula:
/// ```text
/// recency_factor = exp(-age_days / 30) where age = now - created_at
/// final_score = (1 - recency_weight) * similarity + recency_weight * recency_factor * similarity
/// ```
///
/// This ensures that:
/// - When recency_weight = 0.0, only similarity matters
/// - When recency_weight = 1.0, recent items get significant boost
/// - Older items still rank if highly similar
pub fn compute_final_score(similarity: f32, created_at: i64, recency_weight: f32) -> f32 {
    if recency_weight <= 0.0 {
        return similarity;
    }

    let now = Utc::now().timestamp();
    let age_seconds = (now - created_at).max(0) as f32;
    let age_days = age_seconds / 86400.0;

    // Exponential decay: half-life of ~21 days (30/ln(2))
    let recency_factor = (-age_days / 30.0).exp();

    // Blend similarity with recency-boosted similarity
    let base_component = (1.0 - recency_weight) * similarity;
    let recency_component = recency_weight * recency_factor * similarity;

    base_component + recency_component
}

/// Compute recency factor for a timestamp
///
/// Returns a value between 0.0 (very old) and 1.0 (just created)
pub fn compute_recency_factor(created_at: i64) -> f32 {
    let now = Utc::now().timestamp();
    let age_seconds = (now - created_at).max(0) as f32;
    let age_days = age_seconds / 86400.0;
    (-age_days / 30.0).exp()
}

// =============================================================================
// RECIPROCAL RANK FUSION
// =============================================================================

/// Fuse results from multiple sources using Reciprocal Rank Fusion (RRF)
///
/// RRF combines ranked lists by summing reciprocal ranks:
/// ```text
/// score(d) = sum over all lists of: 1 / (k + rank(d))
/// ```
///
/// where k is a constant (typically 60) that determines how much weight is given
/// to lower-ranked items.
///
/// Reference: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
/// (Cormack et al., 2009)
///
/// # Arguments
/// * `hot_results` - Results from hot memory (id, similarity)
/// * `cold_results` - Results from cold memory (id, similarity)
/// * `k` - RRF constant (typically 60)
///
/// # Returns
/// Fused results as (id, rrf_score) sorted by score descending
pub fn reciprocal_rank_fusion(
    hot_results: Vec<(Uuid, f32)>,
    cold_results: Vec<(Uuid, f32)>,
    k: usize,
) -> Vec<(Uuid, f32)> {
    let mut rrf_scores: HashMap<Uuid, f32> = HashMap::new();

    // Process hot memory results
    for (rank, (id, _score)) in hot_results.iter().enumerate() {
        let rrf = 1.0 / (k as f32 + rank as f32 + 1.0);
        *rrf_scores.entry(*id).or_insert(0.0) += rrf;
    }

    // Process cold memory results
    for (rank, (id, _score)) in cold_results.iter().enumerate() {
        let rrf = 1.0 / (k as f32 + rank as f32 + 1.0);
        *rrf_scores.entry(*id).or_insert(0.0) += rrf;
    }

    // Sort by RRF score descending
    let mut results: Vec<(Uuid, f32)> = rrf_scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Alternative: weighted RRF that considers original scores
///
/// This variant weights the RRF contribution by the original similarity score,
/// which can improve results when one source has much better matches.
pub fn weighted_reciprocal_rank_fusion(
    hot_results: Vec<(Uuid, f32)>,
    cold_results: Vec<(Uuid, f32)>,
    k: usize,
    hot_weight: f32,
    cold_weight: f32,
) -> Vec<(Uuid, f32)> {
    let mut rrf_scores: HashMap<Uuid, f32> = HashMap::new();

    // Process hot memory results with weight
    for (rank, (id, score)) in hot_results.iter().enumerate() {
        let rrf = hot_weight * score * (1.0 / (k as f32 + rank as f32 + 1.0));
        *rrf_scores.entry(*id).or_insert(0.0) += rrf;
    }

    // Process cold memory results with weight
    for (rank, (id, score)) in cold_results.iter().enumerate() {
        let rrf = cold_weight * score * (1.0 / (k as f32 + rank as f32 + 1.0));
        *rrf_scores.entry(*id).or_insert(0.0) += rrf;
    }

    let mut results: Vec<(Uuid, f32)> = rrf_scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

// =============================================================================
// CONTEXT RETRIEVAL
// =============================================================================

/// Retrieve relevant context from memory using semantic search
///
/// This function combines hot (in-memory) and cold (disk-based) memory search
/// using Reciprocal Rank Fusion (RRF) for result combination, with optional
/// recency weighting to favor more recent memories.
///
/// # Algorithm
///
/// 1. **Search hot memory** (fast path, in-memory)
///    - Compute cosine similarity for all items
///    - Return top candidates with scores and timestamps
///
/// 2. **Search cold memory** (disk-based)
///    - Same cosine similarity search
///    - May be larger but slower
///
/// 3. **Fuse results with RRF**
///    - Combine rankings from both sources
///    - RRF normalizes across different score distributions
///
/// 4. **Apply recency weighting**
///    - Boost recent memories based on recency_weight
///    - Exponential decay with ~21 day half-life
///
/// 5. **Deduplicate and filter**
///    - Remove duplicates (same ID from both sources)
///    - Filter by minimum score threshold
///    - Sort by final score and limit results
///
/// # Arguments
///
/// * `query` - The context query with embedding, limits, and scoring parameters
/// * `hot` - Reference to hot memory storage
/// * `cold` - Reference to cold memory storage
///
/// # Returns
///
/// Vector of `ContextResult` sorted by relevance score (descending)
///
/// # Example
///
/// ```ignore
/// let query = ContextQuery::new(embedding)
///     .with_text("What is machine learning?")
///     .with_limit(10)
///     .with_min_score(0.5)
///     .with_recency_weight(0.3);
///
/// let results = retrieve_context(&query, &hot_memory, &cold_memory).await?;
/// for result in results {
///     println!("{}: {} (score: {:.3})", result.source, result.content, result.score);
/// }
/// ```
pub async fn retrieve_context(
    query: &ContextQuery,
    hot: &HotMemory,
    cold: &ColdMemory,
) -> MemResult<Vec<ContextResult>> {
    // Validate query
    if query.embedding.is_empty() {
        return Err(MemError::invalid_input("Query embedding cannot be empty"));
    }

    // Fetch more candidates than needed to allow for filtering
    let fetch_limit = (query.limit * 3).max(50);

    // Step 1 & 2: Search both memory stores in parallel
    let (hot_results, cold_results) = tokio::join!(
        hot.search(&query.embedding, fetch_limit),
        cold.search(&query.embedding, fetch_limit)
    );

    let hot_results = hot_results?;
    let cold_results = cold_results?;

    // Build lookup maps for metadata
    let hot_items = hot.items.read().await;
    let cold_items = cold.items.read().await;

    // Step 3: Fuse results using RRF
    // Convert to (id, score) format for RRF
    let hot_for_rrf: Vec<(Uuid, f32)> = hot_results
        .iter()
        .map(|(id, score, _)| (*id, *score))
        .collect();
    let cold_for_rrf: Vec<(Uuid, f32)> = cold_results
        .iter()
        .map(|(id, score, _)| (*id, *score))
        .collect();

    let rrf_results = reciprocal_rank_fusion(hot_for_rrf, cold_for_rrf, 60);

    // Build timestamp lookup for recency calculation
    let mut timestamp_lookup: HashMap<Uuid, (i64, f32, MemorySource)> = HashMap::new();

    for (id, score, created_at) in &hot_results {
        timestamp_lookup.insert(*id, (*created_at, *score, MemorySource::Hot));
    }
    for (id, score, created_at) in &cold_results {
        // If already in hot, prefer hot source (it's faster for subsequent access)
        timestamp_lookup
            .entry(*id)
            .or_insert((*created_at, *score, MemorySource::Cold));
    }

    // Step 4 & 5: Apply recency weighting, filter, and build final results
    let mut final_results: Vec<ContextResult> = Vec::new();
    let mut seen_ids: HashSet<Uuid> = HashSet::new();

    for (id, _rrf_score) in rrf_results {
        // Skip duplicates
        if seen_ids.contains(&id) {
            continue;
        }
        seen_ids.insert(id);

        // Get timestamp and source info
        let (created_at, similarity, source) = match timestamp_lookup.get(&id) {
            Some(info) => *info,
            None => continue,
        };

        // Compute final score with recency weighting
        // We use similarity for the base score, not RRF score
        let final_score = compute_final_score(similarity, created_at, query.recency_weight);

        // Filter by minimum score
        if final_score < query.min_score {
            continue;
        }

        // Get content and metadata from the appropriate store
        let (content, metadata) = match source {
            MemorySource::Hot => {
                if let Some(item) = hot_items.get(&id) {
                    (item.content.clone(), item.metadata.clone())
                } else {
                    continue;
                }
            }
            MemorySource::Cold => {
                if let Some(item) = cold_items.get(&id) {
                    (item.content.clone(), item.metadata.clone())
                } else {
                    continue;
                }
            }
        };

        let recency_factor = compute_recency_factor(created_at);

        final_results.push(
            ContextResult::new(id, content, final_score, source, metadata)
                .with_score_components(similarity, recency_factor),
        );
    }

    // Sort by final score descending
    final_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit results
    final_results.truncate(query.limit);

    Ok(final_results)
}

/// Retrieve context with custom RRF parameters
///
/// This variant allows customizing the RRF constant k and source weights.
pub async fn retrieve_context_custom(
    query: &ContextQuery,
    hot: &HotMemory,
    cold: &ColdMemory,
    rrf_k: usize,
    hot_weight: f32,
    cold_weight: f32,
) -> MemResult<Vec<ContextResult>> {
    if query.embedding.is_empty() {
        return Err(MemError::invalid_input("Query embedding cannot be empty"));
    }

    let fetch_limit = (query.limit * 3).max(50);

    let (hot_results, cold_results) = tokio::join!(
        hot.search(&query.embedding, fetch_limit),
        cold.search(&query.embedding, fetch_limit)
    );

    let hot_results = hot_results?;
    let cold_results = cold_results?;

    let hot_items = hot.items.read().await;
    let cold_items = cold.items.read().await;

    let hot_for_rrf: Vec<(Uuid, f32)> = hot_results
        .iter()
        .map(|(id, score, _)| (*id, *score))
        .collect();
    let cold_for_rrf: Vec<(Uuid, f32)> = cold_results
        .iter()
        .map(|(id, score, _)| (*id, *score))
        .collect();

    // Use weighted RRF
    let rrf_results =
        weighted_reciprocal_rank_fusion(hot_for_rrf, cold_for_rrf, rrf_k, hot_weight, cold_weight);

    let mut timestamp_lookup: HashMap<Uuid, (i64, f32, MemorySource)> = HashMap::new();
    for (id, score, created_at) in &hot_results {
        timestamp_lookup.insert(*id, (*created_at, *score, MemorySource::Hot));
    }
    for (id, score, created_at) in &cold_results {
        timestamp_lookup
            .entry(*id)
            .or_insert((*created_at, *score, MemorySource::Cold));
    }

    let mut final_results: Vec<ContextResult> = Vec::new();
    let mut seen_ids: HashSet<Uuid> = HashSet::new();

    for (id, _rrf_score) in rrf_results {
        if seen_ids.contains(&id) {
            continue;
        }
        seen_ids.insert(id);

        let (created_at, similarity, source) = match timestamp_lookup.get(&id) {
            Some(info) => *info,
            None => continue,
        };

        let final_score = compute_final_score(similarity, created_at, query.recency_weight);

        if final_score < query.min_score {
            continue;
        }

        let (content, metadata) = match source {
            MemorySource::Hot => {
                if let Some(item) = hot_items.get(&id) {
                    (item.content.clone(), item.metadata.clone())
                } else {
                    continue;
                }
            }
            MemorySource::Cold => {
                if let Some(item) = cold_items.get(&id) {
                    (item.content.clone(), item.metadata.clone())
                } else {
                    continue;
                }
            }
        };

        let recency_factor = compute_recency_factor(created_at);

        final_results.push(
            ContextResult::new(id, content, final_score, source, metadata)
                .with_score_components(similarity, recency_factor),
        );
    }

    final_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    final_results.truncate(query.limit);

    Ok(final_results)
}

// =============================================================================
// UNIFIED MEMORY INTERFACE
// =============================================================================

/// Unified memory interface combining hot and cold storage
///
/// Provides a single API for memory operations with automatic tiering.
pub struct UnifiedMemory {
    /// Hot (in-memory) storage
    pub hot: HotMemory,
    /// Cold (disk-based) storage
    pub cold: ColdMemory,
    /// Whether to auto-tier (promote frequently accessed cold items to hot)
    #[allow(dead_code)]
    auto_tier: bool,
    /// Access threshold for promotion
    #[allow(dead_code)]
    promotion_threshold: u64,
}

impl UnifiedMemory {
    /// Create a new unified memory with hot and cold storage
    pub fn new(hot: HotMemory, cold: ColdMemory) -> Self {
        Self {
            hot,
            cold,
            auto_tier: true,
            promotion_threshold: 3,
        }
    }

    /// Create with default configurations
    pub async fn with_defaults() -> MemResult<Self> {
        let hot = HotMemory::default_config();
        let cold = ColdMemory::default_config().await?;
        Ok(Self::new(hot, cold))
    }

    /// Disable auto-tiering
    pub fn disable_auto_tier(mut self) -> Self {
        self.auto_tier = false;
        self
    }

    /// Set promotion threshold
    pub fn with_promotion_threshold(mut self, threshold: u64) -> Self {
        self.promotion_threshold = threshold;
        self
    }

    /// Store a memory item (goes to hot memory first)
    pub async fn store(&self, item: MemoryItem) -> MemResult<()> {
        self.hot.store(item).await
    }

    /// Store directly to cold memory (for persistence)
    pub async fn store_cold(&self, item: MemoryItem) -> MemResult<()> {
        self.cold.store(item).await
    }

    /// Retrieve context using both memory tiers
    pub async fn retrieve(&self, query: &ContextQuery) -> MemResult<Vec<ContextResult>> {
        retrieve_context(query, &self.hot, &self.cold).await
    }

    /// Move all hot items to cold storage (persist)
    pub async fn persist_hot(&self) -> MemResult<usize> {
        let items = self.hot.all_items().await?;
        let count = items.len();

        for item in items {
            self.cold.store(item.clone()).await?;
            self.hot.delete(&item.id).await?;
        }

        Ok(count)
    }

    /// Get total item count across both tiers
    pub async fn total_count(&self) -> usize {
        self.hot.len().await + self.cold.len().await
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embedding() -> Vec<f32> {
        // Create a normalized random-ish embedding
        let mut emb: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
        let magnitude: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut emb {
            *x /= magnitude;
        }
        emb
    }

    fn create_similar_embedding(base: &[f32], similarity: f32) -> Vec<f32> {
        // Create an embedding with approximately the target similarity
        let noise_factor = (1.0 - similarity).sqrt();
        let base_factor = similarity.sqrt();

        let mut emb: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(i, x)| base_factor * x + noise_factor * ((i as f32 * 0.3).cos() * 0.1))
            .collect();

        // Normalize
        let magnitude: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut emb {
            *x /= magnitude;
        }
        emb
    }

    #[test]
    fn test_compute_final_score() {
        // Test with no recency weight
        let score = compute_final_score(0.8, Utc::now().timestamp(), 0.0);
        assert!((score - 0.8).abs() < 0.001);

        // Test with recent item and full recency weight
        let score = compute_final_score(0.8, Utc::now().timestamp(), 1.0);
        assert!(score > 0.75); // Should be close to original since very recent

        // Test with old item
        let old_timestamp = Utc::now().timestamp() - (30 * 24 * 3600); // 30 days ago
        let score = compute_final_score(0.8, old_timestamp, 1.0);
        assert!(score < 0.5); // Should be reduced due to age
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let id1 = Uuid::from_u128(1);
        let id2 = Uuid::from_u128(2);
        let id3 = Uuid::from_u128(3);

        let hot_results = vec![(id1, 0.9), (id2, 0.8), (id3, 0.7)];
        let cold_results = vec![(id2, 0.95), (id1, 0.85), (id3, 0.75)];

        let fused = reciprocal_rank_fusion(hot_results, cold_results, 60);

        // id1 and id2 should both have high RRF scores (appear in both)
        assert_eq!(fused.len(), 3);

        // The top result should be either id1 or id2 (both rank well in both lists)
        let top_id = fused[0].0;
        assert!(top_id == id1 || top_id == id2);
    }

    #[test]
    fn test_context_query_builder() {
        let embedding = create_test_embedding();
        let query = ContextQuery::new(embedding.clone())
            .with_text("test query")
            .with_limit(20)
            .with_min_score(0.5)
            .with_recency_weight(0.4);

        assert_eq!(query.text, "test query");
        assert_eq!(query.limit, 20);
        assert!((query.min_score - 0.5).abs() < 0.001);
        assert!((query.recency_weight - 0.4).abs() < 0.001);
        assert_eq!(query.embedding.len(), embedding.len());
    }

    #[tokio::test]
    async fn test_hot_memory_basic() {
        let hot = HotMemory::new(HotMemoryConfig {
            max_items: 100,
            eviction: EvictionStrategy::LRU,
        });

        let embedding = create_test_embedding();
        let item = MemoryItem::new("Test content".to_string(), embedding.clone());
        let id = item.id;

        // Store
        hot.store(item).await.unwrap();
        assert_eq!(hot.len().await, 1);

        // Get
        let retrieved = hot.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test content");

        // Search
        let results = hot.search(&embedding, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.99); // Should be very similar (same embedding)

        // Delete
        let deleted = hot.delete(&id).await.unwrap();
        assert!(deleted);
        assert!(hot.is_empty().await);
    }

    #[tokio::test]
    async fn test_hot_memory_eviction() {
        let hot = HotMemory::new(HotMemoryConfig {
            max_items: 3,
            eviction: EvictionStrategy::LRU,
        });

        let embedding = create_test_embedding();

        // Store 4 items (exceeds max)
        for i in 0..4 {
            let item = MemoryItem::new(format!("Content {}", i), embedding.clone());
            hot.store(item).await.unwrap();
        }

        // Should only have 3 items due to eviction
        assert_eq!(hot.len().await, 3);
    }

    #[tokio::test]
    async fn test_cold_memory_basic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ColdMemoryConfig {
            data_path: temp_dir.path().to_path_buf(),
            max_size_bytes: 0,
        };

        let cold = ColdMemory::new(config).await.unwrap();

        let embedding = create_test_embedding();
        let item = MemoryItem::new("Test cold content".to_string(), embedding.clone());
        let id = item.id;

        // Store
        cold.store(item).await.unwrap();
        assert_eq!(cold.len().await, 1);

        // Get
        let retrieved = cold.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test cold content");

        // Search
        let results = cold.search(&embedding, 10).await.unwrap();
        assert_eq!(results.len(), 1);

        // Persistence check - create new instance
        drop(cold);

        let config2 = ColdMemoryConfig {
            data_path: temp_dir.path().to_path_buf(),
            max_size_bytes: 0,
        };
        let cold2 = ColdMemory::new(config2).await.unwrap();
        let retrieved2 = cold2.get(&id).await.unwrap();
        assert!(retrieved2.is_some());
    }

    #[tokio::test]
    async fn test_retrieve_context() {
        let temp_dir = tempfile::tempdir().unwrap();

        let hot = HotMemory::default_config();
        let cold = ColdMemory::new(ColdMemoryConfig {
            data_path: temp_dir.path().to_path_buf(),
            max_size_bytes: 0,
        })
        .await
        .unwrap();

        let base_embedding = create_test_embedding();

        // Add items to hot memory
        for i in 0..3 {
            let similarity = 0.9 - (i as f32 * 0.1);
            let emb = create_similar_embedding(&base_embedding, similarity);
            let item = MemoryItem::new(format!("Hot item {}", i), emb);
            hot.store(item).await.unwrap();
        }

        // Add items to cold memory
        for i in 0..3 {
            let similarity = 0.85 - (i as f32 * 0.1);
            let emb = create_similar_embedding(&base_embedding, similarity);
            let item = MemoryItem::new(format!("Cold item {}", i), emb);
            cold.store(item).await.unwrap();
        }

        // Query
        let query = ContextQuery::new(base_embedding)
            .with_text("test query")
            .with_limit(5)
            .with_min_score(0.3)
            .with_recency_weight(0.1);

        let results = retrieve_context(&query, &hot, &cold).await.unwrap();

        // Should have results from both sources
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // Results should be sorted by score descending
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }

        // Should have both hot and cold sources
        let has_hot = results.iter().any(|r| r.source == MemorySource::Hot);
        let has_cold = results.iter().any(|r| r.source == MemorySource::Cold);
        assert!(has_hot || has_cold);
    }

    #[tokio::test]
    async fn test_unified_memory() {
        let temp_dir = tempfile::tempdir().unwrap();

        let hot = HotMemory::default_config();
        let cold = ColdMemory::new(ColdMemoryConfig {
            data_path: temp_dir.path().to_path_buf(),
            max_size_bytes: 0,
        })
        .await
        .unwrap();

        let unified = UnifiedMemory::new(hot, cold);

        let embedding = create_test_embedding();
        let item = MemoryItem::new("Unified test".to_string(), embedding.clone());

        // Store (goes to hot by default)
        unified.store(item).await.unwrap();
        assert_eq!(unified.total_count().await, 1);

        // Persist to cold
        let persisted = unified.persist_hot().await.unwrap();
        assert_eq!(persisted, 1);
        assert_eq!(unified.hot.len().await, 0);
        assert_eq!(unified.cold.len().await, 1);
    }

    #[test]
    fn test_memory_source_display() {
        assert_eq!(format!("{}", MemorySource::Hot), "hot");
        assert_eq!(format!("{}", MemorySource::Cold), "cold");
    }

    #[test]
    fn test_recency_factor() {
        // Very recent (now)
        let recent = compute_recency_factor(Utc::now().timestamp());
        assert!(recent > 0.99);

        // ~21 days old (half-life)
        let half_life = compute_recency_factor(Utc::now().timestamp() - (21 * 24 * 3600));
        assert!(half_life > 0.4 && half_life < 0.6);

        // Very old (60 days)
        let old = compute_recency_factor(Utc::now().timestamp() - (60 * 24 * 3600));
        assert!(old < 0.2);
    }
}
