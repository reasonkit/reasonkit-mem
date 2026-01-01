//! Cross-Encoder Reranking Module
//!
//! Implements document reranking using cross-encoder models for improved
//! retrieval precision.
//!
//! ## Research Foundation
//!
//! Based on: "Cross-Encoders for Document Reranking"
//! - **Paper:** <https://arxiv.org/abs/2010.06467>
//! - **Claim:** Cross-encoders achieve SOTA reranking quality
//! - **Target:** MRR@10 > 0.40 on MS MARCO, latency < 200ms for top-20
//!
//! ## Architecture
//!
//! ```text
//! Query + Document → Cross-Encoder → Relevance Score
//!
//! Unlike bi-encoders (separate embeddings):
//!   Query → Encoder → embedding₁  ─┐
//!   Doc   → Encoder → embedding₂  ─┴→ cosine_similarity
//!
//! Cross-encoders process the pair together:
//!   [CLS] Query [SEP] Document [SEP] → Transformer → Score
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::retrieval::{Reranker, RerankerConfig, HybridResult};
//!
//! // Create reranker with configuration
//! let reranker = Reranker::new(RerankerConfig::default());
//!
//! // Rerank results from hybrid search
//! let reranked = reranker.rerank(query, &results, top_k).await?;
//! ```

use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Configuration for the cross-encoder reranker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Model identifier (e.g., "ms-marco-MiniLM-L-12-v2")
    pub model_id: String,

    /// Maximum sequence length for cross-encoder input
    pub max_length: usize,

    /// Batch size for inference
    pub batch_size: usize,

    /// Whether to use GPU for inference
    pub use_gpu: bool,

    /// Minimum score threshold (filter out below this)
    pub score_threshold: Option<f32>,

    /// Enable caching of (query, doc) pairs
    pub enable_cache: bool,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            model_id: "cross-encoder/ms-marco-MiniLM-L-12-v2".to_string(),
            max_length: 512,
            batch_size: 16,
            use_gpu: false,
            score_threshold: None,
            enable_cache: true,
        }
    }
}

impl RerankerConfig {
    /// Create config for fast reranking (smaller model)
    pub fn fast() -> Self {
        Self {
            model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            max_length: 256,
            batch_size: 32,
            use_gpu: false,
            score_threshold: None,
            enable_cache: true,
        }
    }

    /// Create config for high-quality reranking (larger model)
    pub fn quality() -> Self {
        Self {
            model_id: "cross-encoder/ms-marco-electra-base".to_string(),
            max_length: 512,
            batch_size: 8,
            use_gpu: true,
            score_threshold: None,
            enable_cache: true,
        }
    }
}

/// A candidate document for reranking
#[derive(Debug, Clone)]
pub struct RerankerCandidate {
    /// Unique identifier
    pub id: Uuid,

    /// Document/chunk text
    pub text: String,

    /// Original retrieval score
    pub original_score: f32,

    /// Original rank position
    pub original_rank: usize,
}

/// Reranked result with cross-encoder score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankedResult {
    /// Unique identifier
    pub id: Uuid,

    /// Document/chunk text
    pub text: String,

    /// Cross-encoder relevance score (0-1, higher = more relevant)
    pub rerank_score: f32,

    /// Original retrieval score
    pub original_score: f32,

    /// Original rank position
    pub original_rank: usize,

    /// New rank after reranking
    pub new_rank: usize,

    /// Rank improvement (positive = moved up)
    pub rank_delta: i32,
}

/// Cross-encoder model backend trait
#[async_trait]
pub trait CrossEncoderBackend: Send + Sync {
    /// Score a batch of (query, document) pairs
    async fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;

    /// Get the model identifier
    fn model_id(&self) -> &str;

    /// Get maximum sequence length
    fn max_length(&self) -> usize;
}

/// Placeholder cross-encoder for testing
///
/// Uses a simple heuristic: term overlap between query and document
pub struct HeuristicCrossEncoder {
    model_id: String,
    max_length: usize,
}

impl Default for HeuristicCrossEncoder {
    fn default() -> Self {
        Self {
            model_id: "heuristic-overlap".to_string(),
            max_length: 512,
        }
    }
}

impl HeuristicCrossEncoder {
    /// Create a new heuristic cross-encoder
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute term overlap score between query and document
    fn term_overlap_score(query: &str, document: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let query_terms: std::collections::HashSet<String> = query_lower
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let doc_lower = document.to_lowercase();

        if query_terms.is_empty() {
            return 0.0;
        }

        let matches = query_terms
            .iter()
            .filter(|term| doc_lower.contains(term.as_str()))
            .count();

        matches as f32 / query_terms.len() as f32
    }
}

#[async_trait]
impl CrossEncoderBackend for HeuristicCrossEncoder {
    async fn score_pairs(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        Ok(documents
            .iter()
            .map(|doc| Self::term_overlap_score(query, doc))
            .collect())
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Cross-encoder reranker for improving retrieval precision
pub struct Reranker {
    config: RerankerConfig,
    backend: Arc<dyn CrossEncoderBackend>,
}

impl Reranker {
    /// Create a new reranker with heuristic backend (for testing)
    pub fn new(config: RerankerConfig) -> Self {
        Self {
            config,
            backend: Arc::new(HeuristicCrossEncoder::default()),
        }
    }

    /// Create a reranker with a custom backend
    pub fn with_backend(config: RerankerConfig, backend: Arc<dyn CrossEncoderBackend>) -> Self {
        Self { config, backend }
    }

    /// Get the configuration
    pub fn config(&self) -> &RerankerConfig {
        &self.config
    }

    /// Rerank a list of candidates
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `candidates` - List of candidates to rerank
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Reranked results sorted by cross-encoder score
    pub async fn rerank(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
        top_k: usize,
    ) -> Result<Vec<RerankedResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Extract texts for scoring
        let texts: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();

        // Score all (query, document) pairs
        let scores = self.backend.score_pairs(query, &texts).await?;

        // Create results with scores
        let mut results: Vec<RerankedResult> = candidates
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(idx, (candidate, &score))| RerankedResult {
                id: candidate.id,
                text: candidate.text.clone(),
                rerank_score: score,
                original_score: candidate.original_score,
                original_rank: candidate.original_rank,
                new_rank: idx, // Will be updated after sorting
                rank_delta: 0, // Will be updated after sorting
            })
            .collect();

        // Sort by rerank score (descending)
        results.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply score threshold filter if configured
        if let Some(threshold) = self.config.score_threshold {
            results.retain(|r| r.rerank_score >= threshold);
        }

        // Update new ranks and deltas
        for (new_rank, result) in results.iter_mut().enumerate() {
            result.new_rank = new_rank;
            result.rank_delta = result.original_rank as i32 - new_rank as i32;
        }

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }

    /// Rerank with batching for large candidate sets
    ///
    /// Processes candidates in batches to manage memory and latency.
    pub async fn rerank_batched(
        &self,
        query: &str,
        candidates: &[RerankerCandidate],
        top_k: usize,
    ) -> Result<Vec<RerankedResult>> {
        if candidates.len() <= self.config.batch_size {
            return self.rerank(query, candidates, top_k).await;
        }

        let mut all_results = Vec::new();

        // Process in batches
        for chunk in candidates.chunks(self.config.batch_size) {
            let batch_results = self.rerank(query, chunk, chunk.len()).await?;
            all_results.extend(batch_results);
        }

        // Re-sort all results
        all_results.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update final ranks
        for (new_rank, result) in all_results.iter_mut().enumerate() {
            result.new_rank = new_rank;
            result.rank_delta = result.original_rank as i32 - new_rank as i32;
        }

        // Take top_k
        all_results.truncate(top_k);

        Ok(all_results)
    }
}

/// Statistics for reranking operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankStats {
    /// Number of candidates reranked
    pub candidates_count: usize,

    /// Average rerank score
    pub avg_rerank_score: f32,

    /// Average rank delta (how much positions shifted)
    pub avg_rank_delta: f32,

    /// Number of results that moved up
    pub moved_up_count: usize,

    /// Number of results that moved down
    pub moved_down_count: usize,

    /// Latency in milliseconds
    pub latency_ms: u64,
}

impl RerankStats {
    /// Compute statistics from reranked results
    pub fn from_results(results: &[RerankedResult], latency_ms: u64) -> Self {
        if results.is_empty() {
            return Self {
                latency_ms,
                ..Default::default()
            };
        }

        let sum_score: f32 = results.iter().map(|r| r.rerank_score).sum();
        let sum_delta: i32 = results.iter().map(|r| r.rank_delta).sum();
        let moved_up = results.iter().filter(|r| r.rank_delta > 0).count();
        let moved_down = results.iter().filter(|r| r.rank_delta < 0).count();

        Self {
            candidates_count: results.len(),
            avg_rerank_score: sum_score / results.len() as f32,
            avg_rank_delta: sum_delta as f32 / results.len() as f32,
            moved_up_count: moved_up,
            moved_down_count: moved_down,
            latency_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Test Fixtures and Helpers
    // ============================================================

    fn create_test_candidates() -> Vec<RerankerCandidate> {
        vec![
            RerankerCandidate {
                id: Uuid::new_v4(),
                text: "Machine learning is a subset of artificial intelligence.".to_string(),
                original_score: 0.8,
                original_rank: 0,
            },
            RerankerCandidate {
                id: Uuid::new_v4(),
                text: "Deep learning uses neural networks for pattern recognition.".to_string(),
                original_score: 0.7,
                original_rank: 1,
            },
            RerankerCandidate {
                id: Uuid::new_v4(),
                text: "The weather today is sunny and warm.".to_string(),
                original_score: 0.6,
                original_rank: 2,
            },
        ]
    }

    /// Create a candidate with specific text and rank
    fn make_candidate(text: &str, original_rank: usize) -> RerankerCandidate {
        RerankerCandidate {
            id: Uuid::new_v4(),
            text: text.to_string(),
            original_score: 1.0 - (original_rank as f32 * 0.1),
            original_rank,
        }
    }

    /// Mock cross-encoder that returns predetermined scores
    struct MockCrossEncoder {
        scores: Vec<f32>,
    }

    impl MockCrossEncoder {
        fn new(scores: Vec<f32>) -> Self {
            Self { scores }
        }
    }

    #[async_trait]
    impl CrossEncoderBackend for MockCrossEncoder {
        async fn score_pairs(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            // Return scores in order, cycling if needed
            Ok(documents
                .iter()
                .enumerate()
                .map(|(i, _)| self.scores.get(i).copied().unwrap_or(0.0))
                .collect())
        }

        fn model_id(&self) -> &str {
            "mock-encoder"
        }

        fn max_length(&self) -> usize {
            512
        }
    }

    /// Mock cross-encoder that always returns the same score
    struct ConstantScoreEncoder {
        score: f32,
    }

    #[async_trait]
    impl CrossEncoderBackend for ConstantScoreEncoder {
        async fn score_pairs(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            Ok(vec![self.score; documents.len()])
        }

        fn model_id(&self) -> &str {
            "constant-encoder"
        }

        fn max_length(&self) -> usize {
            512
        }
    }

    // ============================================================
    // Score Calculation Tests
    // ============================================================

    #[test]
    fn test_term_overlap_score_perfect_match() {
        // All query terms are in the document
        let score = HeuristicCrossEncoder::term_overlap_score(
            "rust programming language",
            "Rust is a systems programming language focused on safety.",
        );
        assert!(
            (score - 1.0).abs() < f32::EPSILON,
            "Expected 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_term_overlap_score_partial_match() {
        // Only some query terms match
        let score = HeuristicCrossEncoder::term_overlap_score(
            "rust python java",
            "Rust is a great programming language.",
        );
        // 1 out of 3 terms match
        let expected = 1.0 / 3.0;
        assert!(
            (score - expected).abs() < 0.01,
            "Expected ~{:.3}, got {:.3}",
            expected,
            score
        );
    }

    #[test]
    fn test_term_overlap_score_no_match() {
        let score =
            HeuristicCrossEncoder::term_overlap_score("quantum physics", "The cat sat on the mat.");
        assert!(score < f32::EPSILON, "Expected 0.0, got {}", score);
    }

    #[test]
    fn test_term_overlap_score_empty_query() {
        let score = HeuristicCrossEncoder::term_overlap_score("", "Some document text here.");
        assert!(
            score < f32::EPSILON,
            "Expected 0.0 for empty query, got {}",
            score
        );
    }

    #[test]
    fn test_term_overlap_score_whitespace_only_query() {
        let score =
            HeuristicCrossEncoder::term_overlap_score("   \t\n  ", "Some document text here.");
        assert!(
            score < f32::EPSILON,
            "Expected 0.0 for whitespace-only query, got {}",
            score
        );
    }

    #[test]
    fn test_term_overlap_score_empty_document() {
        let score = HeuristicCrossEncoder::term_overlap_score("machine learning", "");
        assert!(
            score < f32::EPSILON,
            "Expected 0.0 for empty document, got {}",
            score
        );
    }

    #[test]
    fn test_term_overlap_score_case_insensitive() {
        let score_lower = HeuristicCrossEncoder::term_overlap_score("rust", "rust is great");
        let score_upper = HeuristicCrossEncoder::term_overlap_score("RUST", "rust is great");
        let score_mixed = HeuristicCrossEncoder::term_overlap_score("RuSt", "RUST is great");

        assert!((score_lower - 1.0).abs() < f32::EPSILON);
        assert!((score_upper - 1.0).abs() < f32::EPSILON);
        assert!((score_mixed - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_term_overlap_score_duplicate_query_terms() {
        // Duplicate terms should be deduplicated
        let score =
            HeuristicCrossEncoder::term_overlap_score("rust rust rust", "Rust programming.");
        // HashSet deduplicates, so only 1 unique term
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_term_overlap_score_substring_match() {
        // "learn" should match within "learning"
        let score =
            HeuristicCrossEncoder::term_overlap_score("learn", "Machine learning is great.");
        assert!((score - 1.0).abs() < f32::EPSILON, "Substring should match");
    }

    // ============================================================
    // Result Reordering Tests
    // ============================================================

    #[tokio::test]
    async fn test_reranker_reorders_by_score() {
        // Use mock encoder to control exact scores
        let mock = MockCrossEncoder::new(vec![0.3, 0.9, 0.5]); // Second doc gets highest score
        let config = RerankerConfig::default();
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = vec![
            make_candidate("doc A", 0),
            make_candidate("doc B", 1),
            make_candidate("doc C", 2),
        ];

        let results = reranker.rerank("query", &candidates, 3).await.unwrap();

        // Should be reordered: B (0.9), C (0.5), A (0.3)
        assert_eq!(results[0].original_rank, 1, "Highest score should be first");
        assert_eq!(
            results[1].original_rank, 2,
            "Second highest should be second"
        );
        assert_eq!(results[2].original_rank, 0, "Lowest score should be last");

        // Verify new ranks are correct
        assert_eq!(results[0].new_rank, 0);
        assert_eq!(results[1].new_rank, 1);
        assert_eq!(results[2].new_rank, 2);
    }

    #[tokio::test]
    async fn test_rank_delta_calculation() {
        let mock = MockCrossEncoder::new(vec![0.1, 0.9, 0.5]); // Reverse order
        let config = RerankerConfig::default();
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = vec![
            make_candidate("doc A", 0), // Will move from 0 to 2 (delta = -2)
            make_candidate("doc B", 1), // Will move from 1 to 0 (delta = +1)
            make_candidate("doc C", 2), // Will move from 2 to 1 (delta = +1)
        ];

        let results = reranker.rerank("query", &candidates, 3).await.unwrap();

        // Find each doc by original_rank
        let doc_a = results.iter().find(|r| r.original_rank == 0).unwrap();
        let doc_b = results.iter().find(|r| r.original_rank == 1).unwrap();
        let doc_c = results.iter().find(|r| r.original_rank == 2).unwrap();

        assert_eq!(doc_b.rank_delta, 1, "Doc B moved up from 1 to 0");
        assert_eq!(doc_c.rank_delta, 1, "Doc C moved up from 2 to 1");
        assert_eq!(doc_a.rank_delta, -2, "Doc A moved down from 0 to 2");
    }

    #[tokio::test]
    async fn test_reranker_preserves_scores() {
        let mock = MockCrossEncoder::new(vec![0.75, 0.85, 0.95]);
        let config = RerankerConfig::default();
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = vec![
            make_candidate("doc A", 0),
            make_candidate("doc B", 1),
            make_candidate("doc C", 2),
        ];

        let results = reranker.rerank("query", &candidates, 3).await.unwrap();

        // Verify scores are preserved correctly
        for result in &results {
            match result.original_rank {
                0 => assert!((result.rerank_score - 0.75).abs() < 0.01),
                1 => assert!((result.rerank_score - 0.85).abs() < 0.01),
                2 => assert!((result.rerank_score - 0.95).abs() < 0.01),
                _ => panic!("Unexpected original_rank"),
            }
        }
    }

    // ============================================================
    // Cross-Encoder Integration Tests (Mock Backend)
    // ============================================================

    #[tokio::test]
    async fn test_custom_backend_integration() {
        let mock = MockCrossEncoder::new(vec![0.5, 0.8, 0.3]);
        let config = RerankerConfig::default();
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = create_test_candidates();
        let results = reranker.rerank("test", &candidates, 3).await.unwrap();

        // Verify the mock scores were used
        assert!((results[0].rerank_score - 0.8).abs() < 0.01); // Highest
        assert!((results[1].rerank_score - 0.5).abs() < 0.01);
        assert!((results[2].rerank_score - 0.3).abs() < 0.01); // Lowest
    }

    #[tokio::test]
    async fn test_backend_model_id() {
        let mock = MockCrossEncoder::new(vec![1.0]);
        assert_eq!(mock.model_id(), "mock-encoder");
        assert_eq!(mock.max_length(), 512);
    }

    #[tokio::test]
    async fn test_heuristic_backend_properties() {
        let heuristic = HeuristicCrossEncoder::new();
        assert_eq!(heuristic.model_id(), "heuristic-overlap");
        assert_eq!(heuristic.max_length(), 512);
    }

    // ============================================================
    // Edge Cases Tests
    // ============================================================

    #[tokio::test]
    async fn test_reranker_empty_candidates() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);

        let results = reranker.rerank("test query", &[], 10).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_reranker_single_candidate() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);

        let candidates = vec![make_candidate("single document about rust", 0)];

        let results = reranker.rerank("rust", &candidates, 10).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].new_rank, 0);
        assert_eq!(results[0].rank_delta, 0); // No change for single item
    }

    #[tokio::test]
    async fn test_reranker_identical_scores() {
        // All docs have the same score - order should be stable or consistent
        let mock = ConstantScoreEncoder { score: 0.5 };
        let config = RerankerConfig::default();
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = vec![
            make_candidate("doc A", 0),
            make_candidate("doc B", 1),
            make_candidate("doc C", 2),
        ];

        let results = reranker.rerank("query", &candidates, 3).await.unwrap();

        // All should have the same rerank_score
        assert!(results
            .iter()
            .all(|r| (r.rerank_score - 0.5).abs() < f32::EPSILON));

        // Verify we still get 3 results
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_reranker_top_k_less_than_candidates() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates(); // 3 candidates

        let results = reranker
            .rerank("machine learning", &candidates, 1)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_reranker_top_k_greater_than_candidates() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates(); // 3 candidates

        let results = reranker
            .rerank("machine learning", &candidates, 100)
            .await
            .unwrap();

        // Should return all 3, not 100
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_reranker_top_k_zero() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker
            .rerank("machine learning", &candidates, 0)
            .await
            .unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_reranker_empty_query() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker.rerank("", &candidates, 3).await.unwrap();

        // Empty query = 0 score for all (via heuristic encoder)
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.rerank_score < f32::EPSILON));
    }

    #[tokio::test]
    async fn test_reranker_very_long_text() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);

        // Create a very long document
        let long_text = "rust ".repeat(10000);
        let candidates = vec![make_candidate(&long_text, 0)];

        let results = reranker.rerank("rust", &candidates, 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].rerank_score > 0.9); // Should match "rust"
    }

    #[tokio::test]
    async fn test_reranker_special_characters() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);

        let candidates = vec![
            make_candidate("C++ and C# are programming languages", 0),
            make_candidate("@mention #hashtag $money", 1),
            make_candidate("Special chars: <>&\"'", 2),
        ];

        // Should not crash with special characters
        let results = reranker.rerank("C++", &candidates, 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_reranker_unicode_text() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);

        let candidates = vec![
            make_candidate("Rust", 0),
            make_candidate("Rust", 1),
            make_candidate("Rust in Japanese", 2),
        ];

        let results = reranker.rerank("Rust", &candidates, 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    // ============================================================
    // Threshold Filtering Tests
    // ============================================================

    #[tokio::test]
    async fn test_reranker_threshold_filters_low_scores() {
        let mock = MockCrossEncoder::new(vec![0.3, 0.6, 0.9]);
        let config = RerankerConfig {
            score_threshold: Some(0.5),
            ..Default::default()
        };
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = vec![
            make_candidate("low score doc", 0),
            make_candidate("medium score doc", 1),
            make_candidate("high score doc", 2),
        ];

        let results = reranker.rerank("query", &candidates, 10).await.unwrap();

        // Only scores >= 0.5 should remain
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.rerank_score >= 0.5));
    }

    #[tokio::test]
    async fn test_reranker_threshold_filters_all() {
        let mock = MockCrossEncoder::new(vec![0.1, 0.2, 0.3]);
        let config = RerankerConfig {
            score_threshold: Some(0.9), // Very high threshold
            ..Default::default()
        };
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = create_test_candidates();

        let results = reranker.rerank("query", &candidates, 10).await.unwrap();

        // All scores are below threshold
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_reranker_threshold_none_keeps_all() {
        let mock = MockCrossEncoder::new(vec![0.01, 0.02, 0.03]);
        let config = RerankerConfig {
            score_threshold: None,
            ..Default::default()
        };
        let reranker = Reranker::with_backend(config, Arc::new(mock));

        let candidates = create_test_candidates();

        let results = reranker.rerank("query", &candidates, 10).await.unwrap();

        // No threshold, all should be returned
        assert_eq!(results.len(), 3);
    }

    // ============================================================
    // Batched Reranking Tests
    // ============================================================

    #[tokio::test]
    async fn test_rerank_batched_small_set() {
        // When candidates <= batch_size, should behave like regular rerank
        let config = RerankerConfig {
            batch_size: 10, // Larger than our candidate set
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates(); // 3 candidates

        let results = reranker
            .rerank_batched("machine learning", &candidates, 3)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_rerank_batched_large_set() {
        let config = RerankerConfig {
            batch_size: 2, // Small batch size to force batching
            ..Default::default()
        };
        let reranker = Reranker::new(config);

        // Create 5 candidates (will need 3 batches with batch_size=2)
        let candidates = vec![
            make_candidate("rust programming language", 0),
            make_candidate("python scripting", 1),
            make_candidate("rust memory safety", 2),
            make_candidate("java enterprise", 3),
            make_candidate("rust concurrency", 4),
        ];

        let results = reranker
            .rerank_batched("rust", &candidates, 3)
            .await
            .unwrap();

        // Should return top 3 even though processed in batches
        assert_eq!(results.len(), 3);

        // All results should have "rust" in them (highest scores)
        assert!(results.iter().all(|r| r.text.contains("rust")));
    }

    #[tokio::test]
    async fn test_rerank_batched_preserves_global_ranking() {
        let config = RerankerConfig {
            batch_size: 2,
            ..Default::default()
        };
        let reranker = Reranker::new(config);

        let candidates = vec![
            make_candidate("rust rust rust", 0), // Score: 1.0
            make_candidate("no match here", 1),  // Score: 0.0
            make_candidate("rust rust", 2),      // Score: 1.0
            make_candidate("nothing", 3),        // Score: 0.0
        ];

        let results = reranker
            .rerank_batched("rust", &candidates, 4)
            .await
            .unwrap();

        // Rust-containing docs should come first
        assert!(results[0].text.contains("rust"));
        assert!(results[1].text.contains("rust"));
    }

    #[tokio::test]
    async fn test_rerank_batched_empty() {
        let config = RerankerConfig {
            batch_size: 2,
            ..Default::default()
        };
        let reranker = Reranker::new(config);

        let results = reranker.rerank_batched("query", &[], 10).await.unwrap();

        assert!(results.is_empty());
    }

    // ============================================================
    // Statistics Tests
    // ============================================================

    #[test]
    fn test_rerank_stats_empty_results() {
        let stats = RerankStats::from_results(&[], 100);

        assert_eq!(stats.candidates_count, 0);
        assert!(stats.avg_rerank_score.abs() < f32::EPSILON);
        assert!(stats.avg_rank_delta.abs() < f32::EPSILON);
        assert_eq!(stats.moved_up_count, 0);
        assert_eq!(stats.moved_down_count, 0);
        assert_eq!(stats.latency_ms, 100);
    }

    #[test]
    fn test_rerank_stats_single_result() {
        let results = vec![RerankedResult {
            id: Uuid::new_v4(),
            text: "test".to_string(),
            rerank_score: 0.8,
            original_score: 0.7,
            original_rank: 0,
            new_rank: 0,
            rank_delta: 0,
        }];

        let stats = RerankStats::from_results(&results, 25);

        assert_eq!(stats.candidates_count, 1);
        assert!((stats.avg_rerank_score - 0.8).abs() < f32::EPSILON);
        assert!(stats.avg_rank_delta.abs() < f32::EPSILON);
        assert_eq!(stats.moved_up_count, 0);
        assert_eq!(stats.moved_down_count, 0);
        assert_eq!(stats.latency_ms, 25);
    }

    #[test]
    fn test_rerank_stats_mixed_movement() {
        let results = vec![
            RerankedResult {
                id: Uuid::new_v4(),
                text: "a".to_string(),
                rerank_score: 0.9,
                original_score: 0.5,
                original_rank: 3,
                new_rank: 0,
                rank_delta: 3, // Moved up
            },
            RerankedResult {
                id: Uuid::new_v4(),
                text: "b".to_string(),
                rerank_score: 0.8,
                original_score: 0.9,
                original_rank: 0,
                new_rank: 1,
                rank_delta: -1, // Moved down
            },
            RerankedResult {
                id: Uuid::new_v4(),
                text: "c".to_string(),
                rerank_score: 0.7,
                original_score: 0.7,
                original_rank: 2,
                new_rank: 2,
                rank_delta: 0, // No change
            },
        ];

        let stats = RerankStats::from_results(&results, 50);

        assert_eq!(stats.candidates_count, 3);
        assert_eq!(stats.moved_up_count, 1);
        assert_eq!(stats.moved_down_count, 1);
        // avg_rank_delta = (3 + (-1) + 0) / 3 = 0.666...
        assert!((stats.avg_rank_delta - (2.0 / 3.0)).abs() < 0.01);
        // avg_rerank_score = (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert!((stats.avg_rerank_score - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_rerank_stats_all_moved_up() {
        let results = vec![
            RerankedResult {
                id: Uuid::new_v4(),
                text: "a".to_string(),
                rerank_score: 0.9,
                original_score: 0.5,
                original_rank: 2,
                new_rank: 0,
                rank_delta: 2,
            },
            RerankedResult {
                id: Uuid::new_v4(),
                text: "b".to_string(),
                rerank_score: 0.8,
                original_score: 0.6,
                original_rank: 3,
                new_rank: 1,
                rank_delta: 2,
            },
        ];

        let stats = RerankStats::from_results(&results, 30);

        assert_eq!(stats.moved_up_count, 2);
        assert_eq!(stats.moved_down_count, 0);
    }

    // ============================================================
    // Configuration Tests
    // ============================================================

    #[test]
    fn test_config_default() {
        let config = RerankerConfig::default();

        assert_eq!(config.model_id, "cross-encoder/ms-marco-MiniLM-L-12-v2");
        assert_eq!(config.max_length, 512);
        assert_eq!(config.batch_size, 16);
        assert!(!config.use_gpu);
        assert!(config.score_threshold.is_none());
        assert!(config.enable_cache);
    }

    #[test]
    fn test_config_fast() {
        let config = RerankerConfig::fast();

        assert_eq!(config.model_id, "cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert_eq!(config.max_length, 256);
        assert_eq!(config.batch_size, 32);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_config_quality() {
        let config = RerankerConfig::quality();

        assert_eq!(config.model_id, "cross-encoder/ms-marco-electra-base");
        assert_eq!(config.max_length, 512);
        assert_eq!(config.batch_size, 8);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_config_serialization() {
        let config = RerankerConfig {
            model_id: "test-model".to_string(),
            max_length: 256,
            batch_size: 8,
            use_gpu: true,
            score_threshold: Some(0.5),
            enable_cache: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RerankerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.model_id, "test-model");
        assert_eq!(deserialized.max_length, 256);
        assert_eq!(deserialized.batch_size, 8);
        assert!(deserialized.use_gpu);
        assert_eq!(deserialized.score_threshold, Some(0.5));
        assert!(!deserialized.enable_cache);
    }

    #[tokio::test]
    async fn test_reranker_config_accessor() {
        let config = RerankerConfig {
            model_id: "my-custom-model".to_string(),
            ..Default::default()
        };
        let reranker = Reranker::new(config.clone());

        assert_eq!(reranker.config().model_id, "my-custom-model");
    }

    // ============================================================
    // RerankedResult Serialization Tests
    // ============================================================

    #[test]
    fn test_reranked_result_serialization() {
        let id = Uuid::new_v4();
        let result = RerankedResult {
            id,
            text: "Test document".to_string(),
            rerank_score: 0.85,
            original_score: 0.7,
            original_rank: 3,
            new_rank: 1,
            rank_delta: 2,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: RerankedResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, id);
        assert_eq!(deserialized.text, "Test document");
        assert!((deserialized.rerank_score - 0.85).abs() < f32::EPSILON);
        assert_eq!(deserialized.original_rank, 3);
        assert_eq!(deserialized.new_rank, 1);
        assert_eq!(deserialized.rank_delta, 2);
    }

    // ============================================================
    // Original Tests (preserved from existing code)
    // ============================================================

    #[tokio::test]
    async fn test_reranker_basic() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker
            .rerank("machine learning AI", &candidates, 3)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);

        // First result should be the most relevant (machine learning)
        assert!(results[0].text.contains("Machine learning"));

        // Weather result should be last (least relevant to query)
        assert!(results[2].text.contains("weather"));
    }

    #[tokio::test]
    async fn test_reranker_top_k() {
        let config = RerankerConfig::default();
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker
            .rerank("machine learning", &candidates, 1)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_reranker_with_threshold() {
        let config = RerankerConfig {
            score_threshold: Some(0.5),
            ..Default::default()
        };
        let reranker = Reranker::new(config);
        let candidates = create_test_candidates();

        let results = reranker
            .rerank("machine learning", &candidates, 10)
            .await
            .unwrap();

        // Only results above threshold should be returned
        assert!(results.iter().all(|r| r.rerank_score >= 0.5));
    }

    #[test]
    fn test_heuristic_overlap() {
        let score = HeuristicCrossEncoder::term_overlap_score(
            "machine learning neural",
            "Machine learning is a powerful technique for neural networks.",
        );

        // Should have high overlap
        assert!(score > 0.5);

        let low_score = HeuristicCrossEncoder::term_overlap_score(
            "machine learning",
            "The weather is sunny today.",
        );

        // Should have low overlap
        assert!(low_score < 0.2);
    }

    #[test]
    fn test_rerank_stats() {
        let results = vec![
            RerankedResult {
                id: Uuid::new_v4(),
                text: "Test".to_string(),
                rerank_score: 0.9,
                original_score: 0.8,
                original_rank: 2,
                new_rank: 0,
                rank_delta: 2,
            },
            RerankedResult {
                id: Uuid::new_v4(),
                text: "Test 2".to_string(),
                rerank_score: 0.7,
                original_score: 0.9,
                original_rank: 0,
                new_rank: 1,
                rank_delta: -1,
            },
        ];

        let stats = RerankStats::from_results(&results, 50);

        assert_eq!(stats.candidates_count, 2);
        assert_eq!(stats.moved_up_count, 1);
        assert_eq!(stats.moved_down_count, 1);
        assert_eq!(stats.latency_ms, 50);
    }

    #[test]
    fn test_config_presets() {
        let fast = RerankerConfig::fast();
        assert!(fast.max_length < 512);
        assert!(fast.batch_size > 16);

        let quality = RerankerConfig::quality();
        assert!(quality.use_gpu);
    }
}
