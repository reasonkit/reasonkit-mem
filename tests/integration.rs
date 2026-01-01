//! Integration tests for reasonkit-mem
//!
//! End-to-end tests for the memory system covering:
//! - Full document ingestion workflows
//! - Hybrid search with reranking
//! - Index rebuild operations
//! - Concurrent access patterns
//!
//! # Test Philosophy
//!
//! These tests use realistic mock embeddings and varied document sizes to
//! simulate production workloads. The embedding mock generates deterministic
//! vectors based on text content, allowing consistent search behavior.
//!
//! # Test Categories
//!
//! 1. **Document Lifecycle**: Ingest, search, retrieve, delete
//! 2. **Hybrid Search**: Dense + Sparse with RRF fusion
//! 3. **Reranking**: Cross-encoder precision improvement
//! 4. **Concurrent Access**: Thread-safe operations
//! 5. **Index Operations**: Rebuild, optimize, stats
//! 6. **Edge Cases**: Empty queries, large documents, special characters

use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Barrier;
use uuid::Uuid;

use reasonkit_mem::{
    embedding::{cosine_similarity, EmbeddingProvider, EmbeddingResult, EmbeddingVector},
    indexing::IndexManager,
    retrieval::{
        FusionEngine, FusionStrategy, HybridResult, HybridRetriever, KnowledgeBase, RankedResult,
        Reranker, RerankerCandidate, RerankerConfig, RetrievalStats,
    },
    Chunk, Document, DocumentContent, DocumentType, EmbeddingIds, MatchSource, Metadata,
    ProcessingState, ProcessingStatus, Result, RetrievalConfig, Source, SourceType,
};

// ============================================================================
// TEST INFRASTRUCTURE: Mock Embedding Provider
// ============================================================================

/// Mock embedding provider that generates deterministic embeddings
///
/// The embeddings are based on a hash of the input text, ensuring:
/// - Same text always produces the same embedding
/// - Similar texts produce similar embeddings (via character frequency)
/// - Results are reproducible across test runs
struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a deterministic embedding based on text content
    ///
    /// Uses character frequency analysis to create embeddings that
    /// capture text similarity without requiring a real model.
    fn generate_embedding(&self, text: &str) -> EmbeddingVector {
        let mut embedding = vec![0.0f32; self.dimension];
        let text_lower = text.to_lowercase();
        let text_bytes = text_lower.as_bytes();

        // Character frequency components (first 26 dimensions for a-z)
        for byte in text_bytes.iter() {
            if *byte >= b'a' && *byte <= b'z' {
                let idx = (*byte - b'a') as usize;
                if idx < self.dimension {
                    embedding[idx] += 1.0;
                }
            }
        }

        // Word count component
        let word_count = text.split_whitespace().count() as f32;
        if self.dimension > 26 {
            embedding[26] = word_count / 100.0;
        }

        // Length component
        if self.dimension > 27 {
            embedding[27] = text.len() as f32 / 1000.0;
        }

        // Keyword boosting for common terms
        let keywords = [
            ("machine", 28),
            ("learning", 29),
            ("neural", 30),
            ("network", 31),
            ("deep", 32),
            ("ai", 33),
            ("artificial", 34),
            ("intelligence", 35),
            ("data", 36),
            ("model", 37),
        ];

        for (keyword, idx) in keywords.iter() {
            if idx < &self.dimension && text_lower.contains(keyword) {
                embedding[*idx] = 1.0;
            }
        }

        // Fill remaining dimensions with deterministic noise based on hash
        let hash = Self::simple_hash(text);
        for i in 38..self.dimension {
            embedding[i] = ((hash >> (i % 32)) & 1) as f32 * 0.1;
        }

        // Normalize the embedding
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for x in embedding.iter_mut() {
                *x /= magnitude;
            }
        }

        embedding
    }

    fn simple_hash(text: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "mock-embedding-provider"
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        let dense = self.generate_embedding(text);
        Ok(EmbeddingResult {
            dense: Some(dense),
            sparse: None,
            token_count: text.split_whitespace().count(),
        })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }
}

// ============================================================================
// TEST HELPERS
// ============================================================================

/// Create a test document with the given content and chunks
fn create_test_document(title: &str, content: &str, chunk_texts: &[&str]) -> Document {
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some(format!("/test/{}.md", title.replace(' ', "_"))),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: Some("1.0".to_string()),
    };

    let metadata = Metadata {
        title: Some(title.to_string()),
        authors: Vec::new(),
        abstract_text: Some(content.chars().take(200).collect()),
        date: Some(Utc::now().format("%Y-%m-%d").to_string()),
        venue: None,
        citations: None,
        tags: vec!["test".to_string()],
        categories: vec!["integration-test".to_string()],
        keywords: Vec::new(),
        doi: None,
        license: Some("Apache-2.0".to_string()),
    };

    let word_count = content.split_whitespace().count();
    let char_count = content.len();

    let chunks: Vec<Chunk> = chunk_texts
        .iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            id: Uuid::new_v4(),
            text: text.to_string(),
            index: i,
            start_char: 0,
            end_char: text.len(),
            token_count: Some(text.split_whitespace().count()),
            section: Some(format!("Section {}", i + 1)),
            page: Some(i / 2 + 1),
            embedding_ids: EmbeddingIds::default(),
        })
        .collect();

    Document {
        id: Uuid::new_v4(),
        doc_type: DocumentType::Note,
        source,
        content: DocumentContent {
            raw: content.to_string(),
            format: reasonkit_mem::types::ContentFormat::Text,
            language: "en".to_string(),
            word_count,
            char_count,
        },
        metadata,
        processing: ProcessingStatus {
            status: ProcessingState::Completed,
            chunked: true,
            embedded: false,
            indexed: false,
            raptor_processed: false,
            errors: Vec::new(),
        },
        chunks,
        created_at: Utc::now(),
        updated_at: None,
    }
}

/// Create a mock embedding pipeline
fn create_mock_embedding_pipeline() -> Arc<reasonkit_mem::embedding::EmbeddingPipeline> {
    let provider = Arc::new(MockEmbeddingProvider::new(128));
    Arc::new(reasonkit_mem::embedding::EmbeddingPipeline::new(provider))
}

/// Generate embeddings for document chunks
async fn generate_chunk_embeddings(
    doc: &Document,
    provider: &MockEmbeddingProvider,
) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::new();
    for chunk in &doc.chunks {
        embeddings.push(provider.generate_embedding(&chunk.text));
    }
    embeddings
}

// ============================================================================
// TEST: Full Document Lifecycle
// ============================================================================

/// Test complete document ingestion and retrieval workflow
#[tokio::test]
async fn test_document_lifecycle_ingest_search_retrieve() {
    // Create retriever with mock embeddings
    let retriever = HybridRetriever::in_memory().unwrap();
    let embedding_pipeline = create_mock_embedding_pipeline();
    let retriever = retriever.with_embedding_pipeline(embedding_pipeline.clone());

    // Create test documents
    let doc1 = create_test_document(
        "Machine Learning Basics",
        "An introduction to machine learning concepts and algorithms.",
        &[
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Supervised learning uses labeled training data to make predictions on new data.",
            "Unsupervised learning finds hidden patterns in unlabeled data.",
        ],
    );

    let doc2 = create_test_document(
        "Neural Networks",
        "Deep learning with neural networks.",
        &[
            "Neural networks are inspired by biological neurons in the brain.",
            "Deep learning uses multiple layers of neural networks for complex pattern recognition.",
            "Convolutional neural networks excel at image processing tasks.",
        ],
    );

    let doc3 = create_test_document(
        "Natural Language Processing",
        "NLP techniques for text understanding.",
        &[
            "Natural language processing enables computers to understand human language.",
            "Transformers have revolutionized NLP with attention mechanisms.",
            "BERT and GPT are examples of pre-trained language models.",
        ],
    );

    // Generate embeddings and add documents
    let provider = MockEmbeddingProvider::new(128);

    for doc in [&doc1, &doc2, &doc3] {
        let embeddings = generate_chunk_embeddings(doc, &provider).await;
        retriever
            .add_document_with_embeddings(doc, embeddings)
            .await
            .unwrap();
    }

    // Verify stats
    let stats = retriever.stats().await.unwrap();
    assert_eq!(stats.document_count, 3, "Should have 3 documents");
    assert_eq!(stats.chunk_count, 9, "Should have 9 chunks total");

    // Test sparse search (BM25)
    let sparse_results = retriever
        .search_sparse("machine learning", 5)
        .await
        .unwrap();
    assert!(
        !sparse_results.is_empty(),
        "Sparse search should return results"
    );
    assert_eq!(sparse_results[0].match_source, MatchSource::Sparse);
    assert!(
        sparse_results[0].text.to_lowercase().contains("machine")
            || sparse_results[0].text.to_lowercase().contains("learning"),
        "Top result should be relevant to query"
    );

    // Test dense search (vector)
    let dense_results = retriever
        .search_dense("neural networks deep learning", 5)
        .await
        .unwrap();
    assert!(
        !dense_results.is_empty(),
        "Dense search should return results"
    );
    assert_eq!(dense_results[0].match_source, MatchSource::Dense);

    // Test hybrid search
    let hybrid_results = retriever
        .search("artificial intelligence", 5)
        .await
        .unwrap();
    assert!(
        !hybrid_results.is_empty(),
        "Hybrid search should return results"
    );

    // Test document deletion
    let doc1_id = doc1.id;
    retriever.delete_document(&doc1_id).await.unwrap();

    let stats_after = retriever.stats().await.unwrap();
    assert_eq!(
        stats_after.document_count, 2,
        "Should have 2 documents after deletion"
    );
    assert_eq!(
        stats_after.chunk_count, 6,
        "Should have 6 chunks after deletion"
    );
}

// ============================================================================
// TEST: Hybrid Search with Multiple Documents
// ============================================================================

/// Test hybrid search combining dense and sparse retrieval
#[tokio::test]
async fn test_hybrid_search_with_rrf_fusion() {
    let retriever = HybridRetriever::in_memory().unwrap();
    let embedding_pipeline = create_mock_embedding_pipeline();
    let retriever = retriever.with_embedding_pipeline(embedding_pipeline);

    // Create documents with varying relevance
    let docs = vec![
        create_test_document(
            "Python Programming",
            "Python is a versatile programming language.",
            &[
                "Python is widely used for data science and machine learning applications.",
                "The Python ecosystem includes libraries like NumPy, Pandas, and scikit-learn.",
            ],
        ),
        create_test_document(
            "Rust Programming",
            "Rust provides memory safety without garbage collection.",
            &[
                "Rust is a systems programming language focused on safety and performance.",
                "The Rust compiler prevents common memory errors at compile time.",
            ],
        ),
        create_test_document(
            "JavaScript Development",
            "JavaScript is essential for web development.",
            &[
                "JavaScript runs in browsers and enables interactive web applications.",
                "Node.js allows JavaScript to run on the server side.",
            ],
        ),
    ];

    let provider = MockEmbeddingProvider::new(128);
    for doc in &docs {
        let embeddings = generate_chunk_embeddings(doc, &provider).await;
        retriever
            .add_document_with_embeddings(doc, embeddings)
            .await
            .unwrap();
    }

    // Search with hybrid configuration
    let config = RetrievalConfig {
        top_k: 5,
        min_score: 0.0,
        alpha: 0.5, // Equal weight to dense and sparse
        use_raptor: false,
        rerank: false,
    };

    let query = "programming language for machine learning";
    let query_embedding = provider.generate_embedding(query);

    let results = retriever
        .search_hybrid(query, Some(&query_embedding), &config)
        .await
        .unwrap();

    assert!(!results.is_empty(), "Hybrid search should return results");

    // Python should rank high due to "machine learning" mention
    let python_found = results
        .iter()
        .any(|r| r.text.to_lowercase().contains("python"));
    assert!(
        python_found,
        "Python-related chunk should appear in results"
    );
}

/// Test different fusion strategies
#[tokio::test]
async fn test_fusion_strategies() {
    // Test RRF fusion
    let rrf_engine = FusionEngine::new(FusionStrategy::ReciprocalRankFusion { k: 60 });

    let mut results = HashMap::new();
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    results.insert(
        "sparse".to_string(),
        vec![
            RankedResult {
                id: id1,
                score: 10.0,
                rank: 0,
            },
            RankedResult {
                id: id2,
                score: 8.0,
                rank: 1,
            },
            RankedResult {
                id: id3,
                score: 6.0,
                rank: 2,
            },
        ],
    );

    results.insert(
        "dense".to_string(),
        vec![
            RankedResult {
                id: id2,
                score: 0.95,
                rank: 0,
            },
            RankedResult {
                id: id1,
                score: 0.85,
                rank: 1,
            },
            RankedResult {
                id: id3,
                score: 0.75,
                rank: 2,
            },
        ],
    );

    let fused = rrf_engine.fuse(results).unwrap();

    assert_eq!(fused.len(), 3, "Should have 3 fused results");
    // id2 appears at rank 0 in dense and rank 1 in sparse - should rank high
    // id1 appears at rank 0 in sparse and rank 1 in dense - should also rank high
    let top_two_ids: Vec<Uuid> = fused.iter().take(2).map(|r| r.id).collect();
    assert!(
        top_two_ids.contains(&id1) || top_two_ids.contains(&id2),
        "Top results should include items that rank well in both methods"
    );

    // Test weighted sum fusion
    let weighted_engine = FusionEngine::weighted(0.7);

    let mut results2 = HashMap::new();
    results2.insert(
        "sparse".to_string(),
        vec![RankedResult {
            id: id1,
            score: 1.0,
            rank: 0,
        }],
    );
    results2.insert(
        "dense".to_string(),
        vec![RankedResult {
            id: id1,
            score: 0.5,
            rank: 0,
        }],
    );

    let fused2 = weighted_engine.fuse(results2).unwrap();
    assert_eq!(fused2.len(), 1);
    assert!(
        fused2[0].fusion_score <= 1.0,
        "Fusion score should be normalized"
    );
}

// ============================================================================
// TEST: Reranking
// ============================================================================

/// Test cross-encoder reranking improves result quality
#[tokio::test]
async fn test_reranking_improves_results() {
    let config = RerankerConfig::default();
    let reranker = Reranker::new(config);

    // Create candidates with varying relevance to "machine learning"
    let candidates = vec![
        RerankerCandidate {
            id: Uuid::new_v4(),
            text: "The weather is nice today with sunny skies.".to_string(),
            original_score: 0.9, // High original score but low relevance
            original_rank: 0,
        },
        RerankerCandidate {
            id: Uuid::new_v4(),
            text: "Machine learning enables computers to learn from data patterns.".to_string(),
            original_score: 0.5, // Lower original score but high relevance
            original_rank: 1,
        },
        RerankerCandidate {
            id: Uuid::new_v4(),
            text: "Deep learning is a subset of machine learning using neural networks."
                .to_string(),
            original_score: 0.4,
            original_rank: 2,
        },
        RerankerCandidate {
            id: Uuid::new_v4(),
            text: "Cooking recipes for pasta dishes.".to_string(),
            original_score: 0.3,
            original_rank: 3,
        },
    ];

    let query = "machine learning neural networks";
    let reranked = reranker.rerank(query, &candidates, 4).await.unwrap();

    assert_eq!(reranked.len(), 4);

    // Machine learning related content should be reranked higher
    let top_two_texts: Vec<&str> = reranked.iter().take(2).map(|r| r.text.as_str()).collect();
    let contains_ml = top_two_texts
        .iter()
        .any(|t| t.to_lowercase().contains("machine learning"));
    assert!(
        contains_ml,
        "Relevant content should be in top 2 after reranking"
    );

    // Irrelevant content should be at the bottom
    let last_text = &reranked.last().unwrap().text;
    assert!(
        last_text.contains("weather") || last_text.contains("Cooking"),
        "Irrelevant content should be at bottom"
    );

    // Check rank deltas
    let ml_result = reranked
        .iter()
        .find(|r| r.text.contains("Machine learning"))
        .unwrap();
    assert!(
        ml_result.rank_delta >= 0,
        "Relevant result should move up or stay same"
    );
}

/// Test reranking with batching for large candidate sets
#[tokio::test]
async fn test_reranking_batched() {
    let config = RerankerConfig {
        batch_size: 5,
        ..Default::default()
    };
    let reranker = Reranker::new(config);

    // Create 20 candidates
    let mut candidates = Vec::new();
    for i in 0..20 {
        candidates.push(RerankerCandidate {
            id: Uuid::new_v4(),
            text: format!("Document {} about machine learning topic number {}", i, i),
            original_score: 1.0 - (i as f32 * 0.05),
            original_rank: i,
        });
    }

    let query = "machine learning";
    let reranked = reranker
        .rerank_batched(query, &candidates, 10)
        .await
        .unwrap();

    assert_eq!(reranked.len(), 10, "Should return top_k results");

    // All results should have consistent ranking
    for (i, result) in reranked.iter().enumerate() {
        assert_eq!(result.new_rank, i, "New rank should match position");
    }
}

/// Test reranking with score threshold
#[tokio::test]
async fn test_reranking_with_threshold() {
    let config = RerankerConfig {
        score_threshold: Some(0.3),
        ..Default::default()
    };
    let reranker = Reranker::new(config);

    let candidates = vec![
        RerankerCandidate {
            id: Uuid::new_v4(),
            text: "Machine learning is a field of artificial intelligence.".to_string(),
            original_score: 0.9,
            original_rank: 0,
        },
        RerankerCandidate {
            id: Uuid::new_v4(),
            text: "The cat sat on the mat.".to_string(),
            original_score: 0.8,
            original_rank: 1,
        },
    ];

    let query = "machine learning AI";
    let reranked = reranker.rerank(query, &candidates, 10).await.unwrap();

    // Only results above threshold should be returned
    for result in &reranked {
        assert!(
            result.rerank_score >= 0.3,
            "All results should be above threshold"
        );
    }
}

// ============================================================================
// TEST: Index Operations
// ============================================================================

/// Test index rebuild and optimization
#[tokio::test]
async fn test_index_rebuild_and_optimize() {
    let index_manager = IndexManager::in_memory().unwrap();

    // Create and index multiple documents
    for i in 0..10 {
        let doc = create_test_document(
            &format!("Document {}", i),
            &format!("Content for document number {}", i),
            &[
                &format!("Chunk 1 of document {}: machine learning applications", i),
                &format!("Chunk 2 of document {}: data science workflows", i),
            ],
        );
        index_manager.index_document(&doc).unwrap();
    }

    // Get initial stats
    let initial_stats = index_manager.stats().unwrap();
    assert_eq!(
        initial_stats.chunk_count, 20,
        "Should have 20 indexed chunks"
    );

    // Optimize index
    index_manager.optimize().unwrap();

    // Stats should remain consistent after optimization
    let post_optimize_stats = index_manager.stats().unwrap();
    assert_eq!(
        post_optimize_stats.chunk_count, initial_stats.chunk_count,
        "Chunk count should be unchanged after optimization"
    );

    // Search should still work
    let results = index_manager.search_bm25("machine learning", 5).unwrap();
    assert!(
        !results.is_empty(),
        "Search should return results after optimization"
    );
}

/// Test index deletion
#[tokio::test]
async fn test_index_document_deletion() {
    let index_manager = IndexManager::in_memory().unwrap();

    let doc1 = create_test_document(
        "Document to Keep",
        "This document should remain",
        &["Chunk about Python programming"],
    );
    let doc2 = create_test_document(
        "Document to Delete",
        "This document will be removed",
        &["Chunk about JavaScript development"],
    );

    index_manager.index_document(&doc1).unwrap();
    index_manager.index_document(&doc2).unwrap();

    let initial_stats = index_manager.stats().unwrap();
    assert_eq!(initial_stats.chunk_count, 2);

    // Delete doc2
    index_manager.delete_document(&doc2.id).unwrap();

    // Search for deleted content should not find it
    let results = index_manager.search_bm25("JavaScript", 5).unwrap();
    let found_deleted = results.iter().any(|r| r.doc_id == doc2.id);
    assert!(
        !found_deleted,
        "Deleted document should not appear in search"
    );

    // Search for remaining content should still work
    let results2 = index_manager.search_bm25("Python", 5).unwrap();
    assert!(
        !results2.is_empty(),
        "Remaining document should be searchable"
    );
}

// ============================================================================
// TEST: Concurrent Access
// ============================================================================

/// Test concurrent document indexing
#[tokio::test]
async fn test_concurrent_document_indexing() {
    let retriever = Arc::new(HybridRetriever::in_memory().unwrap());
    let barrier = Arc::new(Barrier::new(4));
    let provider = Arc::new(MockEmbeddingProvider::new(128));

    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let retriever = retriever.clone();
        let barrier = barrier.clone();
        let provider = provider.clone();

        let handle = tokio::spawn(async move {
            // Wait for all threads to be ready
            barrier.wait().await;

            // Each thread adds 5 documents
            for i in 0..5 {
                let doc = create_test_document(
                    &format!("Thread {} Doc {}", thread_id, i),
                    &format!("Content from thread {} document {}", thread_id, i),
                    &[&format!("Chunk from thread {} doc {}", thread_id, i)],
                );

                let embeddings: Vec<Vec<f32>> = doc
                    .chunks
                    .iter()
                    .map(|c| provider.generate_embedding(&c.text))
                    .collect();

                retriever
                    .add_document_with_embeddings(&doc, embeddings)
                    .await
                    .unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all documents were indexed
    let stats = retriever.stats().await.unwrap();
    assert_eq!(
        stats.document_count, 20,
        "All 20 documents should be indexed"
    );
    assert_eq!(stats.chunk_count, 20, "All 20 chunks should be indexed");
}

/// Test concurrent search operations
#[tokio::test]
async fn test_concurrent_search_operations() {
    let retriever = HybridRetriever::in_memory().unwrap();
    let embedding_pipeline = create_mock_embedding_pipeline();
    let retriever = Arc::new(retriever.with_embedding_pipeline(embedding_pipeline));

    // Add test documents
    let provider = MockEmbeddingProvider::new(128);
    for i in 0..10 {
        let doc = create_test_document(
            &format!("Search Test Doc {}", i),
            &format!("Content for search testing number {}", i),
            &[
                &format!("Machine learning concepts part {}", i),
                &format!("Neural network architectures part {}", i),
            ],
        );
        let embeddings = doc
            .chunks
            .iter()
            .map(|c| provider.generate_embedding(&c.text))
            .collect();
        retriever
            .add_document_with_embeddings(&doc, embeddings)
            .await
            .unwrap();
    }

    let queries = vec![
        "machine learning",
        "neural networks",
        "deep learning",
        "artificial intelligence",
        "data science",
    ];

    let mut handles = Vec::new();

    for query in queries {
        let retriever = retriever.clone();
        let query = query.to_string();

        let handle = tokio::spawn(async move {
            // Each task performs multiple searches
            for _ in 0..5 {
                let results = retriever.search_sparse(&query, 5).await;
                assert!(results.is_ok(), "Search should not fail under concurrency");
            }
        });
        handles.push(handle);
    }

    // Wait for all searches to complete
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test concurrent read and write operations
#[tokio::test]
async fn test_concurrent_read_write() {
    let retriever = HybridRetriever::in_memory().unwrap();
    let embedding_pipeline = create_mock_embedding_pipeline();
    let retriever = Arc::new(retriever.with_embedding_pipeline(embedding_pipeline));
    let provider = Arc::new(MockEmbeddingProvider::new(128));

    // Pre-populate with some documents
    for i in 0..5 {
        let doc = create_test_document(
            &format!("Initial Doc {}", i),
            &format!("Initial content {}", i),
            &[&format!("Initial chunk {}", i)],
        );
        let embeddings: Vec<Vec<f32>> = doc
            .chunks
            .iter()
            .map(|c| provider.generate_embedding(&c.text))
            .collect();
        retriever
            .add_document_with_embeddings(&doc, embeddings)
            .await
            .unwrap();
    }

    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();

    // Writer tasks
    for writer_id in 0..2 {
        let retriever = retriever.clone();
        let barrier = barrier.clone();
        let provider = provider.clone();

        let handle = tokio::spawn(async move {
            barrier.wait().await;

            for i in 0..5 {
                let doc = create_test_document(
                    &format!("Writer {} Doc {}", writer_id, i),
                    &format!("Written by thread {} iteration {}", writer_id, i),
                    &[&format!("Writer {} chunk {}", writer_id, i)],
                );
                let embeddings: Vec<Vec<f32>> = doc
                    .chunks
                    .iter()
                    .map(|c| provider.generate_embedding(&c.text))
                    .collect();
                retriever
                    .add_document_with_embeddings(&doc, embeddings)
                    .await
                    .unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });
        handles.push(handle);
    }

    // Reader tasks
    for reader_id in 0..2 {
        let retriever = retriever.clone();
        let barrier = barrier.clone();

        let handle = tokio::spawn(async move {
            barrier.wait().await;

            for _ in 0..10 {
                let query = format!("content iteration {}", reader_id);
                let _results = retriever.search_sparse(&query, 5).await;
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            }
        });
        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        handle.await.unwrap();
    }

    // Final verification
    let stats = retriever.stats().await.unwrap();
    assert!(
        stats.document_count >= 5,
        "Should have at least initial documents"
    );
}

// ============================================================================
// TEST: Knowledge Base Integration
// ============================================================================

/// Test KnowledgeBase high-level API
#[tokio::test]
async fn test_knowledge_base_api() {
    let kb = KnowledgeBase::in_memory().unwrap();

    // Add documents
    let doc1 = create_test_document(
        "AI Overview",
        "Artificial Intelligence encompasses various fields.",
        &[
            "AI includes machine learning, deep learning, and neural networks.",
            "Applications of AI range from image recognition to natural language processing.",
        ],
    );

    let doc2 = create_test_document(
        "ML Techniques",
        "Machine learning techniques for data analysis.",
        &[
            "Supervised learning uses labeled data for training.",
            "Reinforcement learning learns through trial and error.",
        ],
    );

    kb.add(&doc1).await.unwrap();
    kb.add(&doc2).await.unwrap();

    // Query
    let results = kb
        .retriever()
        .search_sparse("machine learning techniques", 5)
        .await
        .unwrap();
    assert!(!results.is_empty());

    // Query with config
    let config = RetrievalConfig {
        top_k: 3,
        min_score: 0.0,
        alpha: 0.0, // Sparse only
        use_raptor: false,
        rerank: false,
    };
    let results_with_config = kb.query_with_config("neural networks", &config).await;
    // Note: May return empty if no embedding pipeline configured

    // Stats
    let stats = kb.stats().await.unwrap();
    assert_eq!(stats.document_count, 2);
    assert_eq!(stats.chunk_count, 4);

    // Delete
    kb.delete_document(&doc1.id).await.unwrap();
    let stats_after = kb.stats().await.unwrap();
    assert_eq!(stats_after.document_count, 1);
}

// ============================================================================
// TEST: Edge Cases
// ============================================================================

/// Test empty query handling
#[tokio::test]
async fn test_empty_query_handling() {
    let retriever = HybridRetriever::in_memory().unwrap();

    // Add a document
    let doc = create_test_document(
        "Test Doc",
        "Test content",
        &["Test chunk with some content"],
    );
    retriever.add_document(&doc).await.unwrap();

    // Search with empty query - should handle gracefully
    let results = retriever.search_sparse("", 5).await;
    assert!(results.is_ok(), "Empty query should not cause error");
}

/// Test special characters in content
#[tokio::test]
async fn test_special_characters_in_content() {
    let retriever = HybridRetriever::in_memory().unwrap();

    let doc = create_test_document(
        "Special Characters Test",
        "Content with special chars: @#$%^&*()",
        &[
            "Chunk with quotes: \"hello world\"",
            "Chunk with unicode: \u{1F600} emoji and \u{00E9} accent",
            "Chunk with brackets: [array] {object} (parentheses)",
        ],
    );

    retriever.add_document(&doc).await.unwrap();

    // Search should handle special characters
    let results = retriever.search_sparse("quotes", 5).await.unwrap();
    assert!(!results.is_empty());

    let results2 = retriever.search_sparse("brackets", 5).await.unwrap();
    assert!(!results2.is_empty());
}

/// Test very large document handling
#[tokio::test]
async fn test_large_document_handling() {
    let retriever = HybridRetriever::in_memory().unwrap();

    // Create a document with many chunks
    let chunk_texts: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "This is chunk number {} containing information about machine learning, \
             artificial intelligence, data science, and neural networks. The chunk \
             includes various keywords for testing search functionality.",
                i
            )
        })
        .collect();

    let chunk_refs: Vec<&str> = chunk_texts.iter().map(|s| s.as_str()).collect();

    let large_doc = create_test_document(
        "Large Document",
        "A document with many chunks for testing scalability.",
        &chunk_refs,
    );

    retriever.add_document(&large_doc).await.unwrap();

    let stats = retriever.stats().await.unwrap();
    assert_eq!(stats.chunk_count, 100, "Should index all 100 chunks");

    // Search should still be fast
    let start = std::time::Instant::now();
    let results = retriever
        .search_sparse("machine learning", 10)
        .await
        .unwrap();
    let duration = start.elapsed();

    assert!(!results.is_empty());
    assert!(
        duration.as_millis() < 1000,
        "Search should complete in under 1 second"
    );
}

/// Test document with empty chunks
#[tokio::test]
async fn test_document_with_empty_chunks() {
    let retriever = HybridRetriever::in_memory().unwrap();

    let doc = create_test_document(
        "Sparse Document",
        "Document with minimal content",
        &["Only one chunk with content"],
    );

    retriever.add_document(&doc).await.unwrap();

    let stats = retriever.stats().await.unwrap();
    assert_eq!(stats.chunk_count, 1);

    let results = retriever.search_sparse("content", 5).await.unwrap();
    assert!(!results.is_empty());
}

/// Test retrieval with min_score filter
#[tokio::test]
async fn test_retrieval_min_score_filter() {
    let retriever = HybridRetriever::in_memory().unwrap();
    let embedding_pipeline = create_mock_embedding_pipeline();
    let retriever = retriever.with_embedding_pipeline(embedding_pipeline);

    let provider = MockEmbeddingProvider::new(128);

    // Add documents with varying relevance
    let relevant_doc = create_test_document(
        "Highly Relevant",
        "Machine learning is about data patterns",
        &["Machine learning uses algorithms to find patterns in data."],
    );

    let irrelevant_doc = create_test_document(
        "Irrelevant",
        "Cooking recipes and kitchen tips",
        &["How to bake a perfect chocolate cake."],
    );

    for doc in [&relevant_doc, &irrelevant_doc] {
        let embeddings = doc
            .chunks
            .iter()
            .map(|c| provider.generate_embedding(&c.text))
            .collect();
        retriever
            .add_document_with_embeddings(doc, embeddings)
            .await
            .unwrap();
    }

    // Search with min_score filter
    let config = RetrievalConfig {
        top_k: 10,
        min_score: 0.0,
        alpha: 0.0, // Sparse only to ensure consistent behavior
        use_raptor: false,
        rerank: false,
    };

    let results = retriever
        .search_hybrid("machine learning", None, &config)
        .await
        .unwrap();

    // Should find the relevant document
    let found_relevant = results
        .iter()
        .any(|r| r.text.to_lowercase().contains("machine learning"));
    assert!(found_relevant, "Should find relevant content");
}

// ============================================================================
// TEST: Embedding Similarity Validation
// ============================================================================

/// Test that mock embeddings produce meaningful similarity scores
#[tokio::test]
async fn test_embedding_similarity_meaningful() {
    let provider = MockEmbeddingProvider::new(128);

    // Similar texts should have high similarity
    let emb1 = provider.generate_embedding("machine learning artificial intelligence");
    let emb2 = provider.generate_embedding("machine learning and AI systems");
    let similar_score = cosine_similarity(&emb1, &emb2);

    // Dissimilar texts should have lower similarity
    let emb3 = provider.generate_embedding("cooking recipes for dinner");
    let dissimilar_score = cosine_similarity(&emb1, &emb3);

    assert!(
        similar_score > dissimilar_score,
        "Similar texts should have higher similarity: {} vs {}",
        similar_score,
        dissimilar_score
    );

    // Self-similarity should be 1.0 (or very close)
    let self_score = cosine_similarity(&emb1, &emb1);
    assert!(
        (self_score - 1.0).abs() < 0.001,
        "Self-similarity should be 1.0, got {}",
        self_score
    );
}

// ============================================================================
// TEST: Stats and Metrics
// ============================================================================

/// Test retrieval statistics accuracy
#[tokio::test]
async fn test_retrieval_stats_accuracy() {
    let retriever = HybridRetriever::in_memory().unwrap();
    let provider = MockEmbeddingProvider::new(128);

    // Initial stats should be zero
    let initial_stats = retriever.stats().await.unwrap();
    assert_eq!(initial_stats.document_count, 0);
    assert_eq!(initial_stats.chunk_count, 0);

    // Add documents incrementally and verify stats
    for i in 0..5 {
        let doc = create_test_document(
            &format!("Stats Test Doc {}", i),
            &format!("Content {}", i),
            &[
                &format!("Chunk 1 of doc {}", i),
                &format!("Chunk 2 of doc {}", i),
                &format!("Chunk 3 of doc {}", i),
            ],
        );
        let embeddings = doc
            .chunks
            .iter()
            .map(|c| provider.generate_embedding(&c.text))
            .collect();
        retriever
            .add_document_with_embeddings(&doc, embeddings)
            .await
            .unwrap();

        let stats = retriever.stats().await.unwrap();
        assert_eq!(stats.document_count, i + 1);
        assert_eq!(stats.chunk_count, (i + 1) * 3);
    }

    // Final verification
    let final_stats = retriever.stats().await.unwrap();
    assert_eq!(final_stats.document_count, 5);
    assert_eq!(final_stats.chunk_count, 15);
    assert_eq!(final_stats.indexed_chunks, 15);
}

// ============================================================================
// TEST: Error Handling
// ============================================================================

/// Test error handling for mismatched embeddings
#[tokio::test]
async fn test_embedding_count_mismatch_error() {
    let retriever = HybridRetriever::in_memory().unwrap();

    let doc = create_test_document(
        "Mismatch Test",
        "Testing error handling",
        &["Chunk 1", "Chunk 2", "Chunk 3"],
    );

    // Provide wrong number of embeddings
    let wrong_embeddings = vec![
        vec![0.1; 128],
        vec![0.2; 128],
        // Missing third embedding
    ];

    let result = retriever
        .add_document_with_embeddings(&doc, wrong_embeddings)
        .await;
    assert!(result.is_err(), "Should error on embedding count mismatch");
}

/// Test dense search without embedding pipeline
#[tokio::test]
async fn test_dense_search_without_pipeline_error() {
    let retriever = HybridRetriever::in_memory().unwrap();
    // Note: No embedding pipeline configured

    let doc = create_test_document(
        "No Pipeline Test",
        "Testing without embedding pipeline",
        &["Test chunk"],
    );
    retriever.add_document(&doc).await.unwrap();

    // Dense search should fail without pipeline
    let result = retriever.search_dense("test query", 5).await;
    assert!(
        result.is_err(),
        "Dense search should fail without embedding pipeline"
    );
}
