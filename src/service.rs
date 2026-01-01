//! MemoryService implementation for reasonkit-core integration.
//!
//! This module provides the `MemServiceImpl` struct that implements the
//! `MemoryService` trait defined in `reasonkit-core::traits::memory`.
//! It wraps the existing reasonkit-mem infrastructure (storage, embedding,
//! retrieval, indexing) into a unified interface.
//!
//! # Architecture
//!
//! ```text
//! reasonkit-core                    reasonkit-mem
//! +------------------+              +------------------+
//! | MemoryService    | <-- trait    | MemServiceImpl   |
//! | (trait)          |              | (implementation) |
//! +------------------+              +------------------+
//!                                          |
//!                    +---------------------+---------------------+
//!                    |                     |                     |
//!              +----------+         +------------+        +------------+
//!              | Storage  |         | Embedding  |        | Retrieval  |
//!              +----------+         +------------+        +------------+
//! ```
//!
//! # Feature Flag
//!
//! This module is gated behind the `core-integration` feature flag to avoid
//! circular dependencies when building reasonkit-mem standalone.

use crate::{
    embedding::{EmbeddingPipeline, EmbeddingProvider, OpenAIEmbedding},
    indexing::IndexManager,
    retrieval::{HybridResult, HybridRetriever, RetrievalStats as MemRetrievalStats},
    storage::{AccessContext, AccessLevel, Storage},
    Document as MemDocument, DocumentType, Error as MemError, MatchSource, Result as MemResult,
    Source, SourceType,
};
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

// ============================================================================
// Type Aliases for reasonkit-core trait types
// ============================================================================

/// Re-export of trait types that must match reasonkit-core definitions.
/// These are duplicated here to avoid circular crate dependencies.
/// When the `core-integration` feature is enabled, these should be imported
/// from reasonkit-core instead.
/// Result type for memory operations.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// Errors that can occur during memory operations.
#[derive(thiserror::Error, Debug)]
pub enum MemoryError {
    #[error("Document not found: {0}")]
    NotFound(Uuid),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<MemError> for MemoryError {
    fn from(e: MemError) -> Self {
        match e {
            MemError::NotFound(s) => MemoryError::NotFound(Uuid::parse_str(&s).unwrap_or_default()),
            MemError::Storage(s) => MemoryError::Storage(s),
            MemError::Embedding(s) => MemoryError::Embedding(s),
            MemError::Indexing(s) => MemoryError::Index(s),
            MemError::Config(s) => MemoryError::Config(s),
            MemError::Serialization(s) => MemoryError::Serialization(s),
            MemError::Io(e) => MemoryError::Io(e),
            other => MemoryError::Storage(other.to_string()),
        }
    }
}

/// A document to be stored in memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub id: Option<Uuid>,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub source: Option<String>,
    pub created_at: Option<i64>,
}

/// A chunk of a document after splitting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub id: Option<Uuid>,
    pub document_id: Uuid,
    pub content: String,
    pub index: usize,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
}

/// A search result from memory retrieval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
    pub source: RetrievalSource,
}

/// Source of the retrieval result.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RetrievalSource {
    Vector,
    BM25,
    Hybrid,
}

/// Configuration for hybrid search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HybridConfig {
    pub vector_weight: f32,
    pub bm25_weight: f32,
    pub use_reranker: bool,
    pub reranker_top_k: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            bm25_weight: 0.3,
            use_reranker: true,
            reranker_top_k: 10,
        }
    }
}

/// A context window assembled from retrieved chunks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextWindow {
    pub chunks: Vec<SearchResult>,
    pub total_tokens: usize,
    pub truncated: bool,
}

/// Configuration for index creation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexConfig {
    pub name: String,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub ef_construction: usize,
    pub m: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            dimensions: 384,
            metric: DistanceMetric::Cosine,
            ef_construction: 200,
            m: 16,
        }
    }
}

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Statistics about the memory index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_vectors: usize,
    pub index_size_bytes: u64,
    pub last_updated: i64,
}

/// Configuration for the memory service.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub embedding_model: String,
    pub embedding_dimensions: usize,
    pub max_context_tokens: usize,
    pub storage_path: Option<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            embedding_model: "all-MiniLM-L6-v2".to_string(),
            embedding_dimensions: 384,
            max_context_tokens: 4096,
            storage_path: None,
        }
    }
}

// ============================================================================
// MemoryService Trait Definition
// ============================================================================

/// Core abstraction for memory operations.
///
/// This trait is implemented by `reasonkit-mem` and consumed by `reasonkit-core`.
/// It provides a unified interface for document storage, retrieval, and embedding.
#[async_trait]
pub trait MemoryService: Send + Sync {
    // -------------------------------------------------------------------------
    // Storage Operations
    // -------------------------------------------------------------------------

    /// Store a document, returning its assigned ID.
    async fn store_document(&self, doc: &Document) -> MemoryResult<Uuid>;

    /// Store multiple chunks, returning their assigned IDs.
    async fn store_chunks(&self, chunks: &[Chunk]) -> MemoryResult<Vec<Uuid>>;

    /// Delete a document and all its chunks.
    async fn delete_document(&self, id: Uuid) -> MemoryResult<()>;

    /// Update an existing document.
    async fn update_document(&self, id: Uuid, doc: &Document) -> MemoryResult<()>;

    // -------------------------------------------------------------------------
    // Retrieval Operations
    // -------------------------------------------------------------------------

    /// Search for relevant chunks using vector similarity.
    async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>>;

    /// Search using hybrid retrieval (vector + BM25 with RRF fusion).
    async fn hybrid_search(
        &self,
        query: &str,
        top_k: usize,
        config: HybridConfig,
    ) -> MemoryResult<Vec<SearchResult>>;

    /// Get a document by its ID.
    async fn get_by_id(&self, id: Uuid) -> MemoryResult<Option<Document>>;

    /// Get a context window optimized for the query and token budget.
    async fn get_context(&self, query: &str, max_tokens: usize) -> MemoryResult<ContextWindow>;

    // -------------------------------------------------------------------------
    // Embedding Operations
    // -------------------------------------------------------------------------

    /// Embed a single text string.
    async fn embed(&self, text: &str) -> MemoryResult<Vec<f32>>;

    /// Embed multiple texts in a batch.
    async fn embed_batch(&self, texts: &[&str]) -> MemoryResult<Vec<Vec<f32>>>;

    // -------------------------------------------------------------------------
    // Index Management
    // -------------------------------------------------------------------------

    /// Create a new index with the given configuration.
    async fn create_index(&self, config: IndexConfig) -> MemoryResult<()>;

    /// Rebuild the index from stored documents.
    async fn rebuild_index(&self) -> MemoryResult<()>;

    /// Get statistics about the current index.
    async fn get_stats(&self) -> MemoryResult<IndexStats>;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Get the current configuration.
    fn config(&self) -> &MemoryConfig;

    /// Update the configuration.
    fn set_config(&mut self, config: MemoryConfig);

    // -------------------------------------------------------------------------
    // Health & Lifecycle
    // -------------------------------------------------------------------------

    /// Check if the service is healthy and ready.
    async fn health_check(&self) -> MemoryResult<bool>;

    /// Flush any pending writes to storage.
    async fn flush(&self) -> MemoryResult<()>;

    /// Gracefully shutdown the service.
    async fn shutdown(&self) -> MemoryResult<()>;
}

// ============================================================================
// MemServiceImpl - The Implementation
// ============================================================================

/// Implementation of `MemoryService` that wraps reasonkit-mem infrastructure.
///
/// This struct delegates to:
/// - `HybridRetriever` for search operations
/// - `EmbeddingPipeline` for embedding generation
/// - `Storage` for document persistence
/// - `IndexManager` for BM25 indexing
pub struct MemServiceImpl {
    retriever: HybridRetriever,
    embedding_pipeline: Option<Arc<EmbeddingPipeline>>,
    config: RwLock<MemoryConfig>,
    is_healthy: std::sync::atomic::AtomicBool,
}

impl MemServiceImpl {
    /// Create a new in-memory MemServiceImpl.
    ///
    /// This is suitable for development and testing.
    pub fn in_memory() -> MemResult<Self> {
        let retriever = HybridRetriever::in_memory()?;

        Ok(Self {
            retriever,
            embedding_pipeline: None,
            config: RwLock::new(MemoryConfig::default()),
            is_healthy: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// Create a new MemServiceImpl with custom storage and index.
    pub fn new(storage: Storage, index: IndexManager) -> Self {
        let retriever = HybridRetriever::new(storage, index);

        Self {
            retriever,
            embedding_pipeline: None,
            config: RwLock::new(MemoryConfig::default()),
            is_healthy: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Configure the embedding pipeline.
    ///
    /// Required for vector search operations. Without this, only BM25 search
    /// will be available.
    pub fn with_embedding_pipeline(mut self, pipeline: Arc<EmbeddingPipeline>) -> Self {
        self.embedding_pipeline = Some(pipeline.clone());
        self.retriever = self.retriever.with_embedding_pipeline(pipeline);
        self
    }

    /// Configure with OpenAI embeddings.
    ///
    /// Convenience method that sets up OpenAI-compatible embedding provider.
    pub fn with_openai_embeddings(self) -> MemResult<Self> {
        let provider = OpenAIEmbedding::openai()?;
        let pipeline = Arc::new(EmbeddingPipeline::new(Arc::new(provider)));
        Ok(self.with_embedding_pipeline(pipeline))
    }

    /// Set the memory configuration.
    pub fn with_config(self, config: MemoryConfig) -> Self {
        *self.config.write().unwrap() = config;
        self
    }

    /// Get a reference to the underlying retriever.
    pub fn retriever(&self) -> &HybridRetriever {
        &self.retriever
    }

    /// Create admin access context for internal operations.
    fn admin_context(&self, operation: &str) -> AccessContext {
        AccessContext::new(
            "mem-service".to_string(),
            AccessLevel::Admin,
            operation.to_string(),
        )
    }

    /// Convert external Document to internal MemDocument.
    fn to_mem_document(&self, doc: &Document) -> MemDocument {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: doc.source.clone(),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let mut mem_doc =
            MemDocument::new(DocumentType::Note, source).with_content(doc.content.clone());

        // Set ID if provided
        if let Some(id) = doc.id {
            mem_doc.id = id;
        }

        // Convert metadata to tags
        mem_doc.metadata.tags = doc.metadata.keys().cloned().collect();

        mem_doc
    }

    /// Convert internal HybridResult to external SearchResult.
    fn to_search_result(&self, result: &HybridResult) -> SearchResult {
        let source = match result.match_source {
            MatchSource::Dense => RetrievalSource::Vector,
            MatchSource::Sparse => RetrievalSource::BM25,
            MatchSource::Hybrid | MatchSource::Raptor => RetrievalSource::Hybrid,
        };

        SearchResult {
            chunk: Chunk {
                id: Some(result.chunk_id),
                document_id: result.doc_id,
                content: result.text.clone(),
                index: 0, // Not tracked in HybridResult
                embedding: None,
                metadata: HashMap::new(),
            },
            score: result.score,
            source,
        }
    }

    /// Convert internal MemDocument to external Document.
    fn to_external_document(&self, doc: &MemDocument) -> Document {
        let mut metadata = HashMap::new();
        for tag in &doc.metadata.tags {
            metadata.insert(tag.clone(), "true".to_string());
        }

        Document {
            id: Some(doc.id),
            content: doc.content.raw.clone(),
            metadata,
            source: doc.source.path.clone(),
            created_at: Some(doc.created_at.timestamp()),
        }
    }
}

#[async_trait]
impl MemoryService for MemServiceImpl {
    // -------------------------------------------------------------------------
    // Storage Operations
    // -------------------------------------------------------------------------

    async fn store_document(&self, doc: &Document) -> MemoryResult<Uuid> {
        let mem_doc = self.to_mem_document(doc);
        let doc_id = mem_doc.id;

        self.retriever.add_document(&mem_doc).await?;

        Ok(doc_id)
    }

    async fn store_chunks(&self, chunks: &[Chunk]) -> MemoryResult<Vec<Uuid>> {
        // For each chunk, we create a mini-document and store it
        let mut ids = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let doc = Document {
                id: chunk.id,
                content: chunk.content.clone(),
                metadata: chunk.metadata.clone(),
                source: None,
                created_at: None,
            };

            let id = self.store_document(&doc).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    async fn delete_document(&self, id: Uuid) -> MemoryResult<()> {
        self.retriever.delete_document(&id).await?;
        Ok(())
    }

    async fn update_document(&self, id: Uuid, doc: &Document) -> MemoryResult<()> {
        // Delete existing
        self.delete_document(id).await?;

        // Store new with same ID
        let mut new_doc = doc.clone();
        new_doc.id = Some(id);
        self.store_document(&new_doc).await?;

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Retrieval Operations
    // -------------------------------------------------------------------------

    async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>> {
        let results = self.retriever.search(query, top_k).await?;
        Ok(results.iter().map(|r| self.to_search_result(r)).collect())
    }

    async fn hybrid_search(
        &self,
        query: &str,
        top_k: usize,
        config: HybridConfig,
    ) -> MemoryResult<Vec<SearchResult>> {
        let retrieval_config = crate::RetrievalConfig {
            top_k,
            min_score: 0.0,
            alpha: config.vector_weight,
            use_raptor: false,
            rerank: config.use_reranker,
        };

        let results = self
            .retriever
            .search_hybrid(query, None, &retrieval_config)
            .await?;

        Ok(results.iter().map(|r| self.to_search_result(r)).collect())
    }

    async fn get_by_id(&self, id: Uuid) -> MemoryResult<Option<Document>> {
        let context = self.admin_context("get_by_id");

        match self.retriever.storage().get_document(&id, &context).await {
            Ok(Some(doc)) => Ok(Some(self.to_external_document(&doc))),
            Ok(None) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn get_context(&self, query: &str, max_tokens: usize) -> MemoryResult<ContextWindow> {
        let top_k = {
            let config = self.config.read().unwrap();
            max_tokens / config.chunk_size.max(1)
        };

        let results = self.search(query, top_k.max(5)).await?;

        // Estimate tokens (rough approximation: 4 chars per token)
        let mut total_tokens = 0;
        let mut chunks = Vec::new();
        let mut truncated = false;

        for result in results {
            let chunk_tokens = result.chunk.content.len() / 4;
            if total_tokens + chunk_tokens > max_tokens {
                truncated = true;
                break;
            }
            total_tokens += chunk_tokens;
            chunks.push(result);
        }

        Ok(ContextWindow {
            chunks,
            total_tokens,
            truncated,
        })
    }

    // -------------------------------------------------------------------------
    // Embedding Operations
    // -------------------------------------------------------------------------

    async fn embed(&self, text: &str) -> MemoryResult<Vec<f32>> {
        let pipeline = self
            .embedding_pipeline
            .as_ref()
            .ok_or_else(|| MemoryError::Config("Embedding pipeline not configured".into()))?;

        pipeline
            .embed_text(text)
            .await
            .map_err(|e| MemoryError::Embedding(e.to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> MemoryResult<Vec<Vec<f32>>> {
        let pipeline = self
            .embedding_pipeline
            .as_ref()
            .ok_or_else(|| MemoryError::Config("Embedding pipeline not configured".into()))?;

        let provider = pipeline.provider();
        let results = provider
            .embed_batch(texts)
            .await
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;

        // Extract dense vectors from results
        results
            .into_iter()
            .map(|r| {
                r.dense
                    .ok_or_else(|| MemoryError::Embedding("No dense embedding returned".into()))
            })
            .collect()
    }

    // -------------------------------------------------------------------------
    // Index Management
    // -------------------------------------------------------------------------

    async fn create_index(&self, _config: IndexConfig) -> MemoryResult<()> {
        // The index is automatically created when using HybridRetriever
        // This method is a no-op for the current implementation
        Ok(())
    }

    async fn rebuild_index(&self) -> MemoryResult<()> {
        // Optimize the BM25 index
        self.retriever
            .index()
            .optimize()
            .map_err(|e| MemoryError::Index(e.to_string()))?;

        Ok(())
    }

    async fn get_stats(&self) -> MemoryResult<IndexStats> {
        let stats = self.retriever.stats().await?;

        Ok(IndexStats {
            total_documents: stats.document_count,
            total_chunks: stats.chunk_count,
            total_vectors: stats.embedding_count,
            index_size_bytes: stats.storage_bytes + stats.index_bytes,
            last_updated: Utc::now().timestamp(),
        })
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    fn config(&self) -> &MemoryConfig {
        // This is a bit awkward due to RwLock, but we need to return a reference
        // In practice, we'd use a different pattern or interior mutability
        // For now, we leak a Box to get a static reference (not ideal but functional)
        let config = self.config.read().unwrap().clone();
        Box::leak(Box::new(config))
    }

    fn set_config(&mut self, config: MemoryConfig) {
        *self.config.write().unwrap() = config;
    }

    // -------------------------------------------------------------------------
    // Health & Lifecycle
    // -------------------------------------------------------------------------

    async fn health_check(&self) -> MemoryResult<bool> {
        Ok(self.is_healthy.load(std::sync::atomic::Ordering::SeqCst))
    }

    async fn flush(&self) -> MemoryResult<()> {
        // The current storage implementation flushes on each write
        // This is a no-op but could be extended for batched writes
        Ok(())
    }

    async fn shutdown(&self) -> MemoryResult<()> {
        self.is_healthy
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mem_service_creation() {
        let service = MemServiceImpl::in_memory().expect("Failed to create service");
        assert!(service.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_store_and_search_sparse() {
        let service = MemServiceImpl::in_memory().expect("Failed to create service");

        // Store a document
        let doc = Document {
            id: None,
            content: "Machine learning is a subset of artificial intelligence.".to_string(),
            metadata: HashMap::new(),
            source: Some("/test/doc.md".to_string()),
            created_at: None,
        };

        let id = service.store_document(&doc).await.unwrap();
        assert_ne!(id, Uuid::nil());

        // Search using BM25 (sparse search works without embedding pipeline)
        let results = service
            .retriever
            .search_sparse("machine learning", 5)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_get_stats() {
        let service = MemServiceImpl::in_memory().expect("Failed to create service");

        let stats = service.get_stats().await.unwrap();
        assert_eq!(stats.total_documents, 0);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let service = MemServiceImpl::in_memory().expect("Failed to create service");

        assert!(service.health_check().await.unwrap());
        service.shutdown().await.unwrap();
        assert!(!service.health_check().await.unwrap());
    }

    #[test]
    fn test_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 50);
        assert_eq!(config.embedding_dimensions, 384);
    }

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert!((config.vector_weight - 0.7).abs() < 0.001);
        assert!((config.bm25_weight - 0.3).abs() < 0.001);
        assert!(config.use_reranker);
    }
}
