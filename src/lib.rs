#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! # ReasonKit Memory Infrastructure
//!
//! `reasonkit-mem` provides long-term memory, retrieval, and hybrid search infrastructure
//! for the ReasonKit ecosystem. This crate serves as the "Hippocampus" - the memory and
//! retrieval backbone for AI reasoning systems.
//!
//! ## Overview
//!
//! This crate implements a production-grade memory system combining:
//! - **Vector storage** with Qdrant for semantic search
//! - **BM25 indexing** with Tantivy for keyword matching
//! - **Hybrid retrieval** combining dense and sparse methods
//! - **RAPTOR trees** for hierarchical document understanding
//! - **Dual-layer memory** with hot/cold storage tiers
//!
//! ## Architecture
//!
//! ```text
//! reasonkit-mem/
//! +-- storage/      # Qdrant vector + dual-layer memory
//! +-- embedding/    # Dense vector embeddings (BGE-M3, OpenAI)
//! +-- retrieval/    # Hybrid search, fusion, reranking
//! +-- raptor/       # RAPTOR hierarchical tree structure
//! +-- indexing/     # BM25/Tantivy sparse indexing
//! +-- rag/          # RAG pipeline orchestration
//! +-- service/      # MemoryService trait implementation for reasonkit-core
//! ```
//!
//! ## Quick Start
//!
//! ### Basic Document Indexing and Search
//!
//! ```rust,ignore
//! use reasonkit_mem::{
//!     retrieval::{KnowledgeBase, HybridResult},
//!     Document, DocumentType, Source, SourceType,
//! };
//! use chrono::Utc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an in-memory knowledge base
//!     let kb = KnowledgeBase::in_memory()?;
//!
//!     // Create a document
//!     let source = Source {
//!         source_type: SourceType::Local,
//!         url: None,
//!         path: Some("/docs/example.md".to_string()),
//!         arxiv_id: None,
//!         github_repo: None,
//!         retrieved_at: Utc::now(),
//!         version: None,
//!     };
//!
//!     let doc = Document::new(DocumentType::Documentation, source)
//!         .with_content("Machine learning enables computers to learn from data.".into());
//!
//!     // Add to knowledge base
//!     kb.add(&doc).await?;
//!
//!     // Search using BM25 (no embeddings required)
//!     let results = kb.retriever().search_sparse("machine learning", 10).await?;
//!
//!     for result in results {
//!         println!("Score: {:.3}, Text: {}", result.score, result.text);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Using the MemoryService Trait
//!
//! The `service` module provides a standardized interface for integration with
//! `reasonkit-core`. This enables loose coupling between crates.
//!
//! ```rust,ignore
//! use reasonkit_mem::service::{MemServiceImpl, MemoryService, Document};
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create in-memory service
//!     let service = MemServiceImpl::in_memory()?;
//!
//!     // Store a document
//!     let doc = Document {
//!         id: None,
//!         content: "Machine learning is a subset of AI.".to_string(),
//!         metadata: HashMap::new(),
//!         source: Some("/path/to/doc.md".to_string()),
//!         created_at: None,
//!     };
//!
//!     let id = service.store_document(&doc).await?;
//!     println!("Stored document with ID: {}", id);
//!
//!     // Get statistics
//!     let stats = service.get_stats().await?;
//!     println!("Total documents: {}", stats.total_documents);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Hybrid Search with Embeddings
//!
//! ```rust,ignore
//! use reasonkit_mem::{
//!     embedding::{EmbeddingPipeline, OpenAIEmbedding, EmbeddingConfig},
//!     retrieval::HybridRetriever,
//!     RetrievalConfig,
//! };
//! use std::sync::Arc;
//!
//! // Create embedding pipeline
//! let config = EmbeddingConfig::default();
//! let provider = Arc::new(OpenAIEmbedding::new(config)?);
//! let pipeline = Arc::new(EmbeddingPipeline::new(provider));
//!
//! // Create hybrid retriever with embeddings
//! let retriever = HybridRetriever::in_memory()?
//!     .with_embedding_pipeline(pipeline);
//!
//! // Search combines vector + BM25 results
//! let results = retriever.search("semantic search query", 10).await?;
//! ```
//!
//! ### Dual-Layer Memory System
//!
//! ```rust,ignore
//! use reasonkit_mem::storage::{DualLayerMemory, DualLayerConfig, MemoryEntry};
//!
//! // Create dual-layer memory with hot/cold tiers
//! let config = DualLayerConfig::default();
//! let memory = DualLayerMemory::new(config).await?;
//!
//! // Store a memory entry (goes to hot layer first)
//! let entry = MemoryEntry::new("Important context to remember")
//!     .with_importance(0.9)
//!     .with_metadata("source", "user_input");
//!
//! let id = memory.store(entry).await?;
//!
//! // Retrieve context for a query
//! let results = memory.retrieve_context("context query", 10).await?;
//!
//! // Graceful shutdown
//! memory.shutdown().await?;
//! ```
//!
//! ## Features
//!
//! - **Hybrid Search**: Dense (Qdrant) + Sparse (Tantivy BM25) fusion with RRF
//! - **RAPTOR Trees**: Hierarchical retrieval for long-form QA (Sarthi et al. 2024)
//! - **Cross-Encoder Reranking**: Precision boosting for final results
//! - **Embedded Mode**: Zero-config development with Qdrant embedded
//! - **Dual-Layer Memory**: Hot/cold storage with automatic migration
//! - **Write-Ahead Log**: Durability and crash recovery
//!
//! ## Cargo Features
//!
//! - `default`: Core functionality
//! - `local-embeddings`: Enable local ONNX-based embeddings (BGE-M3, E5)
//! - `python`: Python bindings via PyO3
//!
//! ## Research Foundation
//!
//! This crate implements techniques from:
//! - **RAPTOR**: Sarthi et al. 2024 - "Recursive Abstractive Processing for Tree-Organized Retrieval"
//! - **RRF Fusion**: Cormack et al. 2009 - "Reciprocal Rank Fusion"
//! - **Cross-Encoders**: Nogueira et al. 2020 - arXiv:2010.06467

#![warn(clippy::all)]
#![allow(missing_docs)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![cfg_attr(feature = "python", allow(clippy::useless_conversion))]

// Re-export commonly used types
pub use error::{MemError, MemResult};
pub use types::*;

/// Error types for memory operations.
///
/// This module provides a unified error type [`MemError`] for all memory operations,
/// including storage, embedding, retrieval, and indexing errors.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::error::{MemError, MemResult};
///
/// fn process_document() -> MemResult<()> {
///     // Operations that might fail
///     Err(MemError::storage("Connection failed"))
/// }
/// ```
pub mod error;

/// Core types shared across all modules.
///
/// This module defines the fundamental data structures used throughout the crate:
/// - [`Document`]: A document in the knowledge base
/// - [`Chunk`]: A text chunk from a document
/// - [`Source`]: Source information for a document
/// - [`Metadata`]: Document metadata (title, authors, etc.)
/// - [`RetrievalConfig`]: Configuration for retrieval operations
/// - [`SearchResult`]: Result from a search query
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::{Document, DocumentType, Source, SourceType};
/// use chrono::Utc;
///
/// let source = Source {
///     source_type: SourceType::Local,
///     url: None,
///     path: Some("/path/to/file.md".to_string()),
///     arxiv_id: None,
///     github_repo: None,
///     retrieved_at: Utc::now(),
///     version: None,
/// };
///
/// let doc = Document::new(DocumentType::Note, source)
///     .with_content("Document content here".to_string());
/// ```
pub mod types;

/// Alias for backward compatibility with [`MemError`].
pub type Error = MemError;

/// Result type alias using [`MemError`].
pub type Result<T> = MemResult<T>;

/// Vector and file-based storage backends.
///
/// This module provides storage functionality including:
/// - **Qdrant Integration**: Vector storage for semantic search
/// - **Dual-Layer Memory**: Hot/cold tier storage with automatic migration
/// - **Write-Ahead Log**: Durability and crash recovery
///
/// # Dual-Layer Architecture
///
/// ```text
/// +-------------------+
/// |  DualLayerMemory  |  Unified interface
/// +-------------------+
///          |
///    +-----+-----+
///    |           |
/// +------+   +------+
/// | Hot  |   | Cold |   Hot = recent/active, Cold = historical
/// +------+   +------+
///    |           |
///    +-----+-----+
///          |
///    +----------+
///    |   WAL    |   Write-ahead log for durability
///    +----------+
/// ```
///
/// # Key Types
///
/// - [`storage::Storage`]: Main storage interface for documents
/// - [`storage::DualLayerMemory`]: Dual-layer memory with hot/cold tiers
/// - [`storage::MemoryEntry`]: A memory entry for storage
/// - [`storage::DualLayerConfig`]: Configuration for dual-layer storage
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_mem::storage::{DualLayerMemory, DualLayerConfig, MemoryEntry};
///
/// let config = DualLayerConfig::default();
/// let memory = DualLayerMemory::new(config).await?;
///
/// let entry = MemoryEntry::new("Context to remember")
///     .with_importance(0.8);
///
/// let id = memory.store(entry).await?;
/// ```
pub mod storage;

/// Dense vector embedding services.
///
/// This module provides text embedding functionality for semantic search:
/// - **API-based**: OpenAI, Anthropic, Cohere, local servers
/// - **Local ONNX**: BGE-M3, E5 (requires `local-embeddings` feature)
/// - **Caching**: Built-in embedding cache to reduce API calls
///
/// # Key Types
///
/// - [`embedding::EmbeddingPipeline`]: Pipeline for batch embedding operations
/// - [`embedding::OpenAIEmbedding`]: OpenAI-compatible embedding provider
/// - [`embedding::EmbeddingConfig`]: Configuration for embedding providers
/// - [`embedding::EmbeddingCache`]: LRU cache for embeddings
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_mem::embedding::{
///     EmbeddingPipeline, OpenAIEmbedding, EmbeddingConfig, EmbeddingProvider
/// };
/// use std::sync::Arc;
///
/// // Create OpenAI embedding provider
/// let config = EmbeddingConfig::default();
/// let provider = Arc::new(OpenAIEmbedding::new(config)?);
/// let pipeline = EmbeddingPipeline::new(provider);
///
/// // Embed text
/// let embedding = pipeline.embed_text("Hello, world!").await?;
/// println!("Embedding dimension: {}", embedding.len());
/// ```
///
/// # Utility Functions
///
/// - [`embedding::normalize_vector`]: Normalize a vector to unit length
/// - [`embedding::cosine_similarity`]: Compute cosine similarity between vectors
pub mod embedding;

/// Hybrid retrieval with fusion and reranking.
///
/// This module implements state-of-the-art retrieval techniques:
/// - **Hybrid Search**: Combines BM25 (sparse) and vector (dense) retrieval
/// - **RRF Fusion**: Reciprocal Rank Fusion for combining result sets
/// - **Cross-Encoder Reranking**: Precision improvement using cross-encoders
/// - **Query Expansion**: Better recall through query augmentation
///
/// # Key Types
///
/// - [`retrieval::HybridRetriever`]: Main hybrid search interface
/// - [`retrieval::KnowledgeBase`]: High-level knowledge base wrapper
/// - [`retrieval::FusionEngine`]: Combines results from multiple methods
/// - [`retrieval::Reranker`]: Cross-encoder reranking for precision
///
/// # Research Foundation
///
/// - **RRF Fusion**: Cormack et al. 2009 - "Reciprocal Rank Fusion"
/// - **Cross-Encoder**: Nogueira et al. 2020 - arXiv:2010.06467
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_mem::retrieval::{HybridRetriever, KnowledgeBase};
///
/// // Create in-memory knowledge base
/// let kb = KnowledgeBase::in_memory()?;
///
/// // Add documents
/// kb.add(&document).await?;
///
/// // Query with hybrid search
/// let results = kb.query("search query", 10).await?;
///
/// for result in results {
///     println!("Score: {:.3}, Source: {:?}", result.score, result.match_source);
/// }
/// ```
pub mod retrieval;

/// RAPTOR hierarchical tree structure.
///
/// Implements the RAPTOR (Recursive Abstractive Processing for Tree-Organized
/// Retrieval) algorithm for hierarchical document understanding and retrieval.
///
/// # Overview
///
/// RAPTOR builds a tree structure where:
/// - **Leaf nodes**: Original document chunks with embeddings
/// - **Internal nodes**: Summaries of child node clusters
/// - **Root nodes**: High-level summaries of the entire document
///
/// This enables retrieval at multiple levels of abstraction.
///
/// # Key Types
///
/// - [`raptor::RaptorTree`]: Main RAPTOR tree structure
/// - [`raptor::RaptorNode`]: A node in the RAPTOR tree
/// - [`raptor::RaptorStats`]: Statistics about the tree
/// - [`raptor::OptimizedRaptorTree`]: Performance-optimized implementation
/// - [`raptor::CodeGraph`]: Code entity graph for code understanding
///
/// # Research Foundation
///
/// Based on: Sarthi et al. 2024 - "RAPTOR: Recursive Abstractive Processing
/// for Tree-Organized Retrieval"
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_mem::raptor::RaptorTree;
///
/// let mut tree = RaptorTree::new(
///     3,  // max_depth
///     4,  // cluster_size
/// );
///
/// // Build tree from chunks
/// tree.build_from_chunks(&chunks, &embedder, &summarizer).await?;
///
/// // Search the tree
/// let results = tree.search(&query_embedding, 10)?;
///
/// let stats = tree.stats();
/// println!("Total nodes: {}, Leaf nodes: {}", stats.total_nodes, stats.leaf_nodes);
/// ```
pub mod raptor;

/// BM25/Tantivy sparse indexing.
///
/// This module provides full-text search indexing using the Tantivy search engine,
/// implementing BM25 scoring for keyword-based retrieval.
///
/// # Key Types
///
/// - [`indexing::IndexManager`]: Manages all index types
/// - [`indexing::BM25Index`]: BM25 text index using Tantivy
/// - [`indexing::IndexConfig`]: Configuration for indexing
/// - [`indexing::IndexStats`]: Statistics about the index
/// - [`indexing::BM25Result`]: Result from a BM25 search
///
/// # Features
///
/// - Fast full-text search with BM25 scoring
/// - In-memory or persistent index storage
/// - Incremental document updates
/// - Index optimization (segment merging)
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_mem::indexing::{IndexManager, BM25Index};
///
/// // Create in-memory index manager
/// let manager = IndexManager::in_memory()?;
///
/// // Index a document
/// manager.index_document(&document)?;
///
/// // Search using BM25
/// let results = manager.search_bm25("machine learning", 10)?;
///
/// for result in results {
///     println!("Score: {:.2}, Text: {}", result.score, result.text);
/// }
/// ```
pub mod indexing;

/// RAG pipeline orchestration.
///
/// This module coordinates the full RAG (Retrieval-Augmented Generation) pipeline:
/// - Query processing and analysis
/// - Multi-stage retrieval (sparse, dense, hybrid)
/// - Context assembly and ranking
/// - Response generation coordination
pub mod rag;

/// MemoryService trait implementation for reasonkit-core integration.
///
/// This module provides the [`service::MemServiceImpl`] struct that implements the
/// `MemoryService` trait defined in `reasonkit-core::traits::memory`. It wraps the
/// existing reasonkit-mem infrastructure into a unified interface for cross-crate
/// integration.
///
/// # Architecture
///
/// ```text
/// reasonkit-core                    reasonkit-mem
/// +------------------+              +------------------+
/// | MemoryService    | <-- trait    | MemServiceImpl   |
/// | (trait)          |              | (implementation) |
/// +------------------+              +------------------+
///                                          |
///                    +---------------------+---------------------+
///                    |                     |                     |
///              +----------+         +------------+        +------------+
///              | Storage  |         | Embedding  |        | Retrieval  |
///              +----------+         +------------+        +------------+
/// ```
///
/// # Key Types
///
/// - [`service::MemServiceImpl`]: Implementation of the MemoryService trait
/// - [`service::MemoryService`]: The trait interface (mirrored from reasonkit-core)
/// - [`service::Document`]: Simplified document type for the service interface
/// - [`service::SearchResult`]: Search result from the service
/// - [`service::MemoryConfig`]: Configuration for the memory service
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit_mem::service::{MemServiceImpl, MemoryService, Document};
/// use std::collections::HashMap;
///
/// // Create in-memory service
/// let service = MemServiceImpl::in_memory()?;
///
/// // Store a document
/// let doc = Document {
///     id: None,
///     content: "Machine learning is a subset of AI.".to_string(),
///     metadata: HashMap::new(),
///     source: Some("/path/to/doc.md".to_string()),
///     created_at: None,
/// };
///
/// let id = service.store_document(&doc).await?;
///
/// // Search for documents
/// let results = service.search("machine learning", 10).await?;
///
/// // Get index statistics
/// let stats = service.get_stats().await?;
/// println!("Documents: {}, Chunks: {}", stats.total_documents, stats.total_chunks);
/// ```
pub mod service;

/// Prelude for convenient imports.
///
/// This module re-exports the most commonly used types for convenience.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::prelude::*;
/// ```
pub mod prelude {
    pub use crate::error::{MemError, MemResult};
    pub use crate::service::{MemServiceImpl, MemoryService};
    pub use crate::types::*;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lib_compiles() {
        // Basic compilation test
    }
}

// ============================================================================
// Python Bindings Infrastructure
// ============================================================================

#[cfg(feature = "python")]
use crate::raptor::optimized::{OptimizedRaptorTree, RaptorOptConfig};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::sync::Arc;
#[cfg(feature = "python")]
use tokio::sync::Mutex;

/// Python wrapper for the RAPTOR tree.
///
/// Provides a Python-friendly interface to the RAPTOR hierarchical tree
/// for document retrieval. Requires the `python` feature.
///
/// # Python Example
///
/// ```python
/// from reasonkit_mem import RaptorTree
///
/// # Create tree with optional configuration
/// tree = RaptorTree(max_depth=3, cluster_size=4)
///
/// # Ingest documents with custom embed and summarize functions
/// tree.ingest_documents(
///     documents=["doc1", "doc2", "doc3"],
///     embed_fn=my_embed_function,
///     summarize_fn=my_summarize_function
/// )
///
/// # Query the tree
/// results = tree.query(query_vector, top_k=10)
/// for chunk_id, score in results:
///     print(f"Chunk {chunk_id}: {score:.3f}")
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "RaptorTree")]
pub struct PyRaptorTree {
    inner: Arc<Mutex<OptimizedRaptorTree>>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRaptorTree {
    /// Create a new RAPTOR tree.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum tree depth (default: 3)
    /// * `cluster_size` - Number of nodes per cluster (default: 4)
    ///
    /// # Returns
    ///
    /// A new PyRaptorTree instance.
    #[new]
    #[pyo3(signature = (max_depth=None, cluster_size=None))]
    fn new(max_depth: Option<usize>, cluster_size: Option<usize>) -> PyResult<Self> {
        let mut config = RaptorOptConfig::default();
        if let Some(d) = max_depth {
            config.max_depth = d;
        }
        if let Some(c) = cluster_size {
            config.cluster_size = c;
        }

        let tree = OptimizedRaptorTree::new(config);

        // Create a runtime for this instance
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(PyRaptorTree {
            inner: Arc::new(Mutex::new(tree)),
            rt: Arc::new(rt),
        })
    }

    /// Ingest documents into the RAPTOR tree.
    ///
    /// # Arguments
    ///
    /// * `documents` - List of document strings to ingest
    /// * `embed_fn` - Python callable that takes a string and returns a list of floats
    /// * `summarize_fn` - Python callable that takes a string and returns a summary string
    ///
    /// # Returns
    ///
    /// None on success, raises RuntimeError on failure.
    #[allow(clippy::useless_conversion)]
    fn ingest_documents(
        &self,
        py: Python<'_>,
        documents: Vec<String>,
        embed_fn: PyObject,
        summarize_fn: PyObject,
    ) -> PyResult<()> {
        let chunks: Vec<Chunk> = documents
            .into_iter()
            .enumerate()
            .map(|(i, text)| Chunk {
                id: uuid::Uuid::new_v4(),
                text,
                index: i,
                start_char: 0,
                end_char: 0,
                token_count: None,
                section: None,
                page: None,
                embedding_ids: Default::default(),
            })
            .collect();

        // Capture for async block
        let inner = self.inner.clone();
        let embed_fn = embed_fn.clone_ref(py);
        let summarize_fn = summarize_fn.clone_ref(py);
        let rt = self.rt.clone();

        // Release GIL to allow threading/async
        py.allow_threads(move || {
            rt.block_on(async move {
                let mut tree = inner.lock().await;

                // Callbacks that re-acquire GIL
                let embedder = |text: &str| -> crate::Result<Vec<f32>> {
                    Python::with_gil(|py| {
                        let res = embed_fn
                            .call1(py, (text,))
                            .map_err(|e| crate::MemError::embedding(e.to_string()))?;
                        res.extract::<Vec<f32>>(py)
                            .map_err(|e| crate::MemError::embedding(e.to_string()))
                    })
                };

                let summarizer = |text: &str| -> crate::Result<String> {
                    Python::with_gil(|py| {
                        let res = summarize_fn
                            .call1(py, (text,))
                            .map_err(|e| crate::MemError::generation(e.to_string()))?;
                        res.extract::<String>(py)
                            .map_err(|e| crate::MemError::generation(e.to_string()))
                    })
                };

                tree.build_from_chunks(&chunks, &embedder, &summarizer)
                    .await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    /// Query the RAPTOR tree with a vector.
    ///
    /// # Arguments
    ///
    /// * `vector` - Query embedding vector as a list of floats
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    ///
    /// List of (chunk_id, score) tuples, sorted by score descending.
    #[allow(clippy::useless_conversion)]
    fn query(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let inner = self.inner.clone();
        let rt = self.rt.clone();

        py.allow_threads(move || {
            rt.block_on(async move {
                let tree = inner.lock().await;
                // search_beam is synchronous but we access it within the async lock
                let results = tree.search_beam(&vector, top_k).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;

                // Convert UUIDs to strings
                Ok(results
                    .into_iter()
                    .map(|(uuid, score)| (uuid.to_string(), score))
                    .collect())
            })
        })
    }
}

/// Python module definition for reasonkit_mem.
///
/// Exposes the RAPTOR tree functionality to Python.
#[cfg(feature = "python")]
#[pymodule]
fn reasonkit_mem(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRaptorTree>()?;
    Ok(())
}
