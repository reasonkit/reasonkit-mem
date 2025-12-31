//! # ReasonKit Memory Infrastructure
//!
//! Long-term memory, retrieval, and hybrid search infrastructure for ReasonKit.
//! This crate provides the "Hippocampus" - memory & retrieval capabilities.
//!
//! ## Architecture
//!
//! ```text
//! reasonkit-mem/
//! ├── storage/      # Qdrant vector + file-based storage
//! ├── embedding/    # Dense vector embeddings (BGE-M3, OpenAI)
//! ├── retrieval/    # Hybrid search, fusion, reranking
//! ├── raptor/       # RAPTOR hierarchical tree structure
//! ├── indexing/     # BM25/Tantivy sparse indexing
//! └── rag/          # RAG pipeline orchestration
//! ```
//!
//! ## Features
//!
//! - **Hybrid Search**: Dense (Qdrant) + Sparse (Tantivy BM25) fusion
//! - **RAPTOR Trees**: Hierarchical retrieval for long-form QA
//! - **Cross-Encoder Reranking**: Precision boosting for final results
//! - **Embedded Mode**: Zero-config development with Qdrant embedded
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit_mem::{Storage, EmbeddingService, HybridRetriever};
//!
//! // Create storage backend
//! let storage = Storage::new_embedded("./data").await?;
//!
//! // Index documents
//! storage.index_documents(&docs).await?;
//!
//! // Hybrid search
//! let results = retriever.search("query", 10).await?;
//! ```

#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![cfg_attr(feature = "python", allow(clippy::useless_conversion))]

// Re-export commonly used types
pub use error::{MemError, MemResult};
pub use types::*;

/// Error types for memory operations
pub mod error;

/// Core types shared across all modules
pub mod types;

/// Alias for backward compatibility
pub type Error = MemError;
/// Result alias
pub type Result<T> = MemResult<T>;

/// Vector and file-based storage backends
///
/// Provides:
/// - Qdrant vector storage (embedded and cluster modes)
/// - File-based fallback storage
/// - Document and chunk management
pub mod storage;

/// Dense vector embedding services
///
/// Supports:
/// - Local embeddings (BGE-M3 via ONNX)
/// - Remote embeddings (OpenAI, Anthropic, etc.)
/// - Caching and batching
pub mod embedding;

/// Hybrid retrieval with fusion and reranking
///
/// Implements:
/// - Dense + Sparse hybrid search
/// - Reciprocal Rank Fusion (RRF)
/// - Cross-encoder reranking
/// - Query expansion
pub mod retrieval;

/// RAPTOR hierarchical tree structure
///
/// Based on RAPTOR paper (Sarthi et al. 2024):
/// - Multi-level summarization
/// - Recursive clustering
/// - Long-form QA optimization
pub mod raptor;

/// BM25/Tantivy sparse indexing
///
/// Provides:
/// - Full-text search indexing
/// - Custom analyzers
/// - Incremental updates
pub mod indexing;

/// RAG pipeline orchestration
///
/// Coordinates:
/// - Query processing
/// - Multi-stage retrieval
/// - Context assembly
pub mod rag;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::error::{MemError, MemResult};
    pub use crate::types::*;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lib_compiles() {
        // Basic compilation test
    }
}

// Python Bindings Infrastructure
#[cfg(feature = "python")]
use crate::raptor::optimized::{OptimizedRaptorTree, RaptorOptConfig};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::sync::Arc;
#[cfg(feature = "python")]
use tokio::sync::Mutex;

#[cfg(feature = "python")]
#[pyclass(name = "RaptorTree")]
pub struct PyRaptorTree {
    inner: Arc<Mutex<OptimizedRaptorTree>>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRaptorTree {
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

#[cfg(feature = "python")]
#[pymodule]
fn reasonkit_mem(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRaptorTree>()?;
    Ok(())
}
