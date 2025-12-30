//! Error types for ReasonKit Memory operations

use std::io;
use thiserror::Error;

/// Result type for memory operations
pub type MemResult<T> = Result<T, MemError>;

/// Unified error type for all memory operations
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum MemError {
    /// Storage backend errors (Qdrant, file system)
    #[error("Storage error: {0}")]
    Storage(String),

    /// Embedding service errors
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Retrieval errors (search, fusion, reranking)
    #[error("Retrieval error: {0}")]
    Retrieval(String),

    /// RAPTOR tree errors
    #[error("RAPTOR error: {0}")]
    Raptor(String),

    /// Indexing errors (BM25/Tantivy)
    #[error("Indexing error: {0}")]
    Indexing(String),

    /// RAG pipeline errors
    #[error("RAG error: {0}")]
    Rag(String),

    /// I/O errors
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network errors
    #[error("Network error: {0}")]
    Network(String),

    /// Qdrant-specific errors
    #[error("Qdrant error: {0}")]
    Qdrant(String),

    /// Tantivy-specific errors
    #[error("Tantivy error: {0}")]
    Tantivy(String),

    /// Document not found
    #[error("Document not found: {0}")]
    NotFound(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl From<serde_json::Error> for MemError {
    fn from(err: serde_json::Error) -> Self {
        MemError::Serialization(err.to_string())
    }
}

impl From<reqwest::Error> for MemError {
    fn from(err: reqwest::Error) -> Self {
        MemError::Network(err.to_string())
    }
}

/// Helper methods for creating errors (matches reasonkit-core API)
impl MemError {
    /// Create an IO error from a string
    pub fn io(msg: impl Into<String>) -> Self {
        MemError::Storage(msg.into())
    }

    /// Create a storage error
    pub fn storage(msg: impl Into<String>) -> Self {
        MemError::Storage(msg.into())
    }

    /// Create an embedding error
    pub fn embedding(msg: impl Into<String>) -> Self {
        MemError::Embedding(msg.into())
    }

    /// Create a retrieval error
    pub fn retrieval(msg: impl Into<String>) -> Self {
        MemError::Retrieval(msg.into())
    }

    /// Create a qdrant error
    pub fn qdrant(msg: impl Into<String>) -> Self {
        MemError::Qdrant(msg.into())
    }

    /// Create a config error
    pub fn config(msg: impl Into<String>) -> Self {
        MemError::Config(msg.into())
    }

    /// Create a not found error
    pub fn not_found(msg: impl Into<String>) -> Self {
        MemError::NotFound(msg.into())
    }

    /// Create a parse/serialization error
    pub fn parse(msg: impl Into<String>) -> Self {
        MemError::Serialization(msg.into())
    }

    /// Create a tantivy error
    pub fn tantivy(msg: impl Into<String>) -> Self {
        MemError::Tantivy(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        MemError::InvalidInput(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        MemError::InvalidInput(msg.into())
    }

    /// Create a network error
    pub fn network(msg: impl Into<String>) -> Self {
        MemError::Network(msg.into())
    }

    /// Create a raptor error
    pub fn raptor(msg: impl Into<String>) -> Self {
        MemError::Raptor(msg.into())
    }

    /// Create a RAG error
    pub fn rag(msg: impl Into<String>) -> Self {
        MemError::Rag(msg.into())
    }

    /// Create an indexing error
    pub fn indexing(msg: impl Into<String>) -> Self {
        MemError::Indexing(msg.into())
    }

    /// Create a query error
    pub fn query(msg: impl Into<String>) -> Self {
        MemError::Retrieval(msg.into())
    }

    /// Create a generation error (for summarization/LLM generation)
    pub fn generation(msg: impl Into<String>) -> Self {
        MemError::Retrieval(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MemError::Storage("connection failed".to_string());
        assert_eq!(err.to_string(), "Storage error: connection failed");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let mem_err: MemError = io_err.into();
        assert!(matches!(mem_err, MemError::Io(_)));
    }
}
