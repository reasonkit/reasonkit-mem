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

    // ============================================================
    // Dual-Layer Memory System Errors
    // ============================================================
    /// Hot memory errors (in-memory cache layer)
    #[error("Hot memory error: {0}")]
    HotMemory(String),

    /// Cold memory errors (persistent storage layer)
    #[error("Cold memory error: {0}")]
    ColdMemory(String),

    /// Write-Ahead Log (WAL) errors
    #[error("WAL error: {0}")]
    Wal(String),

    /// Synchronization errors between hot and cold layers
    #[error("Sync error: {0}")]
    Sync(String),

    /// Checksum verification failed
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Recovery error during WAL replay or crash recovery
    #[error("Recovery error: {0}")]
    Recovery(String),

    /// Capacity exceeded for memory limits
    #[error("Capacity exceeded: {current} / {max}")]
    CapacityExceeded { current: usize, max: usize },
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

    // ============================================================
    // Dual-Layer Memory System Helper Methods
    // ============================================================

    /// Create a hot memory error (in-memory cache layer)
    pub fn hot_memory(msg: impl Into<String>) -> Self {
        MemError::HotMemory(msg.into())
    }

    /// Create a cold memory error (persistent storage layer)
    pub fn cold_memory(msg: impl Into<String>) -> Self {
        MemError::ColdMemory(msg.into())
    }

    /// Create a WAL (Write-Ahead Log) error
    pub fn wal(msg: impl Into<String>) -> Self {
        MemError::Wal(msg.into())
    }

    /// Create a sync error between hot and cold layers
    pub fn sync(msg: impl Into<String>) -> Self {
        MemError::Sync(msg.into())
    }

    /// Create a checksum mismatch error
    pub fn checksum_mismatch(expected: u32, actual: u32) -> Self {
        MemError::ChecksumMismatch { expected, actual }
    }

    /// Create a recovery error
    pub fn recovery(msg: impl Into<String>) -> Self {
        MemError::Recovery(msg.into())
    }

    /// Create a capacity exceeded error
    pub fn capacity_exceeded(current: usize, max: usize) -> Self {
        MemError::CapacityExceeded { current, max }
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

    #[test]
    fn test_hot_memory_error() {
        let err = MemError::hot_memory("cache eviction failed");
        assert_eq!(err.to_string(), "Hot memory error: cache eviction failed");
    }

    #[test]
    fn test_cold_memory_error() {
        let err = MemError::cold_memory("disk write failed");
        assert_eq!(err.to_string(), "Cold memory error: disk write failed");
    }

    #[test]
    fn test_wal_error() {
        let err = MemError::wal("log corruption detected");
        assert_eq!(err.to_string(), "WAL error: log corruption detected");
    }

    #[test]
    fn test_sync_error() {
        let err = MemError::sync("hot-cold sync timeout");
        assert_eq!(err.to_string(), "Sync error: hot-cold sync timeout");
    }

    #[test]
    fn test_checksum_mismatch() {
        let err = MemError::checksum_mismatch(0xDEADBEEF, 0xCAFEBABE);
        assert_eq!(
            err.to_string(),
            "Checksum mismatch: expected 3735928559, got 3405691582"
        );
    }

    #[test]
    fn test_recovery_error() {
        let err = MemError::recovery("WAL replay failed at entry 42");
        assert_eq!(
            err.to_string(),
            "Recovery error: WAL replay failed at entry 42"
        );
    }

    #[test]
    fn test_capacity_exceeded() {
        let err = MemError::capacity_exceeded(1500, 1000);
        assert_eq!(err.to_string(), "Capacity exceeded: 1500 / 1000");
    }
}
