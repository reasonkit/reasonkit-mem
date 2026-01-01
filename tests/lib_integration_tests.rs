//! Integration tests for reasonkit-mem library interface
//!
//! These tests cover the main public API exposed through lib.rs:
//! - Core types (Document, Chunk, Source, Metadata, etc.)
//! - Error handling (MemError variants and helpers)
//! - RetrievalConfig and related configuration
//! - Prelude module exports
//!
//! # Test Organization
//!
//! Tests are organized by functionality:
//! - Document lifecycle (creation, modification, content handling)
//! - Type serialization (JSON round-trip, defaults)
//! - Error handling (all error variants, conversions)
//! - Configuration (defaults, custom values)
//! - Prelude exports (verify all expected types are accessible)

use chrono::Utc;
use reasonkit_mem::prelude::*;
use reasonkit_mem::{
    error::{MemError, MemResult},
    types::*,
    Error, Result,
};
use serde_json;
use std::collections::HashMap;
use uuid::Uuid;

// ============================================================================
// DOCUMENT LIFECYCLE TESTS
// ============================================================================

mod document_lifecycle {
    use super::*;

    /// Test creating a new document with minimal required fields
    #[test]
    fn test_document_creation_minimal() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/document.md".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc = Document::new(DocumentType::Note, source);

        assert_eq!(doc.doc_type, DocumentType::Note);
        assert!(doc.chunks.is_empty());
        assert_eq!(doc.content.raw, "");
        assert_eq!(doc.processing.status, ProcessingState::Pending);
    }

    /// Test creating a document with full content
    #[test]
    fn test_document_with_content() {
        let source = Source {
            source_type: SourceType::Website,
            url: Some("https://example.com/article".to_string()),
            path: None,
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: Some("1.0".to_string()),
        };

        let content = "This is a test document with some content for testing purposes.";
        let doc =
            Document::new(DocumentType::Documentation, source).with_content(content.to_string());

        assert_eq!(doc.content.raw, content);
        assert_eq!(doc.content.word_count, 11);
        assert_eq!(doc.content.char_count, content.len());
        assert_eq!(doc.content.format, ContentFormat::Text);
        assert_eq!(doc.content.language, "en");
    }

    /// Test document with metadata
    #[test]
    fn test_document_with_metadata() {
        let source = Source {
            source_type: SourceType::Arxiv,
            url: Some("https://arxiv.org/abs/2401.18059".to_string()),
            path: None,
            arxiv_id: Some("2401.18059".to_string()),
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let metadata = Metadata {
            title: Some("RAPTOR: Recursive Abstractive Processing".to_string()),
            authors: vec![Author {
                name: "Test Author".to_string(),
                affiliation: Some("Test University".to_string()),
                email: Some("test@example.com".to_string()),
            }],
            abstract_text: Some("A novel approach to hierarchical retrieval.".to_string()),
            date: Some("2024-01-15".to_string()),
            venue: Some("arXiv".to_string()),
            citations: Some(42),
            tags: vec!["rag".to_string(), "retrieval".to_string()],
            categories: vec!["cs.CL".to_string()],
            keywords: vec!["raptor".to_string(), "retrieval".to_string()],
            doi: Some("10.48550/arXiv.2401.18059".to_string()),
            license: Some("CC-BY-4.0".to_string()),
        };

        let doc = Document::new(DocumentType::Paper, source)
            .with_content("Paper content here.".to_string())
            .with_metadata(metadata);

        assert_eq!(doc.doc_type, DocumentType::Paper);
        assert_eq!(
            doc.metadata.title,
            Some("RAPTOR: Recursive Abstractive Processing".to_string())
        );
        assert_eq!(doc.metadata.authors.len(), 1);
        assert_eq!(doc.metadata.citations, Some(42));
    }

    /// Test all document types
    #[test]
    fn test_all_document_types() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let types = [
            DocumentType::Paper,
            DocumentType::Documentation,
            DocumentType::Code,
            DocumentType::Note,
            DocumentType::Transcript,
            DocumentType::Benchmark,
        ];

        for doc_type in types {
            let doc = Document::new(doc_type, source.clone());
            assert_eq!(doc.doc_type, doc_type);
        }
    }

    /// Test document ID uniqueness
    #[test]
    fn test_document_id_uniqueness() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc1 = Document::new(DocumentType::Note, source.clone());
        let doc2 = Document::new(DocumentType::Note, source.clone());

        assert_ne!(doc1.id, doc2.id);
    }

    /// Test document timestamps
    #[test]
    fn test_document_timestamps() {
        let before = Utc::now();

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc = Document::new(DocumentType::Note, source);

        let after = Utc::now();

        assert!(doc.created_at >= before);
        assert!(doc.created_at <= after);
        assert!(doc.updated_at.is_none());
    }
}

// ============================================================================
// CHUNK TESTS
// ============================================================================

mod chunk_tests {
    use super::*;

    /// Test chunk creation and fields
    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk {
            id: Uuid::new_v4(),
            text: "This is a test chunk of text.".to_string(),
            index: 0,
            start_char: 0,
            end_char: 29,
            token_count: Some(7),
            section: Some("Introduction".to_string()),
            page: Some(1),
            embedding_ids: EmbeddingIds::default(),
        };

        assert_eq!(chunk.text, "This is a test chunk of text.");
        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.token_count, Some(7));
        assert_eq!(chunk.section, Some("Introduction".to_string()));
    }

    /// Test EmbeddingIds default values
    #[test]
    fn test_embedding_ids_default() {
        let ids = EmbeddingIds::default();

        assert!(ids.dense.is_none());
        assert!(ids.sparse.is_none());
        assert!(ids.colbert.is_none());
    }

    /// Test EmbeddingIds with values
    #[test]
    fn test_embedding_ids_with_values() {
        let ids = EmbeddingIds {
            dense: Some("dense-123".to_string()),
            sparse: Some("sparse-456".to_string()),
            colbert: Some("colbert-789".to_string()),
        };

        assert_eq!(ids.dense, Some("dense-123".to_string()));
        assert_eq!(ids.sparse, Some("sparse-456".to_string()));
        assert_eq!(ids.colbert, Some("colbert-789".to_string()));
    }

    /// Test chunk serialization
    #[test]
    fn test_chunk_serialization() {
        let chunk = Chunk {
            id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            text: "Test chunk".to_string(),
            index: 5,
            start_char: 100,
            end_char: 200,
            token_count: Some(10),
            section: None,
            page: Some(3),
            embedding_ids: EmbeddingIds::default(),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();

        assert_eq!(chunk.id, deserialized.id);
        assert_eq!(chunk.text, deserialized.text);
        assert_eq!(chunk.index, deserialized.index);
        assert_eq!(chunk.page, deserialized.page);
    }
}

// ============================================================================
// SOURCE TYPE TESTS
// ============================================================================

mod source_type_tests {
    use super::*;

    /// Test all source types
    #[test]
    fn test_all_source_types() {
        let types = [
            SourceType::Arxiv,
            SourceType::Github,
            SourceType::Website,
            SourceType::Local,
            SourceType::Api,
        ];

        for source_type in types {
            let source = Source {
                source_type,
                url: None,
                path: None,
                arxiv_id: None,
                github_repo: None,
                retrieved_at: Utc::now(),
                version: None,
            };
            assert_eq!(source.source_type, source_type);
        }
    }

    /// Test source type serialization
    #[test]
    fn test_source_type_serialization() {
        let source = Source {
            source_type: SourceType::Github,
            url: Some("https://github.com/user/repo".to_string()),
            path: None,
            arxiv_id: None,
            github_repo: Some("user/repo".to_string()),
            retrieved_at: Utc::now(),
            version: Some("main".to_string()),
        };

        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains("\"type\":\"github\""));

        let deserialized: Source = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.source_type, SourceType::Github);
        assert_eq!(deserialized.github_repo, Some("user/repo".to_string()));
    }

    /// Test source with arXiv fields
    #[test]
    fn test_source_arxiv() {
        let source = Source {
            source_type: SourceType::Arxiv,
            url: Some("https://arxiv.org/abs/2401.18059".to_string()),
            path: None,
            arxiv_id: Some("2401.18059".to_string()),
            github_repo: None,
            retrieved_at: Utc::now(),
            version: Some("v2".to_string()),
        };

        assert_eq!(source.arxiv_id, Some("2401.18059".to_string()));
    }
}

// ============================================================================
// PROCESSING STATUS TESTS
// ============================================================================

mod processing_status_tests {
    use super::*;

    /// Test default processing status
    #[test]
    fn test_processing_status_default() {
        let status = ProcessingStatus::default();

        assert_eq!(status.status, ProcessingState::Pending);
        assert!(!status.chunked);
        assert!(!status.embedded);
        assert!(!status.indexed);
        assert!(!status.raptor_processed);
        assert!(status.errors.is_empty());
    }

    /// Test processing state values
    #[test]
    fn test_processing_states() {
        let states = [
            ProcessingState::Pending,
            ProcessingState::Processing,
            ProcessingState::Completed,
            ProcessingState::Failed,
        ];

        for state in states {
            let status = ProcessingStatus {
                status: state,
                ..Default::default()
            };
            assert_eq!(status.status, state);
        }
    }

    /// Test processing status with errors
    #[test]
    fn test_processing_status_with_errors() {
        let status = ProcessingStatus {
            status: ProcessingState::Failed,
            chunked: true,
            embedded: false,
            indexed: false,
            raptor_processed: false,
            errors: vec![
                "Embedding failed: API timeout".to_string(),
                "Retry limit exceeded".to_string(),
            ],
        };

        assert_eq!(status.status, ProcessingState::Failed);
        assert!(status.chunked);
        assert!(!status.embedded);
        assert_eq!(status.errors.len(), 2);
    }

    /// Test processing state serialization
    #[test]
    fn test_processing_state_serialization() {
        let status = ProcessingStatus {
            status: ProcessingState::Completed,
            chunked: true,
            embedded: true,
            indexed: true,
            raptor_processed: true,
            errors: vec![],
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"status\":\"completed\""));

        let deserialized: ProcessingStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.status, ProcessingState::Completed);
    }
}

// ============================================================================
// SEARCH RESULT TESTS
// ============================================================================

mod search_result_tests {
    use super::*;

    /// Test search result creation
    #[test]
    fn test_search_result_creation() {
        let chunk = Chunk {
            id: Uuid::new_v4(),
            text: "Matched text".to_string(),
            index: 0,
            start_char: 0,
            end_char: 12,
            token_count: Some(2),
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        };

        let result = SearchResult {
            score: 0.95,
            document_id: Uuid::new_v4(),
            chunk: chunk.clone(),
            match_source: MatchSource::Hybrid,
        };

        assert!((result.score - 0.95).abs() < f32::EPSILON);
        assert_eq!(result.match_source, MatchSource::Hybrid);
    }

    /// Test all match sources
    #[test]
    fn test_match_sources() {
        let sources = [
            MatchSource::Dense,
            MatchSource::Sparse,
            MatchSource::Hybrid,
            MatchSource::Raptor,
        ];

        for source in sources {
            let chunk = Chunk {
                id: Uuid::new_v4(),
                text: "Test".to_string(),
                index: 0,
                start_char: 0,
                end_char: 4,
                token_count: None,
                section: None,
                page: None,
                embedding_ids: EmbeddingIds::default(),
            };

            let result = SearchResult {
                score: 0.5,
                document_id: Uuid::new_v4(),
                chunk,
                match_source: source,
            };

            assert_eq!(result.match_source, source);
        }
    }

    /// Test match source serialization
    #[test]
    fn test_match_source_serialization() {
        let sources = [
            (MatchSource::Dense, "\"dense\""),
            (MatchSource::Sparse, "\"sparse\""),
            (MatchSource::Hybrid, "\"hybrid\""),
            (MatchSource::Raptor, "\"raptor\""),
        ];

        for (source, expected) in sources {
            let json = serde_json::to_string(&source).unwrap();
            assert_eq!(json, expected);
        }
    }
}

// ============================================================================
// RETRIEVAL CONFIG TESTS
// ============================================================================

mod retrieval_config_tests {
    use super::*;

    /// Test default retrieval configuration
    #[test]
    fn test_retrieval_config_default() {
        let config = RetrievalConfig::default();

        assert_eq!(config.top_k, 10);
        assert!((config.min_score - 0.0).abs() < f32::EPSILON);
        assert!((config.alpha - 0.7).abs() < f32::EPSILON);
        assert!(!config.use_raptor);
        assert!(!config.rerank);
    }

    /// Test custom retrieval configuration
    #[test]
    fn test_retrieval_config_custom() {
        let config = RetrievalConfig {
            top_k: 20,
            min_score: 0.5,
            alpha: 0.3,
            use_raptor: true,
            rerank: true,
        };

        assert_eq!(config.top_k, 20);
        assert!((config.min_score - 0.5).abs() < f32::EPSILON);
        assert!((config.alpha - 0.3).abs() < f32::EPSILON);
        assert!(config.use_raptor);
        assert!(config.rerank);
    }

    /// Test retrieval config alpha boundaries
    #[test]
    fn test_retrieval_config_alpha_values() {
        // Alpha = 0 means sparse only
        let sparse_only = RetrievalConfig {
            alpha: 0.0,
            ..Default::default()
        };
        assert!((sparse_only.alpha - 0.0).abs() < f32::EPSILON);

        // Alpha = 1 means dense only
        let dense_only = RetrievalConfig {
            alpha: 1.0,
            ..Default::default()
        };
        assert!((dense_only.alpha - 1.0).abs() < f32::EPSILON);

        // Alpha = 0.5 means balanced
        let balanced = RetrievalConfig {
            alpha: 0.5,
            ..Default::default()
        };
        assert!((balanced.alpha - 0.5).abs() < f32::EPSILON);
    }

    /// Test retrieval config serialization
    #[test]
    fn test_retrieval_config_serialization() {
        let config = RetrievalConfig {
            top_k: 15,
            min_score: 0.3,
            alpha: 0.8,
            use_raptor: true,
            rerank: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RetrievalConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.top_k, deserialized.top_k);
        assert!((config.min_score - deserialized.min_score).abs() < f32::EPSILON);
        assert!((config.alpha - deserialized.alpha).abs() < f32::EPSILON);
        assert_eq!(config.use_raptor, deserialized.use_raptor);
        assert_eq!(config.rerank, deserialized.rerank);
    }
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

mod error_handling_tests {
    use super::*;
    use std::io;

    /// Test all error helper methods
    #[test]
    fn test_error_helper_methods() {
        let errors = vec![
            (
                MemError::storage("storage failed"),
                "Storage error: storage failed",
            ),
            (
                MemError::embedding("embedding failed"),
                "Embedding error: embedding failed",
            ),
            (
                MemError::retrieval("retrieval failed"),
                "Retrieval error: retrieval failed",
            ),
            (
                MemError::qdrant("qdrant failed"),
                "Qdrant error: qdrant failed",
            ),
            (
                MemError::config("config failed"),
                "Configuration error: config failed",
            ),
            (
                MemError::not_found("doc not found"),
                "Document not found: doc not found",
            ),
            (
                MemError::parse("parse failed"),
                "Serialization error: parse failed",
            ),
            (
                MemError::tantivy("tantivy failed"),
                "Tantivy error: tantivy failed",
            ),
            (MemError::invalid_input("invalid"), "Invalid input: invalid"),
            (
                MemError::validation("validation failed"),
                "Invalid input: validation failed",
            ),
            (
                MemError::network("network failed"),
                "Network error: network failed",
            ),
            (
                MemError::raptor("raptor failed"),
                "RAPTOR error: raptor failed",
            ),
            (MemError::rag("rag failed"), "RAG error: rag failed"),
            (
                MemError::indexing("indexing failed"),
                "Indexing error: indexing failed",
            ),
            (
                MemError::query("query failed"),
                "Retrieval error: query failed",
            ),
            (
                MemError::generation("generation failed"),
                "Retrieval error: generation failed",
            ),
        ];

        for (error, expected_msg) in errors {
            assert_eq!(error.to_string(), expected_msg);
        }
    }

    /// Test dual-layer memory error helpers
    #[test]
    fn test_dual_layer_error_helpers() {
        let errors = vec![
            (
                MemError::hot_memory("hot failed"),
                "Hot memory error: hot failed",
            ),
            (
                MemError::cold_memory("cold failed"),
                "Cold memory error: cold failed",
            ),
            (MemError::wal("wal failed"), "WAL error: wal failed"),
            (MemError::sync("sync failed"), "Sync error: sync failed"),
            (
                MemError::recovery("recovery failed"),
                "Recovery error: recovery failed",
            ),
        ];

        for (error, expected_msg) in errors {
            assert_eq!(error.to_string(), expected_msg);
        }
    }

    /// Test checksum mismatch error
    #[test]
    fn test_checksum_mismatch_error() {
        let error = MemError::checksum_mismatch(0xDEADBEEF, 0xCAFEBABE);
        let msg = error.to_string();

        assert!(msg.contains("Checksum mismatch"));
        assert!(msg.contains("3735928559")); // 0xDEADBEEF in decimal
        assert!(msg.contains("3405691582")); // 0xCAFEBABE in decimal
    }

    /// Test capacity exceeded error
    #[test]
    fn test_capacity_exceeded_error() {
        let error = MemError::capacity_exceeded(1500, 1000);
        assert_eq!(error.to_string(), "Capacity exceeded: 1500 / 1000");
    }

    /// Test error conversion from io::Error
    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let mem_err: MemError = io_err.into();

        match mem_err {
            MemError::Io(_) => (),
            _ => panic!("Expected MemError::Io"),
        }
    }

    /// Test error conversion from serde_json::Error
    #[test]
    fn test_error_from_serde_json() {
        let json_err = serde_json::from_str::<i32>("invalid").unwrap_err();
        let mem_err: MemError = json_err.into();

        match mem_err {
            MemError::Serialization(msg) => {
                assert!(!msg.is_empty());
            }
            _ => panic!("Expected MemError::Serialization"),
        }
    }

    /// Test Result type alias
    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }

        fn returns_err() -> Result<i32> {
            Err(Error::storage("test error"))
        }

        assert_eq!(returns_ok().unwrap(), 42);
        assert!(returns_err().is_err());
    }

    /// Test MemResult type alias
    #[test]
    fn test_mem_result_type_alias() {
        fn returns_ok() -> MemResult<String> {
            Ok("success".to_string())
        }

        fn returns_err() -> MemResult<String> {
            Err(MemError::not_found("missing"))
        }

        assert_eq!(returns_ok().unwrap(), "success");
        assert!(returns_err().is_err());
    }
}

// ============================================================================
// CONTENT FORMAT TESTS
// ============================================================================

mod content_format_tests {
    use super::*;

    /// Test all content formats
    #[test]
    fn test_content_formats() {
        let formats = [
            ContentFormat::Text,
            ContentFormat::Markdown,
            ContentFormat::Html,
            ContentFormat::Latex,
        ];

        for format in formats {
            let content = DocumentContent {
                raw: "content".to_string(),
                format,
                language: "en".to_string(),
                word_count: 1,
                char_count: 7,
            };
            assert_eq!(content.format, format);
        }
    }

    /// Test content format default
    #[test]
    fn test_content_format_default() {
        let content = DocumentContent::default();
        assert_eq!(content.format, ContentFormat::Text);
    }

    /// Test content format serialization
    #[test]
    fn test_content_format_serialization() {
        let formats = [
            (ContentFormat::Text, "\"text\""),
            (ContentFormat::Markdown, "\"markdown\""),
            (ContentFormat::Html, "\"html\""),
            (ContentFormat::Latex, "\"latex\""),
        ];

        for (format, expected) in formats {
            let json = serde_json::to_string(&format).unwrap();
            assert_eq!(json, expected);
        }
    }
}

// ============================================================================
// METADATA TESTS
// ============================================================================

mod metadata_tests {
    use super::*;

    /// Test metadata default values
    #[test]
    fn test_metadata_default() {
        let metadata = Metadata::default();

        assert!(metadata.title.is_none());
        assert!(metadata.authors.is_empty());
        assert!(metadata.abstract_text.is_none());
        assert!(metadata.date.is_none());
        assert!(metadata.venue.is_none());
        assert!(metadata.citations.is_none());
        assert!(metadata.tags.is_empty());
        assert!(metadata.categories.is_empty());
        assert!(metadata.keywords.is_empty());
        assert!(metadata.doi.is_none());
        assert!(metadata.license.is_none());
    }

    /// Test author creation
    #[test]
    fn test_author_creation() {
        let author = Author {
            name: "Jane Doe".to_string(),
            affiliation: Some("MIT".to_string()),
            email: Some("jane@mit.edu".to_string()),
        };

        assert_eq!(author.name, "Jane Doe");
        assert_eq!(author.affiliation, Some("MIT".to_string()));
    }

    /// Test metadata serialization
    #[test]
    fn test_metadata_serialization() {
        let metadata = Metadata {
            title: Some("Test Paper".to_string()),
            authors: vec![Author {
                name: "Test Author".to_string(),
                affiliation: None,
                email: None,
            }],
            abstract_text: Some("Abstract here.".to_string()),
            date: Some("2024-01-01".to_string()),
            venue: None,
            citations: Some(10),
            tags: vec!["ai".to_string()],
            categories: vec!["cs.AI".to_string()],
            keywords: vec!["machine learning".to_string()],
            doi: None,
            license: Some("MIT".to_string()),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: Metadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata.title, deserialized.title);
        assert_eq!(metadata.authors.len(), deserialized.authors.len());
        assert_eq!(metadata.citations, deserialized.citations);
    }
}

// ============================================================================
// PRELUDE TESTS
// ============================================================================

mod prelude_tests {
    use reasonkit_mem::prelude::*;

    /// Test that prelude exports MemError
    #[test]
    fn test_prelude_exports_mem_error() {
        let _error: MemError = MemError::Storage("test".to_string());
    }

    /// Test that prelude exports MemResult
    #[test]
    fn test_prelude_exports_mem_result() {
        let _result: MemResult<i32> = Ok(42);
    }

    /// Test that prelude exports core types
    #[test]
    fn test_prelude_exports_types() {
        // Document type is accessible
        let _doc_type = DocumentType::Note;

        // Source type is accessible
        let _source_type = SourceType::Local;

        // Content format is accessible
        let _format = ContentFormat::Text;

        // Processing state is accessible
        let _state = ProcessingState::Pending;

        // Match source is accessible
        let _match = MatchSource::Hybrid;
    }
}

// ============================================================================
// DOCUMENT SERIALIZATION TESTS
// ============================================================================

mod document_serialization_tests {
    use super::*;

    /// Test full document serialization round-trip
    #[test]
    fn test_document_full_serialization() {
        let source = Source {
            source_type: SourceType::Github,
            url: Some("https://github.com/user/repo".to_string()),
            path: None,
            arxiv_id: None,
            github_repo: Some("user/repo".to_string()),
            retrieved_at: Utc::now(),
            version: Some("v1.0.0".to_string()),
        };

        let mut doc = Document::new(DocumentType::Code, source)
            .with_content("fn main() { println!(\"Hello\"); }".to_string());

        doc.chunks = vec![Chunk {
            id: Uuid::new_v4(),
            text: "fn main() { println!(\"Hello\"); }".to_string(),
            index: 0,
            start_char: 0,
            end_char: 33,
            token_count: Some(6),
            section: Some("main".to_string()),
            page: None,
            embedding_ids: EmbeddingIds {
                dense: Some("emb-001".to_string()),
                sparse: None,
                colbert: None,
            },
        }];

        let json = serde_json::to_string_pretty(&doc).unwrap();
        let deserialized: Document = serde_json::from_str(&json).unwrap();

        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.doc_type, deserialized.doc_type);
        assert_eq!(doc.content.raw, deserialized.content.raw);
        assert_eq!(doc.chunks.len(), deserialized.chunks.len());
        assert_eq!(doc.chunks[0].text, deserialized.chunks[0].text);
    }

    /// Test document with all fields populated
    #[test]
    fn test_document_comprehensive() {
        let source = Source {
            source_type: SourceType::Arxiv,
            url: Some("https://arxiv.org/abs/2401.18059".to_string()),
            path: Some("/cache/2401.18059.pdf".to_string()),
            arxiv_id: Some("2401.18059".to_string()),
            github_repo: None,
            retrieved_at: Utc::now(),
            version: Some("v1".to_string()),
        };

        let metadata = Metadata {
            title: Some("RAPTOR Paper".to_string()),
            authors: vec![
                Author {
                    name: "Author One".to_string(),
                    affiliation: Some("Stanford".to_string()),
                    email: Some("a1@stanford.edu".to_string()),
                },
                Author {
                    name: "Author Two".to_string(),
                    affiliation: Some("MIT".to_string()),
                    email: None,
                },
            ],
            abstract_text: Some("This paper presents RAPTOR.".to_string()),
            date: Some("2024-01-15".to_string()),
            venue: Some("arXiv".to_string()),
            citations: Some(100),
            tags: vec!["rag".to_string(), "retrieval".to_string()],
            categories: vec!["cs.CL".to_string(), "cs.AI".to_string()],
            keywords: vec!["raptor".to_string(), "tree".to_string()],
            doi: Some("10.48550/arXiv.2401.18059".to_string()),
            license: Some("arXiv".to_string()),
        };

        let mut doc = Document::new(DocumentType::Paper, source)
            .with_content("Full paper content here with many words.".to_string())
            .with_metadata(metadata);

        doc.processing = ProcessingStatus {
            status: ProcessingState::Completed,
            chunked: true,
            embedded: true,
            indexed: true,
            raptor_processed: true,
            errors: vec![],
        };

        let json = serde_json::to_string(&doc).unwrap();
        assert!(json.len() > 100);

        let deserialized: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(
            doc.metadata.authors.len(),
            deserialized.metadata.authors.len()
        );
        assert_eq!(doc.processing.status, deserialized.processing.status);
    }
}

// ============================================================================
// EDGE CASES AND BOUNDARY TESTS
// ============================================================================

mod edge_cases {
    use super::*;

    /// Test document with empty content
    #[test]
    fn test_document_empty_content() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/empty.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc = Document::new(DocumentType::Note, source).with_content("".to_string());

        assert_eq!(doc.content.raw, "");
        assert_eq!(doc.content.word_count, 0);
        assert_eq!(doc.content.char_count, 0);
    }

    /// Test document with very long content
    #[test]
    fn test_document_long_content() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/long.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let long_content = "word ".repeat(10000);
        let doc =
            Document::new(DocumentType::Documentation, source).with_content(long_content.clone());

        assert_eq!(doc.content.word_count, 10000);
        assert_eq!(doc.content.char_count, long_content.len());
    }

    /// Test chunk with unicode content
    #[test]
    fn test_chunk_unicode() {
        let chunk = Chunk {
            id: Uuid::new_v4(),
            text: "Hello World".to_string(),
            index: 0,
            start_char: 0,
            end_char: 15, // Unicode chars may have different byte lengths
            token_count: Some(2),
            section: Some("Intro".to_string()),
            page: None,
            embedding_ids: EmbeddingIds::default(),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();

        assert!(deserialized.text.contains("World"));
    }

    /// Test retrieval config with extreme values
    #[test]
    fn test_retrieval_config_extreme_values() {
        let config = RetrievalConfig {
            top_k: usize::MAX,
            min_score: f32::MIN,
            alpha: 0.99999,
            use_raptor: true,
            rerank: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RetrievalConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.top_k, deserialized.top_k);
    }

    /// Test document with many chunks
    #[test]
    fn test_document_many_chunks() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/large_doc.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let mut doc = Document::new(DocumentType::Documentation, source);

        // Add 1000 chunks
        for i in 0..1000 {
            doc.chunks.push(Chunk {
                id: Uuid::new_v4(),
                text: format!("Chunk {} content", i),
                index: i,
                start_char: i * 20,
                end_char: (i + 1) * 20,
                token_count: Some(3),
                section: None,
                page: Some(i / 50),
                embedding_ids: EmbeddingIds::default(),
            });
        }

        assert_eq!(doc.chunks.len(), 1000);

        // Ensure serialization works with many chunks
        let json = serde_json::to_string(&doc).unwrap();
        let deserialized: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.chunks.len(), 1000);
    }

    /// Test metadata with empty arrays
    #[test]
    fn test_metadata_empty_arrays() {
        let metadata = Metadata {
            title: Some("Title".to_string()),
            authors: vec![],
            abstract_text: None,
            date: None,
            venue: None,
            citations: None,
            tags: vec![],
            categories: vec![],
            keywords: vec![],
            doi: None,
            license: None,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: Metadata = serde_json::from_str(&json).unwrap();

        assert!(deserialized.authors.is_empty());
        assert!(deserialized.tags.is_empty());
    }
}

// ============================================================================
// TYPE EQUALITY TESTS
// ============================================================================

mod type_equality {
    use super::*;

    /// Test DocumentType equality
    #[test]
    fn test_document_type_equality() {
        assert_eq!(DocumentType::Paper, DocumentType::Paper);
        assert_ne!(DocumentType::Paper, DocumentType::Code);
    }

    /// Test SourceType equality
    #[test]
    fn test_source_type_equality() {
        assert_eq!(SourceType::Arxiv, SourceType::Arxiv);
        assert_ne!(SourceType::Arxiv, SourceType::Github);
    }

    /// Test ContentFormat equality
    #[test]
    fn test_content_format_equality() {
        assert_eq!(ContentFormat::Text, ContentFormat::Text);
        assert_ne!(ContentFormat::Text, ContentFormat::Markdown);
    }

    /// Test ProcessingState equality
    #[test]
    fn test_processing_state_equality() {
        assert_eq!(ProcessingState::Pending, ProcessingState::Pending);
        assert_ne!(ProcessingState::Pending, ProcessingState::Completed);
    }

    /// Test MatchSource equality
    #[test]
    fn test_match_source_equality() {
        assert_eq!(MatchSource::Dense, MatchSource::Dense);
        assert_ne!(MatchSource::Dense, MatchSource::Sparse);
    }
}

// ============================================================================
// CLONE AND DEBUG TESTS
// ============================================================================

mod clone_debug_tests {
    use super::*;

    /// Test Document clone
    #[test]
    fn test_document_clone() {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc = Document::new(DocumentType::Note, source).with_content("Test".to_string());
        let cloned = doc.clone();

        assert_eq!(doc.id, cloned.id);
        assert_eq!(doc.content.raw, cloned.content.raw);
    }

    /// Test Chunk clone
    #[test]
    fn test_chunk_clone() {
        let chunk = Chunk {
            id: Uuid::new_v4(),
            text: "Test chunk".to_string(),
            index: 0,
            start_char: 0,
            end_char: 10,
            token_count: Some(2),
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        };

        let cloned = chunk.clone();
        assert_eq!(chunk.id, cloned.id);
        assert_eq!(chunk.text, cloned.text);
    }

    /// Test RetrievalConfig clone
    #[test]
    fn test_retrieval_config_clone() {
        let config = RetrievalConfig {
            top_k: 20,
            min_score: 0.5,
            alpha: 0.8,
            use_raptor: true,
            rerank: true,
        };

        let cloned = config.clone();
        assert_eq!(config.top_k, cloned.top_k);
        assert!((config.alpha - cloned.alpha).abs() < f32::EPSILON);
    }

    /// Test Debug implementations
    #[test]
    fn test_debug_implementations() {
        let chunk = Chunk {
            id: Uuid::new_v4(),
            text: "Debug test".to_string(),
            index: 0,
            start_char: 0,
            end_char: 10,
            token_count: None,
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        };

        let debug_str = format!("{:?}", chunk);
        assert!(debug_str.contains("Debug test"));

        let config = RetrievalConfig::default();
        let config_debug = format!("{:?}", config);
        assert!(config_debug.contains("top_k"));

        let error = MemError::storage("test error");
        let error_debug = format!("{:?}", error);
        assert!(error_debug.contains("Storage"));
    }
}
