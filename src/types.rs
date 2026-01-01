//! Core types for memory and retrieval operations.
//!
//! This module defines the fundamental data structures used throughout the
//! `reasonkit-mem` crate. These types represent documents, chunks, metadata,
//! and retrieval configurations.
//!
//! # Overview
//!
//! The type hierarchy is:
//!
//! ```text
//! Document
//! +-- id: Uuid
//! +-- doc_type: DocumentType
//! +-- source: Source
//! +-- content: DocumentContent
//! +-- metadata: Metadata
//! +-- processing: ProcessingStatus
//! +-- chunks: Vec<Chunk>
//!     +-- id: Uuid
//!     +-- text: String
//!     +-- embedding_ids: EmbeddingIds
//! ```
//!
//! # Example
//!
//! ```rust
//! use reasonkit_mem::{Document, DocumentType, Source, SourceType, Metadata, Author};
//! use chrono::Utc;
//!
//! // Create a source
//! let source = Source {
//!     source_type: SourceType::Arxiv,
//!     url: Some("https://arxiv.org/abs/2401.18059".to_string()),
//!     path: None,
//!     arxiv_id: Some("2401.18059".to_string()),
//!     github_repo: None,
//!     retrieved_at: Utc::now(),
//!     version: None,
//! };
//!
//! // Create a document
//! let mut doc = Document::new(DocumentType::Paper, source)
//!     .with_content("Abstract: This paper presents...".to_string());
//!
//! // Add metadata
//! doc.metadata = Metadata {
//!     title: Some("RAPTOR: Recursive Abstractive Processing".to_string()),
//!     authors: vec![Author {
//!         name: "Sarthi et al.".to_string(),
//!         affiliation: Some("Stanford University".to_string()),
//!         email: None,
//!     }],
//!     ..Default::default()
//! };
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Document type categorization.
///
/// Categorizes documents by their type to enable type-specific processing
/// and retrieval strategies.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::DocumentType;
///
/// let doc_type = DocumentType::Paper;
/// assert_eq!(doc_type, DocumentType::Paper);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    /// Academic paper (PDF, typically from arXiv or journals)
    Paper,
    /// Technical documentation (README, API docs, guides)
    Documentation,
    /// Source code (Rust, Python, etc.)
    Code,
    /// User notes (personal notes, meeting notes)
    Note,
    /// Meeting/interview transcript
    Transcript,
    /// Benchmark data (performance metrics, test results)
    Benchmark,
}

/// Source information for a document.
///
/// Tracks where a document came from, when it was retrieved, and any
/// relevant identifiers (arXiv ID, GitHub repo, etc.).
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::{Source, SourceType};
/// use chrono::Utc;
///
/// let source = Source {
///     source_type: SourceType::Github,
///     url: Some("https://github.com/org/repo".to_string()),
///     path: None,
///     arxiv_id: None,
///     github_repo: Some("org/repo".to_string()),
///     retrieved_at: Utc::now(),
///     version: Some("v1.0.0".to_string()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Source type (arXiv, GitHub, local file, etc.)
    #[serde(rename = "type")]
    pub source_type: SourceType,
    /// Original URL where the document was retrieved
    pub url: Option<String>,
    /// Local file path (if applicable)
    pub path: Option<String>,
    /// arXiv ID (e.g., "2401.18059")
    pub arxiv_id: Option<String>,
    /// GitHub repository (e.g., "anthropics/claude-code")
    pub github_repo: Option<String>,
    /// When the document was retrieved
    pub retrieved_at: DateTime<Utc>,
    /// Version or commit hash
    pub version: Option<String>,
}

/// Source type enumeration.
///
/// Identifies the origin of a document for provenance tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// arXiv paper
    Arxiv,
    /// GitHub repository
    Github,
    /// Website
    Website,
    /// Local file
    Local,
    /// API response
    Api,
}

/// Document metadata.
///
/// Contains bibliographic and descriptive metadata about a document,
/// including title, authors, abstract, publication information, and tags.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::{Metadata, Author};
///
/// let metadata = Metadata {
///     title: Some("Machine Learning Fundamentals".to_string()),
///     authors: vec![
///         Author {
///             name: "Alice Smith".to_string(),
///             affiliation: Some("MIT".to_string()),
///             email: None,
///         }
///     ],
///     tags: vec!["ml".to_string(), "tutorial".to_string()],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Document title
    pub title: Option<String>,
    /// Authors of the document
    pub authors: Vec<Author>,
    /// Abstract or summary
    #[serde(rename = "abstract")]
    pub abstract_text: Option<String>,
    /// Publication/creation date (ISO 8601 string)
    pub date: Option<String>,
    /// Publication venue (journal, conference, etc.)
    pub venue: Option<String>,
    /// Citation count (if available)
    pub citations: Option<i32>,
    /// User-defined tags for categorization
    pub tags: Vec<String>,
    /// ReasonKit-specific categories
    pub categories: Vec<String>,
    /// Extracted keywords
    pub keywords: Vec<String>,
    /// Digital Object Identifier
    pub doi: Option<String>,
    /// Content license
    pub license: Option<String>,
}

/// Author information.
///
/// Represents an author with name, affiliation, and contact information.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::Author;
///
/// let author = Author {
///     name: "John Doe".to_string(),
///     affiliation: Some("Stanford University".to_string()),
///     email: Some("jdoe@stanford.edu".to_string()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    /// Author name
    pub name: String,
    /// Institutional affiliation
    pub affiliation: Option<String>,
    /// Email address
    pub email: Option<String>,
}

/// A chunk of text from a document.
///
/// Documents are split into chunks for efficient retrieval. Each chunk
/// contains the text content, position information, and references to
/// associated embeddings.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::{Chunk, EmbeddingIds};
/// use uuid::Uuid;
///
/// let chunk = Chunk {
///     id: Uuid::new_v4(),
///     text: "This is the chunk content.".to_string(),
///     index: 0,
///     start_char: 0,
///     end_char: 26,
///     token_count: Some(6),
///     section: Some("Introduction".to_string()),
///     page: Some(1),
///     embedding_ids: EmbeddingIds::default(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: Uuid,
    /// Chunk text content
    pub text: String,
    /// Position in document (0-indexed)
    pub index: usize,
    /// Start character offset in original document
    pub start_char: usize,
    /// End character offset in original document
    pub end_char: usize,
    /// Approximate token count
    pub token_count: Option<usize>,
    /// Section heading (if applicable)
    pub section: Option<String>,
    /// Page number (for PDFs)
    pub page: Option<usize>,
    /// Associated embedding IDs
    pub embedding_ids: EmbeddingIds,
}

/// References to different embedding types for a chunk.
///
/// Tracks the IDs of embeddings stored in different vector stores.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::EmbeddingIds;
///
/// let ids = EmbeddingIds {
///     dense: Some("emb_123456".to_string()),
///     sparse: None,
///     colbert: None,
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingIds {
    /// Dense embedding ID (in Qdrant)
    pub dense: Option<String>,
    /// Sparse embedding ID (for BM25/SPLADE)
    pub sparse: Option<String>,
    /// ColBERT embedding ID (for late interaction)
    pub colbert: Option<String>,
}

/// Processing status for a document.
///
/// Tracks the processing pipeline status for a document, including
/// chunking, embedding, indexing, and RAPTOR tree building.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::{ProcessingStatus, ProcessingState};
///
/// let status = ProcessingStatus {
///     status: ProcessingState::Completed,
///     chunked: true,
///     embedded: true,
///     indexed: true,
///     raptor_processed: false,
///     errors: vec![],
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStatus {
    /// Overall processing status
    pub status: ProcessingState,
    /// Whether the document has been chunked
    pub chunked: bool,
    /// Whether embeddings have been generated
    pub embedded: bool,
    /// Whether the document has been indexed (BM25)
    pub indexed: bool,
    /// Whether RAPTOR tree has been built
    pub raptor_processed: bool,
    /// Error messages from processing
    pub errors: Vec<String>,
}

/// Processing state enumeration.
///
/// Represents the current state of document processing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingState {
    /// Not yet processed
    #[default]
    Pending,
    /// Currently processing
    Processing,
    /// Successfully completed
    Completed,
    /// Failed with errors
    Failed,
}

/// A document in the knowledge base.
///
/// The primary data structure for storing documents. Contains all information
/// about a document including content, metadata, processing status, and chunks.
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
///     path: Some("/path/to/doc.md".to_string()),
///     arxiv_id: None,
///     github_repo: None,
///     retrieved_at: Utc::now(),
///     version: None,
/// };
///
/// let doc = Document::new(DocumentType::Note, source)
///     .with_content("Document content here.".to_string());
///
/// assert_eq!(doc.doc_type, DocumentType::Note);
/// assert_eq!(doc.content.word_count, 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: Uuid,
    /// Document type
    #[serde(rename = "type")]
    pub doc_type: DocumentType,
    /// Source information
    pub source: Source,
    /// Raw content
    pub content: DocumentContent,
    /// Metadata
    pub metadata: Metadata,
    /// Processing status
    pub processing: ProcessingStatus,
    /// Document chunks
    pub chunks: Vec<Chunk>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: Option<DateTime<Utc>>,
}

/// Document content.
///
/// Contains the raw text content of a document along with format
/// information and statistics.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::{DocumentContent, ContentFormat};
///
/// let content = DocumentContent {
///     raw: "Hello, world!".to_string(),
///     format: ContentFormat::Text,
///     language: "en".to_string(),
///     word_count: 2,
///     char_count: 13,
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentContent {
    /// Full extracted text
    pub raw: String,
    /// Content format
    pub format: ContentFormat,
    /// Language code (ISO 639-1)
    pub language: String,
    /// Word count
    pub word_count: usize,
    /// Character count
    pub char_count: usize,
}

/// Content format.
///
/// Identifies the format of the document content for proper rendering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentFormat {
    /// Plain text
    #[default]
    Text,
    /// Markdown
    Markdown,
    /// HTML
    Html,
    /// LaTeX
    Latex,
}

impl Document {
    /// Create a new document with the given type and source.
    ///
    /// # Arguments
    ///
    /// * `doc_type` - The type of document (Paper, Code, Note, etc.)
    /// * `source` - Source information for the document
    ///
    /// # Returns
    ///
    /// A new `Document` with default content, metadata, and processing status.
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
    /// let doc = Document::new(DocumentType::Note, source);
    /// ```
    pub fn new(doc_type: DocumentType, source: Source) -> Self {
        Self {
            id: Uuid::new_v4(),
            doc_type,
            source,
            content: DocumentContent::default(),
            metadata: Metadata::default(),
            processing: ProcessingStatus::default(),
            chunks: Vec::new(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    /// Set the raw content of the document.
    ///
    /// Automatically computes word and character counts.
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw text content
    ///
    /// # Returns
    ///
    /// The document with updated content.
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
    ///     path: None,
    ///     arxiv_id: None,
    ///     github_repo: None,
    ///     retrieved_at: Utc::now(),
    ///     version: None,
    /// };
    ///
    /// let doc = Document::new(DocumentType::Note, source)
    ///     .with_content("Hello, world!".to_string());
    ///
    /// assert_eq!(doc.content.word_count, 2);
    /// assert_eq!(doc.content.char_count, 13);
    /// ```
    pub fn with_content(mut self, raw: String) -> Self {
        let word_count = raw.split_whitespace().count();
        let char_count = raw.len();
        self.content = DocumentContent {
            raw,
            format: ContentFormat::Text,
            language: "en".to_string(),
            word_count,
            char_count,
        };
        self
    }

    /// Set the metadata for the document.
    ///
    /// # Arguments
    ///
    /// * `metadata` - The document metadata
    ///
    /// # Returns
    ///
    /// The document with updated metadata.
    ///
    /// # Example
    ///
    /// ```rust
    /// use reasonkit_mem::{Document, DocumentType, Source, SourceType, Metadata};
    /// use chrono::Utc;
    ///
    /// let source = Source {
    ///     source_type: SourceType::Local,
    ///     url: None,
    ///     path: None,
    ///     arxiv_id: None,
    ///     github_repo: None,
    ///     retrieved_at: Utc::now(),
    ///     version: None,
    /// };
    ///
    /// let metadata = Metadata {
    ///     title: Some("My Document".to_string()),
    ///     ..Default::default()
    /// };
    ///
    /// let doc = Document::new(DocumentType::Note, source)
    ///     .with_metadata(metadata);
    ///
    /// assert_eq!(doc.metadata.title, Some("My Document".to_string()));
    /// ```
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Search result from a query.
///
/// Contains the matched chunk, relevance score, and match source information.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::{SearchResult, Chunk, EmbeddingIds, MatchSource};
/// use uuid::Uuid;
///
/// let chunk = Chunk {
///     id: Uuid::new_v4(),
///     text: "Matched content".to_string(),
///     index: 0,
///     start_char: 0,
///     end_char: 15,
///     token_count: Some(2),
///     section: None,
///     page: None,
///     embedding_ids: EmbeddingIds::default(),
/// };
///
/// let result = SearchResult {
///     score: 0.95,
///     document_id: Uuid::new_v4(),
///     chunk,
///     match_source: MatchSource::Hybrid,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Relevance score (higher is more relevant)
    pub score: f32,
    /// Document ID containing the matched chunk
    pub document_id: Uuid,
    /// The matched chunk
    pub chunk: Chunk,
    /// Source of the match (dense, sparse, hybrid, raptor)
    pub match_source: MatchSource,
}

/// Source of a search match.
///
/// Indicates which retrieval method produced the match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchSource {
    /// Dense vector match (semantic similarity)
    Dense,
    /// Sparse (BM25) match (keyword matching)
    Sparse,
    /// Hybrid search (combined dense + sparse)
    Hybrid,
    /// RAPTOR tree match (hierarchical retrieval)
    Raptor,
}

/// Configuration for retrieval operations.
///
/// Controls the behavior of search operations including result count,
/// scoring thresholds, and retrieval method weights.
///
/// # Example
///
/// ```rust
/// use reasonkit_mem::RetrievalConfig;
///
/// // Default configuration (favors semantic search)
/// let config = RetrievalConfig::default();
/// assert_eq!(config.top_k, 10);
/// assert_eq!(config.alpha, 0.7);
///
/// // Custom configuration for keyword-heavy search
/// let keyword_config = RetrievalConfig {
///     top_k: 20,
///     alpha: 0.3,  // More weight on BM25
///     rerank: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Minimum score threshold (results below this are filtered)
    pub min_score: f32,
    /// Alpha for hybrid search weight.
    /// - `0.0` = sparse (BM25) only
    /// - `1.0` = dense (vector) only
    /// - `0.7` (default) = 70% dense, 30% sparse
    pub alpha: f32,
    /// Whether to use RAPTOR tree for hierarchical retrieval
    pub use_raptor: bool,
    /// Whether to apply cross-encoder reranking
    pub rerank: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: 0.0,
            alpha: 0.7, // Favor semantic search
            use_raptor: false,
            rerank: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let source = Source {
            source_type: SourceType::Arxiv,
            url: Some("https://arxiv.org/abs/2401.18059".to_string()),
            path: None,
            arxiv_id: Some("2401.18059".to_string()),
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc = Document::new(DocumentType::Paper, source)
            .with_content("This is a test paper about RAPTOR.".to_string());

        assert_eq!(doc.doc_type, DocumentType::Paper);
        assert_eq!(doc.content.word_count, 7);
        assert!(doc.content.raw.contains("RAPTOR"));
    }

    #[test]
    fn test_retrieval_config_default() {
        let config = RetrievalConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.alpha, 0.7);
        assert!(!config.use_raptor);
    }
}
