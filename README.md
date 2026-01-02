<div align="center">

# ReasonKit Mem

**Memory & Retrieval Infrastructure for ReasonKit**

[![Crates.io](https://img.shields.io/crates/v/reasonkit-mem?style=flat-square&color=%2306b6d4)](https://crates.io/crates/reasonkit-mem)
[![docs.rs](https://img.shields.io/docsrs/reasonkit-mem?style=flat-square&color=%2310b981)](https://docs.rs/reasonkit-mem)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square&color=%23a855f7)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.74+-orange?style=flat-square&logo=rust&color=%23f97316)](https://www.rust-lang.org/)

*The Long-Term Memory Layer ("Hippocampus") for AI Reasoning*

[Documentation](https://docs.rs/reasonkit-mem) | [ReasonKit Core](https://github.com/ReasonKit/reasonkit-core) | [Website](https://reasonkit.sh)

</div>

---

**ReasonKit Mem** is the memory layer ("Hippocampus") for ReasonKit. It provides vector storage, hybrid search, RAPTOR trees, and embedding support.

## Features

- **Vector Storage** - Qdrant-based dense vector storage with embedded mode
- **Hybrid Search** - Dense (Qdrant) + Sparse (Tantivy BM25) fusion
- **RAPTOR Trees** - Hierarchical retrieval for long-form QA
- **Embeddings** - Local (BGE-M3) and remote (OpenAI) embedding support
- **Reranking** - Cross-encoder reranking for precision

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
reasonkit-mem = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Usage

### Basic Usage (Embedded Mode)

```rust,ignore
use reasonkit_mem::storage::Storage;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create embedded storage (automatic file storage fallback)
    let storage = Storage::new_embedded().await?;

    // Use storage...
    Ok(())
}
```

### Storage with Custom Configuration

```rust,ignore
use reasonkit_mem::storage::{Storage, EmbeddedStorageConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create storage with custom file path
    let config = EmbeddedStorageConfig::file_only(PathBuf::from("./data"));
    let storage = Storage::new_embedded_with_config(config).await?;

    // Or use Qdrant (requires running server)
    let qdrant_config = EmbeddedStorageConfig::with_qdrant(
        "http://localhost:6333",
        "my_collection",
        1536,
    );
    let qdrant_storage = Storage::new_embedded_with_config(qdrant_config).await?;

    Ok(())
}
```

### Hybrid Search with KnowledgeBase

```rust,ignore
use reasonkit_mem::retrieval::KnowledgeBase;
use reasonkit_mem::{Document, DocumentType, Source, SourceType};
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create in-memory knowledge base
    let kb = KnowledgeBase::in_memory()?;

    // Create a document
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some("notes.md".to_string()),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: None,
    };

    let doc = Document::new(DocumentType::Note, source)
        .with_content("Machine learning is a subset of artificial intelligence.".to_string());

    // Add document to knowledge base
    kb.add(&doc).await?;

    // Search using sparse retrieval (BM25)
    let results = kb.retriever().search_sparse("machine learning", 5).await?;

    for result in results {
        println!("Score: {:.3}, Text: {}", result.score, result.text);
    }

    Ok(())
}
```

### Using Embeddings

```rust,ignore
use reasonkit_mem::embedding::{EmbeddingConfig, EmbeddingPipeline, OpenAIEmbedding};
use reasonkit_mem::retrieval::KnowledgeBase;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create OpenAI embedding provider (requires OPENAI_API_KEY env var)
    let embedding_provider = OpenAIEmbedding::openai()?;
    let pipeline = Arc::new(EmbeddingPipeline::new(Arc::new(embedding_provider)));

    // Create knowledge base with embedding support
    let kb = KnowledgeBase::in_memory()?
        .with_embedding_pipeline(pipeline);

    // Now hybrid search will use both dense (vector) and sparse (BM25)
    // let results = kb.query("semantic search query", 10).await?;

    Ok(())
}
```

### Embedded Mode Documentation

For detailed information about embedded mode, see [docs/EMBEDDED_MODE_GUIDE.md](docs/EMBEDDED_MODE_GUIDE.md).

## Architecture

![ReasonKit Mem Hybrid Architecture](https://reasonkit.sh/assets/brand/mem/hybrid_architecture.png)
![ReasonKit Mem Hybrid Architecture Technical Diagram](https://reasonkit.sh/assets/brand/mem/hybrid_retrieval_engine.svg)

### The RAPTOR Algorithm (Hierarchical Indexing)

ReasonKit Mem implements **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) to answer high-level questions across large document sets.

![ReasonKit Mem RAPTOR Tree Structure](https://reasonkit.sh/assets/brand/mem/raptor_tree_structure.svg)
![ReasonKit Mem RAPTOR Tree](https://reasonkit.sh/assets/brand/mem/raptor_tree.png)

### The Memory Dashboard

![ReasonKit Mem Dashboard](https://reasonkit.sh/assets/brand/mem/memory_dashboard.png)

### Integration Ecosystem

![ReasonKit Mem Ecosystem](https://reasonkit.sh/assets/brand/mem/mem_ecosystem.png)

## Technology Stack

| Component      | Technology          | Purpose                |
| -------------- | ------------------- | ---------------------- |
| **Qdrant**     | qdrant-client 1.10+ | Dense vector storage   |
| **Tantivy**    | tantivy 0.22+       | BM25 sparse search     |
| **RAPTOR**     | Custom Rust         | Hierarchical retrieval |
| **Embeddings** | BGE-M3 / OpenAI     | Dense representations  |
| **Reranking**  | Cross-encoder       | Final precision boost  |

## Project Structure

```text
reasonkit-mem/
├── src/
│   ├── storage/      # Qdrant vector + file-based storage
│   ├── embedding/    # Dense vector embeddings
│   ├── retrieval/    # Hybrid search, fusion, reranking
│   ├── raptor/       # RAPTOR hierarchical tree structure
│   ├── indexing/     # BM25/Tantivy sparse indexing
│   └── rag/          # RAG pipeline orchestration
├── benches/          # Performance benchmarks
├── examples/         # Usage examples
├── docs/             # Additional documentation
└── Cargo.toml
```

## Feature Flags

| Feature            | Description                              |
| ------------------ | ---------------------------------------- |
| `default`          | Core functionality                       |
| `python`           | Python bindings via PyO3                 |
| `local-embeddings` | Local BGE-M3 embeddings via ONNX Runtime |

## API Reference

### Core Types (re-exported at crate root)

```rust,ignore
use reasonkit_mem::{
    // Documents
    Document, DocumentType, DocumentContent,
    // Chunks
    Chunk, EmbeddingIds,
    // Sources
    Source, SourceType,
    // Metadata
    Metadata, Author,
    // Search
    SearchResult, MatchSource, RetrievalConfig,
    // Processing
    ProcessingStatus, ProcessingState, ContentFormat,
    // Errors
    MemError, MemResult,
};
```

### Storage Module

```rust,ignore
use reasonkit_mem::storage::{
    Storage,
    EmbeddedStorageConfig,
    StorageBackend,
    InMemoryStorage,
    FileStorage,
    QdrantStorage,
    AccessContext,
    AccessLevel,
};
```

### Embedding Module

```rust,ignore
use reasonkit_mem::embedding::{
    EmbeddingProvider,      // Trait for embedding backends
    OpenAIEmbedding,        // OpenAI API embeddings
    EmbeddingConfig,        // Configuration
    EmbeddingPipeline,      // Batch processing pipeline
    EmbeddingResult,        // Single embedding result
    EmbeddingVector,        // Vec<f32> alias
    cosine_similarity,      // Utility function
    normalize_vector,       // Utility function
};
```

### Retrieval Module

```rust,ignore
use reasonkit_mem::retrieval::{
    HybridRetriever,        // Main retrieval engine
    KnowledgeBase,          // High-level API
    HybridResult,           // Search result
    RetrievalStats,         // Statistics
    // Fusion
    FusionEngine,
    FusionStrategy,
    // Reranking
    Reranker,
    RerankerConfig,
};
```

## Version & Maturity

| Component | Status | Notes |
|-----------|--------|-------|
| **Vector Storage** | ✅ Stable | Qdrant integration production-ready |
| **Hybrid Search** | ✅ Stable | Dense + Sparse fusion working |
| **RAPTOR Trees** | ✅ Stable | Hierarchical retrieval implemented |
| **Embeddings** | ✅ Stable | OpenAI API fully supported |
| **Local Embeddings** | 🔶 Beta | BGE-M3 ONNX (enable with `local-embeddings` feature) |
| **Python Bindings** | 🔶 Beta | Build from source with `--features python` |

**Current Version:** v0.1.2 | [CHANGELOG](CHANGELOG.md) | [Releases](https://github.com/reasonkit/reasonkit-mem/releases)

### Verify Installation

```rust
use reasonkit_mem::storage::Storage;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Quick verification - creates in-memory storage
    let storage = Storage::new_embedded().await?;
    println!("ReasonKit Mem initialized successfully!");
    Ok(())
}
```

## License

Apache License 2.0 - see [LICENSE](https://github.com/reasonkit/reasonkit-mem/blob/main/LICENSE)

---

<div align="center">

![ReasonKit Ecosystem Connection](https://reasonkit.sh/assets/brand/mem/ecosystem_connection.png)

**Part of the ReasonKit Ecosystem**

[ReasonKit Core](https://github.com/reasonkit/reasonkit-core) | [ReasonKit Web](https://github.com/reasonkit/reasonkit-web) | [Website](https://reasonkit.sh)

*"See How Your AI Thinks"*

</div>
