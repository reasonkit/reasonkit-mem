# ReasonKit Mem

> Memory & Retrieval Infrastructure for ReasonKit

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

```rust
use reasonkit_mem::storage::Storage;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create embedded storage (automatic file storage fallback)
    let storage = Storage::new_embedded().await?;

    // Use storage...
    Ok(())
}
```

### Advanced Usage (Custom Configuration)

```rust
use reasonkit_mem::{
    storage::{Storage, EmbeddedStorageConfig},
    embedding::EmbeddingProvider,
    retrieval::HybridRetriever,
    Document, RetrievalConfig,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create storage with custom config
    let config = EmbeddedStorageConfig::file_only(PathBuf::from("./data"));
    let storage = Storage::new_embedded_with_config(config).await?;

    // Or use Qdrant (requires running server)
    let qdrant_config = EmbeddedStorageConfig::with_qdrant(
        "http://localhost:6333",
        "my_collection",
        1536,
    );
    let storage = Storage::new_embedded_with_config(qdrant_config).await?;

    // Index documents
    storage.store_document(&doc, &context).await?;

    // Hybrid search
    let retriever = HybridRetriever::new(storage.clone());
    let results = retriever.search("query", &RetrievalConfig::default()).await?;

    Ok(())
}
```

### Embedded Mode Documentation

For detailed information about embedded mode, see [docs/EMBEDDED_MODE_GUIDE.md](docs/EMBEDDED_MODE_GUIDE.md).

## Architecture

```
Query → [Dense Encoder] → Qdrant ANN Search → Top-K Dense
      → [BM25 Tokenizer] → Tantivy Search → Top-K Sparse
                               ↓
                    [Reciprocal Rank Fusion]
                               ↓
                    [Cross-Encoder Rerank]
                               ↓
                         Final Results
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Qdrant** | qdrant-client 1.10+ | Dense vector storage |
| **Tantivy** | tantivy 0.22+ | BM25 sparse search |
| **RAPTOR** | Custom Rust | Hierarchical retrieval |
| **Embeddings** | BGE-M3 / OpenAI | Dense representations |
| **Reranking** | Cross-encoder | Final precision boost |

## Project Structure

```
reasonkit-mem/
├── src/
│   ├── storage/      # Qdrant vector + file-based storage
│   ├── embedding/    # Dense vector embeddings
│   ├── retrieval/    # Hybrid search, fusion, reranking
│   ├── raptor/       # RAPTOR hierarchical tree structure
│   ├── indexing/     # BM25/Tantivy sparse indexing
│   └── rag/          # RAG pipeline orchestration
└── Cargo.toml
```

## License

Apache License 2.0 - see [LICENSE](LICENSE)

## Links

- [ReasonKit Core](https://github.com/reasonkit/reasonkit-core) - The reasoning engine
- [ReasonKit Web](https://github.com/reasonkit/reasonkit-web) - Web sensing layer
- [Website](https://reasonkit.sh) - Official website
