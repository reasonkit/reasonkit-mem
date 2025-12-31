<div align="center">

<!-- Hero Banner - Memory Infrastructure -->
<img src="https://raw.githubusercontent.com/ReasonKit/reasonkit-core/main/brand/banners/hero-tree.png" alt="ReasonKit Mem" width="100%" style="border-radius: 12px;">

# ReasonKit Mem

**Memory & Retrieval Infrastructure for ReasonKit**

[![Crates.io](https://img.shields.io/crates/v/reasonkit-mem?style=flat-square&color=%2306b6d4)](https://crates.io/crates/reasonkit-mem)
[![docs.rs](https://img.shields.io/docsrs/reasonkit-mem?style=flat-square&color=%2310b981)](https://docs.rs/reasonkit-mem)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square&color=%23a855f7)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.74+-orange?style=flat-square&logo=rust&color=%23f97316)](https://www.rust-lang.org/)

_The Long-Term Memory Layer ("Hippocampus") for AI Reasoning_

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

<div align="center">

```
+---------------------------------------------------------------------+
|                      HYBRID RETRIEVAL                               |
+---------------------------------------------------------------------+
|                                                                     |
|   Query --+---> [Dense Encoder] ---> Qdrant ANN ---> Top-K Dense   |
|           |                                                         |
|           +---> [BM25 Tokenizer] ---> Tantivy ---> Top-K Sparse    |
|                                                                     |
|                       v                                             |
|            +----------------------+                                 |
|            | Reciprocal Rank Fusion|                                |
|            +----------+-----------+                                 |
|                       v                                             |
|            +----------------------+                                 |
|            | Cross-Encoder Rerank |                                 |
|            +----------+-----------+                                 |
|                       v                                             |
|                 Final Results                                       |
+---------------------------------------------------------------------+
```

</div>

## Technology Stack

| Component      | Technology          | Purpose                |
| -------------- | ------------------- | ---------------------- |
| **Qdrant**     | qdrant-client 1.10+ | Dense vector storage   |
| **Tantivy**    | tantivy 0.22+       | BM25 sparse search     |
| **RAPTOR**     | Custom Rust         | Hierarchical retrieval |
| **Embeddings** | BGE-M3 / OpenAI     | Dense representations  |
| **Reranking**  | Cross-encoder       | Final precision boost  |

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

## License

Apache License 2.0 - see [LICENSE](LICENSE)

---

<div align="center">

**Part of the ReasonKit Ecosystem**

[ReasonKit Core](https://github.com/ReasonKit/reasonkit-core) | [ReasonKit Web](https://github.com/ReasonKit/reasonkit-web) | [Website](https://reasonkit.sh)

_"See How Your AI Thinks"_

</div>
