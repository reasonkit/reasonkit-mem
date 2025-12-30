# RAPTOR Tree Implementation Guide

> **Version:** 1.0  
> **Last Updated:** 2025-12-29  
> **Status:** ✅ Complete

---

## Overview

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) implements hierarchical clustering and summarization for improved retrieval quality, especially for long-form question answering.

## What is RAPTOR?

RAPTOR builds a tree structure where:
- **Leaf nodes** = Original document chunks
- **Internal nodes** = Summaries of child clusters
- **Root nodes** = Highest-level abstractions

This enables multi-level retrieval: search at abstract levels first, then drill down to specific chunks.

## Architecture

```
Level 2 (Roots)
    ├─ Summary: "AI and Machine Learning Overview"
    │   └─ Level 1
    │       ├─ Summary: "Neural Networks"
    │       │   └─ Level 0 (Leaves)
    │       │       ├─ Chunk: "Neural networks are..."
    │       │       ├─ Chunk: "Backpropagation works by..."
    │       │       └─ Chunk: "CNNs use convolutional layers..."
    │       └─ Summary: "Deep Learning"
    │           └─ Level 0 (Leaves)
    │               ├─ Chunk: "Deep learning involves..."
    │               └─ Chunk: "Transformers use attention..."
    └─ Summary: "Natural Language Processing"
        └─ Level 1
            └─ Level 0 (Leaves)
                └─ Chunk: "NLP techniques include..."
```

## Components

### RaptorTree

Main structure for hierarchical retrieval:

```rust
use reasonkit_mem::raptor::RaptorTree;

let mut tree = RaptorTree::new(
    max_depth: 3,      // Maximum tree depth
    cluster_size: 5,   // Nodes per cluster
);
```

### Building a RAPTOR Tree

```rust
use reasonkit_mem::{Chunk, raptor::RaptorTree};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Prepare chunks
    let chunks: Vec<Chunk> = /* ... */;
    
    // Create tree
    let mut tree = RaptorTree::new(3, 5);
    
    // Build tree with embedder and summarizer
    tree.build_from_chunks(
        &chunks,
        &|text| {
            // Embedding function
            Ok(generate_embedding(text)?)
        },
        &|text| {
            // Summarization function
            Ok(summarize_text(text)?)
        },
    ).await?;
    
    Ok(())
}
```

## Usage Examples

### Basic RAPTOR Tree

```rust
use reasonkit_mem::{
    Chunk, Document, raptor::RaptorTree,
    storage::Storage,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create tree
    let mut tree = RaptorTree::new(3, 5);
    
    // Build from document chunks
    let chunks = vec![/* your chunks */];
    tree.build_from_chunks(
        &chunks,
        &embed_text,      // Embedding function
        &summarize_text,  // Summarization function
    ).await?;
    
    // Search the tree
    let query_embedding = embed_text("What is machine learning?")?;
    let results = tree.search(&query_embedding, 10)?;
    
    for (node_id, score) in results {
        if let Some(node) = tree.get_node(&node_id) {
            println!("Score: {:.3}, Text: {}", score, node.text);
        }
    }
    
    Ok(())
}
```

### Integration with Hybrid Search

```rust
use reasonkit_mem::retrieval::HybridRetriever;

let retriever = HybridRetriever::new(storage, index, config)
    .with_raptor_tree(Some(tree));

// RAPTOR-aware hybrid search
let results = retriever.search_with_raptor("query", 10).await?;
```

### Tree Statistics

```rust
let stats = tree.stats();
println!("Total nodes: {}", stats.total_nodes);
println!("Leaf nodes: {}", stats.leaf_nodes);
println!("Max depth: {}", stats.max_depth);
println!("Root count: {}", stats.root_count);
```

## Tree Building Process

1. **Level 0 (Leaves)**: Create nodes from original chunks
2. **Level 1+ (Internal)**: 
   - Cluster nodes from previous level
   - Summarize each cluster
   - Create parent nodes with summaries
   - Link children to parents
3. **Roots**: Remaining nodes at highest level

## Configuration

### max_depth

Maximum number of hierarchical levels (excluding leaves).

- **1**: Single summarization level
- **2-3**: Recommended for most use cases
- **4+**: Very deep hierarchies (may lose information)

### cluster_size

Number of nodes to cluster together at each level.

- **3-5**: Small clusters, more granular
- **5-10**: Recommended default
- **10+**: Large clusters, more abstract

## Search Algorithm

RAPTOR search uses a multi-level approach:

1. **Search all levels**: Compare query embedding with all node embeddings
2. **Top-K selection**: Select top candidates across all levels
3. **Expand to leaves**: For each candidate, expand to leaf descendants
4. **Deduplicate**: Remove duplicate leaf nodes
5. **Return top-K**: Final ranked leaf nodes

## Optimized RAPTOR

An optimized version is available with:

- **Beam search**: Efficient tree traversal
- **Embedding caching**: Reuse computed embeddings
- **Parallel clustering**: Faster tree building
- **Early termination**: Prune low-similarity paths

```rust
use reasonkit_mem::raptor::optimized::{OptimizedRaptorTree, RaptorOptConfig};

let config = RaptorOptConfig {
    max_depth: 3,
    cluster_size: 5,
    beam_width: 3,
    cache_size: 1000,
    parallel_clustering: true,
    min_similarity: 0.1,
};

let mut tree = OptimizedRaptorTree::new(config);
tree.build_from_chunks(&chunks, &embed_text, &summarize_text).await?;
```

## Performance Considerations

| Configuration | Build Time | Search Time | Memory | Use Case |
|---------------|------------|-------------|--------|----------|
| **Basic (depth=2, size=5)** | Fast | Fast | Low | Small datasets |
| **Standard (depth=3, size=5)** | Medium | Medium | Medium | General purpose |
| **Deep (depth=4, size=10)** | Slow | Slow | High | Large datasets |
| **Optimized** | Medium | Fast | Medium | Production use |

## Best Practices

1. **Depth**: Start with 2-3 levels, increase only if needed
2. **Cluster Size**: 5-10 nodes per cluster is optimal
3. **Summarization**: Use high-quality summarization for better abstractions
4. **Embeddings**: Consistent embedding model across all levels
5. **Testing**: Verify tree structure with `tree.stats()`

## Integration Examples

### With Storage

```rust
// Store RAPTOR tree in storage
storage.store_raptor_tree(&tree, &context).await?;

// Load RAPTOR tree from storage
let tree = storage.load_raptor_tree(collection_id, &context).await?;
```

### With Hybrid Search

```rust
let retriever = HybridRetriever::new(storage, index, config)
    .with_raptor_tree(Some(tree))
    .with_embedding_pipeline(Some(pipeline));

// Multi-level hybrid search
let results = retriever.search_with_raptor("query", 10).await?;
```

## Testing

Comprehensive tests are available in:
- `reasonkit-mem/src/raptor/mod.rs` - Basic RAPTOR tests
- `reasonkit-mem/src/raptor/optimized.rs` - Optimized RAPTOR tests

Run tests:
```bash
cargo test --lib raptor
```

## References

- **RAPTOR Paper**: Sarthi et al. (2024) - "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
- **Hierarchical Clustering**: Standard clustering algorithms (K-means, hierarchical)
- **Tree Search**: Beam search, best-first search

---

**Status**: ✅ Complete - RAPTOR tree structure is fully implemented with basic and optimized versions. Comprehensive tests and tree building/search functionality included.

