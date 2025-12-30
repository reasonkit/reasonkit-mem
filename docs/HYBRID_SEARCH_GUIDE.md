# Hybrid Search Implementation Guide

> **Version:** 1.0  
> **Last Updated:** 2025-12-29  
> **Status:** ✅ Complete

---

## Overview

ReasonKit-Mem implements **hybrid search** combining dense (vector) and sparse (BM25) retrieval with multiple fusion strategies. This provides the best of both worlds: semantic understanding from embeddings and keyword precision from BM25.

## Architecture

```
Query
  │
  ├─→ [Dense Encoder] ──→ Qdrant Vector Search ──→ Top-K Dense Results
  │
  └─→ [BM25 Tokenizer] ──→ Tantivy Search ──→ Top-K Sparse Results
                          │
                          ↓
              [Reciprocal Rank Fusion (RRF)]
                          │
                          ↓
              [Cross-Encoder Reranking] (optional)
                          │
                          ↓
                   Final Hybrid Results
```

## Components

### 1. HybridRetriever

The main interface for hybrid search operations.

```rust
use reasonkit_mem::{
    storage::Storage,
    retrieval::{HybridRetriever, RetrievalConfig},
    indexing::IndexManager,
};

let retriever = HybridRetriever::new(
    storage,
    index_manager,
    RetrievalConfig::default(),
);
```

### 2. FusionEngine

Combines results from multiple retrieval methods using various strategies:

- **RRF (Reciprocal Rank Fusion)** - Recommended default
- **Weighted Sum** - Score-based fusion with normalization
- **Rank-Biased Fusion (RBF)** - Decay-based fusion

### 3. RetrievalConfig

Configuration for hybrid search behavior:

```rust
pub struct RetrievalConfig {
    pub top_k: usize,              // Number of results to return
    pub alpha: f32,                 // Dense/sparse balance (0.0 = BM25 only, 1.0 = vector only)
    pub fusion_strategy: FusionStrategy,
    pub rerank: bool,               // Enable cross-encoder reranking
    pub rerank_top_k: usize,        // Rerank top N candidates
}
```

## Usage Examples

### Basic Hybrid Search

```rust
use reasonkit_mem::{
    storage::Storage,
    retrieval::{HybridRetriever, RetrievalConfig},
    indexing::IndexManager,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize storage and index
    let storage = Storage::new_embedded().await?;
    let index = IndexManager::new("./index").await?;
    
    // Create retriever
    let retriever = HybridRetriever::new(
        storage,
        index,
        RetrievalConfig::default(),
    );
    
    // Perform hybrid search
    let results = retriever.search_hybrid(
        "What is machine learning?",
        None, // Will generate embedding automatically
        &RetrievalConfig {
            top_k: 10,
            alpha: 0.7, // 70% dense, 30% sparse
            ..Default::default()
        },
    ).await?;
    
    for result in results {
        println!("Score: {:.3}, Text: {}", result.score, result.text);
    }
    
    Ok(())
}
```

### Dense-Only Search

```rust
// Pure vector search (alpha = 1.0)
let results = retriever.search_dense("query", 10).await?;
```

### Sparse-Only Search

```rust
// Pure BM25 search (alpha = 0.0)
let results = retriever.search_sparse("query", 10).await?;
```

### Custom Fusion Strategy

```rust
use reasonkit_mem::retrieval::{FusionEngine, FusionStrategy};

// Use Rank-Biased Fusion
let fusion_engine = FusionEngine::new(FusionStrategy::RankBiasedFusion {
    rho: 0.8, // Persistence parameter
});

// Or use Weighted Sum
let fusion_engine = FusionEngine::weighted(0.7); // 70% dense weight
```

## Fusion Strategies

### Reciprocal Rank Fusion (RRF) - Default

**Formula:** `score(d) = Σ 1 / (k + rank(d))`

- **k**: Constant (typically 60)
- **Advantages**: Rank-based, doesn't require score normalization
- **Use Case**: General purpose, recommended default

```rust
let config = RetrievalConfig {
    fusion_strategy: FusionStrategy::ReciprocalRankFusion { k: 60 },
    ..Default::default()
};
```

### Weighted Sum

**Formula:** `score(d) = α × dense_score(d) + (1-α) × sparse_score(d)`

- **α**: Dense weight (0.0-1.0)
- **Advantages**: Simple, interpretable weights
- **Use Case**: When you know the relative importance of dense vs sparse

```rust
let config = RetrievalConfig {
    fusion_strategy: FusionStrategy::WeightedSum { dense_weight: 0.7 },
    ..Default::default()
};
```

### Rank-Biased Fusion (RBF)

**Formula:** `score(d) = Σ ρ^rank(d)`

- **ρ**: Persistence parameter (0.0-1.0, typically 0.8)
- **Advantages**: Decay-based, emphasizes top ranks
- **Use Case**: When rank position is more important than absolute scores

```rust
let config = RetrievalConfig {
    fusion_strategy: FusionStrategy::RankBiasedFusion { rho: 0.8 },
    ..Default::default()
};
```

## Alpha Parameter

The `alpha` parameter controls the balance between dense and sparse retrieval:

| Alpha | Dense | Sparse | Use Case |
|-------|-------|--------|----------|
| `0.0` | 0% | 100% | Pure keyword search (BM25 only) |
| `0.3` | 30% | 70% | Keyword-heavy queries |
| `0.5` | 50% | 50% | Balanced (default) |
| `0.7` | 70% | 30% | Semantic-heavy queries |
| `1.0` | 100% | 0% | Pure semantic search (vector only) |

## Reranking

Optional cross-encoder reranking improves precision by re-scoring top candidates:

```rust
let config = RetrievalConfig {
    top_k: 20,           // Retrieve 20 candidates
    rerank: true,        // Enable reranking
    rerank_top_k: 10,    // Rerank top 10, return top 10
    ..Default::default()
};
```

## Performance Considerations

| Method | Latency | Precision | Recall | Use Case |
|--------|---------|-----------|--------|----------|
| **Dense Only** | Low | Medium | High | Semantic queries |
| **Sparse Only** | Very Low | High | Medium | Keyword queries |
| **Hybrid (RRF)** | Medium | High | High | General purpose |
| **Hybrid + Rerank** | High | Very High | High | Maximum precision |

## Best Practices

1. **Default Configuration**: Use `RetrievalConfig::default()` for most cases (alpha=0.7, RRF)
2. **Alpha Tuning**: Start with 0.7, adjust based on query type:
   - Keyword-heavy: Lower alpha (0.3-0.5)
   - Semantic-heavy: Higher alpha (0.7-0.9)
3. **Reranking**: Enable for final results when precision is critical
4. **Top-K**: Retrieve more candidates (2-3x) than needed if using reranking

## Integration with RAPTOR

Hybrid search can be combined with RAPTOR trees for hierarchical retrieval:

```rust
let retriever = HybridRetriever::new(storage, index, config)
    .with_raptor_tree(raptor_tree)
    .with_embedding_pipeline(embedding_pipeline);

// RAPTOR-aware hybrid search
let results = retriever.search_with_raptor("query", 10).await?;
```

## Testing

Comprehensive tests are available in:
- `reasonkit-mem/src/retrieval/fusion.rs` - Fusion algorithm tests
- `reasonkit-mem/src/retrieval/mod.rs` - Integration tests

Run tests:
```bash
cargo test --lib retrieval
```

## References

- **RRF**: Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet"
- **RBF**: Moffat & Zobel (2008) - "Rank-Biased Precision"
- **Cross-Encoder**: Nogueira et al. (2020) - arXiv:2010.06467

---

**Status**: ✅ Complete - Hybrid search is fully implemented with RRF, Weighted Sum, and RBF fusion strategies. Comprehensive tests and documentation included.

