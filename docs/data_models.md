# ReasonKit Memory Data Models

> **Version:** 0.1.0

## Core Concepts

ReasonKit Memory uses a hierarchical data model optimized for RAG (Retrieval-Augmented Generation) and long-term agent memory.

### 1. MemoryUnit (The Atom)

The fundamental unit of storage.

```rust
struct MemoryUnit {
    id: Uuid,
    content: String,
    metadata: HashMap<String, Value>,
    embedding: Vec<f32>,
    timestamp: DateTime<Utc>,
    source_uri: Option<String>,
}
```

### 2. Episodic Memory

Stores sequences of events or interactions.

- **Structure:** Time-ordered list of `MemoryUnit`s.
- **Use Case:** Chat history, activity logs.
- **Indexing:** Chronological + Semantic.

### 3. Semantic Memory

Stores facts, concepts, and generalized knowledge.

- **Structure:** Graph-based or clustered vector space.
- **Use Case:** "What is the capital of France?", "User prefers dark mode".
- **Indexing:** RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).

## RAPTOR Tree Structure

For large knowledge bases, we use a RAPTOR tree:

- **Leaf Nodes:** Original chunks of text (`MemoryUnit`).
- **Parent Nodes:** Summaries of child nodes.
- **Root Node:** High-level summary of the entire cluster/document.

Retrieval traverses this tree to find the right level of abstraction for a query.

## Vector Schema

- **Dimensions:** 1536 (default, compatible with OpenAI text-embedding-3-small) or 768 (local models).
- **Metric:** Cosine Similarity.
- **Engine:** Qdrant / pgvector (pluggable).
