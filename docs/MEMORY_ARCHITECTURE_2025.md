# ReasonKit-Mem: 2025 Memory Architecture Guide

> **Based on:** AI Memory Systems 2025 Foundation Research
> **Goal:** Modular, optional memory layer for reasonkit-core
> **Design Principle:** Plug-and-play integration, standalone value

---

## Architecture Overview

ReasonKit-Mem implements the **EvolveLab modular design** (encode→store→retrieve→manage) synthesized from 2025's leading memory research.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        REASONKIT-MEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  reasonkit-core ←──────────────────────────────────────┐                │
│       │                                                  │                │
│       │ (optional integration)                          │                │
│       ▼                                                  │                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      REASONKIT-MEM                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐         │   │
│  │  │ ENCODE  │──►│  STORE  │──►│RETRIEVE │──►│ MANAGE  │         │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘         │   │
│  │       │             │             │             │               │   │
│  │       ▼             ▼             ▼             ▼               │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │              MEMORY LAYER ABSTRACTION                    │   │   │
│  │  │  • Factual Memory (BedRock axioms)                      │   │   │
│  │  │  • Experiential Memory (GigaThink patterns)             │   │   │
│  │  │  • Working Memory (LaserLogic state)                    │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                           │                                     │   │
│  │  ┌────────────────────────┴────────────────────────────────┐   │   │
│  │  │                    STORAGE BACKENDS                      │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │   │
│  │  │  │  Qdrant  │  │ Tantivy  │  │  RAPTOR  │              │   │   │
│  │  │  │ (Vector) │  │  (BM25)  │  │  (Tree)  │              │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘              │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  reasonkit-web ←─────────────────────────────────────────────────────┘   │
│  (web sensing feeds memory)                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules (EvolveLab Design)

### 1. ENCODE Module

Transforms raw input into memory-ready representations.

```rust
// src/encode/mod.rs
pub trait MemoryEncoder {
    /// Encode raw content into storable form
    fn encode(&self, content: &str, context: &EncodingContext) -> MemoryEntry;

    /// Select what to remember (acquisition)
    fn select(&self, content: &str) -> SelectionResult;

    /// Summarize for compression
    fn summarize(&self, content: &str, target_tokens: usize) -> String;
}

pub struct EncodingContext {
    pub source: MemorySource,        // Where content came from
    pub timestamp: DateTime<Utc>,
    pub confidence: f32,             // 0.0-1.0 reliability
    pub provenance: Provenance,      // Full lineage (MemCube-style)
}

/// MemCube-inspired metadata structure
pub struct Provenance {
    pub origin: String,              // Original source URL/path
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub version: u32,
    pub parent_ids: Vec<Uuid>,       // For derived memories
    pub access_count: u64,
    pub last_accessed: DateTime<Utc>,
}
```

### 2. STORE Module

Organizes memory for efficient access across multiple backends.

```rust
// src/store/mod.rs
pub trait MemoryStore {
    /// Store a memory entry
    fn store(&self, entry: MemoryEntry) -> Result<MemoryId>;

    /// Update existing memory
    fn update(&self, id: MemoryId, entry: MemoryEntry) -> Result<()>;

    /// Delete memory
    fn delete(&self, id: MemoryId) -> Result<()>;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Hierarchical memory levels (3D topology from research)
pub enum MemoryLevel {
    /// L1: Raw leaf nodes (1D flat)
    Leaf,
    /// L2: Semantic clusters (2D planar)
    Cluster,
    /// L3: Summary abstractions (3D hierarchical)
    Summary,
}

/// Storage backend implementations
pub enum StorageBackend {
    /// Qdrant embedded for vector similarity
    Qdrant(QdrantStore),
    /// Tantivy for BM25 keyword search
    Tantivy(TantivyStore),
    /// RAPTOR tree for hierarchical retrieval
    Raptor(RaptorStore),
    /// Hybrid combining all three
    Hybrid(HybridStore),
}
```

### 3. RETRIEVE Module

Query memory efficiently with multiple strategies.

```rust
// src/retrieve/mod.rs
pub trait MemoryRetriever {
    /// Semantic similarity search
    fn semantic_search(&self, query: &str, k: usize) -> Vec<MemoryEntry>;

    /// Keyword-based search
    fn keyword_search(&self, query: &str, k: usize) -> Vec<MemoryEntry>;

    /// Hybrid search (RRF fusion)
    fn hybrid_search(&self, query: &str, k: usize) -> Vec<MemoryEntry>;

    /// Hierarchical RAPTOR retrieval
    fn tree_search(&self, query: &str, depth: usize) -> Vec<MemoryEntry>;
}

/// Query expansion strategies (from HyDE paper)
pub struct QueryExpander {
    /// Generate hypothetical document for better retrieval
    pub fn expand_hyde(&self, query: &str) -> Vec<String>;

    /// Multi-query expansion
    pub fn expand_multi(&self, query: &str, n: usize) -> Vec<String>;
}
```

### 4. MANAGE Module

Maintain memory over time (consolidation, forgetting, evolution).

```rust
// src/manage/mod.rs
pub trait MemoryManager {
    /// Consolidate related memories (like sleep consolidation)
    fn consolidate(&self) -> Result<ConsolidationReport>;

    /// Apply forgetting curve to stale memories
    fn apply_forgetting(&self, policy: ForgettingPolicy) -> Result<u64>;

    /// Resolve conflicts between contradicting memories
    fn resolve_conflicts(&self) -> Result<Vec<ConflictResolution>>;

    /// Migrate memory between storage tiers
    fn migrate(&self, id: MemoryId, target: StorageBackend) -> Result<()>;
}

/// Memory lifecycle policies
pub struct ForgettingPolicy {
    /// Time-to-live for unaccessed memories
    pub ttl: Duration,
    /// Minimum access count to preserve
    pub min_access_count: u64,
    /// Relevance decay rate
    pub decay_rate: f32,
    /// Never forget if confidence above threshold
    pub confidence_threshold: f32,
}

/// Consolidation strategies (from MemEvolve)
pub enum ConsolidationStrategy {
    /// Merge similar memories
    Merge,
    /// Abstract to higher level
    Abstract,
    /// Compress without loss
    Compress,
    /// Promote to parameter memory
    Parameterize,
}
```

---

## Memory Types (Human-AI Parallel)

Based on arXiv:2504.15965 taxonomy:

```rust
/// Memory function types mapped to ReasonKit modules
pub enum MemoryFunction {
    /// Factual/Declarative - "What" knowledge
    /// Maps to: BedRock axioms, verified facts
    Factual {
        source: VerifiedSource,
        confidence: f32,
    },

    /// Experiential/Procedural - "How" knowledge
    /// Maps to: GigaThink exploration patterns
    Experiential {
        case_based: Vec<ProblemSolution>,
        strategy_based: Vec<Strategy>,
        skill_based: Vec<Procedure>,
    },

    /// Working - Active reasoning state
    /// Maps to: LaserLogic current context
    Working {
        context: Vec<ContextItem>,
        attention: Vec<AttentionWeight>,
    },
}
```

---

## Integration with ReasonKit Core

### Optional Flag Installation

```bash
# Install reasonkit-core without memory (default)
cargo install reasonkit

# Install with memory support
cargo install reasonkit --features memory

# Or add to Cargo.toml
[dependencies]
reasonkit-core = { version = "1.0", features = ["memory"] }
reasonkit-mem = { version = "0.1", optional = true }
```

### Feature-Gated Integration

```rust
// reasonkit-core/src/lib.rs

#[cfg(feature = "memory")]
use reasonkit_mem::{MemoryStore, MemoryRetriever};

pub struct ReasonKit {
    think_tools: ThinkToolRegistry,

    #[cfg(feature = "memory")]
    memory: Option<reasonkit_mem::MemorySystem>,
}

impl ReasonKit {
    #[cfg(feature = "memory")]
    pub fn with_memory(mut self, config: MemoryConfig) -> Self {
        self.memory = Some(reasonkit_mem::MemorySystem::new(config));
        self
    }

    pub fn execute(&self, input: &str) -> Result<Output> {
        // Retrieve relevant context from memory
        #[cfg(feature = "memory")]
        let context = if let Some(mem) = &self.memory {
            mem.retrieve_context(input)?
        } else {
            vec![]
        };

        // Execute ThinkTool chain
        let output = self.think_tools.execute(input, &context)?;

        // Store experience to memory
        #[cfg(feature = "memory")]
        if let Some(mem) = &self.memory {
            mem.store_experience(input, &output)?;
        }

        Ok(output)
    }
}
```

### Web Integration (reasonkit-web)

```python
# reasonkit-web can feed captures to memory
from reasonkit_mem import MemoryClient

class WebCapture:
    def __init__(self, memory_endpoint: str = None):
        self.memory = MemoryClient(memory_endpoint) if memory_endpoint else None

    def capture(self, url: str) -> CaptureResult:
        result = self._capture_page(url)

        # Store to memory if connected
        if self.memory:
            self.memory.store(
                content=result.content,
                source=url,
                memory_type="factual",
                confidence=0.9
            )

        return result
```

---

## GPU Acceleration Path

When GPU cluster is available:

```rust
// src/store/gpu.rs

#[cfg(feature = "gpu")]
pub struct GpuAcceleratedStore {
    qdrant: QdrantGpuClient,  // v1.13+ with GPU
    device: cuda::Device,
}

#[cfg(feature = "gpu")]
impl MemoryStore for GpuAcceleratedStore {
    fn store(&self, entry: MemoryEntry) -> Result<MemoryId> {
        // GPU-accelerated embedding
        let embedding = self.embed_gpu(&entry.content)?;

        // GPU-accelerated indexing (10x faster)
        self.qdrant.upsert_gpu(embedding, entry.metadata)?;

        Ok(entry.id)
    }
}
```

---

## Configuration

```toml
# reasonkit-mem.toml

[memory]
# Storage backend selection
backend = "hybrid"  # "qdrant" | "tantivy" | "raptor" | "hybrid"

[memory.qdrant]
path = "./data/qdrant"
collection = "reasonkit_memory"
embedding_dim = 768

[memory.tantivy]
path = "./data/tantivy"

[memory.raptor]
path = "./data/raptor"
max_depth = 4
cluster_size = 10

[memory.forgetting]
ttl_days = 90
min_access_count = 3
decay_rate = 0.1
confidence_threshold = 0.95

[memory.consolidation]
enabled = true
interval_hours = 24
strategy = "abstract"

[memory.gpu]
enabled = false
device = 0
```

---

## Benchmark Targets

Based on MemOS research (159% improvement baseline):

| Operation | Target Latency | Target Throughput |
|-----------|----------------|-------------------|
| Store (single) | < 5ms | 1,000/sec |
| Retrieve (k=10) | < 20ms | 500 queries/sec |
| Hybrid search | < 50ms | 200 queries/sec |
| Consolidation | < 1s per 1000 entries | Background |

---

## Roadmap

### v0.1 - Foundation
- [ ] Core encode/store/retrieve/manage traits
- [ ] Qdrant embedded backend
- [ ] Basic RAPTOR integration
- [ ] Feature-gated integration with reasonkit-core

### v0.2 - Intelligence
- [ ] Automatic consolidation scheduler
- [ ] Forgetting policy implementation
- [ ] Conflict resolution
- [ ] Provenance tracking (MemCube-style)

### v0.3 - Performance
- [ ] GPU acceleration support
- [ ] Hybrid search optimization
- [ ] Query expansion (HyDE)
- [ ] Benchmarks vs MemOS

### v1.0 - Production
- [ ] Multi-tenant isolation
- [ ] Horizontal scaling
- [ ] reasonkit-web integration
- [ ] Enterprise features

---

## References

- [AI Memory Systems 2025 Foundation](../rk-research/AI_MEMORY_SYSTEMS_2025_FOUNDATION.md)
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
- [MemOS](https://github.com/MemTensor/MemOS)
- [MemEvolve](https://github.com/bingreeky/MemEvolve)

---

**Document Version:** 1.0.0
**Last Updated:** 2025-12-30
