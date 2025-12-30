# REASONKIT-MEM PROJECT CONTEXT

> Memory & Retrieval Infrastructure for ReasonKit ("Hippocampus")
> "The Long-Term Memory Layer"

**LICENSE:** Apache 2.0
**STATUS:** Stable (Migration Complete - 2025-12-29)
**REPOSITORY:** https://github.com/ReasonKit/reasonkit-mem

---

## WHAT REASONKIT-MEM IS

**ReasonKit-Mem is the memory and retrieval infrastructure layer for ReasonKit.**

It provides:
- **Vector Storage**: Qdrant-based dense vector storage with embedded mode
- **Hybrid Search**: Dense (Qdrant) + Sparse (Tantivy BM25) fusion
- **RAPTOR Trees**: Hierarchical retrieval for long-form QA
- **Embeddings**: Local (BGE-M3) and remote (OpenAI) embedding support
- **Reranking**: Cross-encoder reranking for precision

### Architecture

```
reasonkit-mem/
├── src/
│   ├── storage/      # Qdrant vector + file-based storage
│   ├── embedding/    # Dense vector embeddings (BGE-M3, OpenAI)
│   ├── retrieval/    # Hybrid search, fusion, reranking
│   ├── raptor/       # RAPTOR hierarchical tree structure
│   ├── indexing/     # BM25/Tantivy sparse indexing
│   ├── rag/          # RAG pipeline orchestration
│   ├── types.rs      # Shared types (Document, Chunk, etc.)
│   └── error.rs      # MemError types
└── Cargo.toml
```

---

## SEPARATION FROM REASONKIT-CORE

**reasonkit-core** = PURE REASONING ENGINE (ThinkTools, protocols, evaluation)
**reasonkit-mem** = MEMORY INFRASTRUCTURE (storage, retrieval, embeddings, RAPTOR)

This separation allows:
- Independent scaling of memory vs reasoning
- Clear architectural boundaries
- Potential separate release/versioning

---

## TECHNOLOGY STACK

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Qdrant** | qdrant-client 1.10+ | Dense vector storage |
| **Tantivy** | tantivy 0.22+ | BM25 sparse search |
| **RAPTOR** | Custom Rust | Hierarchical retrieval |
| **Embeddings** | BGE-M3 / OpenAI | Dense representations |
| **Reranking** | Cross-encoder | Final precision boost |

---

## HYBRID SEARCH ARCHITECTURE

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

---

## USAGE

```rust
use reasonkit_mem::{
    storage::{Storage, EmbeddedStorageConfig},
    embedding::EmbeddingProvider,
    retrieval::HybridRetriever,
    Document, RetrievalConfig,
};

// Create storage backend
let config = EmbeddedStorageConfig::default();
let storage = Storage::new_embedded(config).await?;

// Index documents
storage.store_document(&doc).await?;

// Hybrid search
let retriever = HybridRetriever::new(storage.clone());
let results = retriever.search("query", &RetrievalConfig::default()).await?;
```

---

## QUALITY STATUS

| Gate | Status | Notes |
|------|--------|-------|
| **Build** | ✅ PASS | `cargo build --release` |
| **Clippy** | ✅ PASS | No warnings |
| **Tests** | ✅ PASS | 54 tests passing |
| **Format** | ✅ PASS | `cargo fmt --check` |

---

## TASK MANAGEMENT (MANDATORY - CONS-007)

> **Axiom:** No work exists without task tracking. ALL AI agents MUST use the full task system.

### Taskwarrior Integration

**ALL AI agents MUST use Taskwarrior for task tracking.**

```bash
# Create task (MANDATORY format for RK-PROJECT)
task add project:rk-project.mem.{component} "{description}" priority:{H|M|L} due:{date} +{tags}

# Examples:
task add project:rk-project.mem.storage "Optimize Qdrant embedded mode" priority:H due:today +rust +performance
task add project:rk-project.mem.retrieval "Implement RRF fusion" priority:M due:friday +search
task add project:rk-project.mem.raptor "Build RAPTOR tree structure" priority:M due:tomorrow +rag

# Start working (CRITICAL: Auto-starts timewarrior!)
task {id} start

# Stop working (pauses time tracking)
task {id} stop

# Complete task (stops timewarrior, records completion)
task {id} done

# Add annotations (progress notes, decisions, blockers)
task {id} annotate "Completed hybrid search implementation, 20% improvement in recall"
task {id} annotate "BLOCKED: Waiting for Qdrant 1.10 release"
task {id} annotate "DECISION: Using Tantivy for BM25 over alternatives"

# View status
task project:rk-project.mem list
task project:rk-project.mem summary
timew summary :week
```

**Components:**
- `mem.storage` → Qdrant vector storage
- `mem.retrieval` → Hybrid search, fusion, reranking
- `mem.raptor` → RAPTOR hierarchical trees
- `mem.embedding` → Embedding services (BGE-M3, OpenAI)
- `mem.indexing` → BM25/Tantivy sparse indexing

**Full Documentation:** See `ORCHESTRATOR.md` for complete Taskwarrior reference.

---

## MCP SERVERS, SKILLS & PLUGINS (MAXIMIZE)

### MCP Server Usage

**Agents MUST leverage MCP servers for all compatible operations.**

```yaml
MCP_SERVERS_PRIORITY:
  - sequential-thinking   # ALWAYS use for complex reasoning chains
  - filesystem            # File operations
  - github               # Repository operations
  - memory               # Persistent memory
  - puppeteer            # Web automation
  - fetch                # HTTP requests with caching

USAGE_PATTERN:
  1. Check if MCP server exists for operation
  2. If yes: USE IT (preferred over direct implementation)
  3. If no: Implement in Rust, consider creating MCP server
```

### Skills & Plugins

```yaml
SKILLS_MAXIMIZATION:
  - Use pdf skill for PDF operations
  - Use xlsx skill for spreadsheet operations
  - Use docx skill for document operations
  - Use frontend-design skill for UI work
  - Use mcp-builder skill for MCP server creation

PLUGIN_PRIORITY:
  - api-contract-sync for API validation
  - math for deterministic calculations
  - experienced-engineer agents for specialized tasks
```

### Extensions

```yaml
BROWSER_EXTENSIONS:
  - Use when web research needed
  - Prefer official provider extensions

IDE_EXTENSIONS:
  - Cursor: .cursorrules enforcement
  - VS Code: copilot-instructions.md
  - Windsurf: .windsurfrules
```

**Full Reference:** See [ORCHESTRATOR.md](../../ORCHESTRATOR.md#mcp-servers-skills--plugins-maximize) for complete MCP/Skills/Plugins documentation.

---

## CONSTRAINTS

| Constraint | Details |
|------------|---------|
| Rust-only | All core code in Rust |
| Performance | All hot paths optimized |
| No ThinkTools | ThinkTools stay in reasonkit-core |
| API Stability | Breaking changes require version bump |

---

*reasonkit-mem v0.1.0 | Memory Infrastructure | Apache 2.0*
*Migration Complete: 2025-12-29 | Ready for crates.io publication*
