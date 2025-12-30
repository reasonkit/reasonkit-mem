# Changelog

All notable changes to reasonkit-mem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-29

### Added

#### Initial Release - Memory Infrastructure Extraction

- **Storage Module**
  - Qdrant vector database integration with embedded mode
  - File-based storage fallback
  - Access control and context management
  - Document storage and retrieval
  - Vector operations (upsert, search, delete)

- **Embedding Module**
  - BGE-M3 local inference support (ONNX)
  - OpenAI embedding provider
  - Embedding caching with TTL
  - Batch processing support
  - Multiple embedding providers

- **Retrieval Module**
  - Hybrid search (dense + sparse)
  - Reciprocal Rank Fusion (RRF)
  - Cross-encoder reranking
  - Query expansion
  - Retrieval statistics and metrics

- **RAPTOR Module**
  - Hierarchical retrieval trees
  - Optimized tree construction
  - Multi-level document clustering
  - Long-form QA support

- **Indexing Module**
  - Tantivy BM25 sparse indexing
  - Document indexing and search
  - Index optimization
  - Full-text search capabilities

- **RAG Pipeline**
  - Simple RAG pipeline orchestration
  - Document processing
  - Chunk management

### Changed

- Extracted from reasonkit-core as standalone crate
- Clean separation of memory infrastructure from reasoning engine
- Independent versioning and deployment

### Technical Details

- **Rust Edition:** 2021
- **Minimum Rust Version:** 1.74
- **License:** Apache-2.0
- **Dependencies:** All published to crates.io

### Migration Notes

This crate was extracted from `reasonkit-core` as part of a clean architectural separation. The migration was completed on 2025-12-29 with:

- 19 files + 5 directories migrated
- 319 tests passing (54 in this crate)
- All quality gates passing
- Zero breaking changes (backward compatible via feature flags)

### Documentation

- Complete API documentation
- README with usage examples
- Migration documentation in parent project

---

## [Unreleased]

### Planned Features

- Performance optimizations
- Additional embedding providers
- Enhanced RAPTOR algorithms
- More retrieval strategies

---

[0.1.0]: https://github.com/reasonkit/reasonkit-mem/releases/tag/v0.1.0

