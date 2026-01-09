# Persistence Strategies

> **Version:** 0.1.0

ReasonKit Memory supports a dual-layer persistence strategy: Hot (Fast) and Cold (Archive).

## 1. Hot Storage (Vector Database)

Designed for sub-millisecond retrieval during active reasoning.

- **Technology:** Qdrant (primary), or pgvector (PostgreSQL).
- **Data:** Embeddings, metadata, recent ephemeral context.
- **Retention:** Configurable (e.g., last 30 days or active working set).

## 2. Cold Storage (Object/Relational)

Designed for durability, audit trails, and full reconstruction.

- **Technology:** SQLite (local), PostgreSQL (server), or S3-compatible Blob Storage.
- **Data:** Full raw text, original documents, complete conversation logs, snapshots of the vector state.
- **Format:** Parquet (for analytics) or JSONL (for portability).

## Sync Strategy

1.  **Write Path:**
    - Agent writes to `MemoryInterface`.
    - System writes to **Cold Storage** (WAL/Log) immediately for durability.
    - System asynchronously computes embeddings and updates **Hot Storage**.

2.  **Read Path:**
    - Query hits **Hot Storage** (Vector Index).
    - If payload is missing/truncated in Hot, fetch full content from **Cold Storage** using ID.

## Backup & Recovery

- **Snapshotting:** Qdrant snapshots are taken daily.
- **PITR:** PostgreSQL Point-in-Time Recovery is enabled for the Cold layer.
- **Export:** `reasonkit-mem export --format jsonl` allows dumping the entire memory state for migration.
