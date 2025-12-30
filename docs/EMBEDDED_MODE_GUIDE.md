# Qdrant Embedded Mode Guide

> **Version:** 1.0  
> **Last Updated:** 2025-12-29  
> **Status:** Complete

---

## Overview

ReasonKit-Mem supports **embedded mode** for Qdrant, which allows you to use Qdrant locally without requiring external cloud infrastructure. Embedded mode provides automatic fallback to file-based storage when Qdrant is not available, making it ideal for development and local-first deployments.

## What is Embedded Mode?

**Embedded mode** in ReasonKit-Mem means:

1. **Local Qdrant Server**: Qdrant runs on `localhost` (typically port 6333) without external dependencies
2. **Automatic Fallback**: If Qdrant is not available, the system automatically falls back to file-based storage
3. **Zero Configuration**: Works out of the box with sensible defaults
4. **Development-Friendly**: No need to set up external services for local development

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              create_embedded_storage()                   │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐              ┌───────────────┐
│ require_qdrant│              │ require_qdrant│
│    = true     │              │    = false     │
└───────┬───────┘              └───────┬───────┘
        │                               │
        ▼                               ▼
┌───────────────┐              ┌───────────────┐
│ Check Qdrant  │              │ File Storage  │
│   Health      │              │   (Direct)     │
└───────┬───────┘              └───────────────┘
        │
        ├─── Available? ──YES──► QdrantStorage
        │
        └─── NO ──► Error (if require_qdrant=true)
                    OR FileStorage (if require_qdrant=false)
```

## Usage

### Basic Usage (File Storage Fallback)

```rust
use reasonkit_mem::storage::{Storage, EmbeddedStorageConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Default config uses file storage (require_qdrant=false)
    let config = EmbeddedStorageConfig::default();
    let storage = Storage::new_embedded(config).await?;
    
    // Use storage...
    Ok(())
}
```

### File-Only Mode (Explicit)

```rust
use reasonkit_mem::storage::{Storage, EmbeddedStorageConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let data_path = PathBuf::from("./data");
    let config = EmbeddedStorageConfig::file_only(data_path);
    let storage = Storage::new_embedded(config).await?;
    
    // Guaranteed to use file storage
    Ok(())
}
```

### Qdrant Mode (Requires Running Server)

```rust
use reasonkit_mem::storage::{Storage, EmbeddedStorageConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Requires Qdrant server running on localhost:6333
    let config = EmbeddedStorageConfig::with_qdrant(
        "http://localhost:6333",
        "my_collection",
        1536, // vector dimension
    );
    let storage = Storage::new_embedded(config).await?;
    
    // Uses Qdrant if available, fails if not
    Ok(())
}
```

## Configuration Options

### `EmbeddedStorageConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_path` | `PathBuf` | `~/.local/share/reasonkit/storage` | Path for file-based storage |
| `collection_name` | `String` | `"reasonkit_default"` | Qdrant collection name |
| `vector_size` | `usize` | `1536` | Vector dimension (OpenAI ada-002 default) |
| `require_qdrant` | `bool` | `false` | Whether Qdrant is required (vs optional) |
| `qdrant_url` | `String` | `"http://localhost:6333"` | Qdrant server URL |

### Constructor Methods

#### `EmbeddedStorageConfig::default()`
Creates a default config with file storage fallback.

#### `EmbeddedStorageConfig::file_only(data_path: PathBuf)`
Creates a config that explicitly uses file storage only.

#### `EmbeddedStorageConfig::with_qdrant(url, collection, vector_size)`
Creates a config that requires Qdrant. Will fail if Qdrant is not available.

## Setting Up Qdrant for Embedded Mode

### Option 1: Docker (Recommended)

```bash
# Run Qdrant in Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Verify it's running
curl http://localhost:6333/readyz
```

### Option 2: Binary

```bash
# Download Qdrant binary
wget https://github.com/qdrant/qdrant/releases/download/v1.10.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Run Qdrant
./qdrant
```

### Option 3: File Storage (No Qdrant Required)

If you don't want to run Qdrant, simply use file-only mode:

```rust
let config = EmbeddedStorageConfig::file_only(PathBuf::from("./data"));
```

## URL Parsing

The embedded mode supports various URL formats:

```rust
// All of these work:
"http://localhost:6333"
"localhost:6333"
"localhost"              // Defaults to port 6333
"127.0.0.1:6334"
"https://qdrant.example.com:6333"
```

## Health Checking

The embedded mode includes automatic health checking:

- **Endpoint**: `/readyz` (Qdrant's health check endpoint)
- **Timeout**: 5 seconds
- **Retry**: Not automatic (fails fast if unavailable)

## Error Handling

### When `require_qdrant=true` and Qdrant is unavailable:

```rust
let config = EmbeddedStorageConfig::with_qdrant(
    "http://localhost:99999", // Non-existent
    "test",
    768,
);
let result = Storage::new_embedded(config).await;
// Returns: Err("Qdrant required but not available at http://localhost:99999: ...")
```

### When `require_qdrant=false`:

```rust
let config = EmbeddedStorageConfig::default();
let storage = Storage::new_embedded(config).await?;
// Always succeeds, uses file storage
```

## Performance Considerations

| Mode | Latency | Throughput | Memory | Use Case |
|------|---------|-------------|--------|----------|
| **Qdrant** | Low | High | Medium | Production, large datasets |
| **File Storage** | Medium | Medium | Low | Development, small datasets |
| **In-Memory** | Very Low | Very High | High | Testing, ephemeral data |

## Best Practices

1. **Development**: Use `EmbeddedStorageConfig::default()` for zero-config local development
2. **Testing**: Use `Storage::in_memory()` for fast, isolated tests
3. **Production**: Use `EmbeddedStorageConfig::with_qdrant()` with a managed Qdrant instance
4. **CI/CD**: Use file storage or in-memory to avoid external dependencies

## Troubleshooting

### "Qdrant required but not available"

**Problem**: `require_qdrant=true` but Qdrant server is not running.

**Solutions**:
1. Start Qdrant server: `docker run -p 6333:6333 qdrant/qdrant`
2. Use file storage: `EmbeddedStorageConfig::file_only(...)`
3. Check URL: Verify `qdrant_url` is correct

### "Failed to create storage directory"

**Problem**: Insufficient permissions or invalid path.

**Solutions**:
1. Check directory permissions
2. Use absolute paths
3. Ensure parent directories exist

### Health Check Timeout

**Problem**: Qdrant is slow to respond.

**Solutions**:
1. Increase timeout in `check_qdrant_health()` (currently 5s)
2. Check Qdrant server logs
3. Verify network connectivity

## Examples

See `reasonkit-mem/src/storage/mod.rs` for comprehensive test examples:

- `test_embedded_config_default()`
- `test_embedded_config_file_only()`
- `test_embedded_config_with_qdrant()`
- `test_embedded_storage_file_fallback()`
- `test_embedded_storage_default_config()`
- `test_embedded_storage_with_qdrant_required_but_unavailable()`

## Related Documentation

- [Storage Module API Reference](../src/storage/mod.rs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [ReasonKit-Mem README](../README.md)

---

**Status**: ✅ Complete - Embedded mode is fully functional with comprehensive tests and documentation.

