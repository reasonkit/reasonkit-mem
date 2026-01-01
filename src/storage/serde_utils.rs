//! Efficient serialization/deserialization utilities for memory entries with embeddings.
//!
//! This module provides high-performance binary serialization optimized for:
//! - Dense embedding vectors (f32 arrays) stored as raw little-endian bytes
//! - Memory entries using MessagePack for compact structured data
//! - Optional LZ4 compression for large entries
//! - WAL entries with CRC32 checksums for integrity verification
//!
//! # Performance Characteristics
//!
//! - Embedding serialization: O(n) where n = vector dimension
//! - MessagePack: ~10x smaller than JSON for typical entries
//! - LZ4 compression: ~300MB/s compression, ~1GB/s decompression
//!
//! # Example
//!
//! ```rust,ignore
//! use reasonkit_mem::storage::serde_utils::{
//!     serialize_embedding, deserialize_embedding,
//!     serialize_entry, deserialize_entry,
//! };
//!
//! // Efficient embedding serialization (raw bytes, no overhead)
//! let embedding = vec![0.1, 0.2, 0.3, 0.4];
//! let bytes = serialize_embedding(&embedding);
//! let recovered = deserialize_embedding(&bytes)?;
//! assert_eq!(embedding, recovered);
//!
//! // Memory entry serialization (MessagePack)
//! let entry = MemoryEntry::new(...);
//! let bytes = serialize_entry(&entry)?;
//! let recovered: MemoryEntry = deserialize_entry(&bytes)?;
//! ```

use crate::{MemError, MemResult};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Magic bytes for identifying serialized entry format
#[allow(dead_code)]
const MAGIC_BYTES: [u8; 4] = [b'R', b'K', b'M', b'E']; // ReasonKit Memory Entry

/// Current serialization format version
const FORMAT_VERSION: u8 = 1;

/// Header size in bytes (magic + version + flags)
#[allow(dead_code)]
const HEADER_SIZE: usize = 6;

/// Flag indicating compression is enabled
const FLAG_COMPRESSED: u8 = 0x01;

/// Flag indicating checksum is present
const FLAG_CHECKSUM: u8 = 0x02;

// ============================================================================
// MEMORY ENTRY TYPES
// ============================================================================

/// A memory entry containing text content and its embedding vector.
///
/// This is the primary data structure for storage, optimized for
/// efficient serialization with separate handling of embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: Uuid,
    /// Text content
    pub content: String,
    /// Embedding vector (stored separately for efficiency)
    #[serde(skip)]
    pub embedding: Vec<f32>,
    /// Metadata as key-value pairs
    pub metadata: std::collections::HashMap<String, String>,
    /// Creation timestamp (Unix epoch seconds)
    pub created_at: i64,
    /// Last update timestamp
    pub updated_at: Option<i64>,
    /// Source document ID
    pub document_id: Option<Uuid>,
    /// Chunk index within document
    pub chunk_index: Option<usize>,
    /// Additional tags
    pub tags: Vec<String>,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(content: String, embedding: Vec<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            metadata: std::collections::HashMap::new(),
            created_at: chrono::Utc::now().timestamp(),
            updated_at: None,
            document_id: None,
            chunk_index: None,
            tags: Vec::new(),
        }
    }

    /// Create with specific ID
    pub fn with_id(id: Uuid, content: String, embedding: Vec<f32>) -> Self {
        let mut entry = Self::new(content, embedding);
        entry.id = id;
        entry
    }
}

/// WAL (Write-Ahead Log) entry for durability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Sequence number
    pub sequence: u64,
    /// Operation type
    pub operation: WalOperation,
    /// Entry data
    pub data: WalEntryData,
    /// Timestamp (Unix epoch nanoseconds for precision)
    pub timestamp_ns: u64,
}

/// WAL operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalOperation {
    /// Insert new entry
    Insert,
    /// Update existing entry
    Update,
    /// Delete entry
    Delete,
    /// Checkpoint marker
    Checkpoint,
}

/// WAL entry data variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntryData {
    /// Memory entry data (for Insert/Update)
    Entry(MemoryEntryCompact),
    /// Entry ID only (for Delete)
    EntryId(Uuid),
    /// Checkpoint data
    Checkpoint { last_sequence: u64 },
}

/// Compact version of MemoryEntry for WAL storage
/// Embedding is stored separately as raw bytes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntryCompact {
    /// Entry ID
    pub id: Uuid,
    /// Content
    pub content: String,
    /// Embedding as raw bytes (little-endian f32)
    #[serde(with = "serde_bytes")]
    pub embedding_bytes: Vec<u8>,
    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Created timestamp
    pub created_at: i64,
    /// Document ID
    pub document_id: Option<Uuid>,
    /// Chunk index
    pub chunk_index: Option<usize>,
    /// Tags
    pub tags: Vec<String>,
}

impl From<&MemoryEntry> for MemoryEntryCompact {
    fn from(entry: &MemoryEntry) -> Self {
        Self {
            id: entry.id,
            content: entry.content.clone(),
            embedding_bytes: serialize_embedding(&entry.embedding),
            metadata: entry.metadata.clone(),
            created_at: entry.created_at,
            document_id: entry.document_id,
            chunk_index: entry.chunk_index,
            tags: entry.tags.clone(),
        }
    }
}

impl MemoryEntryCompact {
    /// Convert to full MemoryEntry
    pub fn to_memory_entry(&self) -> MemResult<MemoryEntry> {
        let embedding = deserialize_embedding(&self.embedding_bytes)?;
        Ok(MemoryEntry {
            id: self.id,
            content: self.content.clone(),
            embedding,
            metadata: self.metadata.clone(),
            created_at: self.created_at,
            updated_at: None,
            document_id: self.document_id,
            chunk_index: self.chunk_index,
            tags: self.tags.clone(),
        })
    }
}

// ============================================================================
// EMBEDDING SERIALIZATION
// ============================================================================

/// Serialize embedding vector to raw little-endian bytes.
///
/// This is the most efficient representation for f32 vectors:
/// - Zero overhead (no format markers, length prefixes per element)
/// - Direct memory layout compatible with SIMD operations
/// - 4 bytes per f32 element
///
/// # Arguments
/// * `embedding` - Slice of f32 values representing the embedding vector
///
/// # Returns
/// Raw bytes in little-endian format
///
/// # Example
/// ```rust,ignore
/// let embedding = vec![1.0, 2.0, 3.0];
/// let bytes = serialize_embedding(&embedding);
/// assert_eq!(bytes.len(), 12); // 3 * 4 bytes
/// ```
#[inline]
pub fn serialize_embedding(embedding: &[f32]) -> Vec<u8> {
    // Pre-allocate exact size needed
    let mut bytes = Vec::with_capacity(embedding.len() * 4);

    // Use direct byte conversion for maximum efficiency
    for &value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    bytes
}

/// Deserialize embedding vector from raw little-endian bytes.
///
/// # Arguments
/// * `bytes` - Raw bytes in little-endian format
///
/// # Returns
/// * `Ok(Vec<f32>)` - Reconstructed embedding vector
/// * `Err` - If byte count is not divisible by 4
///
/// # Example
/// ```rust,ignore
/// let bytes = serialize_embedding(&vec![1.0, 2.0, 3.0]);
/// let embedding = deserialize_embedding(&bytes)?;
/// assert_eq!(embedding, vec![1.0, 2.0, 3.0]);
/// ```
#[inline]
pub fn deserialize_embedding(bytes: &[u8]) -> MemResult<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(MemError::Serialization(format!(
            "Invalid embedding byte length: {} (must be divisible by 4)",
            bytes.len()
        )));
    }

    let capacity = bytes.len() / 4;
    let mut embedding = Vec::with_capacity(capacity);

    // Use chunks for efficient iteration
    for chunk in bytes.chunks_exact(4) {
        // SAFETY: chunks_exact guarantees exactly 4 bytes
        let arr: [u8; 4] = chunk.try_into().unwrap();
        embedding.push(f32::from_le_bytes(arr));
    }

    Ok(embedding)
}

/// Serialize embedding with dimension prefix.
///
/// Includes a 4-byte little-endian u32 dimension count prefix.
/// Useful when dimension needs to be validated on deserialization.
#[inline]
pub fn serialize_embedding_with_dim(embedding: &[f32]) -> Vec<u8> {
    let dim = embedding.len() as u32;
    let mut bytes = Vec::with_capacity(4 + embedding.len() * 4);
    bytes.extend_from_slice(&dim.to_le_bytes());
    bytes.extend(serialize_embedding(embedding));
    bytes
}

/// Deserialize embedding with dimension validation.
///
/// Reads the dimension prefix and validates it matches expected.
#[inline]
pub fn deserialize_embedding_with_dim(
    bytes: &[u8],
    expected_dim: Option<usize>,
) -> MemResult<Vec<f32>> {
    if bytes.len() < 4 {
        return Err(MemError::Serialization(
            "Invalid embedding bytes: too short for dimension prefix".to_string(),
        ));
    }

    let dim_bytes: [u8; 4] = bytes[..4].try_into().unwrap();
    let dim = u32::from_le_bytes(dim_bytes) as usize;

    if let Some(expected) = expected_dim {
        if dim != expected {
            return Err(MemError::Serialization(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                expected, dim
            )));
        }
    }

    let expected_len = 4 + dim * 4;
    if bytes.len() != expected_len {
        return Err(MemError::Serialization(format!(
            "Invalid embedding byte length: expected {}, got {}",
            expected_len,
            bytes.len()
        )));
    }

    deserialize_embedding(&bytes[4..])
}

// ============================================================================
// MEMORY ENTRY SERIALIZATION (MessagePack)
// ============================================================================

/// Serialize a memory entry to compact MessagePack bytes.
///
/// The entry is serialized as:
/// 1. Header (6 bytes): magic + version + flags
/// 2. Metadata MessagePack (variable)
/// 3. Embedding raw bytes (4 * dim)
///
/// This separates the structured metadata from the embedding for efficiency.
///
/// # Arguments
/// * `entry` - Memory entry to serialize
///
/// # Returns
/// Compact binary representation
pub fn serialize_entry(entry: &MemoryEntry) -> MemResult<Vec<u8>> {
    // Create compact representation with embedded bytes
    let compact = MemoryEntryCompact::from(entry);

    // Serialize with MessagePack
    serialize_msgpack(&compact)
}

/// Deserialize bytes to a memory entry.
///
/// # Arguments
/// * `bytes` - Serialized entry bytes
///
/// # Returns
/// Reconstructed memory entry
pub fn deserialize_entry(bytes: &[u8]) -> MemResult<MemoryEntry> {
    let compact: MemoryEntryCompact = deserialize_msgpack(bytes)?;
    compact.to_memory_entry()
}

/// Serialize any serde-compatible value to MessagePack.
pub fn serialize_msgpack<T: Serialize>(value: &T) -> MemResult<Vec<u8>> {
    rmp_serde::to_vec(value)
        .map_err(|e| MemError::Serialization(format!("MessagePack serialization failed: {}", e)))
}

/// Deserialize any serde-compatible value from MessagePack.
pub fn deserialize_msgpack<T: DeserializeOwned>(bytes: &[u8]) -> MemResult<T> {
    rmp_serde::from_slice(bytes)
        .map_err(|e| MemError::Serialization(format!("MessagePack deserialization failed: {}", e)))
}

// ============================================================================
// WAL ENTRY SERIALIZATION (with checksum)
// ============================================================================

/// Serialize a WAL entry with CRC32 checksum for integrity.
///
/// Format:
/// - 4 bytes: Magic bytes "RKWE" (ReasonKit WAL Entry)
/// - 1 byte: Format version
/// - 1 byte: Flags
/// - 4 bytes: CRC32 checksum of payload
/// - N bytes: MessagePack payload
///
/// # Arguments
/// * `entry` - WAL entry to serialize
///
/// # Returns
/// Serialized bytes with integrity checksum
pub fn serialize_wal_entry(entry: &WalEntry) -> MemResult<Vec<u8>> {
    const WAL_MAGIC: [u8; 4] = [b'R', b'K', b'W', b'E'];

    // Serialize payload first
    let payload = serialize_msgpack(entry)?;

    // Calculate CRC32 checksum
    let checksum = crc32_checksum(&payload);

    // Build final buffer: magic + version + flags + checksum + payload
    let mut result = Vec::with_capacity(10 + payload.len());
    result.extend_from_slice(&WAL_MAGIC);
    result.push(FORMAT_VERSION);
    result.push(FLAG_CHECKSUM);
    result.extend_from_slice(&checksum.to_le_bytes());
    result.extend(payload);

    Ok(result)
}

/// Deserialize WAL entry with checksum verification.
///
/// # Arguments
/// * `bytes` - Serialized WAL entry bytes
///
/// # Returns
/// * `Ok(WalEntry)` - Verified and deserialized entry
/// * `Err` - If checksum fails or format is invalid
pub fn deserialize_wal_entry(bytes: &[u8]) -> MemResult<WalEntry> {
    const WAL_MAGIC: [u8; 4] = [b'R', b'K', b'W', b'E'];
    const MIN_SIZE: usize = 10; // magic(4) + version(1) + flags(1) + checksum(4)

    if bytes.len() < MIN_SIZE {
        return Err(MemError::Serialization(format!(
            "WAL entry too short: {} bytes (minimum {})",
            bytes.len(),
            MIN_SIZE
        )));
    }

    // Verify magic bytes
    if bytes[..4] != WAL_MAGIC {
        return Err(MemError::Serialization(
            "Invalid WAL entry: wrong magic bytes".to_string(),
        ));
    }

    // Check version
    let version = bytes[4];
    if version != FORMAT_VERSION {
        return Err(MemError::Serialization(format!(
            "Unsupported WAL format version: {} (expected {})",
            version, FORMAT_VERSION
        )));
    }

    // Check flags
    let flags = bytes[5];
    let has_checksum = (flags & FLAG_CHECKSUM) != 0;

    // Extract and verify checksum
    let stored_checksum = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    let payload = &bytes[10..];

    if has_checksum {
        let computed_checksum = crc32_checksum(payload);
        if stored_checksum != computed_checksum {
            return Err(MemError::Serialization(format!(
                "WAL entry checksum mismatch: stored {:08x}, computed {:08x}",
                stored_checksum, computed_checksum
            )));
        }
    }

    deserialize_msgpack(payload)
}

// ============================================================================
// COMPRESSION (Optional LZ4)
// ============================================================================

/// Compress data using LZ4.
///
/// LZ4 provides excellent decompression speed (~1GB/s) with reasonable
/// compression ratios. Ideal for embedding data where read performance
/// is critical.
///
/// # Arguments
/// * `data` - Uncompressed data
///
/// # Returns
/// LZ4-compressed data with size prefix
#[cfg(feature = "compression")]
pub fn compress(data: &[u8]) -> MemResult<Vec<u8>> {
    use lz4_flex::compress_prepend_size;
    Ok(compress_prepend_size(data))
}

/// Decompress LZ4-compressed data.
///
/// # Arguments
/// * `data` - LZ4-compressed data with size prefix
///
/// # Returns
/// Decompressed original data
#[cfg(feature = "compression")]
pub fn decompress(data: &[u8]) -> MemResult<Vec<u8>> {
    use lz4_flex::decompress_size_prepended;
    decompress_size_prepended(data)
        .map_err(|e| MemError::Serialization(format!("LZ4 decompression failed: {}", e)))
}

/// Stub implementations when compression feature is disabled
#[cfg(not(feature = "compression"))]
pub fn compress(data: &[u8]) -> MemResult<Vec<u8>> {
    // Without compression feature, just return data with a header indicating no compression
    let mut result = Vec::with_capacity(1 + data.len());
    result.push(0x00); // No compression marker
    result.extend_from_slice(data);
    Ok(result)
}

#[cfg(not(feature = "compression"))]
pub fn decompress(data: &[u8]) -> MemResult<Vec<u8>> {
    if data.is_empty() {
        return Err(MemError::Serialization(
            "Empty data cannot be decompressed".to_string(),
        ));
    }
    // First byte is compression marker
    if data[0] != 0x00 {
        return Err(MemError::Serialization(
            "Compressed data requires 'compression' feature".to_string(),
        ));
    }
    Ok(data[1..].to_vec())
}

/// Serialize entry with optional compression.
///
/// Automatically compresses if data exceeds threshold.
pub fn serialize_entry_compressed(
    entry: &MemoryEntry,
    compression_threshold: usize,
) -> MemResult<Vec<u8>> {
    let raw = serialize_entry(entry)?;

    if raw.len() >= compression_threshold {
        let compressed = compress(&raw)?;
        // Only use compressed if it's actually smaller
        if compressed.len() < raw.len() {
            let mut result = Vec::with_capacity(1 + compressed.len());
            result.push(FLAG_COMPRESSED);
            result.extend(compressed);
            return Ok(result);
        }
    }

    // Not compressed
    let mut result = Vec::with_capacity(1 + raw.len());
    result.push(0x00);
    result.extend(raw);
    Ok(result)
}

/// Deserialize entry with automatic decompression detection.
pub fn deserialize_entry_compressed(bytes: &[u8]) -> MemResult<MemoryEntry> {
    if bytes.is_empty() {
        return Err(MemError::Serialization("Empty data".to_string()));
    }

    let is_compressed = bytes[0] == FLAG_COMPRESSED;
    let payload = &bytes[1..];

    let raw = if is_compressed {
        decompress(payload)?
    } else {
        payload.to_vec()
    };

    deserialize_entry(&raw)
}

// ============================================================================
// BATCH SERIALIZATION
// ============================================================================

/// Serialize multiple entries efficiently.
///
/// Uses a compact format optimized for batch operations:
/// - 4 bytes: Entry count (u32 LE)
/// - For each entry: 4 bytes length + serialized entry
pub fn serialize_batch(entries: &[MemoryEntry]) -> MemResult<Vec<u8>> {
    let count = entries.len() as u32;

    // Estimate size (rough approximation)
    let estimated_size = 4 + entries.len() * 1024;
    let mut buffer = Vec::with_capacity(estimated_size);

    // Write count
    buffer.extend_from_slice(&count.to_le_bytes());

    // Write each entry with length prefix
    for entry in entries {
        let serialized = serialize_entry(entry)?;
        let len = serialized.len() as u32;
        buffer.extend_from_slice(&len.to_le_bytes());
        buffer.extend(serialized);
    }

    Ok(buffer)
}

/// Deserialize batch of entries.
pub fn deserialize_batch(bytes: &[u8]) -> MemResult<Vec<MemoryEntry>> {
    if bytes.len() < 4 {
        return Err(MemError::Serialization("Batch data too short".to_string()));
    }

    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut entries = Vec::with_capacity(count);
    let mut offset = 4;

    for _ in 0..count {
        if offset + 4 > bytes.len() {
            return Err(MemError::Serialization("Batch data truncated".to_string()));
        }

        let len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + len > bytes.len() {
            return Err(MemError::Serialization("Batch entry truncated".to_string()));
        }

        let entry = deserialize_entry(&bytes[offset..offset + len])?;
        entries.push(entry);
        offset += len;
    }

    Ok(entries)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate CRC32 checksum using the IEEE polynomial.
///
/// Uses a simple implementation for portability. For high-throughput
/// scenarios, consider using hardware-accelerated crc32c.
fn crc32_checksum(data: &[u8]) -> u32 {
    // CRC32 IEEE polynomial lookup table
    const CRC32_TABLE: [u32; 256] = generate_crc32_table();

    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Generate CRC32 lookup table at compile time.
const fn generate_crc32_table() -> [u32; 256] {
    const POLYNOMIAL: u32 = 0xEDB88320;
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Estimate serialized size for capacity planning.
pub fn estimate_entry_size(entry: &MemoryEntry) -> usize {
    // Base overhead (header, UUIDs, timestamps)
    let base = 64;
    // Content
    let content = entry.content.len();
    // Embedding (4 bytes per f32)
    let embedding = entry.embedding.len() * 4;
    // Metadata (rough estimate)
    let metadata: usize = entry
        .metadata
        .iter()
        .map(|(k, v)| k.len() + v.len() + 8)
        .sum();
    // Tags
    let tags: usize = entry.tags.iter().map(|t| t.len() + 4).sum();

    base + content + embedding + metadata + tags
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_embedding_roundtrip() {
        let original: Vec<f32> = vec![1.0, -2.5, 3.14159, 0.0, f32::MAX, f32::MIN];
        let bytes = serialize_embedding(&original);
        let recovered = deserialize_embedding(&bytes).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_serialize_embedding_empty() {
        let original: Vec<f32> = vec![];
        let bytes = serialize_embedding(&original);
        assert!(bytes.is_empty());
        let recovered = deserialize_embedding(&bytes).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_deserialize_embedding_invalid_length() {
        let bytes = vec![1, 2, 3]; // Not divisible by 4
        let result = deserialize_embedding(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_embedding_with_dim() {
        let original = vec![1.0, 2.0, 3.0];
        let bytes = serialize_embedding_with_dim(&original);
        assert_eq!(bytes.len(), 4 + 12); // 4 dim + 12 data

        let recovered = deserialize_embedding_with_dim(&bytes, Some(3)).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_deserialize_embedding_dim_mismatch() {
        let original = vec![1.0, 2.0, 3.0];
        let bytes = serialize_embedding_with_dim(&original);
        let result = deserialize_embedding_with_dim(&bytes, Some(4));
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_entry_roundtrip() {
        let mut entry = MemoryEntry::new(
            "Test content for memory entry".to_string(),
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
        );
        entry
            .metadata
            .insert("key".to_string(), "value".to_string());
        entry.tags.push("test".to_string());

        let bytes = serialize_entry(&entry).unwrap();
        let recovered = deserialize_entry(&bytes).unwrap();

        assert_eq!(entry.id, recovered.id);
        assert_eq!(entry.content, recovered.content);
        assert_eq!(entry.embedding, recovered.embedding);
        assert_eq!(entry.metadata, recovered.metadata);
        assert_eq!(entry.tags, recovered.tags);
    }

    #[test]
    fn test_wal_entry_roundtrip() {
        let entry = WalEntry {
            sequence: 42,
            operation: WalOperation::Insert,
            data: WalEntryData::Entry(MemoryEntryCompact {
                id: Uuid::new_v4(),
                content: "WAL test".to_string(),
                embedding_bytes: serialize_embedding(&vec![1.0, 2.0]),
                metadata: std::collections::HashMap::new(),
                created_at: 1234567890,
                document_id: None,
                chunk_index: Some(0),
                tags: vec![],
            }),
            timestamp_ns: 1234567890123456789,
        };

        let bytes = serialize_wal_entry(&entry).unwrap();
        let recovered = deserialize_wal_entry(&bytes).unwrap();

        assert_eq!(entry.sequence, recovered.sequence);
        assert_eq!(entry.operation, recovered.operation);
        assert_eq!(entry.timestamp_ns, recovered.timestamp_ns);
    }

    #[test]
    fn test_wal_entry_checksum_corruption() {
        let entry = WalEntry {
            sequence: 1,
            operation: WalOperation::Delete,
            data: WalEntryData::EntryId(Uuid::new_v4()),
            timestamp_ns: 0,
        };

        let mut bytes = serialize_wal_entry(&entry).unwrap();
        // Corrupt a byte in the payload
        if let Some(last) = bytes.last_mut() {
            *last ^= 0xFF;
        }

        let result = deserialize_wal_entry(&bytes);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_wal_entry_invalid_magic() {
        let bytes = vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
        let result = deserialize_wal_entry(&bytes);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("wrong magic bytes"));
    }

    #[test]
    fn test_batch_serialization() {
        let entries: Vec<MemoryEntry> = (0..5)
            .map(|i| MemoryEntry::new(format!("Entry {}", i), vec![i as f32; 4]))
            .collect();

        let bytes = serialize_batch(&entries).unwrap();
        let recovered = deserialize_batch(&bytes).unwrap();

        assert_eq!(entries.len(), recovered.len());
        for (orig, rec) in entries.iter().zip(recovered.iter()) {
            assert_eq!(orig.content, rec.content);
            assert_eq!(orig.embedding, rec.embedding);
        }
    }

    #[test]
    fn test_compression_roundtrip() {
        let entry = MemoryEntry::new(
            "A".repeat(10000), // Large content to benefit from compression
            vec![0.1; 1000],
        );

        let bytes = serialize_entry_compressed(&entry, 1000).unwrap();
        let recovered = deserialize_entry_compressed(&bytes).unwrap();

        assert_eq!(entry.content, recovered.content);
        assert_eq!(entry.embedding, recovered.embedding);
    }

    #[test]
    fn test_estimate_entry_size() {
        let entry = MemoryEntry::new("Hello".to_string(), vec![1.0; 384]);
        let estimated = estimate_entry_size(&entry);
        let actual = serialize_entry(&entry).unwrap().len();

        // Estimate should be in reasonable range (not exact due to MessagePack encoding)
        assert!(estimated > actual / 2);
        assert!(estimated < actual * 3);
    }

    #[test]
    fn test_crc32_known_values() {
        // Test against known CRC32 values
        let checksum = crc32_checksum(b"123456789");
        // CRC32 of "123456789" with IEEE polynomial
        assert_eq!(checksum, 0xCBF43926);
    }

    #[test]
    fn test_large_embedding() {
        // Test with realistic embedding size (OpenAI ada-002 = 1536 dims)
        let large_embedding: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        let bytes = serialize_embedding(&large_embedding);
        assert_eq!(bytes.len(), 1536 * 4);

        let recovered = deserialize_embedding(&bytes).unwrap();
        assert_eq!(large_embedding, recovered);
    }

    #[test]
    fn test_special_float_values() {
        let special = vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.0,
            -0.0,
            f32::EPSILON,
        ];
        let bytes = serialize_embedding(&special);
        let recovered = deserialize_embedding(&bytes).unwrap();

        assert!(recovered[0].is_infinite() && recovered[0].is_sign_positive());
        assert!(recovered[1].is_infinite() && recovered[1].is_sign_negative());
        assert!(recovered[2].is_nan());
        assert_eq!(recovered[3], 0.0);
        assert_eq!(recovered[4], -0.0);
        assert_eq!(recovered[5], f32::EPSILON);
    }
}
