//! Benchmarks for dual-layer memory system
//!
//! Performance targets:
//! - Hot memory write: < 50us
//! - Hot memory read: < 10us
//! - Cold memory write: < 1ms
//! - Cold memory read: < 500us
//! - WAL append: < 100us
//! - WAL sync: < 5ms
//! - Recovery: < 1s per 10k entries
//! - Hybrid search: < 10ms for 10k entries

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use uuid::Uuid;

// ============================================================================
// Mock Dual-Layer Memory Types for Benchmarking
// ============================================================================

/// Entry stored in memory layers
#[derive(Clone, Debug)]
struct MemoryEntry {
    id: Uuid,
    embedding: Vec<f32>,
    text: String,
    timestamp: u64,
    checksum: u32,
}

impl MemoryEntry {
    fn new(embedding: Vec<f32>, text: String) -> Self {
        let id = Uuid::new_v4();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let checksum = crc32fast::hash(text.as_bytes());
        Self {
            id,
            embedding,
            text,
            timestamp,
            checksum,
        }
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<Uuid>()
            + self.embedding.len() * std::mem::size_of::<f32>()
            + self.text.len()
            + std::mem::size_of::<u64>()
            + std::mem::size_of::<u32>()
    }
}

/// Hot memory layer (in-memory cache with LRU eviction)
struct HotMemory {
    entries: RwLock<HashMap<Uuid, MemoryEntry>>,
    access_order: RwLock<Vec<Uuid>>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl HotMemory {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_entries)),
            access_order: RwLock::new(Vec::with_capacity(max_entries)),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    fn write(&self, entry: MemoryEntry) -> Result<(), String> {
        let mut entries = self.entries.write().map_err(|e| e.to_string())?;
        let mut order = self.access_order.write().map_err(|e| e.to_string())?;

        // Remove if exists
        if entries.contains_key(&entry.id) {
            order.retain(|id| *id != entry.id);
        }

        // Evict if at capacity
        while entries.len() >= self.max_entries && !order.is_empty() {
            let oldest = order.remove(0);
            entries.remove(&oldest);
        }

        // Insert new entry
        let id = entry.id;
        entries.insert(id, entry);
        order.push(id);

        Ok(())
    }

    fn read(&self, id: &Uuid) -> Option<MemoryEntry> {
        let entries = self.entries.read().ok()?;
        if let Some(entry) = entries.get(id).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            // Update access order
            if let Ok(mut order) = self.access_order.write() {
                order.retain(|oid| oid != id);
                order.push(*id);
            }
            Some(entry)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<(Uuid, f32)> {
        let entries = self.entries.read().unwrap();
        let mut results: Vec<(Uuid, f32)> = entries
            .iter()
            .map(|(id, entry)| {
                let score = cosine_similarity(query, &entry.embedding);
                (*id, score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        }
    }
}

/// Cold memory layer (simulated persistent storage)
struct ColdMemory {
    entries: RwLock<HashMap<Uuid, MemoryEntry>>,
    write_latency_us: u64,
    read_latency_us: u64,
}

impl ColdMemory {
    fn new(write_latency_us: u64, read_latency_us: u64) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            write_latency_us,
            read_latency_us,
        }
    }

    fn write(&self, entry: MemoryEntry) -> Result<(), String> {
        // Simulate disk I/O latency
        std::thread::sleep(Duration::from_micros(self.write_latency_us));

        let mut entries = self.entries.write().map_err(|e| e.to_string())?;
        entries.insert(entry.id, entry);
        Ok(())
    }

    fn read(&self, id: &Uuid) -> Option<MemoryEntry> {
        // Simulate disk I/O latency
        std::thread::sleep(Duration::from_micros(self.read_latency_us));

        self.entries.read().ok()?.get(id).cloned()
    }

    fn write_batch(&self, entries: Vec<MemoryEntry>) -> Result<usize, String> {
        // Batch write has amortized latency
        std::thread::sleep(Duration::from_micros(
            self.write_latency_us * (entries.len() as u64 / 10).max(1),
        ));

        let mut storage = self.entries.write().map_err(|e| e.to_string())?;
        let count = entries.len();
        for entry in entries {
            storage.insert(entry.id, entry);
        }
        Ok(count)
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<(Uuid, f32)> {
        // Simulate disk scan latency
        std::thread::sleep(Duration::from_micros(self.read_latency_us * 10));

        let entries = self.entries.read().unwrap();
        let mut results: Vec<(Uuid, f32)> = entries
            .iter()
            .map(|(id, entry)| {
                let score = cosine_similarity(query, &entry.embedding);
                (*id, score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }
}

/// Write-Ahead Log entry
#[derive(Clone, Debug)]
struct WalEntry {
    sequence: u64,
    operation: WalOperation,
    timestamp: u64,
    checksum: u32,
}

#[derive(Clone, Debug)]
enum WalOperation {
    Write(MemoryEntry),
    Delete(Uuid),
}

/// Write-Ahead Log for durability
struct WriteAheadLog {
    entries: RwLock<Vec<WalEntry>>,
    sequence: AtomicU64,
    sync_latency_us: u64,
}

impl WriteAheadLog {
    fn new(sync_latency_us: u64) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            sequence: AtomicU64::new(0),
            sync_latency_us,
        }
    }

    fn append(&self, operation: WalOperation) -> Result<u64, String> {
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Compute checksum of operation
        let checksum = match &operation {
            WalOperation::Write(entry) => entry.checksum,
            WalOperation::Delete(id) => crc32fast::hash(id.as_bytes()),
        };

        let wal_entry = WalEntry {
            sequence: seq,
            operation,
            timestamp,
            checksum,
        };

        let mut entries = self.entries.write().map_err(|e| e.to_string())?;
        entries.push(wal_entry);

        Ok(seq)
    }

    fn sync(&self) -> Result<(), String> {
        // Simulate fsync latency
        std::thread::sleep(Duration::from_micros(self.sync_latency_us));
        Ok(())
    }

    fn replay(&self) -> Result<Vec<WalEntry>, String> {
        let entries = self.entries.read().map_err(|e| e.to_string())?;
        Ok(entries.clone())
    }

    fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    fn truncate(&self, until_sequence: u64) -> Result<usize, String> {
        let mut entries = self.entries.write().map_err(|e| e.to_string())?;
        let original_len = entries.len();
        entries.retain(|e| e.sequence > until_sequence);
        Ok(original_len - entries.len())
    }
}

/// Dual-layer memory combining hot and cold storage with WAL
struct DualLayerMemory {
    hot: Arc<HotMemory>,
    cold: Arc<ColdMemory>,
    wal: Arc<WriteAheadLog>,
    sync_threshold: usize,
}

impl DualLayerMemory {
    fn new(hot_capacity: usize, sync_threshold: usize) -> Self {
        Self {
            hot: Arc::new(HotMemory::new(hot_capacity)),
            cold: Arc::new(ColdMemory::new(500, 100)), // 500us write, 100us read
            wal: Arc::new(WriteAheadLog::new(2000)),   // 2ms sync
            sync_threshold,
        }
    }

    fn write(&self, entry: MemoryEntry) -> Result<(), String> {
        // Write to WAL first (durability)
        self.wal.append(WalOperation::Write(entry.clone()))?;

        // Write to hot memory
        self.hot.write(entry.clone())?;

        // Async sync to cold if threshold reached
        if self.wal.len() >= self.sync_threshold {
            self.sync_to_cold()?;
        }

        Ok(())
    }

    fn read(&self, id: &Uuid) -> Option<MemoryEntry> {
        // Try hot first
        if let Some(entry) = self.hot.read(id) {
            return Some(entry);
        }

        // Fall back to cold
        if let Some(entry) = self.cold.read(id) {
            // Promote to hot
            let _ = self.hot.write(entry.clone());
            return Some(entry);
        }

        None
    }

    fn search(&self, query: &[f32], top_k: usize) -> Vec<(Uuid, f32)> {
        // Search both layers and merge
        let mut hot_results = self.hot.search(query, top_k * 2);
        let cold_results = self.cold.search(query, top_k * 2);

        // Merge and deduplicate
        let mut seen = std::collections::HashSet::new();
        for (id, score) in hot_results.iter() {
            seen.insert(*id);
        }

        for (id, score) in cold_results {
            if !seen.contains(&id) {
                hot_results.push((id, score));
                seen.insert(id);
            }
        }

        hot_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hot_results.truncate(top_k);
        hot_results
    }

    fn sync_to_cold(&self) -> Result<(), String> {
        // Get WAL entries to sync
        let entries = self.wal.replay()?;

        // Batch write to cold storage
        let memory_entries: Vec<MemoryEntry> = entries
            .into_iter()
            .filter_map(|e| match e.operation {
                WalOperation::Write(entry) => Some(entry),
                WalOperation::Delete(_) => None,
            })
            .collect();

        if !memory_entries.is_empty() {
            let last_seq = self.wal.sequence.load(Ordering::SeqCst);
            self.cold.write_batch(memory_entries)?;
            self.wal.sync()?;
            self.wal.truncate(last_seq)?;
        }

        Ok(())
    }

    fn recover(&self) -> Result<usize, String> {
        let entries = self.wal.replay()?;
        let count = entries.len();

        for wal_entry in entries {
            match wal_entry.operation {
                WalOperation::Write(entry) => {
                    self.hot.write(entry)?;
                }
                WalOperation::Delete(id) => {
                    // Handle delete during recovery
                    let _ = id; // No-op for benchmark
                }
            }
        }

        Ok(count)
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_embedding(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim).map(|i| ((seed + i) as f32 * 0.01).sin()).collect()
}

fn generate_entry(dim: usize, seed: usize) -> MemoryEntry {
    MemoryEntry::new(
        generate_embedding(dim, seed),
        format!("Document content for entry {}", seed),
    )
}

fn generate_entries(count: usize, dim: usize) -> Vec<MemoryEntry> {
    (0..count).map(|i| generate_entry(dim, i)).collect()
}

// ============================================================================
// Hot Memory Benchmarks
// ============================================================================

fn hot_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_memory");

    // Benchmark: write throughput
    let sizes = vec![100, 1000, 10000];

    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("write_throughput", size),
            size,
            |b, &size| {
                let entries = generate_entries(size, 384);

                b.iter(|| {
                    let hot = HotMemory::new(size * 2);
                    for entry in &entries {
                        hot.write(entry.clone()).unwrap();
                    }
                    black_box(hot.len());
                });
            },
        );
    }

    // Benchmark: read throughput
    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("read_throughput", size),
            size,
            |b, &size| {
                let hot = HotMemory::new(size * 2);
                let entries = generate_entries(size, 384);
                let ids: Vec<Uuid> = entries
                    .iter()
                    .map(|e| {
                        hot.write(e.clone()).unwrap();
                        e.id
                    })
                    .collect();

                b.iter(|| {
                    for id in &ids {
                        black_box(hot.read(id));
                    }
                });
            },
        );
    }

    // Benchmark: concurrent access
    group.bench_function("concurrent_access", |b| {
        let hot = Arc::new(HotMemory::new(10000));
        let entries = generate_entries(1000, 384);
        for entry in &entries {
            hot.write(entry.clone()).unwrap();
        }
        let ids: Vec<Uuid> = entries.iter().map(|e| e.id).collect();

        b.iter(|| {
            let hot_clone = hot.clone();
            let ids_clone = ids.clone();

            // Simulate concurrent reads and writes
            std::thread::scope(|s| {
                // Reader threads
                for _ in 0..4 {
                    let hot_ref = hot_clone.clone();
                    let ids_ref = ids_clone.clone();
                    s.spawn(move || {
                        for id in ids_ref.iter().take(100) {
                            black_box(hot_ref.read(id));
                        }
                    });
                }

                // Writer thread
                let hot_ref = hot_clone.clone();
                s.spawn(move || {
                    for i in 0..50 {
                        let entry = generate_entry(384, 10000 + i);
                        let _ = hot_ref.write(entry);
                    }
                });
            });
        });
    });

    // Benchmark: similarity search
    let search_sizes = vec![100, 1000, 5000, 10000];

    for size in search_sizes {
        group.bench_with_input(
            BenchmarkId::new("similarity_search", size),
            &size,
            |b, &size| {
                let hot = HotMemory::new(size * 2);
                let entries = generate_entries(size, 384);
                for entry in entries {
                    hot.write(entry).unwrap();
                }
                let query = generate_embedding(384, 999);

                b.iter(|| {
                    let results = hot.search(black_box(&query), 10);
                    black_box(results);
                });
            },
        );
    }

    // Benchmark: LRU eviction
    group.bench_function("lru_eviction", |b| {
        b.iter(|| {
            let hot = HotMemory::new(1000);
            // Write 2000 entries to trigger eviction
            for i in 0..2000 {
                let entry = generate_entry(384, i);
                hot.write(entry).unwrap();
            }
            black_box(hot.len());
        });
    });

    group.finish();
}

// ============================================================================
// Cold Memory Benchmarks
// ============================================================================

fn cold_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_memory");
    // Use shorter measurement time for I/O-bound benchmarks
    group.measurement_time(Duration::from_secs(5));

    let sizes = vec![10, 50, 100];

    // Benchmark: write throughput
    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("write_throughput", size),
            size,
            |b, &size| {
                // Use minimal latency for throughput testing
                let cold = ColdMemory::new(10, 5);
                let entries = generate_entries(size, 384);

                b.iter(|| {
                    for entry in &entries {
                        cold.write(entry.clone()).unwrap();
                    }
                    black_box(cold.len());
                });
            },
        );
    }

    // Benchmark: batch write
    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("batch_write", size), size, |b, &size| {
            let cold = ColdMemory::new(10, 5);

            b.iter(|| {
                let entries = generate_entries(size, 384);
                cold.write_batch(entries).unwrap();
                black_box(cold.len());
            });
        });
    }

    // Benchmark: read throughput
    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("read_throughput", size),
            size,
            |b, &size| {
                let cold = ColdMemory::new(10, 5);
                let entries = generate_entries(size, 384);
                let ids: Vec<Uuid> = entries
                    .iter()
                    .map(|e| {
                        cold.write(e.clone()).unwrap();
                        e.id
                    })
                    .collect();

                b.iter(|| {
                    for id in &ids {
                        black_box(cold.read(id));
                    }
                });
            },
        );
    }

    // Benchmark: similarity search
    for size in &sizes {
        group.bench_with_input(
            BenchmarkId::new("similarity_search", size),
            size,
            |b, &size| {
                let cold = ColdMemory::new(10, 5);
                let entries = generate_entries(size, 384);
                for entry in entries {
                    cold.write(entry).unwrap();
                }
                let query = generate_embedding(384, 999);

                b.iter(|| {
                    let results = cold.search(black_box(&query), 10);
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// WAL Benchmarks
// ============================================================================

fn wal_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal");
    group.measurement_time(Duration::from_secs(5));

    // Benchmark: append throughput
    let sizes = vec![100, 500, 1000, 5000];

    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("append_throughput", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let wal = WriteAheadLog::new(10); // Minimal sync latency
                    for i in 0..size {
                        let entry = generate_entry(384, i);
                        wal.append(WalOperation::Write(entry)).unwrap();
                    }
                    black_box(wal.len());
                });
            },
        );
    }

    // Benchmark: sync latency
    group.bench_function("sync_latency", |b| {
        let wal = WriteAheadLog::new(100); // 100us sync

        // Pre-populate
        for i in 0..100 {
            let entry = generate_entry(384, i);
            wal.append(WalOperation::Write(entry)).unwrap();
        }

        b.iter(|| {
            black_box(wal.sync().unwrap());
        });
    });

    // Benchmark: recovery time
    let recovery_sizes = vec![100, 500, 1000, 5000];

    for size in recovery_sizes {
        group.bench_with_input(
            BenchmarkId::new("recovery_time", size),
            &size,
            |b, &size| {
                let wal = WriteAheadLog::new(10);
                for i in 0..size {
                    let entry = generate_entry(384, i);
                    wal.append(WalOperation::Write(entry)).unwrap();
                }

                b.iter(|| {
                    let entries = wal.replay().unwrap();
                    black_box(entries.len());
                });
            },
        );
    }

    // Benchmark: truncation
    group.bench_function("truncate", |b| {
        b.iter_with_setup(
            || {
                let wal = WriteAheadLog::new(10);
                for i in 0..1000 {
                    let entry = generate_entry(384, i);
                    wal.append(WalOperation::Write(entry)).unwrap();
                }
                wal
            },
            |wal| {
                let truncated = wal.truncate(500).unwrap();
                black_box(truncated);
            },
        );
    });

    group.finish();
}

// ============================================================================
// Dual-Layer Memory Benchmarks
// ============================================================================

fn dual_layer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_layer_memory");
    group.measurement_time(Duration::from_secs(5));

    // Benchmark: write path (hot + WAL)
    let sizes = vec![100, 500, 1000];

    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("write_path", size), size, |b, &size| {
            b.iter(|| {
                let dlm = DualLayerMemory::new(size * 2, size * 10);
                for i in 0..size {
                    let entry = generate_entry(384, i);
                    dlm.write(entry).unwrap();
                }
                black_box(dlm.hot.len());
            });
        });
    }

    // Benchmark: read path (hot hit)
    for size in &sizes {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("read_hot_hit", size), size, |b, &size| {
            let dlm = DualLayerMemory::new(size * 2, size * 10);
            let entries = generate_entries(size, 384);
            let ids: Vec<Uuid> = entries
                .iter()
                .map(|e| {
                    dlm.write(e.clone()).unwrap();
                    e.id
                })
                .collect();

            b.iter(|| {
                for id in &ids {
                    black_box(dlm.read(id));
                }
            });
        });
    }

    // Benchmark: read path (cold miss -> promotion)
    group.bench_function("read_cold_promotion", |b| {
        b.iter_with_setup(
            || {
                let dlm = DualLayerMemory::new(100, 1000);
                let entries = generate_entries(200, 384);
                let mut ids = Vec::new();

                // Write to cold via sync
                for entry in &entries {
                    dlm.write(entry.clone()).unwrap();
                    ids.push(entry.id);
                }
                dlm.sync_to_cold().unwrap();

                // Clear hot cache by writing new entries
                for i in 0..150 {
                    let entry = generate_entry(384, 1000 + i);
                    dlm.write(entry).unwrap();
                }

                (dlm, ids)
            },
            |(dlm, ids)| {
                // Read entries that should be in cold
                for id in ids.iter().take(50) {
                    black_box(dlm.read(id));
                }
            },
        );
    });

    // Benchmark: sync to cold
    group.bench_function("sync_to_cold", |b| {
        b.iter_with_setup(
            || {
                let dlm = DualLayerMemory::new(1000, 10000);
                for i in 0..500 {
                    let entry = generate_entry(384, i);
                    dlm.write(entry).unwrap();
                }
                dlm
            },
            |dlm| {
                dlm.sync_to_cold().unwrap();
            },
        );
    });

    // Benchmark: recovery
    group.bench_function("recovery", |b| {
        b.iter_with_setup(
            || {
                let dlm = DualLayerMemory::new(1000, 10000);
                for i in 0..500 {
                    let entry = generate_entry(384, i);
                    dlm.write(entry).unwrap();
                }
                dlm
            },
            |dlm| {
                let recovered = dlm.recover().unwrap();
                black_box(recovered);
            },
        );
    });

    group.finish();
}

// ============================================================================
// Hybrid Search / Retrieve Context Benchmarks
// ============================================================================

fn retrieve_context_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieve_context");
    group.measurement_time(Duration::from_secs(5));

    // Benchmark: hybrid search across hot and cold
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("hybrid_search", size),
            &size,
            |b, &size| {
                let dlm = DualLayerMemory::new(size / 2, size * 10);
                let entries = generate_entries(size, 384);
                for entry in entries {
                    dlm.write(entry).unwrap();
                }
                // Sync half to cold
                dlm.sync_to_cold().unwrap();

                let query = generate_embedding(384, 999);

                b.iter(|| {
                    let results = dlm.search(black_box(&query), 10);
                    black_box(results);
                });
            },
        );
    }

    // Benchmark: different top_k values
    let top_k_values = vec![5, 10, 20, 50, 100];

    for top_k in top_k_values {
        group.bench_with_input(
            BenchmarkId::new("top_k_values", top_k),
            &top_k,
            |b, &top_k| {
                let dlm = DualLayerMemory::new(2000, 10000);
                let entries = generate_entries(2000, 384);
                for entry in entries {
                    dlm.write(entry).unwrap();
                }
                let query = generate_embedding(384, 999);

                b.iter(|| {
                    let results = dlm.search(black_box(&query), top_k);
                    black_box(results);
                });
            },
        );
    }

    // Benchmark: hit rate impact on performance
    group.bench_function("hot_hit_rate_100pct", |b| {
        let dlm = DualLayerMemory::new(1000, 10000);
        let entries = generate_entries(500, 384);
        for entry in entries {
            dlm.write(entry).unwrap();
        }
        let query = generate_embedding(384, 999);

        b.iter(|| {
            let results = dlm.search(black_box(&query), 10);
            black_box(results);
        });
    });

    group.bench_function("hot_hit_rate_50pct", |b| {
        let dlm = DualLayerMemory::new(250, 10000);
        let entries = generate_entries(500, 384);
        for entry in entries {
            dlm.write(entry).unwrap();
        }
        dlm.sync_to_cold().unwrap();
        let query = generate_embedding(384, 999);

        b.iter(|| {
            let results = dlm.search(black_box(&query), 10);
            black_box(results);
        });
    });

    group.finish();
}

// ============================================================================
// Memory Efficiency Benchmarks
// ============================================================================

fn memory_efficiency_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Benchmark: memory usage per entry
    group.bench_function("entry_size_calculation", |b| {
        b.iter(|| {
            let entry = generate_entry(384, 0);
            black_box(entry.size_bytes());
        });
    });

    // Benchmark: cache hit rate under different loads
    let load_patterns = vec![
        ("uniform", |i: usize| i % 1000),
        ("zipf_like", |i: usize| (i as f64).sqrt() as usize % 1000),
        (
            "hot_cold",
            |i: usize| if i % 10 == 0 { i % 100 } else { i % 10 },
        ),
    ];

    for (name, pattern_fn) in load_patterns {
        group.bench_function(format!("hit_rate_{}", name), |b| {
            let hot = HotMemory::new(500);
            let entries = generate_entries(1000, 384);
            for entry in &entries {
                hot.write(entry.clone()).unwrap();
            }
            let ids: Vec<Uuid> = entries.iter().map(|e| e.id).collect();

            b.iter(|| {
                for i in 0..1000 {
                    let idx = pattern_fn(i);
                    black_box(hot.read(&ids[idx]));
                }
                black_box(hot.hit_rate());
            });
        });
    }

    group.finish();
}

// ============================================================================
// Async Benchmarks (for integration with tokio runtime)
// ============================================================================

fn async_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("async_operations");
    group.measurement_time(Duration::from_secs(5));

    // Benchmark: async write with simulated I/O
    group.bench_function("async_write_batch", |b| {
        b.iter(|| {
            rt.block_on(async {
                let entries = generate_entries(100, 384);
                let dlm = DualLayerMemory::new(500, 1000);

                for entry in entries {
                    dlm.write(entry).unwrap();
                }

                // Simulate async sync
                tokio::task::yield_now().await;
                dlm.sync_to_cold().unwrap();

                black_box(dlm.hot.len());
            });
        });
    });

    // Benchmark: async search
    group.bench_function("async_search", |b| {
        let dlm = DualLayerMemory::new(500, 1000);
        let entries = generate_entries(500, 384);
        for entry in entries {
            dlm.write(entry).unwrap();
        }
        let query = generate_embedding(384, 999);

        b.iter(|| {
            rt.block_on(async {
                let results = dlm.search(&query, 10);
                tokio::task::yield_now().await;
                black_box(results);
            });
        });
    });

    // Benchmark: concurrent async operations
    group.bench_function("async_concurrent_ops", |b| {
        let dlm = Arc::new(DualLayerMemory::new(500, 1000));
        let entries = generate_entries(200, 384);
        for entry in &entries {
            dlm.write(entry.clone()).unwrap();
        }
        let ids: Vec<Uuid> = entries.iter().map(|e| e.id).collect();

        b.iter(|| {
            rt.block_on(async {
                let dlm_clone = dlm.clone();
                let ids_clone = ids.clone();

                let handles: Vec<_> = (0..4)
                    .map(|t| {
                        let dlm_ref = dlm_clone.clone();
                        let ids_ref = ids_clone.clone();
                        tokio::spawn(async move {
                            for i in (t * 50)..((t + 1) * 50) {
                                if i < ids_ref.len() {
                                    black_box(dlm_ref.read(&ids_ref[i]));
                                }
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.await.unwrap();
                }
            });
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Groups and Main
// ============================================================================

criterion_group!(
    benches,
    hot_memory_benchmarks,
    cold_memory_benchmarks,
    wal_benchmarks,
    dual_layer_benchmarks,
    retrieve_context_benchmarks,
    memory_efficiency_benchmarks,
    async_benchmarks,
);

criterion_main!(benches);
