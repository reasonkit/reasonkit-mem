//! Configuration validation and defaults for the dual-layer memory system.
//!
//! This module provides configuration types and validation for the hot/cold
//! memory architecture, Write-Ahead Log (WAL), and synchronization policies.
//!
//! # Architecture
//!
//! ```text
//! +------------------+     +-------------------+
//! |   Hot Memory     |     |   Cold Memory     |
//! | (In-Memory LRU)  |---->|   (Sled on Disk)  |
//! +------------------+     +-------------------+
//!          |                        |
//!          v                        v
//! +------------------------------------------+
//! |         Write-Ahead Log (WAL)            |
//! |      (Durability & Crash Recovery)       |
//! +------------------------------------------+
//! ```
//!
//! # Usage
//!
//! ```rust
//! use std::path::PathBuf;
//! use reasonkit_mem::storage::config::DualLayerConfig;
//!
//! // Use defaults
//! let config = DualLayerConfig::default();
//!
//! // Or create with custom data directory
//! let config = DualLayerConfig::with_data_dir(PathBuf::from("/data/reasonkit"));
//!
//! // Validate before use
//! config.validate().expect("Invalid configuration");
//! ```

use crate::error::{MemError, MemResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Write-Ahead Log synchronization mode.
///
/// Controls durability vs performance tradeoff for WAL writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SyncMode {
    /// Synchronous flush after every write (maximum durability).
    /// Best for critical data, worst for throughput.
    Sync,

    /// Flush on segment completion (balanced).
    /// Default mode - good balance of durability and performance.
    #[default]
    Balanced,

    /// Asynchronous writes with periodic flush (maximum performance).
    /// Risk of data loss on crash, but highest throughput.
    Async,

    /// Operating system decides when to flush.
    /// Fastest but least durable.
    OsDefault,
}

impl SyncMode {
    /// Get the flush interval for this mode (if applicable).
    pub fn flush_interval(&self) -> Option<Duration> {
        match self {
            SyncMode::Sync => None, // Immediate flush
            SyncMode::Balanced => Some(Duration::from_millis(100)),
            SyncMode::Async => Some(Duration::from_secs(1)),
            SyncMode::OsDefault => None, // OS controlled
        }
    }

    /// Whether this mode requires explicit fsync calls.
    pub fn requires_fsync(&self) -> bool {
        matches!(self, SyncMode::Sync | SyncMode::Balanced)
    }
}

/// Configuration for the hot (in-memory) storage layer.
///
/// The hot layer provides fast access to recently accessed entries
/// using an LRU eviction policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotMemoryConfig {
    /// Maximum number of entries in hot memory.
    /// When exceeded, oldest entries are evicted to cold storage.
    ///
    /// Default: 10,000 entries
    pub max_entries: usize,

    /// Time-to-live for entries in seconds.
    /// Entries older than this are eligible for eviction regardless of access.
    ///
    /// Default: 3600 (1 hour)
    pub ttl_secs: u64,

    /// Number of entries to evict in a single batch.
    /// Larger batches are more efficient but may cause latency spikes.
    ///
    /// Default: 100
    pub eviction_batch_size: usize,

    /// Whether to enable access tracking for LRU.
    /// Disabling improves write performance but uses FIFO eviction.
    ///
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_lru_tracking: bool,

    /// Pre-allocate capacity for this many entries.
    /// Reduces allocation overhead during operation.
    ///
    /// Default: 1000
    #[serde(default = "default_preallocate")]
    pub preallocate_capacity: usize,
}

fn default_true() -> bool {
    true
}

fn default_preallocate() -> usize {
    1000
}

impl Default for HotMemoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl_secs: 3600,
            eviction_batch_size: 100,
            enable_lru_tracking: true,
            preallocate_capacity: 1000,
        }
    }
}

impl HotMemoryConfig {
    /// Create a high-performance configuration for read-heavy workloads.
    ///
    /// - Larger cache (50K entries)
    /// - Longer TTL (4 hours)
    /// - LRU tracking enabled
    pub fn high_performance() -> Self {
        Self {
            max_entries: 50_000,
            ttl_secs: 14400, // 4 hours
            eviction_batch_size: 500,
            enable_lru_tracking: true,
            preallocate_capacity: 10_000,
        }
    }

    /// Create a low-memory configuration for constrained environments.
    ///
    /// - Small cache (1K entries)
    /// - Short TTL (10 minutes)
    /// - LRU tracking disabled for performance
    pub fn low_memory() -> Self {
        Self {
            max_entries: 1_000,
            ttl_secs: 600, // 10 minutes
            eviction_batch_size: 50,
            enable_lru_tracking: false,
            preallocate_capacity: 100,
        }
    }

    /// Get TTL as Duration.
    pub fn ttl(&self) -> Duration {
        Duration::from_secs(self.ttl_secs)
    }

    /// Validate this configuration.
    pub fn validate(&self) -> MemResult<()> {
        if self.max_entries == 0 {
            return Err(MemError::config("max_entries must be greater than 0"));
        }

        if self.eviction_batch_size == 0 {
            return Err(MemError::config(
                "eviction_batch_size must be greater than 0",
            ));
        }

        if self.eviction_batch_size > self.max_entries {
            return Err(MemError::config(
                "eviction_batch_size cannot exceed max_entries",
            ));
        }

        if self.preallocate_capacity > self.max_entries {
            return Err(MemError::config(
                "preallocate_capacity cannot exceed max_entries",
            ));
        }

        Ok(())
    }
}

/// Configuration for the cold (disk-based) storage layer.
///
/// The cold layer uses Sled embedded database for persistent storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdMemoryConfig {
    /// Path to the Sled database directory.
    /// Will be created if it doesn't exist.
    ///
    /// Default: `$XDG_DATA_HOME/reasonkit/cold` or `./data/cold`
    pub db_path: PathBuf,

    /// Sled cache size in megabytes.
    /// Larger cache improves read performance but uses more memory.
    ///
    /// Default: 128 MB
    pub cache_size_mb: usize,

    /// Interval between automatic flushes in seconds.
    /// Lower values improve durability but reduce write throughput.
    ///
    /// Default: 5 seconds
    pub flush_interval_secs: u64,

    /// Whether to enable compression for cold storage.
    /// Reduces disk usage but increases CPU overhead.
    ///
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_compression: bool,

    /// Maximum size for the cold storage in megabytes.
    /// 0 means unlimited.
    ///
    /// Default: 0 (unlimited)
    #[serde(default)]
    pub max_size_mb: usize,

    /// Mode for database opening.
    ///
    /// Default: CreateOrOpen
    #[serde(default)]
    pub mode: ColdStorageMode,
}

/// Mode for opening cold storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ColdStorageMode {
    /// Create if doesn't exist, otherwise open existing.
    #[default]
    CreateOrOpen,

    /// Always create new (will clear existing data).
    Create,

    /// Only open existing (error if doesn't exist).
    Open,

    /// Open in read-only mode.
    ReadOnly,
}

impl Default for ColdMemoryConfig {
    fn default() -> Self {
        let db_path = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("reasonkit")
            .join("cold");

        Self {
            db_path,
            cache_size_mb: 128,
            flush_interval_secs: 5,
            enable_compression: true,
            max_size_mb: 0,
            mode: ColdStorageMode::default(),
        }
    }
}

impl ColdMemoryConfig {
    /// Create configuration for a specific database path.
    pub fn with_path(db_path: PathBuf) -> Self {
        Self {
            db_path,
            ..Default::default()
        }
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            cache_size_mb: 512,
            flush_interval_secs: 30,
            enable_compression: false, // Trade disk for CPU
            ..Default::default()
        }
    }

    /// Get cache size in bytes.
    pub fn cache_size_bytes(&self) -> usize {
        self.cache_size_mb * 1024 * 1024
    }

    /// Get flush interval as Duration.
    pub fn flush_interval(&self) -> Duration {
        Duration::from_secs(self.flush_interval_secs)
    }

    /// Validate this configuration.
    pub fn validate(&self) -> MemResult<()> {
        if self.cache_size_mb == 0 {
            return Err(MemError::config("cache_size_mb must be greater than 0"));
        }

        if self.cache_size_mb > 16384 {
            // 16 GB max sanity check
            return Err(MemError::config(
                "cache_size_mb exceeds maximum of 16384 (16 GB)",
            ));
        }

        if self.flush_interval_secs == 0 {
            return Err(MemError::config(
                "flush_interval_secs must be greater than 0",
            ));
        }

        // Validate path is not empty
        if self.db_path.as_os_str().is_empty() {
            return Err(MemError::config("db_path cannot be empty"));
        }

        Ok(())
    }
}

/// Configuration for the Write-Ahead Log (WAL).
///
/// The WAL provides durability and crash recovery by logging
/// all operations before they are applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfig {
    /// Directory for WAL segment files.
    ///
    /// Default: `$XDG_DATA_HOME/reasonkit/wal` or `./data/wal`
    pub dir: PathBuf,

    /// Maximum size of each WAL segment in megabytes.
    /// Larger segments reduce file overhead but increase recovery time.
    ///
    /// Default: 64 MB
    pub segment_size_mb: usize,

    /// Synchronization mode for WAL writes.
    ///
    /// Default: Balanced
    pub sync_mode: SyncMode,

    /// Whether to enable WAL.
    /// Disabling improves performance but risks data loss on crash.
    ///
    /// Default: true
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum number of WAL segments to retain.
    /// Older segments are deleted after checkpoint.
    /// 0 means unlimited.
    ///
    /// Default: 10
    #[serde(default = "default_max_segments")]
    pub max_segments: usize,

    /// Whether to preallocate segment files.
    /// Improves write performance but uses more disk space initially.
    ///
    /// Default: true
    #[serde(default = "default_true")]
    pub preallocate: bool,
}

fn default_max_segments() -> usize {
    10
}

impl Default for WalConfig {
    fn default() -> Self {
        let dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("reasonkit")
            .join("wal");

        Self {
            dir,
            segment_size_mb: 64,
            sync_mode: SyncMode::default(),
            enabled: true,
            max_segments: 10,
            preallocate: true,
        }
    }
}

impl WalConfig {
    /// Create configuration for a specific directory.
    pub fn with_dir(dir: PathBuf) -> Self {
        Self {
            dir,
            ..Default::default()
        }
    }

    /// Create a high-durability configuration.
    pub fn high_durability() -> Self {
        Self {
            sync_mode: SyncMode::Sync,
            segment_size_mb: 32, // Smaller segments for faster recovery
            preallocate: true,
            ..Default::default()
        }
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            sync_mode: SyncMode::Async,
            segment_size_mb: 128, // Larger segments for fewer file operations
            preallocate: true,
            ..Default::default()
        }
    }

    /// Disable WAL entirely (not recommended for production).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Get segment size in bytes.
    pub fn segment_size_bytes(&self) -> usize {
        self.segment_size_mb * 1024 * 1024
    }

    /// Validate this configuration.
    pub fn validate(&self) -> MemResult<()> {
        if !self.enabled {
            return Ok(()); // No validation needed if disabled
        }

        if self.segment_size_mb == 0 {
            return Err(MemError::config("segment_size_mb must be greater than 0"));
        }

        if self.segment_size_mb > 1024 {
            // 1 GB max sanity check
            return Err(MemError::config(
                "segment_size_mb exceeds maximum of 1024 (1 GB)",
            ));
        }

        if self.dir.as_os_str().is_empty() {
            return Err(MemError::config("WAL dir cannot be empty"));
        }

        Ok(())
    }
}

/// Configuration for hot-to-cold synchronization.
///
/// Controls when and how entries are migrated from hot to cold storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Interval between sync operations in seconds.
    /// Lower values keep cold storage more up-to-date.
    ///
    /// Default: 60 seconds
    pub interval_secs: u64,

    /// Age threshold in seconds for hot-to-cold migration.
    /// Entries older than this are migrated during sync.
    ///
    /// Default: 300 (5 minutes)
    pub hot_to_cold_age_secs: u64,

    /// Number of entries to migrate per sync batch.
    /// Larger batches are more efficient but may cause latency.
    ///
    /// Default: 1000
    pub batch_size: usize,

    /// Whether to enable automatic sync.
    /// If disabled, sync must be triggered manually.
    ///
    /// Default: true
    #[serde(default = "default_true")]
    pub auto_sync_enabled: bool,

    /// Whether to sync on shutdown.
    ///
    /// Default: true
    #[serde(default = "default_true")]
    pub sync_on_shutdown: bool,

    /// Maximum time to spend on a single sync operation in milliseconds.
    /// 0 means unlimited.
    ///
    /// Default: 5000 (5 seconds)
    #[serde(default = "default_max_sync_time")]
    pub max_sync_time_ms: u64,
}

fn default_max_sync_time() -> u64 {
    5000
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            interval_secs: 60,
            hot_to_cold_age_secs: 300,
            batch_size: 1000,
            auto_sync_enabled: true,
            sync_on_shutdown: true,
            max_sync_time_ms: 5000,
        }
    }
}

impl SyncConfig {
    /// Create a configuration for write-heavy workloads.
    /// More aggressive sync to prevent hot memory overflow.
    pub fn write_heavy() -> Self {
        Self {
            interval_secs: 30,
            hot_to_cold_age_secs: 60,
            batch_size: 2000,
            auto_sync_enabled: true,
            sync_on_shutdown: true,
            max_sync_time_ms: 10000,
        }
    }

    /// Create a configuration for read-heavy workloads.
    /// Less aggressive sync to keep more data in hot memory.
    pub fn read_heavy() -> Self {
        Self {
            interval_secs: 300,        // 5 minutes
            hot_to_cold_age_secs: 900, // 15 minutes
            batch_size: 500,
            auto_sync_enabled: true,
            sync_on_shutdown: true,
            max_sync_time_ms: 3000,
        }
    }

    /// Get sync interval as Duration.
    pub fn interval(&self) -> Duration {
        Duration::from_secs(self.interval_secs)
    }

    /// Get hot-to-cold age threshold as Duration.
    pub fn hot_to_cold_age(&self) -> Duration {
        Duration::from_secs(self.hot_to_cold_age_secs)
    }

    /// Get maximum sync time as Duration.
    pub fn max_sync_time(&self) -> Option<Duration> {
        if self.max_sync_time_ms == 0 {
            None
        } else {
            Some(Duration::from_millis(self.max_sync_time_ms))
        }
    }

    /// Validate this configuration.
    pub fn validate(&self) -> MemResult<()> {
        if self.interval_secs == 0 {
            return Err(MemError::config("interval_secs must be greater than 0"));
        }

        if self.batch_size == 0 {
            return Err(MemError::config("batch_size must be greater than 0"));
        }

        // Sanity check: sync interval shouldn't be too long
        if self.interval_secs > 86400 {
            // 24 hours
            return Err(MemError::config(
                "interval_secs exceeds maximum of 86400 (24 hours)",
            ));
        }

        // Migration age should be less than or equal to sync interval
        // Otherwise entries might never get migrated
        if self.hot_to_cold_age_secs > self.interval_secs * 10 {
            tracing::warn!(
                "hot_to_cold_age_secs ({}) is much larger than interval_secs ({}), \
                 entries may stay in hot memory for a long time",
                self.hot_to_cold_age_secs,
                self.interval_secs
            );
        }

        Ok(())
    }
}

/// Complete configuration for the dual-layer memory system.
///
/// This is the main configuration type that combines all sub-configurations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DualLayerConfig {
    /// Hot (in-memory) layer configuration.
    pub hot: HotMemoryConfig,

    /// Cold (disk-based) layer configuration.
    pub cold: ColdMemoryConfig,

    /// Write-Ahead Log configuration.
    pub wal: WalConfig,

    /// Synchronization configuration.
    pub sync: SyncConfig,
}

impl DualLayerConfig {
    /// Create configuration with a custom data directory.
    ///
    /// All storage paths (cold DB, WAL) will be placed under this directory.
    ///
    /// # Arguments
    /// * `data_dir` - Base directory for all storage files
    ///
    /// # Example
    /// ```rust
    /// use std::path::PathBuf;
    /// use reasonkit_mem::storage::config::DualLayerConfig;
    ///
    /// let config = DualLayerConfig::with_data_dir(PathBuf::from("/data/reasonkit"));
    /// assert!(config.cold.db_path.starts_with("/data/reasonkit"));
    /// assert!(config.wal.dir.starts_with("/data/reasonkit"));
    /// ```
    pub fn with_data_dir(data_dir: PathBuf) -> Self {
        Self {
            hot: HotMemoryConfig::default(),
            cold: ColdMemoryConfig {
                db_path: data_dir.join("cold"),
                ..Default::default()
            },
            wal: WalConfig {
                dir: data_dir.join("wal"),
                ..Default::default()
            },
            sync: SyncConfig::default(),
        }
    }

    /// Create a high-performance configuration.
    ///
    /// Optimized for throughput with larger caches and async writes.
    /// May sacrifice some durability for performance.
    pub fn high_performance(data_dir: PathBuf) -> Self {
        Self {
            hot: HotMemoryConfig::high_performance(),
            cold: ColdMemoryConfig {
                db_path: data_dir.join("cold"),
                ..ColdMemoryConfig::high_performance()
            },
            wal: WalConfig {
                dir: data_dir.join("wal"),
                ..WalConfig::high_performance()
            },
            sync: SyncConfig::write_heavy(),
        }
    }

    /// Create a high-durability configuration.
    ///
    /// Optimized for data safety with synchronous writes.
    /// May sacrifice some performance for durability.
    pub fn high_durability(data_dir: PathBuf) -> Self {
        Self {
            hot: HotMemoryConfig::default(),
            cold: ColdMemoryConfig {
                db_path: data_dir.join("cold"),
                ..Default::default()
            },
            wal: WalConfig {
                dir: data_dir.join("wal"),
                ..WalConfig::high_durability()
            },
            sync: SyncConfig {
                interval_secs: 30,
                sync_on_shutdown: true,
                ..Default::default()
            },
        }
    }

    /// Create a low-memory configuration.
    ///
    /// For constrained environments with limited RAM.
    pub fn low_memory(data_dir: PathBuf) -> Self {
        Self {
            hot: HotMemoryConfig::low_memory(),
            cold: ColdMemoryConfig {
                db_path: data_dir.join("cold"),
                cache_size_mb: 32,
                ..Default::default()
            },
            wal: WalConfig {
                dir: data_dir.join("wal"),
                segment_size_mb: 16,
                ..Default::default()
            },
            sync: SyncConfig {
                batch_size: 100,
                ..Default::default()
            },
        }
    }

    /// Create a configuration optimized for testing.
    ///
    /// Uses temporary paths and aggressive sync for predictable behavior.
    #[cfg(test)]
    pub fn for_testing() -> Self {
        let temp_dir =
            std::env::temp_dir().join(format!("reasonkit_test_{}", uuid::Uuid::new_v4()));

        Self {
            hot: HotMemoryConfig {
                max_entries: 100,
                ttl_secs: 60,
                eviction_batch_size: 10,
                enable_lru_tracking: true,
                preallocate_capacity: 10,
            },
            cold: ColdMemoryConfig {
                db_path: temp_dir.join("cold"),
                cache_size_mb: 8,
                flush_interval_secs: 1,
                enable_compression: false,
                max_size_mb: 100,
                mode: ColdStorageMode::Create,
            },
            wal: WalConfig {
                dir: temp_dir.join("wal"),
                segment_size_mb: 1,
                sync_mode: SyncMode::Sync,
                enabled: true,
                max_segments: 3,
                preallocate: false,
            },
            sync: SyncConfig {
                interval_secs: 1,
                hot_to_cold_age_secs: 1,
                batch_size: 10,
                auto_sync_enabled: true,
                sync_on_shutdown: true,
                max_sync_time_ms: 1000,
            },
        }
    }

    /// Validate the complete configuration.
    ///
    /// Checks all sub-configurations and cross-configuration constraints.
    ///
    /// # Errors
    /// Returns an error if any configuration is invalid.
    pub fn validate(&self) -> MemResult<()> {
        // Validate each sub-configuration
        self.hot.validate()?;
        self.cold.validate()?;
        self.wal.validate()?;
        self.sync.validate()?;

        // Cross-configuration validation

        // Eviction batch should be reasonable compared to max entries
        if self.hot.eviction_batch_size > self.hot.max_entries / 2 {
            tracing::warn!(
                "eviction_batch_size ({}) is more than half of max_entries ({}), \
                 this may cause large eviction operations",
                self.hot.eviction_batch_size,
                self.hot.max_entries
            );
        }

        // Sync batch should be reasonable
        if self.sync.batch_size > self.hot.max_entries {
            return Err(MemError::config(
                "sync batch_size cannot exceed hot max_entries",
            ));
        }

        // Ensure cold path and WAL path are different
        if self.cold.db_path == self.wal.dir {
            return Err(MemError::config(
                "cold db_path and wal dir cannot be the same",
            ));
        }

        // Warn if WAL is disabled
        if !self.wal.enabled {
            tracing::warn!(
                "WAL is disabled - data may be lost on crash. \
                 This is not recommended for production use."
            );
        }

        Ok(())
    }

    /// Ensure all required directories exist.
    ///
    /// Creates directories if they don't exist.
    ///
    /// # Errors
    /// Returns an error if directories cannot be created.
    pub fn ensure_directories(&self) -> MemResult<()> {
        if let Some(parent) = self.cold.db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemError::io(format!(
                    "Failed to create cold storage directory {:?}: {}",
                    parent, e
                ))
            })?;
        }

        if self.wal.enabled {
            std::fs::create_dir_all(&self.wal.dir).map_err(|e| {
                MemError::io(format!(
                    "Failed to create WAL directory {:?}: {}",
                    self.wal.dir, e
                ))
            })?;
        }

        Ok(())
    }

    /// Load configuration from a TOML file.
    ///
    /// # Arguments
    /// * `path` - Path to the TOML configuration file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_toml_file(path: &std::path::Path) -> MemResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MemError::io(format!("Failed to read config file {:?}: {}", path, e)))?;

        Self::from_toml(&content)
    }

    /// Load configuration from a TOML string.
    ///
    /// # Arguments
    /// * `toml_str` - TOML configuration string
    ///
    /// # Errors
    /// Returns an error if the TOML cannot be parsed.
    pub fn from_toml(toml_str: &str) -> MemResult<Self> {
        // Note: This requires the `toml` crate. For now, we use serde_json
        // as a placeholder. In production, you'd use:
        // toml::from_str(toml_str).map_err(|e| MemError::config(e.to_string()))

        // Fallback: try to parse as JSON (for testing)
        serde_json::from_str(toml_str)
            .map_err(|e| MemError::config(format!("Failed to parse config: {}", e)))
    }

    /// Serialize configuration to TOML string.
    ///
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn to_toml(&self) -> MemResult<String> {
        // Note: This requires the `toml` crate. For now, we use JSON.
        // In production, you'd use:
        // toml::to_string_pretty(self).map_err(|e| MemError::config(e.to_string()))

        serde_json::to_string_pretty(self)
            .map_err(|e| MemError::config(format!("Failed to serialize config: {}", e)))
    }

    /// Create a builder for more fluent configuration.
    pub fn builder() -> DualLayerConfigBuilder {
        DualLayerConfigBuilder::default()
    }
}

/// Builder for DualLayerConfig.
///
/// Provides a fluent API for constructing configurations.
#[derive(Debug, Default)]
pub struct DualLayerConfigBuilder {
    hot: Option<HotMemoryConfig>,
    cold: Option<ColdMemoryConfig>,
    wal: Option<WalConfig>,
    sync: Option<SyncConfig>,
    data_dir: Option<PathBuf>,
}

impl DualLayerConfigBuilder {
    /// Set the base data directory.
    pub fn data_dir(mut self, dir: PathBuf) -> Self {
        self.data_dir = Some(dir);
        self
    }

    /// Set hot memory configuration.
    pub fn hot(mut self, config: HotMemoryConfig) -> Self {
        self.hot = Some(config);
        self
    }

    /// Set hot memory max entries.
    pub fn hot_max_entries(mut self, max_entries: usize) -> Self {
        let mut config = self.hot.take().unwrap_or_default();
        config.max_entries = max_entries;
        self.hot = Some(config);
        self
    }

    /// Set hot memory TTL.
    pub fn hot_ttl_secs(mut self, ttl_secs: u64) -> Self {
        let mut config = self.hot.take().unwrap_or_default();
        config.ttl_secs = ttl_secs;
        self.hot = Some(config);
        self
    }

    /// Set cold memory configuration.
    pub fn cold(mut self, config: ColdMemoryConfig) -> Self {
        self.cold = Some(config);
        self
    }

    /// Set cold memory cache size.
    pub fn cold_cache_size_mb(mut self, cache_size_mb: usize) -> Self {
        let mut config = self.cold.take().unwrap_or_default();
        config.cache_size_mb = cache_size_mb;
        self.cold = Some(config);
        self
    }

    /// Set WAL configuration.
    pub fn wal(mut self, config: WalConfig) -> Self {
        self.wal = Some(config);
        self
    }

    /// Set WAL sync mode.
    pub fn wal_sync_mode(mut self, mode: SyncMode) -> Self {
        let mut config = self.wal.take().unwrap_or_default();
        config.sync_mode = mode;
        self.wal = Some(config);
        self
    }

    /// Disable WAL.
    pub fn wal_disabled(mut self) -> Self {
        let mut config = self.wal.take().unwrap_or_default();
        config.enabled = false;
        self.wal = Some(config);
        self
    }

    /// Set sync configuration.
    pub fn sync(mut self, config: SyncConfig) -> Self {
        self.sync = Some(config);
        self
    }

    /// Set sync interval.
    pub fn sync_interval_secs(mut self, interval_secs: u64) -> Self {
        let mut config = self.sync.take().unwrap_or_default();
        config.interval_secs = interval_secs;
        self.sync = Some(config);
        self
    }

    /// Build the configuration.
    ///
    /// # Errors
    /// Returns an error if validation fails.
    pub fn build(self) -> MemResult<DualLayerConfig> {
        let base = if let Some(dir) = self.data_dir {
            DualLayerConfig::with_data_dir(dir)
        } else {
            DualLayerConfig::default()
        };

        let config = DualLayerConfig {
            hot: self.hot.unwrap_or(base.hot),
            cold: self.cold.unwrap_or(base.cold),
            wal: self.wal.unwrap_or(base.wal),
            sync: self.sync.unwrap_or(base.sync),
        };

        config.validate()?;
        Ok(config)
    }

    /// Build without validation (use with caution).
    pub fn build_unchecked(self) -> DualLayerConfig {
        let base = if let Some(dir) = self.data_dir {
            DualLayerConfig::with_data_dir(dir)
        } else {
            DualLayerConfig::default()
        };

        DualLayerConfig {
            hot: self.hot.unwrap_or(base.hot),
            cold: self.cold.unwrap_or(base.cold),
            wal: self.wal.unwrap_or(base.wal),
            sync: self.sync.unwrap_or(base.sync),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DualLayerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.hot.max_entries, 10_000);
        assert_eq!(config.cold.cache_size_mb, 128);
        assert!(config.wal.enabled);
        assert_eq!(config.sync.interval_secs, 60);
    }

    #[test]
    fn test_with_data_dir() {
        let data_dir = PathBuf::from("/tmp/test_reasonkit");
        let config = DualLayerConfig::with_data_dir(data_dir.clone());

        assert_eq!(config.cold.db_path, data_dir.join("cold"));
        assert_eq!(config.wal.dir, data_dir.join("wal"));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_performance_config() {
        let config = DualLayerConfig::high_performance(PathBuf::from("/data"));
        assert!(config.validate().is_ok());
        assert_eq!(config.hot.max_entries, 50_000);
        assert_eq!(config.wal.sync_mode, SyncMode::Async);
    }

    #[test]
    fn test_high_durability_config() {
        let config = DualLayerConfig::high_durability(PathBuf::from("/data"));
        assert!(config.validate().is_ok());
        assert_eq!(config.wal.sync_mode, SyncMode::Sync);
    }

    #[test]
    fn test_low_memory_config() {
        let config = DualLayerConfig::low_memory(PathBuf::from("/data"));
        assert!(config.validate().is_ok());
        assert_eq!(config.hot.max_entries, 1_000);
        assert_eq!(config.cold.cache_size_mb, 32);
    }

    #[test]
    fn test_hot_config_validation() {
        let mut config = HotMemoryConfig::default();
        assert!(config.validate().is_ok());

        config.max_entries = 0;
        assert!(config.validate().is_err());

        config.max_entries = 100;
        config.eviction_batch_size = 200; // Greater than max
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cold_config_validation() {
        let mut config = ColdMemoryConfig::default();
        assert!(config.validate().is_ok());

        config.cache_size_mb = 0;
        assert!(config.validate().is_err());

        config.cache_size_mb = 20000; // > 16GB
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_wal_config_validation() {
        let mut config = WalConfig::default();
        assert!(config.validate().is_ok());

        config.segment_size_mb = 0;
        assert!(config.validate().is_err());

        config.segment_size_mb = 2000; // > 1GB
        assert!(config.validate().is_err());

        // Disabled WAL should pass validation
        let disabled = WalConfig::disabled();
        assert!(disabled.validate().is_ok());
    }

    #[test]
    fn test_sync_config_validation() {
        let mut config = SyncConfig::default();
        assert!(config.validate().is_ok());

        config.interval_secs = 0;
        assert!(config.validate().is_err());

        config.interval_secs = 100000; // > 24 hours
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cross_config_validation() {
        let mut config = DualLayerConfig::default();

        // Same path for cold and WAL should fail
        config.cold.db_path = PathBuf::from("/data/same");
        config.wal.dir = PathBuf::from("/data/same");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sync_mode_properties() {
        assert!(SyncMode::Sync.requires_fsync());
        assert!(SyncMode::Balanced.requires_fsync());
        assert!(!SyncMode::Async.requires_fsync());
        assert!(!SyncMode::OsDefault.requires_fsync());

        assert!(SyncMode::Sync.flush_interval().is_none());
        assert!(SyncMode::Balanced.flush_interval().is_some());
    }

    #[test]
    fn test_builder() {
        let config = DualLayerConfig::builder()
            .data_dir(PathBuf::from("/data"))
            .hot_max_entries(5000)
            .cold_cache_size_mb(256)
            .wal_sync_mode(SyncMode::Sync)
            .sync_interval_secs(30)
            .build()
            .expect("Build failed");

        assert_eq!(config.hot.max_entries, 5000);
        assert_eq!(config.cold.cache_size_mb, 256);
        assert_eq!(config.wal.sync_mode, SyncMode::Sync);
        assert_eq!(config.sync.interval_secs, 30);
    }

    #[test]
    fn test_duration_helpers() {
        let hot = HotMemoryConfig::default();
        assert_eq!(hot.ttl(), Duration::from_secs(3600));

        let cold = ColdMemoryConfig::default();
        assert_eq!(cold.flush_interval(), Duration::from_secs(5));

        let sync = SyncConfig::default();
        assert_eq!(sync.interval(), Duration::from_secs(60));
        assert_eq!(sync.hot_to_cold_age(), Duration::from_secs(300));
    }

    #[test]
    fn test_serialization() {
        let config = DualLayerConfig::default();
        let json = config.to_toml().expect("Serialization failed");
        let parsed: DualLayerConfig = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(parsed.hot.max_entries, config.hot.max_entries);
        assert_eq!(parsed.cold.cache_size_mb, config.cold.cache_size_mb);
    }

    #[test]
    fn test_for_testing_config() {
        let config = DualLayerConfig::for_testing();
        assert!(config.validate().is_ok());
        // Testing config uses small values
        assert!(config.hot.max_entries < 1000);
        assert!(config.sync.interval_secs < 10);
    }

    #[test]
    fn test_presets() {
        // All presets should validate
        assert!(HotMemoryConfig::high_performance().validate().is_ok());
        assert!(HotMemoryConfig::low_memory().validate().is_ok());
        assert!(ColdMemoryConfig::high_performance().validate().is_ok());
        assert!(WalConfig::high_durability().validate().is_ok());
        assert!(WalConfig::high_performance().validate().is_ok());
        assert!(SyncConfig::write_heavy().validate().is_ok());
        assert!(SyncConfig::read_heavy().validate().is_ok());
    }
}
