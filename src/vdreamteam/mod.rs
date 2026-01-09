//! vDreamTeam AI Agent Memory System
//!
//! This module provides structured memory for AI agent teams with:
//! - **Constitutional Layer**: Shared identity, constraints, and governance
//! - **Role-Specific Memory**: Per-agent decisions, lessons, PxP logs
//! - **Cross-Agent Coordination**: Shared logging for multi-agent workflows
//!
//! # Architecture
//!
//! ```text
//! vDreamTeam Memory System
//! ========================
//!
//!            +-----------------------------------+
//!            |       Layer 0: CONSTITUTIONAL    |
//!            |    (Shared, Immutable Core)      |
//!            |  - Core identity & constraints   |
//!            |  - Hard rules (CONS-001 to 016)  |
//!            |  - Mission, values, boundaries   |
//!            +-----------------------------------+
//!                          |
//!     +--------------------+--------------------+
//!     |                    |                    |
//! +-------+          +-------+            +-------+
//! | vCEO  |          | vCTO  |            | vCMO  |
//! | Layer |          | Layer |            | Layer |
//! +-------+          +-------+            +-------+
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use reasonkit_mem::vdreamteam::{VDreamMemory, RoleId, PxPEntry, Consultation};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load vDreamTeam memory from .agents directory
//!     let mut memory = VDreamMemory::load(".agents").await?;
//!
//!     // Log a PxP consultation
//!     let entry = PxPEntry::new("vcto", "Architecture decision")
//!         .add_consultation(Consultation {
//!             model: "deepseek-v3.2".to_string(),
//!             cli_command: "ollama run deepseek-v3.2:cloud".to_string(),
//!             prompt_summary: "Validate 2-layer architecture".to_string(),
//!             response_summary: "Design validated".to_string(),
//!             confidence: 0.90,
//!         });
//!
//!     memory.log_pxp(entry).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! Requires the `vdreamteam` feature flag in Cargo.toml.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::{MemError, MemResult};

// ============================================================================
// Sub-modules
// ============================================================================

/// MCP Server for vDreamTeam memory queries
pub mod mcp_server;

// Re-export MCP server types
pub use mcp_server::VDreamMCPServer;

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for a role (e.g., "vcto", "vceo", "vcmo")
pub type RoleId = String;

/// Main vDreamTeam memory system
///
/// Provides unified access to constitutional and role-specific memory layers.
#[derive(Debug)]
pub struct VDreamMemory {
    /// Base path for .agents directory
    base_path: PathBuf,
    /// Constitutional layer (shared across all roles)
    constitutional: Arc<RwLock<ConstitutionalMemory>>,
    /// Role-specific memory layers
    roles: HashMap<RoleId, Arc<RwLock<RoleMemory>>>,
    /// Cross-agent coordination log path
    cross_agent_log_path: PathBuf,
}

impl VDreamMemory {
    /// Load vDreamTeam memory from a base directory
    ///
    /// # Arguments
    ///
    /// * `base_path` - Path to the .agents directory
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let memory = VDreamMemory::load(".agents").await?;
    /// ```
    pub async fn load(base_path: impl AsRef<Path>) -> MemResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        let constitutional_path = base_path.join("constitutional");
        let roles_path = base_path.join("roles");
        let cross_agent_log_path = base_path.join("logs").join("cross_agent.ndjson");

        // Load constitutional layer
        let constitutional = ConstitutionalMemory::load(&constitutional_path).await?;

        // Discover and load role-specific memory
        let mut roles = HashMap::new();
        if roles_path.exists() {
            let mut entries = tokio::fs::read_dir(&roles_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read roles directory: {}", e)))?;

            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| MemError::storage(format!("Failed to read role entry: {}", e)))?
            {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(role_id) = path.file_name().and_then(|n| n.to_str()) {
                        let role_memory = RoleMemory::load(role_id, &path).await?;
                        roles.insert(role_id.to_string(), Arc::new(RwLock::new(role_memory)));
                    }
                }
            }
        }

        Ok(Self {
            base_path,
            constitutional: Arc::new(RwLock::new(constitutional)),
            roles,
            cross_agent_log_path,
        })
    }

    /// Create a new empty vDreamTeam memory
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        let base_path = base_path.as_ref().to_path_buf();
        Self {
            cross_agent_log_path: base_path.join("logs").join("cross_agent.ndjson"),
            base_path,
            constitutional: Arc::new(RwLock::new(ConstitutionalMemory::default())),
            roles: HashMap::new(),
        }
    }

    /// Get the constitutional memory (read-only)
    pub async fn constitutional(&self) -> MemResult<ConstitutionalMemory> {
        let guard = self.constitutional.read().await;
        Ok(guard.clone())
    }

    /// Get role-specific memory (read-only)
    pub async fn role(&self, role_id: &str) -> MemResult<Option<RoleMemory>> {
        match self.roles.get(role_id) {
            Some(role) => {
                let guard = role.read().await;
                Ok(Some(guard.clone()))
            }
            None => Ok(None),
        }
    }

    /// Log a PxP consultation entry
    ///
    /// Appends to the role's consults.yaml file.
    pub async fn log_pxp(&mut self, entry: PxPEntry) -> MemResult<()> {
        let role_id = entry.role.clone();

        // Ensure role memory exists
        if !self.roles.contains_key(&role_id) {
            return Err(MemError::storage(format!(
                "Role '{}' not found in memory system",
                role_id
            )));
        }

        // Update role memory
        let role = self.roles.get(&role_id).unwrap();
        {
            let mut guard = role.write().await;
            guard.pxp_log.entries.push(entry.clone());
        }

        // Persist to disk
        self.persist_role_pxp(&role_id).await?;

        // Also append to cross-agent log
        self.append_cross_agent_event(CrossAgentEvent::PxPSession {
            timestamp: entry.timestamp,
            role: role_id,
            models: entry
                .consultations
                .iter()
                .map(|c| c.model.clone())
                .collect(),
            topic: entry.decision_context.clone(),
            result: entry.final_decision.clone().unwrap_or_default(),
        })
        .await?;

        Ok(())
    }

    /// Record a decision made by a role
    pub async fn record_decision(&mut self, decision: Decision) -> MemResult<()> {
        let role_id = decision.role.clone();

        if !self.roles.contains_key(&role_id) {
            return Err(MemError::storage(format!(
                "Role '{}' not found in memory system",
                role_id
            )));
        }

        let role = self.roles.get(&role_id).unwrap();
        {
            let mut guard = role.write().await;
            guard.decisions.entries.push(decision.clone());
        }

        self.persist_role_decisions(&role_id).await?;

        // Log to cross-agent
        self.append_cross_agent_event(CrossAgentEvent::Decision {
            timestamp: decision.timestamp,
            role: role_id,
            decision_id: decision.id.clone(),
            title: decision.title.clone(),
            confidence: decision.confidence,
        })
        .await?;

        Ok(())
    }

    /// Record a lesson learned
    pub async fn record_lesson(&mut self, lesson: Lesson) -> MemResult<()> {
        let role_id = lesson.role.clone();

        if !self.roles.contains_key(&role_id) {
            return Err(MemError::storage(format!(
                "Role '{}' not found in memory system",
                role_id
            )));
        }

        let role = self.roles.get(&role_id).unwrap();
        {
            let mut guard = role.write().await;
            guard.lessons.entries.push(lesson.clone());
        }

        self.persist_role_lessons(&role_id).await?;

        Ok(())
    }

    /// Get all roles in the memory system
    pub fn role_ids(&self) -> Vec<&String> {
        self.roles.keys().collect()
    }

    /// Check a constraint by ID
    pub async fn check_constraint(&self, constraint_id: &str) -> MemResult<Option<Constraint>> {
        let guard = self.constitutional.read().await;
        Ok(guard.constraints.get(constraint_id).cloned())
    }

    /// Get PxP statistics for a role
    pub async fn pxp_stats(&self, role_id: &str) -> MemResult<PxPStats> {
        let role = self
            .roles
            .get(role_id)
            .ok_or_else(|| MemError::storage(format!("Role '{}' not found", role_id)))?;

        let guard = role.read().await;
        let entries = &guard.pxp_log.entries;

        let total_consultations: usize = entries.iter().map(|e| e.consultations.len()).sum();

        let avg_confidence = if !entries.is_empty() {
            entries
                .iter()
                .flat_map(|e| e.consultations.iter().map(|c| c.confidence))
                .sum::<f64>()
                / total_consultations.max(1) as f64
        } else {
            0.0
        };

        let mut model_usage: HashMap<String, usize> = HashMap::new();
        for entry in entries {
            for consultation in &entry.consultations {
                *model_usage.entry(consultation.model.clone()).or_default() += 1;
            }
        }

        Ok(PxPStats {
            total_sessions: entries.len(),
            total_consultations,
            avg_confidence,
            model_usage,
        })
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    async fn persist_role_pxp(&self, role_id: &str) -> MemResult<()> {
        let role = self
            .roles
            .get(role_id)
            .ok_or_else(|| MemError::storage(format!("Role '{}' not found", role_id)))?;

        let guard = role.read().await;
        let path = self
            .base_path
            .join("roles")
            .join(role_id)
            .join("memory")
            .join("consults.yaml");

        let yaml = serde_yaml::to_string(&guard.pxp_log)
            .map_err(|e| MemError::storage(format!("Failed to serialize PxP log: {}", e)))?;

        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .map_err(|e| MemError::storage(format!("Failed to create directory: {}", e)))?;

        tokio::fs::write(&path, yaml)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write PxP log: {}", e)))?;

        Ok(())
    }

    async fn persist_role_decisions(&self, role_id: &str) -> MemResult<()> {
        let role = self
            .roles
            .get(role_id)
            .ok_or_else(|| MemError::storage(format!("Role '{}' not found", role_id)))?;

        let guard = role.read().await;
        let path = self
            .base_path
            .join("roles")
            .join(role_id)
            .join("memory")
            .join("decisions.yaml");

        let yaml = serde_yaml::to_string(&guard.decisions)
            .map_err(|e| MemError::storage(format!("Failed to serialize decisions: {}", e)))?;

        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .map_err(|e| MemError::storage(format!("Failed to create directory: {}", e)))?;

        tokio::fs::write(&path, yaml)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write decisions: {}", e)))?;

        Ok(())
    }

    async fn persist_role_lessons(&self, role_id: &str) -> MemResult<()> {
        let role = self
            .roles
            .get(role_id)
            .ok_or_else(|| MemError::storage(format!("Role '{}' not found", role_id)))?;

        let guard = role.read().await;
        let path = self
            .base_path
            .join("roles")
            .join(role_id)
            .join("memory")
            .join("lessons.yaml");

        let yaml = serde_yaml::to_string(&guard.lessons)
            .map_err(|e| MemError::storage(format!("Failed to serialize lessons: {}", e)))?;

        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .map_err(|e| MemError::storage(format!("Failed to create directory: {}", e)))?;

        tokio::fs::write(&path, yaml)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write lessons: {}", e)))?;

        Ok(())
    }

    async fn append_cross_agent_event(&self, event: CrossAgentEvent) -> MemResult<()> {
        let json = serde_json::to_string(&event)
            .map_err(|e| MemError::storage(format!("Failed to serialize event: {}", e)))?;

        tokio::fs::create_dir_all(self.cross_agent_log_path.parent().unwrap())
            .await
            .map_err(|e| MemError::storage(format!("Failed to create logs directory: {}", e)))?;

        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.cross_agent_log_path)
            .await
            .map_err(|e| MemError::storage(format!("Failed to open cross-agent log: {}", e)))?;

        file.write_all(format!("{}\n", json).as_bytes())
            .await
            .map_err(|e| MemError::storage(format!("Failed to write event: {}", e)))?;

        Ok(())
    }
}

// ============================================================================
// Constitutional Memory (Layer 0)
// ============================================================================

/// Constitutional memory - shared immutable core for all agents
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstitutionalMemory {
    /// Core identity (mission, tagline, philosophy)
    pub identity: Identity,
    /// Hard constraints (CONS-001 to CONS-016)
    pub constraints: HashMap<String, Constraint>,
    /// Quality gates
    pub quality_gates: Vec<QualityGate>,
    /// Boundaries (OSS vs Proprietary)
    pub boundaries: Boundaries,
    /// PxP consultation requirements
    pub consultation: ConsultationConfig,
}

impl ConstitutionalMemory {
    /// Load constitutional memory from a directory
    pub async fn load(path: impl AsRef<Path>) -> MemResult<Self> {
        let path = path.as_ref();

        let mut memory = Self::default();

        // Load identity.yaml
        let identity_path = path.join("identity.yaml");
        if identity_path.exists() {
            let content = tokio::fs::read_to_string(&identity_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read identity.yaml: {}", e)))?;
            memory.identity = serde_yaml::from_str(&content)
                .map_err(|e| MemError::storage(format!("Failed to parse identity.yaml: {}", e)))?;
        }

        // Load constraints.yaml
        let constraints_path = path.join("constraints.yaml");
        if constraints_path.exists() {
            let content = tokio::fs::read_to_string(&constraints_path)
                .await
                .map_err(|e| {
                    MemError::storage(format!("Failed to read constraints.yaml: {}", e))
                })?;
            let wrapper: ConstraintsWrapper = serde_yaml::from_str(&content).map_err(|e| {
                MemError::storage(format!("Failed to parse constraints.yaml: {}", e))
            })?;
            // Populate constraint IDs from HashMap keys
            memory.constraints = wrapper
                .constraints
                .into_iter()
                .map(|(key, mut constraint)| {
                    if constraint.id.is_empty() {
                        constraint.id = key.clone();
                    }
                    (key, constraint)
                })
                .collect();
        }

        // Load quality_gates.yaml
        let gates_path = path.join("quality_gates.yaml");
        if gates_path.exists() {
            let content = tokio::fs::read_to_string(&gates_path).await.map_err(|e| {
                MemError::storage(format!("Failed to read quality_gates.yaml: {}", e))
            })?;
            let wrapper: QualityGatesWrapper = serde_yaml::from_str(&content).map_err(|e| {
                MemError::storage(format!("Failed to parse quality_gates.yaml: {}", e))
            })?;
            memory.quality_gates = wrapper.gates;
        }

        // Load boundaries.yaml
        let boundaries_path = path.join("boundaries.yaml");
        if boundaries_path.exists() {
            let content = tokio::fs::read_to_string(&boundaries_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read boundaries.yaml: {}", e)))?;
            memory.boundaries = serde_yaml::from_str(&content).map_err(|e| {
                MemError::storage(format!("Failed to parse boundaries.yaml: {}", e))
            })?;
        }

        // Load consultation.yaml
        let consultation_path = path.join("consultation.yaml");
        if consultation_path.exists() {
            let content = tokio::fs::read_to_string(&consultation_path)
                .await
                .map_err(|e| {
                    MemError::storage(format!("Failed to read consultation.yaml: {}", e))
                })?;
            memory.consultation = serde_yaml::from_str(&content).map_err(|e| {
                MemError::storage(format!("Failed to parse consultation.yaml: {}", e))
            })?;
        }

        Ok(memory)
    }
}

/// Core identity of the vDreamTeam
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Identity {
    pub version: String,
    pub last_updated: String,
    pub mission: String,
    pub tagline: String,
    pub philosophy: String,
    pub principles: Vec<Principle>,
    pub organization: OrganizationInfo,
}

/// A guiding principle
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Principle {
    pub code: String,
    pub description: String,
    pub enforcement: String,
}

/// Organization information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrganizationInfo {
    pub name: String,
    pub website: String,
    pub target_arr: String,
}

/// A hard constraint
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Constraint {
    /// Constraint ID (populated from HashMap key if not in YAML)
    #[serde(default)]
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub enforcement: String,
    #[serde(default)]
    pub consequence: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ConstraintsWrapper {
    constraints: HashMap<String, Constraint>,
}

/// A quality gate
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityGate {
    #[serde(default)]
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub threshold: String,
    #[serde(default)]
    pub required: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct QualityGatesWrapper {
    gates: Vec<QualityGate>,
}

/// OSS vs Proprietary boundaries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Boundaries {
    #[serde(default)]
    pub oss_projects: Vec<String>,
    #[serde(default)]
    pub proprietary_projects: Vec<String>,
    #[serde(default)]
    pub never_oss: Vec<String>,
    /// Additional fields from YAML (classifications, rules, etc.)
    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml::Value>,
}

/// PxP consultation configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsultationConfig {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub philosophy: ConsultationPhilosophy,
    #[serde(default)]
    pub cli_tools: HashMap<String, CliTool>,
    #[serde(default)]
    pub tiers: HashMap<String, ConsultationTier>,
    /// Additional fields from YAML
    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsultationPhilosophy {
    #[serde(default)]
    pub axiom: String,
    #[serde(default)]
    pub requirement: String,
    #[serde(default)]
    pub minimum_per_session: u32,
    #[serde(default)]
    pub maximum_per_session: u32,
    #[serde(default)]
    pub quality_over_speed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CliTool {
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tier: u8,
    #[serde(default)]
    pub specialty: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsultationTier {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub consultations: String,
    #[serde(default)]
    pub time: String,
    #[serde(default)]
    pub use_when: Vec<String>,
    #[serde(default)]
    pub models: Vec<String>,
}

// ============================================================================
// Role-Specific Memory (Layer 2)
// ============================================================================

/// Role-specific memory layer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoleMemory {
    /// Role identifier
    pub role_id: RoleId,
    /// Role identity and configuration
    pub identity: RoleIdentity,
    /// Decision log
    pub decisions: DecisionLog,
    /// Lessons learned
    pub lessons: LessonsLog,
    /// PxP consultation log
    pub pxp_log: PxPLog,
    /// Skills registry
    pub skills: SkillsRegistry,
}

impl RoleMemory {
    /// Load role memory from a directory
    pub async fn load(role_id: &str, path: impl AsRef<Path>) -> MemResult<Self> {
        let path = path.as_ref();
        let memory_path = path.join("memory");

        let mut role = Self {
            role_id: role_id.to_string(),
            ..Default::default()
        };

        // Load identity.yaml
        let identity_path = path.join("identity.yaml");
        if identity_path.exists() {
            let content = tokio::fs::read_to_string(&identity_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read role identity: {}", e)))?;
            role.identity = serde_yaml::from_str(&content)
                .map_err(|e| MemError::storage(format!("Failed to parse role identity: {}", e)))?;
        }

        // Load decisions.yaml
        let decisions_path = memory_path.join("decisions.yaml");
        if decisions_path.exists() {
            let content = tokio::fs::read_to_string(&decisions_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read decisions: {}", e)))?;
            role.decisions = serde_yaml::from_str(&content)
                .map_err(|e| MemError::storage(format!("Failed to parse decisions: {}", e)))?;
        }

        // Load lessons.yaml
        let lessons_path = memory_path.join("lessons.yaml");
        if lessons_path.exists() {
            let content = tokio::fs::read_to_string(&lessons_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read lessons: {}", e)))?;
            role.lessons = serde_yaml::from_str(&content)
                .map_err(|e| MemError::storage(format!("Failed to parse lessons: {}", e)))?;
        }

        // Load consults.yaml (PxP log)
        let pxp_path = memory_path.join("consults.yaml");
        if pxp_path.exists() {
            let content = tokio::fs::read_to_string(&pxp_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read PxP log: {}", e)))?;
            role.pxp_log = serde_yaml::from_str(&content)
                .map_err(|e| MemError::storage(format!("Failed to parse PxP log: {}", e)))?;
        }

        // Load skills.yaml
        let skills_path = memory_path.join("skills.yaml");
        if skills_path.exists() {
            let content = tokio::fs::read_to_string(&skills_path)
                .await
                .map_err(|e| MemError::storage(format!("Failed to read skills: {}", e)))?;
            role.skills = serde_yaml::from_str(&content)
                .map_err(|e| MemError::storage(format!("Failed to parse skills: {}", e)))?;
        }

        Ok(role)
    }
}

/// Role identity and configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoleIdentity {
    pub version: String,
    pub role_id: String,
    pub full_name: String,
    pub tier: u8,
    pub category: String,
    pub model: ModelConfig,
    pub cli_tools: CliToolsConfig,
    pub responsibilities: RoleResponsibilities,
    pub pxp_requirements: HashMap<String, u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    pub primary: String,
    pub fallback: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CliToolsConfig {
    pub primary: String,
    pub alternatives: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoleResponsibilities {
    pub core: Vec<String>,
    pub governance: Vec<String>,
    pub coordination: Vec<String>,
}

// ============================================================================
// Decision Tracking
// ============================================================================

/// Log of decisions made by a role
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionLog {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub role_id: String,
    #[serde(default)]
    pub entries: Vec<Decision>,
}

/// A decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub role: RoleId,
    pub title: String,
    pub context: String,
    pub decision: String,
    pub rationale: String,
    pub alternatives_considered: Vec<String>,
    pub pxp_consultations: Vec<String>,
    pub confidence: f64,
    pub status: DecisionStatus,
}

impl Decision {
    pub fn new(role: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: format!(
                "DEC-{}-{}",
                Utc::now().format("%Y%m%d%H%M%S"),
                uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
            ),
            timestamp: Utc::now(),
            role: role.into(),
            title: title.into(),
            context: String::new(),
            decision: String::new(),
            rationale: String::new(),
            alternatives_considered: Vec::new(),
            pxp_consultations: Vec::new(),
            confidence: 0.0,
            status: DecisionStatus::Pending,
        }
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }

    pub fn with_decision(mut self, decision: impl Into<String>) -> Self {
        self.decision = decision.into();
        self
    }

    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = rationale.into();
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn finalize(mut self) -> Self {
        self.status = DecisionStatus::Finalized;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecisionStatus {
    Pending,
    Finalized,
    Superseded,
    Reverted,
}

// ============================================================================
// Lessons Learned
// ============================================================================

/// Log of lessons learned by a role
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LessonsLog {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub role_id: String,
    #[serde(default)]
    pub entries: Vec<Lesson>,
}

/// A lesson learned record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub role: RoleId,
    pub title: String,
    pub category: LessonCategory,
    pub severity: LessonSeverity,
    pub context: String,
    pub mistake: String,
    pub lesson: String,
    pub corrective_actions: Vec<String>,
    pub prevention: String,
}

impl Lesson {
    pub fn new(role: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: format!(
                "LES-{}-{}",
                Utc::now().format("%Y%m%d%H%M%S"),
                uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
            ),
            timestamp: Utc::now(),
            role: role.into(),
            title: title.into(),
            category: LessonCategory::Process,
            severity: LessonSeverity::Medium,
            context: String::new(),
            mistake: String::new(),
            lesson: String::new(),
            corrective_actions: Vec::new(),
            prevention: String::new(),
        }
    }

    pub fn with_category(mut self, category: LessonCategory) -> Self {
        self.category = category;
        self
    }

    pub fn with_severity(mut self, severity: LessonSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }

    pub fn with_mistake(mut self, mistake: impl Into<String>) -> Self {
        self.mistake = mistake.into();
        self
    }

    pub fn with_lesson(mut self, lesson: impl Into<String>) -> Self {
        self.lesson = lesson.into();
        self
    }

    pub fn with_prevention(mut self, prevention: impl Into<String>) -> Self {
        self.prevention = prevention.into();
        self
    }

    pub fn add_corrective_action(mut self, action: impl Into<String>) -> Self {
        self.corrective_actions.push(action.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LessonCategory {
    Governance,
    Architecture,
    Process,
    Tooling,
    Communication,
    Security,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LessonSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// ============================================================================
// PxP Consultation Logging
// ============================================================================

/// Log of PxP (Prompt-in-Prompt) consultations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PxPLog {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub entries: Vec<PxPEntry>,
}

/// A PxP consultation session entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PxPEntry {
    #[serde(default)]
    pub id: String,
    #[serde(default = "Utc::now")]
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// Role ID (populated from parent log if not in YAML)
    #[serde(default)]
    pub role: RoleId,
    #[serde(default)]
    pub decision_context: String,
    #[serde(default)]
    pub consultations: Vec<Consultation>,
    #[serde(default)]
    pub triangulation_result: Option<String>,
    #[serde(default)]
    pub final_decision: Option<String>,
}

impl PxPEntry {
    pub fn new(role: impl Into<String>, decision_context: impl Into<String>) -> Self {
        Self {
            id: format!("PXP-{}", Utc::now().format("%Y%m%d%H%M%S%f")),
            timestamp: Utc::now(),
            session_id: None,
            role: role.into(),
            decision_context: decision_context.into(),
            consultations: Vec::new(),
            triangulation_result: None,
            final_decision: None,
        }
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn add_consultation(mut self, consultation: Consultation) -> Self {
        self.consultations.push(consultation);
        self
    }

    pub fn with_triangulation(mut self, result: impl Into<String>) -> Self {
        self.triangulation_result = Some(result.into());
        self
    }

    pub fn with_final_decision(mut self, decision: impl Into<String>) -> Self {
        self.final_decision = Some(decision.into());
        self
    }
}

/// A single consultation with an AI model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Consultation {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub cli_command: String,
    #[serde(default)]
    pub prompt_summary: String,
    #[serde(default)]
    pub response_summary: String,
    #[serde(default)]
    pub confidence: f64,
}

// ============================================================================
// Skills Registry
// ============================================================================

/// Registry of skills available to a role
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillsRegistry {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub role_id: String,
    #[serde(default)]
    pub skills: Vec<Skill>,
    /// Additional fields from YAML (cli_tools, etc.)
    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml::Value>,
}

/// A skill available to a role
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Skill {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub skill_type: SkillType,
    #[serde(default)]
    pub invocation: String,
    #[serde(default)]
    pub proficiency: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillType {
    #[default]
    Cli,
    Mcp,
    Internal,
    External,
}

// ============================================================================
// Cross-Agent Coordination
// ============================================================================

/// Events in the cross-agent coordination log
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CrossAgentEvent {
    /// System initialization
    SystemInit {
        timestamp: DateTime<Utc>,
        message: String,
        version: String,
    },
    /// PxP consultation session
    PxPSession {
        timestamp: DateTime<Utc>,
        role: RoleId,
        models: Vec<String>,
        topic: String,
        result: String,
    },
    /// Decision made
    Decision {
        timestamp: DateTime<Utc>,
        role: RoleId,
        decision_id: String,
        title: String,
        confidence: f64,
    },
    /// Role handoff
    Handoff {
        timestamp: DateTime<Utc>,
        from_role: RoleId,
        to_role: RoleId,
        context: String,
    },
    /// Escalation
    Escalation {
        timestamp: DateTime<Utc>,
        from_role: RoleId,
        to_role: RoleId,
        reason: String,
    },
}

// ============================================================================
// Statistics
// ============================================================================

/// PxP consultation statistics for a role
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PxPStats {
    pub total_sessions: usize,
    pub total_consultations: usize,
    pub avg_confidence: f64,
    pub model_usage: HashMap<String, usize>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pxp_entry_creation() {
        let entry = PxPEntry::new("vcto", "Test decision")
            .with_session("test-session-123")
            .add_consultation(Consultation {
                model: "deepseek-v3.2".to_string(),
                cli_command: "ollama run deepseek-v3.2:cloud".to_string(),
                prompt_summary: "Test prompt".to_string(),
                response_summary: "Test response".to_string(),
                confidence: 0.85,
            })
            .with_triangulation("3 models agree")
            .with_final_decision("Proceed with plan A");

        assert_eq!(entry.role, "vcto");
        assert_eq!(entry.consultations.len(), 1);
        assert!(entry.final_decision.is_some());
    }

    #[test]
    fn test_decision_creation() {
        let decision = Decision::new("vcto", "Architecture change")
            .with_context("Need to optimize storage layer")
            .with_decision("Use dual-layer storage")
            .with_rationale("Better performance for hot data")
            .with_confidence(0.92)
            .finalize();

        assert_eq!(decision.role, "vcto");
        assert_eq!(decision.status, DecisionStatus::Finalized);
        assert!(decision.id.starts_with("DEC-"));
    }

    #[test]
    fn test_lesson_creation() {
        let lesson = Lesson::new("vcto", "Quality over Speed")
            .with_category(LessonCategory::Process)
            .with_severity(LessonSeverity::High)
            .with_context("User feedback")
            .with_mistake("Rushing to conclusions")
            .with_lesson("Take more time, use ThinkTools")
            .with_prevention("Apply ThinkTools before concluding")
            .add_corrective_action("Updated consultation.yaml");

        assert_eq!(lesson.role, "vcto");
        assert_eq!(lesson.category, LessonCategory::Process);
        assert_eq!(lesson.severity, LessonSeverity::High);
        assert_eq!(lesson.corrective_actions.len(), 1);
    }

    #[test]
    fn test_cross_agent_event_serialization() {
        let event = CrossAgentEvent::Decision {
            timestamp: Utc::now(),
            role: "vcto".to_string(),
            decision_id: "DEC-001".to_string(),
            title: "Test decision".to_string(),
            confidence: 0.9,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("DECISION"));
        assert!(json.contains("vcto"));
    }

    #[test]
    fn test_vdream_memory_new() {
        let memory = VDreamMemory::new(".agents");
        assert!(memory.role_ids().is_empty());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper: Create a minimal valid .agents directory structure for testing
    async fn setup_test_agents_dir(base: &Path) -> MemResult<()> {
        // Constitutional layer
        let constitutional = base.join("constitutional");
        tokio::fs::create_dir_all(&constitutional)
            .await
            .map_err(|e| {
                MemError::storage(format!("Failed to create constitutional dir: {}", e))
            })?;

        // identity.yaml
        let identity = r#"
version: "1.0.0"
last_updated: "2026-01-03"
mission: "Test mission"
tagline: "Test tagline"
philosophy: "Test philosophy"
principles: []
organization:
  name: "Test Org"
  website: "https://test.org"
  target_arr: "$100K"
"#;
        tokio::fs::write(constitutional.join("identity.yaml"), identity)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write identity.yaml: {}", e)))?;

        // constraints.yaml
        let constraints = r#"
constraints:
  CONS-001:
    id: "CONS-001"
    name: "Test Constraint"
    description: "A test constraint"
    enforcement: "HARD"
    consequence: "Reject"
"#;
        tokio::fs::write(constitutional.join("constraints.yaml"), constraints)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write constraints.yaml: {}", e)))?;

        // quality_gates.yaml
        let gates = r#"
gates:
  - id: "GATE-001"
    name: "Build"
    command: "cargo build"
    threshold: "Exit 0"
    required: true
"#;
        tokio::fs::write(constitutional.join("quality_gates.yaml"), gates)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write quality_gates.yaml: {}", e)))?;

        // boundaries.yaml
        let boundaries = r#"
oss_projects:
  - reasonkit-core
proprietary_projects:
  - reasonkit-pro
never_oss:
  - rk-research
"#;
        tokio::fs::write(constitutional.join("boundaries.yaml"), boundaries)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write boundaries.yaml: {}", e)))?;

        // consultation.yaml
        let consultation = r#"
version: "4.1.0"
philosophy:
  axiom: "Multiple AI perspectives improve outcomes"
  requirement: "Always consult other models"
  minimum_per_session: 2
  maximum_per_session: 15
  quality_over_speed: true
cli_tools: {}
tiers: {}
"#;
        tokio::fs::write(constitutional.join("consultation.yaml"), consultation)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write consultation.yaml: {}", e)))?;

        // Role: vcto
        let vcto_path = base.join("roles").join("vcto");
        let vcto_memory = vcto_path.join("memory");
        tokio::fs::create_dir_all(&vcto_memory)
            .await
            .map_err(|e| MemError::storage(format!("Failed to create vcto/memory dir: {}", e)))?;

        // vcto/identity.yaml
        let vcto_identity = r#"
version: "1.0.0"
role_id: "vcto"
full_name: "Virtual Chief Technology Officer"
tier: 1
category: "executive"
model:
  primary: "claude-opus-4-5"
  fallback: []
cli_tools:
  primary: "claude -p"
  alternatives: []
responsibilities:
  core:
    - "Architecture decisions"
  governance:
    - "Technical standards"
  coordination:
    - "Team coordination"
pxp_requirements: {}
"#;
        tokio::fs::write(vcto_path.join("identity.yaml"), vcto_identity)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write vcto/identity.yaml: {}", e)))?;

        // vcto/memory/decisions.yaml
        let decisions = r#"
version: "1.0.0"
role_id: "vcto"
entries: []
"#;
        tokio::fs::write(vcto_memory.join("decisions.yaml"), decisions)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write decisions.yaml: {}", e)))?;

        // vcto/memory/lessons.yaml
        let lessons = r#"
version: "1.0.0"
role_id: "vcto"
entries: []
"#;
        tokio::fs::write(vcto_memory.join("lessons.yaml"), lessons)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write lessons.yaml: {}", e)))?;

        // vcto/memory/consults.yaml
        let consults = r#"
version: "1.0.0"
entries: []
"#;
        tokio::fs::write(vcto_memory.join("consults.yaml"), consults)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write consults.yaml: {}", e)))?;

        // vcto/memory/skills.yaml
        let skills = r#"
version: "1.0.0"
role_id: "vcto"
skills: []
"#;
        tokio::fs::write(vcto_memory.join("skills.yaml"), skills)
            .await
            .map_err(|e| MemError::storage(format!("Failed to write skills.yaml: {}", e)))?;

        // Logs directory
        tokio::fs::create_dir_all(base.join("logs"))
            .await
            .map_err(|e| MemError::storage(format!("Failed to create logs dir: {}", e)))?;

        Ok(())
    }

    // ========================================================================
    // Integration Test: Load Valid Directory
    // ========================================================================

    #[tokio::test]
    async fn test_load_valid_directory() {
        // Setup temp directory with valid structure
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Load memory
        let memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Verify constitutional layer loaded
        let constitutional = memory
            .constitutional()
            .await
            .expect("Failed to get constitutional");
        assert_eq!(constitutional.identity.mission, "Test mission");
        assert!(constitutional.constraints.contains_key("CONS-001"));
        assert_eq!(constitutional.quality_gates.len(), 1);
        assert_eq!(constitutional.boundaries.oss_projects.len(), 1);
        assert_eq!(constitutional.consultation.version, "4.1.0");

        // Verify role loaded
        assert_eq!(memory.role_ids().len(), 1);
        assert!(memory.role_ids().contains(&&"vcto".to_string()));

        let role = memory.role("vcto").await.expect("Failed to get role");
        assert!(role.is_some());
        let role = role.unwrap();
        assert_eq!(role.role_id, "vcto");
        assert_eq!(role.identity.full_name, "Virtual Chief Technology Officer");
    }

    // ========================================================================
    // Integration Test: Log PxP Persistence
    // ========================================================================

    #[tokio::test]
    async fn test_log_pxp_persistence() {
        // Setup temp directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Load memory
        let mut memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Create and log a PxP entry
        let entry = PxPEntry::new("vcto", "Test architecture decision")
            .with_session("test-session-001")
            .add_consultation(Consultation {
                model: "deepseek-v3.2".to_string(),
                cli_command: "ollama run deepseek-v3.2:cloud".to_string(),
                prompt_summary: "Validate test approach".to_string(),
                response_summary: "Approach validated".to_string(),
                confidence: 0.88,
            })
            .add_consultation(Consultation {
                model: "claude-opus-4.5".to_string(),
                cli_command: "claude -p".to_string(),
                prompt_summary: "Secondary validation".to_string(),
                response_summary: "Confirmed alignment".to_string(),
                confidence: 0.92,
            })
            .with_triangulation("2/2 models agree")
            .with_final_decision("Proceed with implementation");

        memory.log_pxp(entry).await.expect("Failed to log PxP");

        // Verify file was persisted
        let consults_path = base_path
            .join("roles")
            .join("vcto")
            .join("memory")
            .join("consults.yaml");
        assert!(
            consults_path.exists(),
            "consults.yaml should exist after logging"
        );

        let content = fs::read_to_string(&consults_path).expect("Failed to read consults.yaml");
        assert!(
            content.contains("deepseek-v3.2"),
            "Should contain model name"
        );
        assert!(
            content.contains("Test architecture decision"),
            "Should contain decision context"
        );
        assert!(
            content.contains("2/2 models agree"),
            "Should contain triangulation result"
        );

        // Verify cross-agent log
        let cross_agent_path = base_path.join("logs").join("cross_agent.ndjson");
        assert!(cross_agent_path.exists(), "cross_agent.ndjson should exist");

        let cross_content =
            fs::read_to_string(&cross_agent_path).expect("Failed to read cross_agent");
        // The event type is serialized via serde with tag="event" and SCREAMING_SNAKE_CASE
        // Check for known content that must be in the log
        assert!(
            cross_content.contains("vcto") && cross_content.contains("event"),
            "Cross-agent log should contain role and event data. Got: {}",
            cross_content
        );

        // Reload and verify in-memory state
        let reloaded = VDreamMemory::load(base_path)
            .await
            .expect("Failed to reload VDreamMemory");

        let role = reloaded
            .role("vcto")
            .await
            .expect("Failed to get role")
            .unwrap();
        assert_eq!(role.pxp_log.entries.len(), 1);
        assert_eq!(role.pxp_log.entries[0].consultations.len(), 2);
        assert_eq!(
            role.pxp_log.entries[0].final_decision,
            Some("Proceed with implementation".to_string())
        );
    }

    // ========================================================================
    // Integration Test: Record Decision Storage
    // ========================================================================

    #[tokio::test]
    async fn test_record_decision_storage() {
        // Setup temp directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Load memory
        let mut memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Create and record a decision
        let decision = Decision::new("vcto", "Dual-Layer Storage Architecture")
            .with_context("Need optimized storage for hot and cold data")
            .with_decision("Implement 2-layer storage with hot/cold separation")
            .with_rationale("Provides optimal performance for frequently accessed data while maintaining cost efficiency for archival")
            .with_confidence(0.91)
            .finalize();

        memory
            .record_decision(decision)
            .await
            .expect("Failed to record decision");

        // Verify file was persisted
        let decisions_path = base_path
            .join("roles")
            .join("vcto")
            .join("memory")
            .join("decisions.yaml");
        assert!(
            decisions_path.exists(),
            "decisions.yaml should exist after recording"
        );

        let content = fs::read_to_string(&decisions_path).expect("Failed to read decisions.yaml");
        assert!(
            content.contains("Dual-Layer Storage Architecture"),
            "Should contain title"
        );
        assert!(
            content.contains("2-layer storage"),
            "Should contain decision"
        );
        assert!(content.contains("0.91"), "Should contain confidence");
        assert!(content.contains("finalized"), "Should be finalized");

        // Verify cross-agent log
        let cross_agent_path = base_path.join("logs").join("cross_agent.ndjson");
        let cross_content =
            fs::read_to_string(&cross_agent_path).expect("Failed to read cross_agent");
        assert!(
            cross_content.contains("DECISION"),
            "Should contain Decision event"
        );

        // Reload and verify persistence
        let reloaded = VDreamMemory::load(base_path)
            .await
            .expect("Failed to reload VDreamMemory");

        let role = reloaded
            .role("vcto")
            .await
            .expect("Failed to get role")
            .unwrap();
        assert_eq!(role.decisions.entries.len(), 1);
        assert_eq!(
            role.decisions.entries[0].title,
            "Dual-Layer Storage Architecture"
        );
        assert_eq!(role.decisions.entries[0].status, DecisionStatus::Finalized);
    }

    // ========================================================================
    // Integration Test: Record Lesson Storage
    // ========================================================================

    #[tokio::test]
    async fn test_record_lesson_storage() {
        // Setup temp directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Load memory
        let mut memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Create and record a lesson
        let lesson = Lesson::new("vcto", "Quality Over Speed")
            .with_category(LessonCategory::Process)
            .with_severity(LessonSeverity::High)
            .with_context("User feedback indicated rushed conclusions")
            .with_mistake("Skipping ThinkTools application in favor of quick responses")
            .with_lesson("Always apply relevant ThinkTools before finalizing decisions")
            .with_prevention("Enforce minimum PxP consultation count per session")
            .add_corrective_action("Updated consultation.yaml with stricter minimums")
            .add_corrective_action("Added quality gate for ThinkTools usage");

        memory
            .record_lesson(lesson)
            .await
            .expect("Failed to record lesson");

        // Verify file was persisted
        let lessons_path = base_path
            .join("roles")
            .join("vcto")
            .join("memory")
            .join("lessons.yaml");
        assert!(
            lessons_path.exists(),
            "lessons.yaml should exist after recording"
        );

        let content = fs::read_to_string(&lessons_path).expect("Failed to read lessons.yaml");
        assert!(
            content.contains("Quality Over Speed"),
            "Should contain title"
        );
        assert!(
            content.contains("ThinkTools"),
            "Should contain lesson content"
        );
        assert!(content.contains("PROCESS"), "Should contain category");
        assert!(content.contains("HIGH"), "Should contain severity");
        assert!(
            content.contains("consultation.yaml"),
            "Should contain corrective action"
        );

        // Reload and verify persistence
        let reloaded = VDreamMemory::load(base_path)
            .await
            .expect("Failed to reload VDreamMemory");

        let role = reloaded
            .role("vcto")
            .await
            .expect("Failed to get role")
            .unwrap();
        assert_eq!(role.lessons.entries.len(), 1);
        assert_eq!(role.lessons.entries[0].title, "Quality Over Speed");
        assert_eq!(role.lessons.entries[0].category, LessonCategory::Process);
        assert_eq!(role.lessons.entries[0].severity, LessonSeverity::High);
        assert_eq!(role.lessons.entries[0].corrective_actions.len(), 2);
    }

    // ========================================================================
    // Integration Test: Load Nonexistent Directory (Graceful Fallback)
    // ========================================================================

    #[tokio::test]
    async fn test_load_nonexistent_directory() {
        // VDreamMemory::load gracefully handles missing directories by
        // creating default empty structures. This is by design to allow
        // bootstrapping new projects.
        //
        // To truly test failure, we need a path that causes a hard error
        // (e.g., permission denied or invalid path). For now, we verify
        // that loading a nonexistent path returns an empty/default memory.
        let result = VDreamMemory::load("/nonexistent/path/that/should/not/exist").await;

        // The implementation may either:
        // 1. Fail with an error (preferred for strict validation)
        // 2. Return empty memory (graceful fallback)
        match result {
            Ok(memory) => {
                // If it succeeds, it should be empty (no roles loaded)
                assert!(
                    memory.role_ids().is_empty(),
                    "Should have no roles for nonexistent path"
                );
            }
            Err(err) => {
                // If it fails, error should be meaningful
                let err_string = err.to_string();
                assert!(
                    err_string.contains("Failed to read")
                        || err_string.contains("No such file")
                        || err_string.contains("not found")
                        || err_string.contains("directory"),
                    "Error should indicate path/directory issue: {}",
                    err_string
                );
            }
        }
    }

    // ========================================================================
    // Integration Test: Check Constraint
    // ========================================================================

    #[tokio::test]
    async fn test_check_constraint() {
        // Setup temp directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Load memory
        let memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Check existing constraint
        let constraint = memory
            .check_constraint("CONS-001")
            .await
            .expect("Failed to check constraint");
        assert!(constraint.is_some());
        let constraint = constraint.unwrap();
        assert_eq!(constraint.name, "Test Constraint");
        assert_eq!(constraint.enforcement, "HARD");

        // Check non-existing constraint
        let none = memory
            .check_constraint("CONS-999")
            .await
            .expect("Failed to check non-existing constraint");
        assert!(none.is_none());
    }

    // ========================================================================
    // Integration Test: PxP Statistics
    // ========================================================================

    #[tokio::test]
    async fn test_pxp_stats() {
        // Setup temp directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Load memory
        let mut memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Log multiple PxP entries
        for i in 1..=3 {
            let entry = PxPEntry::new("vcto", format!("Decision {}", i))
                .add_consultation(Consultation {
                    model: "deepseek-v3.2".to_string(),
                    cli_command: "ollama run deepseek-v3.2:cloud".to_string(),
                    prompt_summary: "Prompt".to_string(),
                    response_summary: "Response".to_string(),
                    confidence: 0.8 + (i as f64 * 0.05),
                })
                .add_consultation(Consultation {
                    model: "claude-opus-4.5".to_string(),
                    cli_command: "claude -p".to_string(),
                    prompt_summary: "Prompt 2".to_string(),
                    response_summary: "Response 2".to_string(),
                    confidence: 0.85 + (i as f64 * 0.03),
                });

            memory.log_pxp(entry).await.expect("Failed to log PxP");
        }

        // Get stats
        let stats = memory.pxp_stats("vcto").await.expect("Failed to get stats");

        assert_eq!(stats.total_sessions, 3, "Should have 3 sessions");
        assert_eq!(
            stats.total_consultations, 6,
            "Should have 6 total consultations"
        );
        assert!(
            stats.avg_confidence > 0.8,
            "Average confidence should be > 0.8"
        );
        assert_eq!(stats.model_usage.get("deepseek-v3.2"), Some(&3));
        assert_eq!(stats.model_usage.get("claude-opus-4.5"), Some(&3));
    }

    // ========================================================================
    // Integration Test: Role Not Found Error
    // ========================================================================

    #[tokio::test]
    async fn test_role_not_found_error() {
        // Setup temp directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Load memory
        let mut memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Try to log PxP for non-existent role
        let entry = PxPEntry::new("nonexistent_role", "Test");
        let result = memory.log_pxp(entry).await;

        assert!(result.is_err(), "Should fail for non-existent role");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found"),
            "Error should indicate role not found: {}",
            err
        );
    }

    // ========================================================================
    // Integration Test: Multiple Roles
    // ========================================================================

    #[tokio::test]
    async fn test_multiple_roles() {
        // Setup temp directory with multiple roles
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_path = temp_dir.path();

        // Setup base structure
        setup_test_agents_dir(base_path)
            .await
            .expect("Failed to setup test agents dir");

        // Add a second role (vceo)
        let vceo_path = base_path.join("roles").join("vceo");
        let vceo_memory = vceo_path.join("memory");
        tokio::fs::create_dir_all(&vceo_memory)
            .await
            .expect("Failed to create vceo dir");

        let vceo_identity = r#"
version: "1.0.0"
role_id: "vceo"
full_name: "Virtual Chief Executive Officer"
tier: 1
category: "executive"
model:
  primary: "claude-opus-4-5"
  fallback: []
cli_tools:
  primary: "claude -p"
  alternatives: []
responsibilities:
  core:
    - "Strategic decisions"
  governance:
    - "Overall governance"
  coordination:
    - "Executive coordination"
pxp_requirements: {}
"#;
        tokio::fs::write(vceo_path.join("identity.yaml"), vceo_identity)
            .await
            .expect("Failed to write vceo identity");

        for file in &[
            "decisions.yaml",
            "lessons.yaml",
            "consults.yaml",
            "skills.yaml",
        ] {
            let content = match *file {
                "consults.yaml" => "version: \"1.0.0\"\nentries: []\n",
                "skills.yaml" => "version: \"1.0.0\"\nrole_id: \"vceo\"\nskills: []\n",
                _ => "version: \"1.0.0\"\nrole_id: \"vceo\"\nentries: []\n",
            };
            tokio::fs::write(vceo_memory.join(file), content)
                .await
                .expect("Failed to write vceo memory file");
        }

        // Load memory
        let memory = VDreamMemory::load(base_path)
            .await
            .expect("Failed to load VDreamMemory");

        // Verify both roles loaded
        let role_ids = memory.role_ids();
        assert_eq!(role_ids.len(), 2, "Should have 2 roles");
        assert!(role_ids.iter().any(|r| *r == "vcto"));
        assert!(role_ids.iter().any(|r| *r == "vceo"));

        // Verify each role has correct identity
        let vcto = memory
            .role("vcto")
            .await
            .expect("Failed to get vcto")
            .unwrap();
        assert_eq!(vcto.identity.full_name, "Virtual Chief Technology Officer");

        let vceo = memory
            .role("vceo")
            .await
            .expect("Failed to get vceo")
            .unwrap();
        assert_eq!(vceo.identity.full_name, "Virtual Chief Executive Officer");
    }
}
