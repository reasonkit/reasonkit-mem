//! Integration tests for vDreamTeam AI Agent Memory System
//!
//! These tests validate the complete VDreamMemory functionality including:
//! - Loading from valid .agents directory structure
//! - PxP consultation logging and persistence
//! - Decision recording and storage
//! - Lesson learning capture
//! - Cross-agent event coordination
//! - Error handling for nonexistent directories
//!
//! # Test Requirements
//!
//! Requires the `vdreamteam` feature flag:
//! ```bash
//! cargo test --features vdreamteam vdreamteam_integration_tests
//! ```

#![cfg(feature = "vdreamteam")]

use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

use reasonkit_mem::vdreamteam::{
    Consultation, Decision, Lesson, LessonCategory, LessonSeverity, PxPEntry, VDreamMemory,
};

// ============================================================================
// TEST HELPERS
// ============================================================================

/// Create a minimal valid .agents directory structure for testing
async fn create_test_agents_directory(temp_dir: &TempDir) -> PathBuf {
    let base_path = temp_dir.path().join(".agents");

    // Create constitutional layer
    let constitutional_path = base_path.join("constitutional");
    fs::create_dir_all(&constitutional_path).await.unwrap();

    // Create identity.yaml
    let identity_yaml = r#"
version: "1.0.0"
last_updated: "2026-01-03"
mission: "Make AI reasoning structured, auditable, and reliable."
tagline: "Turn Prompts into Protocols"
philosophy: "Designed, Not Dreamed"
principles:
  - code: "PRIN-001"
    description: "Structure beats intelligence"
    enforcement: "HARD"
organization:
  name: "ReasonKit"
  website: "https://reasonkit.sh"
  target_arr: "$719K"
"#;
    fs::write(constitutional_path.join("identity.yaml"), identity_yaml)
        .await
        .unwrap();

    // Create constraints.yaml
    let constraints_yaml = r#"
constraints:
  CONS-001:
    id: "CONS-001"
    name: "No Node.js MCP Servers"
    description: "All MCP servers must be Rust-first"
    enforcement: "HARD"
    consequence: "Reject at code review"
  CONS-002:
    id: "CONS-002"
    name: "Rust for Performance"
    description: "Performance-critical paths must be Rust"
    enforcement: "HARD"
    consequence: "No exceptions"
"#;
    fs::write(
        constitutional_path.join("constraints.yaml"),
        constraints_yaml,
    )
    .await
    .unwrap();

    // Create quality_gates.yaml
    let quality_gates_yaml = r#"
gates:
  - id: "GATE-001"
    name: "Build"
    command: "cargo build --release"
    threshold: "Exit 0"
    required: true
  - id: "GATE-002"
    name: "Clippy"
    command: "cargo clippy -- -D warnings"
    threshold: "0 errors"
    required: true
"#;
    fs::write(
        constitutional_path.join("quality_gates.yaml"),
        quality_gates_yaml,
    )
    .await
    .unwrap();

    // Create boundaries.yaml
    let boundaries_yaml = r#"
oss_projects:
  - "reasonkit-core"
  - "reasonkit-mem"
  - "reasonkit-web"
proprietary_projects:
  - "reasonkit-pro"
never_oss:
  - "rk-research"
  - "rk-startup"
"#;
    fs::write(constitutional_path.join("boundaries.yaml"), boundaries_yaml)
        .await
        .unwrap();

    // Create consultation.yaml
    let consultation_yaml = r#"
version: "4.1.0"
philosophy:
  axiom: "Multiple AI perspectives improve outcomes"
  requirement: "ALWAYS consult other models"
  minimum_per_session: 2
  maximum_per_session: 15
  quality_over_speed: true
cli_tools:
  claude:
    command: "claude -p"
    model: "claude-opus-4-5-20251101"
    tier: 1
    specialty: "Architecture"
  deepseek:
    command: "ollama run deepseek-v3.2:cloud"
    model: "deepseek-v3.2"
    tier: 2
    specialty: "Reasoning"
tiers:
  quick:
    name: "Quick Consultation"
    consultations: "2-3"
    time: "< 5 min"
    use_when:
      - "Simple decisions"
      - "Confidence > 80%"
    models:
      - "deepseek-v3.2"
      - "devstral-2"
  thorough:
    name: "Thorough Consultation"
    consultations: "5-10"
    time: "15-30 min"
    use_when:
      - "Architecture decisions"
      - "Security-critical"
    models:
      - "claude-opus-4.5"
      - "deepseek-v3.2"
      - "mistral-large-3"
"#;
    fs::write(
        constitutional_path.join("consultation.yaml"),
        consultation_yaml,
    )
    .await
    .unwrap();

    // Create roles directory with vcto role
    let vcto_path = base_path.join("roles").join("vcto");
    let vcto_memory_path = vcto_path.join("memory");
    fs::create_dir_all(&vcto_memory_path).await.unwrap();

    // Create vcto identity.yaml
    let vcto_identity_yaml = r#"
version: "1.0.0"
role_id: "vcto"
full_name: "Virtual Chief Technology Officer"
tier: 1
category: "C-SUITE"
model:
  primary: "claude-opus-4-5-20251101"
  fallback:
    - "gpt-5.2-pro"
    - "gemini-3-pro"
cli_tools:
  primary: "claude -p"
  alternatives:
    - "gemini -p"
    - "codex"
responsibilities:
  core:
    - "Technical architecture decisions"
    - "Rust-first enforcement"
    - "Quality gate oversight"
  governance:
    - "MCP server architecture"
    - "Performance standards"
  coordination:
    - "Lead engineer coordination"
    - "Cross-crate integration"
pxp_requirements:
  architecture: 5
  breaking_change: 5
  security: 4
  routine: 2
"#;
    fs::write(vcto_path.join("identity.yaml"), vcto_identity_yaml)
        .await
        .unwrap();

    // Create empty memory files
    let decisions_yaml = r#"
version: "1.0.0"
role_id: "vcto"
entries: []
"#;
    fs::write(vcto_memory_path.join("decisions.yaml"), decisions_yaml)
        .await
        .unwrap();

    let lessons_yaml = r#"
version: "1.0.0"
role_id: "vcto"
entries: []
"#;
    fs::write(vcto_memory_path.join("lessons.yaml"), lessons_yaml)
        .await
        .unwrap();

    let consults_yaml = r#"
version: "1.0.0"
role_id: "vcto"
entries: []
"#;
    fs::write(vcto_memory_path.join("consults.yaml"), consults_yaml)
        .await
        .unwrap();

    let skills_yaml = r#"
version: "1.0.0"
role_id: "vcto"
skills: []
"#;
    fs::write(vcto_memory_path.join("skills.yaml"), skills_yaml)
        .await
        .unwrap();

    // Create logs directory
    fs::create_dir_all(base_path.join("logs")).await.unwrap();

    base_path
}

// ============================================================================
// TEST: Load Valid Directory
// ============================================================================

/// Test loading VDreamMemory from a valid .agents directory structure
#[tokio::test]
async fn test_load_valid_directory() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    // Load the memory system
    let memory = VDreamMemory::load(&agents_path).await;
    assert!(memory.is_ok(), "Should load from valid directory");

    let memory = memory.unwrap();

    // Verify constitutional layer loaded
    let constitutional = memory.constitutional().await.unwrap();
    assert_eq!(
        constitutional.identity.mission,
        "Make AI reasoning structured, auditable, and reliable."
    );
    assert_eq!(
        constitutional.identity.tagline,
        "Turn Prompts into Protocols"
    );
    assert_eq!(constitutional.identity.philosophy, "Designed, Not Dreamed");

    // Verify constraints loaded
    assert!(constitutional.constraints.contains_key("CONS-001"));
    assert!(constitutional.constraints.contains_key("CONS-002"));
    let cons001 = constitutional.constraints.get("CONS-001").unwrap();
    assert_eq!(cons001.name, "No Node.js MCP Servers");

    // Verify quality gates loaded
    assert_eq!(constitutional.quality_gates.len(), 2);
    assert!(constitutional
        .quality_gates
        .iter()
        .any(|g| g.id == "GATE-001"));

    // Verify boundaries loaded
    assert!(constitutional
        .boundaries
        .oss_projects
        .contains(&"reasonkit-core".to_string()));
    assert!(constitutional
        .boundaries
        .never_oss
        .contains(&"rk-research".to_string()));

    // Verify consultation config loaded
    assert_eq!(constitutional.consultation.version, "4.1.0");
    assert_eq!(
        constitutional.consultation.philosophy.minimum_per_session,
        2
    );
    assert_eq!(
        constitutional.consultation.philosophy.maximum_per_session,
        15
    );

    // Verify role loaded
    let role_ids = memory.role_ids();
    assert!(role_ids.iter().any(|r| r.as_str() == "vcto"));

    let vcto = memory.role("vcto").await.unwrap();
    assert!(vcto.is_some());
    let vcto = vcto.unwrap();
    assert_eq!(vcto.identity.full_name, "Virtual Chief Technology Officer");
    assert_eq!(vcto.identity.tier, 1);
    assert_eq!(vcto.identity.category, "C-SUITE");
}

// ============================================================================
// TEST: PxP Logging and Persistence
// ============================================================================

/// Test logging PxP consultation entries with persistence
#[tokio::test]
async fn test_log_pxp_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    // Load memory and log a PxP entry
    let mut memory = VDreamMemory::load(&agents_path).await.unwrap();

    let entry = PxPEntry::new("vcto", "Architecture decision for dual-layer memory")
        .add_consultation(Consultation {
            model: "deepseek-v3.2".to_string(),
            cli_command: "ollama run deepseek-v3.2:cloud".to_string(),
            prompt_summary: "Validate dual-layer memory architecture".to_string(),
            response_summary: "Architecture validated with hot/cold tier design".to_string(),
            confidence: 0.92,
        })
        .add_consultation(Consultation {
            model: "mistral-large-3".to_string(),
            cli_command: "ollama run mistral-large-3:675b-cloud".to_string(),
            prompt_summary: "Review WAL implementation strategy".to_string(),
            response_summary: "WAL approach recommended for durability".to_string(),
            confidence: 0.88,
        })
        .with_final_decision("Use dual-layer memory with WAL for crash recovery".to_string());

    let result = memory.log_pxp(entry).await;
    assert!(result.is_ok(), "Should log PxP entry successfully");

    // Verify the entry was persisted
    let consults_path = agents_path
        .join("roles")
        .join("vcto")
        .join("memory")
        .join("consults.yaml");
    let content = fs::read_to_string(&consults_path).await.unwrap();
    assert!(content.contains("Architecture decision for dual-layer memory"));
    assert!(content.contains("deepseek-v3.2"));
    assert!(content.contains("mistral-large-3"));

    // Verify cross-agent log was updated
    let cross_agent_log = agents_path.join("logs").join("cross_agent.ndjson");
    let log_content = fs::read_to_string(&cross_agent_log)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "Failed to read cross_agent.ndjson: {} at path {:?}",
                e, cross_agent_log
            );
        });
    // Note: serde rename_all="SCREAMING_SNAKE_CASE" converts PxPSession to PX_P_SESSION
    assert!(
        log_content.contains("PX_P_SESSION"),
        "Log should contain PX_P_SESSION event. Got: {}",
        log_content
    );
    assert!(log_content.contains("vcto"));

    // Verify PxP stats
    let stats = memory.pxp_stats("vcto").await.unwrap();
    assert_eq!(stats.total_sessions, 1);
    assert_eq!(stats.total_consultations, 2);
    assert!(stats.avg_confidence > 0.85);
    assert!(stats.model_usage.contains_key("deepseek-v3.2"));
    assert!(stats.model_usage.contains_key("mistral-large-3"));
}

// ============================================================================
// TEST: Decision Recording and Storage
// ============================================================================

/// Test recording decisions with full metadata
#[tokio::test]
async fn test_record_decision_storage() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    let mut memory = VDreamMemory::load(&agents_path).await.unwrap();

    let decision = Decision::new("vcto", "Adopt 2-layer memory architecture")
        .with_context("Need to implement long-term memory for AI agents with different access patterns")
        .with_decision("Use hot/cold tier architecture with WAL for durability")
        .with_rationale("Hot tier for frequently accessed data, cold tier for historical, WAL for crash recovery")
        .with_confidence(0.92)
        .finalize();

    let result = memory.record_decision(decision).await;
    assert!(result.is_ok(), "Should record decision successfully");

    // Verify persistence
    let decisions_path = agents_path
        .join("roles")
        .join("vcto")
        .join("memory")
        .join("decisions.yaml");
    let content = fs::read_to_string(&decisions_path).await.unwrap();
    assert!(content.contains("Adopt 2-layer memory architecture"));
    assert!(content.contains("hot/cold tier architecture"));
    assert!(content.contains("finalized"));

    // Verify cross-agent log contains decision event
    let cross_agent_log = agents_path.join("logs").join("cross_agent.ndjson");
    let log_content = fs::read_to_string(&cross_agent_log).await.unwrap();
    assert!(
        log_content.contains("DECISION"),
        "Log should contain DECISION event"
    );
    assert!(log_content.contains("Adopt 2-layer memory architecture"));
}

// ============================================================================
// TEST: Lesson Recording and Storage
// ============================================================================

/// Test recording lessons learned with full metadata
#[tokio::test]
async fn test_record_lesson_storage() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    let mut memory = VDreamMemory::load(&agents_path).await.unwrap();

    let lesson = Lesson::new("vcto", "Always validate PxP model responses")
        .with_category(LessonCategory::Process)
        .with_severity(LessonSeverity::High)
        .with_context("During architecture review, a model returned outdated syntax")
        .with_mistake("Trusted model response without verification")
        .with_lesson("Always triangulate with 3+ sources for critical decisions")
        .with_prevention("Add mandatory verification step to PxP protocol")
        .add_corrective_action("Update consultation.yaml with verification requirements")
        .add_corrective_action("Add ProofGuard validation to decision workflow");

    let result = memory.record_lesson(lesson).await;
    assert!(result.is_ok(), "Should record lesson successfully");

    // Verify persistence
    let lessons_path = agents_path
        .join("roles")
        .join("vcto")
        .join("memory")
        .join("lessons.yaml");
    let content = fs::read_to_string(&lessons_path).await.unwrap();
    assert!(content.contains("Always validate PxP model responses"));
    assert!(content.contains("triangulate with 3+ sources"));
    assert!(content.contains("ProofGuard validation"));
}

// ============================================================================
// TEST: Nonexistent Directory Error Handling
// ============================================================================

/// Test loading from nonexistent directory returns appropriate error
#[tokio::test]
async fn test_load_nonexistent_directory() {
    let nonexistent_path = PathBuf::from("/nonexistent/path/to/.agents");
    let result = VDreamMemory::load(&nonexistent_path).await;

    // Should succeed but with empty/default values since files don't exist
    // The load function creates defaults for missing files
    assert!(
        result.is_ok(),
        "Should handle nonexistent directory gracefully"
    );

    let memory = result.unwrap();
    let role_ids = memory.role_ids();
    assert!(
        role_ids.is_empty(),
        "Should have no roles for nonexistent directory"
    );
}

// ============================================================================
// TEST: Constraint Checking
// ============================================================================

/// Test constraint checking functionality
#[tokio::test]
async fn test_check_constraint() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    let memory = VDreamMemory::load(&agents_path).await.unwrap();

    // Check existing constraint
    let cons001 = memory.check_constraint("CONS-001").await.unwrap();
    assert!(cons001.is_some());
    let constraint = cons001.unwrap();
    assert_eq!(constraint.id, "CONS-001");
    assert_eq!(constraint.name, "No Node.js MCP Servers");
    assert_eq!(constraint.enforcement, "HARD");

    // Check non-existing constraint
    let nonexistent = memory.check_constraint("CONS-999").await.unwrap();
    assert!(nonexistent.is_none());
}

// ============================================================================
// TEST: Multiple PxP Sessions
// ============================================================================

/// Test logging multiple PxP sessions and verifying stats
#[tokio::test]
async fn test_multiple_pxp_sessions() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    let mut memory = VDreamMemory::load(&agents_path).await.unwrap();

    // Log 5 PxP sessions with varying consultations
    for i in 1..=5 {
        let mut entry = PxPEntry::new("vcto", format!("Decision #{}", i));

        // Add 1-3 consultations per session
        for j in 1..=(i % 3 + 1) {
            entry = entry.add_consultation(Consultation {
                model: format!("model-{}", j),
                cli_command: format!("ollama run model-{}", j),
                prompt_summary: format!("Prompt {} for decision {}", j, i),
                response_summary: format!("Response validated"),
                confidence: 0.80 + (j as f64 * 0.05),
            });
        }

        memory.log_pxp(entry).await.unwrap();
    }

    // Verify stats
    let stats = memory.pxp_stats("vcto").await.unwrap();
    assert_eq!(stats.total_sessions, 5);
    assert!(stats.total_consultations >= 5); // At least 1 per session
    assert!(stats.avg_confidence > 0.80);
}

// ============================================================================
// TEST: Role Not Found Error
// ============================================================================

/// Test error handling when logging to nonexistent role
#[tokio::test]
async fn test_log_to_nonexistent_role() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    let mut memory = VDreamMemory::load(&agents_path).await.unwrap();

    let entry = PxPEntry::new("nonexistent_role", "Test decision");
    let result = memory.log_pxp(entry).await;

    assert!(result.is_err(), "Should fail for nonexistent role");
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"));
}

// ============================================================================
// TEST: New Empty Memory
// ============================================================================

/// Test creating new empty VDreamMemory
#[tokio::test]
async fn test_new_empty_memory() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path().join(".agents");

    let memory = VDreamMemory::new(&base_path);

    // Should have default constitutional memory
    let constitutional = memory.constitutional().await.unwrap();
    assert!(constitutional.identity.mission.is_empty());

    // Should have no roles
    let role_ids = memory.role_ids();
    assert!(role_ids.is_empty());
}

// ============================================================================
// TEST: Cross-Agent Event Logging
// ============================================================================

/// Test that all operations log to cross-agent log
#[tokio::test]
async fn test_cross_agent_event_logging() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    let mut memory = VDreamMemory::load(&agents_path).await.unwrap();

    // Log PxP
    let pxp_entry = PxPEntry::new("vcto", "PxP Test").add_consultation(Consultation {
        model: "test-model".to_string(),
        cli_command: "test".to_string(),
        prompt_summary: "test".to_string(),
        response_summary: "test".to_string(),
        confidence: 0.9,
    });
    memory.log_pxp(pxp_entry).await.unwrap();

    // Record decision
    let decision = Decision::new("vcto", "Decision Test")
        .with_decision("Test decision")
        .with_confidence(0.85)
        .finalize();
    memory.record_decision(decision).await.unwrap();

    // Check cross-agent log contains both events
    let cross_agent_log = agents_path.join("logs").join("cross_agent.ndjson");
    let log_content = fs::read_to_string(&cross_agent_log).await.unwrap();

    let lines: Vec<&str> = log_content.lines().collect();
    assert!(lines.len() >= 2, "Should have at least 2 events logged");

    // Verify event types (serde rename_all="SCREAMING_SNAKE_CASE" - PxPSession becomes PX_P_SESSION)
    assert!(
        log_content.contains("PX_P_SESSION"),
        "Log should contain PX_P_SESSION event"
    );
    assert!(
        log_content.contains("DECISION"),
        "Log should contain DECISION event"
    );
}

// ============================================================================
// TEST: Reload Persisted Data
// ============================================================================

/// Test that persisted data can be reloaded
#[tokio::test]
async fn test_reload_persisted_data() {
    let temp_dir = TempDir::new().unwrap();
    let agents_path = create_test_agents_directory(&temp_dir).await;

    // First session: log data
    {
        let mut memory = VDreamMemory::load(&agents_path).await.unwrap();

        let entry = PxPEntry::new("vcto", "Persistent Decision").add_consultation(Consultation {
            model: "persistent-model".to_string(),
            cli_command: "test".to_string(),
            prompt_summary: "Should persist".to_string(),
            response_summary: "Confirmed".to_string(),
            confidence: 0.95,
        });
        memory.log_pxp(entry).await.unwrap();
    }

    // Second session: reload and verify
    {
        let memory = VDreamMemory::load(&agents_path).await.unwrap();
        let vcto = memory.role("vcto").await.unwrap().unwrap();

        assert_eq!(vcto.pxp_log.entries.len(), 1);
        assert_eq!(
            vcto.pxp_log.entries[0].decision_context,
            "Persistent Decision"
        );
        assert_eq!(
            vcto.pxp_log.entries[0].consultations[0].model,
            "persistent-model"
        );
    }
}
