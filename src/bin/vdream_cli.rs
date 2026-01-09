//! vDreamTeam CLI
//!
//! Command-line interface for vDreamTeam memory operations.
//!
//! # Usage
//!
//! ```bash
//! # Build
//! cargo build --bin vdream_cli --features vdreamteam
//!
//! # Run commands
//! vdream_cli stats                           # Show memory statistics
//! vdream_cli query constitutional identity   # Query constitutional layer
//! vdream_cli query role vcto decisions       # Query role decisions
//! vdream_cli check CONS-001                  # Check constraint
//! vdream_cli log-pxp vcto "Decision context" # Log PxP event
//! ```

use std::env;
use std::path::PathBuf;

#[cfg(feature = "vdreamteam")]
use reasonkit_mem::vdreamteam::{Consultation, PxPEntry, VDreamMemory};

#[cfg(feature = "vdreamteam")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    let agents_path = env::var("VDREAM_AGENTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if let Some(home) = dirs::home_dir() {
                let rk_agents = home.join("RK-PROJECT").join(".agents");
                if rk_agents.exists() {
                    return rk_agents;
                }
            }
            PathBuf::from(".agents")
        });

    let command = args[1].as_str();

    match command {
        "stats" => cmd_stats(&agents_path).await?,
        "query" => {
            if args.len() < 4 {
                eprintln!(
                    "Usage: vdream_cli query <constitutional|role> <section|role_id> [query_type]"
                );
                return Ok(());
            }
            cmd_query(
                &agents_path,
                &args[2],
                &args[3],
                args.get(4).map(|s| s.as_str()),
            )
            .await?;
        }
        "check" => {
            if args.len() < 3 {
                eprintln!("Usage: vdream_cli check <constraint_id> [proposed_action]");
                return Ok(());
            }
            cmd_check(&agents_path, &args[2], args.get(3).map(|s| s.as_str())).await?;
        }
        "log-pxp" => {
            if args.len() < 4 {
                eprintln!("Usage: vdream_cli log-pxp <role_id> <decision_context> [model]");
                return Ok(());
            }
            cmd_log_pxp(
                &agents_path,
                &args[2],
                &args[3],
                args.get(4).map(|s| s.as_str()),
            )
            .await?;
        }
        "roles" => cmd_roles(&agents_path).await?,
        "constraints" => cmd_constraints(&agents_path).await?,
        "help" | "-h" | "--help" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
        }
    }

    Ok(())
}

#[cfg(feature = "vdreamteam")]
fn print_usage() {
    println!(
        r#"vDreamTeam CLI - AI Agent Memory System

USAGE:
    vdream_cli <COMMAND> [OPTIONS]

COMMANDS:
    stats                                   Show memory statistics
    query constitutional <section>          Query constitutional layer
                                           Sections: identity, constraints, boundaries, quality_gates, consultation, all
    query role <role_id> [query_type]       Query role-specific memory
                                           Query types: decisions, lessons, consultations, skills, all
    check <constraint_id> [action]          Check if action violates constraint
    log-pxp <role_id> <context> [model]     Log a PxP consultation event
    roles                                   List available roles
    constraints                             List all constraints
    help                                    Show this help

ENVIRONMENT:
    VDREAM_AGENTS_PATH                      Path to .agents directory (default: ~/RK-PROJECT/.agents)

EXAMPLES:
    vdream_cli stats
    vdream_cli query constitutional identity
    vdream_cli query role vcto decisions
    vdream_cli check CONS-001 "Using Node.js for MCP server"
    vdream_cli log-pxp vcto "Architecture decision" deepseek-v3.2
    vdream_cli roles
"#
    );
}

#[cfg(feature = "vdreamteam")]
async fn cmd_stats(agents_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let memory = VDreamMemory::load(agents_path).await?;

    println!("vDreamTeam Memory Statistics");
    println!("============================");
    println!("Agents Path: {}", agents_path.display());
    println!();

    // Constitutional layer
    match memory.constitutional().await {
        Ok(constitutional) => {
            println!("Constitutional Layer: LOADED");
            println!("  Identity: {}", constitutional.identity.organization.name);
            println!(
                "  Constraints: {} defined",
                constitutional.constraints.len()
            );
            println!(
                "  Boundaries: {} OSS, {} proprietary, {} never-OSS",
                constitutional.boundaries.oss_projects.len(),
                constitutional.boundaries.proprietary_projects.len(),
                constitutional.boundaries.never_oss.len()
            );
            println!(
                "  Quality Gates: {} defined",
                constitutional.quality_gates.len()
            );
        }
        Err(e) => {
            println!("Constitutional Layer: ERROR - {}", e);
        }
    }
    println!();

    // Role stats
    let role_ids = memory.role_ids();
    println!("Roles: {} found", role_ids.len());
    for role_id in &role_ids {
        if let Ok(stats) = memory.pxp_stats(role_id).await {
            println!(
                "  {} - {} consultations, models: {:?}",
                role_id,
                stats.total_consultations,
                stats.model_usage.keys().collect::<Vec<_>>()
            );
        }
    }

    Ok(())
}

#[cfg(feature = "vdreamteam")]
async fn cmd_query(
    agents_path: &PathBuf,
    target: &str,
    section_or_role: &str,
    query_type: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let memory = VDreamMemory::load(agents_path).await?;

    match target {
        "constitutional" => {
            let constitutional = memory.constitutional().await?;
            match section_or_role {
                "identity" => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&constitutional.identity)?
                    );
                }
                "constraints" => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&constitutional.constraints)?
                    );
                }
                "boundaries" => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&constitutional.boundaries)?
                    );
                }
                "quality_gates" => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&constitutional.quality_gates)?
                    );
                }
                "consultation" => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&constitutional.consultation)?
                    );
                }
                "all" => {
                    println!("{}", serde_json::to_string_pretty(&constitutional)?);
                }
                _ => {
                    eprintln!("Unknown constitutional section: {}", section_or_role);
                    eprintln!(
                        "Available: identity, constraints, boundaries, quality_gates, consultation, all"
                    );
                }
            }
        }
        "role" => {
            let role_id = section_or_role;
            let qtype = query_type.unwrap_or("all");

            match memory.role(role_id).await? {
                Some(role_memory) => match qtype {
                    "decisions" => {
                        println!("{}", serde_json::to_string_pretty(&role_memory.decisions)?);
                    }
                    "lessons" => {
                        println!("{}", serde_json::to_string_pretty(&role_memory.lessons)?);
                    }
                    "consultations" => {
                        println!("{}", serde_json::to_string_pretty(&role_memory.pxp_log)?);
                    }
                    "skills" => {
                        println!("{}", serde_json::to_string_pretty(&role_memory.skills)?);
                    }
                    "all" => {
                        println!("{}", serde_json::to_string_pretty(&role_memory)?);
                    }
                    _ => {
                        eprintln!("Unknown query type: {}", qtype);
                        eprintln!("Available: decisions, lessons, consultations, skills, all");
                    }
                },
                None => {
                    eprintln!("Role not found: {}", role_id);
                    eprintln!("Available roles: {:?}", memory.role_ids());
                }
            }
        }
        _ => {
            eprintln!("Unknown target: {}", target);
            eprintln!("Available: constitutional, role");
        }
    }

    Ok(())
}

#[cfg(feature = "vdreamteam")]
async fn cmd_check(
    agents_path: &PathBuf,
    constraint_id: &str,
    proposed_action: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let memory = VDreamMemory::load(agents_path).await?;

    match memory.check_constraint(constraint_id).await? {
        Some(constraint) => {
            println!("Constraint: {}", constraint.id);
            println!("Description: {}", constraint.description);
            println!("Enforcement: {}", constraint.enforcement);
            if !constraint.consequence.is_empty() {
                println!("Consequence: {}", constraint.consequence);
            }

            if let Some(action) = proposed_action {
                println!();
                println!("Proposed Action: {}", action);
                println!("---");
                println!(
                    "NOTE: This constraint has {} enforcement.",
                    constraint.enforcement
                );
                println!("Evaluate the proposed action against the constraint description.");
            }
        }
        None => {
            eprintln!("Constraint not found: {}", constraint_id);
            let constitutional = memory.constitutional().await?;
            let available: Vec<_> = constitutional.constraints.keys().collect();
            eprintln!("Available constraints: {:?}", available);
        }
    }

    Ok(())
}

#[cfg(feature = "vdreamteam")]
async fn cmd_log_pxp(
    agents_path: &PathBuf,
    role_id: &str,
    decision_context: &str,
    model: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = VDreamMemory::load(agents_path).await?;

    let model_name = model.unwrap_or("claude-opus-4.5");

    let entry = PxPEntry::new(role_id, decision_context).add_consultation(Consultation {
        model: model_name.to_string(),
        cli_command: format!("vdream_cli log-pxp {} ...", role_id),
        prompt_summary: format!("CLI PxP log: {}", decision_context),
        response_summary: "Logged via CLI".to_string(),
        confidence: 0.8,
    });

    memory.log_pxp(entry).await?;

    println!("PxP entry logged successfully!");
    println!("  Role: {}", role_id);
    println!("  Context: {}", decision_context);
    println!("  Model: {}", model_name);

    Ok(())
}

#[cfg(feature = "vdreamteam")]
async fn cmd_roles(agents_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let memory = VDreamMemory::load(agents_path).await?;

    println!("Available Roles:");
    println!("================");
    for role_id in memory.role_ids() {
        if let Ok(Some(role)) = memory.role(role_id).await {
            let decision_count = role.decisions.entries.len();
            let lesson_count = role.lessons.entries.len();
            let pxp_count = role.pxp_log.entries.len();
            println!(
                "  {} - {} decisions, {} lessons, {} consultations",
                role_id, decision_count, lesson_count, pxp_count
            );
        } else {
            println!("  {}", role_id);
        }
    }

    Ok(())
}

#[cfg(feature = "vdreamteam")]
async fn cmd_constraints(agents_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let memory = VDreamMemory::load(agents_path).await?;
    let constitutional = memory.constitutional().await?;

    println!("Constraints:");
    println!("============");
    for (id, constraint) in &constitutional.constraints {
        println!("  {} [{}]", id, constraint.enforcement);
        println!("    {}", constraint.description);
        if !constraint.consequence.is_empty() {
            println!("    Consequence: {}", constraint.consequence);
        }
        println!();
    }

    Ok(())
}

#[cfg(not(feature = "vdreamteam"))]
fn main() {
    eprintln!("Error: vdreamteam feature is not enabled.");
    eprintln!("Run with: cargo run --bin vdream_cli --features vdreamteam");
    std::process::exit(1);
}
