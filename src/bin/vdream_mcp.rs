//! vDreamTeam MCP Server Binary
//!
//! Runs the vDreamTeam memory MCP server over stdio.
//!
//! # Usage
//!
//! ```bash
//! # Run directly
//! cargo run --bin vdream_mcp --features vdreamteam
//!
//! # Or install and run
//! cargo install --path . --features vdreamteam
//! vdream_mcp
//! ```
//!
//! # MCP Configuration (claude_desktop_config.json)
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "vdreamteam": {
//!       "command": "vdream_mcp",
//!       "args": [],
//!       "env": {
//!         "VDREAM_AGENTS_PATH": "/path/to/.agents"
//!       }
//!     }
//!   }
//! }
//! ```

use std::env;
use std::path::PathBuf;

#[cfg(feature = "vdreamteam")]
use reasonkit_mem::vdreamteam::VDreamMCPServer;

#[cfg(feature = "vdreamteam")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get agents path from environment or use default
    let agents_path = env::var("VDREAM_AGENTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default: ~/RK-PROJECT/.agents or ./.agents
            if let Some(home) = dirs::home_dir() {
                let rk_agents = home.join("RK-PROJECT").join(".agents");
                if rk_agents.exists() {
                    return rk_agents;
                }
            }
            PathBuf::from(".agents")
        });

    eprintln!(
        "vDreamTeam MCP Server starting with agents path: {}",
        agents_path.display()
    );

    let mut server = VDreamMCPServer::new(agents_path);
    server.run_stdio().await?;

    Ok(())
}

#[cfg(not(feature = "vdreamteam"))]
fn main() {
    eprintln!("Error: vdreamteam feature is not enabled.");
    eprintln!("Run with: cargo run --bin vdream_mcp --features vdreamteam");
    std::process::exit(1);
}
