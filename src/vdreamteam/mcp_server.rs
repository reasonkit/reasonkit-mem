//! vDreamTeam MCP Server
//!
//! Model Context Protocol server for vDreamTeam memory queries.
//! Provides tools for AI agents to query and interact with the
//! constitutional and role-specific memory layers.
//!
//! # Tools
//!
//! - `vdream_query_constitutional`: Query constitutional layer (identity, constraints)
//! - `vdream_query_role`: Query role-specific memory (decisions, lessons, PxP logs)
//! - `vdream_log_pxp`: Log a PxP consultation event
//! - `vdream_log_decision`: Record an agent decision
//! - `vdream_get_stats`: Get memory statistics
//! - `vdream_check_constraint`: Check if an action violates constraints
//!
//! # Transport
//!
//! Uses JSON-RPC 2.0 over stdio (standard MCP transport).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{Consultation, Decision, MemResult, PxPEntry, VDreamMemory};

/// MCP Server version
pub const MCP_VERSION: &str = "2024-11-05";

/// Server info for MCP protocol
#[derive(Debug, Clone, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Server capabilities
#[derive(Debug, Clone, Serialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolsCapability {
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourcesCapability {
    pub subscribe: bool,
    pub list_changed: bool,
}

/// Tool definition for MCP
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl ToolResult {
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            content: Some(vec![ContentBlock {
                content_type: "text".to_string(),
                text: text.into(),
            }]),
            is_error: None,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: Some(vec![ContentBlock {
                content_type: "text".to_string(),
                text: text.into(),
            }]),
            is_error: Some(true),
        }
    }
}

/// JSON-RPC Request
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC Response
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// vDreamTeam MCP Server
pub struct VDreamMCPServer {
    memory: Arc<RwLock<Option<VDreamMemory>>>,
    agents_path: PathBuf,
    initialized: bool,
}

impl VDreamMCPServer {
    /// Create new MCP server instance
    pub fn new(agents_path: impl Into<PathBuf>) -> Self {
        Self {
            memory: Arc::new(RwLock::new(None)),
            agents_path: agents_path.into(),
            initialized: false,
        }
    }

    /// Initialize the server (load memory)
    pub async fn initialize(&mut self) -> MemResult<()> {
        let memory = VDreamMemory::load(&self.agents_path).await?;
        *self.memory.write().await = Some(memory);
        self.initialized = true;
        Ok(())
    }

    /// Get server info
    pub fn get_server_info(&self) -> ServerInfo {
        ServerInfo {
            name: "vDreamTeam Memory Server".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: Some(
                "MCP server for vDreamTeam AI agent memory - constitutional and role-specific layers"
                    .to_string(),
            ),
        }
    }

    /// Get server capabilities
    pub fn get_capabilities(&self) -> ServerCapabilities {
        ServerCapabilities {
            tools: ToolsCapability {
                list_changed: false,
            },
            resources: Some(ResourcesCapability {
                subscribe: false,
                list_changed: false,
            }),
        }
    }

    /// List available tools
    pub fn list_tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "vdream_query_constitutional".to_string(),
                description: "Query the constitutional layer (identity, constraints, boundaries, quality gates)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["identity", "constraints", "boundaries", "quality_gates", "consultation", "all"],
                            "description": "Which constitutional section to query"
                        }
                    },
                    "required": ["section"]
                }),
            },
            ToolDefinition {
                name: "vdream_query_role".to_string(),
                description: "Query role-specific memory (decisions, lessons, consultations, skills)".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "role_id": {
                            "type": "string",
                            "description": "Role ID (e.g., 'vceo', 'vcto', 'lead_core')"
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["decisions", "lessons", "consultations", "skills", "all"],
                            "description": "What type of memory to query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum entries to return (default: 10)"
                        }
                    },
                    "required": ["role_id", "query_type"]
                }),
            },
            ToolDefinition {
                name: "vdream_log_pxp".to_string(),
                description: "Log a PxP (Prompt x Prompt) consultation event".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "role_id": {
                            "type": "string",
                            "description": "Role ID that performed the consultation"
                        },
                        "decision_context": {
                            "type": "string",
                            "description": "Context for the decision being made"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model consulted (e.g., 'deepseek-v3.2', 'claude-opus-4.5')"
                        },
                        "prompt_summary": {
                            "type": "string",
                            "description": "Brief summary of the prompt"
                        },
                        "response_summary": {
                            "type": "string",
                            "description": "Brief summary of the response"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level 0.0-1.0"
                        }
                    },
                    "required": ["role_id", "decision_context", "model", "prompt_summary", "response_summary", "confidence"]
                }),
            },
            ToolDefinition {
                name: "vdream_log_decision".to_string(),
                description: "Record an agent decision".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "role_id": {
                            "type": "string",
                            "description": "Role ID making the decision"
                        },
                        "title": {
                            "type": "string",
                            "description": "Decision title"
                        },
                        "context": {
                            "type": "string",
                            "description": "Decision context"
                        },
                        "alternatives": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Alternatives considered"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Reasoning for the decision"
                        },
                        "outcome": {
                            "type": "string",
                            "description": "Outcome or result"
                        }
                    },
                    "required": ["role_id", "title", "context", "rationale"]
                }),
            },
            ToolDefinition {
                name: "vdream_get_stats".to_string(),
                description: "Get vDreamTeam memory statistics".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            },
            ToolDefinition {
                name: "vdream_check_constraint".to_string(),
                description: "Check if an action would violate a constraint".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "constraint_id": {
                            "type": "string",
                            "description": "Constraint ID to check (e.g., 'CONS-001')"
                        },
                        "proposed_action": {
                            "type": "string",
                            "description": "Description of the proposed action"
                        }
                    },
                    "required": ["constraint_id", "proposed_action"]
                }),
            },
        ]
    }

    /// Handle a tool call
    pub async fn call_tool(&self, name: &str, arguments: Value) -> ToolResult {
        match name {
            "vdream_query_constitutional" => self.handle_query_constitutional(arguments).await,
            "vdream_query_role" => self.handle_query_role(arguments).await,
            "vdream_log_pxp" => self.handle_log_pxp(arguments).await,
            "vdream_log_decision" => self.handle_log_decision(arguments).await,
            "vdream_get_stats" => self.handle_get_stats().await,
            "vdream_check_constraint" => self.handle_check_constraint(arguments).await,
            _ => ToolResult::error(format!("Unknown tool: {}", name)),
        }
    }

    async fn handle_query_constitutional(&self, args: Value) -> ToolResult {
        let section = args
            .get("section")
            .and_then(|v| v.as_str())
            .unwrap_or("all");
        let memory_guard = self.memory.read().await;
        let memory = match memory_guard.as_ref() {
            Some(m) => m,
            None => return ToolResult::error("Memory not initialized"),
        };

        match memory.constitutional().await {
            Ok(constitutional) => match section {
                "identity" => ToolResult::success(
                    serde_json::to_string_pretty(&constitutional.identity).unwrap_or_default(),
                ),
                "constraints" => ToolResult::success(
                    serde_json::to_string_pretty(&constitutional.constraints).unwrap_or_default(),
                ),
                "boundaries" => ToolResult::success(
                    serde_json::to_string_pretty(&constitutional.boundaries).unwrap_or_default(),
                ),
                "quality_gates" => ToolResult::success(
                    serde_json::to_string_pretty(&constitutional.quality_gates).unwrap_or_default(),
                ),
                "consultation" => ToolResult::success(
                    serde_json::to_string_pretty(&constitutional.consultation).unwrap_or_default(),
                ),
                "all" => ToolResult::success(
                    serde_json::to_string_pretty(&constitutional).unwrap_or_default(),
                ),
                _ => ToolResult::error(format!("Unknown section: {}", section)),
            },
            Err(e) => ToolResult::error(format!("Failed to read constitutional: {}", e)),
        }
    }

    async fn handle_query_role(&self, args: Value) -> ToolResult {
        let role_id = match args.get("role_id").and_then(|v| v.as_str()) {
            Some(r) => r,
            None => return ToolResult::error("role_id is required"),
        };

        let query_type = args
            .get("query_type")
            .and_then(|v| v.as_str())
            .unwrap_or("all");
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let memory_guard = self.memory.read().await;
        let memory = match memory_guard.as_ref() {
            Some(m) => m,
            None => return ToolResult::error("Memory not initialized"),
        };

        match memory.role(role_id).await {
            Ok(Some(role_memory)) => match query_type {
                "decisions" => {
                    let decisions: Vec<_> =
                        role_memory.decisions.entries.iter().take(limit).collect();
                    ToolResult::success(
                        serde_json::to_string_pretty(&decisions).unwrap_or_default(),
                    )
                }
                "lessons" => {
                    let lessons: Vec<_> = role_memory.lessons.entries.iter().take(limit).collect();
                    ToolResult::success(serde_json::to_string_pretty(&lessons).unwrap_or_default())
                }
                "consultations" => {
                    let pxp: Vec<_> = role_memory.pxp_log.entries.iter().take(limit).collect();
                    ToolResult::success(serde_json::to_string_pretty(&pxp).unwrap_or_default())
                }
                "skills" => ToolResult::success(
                    serde_json::to_string_pretty(&role_memory.skills).unwrap_or_default(),
                ),
                "all" => ToolResult::success(
                    serde_json::to_string_pretty(&role_memory).unwrap_or_default(),
                ),
                _ => ToolResult::error(format!("Unknown query_type: {}", query_type)),
            },
            Ok(None) => {
                let role_ids = memory.role_ids();
                ToolResult::error(format!(
                    "Role not found: {}. Available: {:?}",
                    role_id, role_ids
                ))
            }
            Err(e) => ToolResult::error(format!("Failed to query role: {}", e)),
        }
    }

    async fn handle_log_pxp(&self, args: Value) -> ToolResult {
        let role_id = match args.get("role_id").and_then(|v| v.as_str()) {
            Some(r) => r,
            None => return ToolResult::error("role_id is required"),
        };

        let decision_context = match args.get("decision_context").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::error("decision_context is required"),
        };

        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let prompt_summary = args
            .get("prompt_summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let response_summary = args
            .get("response_summary")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let confidence = args
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let entry = PxPEntry::new(role_id, decision_context).add_consultation(Consultation {
            model: model.to_string(),
            cli_command: "claude -p (via MCP)".to_string(),
            prompt_summary: prompt_summary.to_string(),
            response_summary: response_summary.to_string(),
            confidence,
        });

        let mut memory_guard = self.memory.write().await;
        match memory_guard.as_mut() {
            Some(memory) => match memory.log_pxp(entry).await {
                Ok(()) => ToolResult::success("PxP entry logged successfully"),
                Err(e) => ToolResult::error(format!("Failed to log PxP: {}", e)),
            },
            None => ToolResult::error("Memory not initialized"),
        }
    }

    async fn handle_log_decision(&self, args: Value) -> ToolResult {
        let role_id = match args.get("role_id").and_then(|v| v.as_str()) {
            Some(r) => r.to_string(),
            None => return ToolResult::error("role_id is required"),
        };

        let title = match args.get("title").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => return ToolResult::error("title is required"),
        };

        let context = args
            .get("context")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Parse alternatives (for future use in Decision builder)
        let _alternatives: Vec<String> = args
            .get("alternatives")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let rationale = args
            .get("rationale")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Parse outcome (for future use in Decision builder)
        let _outcome = args
            .get("outcome")
            .and_then(|v| v.as_str())
            .map(String::from);
        let confidence = args.get("confidence").and_then(|v| v.as_f64());

        let decision = Decision::new(&role_id, &title)
            .with_context(&context)
            .with_decision(&rationale)
            .with_rationale(&rationale);

        let decision = if let Some(conf) = confidence {
            decision.with_confidence(conf)
        } else {
            decision
        };

        let decision = decision.finalize();

        let mut memory_guard = self.memory.write().await;
        match memory_guard.as_mut() {
            Some(memory) => match memory.record_decision(decision).await {
                Ok(()) => ToolResult::success("Decision recorded successfully"),
                Err(e) => ToolResult::error(format!("Failed to record decision: {}", e)),
            },
            None => ToolResult::error("Memory not initialized"),
        }
    }

    async fn handle_get_stats(&self) -> ToolResult {
        let memory_guard = self.memory.read().await;
        let memory = match memory_guard.as_ref() {
            Some(m) => m,
            None => return ToolResult::error("Memory not initialized"),
        };

        let role_ids = memory.role_ids();
        let const_loaded = memory.constitutional().await.is_ok();

        // Collect stats for each role
        let mut role_stats = Vec::new();
        for role_id in &role_ids {
            if let Ok(stats) = memory.pxp_stats(role_id).await {
                role_stats.push(json!({
                    "role_id": role_id,
                    "total_consultations": stats.total_consultations,
                    "model_usage": stats.model_usage,
                }));
            }
        }

        let stats = json!({
            "constitutional_loaded": const_loaded,
            "roles_count": role_ids.len(),
            "roles": role_ids,
            "role_stats": role_stats,
            "agents_path": self.agents_path.display().to_string(),
        });

        ToolResult::success(serde_json::to_string_pretty(&stats).unwrap_or_default())
    }

    async fn handle_check_constraint(&self, args: Value) -> ToolResult {
        let constraint_id = match args.get("constraint_id").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::error("constraint_id is required"),
        };

        let proposed_action = args
            .get("proposed_action")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let memory_guard = self.memory.read().await;
        let memory = match memory_guard.as_ref() {
            Some(m) => m,
            None => return ToolResult::error("Memory not initialized"),
        };

        match memory.check_constraint(constraint_id).await {
            Ok(Some(constraint)) => {
                let result = json!({
                    "constraint": constraint,
                    "proposed_action": proposed_action,
                    "analysis": format!(
                        "Constraint '{}' with enforcement '{}' should be evaluated against the proposed action.",
                        constraint.id, constraint.enforcement
                    ),
                });
                ToolResult::success(serde_json::to_string_pretty(&result).unwrap_or_default())
            }
            Ok(None) => {
                // Get available constraints
                match memory.constitutional().await {
                    Ok(constitutional) => {
                        let available: Vec<_> = constitutional.constraints.keys().collect();
                        ToolResult::error(format!(
                            "Constraint not found: {}. Available: {:?}",
                            constraint_id, available
                        ))
                    }
                    Err(_) => ToolResult::error(format!("Constraint not found: {}", constraint_id)),
                }
            }
            Err(e) => ToolResult::error(format!("Failed to check constraint: {}", e)),
        }
    }

    /// Handle a JSON-RPC request
    pub async fn handle_request(&mut self, request: JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => {
                if let Err(e) = self.initialize().await {
                    return JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32603,
                            message: format!("Failed to initialize: {}", e),
                            data: None,
                        }),
                    };
                }

                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(json!({
                        "protocolVersion": MCP_VERSION,
                        "capabilities": self.get_capabilities(),
                        "serverInfo": self.get_server_info(),
                    })),
                    error: None,
                }
            }

            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(json!({
                    "tools": self.list_tools()
                })),
                error: None,
            },

            "tools/call" => {
                let name = request
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let arguments = request
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(json!({}));

                let result = self.call_tool(name, arguments).await;

                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(serde_json::to_value(result).unwrap_or(json!({}))),
                    error: None,
                }
            }

            "notifications/initialized" | "initialized" => {
                // Client notification, no response needed
                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: None,
                }
            }

            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            },
        }
    }

    /// Run the MCP server over stdio
    pub async fn run_stdio(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let error_response = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: None,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {}", e),
                            data: None,
                        }),
                    };
                    writeln!(stdout, "{}", serde_json::to_string(&error_response)?)?;
                    stdout.flush()?;
                    continue;
                }
            };

            let response = self.handle_request(request).await;

            // Don't send response for notifications (no id)
            if response.id.is_some() || response.error.is_some() {
                writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
                stdout.flush()?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_info() {
        let server = VDreamMCPServer::new("/tmp/agents");
        let info = server.get_server_info();
        assert_eq!(info.name, "vDreamTeam Memory Server");
    }

    #[tokio::test]
    async fn test_list_tools() {
        let server = VDreamMCPServer::new("/tmp/agents");
        let tools = server.list_tools();
        assert!(tools.len() >= 5);
        assert!(tools
            .iter()
            .any(|t| t.name == "vdream_query_constitutional"));
        assert!(tools.iter().any(|t| t.name == "vdream_get_stats"));
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let mut server = VDreamMCPServer::new("/tmp/agents");
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/list".to_string(),
            params: json!({}),
        };

        let response = server.handle_request(request).await;
        assert!(response.result.is_some());
        let result = response.result.unwrap();
        assert!(result.get("tools").is_some());
    }
}
