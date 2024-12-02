use crate::agentic::tool::errors::ToolError;

#[derive(Debug, thiserror::Error)]
pub enum DeciderError {
    #[error("Tool Error: {0}")]
    ToolError(#[from] ToolError),
}
