//! Executes the nodes for coding its main purpose is the following:
//! fn execute(nodes: Vec<&ActionNode>) -> Result<ToolOutput, ExecutionError>;

use std::sync::Arc;

use crate::{
    agentic::{
        symbol::{events::message_event::SymbolEventMessageProperties, tool_box::ToolBox},
        tool::output::ToolOutput,
    },
    mcts::action_node::ActionNode,
};

use super::error::InferenceError;

struct InferenceEngine {}

impl InferenceEngine {
    async fn execute(
        _nodes: Vec<&ActionNode>,
        _tool_box: Arc<ToolBox>,
        _message_properties: SymbolEventMessageProperties,
    ) -> Result<ToolOutput, InferenceError> {
        todo!()
    }
}
