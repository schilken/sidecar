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
        mut nodes_trajectory: Vec<&ActionNode>,
        tool_box: Arc<ToolBox>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<ToolOutput, InferenceError> {
        // split the trajectories between the root and the leaf right now
        if nodes_trajectory.is_empty() {
            return Err(InferenceError::EmptyTrajectory);
        }

        let root_to_leaf_direction = nodes_trajectory.split_off(nodes_trajectory.len() - 1);
        let leaf = nodes_trajectory.pop();
        if leaf.is_none() {
            return Err(InferenceError::EmptyTrajectory);
        }
        let leaf = leaf.expect("is_none to hold");
        todo!()
    }
}
