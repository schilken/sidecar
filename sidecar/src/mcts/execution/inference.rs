//! Executes the nodes for coding its main purpose is the following:
//! fn execute(nodes: Vec<&ActionNode>) -> Result<ToolOutput, ExecutionError>;

use std::{collections::HashMap, sync::Arc};

use llm_client::clients::types::LLMClientMessage;

use crate::{
    agentic::{
        symbol::{events::message_event::SymbolEventMessageProperties, tool_box::ToolBox},
        tool::output::ToolOutput,
    },
    mcts::action_node::{ActionNode, SearchTree},
};

use super::error::InferenceError;

struct InferenceEngine {}

impl InferenceEngine {
    async fn execute(
        mut nodes_trajectory: Vec<&ActionNode>,
        search_tree: &SearchTree,
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

        // keep track of the last updated file
        let mut last_updated_file: HashMap<String, usize> = Default::default();

        root_to_leaf_direction
            .iter()
            .enumerate()
            .for_each(|(index, current_node)| {
                let node_parent = search_tree.parent(current_node);
                let updated_files = match node_parent {
                    None => current_node
                        .user_context()
                        .file_paths()
                        .into_iter()
                        .collect(),
                    Some(node_parent) => current_node
                        .user_context()
                        .get_updated_files(node_parent.user_context()),
                };

                updated_files.into_iter().for_each(|file_path| {
                    last_updated_file.insert(file_path, index);
                });
            });

        // message history
        let mut message_history = vec![];

        // Now create the messages for the previous nodes which we have
        for (_index, current_node) in root_to_leaf_direction.iter().enumerate() {
            if let Some(message) = current_node.message() {
                message_history.push(LLMClientMessage::user(message));
            }

            if let Some(action) = current_node.action() {
                message_history.push(LLMClientMessage::assistant(action.to_string()))
            }

            if let Some(observation) = current_node.observation() {
                if let Some(summary) = observation.summary() {
                    message_history.push(LLMClientMessage::user(summary.to_owned()));
                } else {
                    message_history.push(LLMClientMessage::user(observation.message().to_owned()));
                }
            }
        }

        // do not do anything with the last updated files (yet)
        let _last_updated_files = last_updated_file;

        if let Some(feedback) = leaf.feedback() {
            message_history.push(LLMClientMessage::user(feedback));
        }

        // Now that we have the messages setup we ask the agent to generate the final tool which we want to use

        todo!()
    }
}
