//! The feedback generator over here takes in the node we are on and tries to generate
//! feedback to pick an action which will lead to diversity

use crate::{
    agentic::symbol::events::message_event::SymbolEventMessageProperties,
    mcts::action_node::{ActionNode, SearchTree},
};

use super::error::FeedbackError;

pub struct FeedbackToNode {
    // Analysis of the current task we are on and the different trajectories we have explored
    analysis: String,
    // Direct feedback to the AI agent
    feedback: String,
}

pub struct FeedbackGenerator {}

impl FeedbackGenerator {
    pub fn generate_feedback_for_node(
        &self,
        mut nodes_trajectory: Vec<&ActionNode>,
        search_tree: &SearchTree,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<FeedbackToNode, FeedbackError> {
        todo!("implement the feedback generator")
    }
}
