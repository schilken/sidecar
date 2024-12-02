//! decides which is the best trajectory to take from all the nodes present
//! on the MCTS tree

use crate::{
    agentic::symbol::events::message_event::SymbolEventMessageProperties,
    mcts::action_node::{ActionNode, SearchTree},
};

use super::error::DeciderError;

pub struct Decider {}

impl Decider {
    pub fn decide(
        &self,
        nodes: Vec<&ActionNode>,
        search_tree: &SearchTree,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<&ActionNode, DeciderError> {
        todo!("implement this using agent based debate, we can use multiple algorithms over here honestly once we have a good trajectory")
    }
}
