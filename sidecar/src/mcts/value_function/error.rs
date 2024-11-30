//! The various errors which are part of the reward modul

use crate::agentic::symbol::errors::SymbolError;

#[derive(thiserror::Error, Debug)]
pub enum RewardError {
    #[error("Symbol error: {0}")]
    SymbolError(#[from] SymbolError),

    #[error("Empty trajectory")]
    EmptyTrajectory,

    #[error("Root not found")]
    RootError,

    #[error("Problem statement not found")]
    ProblemStatementNotFound,
}
