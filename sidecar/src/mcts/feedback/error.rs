#[derive(Debug, thiserror::Error)]
pub enum FeedbackError {
    #[error("Empty trajectory")]
    EmptyTrajectory,
}
