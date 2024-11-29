/// The reward for execution on an action node and the value generated out of it
pub struct Reward {
    /// An explanation and the reasoning behind your decision.
    explanation: Option<String>,
    /// Feedback to the alternative branch.
    feedback: Option<String>,
    /// A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue
    value: i32,
}

impl Reward {
    pub fn value(&self) -> i32 {
        self.value
    }

    pub fn feedback(&self) -> Option<String> {
        self.feedback.clone()
    }
}
