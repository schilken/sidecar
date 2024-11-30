use std::sync::Arc;

use llm_client::clients::types::LLMClientMessage;

use crate::{
    agentic::{
        symbol::{events::message_event::SymbolEventMessageProperties, tool_box::ToolBox},
        tool::r#type::ToolType,
    },
    mcts::action_node::{ActionNode, ActionToolParameters, SearchTree},
};

use super::error::RewardError;

/// The reward for execution on an action node and the value generated out of it
pub struct Reward {
    /// An explanation and the reasoning behind your decision.
    explanation: String,
    /// Feedback to the alternative branch.
    feedback: Option<String>,
    /// A single integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue
    value: i32,
}

impl Reward {
    pub fn with_explanation(explanation: String, value: i32) -> Self {
        Self {
            explanation,
            value,
            feedback: None,
        }
    }

    pub fn value(&self) -> i32 {
        self.value
    }

    pub fn feedback(&self) -> Option<String> {
        self.feedback.clone()
    }

    pub fn explanation(&self) -> &str {
        &self.explanation
    }
}

/// Generates the reward for the code and the trajectory
pub struct RewardGeneration {}

impl RewardGeneration {
    pub fn generate_reward(
        &self,
        mut nodes_trajectory: Vec<&ActionNode>,
        search_tree: &SearchTree,
        tool_box: Arc<ToolBox>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<Reward, RewardError> {
        if nodes_trajectory.is_empty() {
            return Err(RewardError::EmptyTrajectory);
        }

        let root_to_leaf_direction = nodes_trajectory.split_off(nodes_trajectory.len() - 1);
        let leaf = nodes_trajectory.pop();
        if leaf.is_none() {
            return Err(RewardError::EmptyTrajectory);
        }
        let leaf = leaf.expect("is_none to hold");

        if let Some(observation) = leaf.observation() {
            // we require a correction, no reward
            if observation.expect_correction() {
                return Ok(Reward::with_explanation(
                    "Expects a correction".to_owned(),
                    0,
                ));
            }
        }

        // check if the action was an error
        if let Some(ActionToolParameters::Errored(_)) = leaf.action() {
            return Ok(Reward::with_explanation(
                "Error action, assigning reward -100".to_owned(),
                -100,
            ));
        }

        // current message
        let current_message = match leaf.action() {
            Some(ActionToolParameters::Errored(_)) => {
                return Ok(Reward::with_explanation(
                    "Error action, assigning reward -100".to_owned(),
                    -100,
                ))
            }
            Some(ActionToolParameters::Tool(tool_input_partial)) => {
                let tool_type = tool_input_partial.to_tool_type();
                match tool_type {
                    ToolType::AttemptCompletion => tool_input_partial.to_string(),
                    _ => {
                        format!(
                            r#"## Last Executed Action:
        Here is the most recent action that was executed and its output. This is the subject of your evaluation.
        <executed_action>
        {}
        </executed_action>
        
        ## Output:
        {}"#,
                            tool_input_partial.to_string(),
                            leaf.observation()
                                .map(|observation| observation.message().to_owned())
                                .unwrap_or("No observation found.".to_owned())
                        )
                    }
                }
            }
            None => {
                return Ok(Reward::with_explanation(
                    "Error, no action assigning reward -100".to_owned(),
                    -100,
                ))
            }
        };

        // messages for the trajectory
        let messages =
            self.messages_for_reward(leaf, root_to_leaf_direction, current_message, search_tree)?;
        todo!()
    }

    fn messages_for_reward(
        &self,
        leaf: &ActionNode,
        root_to_leaf: Vec<&ActionNode>,
        current_message: String,
        search_tree: &SearchTree,
    ) -> Result<Vec<LLMClientMessage>, RewardError> {
        let root_node = search_tree.root();
        if let None = root_node {
            return Err(RewardError::RootError);
        }
        let root_node = root_node.expect("if let None to hold");

        let problem_statement = root_node.message();
        if let None = problem_statement {
            return Err(RewardError::ProblemStatementNotFound);
        }
        let problem_statement = problem_statement.expect("if let None to hold");

        let root_to_leaf_len = root_to_leaf.len();

        let action_observations = root_to_leaf
            .into_iter()
            .enumerate()
            .map(|(idx, current_node)| {
                let action = current_node.action();
                match action {
                    Some(action) => {
                        let action_part =
                            format!(r#"## {} Action: {}"#, idx + 1, action.to_string());
                        let action_part = format!(
                            r#"{}
{}"#,
                            action_part,
                            action.to_string()
                        );

                        let action_observation = match current_node.observation() {
                            Some(observation) => {
                                if observation.summary().is_some() && idx < root_to_leaf_len - 1 {
                                    format!(
                                        r#"{action_part}
Observation: {}"#,
                                        observation.summary().unwrap_or_default()
                                    )
                                } else {
                                    format!(
                                        r#"{action_part}
Observation: {}"#,
                                        observation.message()
                                    )
                                }
                            }
                            None => {
                                format!(
                                    r#"{action_part}
Observation: No output found."#
                                )
                            }
                        };
                        action_observation
                    }
                    None => format!(r#"## {} No action taken at this stage"#, idx + 1),
                }
            })
            .collect::<Vec<String>>();

        let action_observations = format!(
            r#"{problem_statement}

Below is the history of previously executed actions and their observations.
<history>
{}
</history>
        "#,
            action_observations.join("\n")
        );

        // - Now we create the file content (it would be better if we can only keep track
        // of the interested code instead of the whole file)
        let file_content_vec = leaf
            .user_context()
            .variables
            .iter()
            .filter(|variable_information| variable_information.is_file())
            .map(|variable_information| variable_information.clone().to_xml())
            .collect::<Vec<_>>();

        let parent_node = search_tree.parent(leaf);

        let git_patch_diff = if let Some(parent_node) = parent_node {
            parent_node
                .user_context()
                .variables
                .iter()
                .filter(|variable_information| variable_information.is_file())
                .filter_map(|variable_information| {
                    let patch = variable_information.patch_from_root();
                    match patch {
                        Some(patch) => Some(format!(
                            r#"## Changes in {}
{}"#,
                            &variable_information.fs_file_path, patch
                        )),
                        None => None,
                    }
                })
                .collect::<Vec<_>>()
        } else {
            vec!["".to_owned()]
        };

        let current_message = format!(
            r#"{current_message}

The file context the agent has access to:
<files>
{}
</files>

The git diff of the changes until the last action:
{}"#,
            &file_content_vec.join("\n"),
            git_patch_diff.join("\n")
        );

        // generate the system message over here
        Ok(vec![
            LLMClientMessage::user(action_observations),
            LLMClientMessage::user(current_message),
        ])
    }

    // TODO(skcd): Pick up the system message from here and make it work properly
    fn system_message(
        &self,
        action_node: &ActionNode,
        root_to_leaf: Vec<&ActionNode>,
        search_tree: &SearchTree,
        tool_box: Arc<ToolBox>,
    ) -> String {
        let tools = tool_box.tools();
        let tool_types = search_tree.tools();
        let trajectory_length = search_tree.trajectory(action_node.index()).len();
        // generate the system message where we have to show it the format
        // for generating the output
        todo!("generate the system message for reward function")
    }
}
