use async_trait::async_trait;

use crate::agentic::tool::{
    errors::ToolError,
    input::ToolInput,
    output::ToolOutput,
    r#type::{Tool, ToolRewardScale},
};

pub struct TerminalTool {
    client: reqwest::Client,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TerminalInputPartial {
    command: String,
}

impl TerminalInputPartial {
    pub fn new(command: String) -> Self {
        Self { command }
    }

    pub fn command(&self) -> &str {
        &self.command
    }

    pub fn to_string(&self) -> String {
        format!(
            r#"<execute_command>
<command>
{}
</command>
</execute_command>"#,
            self.command
        )
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct TerminalInput {
    command: String,
    editor_url: String,
}

impl TerminalInput {
    pub fn new(command: String, editor_url: String) -> Self {
        Self {
            command,
            editor_url,
        }
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct TerminalOutput {
    output: String,
}

impl TerminalOutput {
    pub fn output(&self) -> &str {
        &self.output
    }
}

impl TerminalTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Tool for TerminalTool {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        let context = input.is_terminal_command()?;
        let editor_endpoint = context.editor_url.to_owned() + "/execute_terminal_command";

        let response = self
            .client
            .post(editor_endpoint)
            .body(serde_json::to_string(&context).map_err(|_e| ToolError::SerdeConversionFailed)?)
            .send()
            .await
            .map_err(|_e| ToolError::ErrorCommunicatingWithEditor)?;

        let terminal_response: TerminalOutput = response
            .json()
            .await
            .map_err(|_e| ToolError::SerdeConversionFailed)?;

        Ok(ToolOutput::TerminalCommand(terminal_response))
    }

    // credit Cline.
    // Current working directory will be known to LLM from higher level context
    fn tool_description(&self) -> String {
        format!(
            r#"### execute_command
Request to execute a CLI command on the system.
Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task.
You must tailor your command to the user's system and provide a clear explanation of what the command does.
Prefer to execute complex CLI commands over creating executable scripts, as they are more flexible and easier to run.
Commands will be executed in the current working directory.
Note: You MUST append a `sleep 0.05` to the end of the command for commands that will complete in under 50ms, as this will circumvent a known issue with the terminal tool where it will sometimes not return the output when the command completes too quickly.
"#
        )
    }

    fn tool_input_format(&self) -> String {
        format!(
            r#"Parameters:
- command: (required) The CLI command to execute. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.

Usage:
<execute_command>
<command>
Your command here
</command>
</execute_command>
"#
        )
    }

    fn get_evaluation_criteria(&self, _trajectory_length: usize) -> Vec<String> {
        vec![]
    }

    fn get_reward_scale(&self) -> Vec<ToolRewardScale> {
        vec![]
    }
}
