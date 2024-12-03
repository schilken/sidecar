//! Executes the nodes for coding its main purpose is the following:
//! fn execute(nodes: Vec<&ActionNode>) -> Result<ToolOutput, ExecutionError>;

use std::{collections::HashMap, path::Path, sync::Arc};

use llm_client::clients::types::LLMClientMessage;

use crate::{
    agentic::{
        symbol::{
            events::{edit::SymbolToEdit, message_event::SymbolEventMessageProperties},
            identifier::SymbolIdentifier,
            tool_box::ToolBox,
        },
        tool::{
            input::{ToolInput, ToolInputPartial},
            lsp::{open_file::OpenFileRequest, search_file::SearchFileContentInput},
            r#type::Tool,
            repo_map::generator::RepoMapGeneratorRequest,
            session::{
                chat::SessionChatMessage,
                tool_use_agent::{ToolUseAgent, ToolUseAgentInput, ToolUseAgentOutput},
            },
            terminal::terminal::TerminalInput,
            test_runner::runner::TestRunnerRequest,
        },
    },
    chunking::text_document::{Position, Range},
    mcts::action_node::{ActionNode, ActionObservation, ActionToolParameters, SearchTree},
};

use super::error::InferenceError;

pub struct InferenceEngineResult {
    action_observation: Option<ActionObservation>,
    action_tool_parameters: ActionToolParameters,
    is_duplicate: bool,
}

impl InferenceEngineResult {
    pub fn new(
        action_observation: Option<ActionObservation>,
        action_tool_parameters: ActionToolParameters,
        is_duplicate: bool,
    ) -> Self {
        Self {
            action_observation,
            action_tool_parameters,
            is_duplicate,
        }
    }

    pub fn action_observation(&self) -> Option<ActionObservation> {
        self.action_observation.clone()
    }

    pub fn action_tool_parameters(&self) -> ActionToolParameters {
        self.action_tool_parameters.clone()
    }

    pub fn is_duplicate(&self) -> bool {
        self.is_duplicate
    }
}

pub struct InferenceEngine {}

impl InferenceEngine {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn execute(
        &self,
        mut nodes_trajectory: Vec<&ActionNode>,
        search_tree: &SearchTree,
        tool_box: Arc<ToolBox>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<InferenceEngineResult, InferenceError> {
        // split the trajectories between the root and the leaf right now
        if nodes_trajectory.is_empty() {
            return Err(InferenceError::EmptyTrajectory);
        }

        let leaf = nodes_trajectory.pop();
        if leaf.is_none() {
            return Err(InferenceError::EmptyTrajectory);
        }
        let leaf = leaf.expect("is_none to hold");
        let root_to_leaf_direction = nodes_trajectory;

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
                // always give the full observation message and not just the summary
                // since we will be generating new actions and they might be based
                // on the read_file output or the code_edit output
                message_history.push(LLMClientMessage::user(observation.message().to_owned()));
            }
        }

        // do not do anything with the last updated files (yet)
        let _last_updated_files = last_updated_file;

        if let Some(feedback) = leaf.feedback() {
            message_history.push(LLMClientMessage::user(feedback));
        }

        // Now that we have the messages setup we ask the agent to generate the final tool which we want to use
        let execution_and_observe = self
            .generate_observation_for_node(
                leaf,
                search_tree,
                message_history,
                tool_box,
                message_properties,
            )
            .await;
        execution_and_observe
    }

    async fn generate_observation_for_node(
        &self,
        current_node: &ActionNode,
        search_tree: &SearchTree,
        messages: Vec<LLMClientMessage>,
        tool_box: Arc<ToolBox>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<InferenceEngineResult, InferenceError> {
        let tool_use_agent = ToolUseAgent::new(
            search_tree.llm_client(),
            search_tree.root_directory(),
            "linux".to_owned(),
            "bash".to_owned(),
            Some(search_tree.repo_name()),
            false,
        );

        let mut session_messages = messages
            .into_iter()
            .map(|message| SessionChatMessage::from_llm_message(message))
            .collect::<Vec<_>>();

        // add a reminder for the output format so it never forgets the thinking tag
        session_messages.push(SessionChatMessage::user(
            r"# Output format reminder:
Always include the <thinking></thinking> section before using the tool.#"
                .to_owned(),
        ));

        let tool_agent_input = ToolUseAgentInput::new(
            session_messages,
            search_tree
                .tools()
                .into_iter()
                .filter_map(|tool_type| tool_box.tools().get_tool_description(&tool_type))
                .collect(),
            None,
            message_properties.clone(),
        );

        // now create the input for the tool use agent
        let tool_use_output = tool_use_agent.invoke(tool_agent_input).await;

        // Now we get the tool use output
        match tool_use_output {
            Ok(tool_use_parameters) => match tool_use_parameters {
                // we are going to execute this branch of the code so we can get the output
                // over here
                ToolUseAgentOutput::Success((tool_input_partial, _)) => {
                    let tool_parameters = ActionToolParameters::tool(tool_input_partial.clone());
                    // we should also detect duplicates over here before we start executing
                    // before executing the tool, check if the tool parameters are equal
                    // we can start with doing something very simple before we do a hard thing
                    let is_duplicate = search_tree.is_duplicate(current_node, &tool_parameters);
                    if is_duplicate {
                        Ok(InferenceEngineResult::new(None, tool_parameters, true))
                    } else {
                        // TODO(skcd): Execute the tool and generate the observation we need
                        // for the node
                        let node_execution_output = self
                            .execute_tool_and_generate_observation(
                                tool_input_partial,
                                tool_box.clone(),
                                message_properties.clone(),
                            )
                            .await;
                        match node_execution_output {
                            Ok(observation) => Ok(InferenceEngineResult::new(
                                Some(observation),
                                tool_parameters,
                                false,
                            )),
                            Err(e) => Ok(InferenceEngineResult::new(
                                // when we have an execution error on the tool we are royally
                                // messed up because we try our best to create an observation
                                // even for the failure cases, generally this means an infra
                                // failure so this is terminal
                                Some(ActionObservation::errored(e.to_string(), false, true)),
                                tool_parameters,
                                false,
                            )),
                        }
                    }
                }
                ToolUseAgentOutput::Failure(failed_string) => Ok(InferenceEngineResult::new(
                    Some(ActionObservation::errored(
                        failed_string.to_owned(),
                        // we failed to parse the tool output, so we can expect an correction
                        // over here
                        true,
                        false,
                    )),
                    ActionToolParameters::errored(failed_string),
                    false,
                )),
            },
            Err(e) => Ok(InferenceEngineResult::new(
                // This is an infra error so we can't expect a correction and this is terminal
                Some(ActionObservation::errored(e.to_string(), false, true)),
                ActionToolParameters::errored(e.to_string()),
                false,
            )),
        }
    }

    async fn execute_tool_and_generate_observation(
        &self,
        tool_input_partial: ToolInputPartial,
        tool_box: Arc<ToolBox>,
        message_properties: SymbolEventMessageProperties,
    ) -> Result<ActionObservation, InferenceError> {
        match tool_input_partial {
            ToolInputPartial::AskFollowupQuestions(_) => {
                // we never hit this branch for ask followup
                Err(InferenceError::WrongToolOutput)
            }
            ToolInputPartial::AttemptCompletion(attemp_completion) => {
                let message = attemp_completion.to_string();
                Ok(ActionObservation::new(message.to_owned(), message, true))
            }
            ToolInputPartial::CodeEditing(code_editing) => {
                let fs_file_path = code_editing.fs_file_path().to_owned();
                let file_contents = tool_box
                    .file_open(fs_file_path.to_owned(), message_properties.clone())
                    .await
                    .map_err(|e| InferenceError::SymbolError(e))?
                    .contents();

                let instruction = code_editing.instruction().to_owned();

                // keep track of the file content which we are about to modify over here
                let _old_file_content = tool_box
                    .file_open(fs_file_path.to_owned(), message_properties.clone())
                    .await;

                // if the file is very very large then we chunk it up and use search and replace
                // on individual chunks instead
                let updated_code = if file_contents.lines().into_iter().collect::<Vec<_>>().len()
                    >= 1300
                    // lets forgo the idea of being smart, do a simple edit
                    // if that does not work then we will know we fucked up in our
                    // observations
                    && false
                {
                    let first_part_lines = file_contents
                        .to_owned()
                        .lines()
                        .into_iter()
                        .enumerate()
                        .filter_map(|(idx, line)| {
                            if idx <= 750 {
                                Some(line.to_owned())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let second_part_lines = file_contents
                        .to_owned()
                        .lines()
                        .into_iter()
                        .enumerate()
                        .filter_map(|(idx, line)| {
                            if idx > 750 {
                                Some(line.to_owned())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let range_to_edit =
                        Range::new(Position::new(0, 0, 0), Position::new(10_000, 0, 0));

                    // First half of the file has been edited
                    let symbol_to_edit = SymbolToEdit::new(
                        fs_file_path.to_owned(),
                        range_to_edit,
                        fs_file_path.to_owned(),
                        vec![instruction.clone()],
                        false,
                        false, // is_new
                        false,
                        "".to_owned(),
                        None,
                        false,
                        None,
                        false,
                        None,
                        vec![], // previous_user_queries
                        None,
                    )
                    .set_should_stream_status(false);

                    let symbol_identifier = SymbolIdentifier::new_symbol(&fs_file_path);

                    let first_part_edited = tool_box
                        .code_editing_with_search_and_replace(
                            &symbol_to_edit,
                            &fs_file_path,
                            &first_part_lines,
                            &range_to_edit,
                            "".to_owned(),
                            instruction.clone(),
                            &symbol_identifier,
                            None,
                            None,
                            message_properties.clone(),
                        )
                        .await
                        .map_err(|e| InferenceError::SymbolError(e))?; // big expectations but can also fail, we should handle it properly

                    // Editing second half of the file
                    let symbol_to_edit = SymbolToEdit::new(
                        fs_file_path.to_owned(),
                        range_to_edit,
                        fs_file_path.to_owned(),
                        vec![instruction.clone()],
                        false,
                        false, // is_new
                        false,
                        "".to_owned(),
                        None,
                        false,
                        None,
                        false,
                        None,
                        vec![], // previous_user_queries
                        None,
                    )
                    .set_should_stream_status(false);

                    let symbol_identifier = SymbolIdentifier::new_symbol(&fs_file_path);

                    let second_part_edited = tool_box
                        .code_editing_with_search_and_replace(
                            &symbol_to_edit,
                            &fs_file_path,
                            &second_part_lines,
                            &range_to_edit,
                            "".to_owned(),
                            format!(r#"{}
This is part of the file which might not contain the method in full, if thats the case do not generate any edits"#, instruction.clone()),
                            &symbol_identifier,
                            None,
                            None,
                            message_properties.clone(),
                        )
                        .await
                        .map_err(|e| InferenceError::SymbolError(e))?; // big expectations but can also fail, we should handle it properly
                    format!(
                        r#"{}
{}"#,
                        first_part_edited, second_part_edited
                    )
                } else {
                    let default_range =
                    // very large end position
                    Range::new(Position::new(0, 0, 0), Position::new(10_000, 0, 0));

                    let symbol_to_edit = SymbolToEdit::new(
                        fs_file_path.to_owned(),
                        default_range,
                        fs_file_path.to_owned(),
                        vec![instruction.clone()],
                        false,
                        false, // is_new
                        false,
                        "".to_owned(),
                        None,
                        false,
                        None,
                        false,
                        None,
                        vec![], // previous_user_queries
                        None,
                    )
                    .set_should_stream_status(false);

                    let symbol_identifier = SymbolIdentifier::new_symbol(&fs_file_path);

                    tool_box
                        .code_editing_with_search_and_replace(
                            &symbol_to_edit,
                            &fs_file_path,
                            &file_contents,
                            &default_range,
                            "".to_owned(),
                            instruction.clone(),
                            &symbol_identifier,
                            None,
                            None,
                            message_properties.clone(),
                        )
                        .await
                        .map_err(|e| InferenceError::SymbolError(e))? // big expectations but can also fail, we should handle it properly
                };
                // This code-block only ever hits for the swe-bench run and nothing else
                // in the future we should create a tool for this, but this will help unblock us
                {
                    // we want to update the whole file content with the new content over here
                    // first we check if the file really exists on the fs, if it does not we create it
                    if let Ok(false) = tokio::fs::try_exists(fs_file_path.to_owned()).await {
                        tokio::fs::create_dir_all(
                            Path::new(&fs_file_path).parent().expect("to exist"),
                        )
                        .await
                        .expect("creating parent directory to work");
                        tokio::fs::File::create(fs_file_path.to_owned())
                            .await
                            .expect("file creation to not fail");
                    }
                    let _ =
                        tokio::fs::write(fs_file_path.to_owned(), updated_code.to_owned()).await;

                    // we have the original file content and the updated code content
                    // we want to generate a git-diff between the 2 and pass that to the LLM implicitly
                    // since we do not have a recent-edits handle easily implemented in python mock editor
                    // This is really bad but we are interested in testing out things for now (DO NOT COMMIT)
                    let client = reqwest::Client::new();
                    let original_content = &file_contents;
                    let request_object = serde_json::json!({
                        "original_content": original_content,
                        "modified_content": updated_code,
                        "fs_file_path": fs_file_path.to_owned(),
                    });
                    let response = client
                        .post(message_properties.editor_url() + "/diff_generator")
                        .body(serde_json::to_string(&request_object).expect("to work"))
                        .send()
                        .await
                        .expect("to get a reply");
                    #[derive(serde::Deserialize)]
                    struct FileEditedResponseStruct {
                        generated_diff: String,
                    }
                    let response: FileEditedResponseStruct =
                        response.json().await.expect("to work");
                    let generated_diff = response.generated_diff;
                    let message = if updated_code == original_content {
                        "Failed to perform the requested edits".to_owned()
                    } else {
                        format!(
                            r#"I performed the edits which you asked me to, and here is the patch with the changes
    {generated_diff}"#
                        )
                    };
                    Ok(ActionObservation::new(message.to_owned(), message, false)
                        .file_content_updated(fs_file_path, updated_code))
                }
            }
            ToolInputPartial::LSPDiagnostics(_) => {
                todo!("LSP diagnostics are not supported right now")
            }
            ToolInputPartial::ListFiles(list_files) => {
                let directory_path = list_files.directory_path().to_owned();
                let input = ToolInput::ListFiles(list_files);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| InferenceError::ToolError(e))?;
                let list_files_output = response
                    .get_list_files_directory()
                    .ok_or(InferenceError::WrongToolOutput)?;
                let response = list_files_output
                    .files()
                    .into_iter()
                    .map(|file_path| file_path.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                let message = format!(
                    r#"Content for directory {directory_path}
{}"#,
                    response.to_owned()
                );
                Ok(ActionObservation::new(
                    message.to_owned(),
                    message.to_owned(),
                    false,
                ))
            }
            ToolInputPartial::OpenFile(open_file) => {
                let open_file_path = open_file.fs_file_path().to_owned();
                let request = OpenFileRequest::new(
                    open_file_path.to_owned(),
                    message_properties.editor_url(),
                );
                let input = ToolInput::OpenFile(request);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| InferenceError::ToolError(e))?
                    .get_file_open_response()
                    .ok_or(InferenceError::WrongToolOutput)?;
                Ok(ActionObservation::new(
                    format!(
                        r#"Here's the content of the file which you wanted to see
{}"#,
                        &response.to_string()
                    ),
                    format!(
                        "Showed the content of the following file {}",
                        &open_file_path
                    ),
                    false,
                )
                .file_content_updated(open_file_path.to_owned(), response.to_content()))
            }
            ToolInputPartial::RepoMapGeneration(repo_map_request) => {
                let directory_path = repo_map_request.directory_path().to_owned();
                let request = ToolInput::RepoMapGeneration(RepoMapGeneratorRequest::new(
                    repo_map_request.directory_path().to_owned(),
                    3000,
                ));
                let tool_output = tool_box
                    .tools()
                    .invoke(request)
                    .await
                    .map_err(|e| InferenceError::ToolError(e))?
                    .repo_map_generator_response()
                    .ok_or(InferenceError::WrongToolOutput)?;
                let repo_map_str = tool_output.repo_map().to_owned();
                let message = format!(
                    r#"Here's the outline of classes and functions present in the directory {directory_path}
{repo_map_str}"#
                );
                Ok(ActionObservation::new(message.to_owned(), message, false))
            }
            ToolInputPartial::SearchFileContentWithRegex(search_file) => {
                let request = SearchFileContentInput::new(
                    search_file.directory_path().to_owned(),
                    search_file.regex_pattern().to_owned(),
                    search_file.file_pattern().map(|s| s.to_owned()),
                    message_properties.editor_url(),
                );
                let input = ToolInput::SearchFileContentWithRegex(request);
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| InferenceError::ToolError(e))?
                    .get_search_file_content_with_regex()
                    .ok_or(InferenceError::WrongToolOutput)?;
                let response = response.response();
                let message = format!(
                    r#"Here's the result of running the search query
{}"#,
                    response
                );
                Ok(ActionObservation::new(message.to_owned(), message, false))
            }
            ToolInputPartial::TerminalCommand(terminal_command) => {
                let command = terminal_command.command().to_owned();
                let request =
                    TerminalInput::new(command.to_owned(), message_properties.editor_url());
                let input = ToolInput::TerminalCommand(request);
                let tool_output = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| InferenceError::ToolError(e))?
                    .terminal_command()
                    .ok_or(InferenceError::WrongToolOutput)?;
                let output = tool_output.output().to_owned();
                let message = format!(
                    r#"Here's the output from running the terminal command
Command: {}
Terminal output: {}"#,
                    command, output
                );
                Ok(ActionObservation::new(message.to_owned(), message, false))
            }
            ToolInputPartial::TestRunner(fs_file_paths) => {
                let editor_url = message_properties.editor_url().to_owned();
                let input =
                    ToolInput::RunTests(TestRunnerRequest::new(fs_file_paths.clone(), editor_url));
                let response = tool_box
                    .tools()
                    .invoke(input)
                    .await
                    .map_err(|e| InferenceError::ToolError(e))?
                    .get_test_runner()
                    .ok_or(InferenceError::WrongToolOutput)?;
                let message = format!(
                    r#"Here's the result of running the tests on the following files:
{}
                
Test Output from the script (we also have to setup the test runner):
Exit code: {}
Output:
{}"#,
                    fs_file_paths.join("\n"),
                    response.exit_code(),
                    response.test_output()
                );
                Ok(ActionObservation::new(message.to_owned(), message, false))
            }
        }
    }
}
