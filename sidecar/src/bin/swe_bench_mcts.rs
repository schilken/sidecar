use clap::Parser;
use llm_client::{
    broker::LLMBroker,
    clients::types::LLMType,
    config::LLMBrokerConfiguration,
    provider::{AnthropicAPIKey, GoogleAIStudioKey, LLMProvider, LLMProviderAPIKeys},
};
use serde::{Deserialize, Serialize};
use sidecar::{
    agentic::{
        symbol::{
            events::{input::SymbolEventRequestId, message_event::SymbolEventMessageProperties},
            identifier::LLMProperties,
            tool_box::ToolBox,
        },
        tool::{
            broker::{ToolBroker, ToolBrokerConfiguration},
            code_edit::models::broker::CodeEditBroker,
            r#type::ToolType,
        },
    },
    chunking::{editor_parsing::EditorParsing, languages::TSLanguageParsing},
    inline_completion::symbols_tracker::SymbolTrackerInline,
    mcts::{action_node::SearchTree, selector::selector::Selector},
};
use std::{path::PathBuf, sync::Arc};

/// Define the command-line arguments
#[derive(Parser, Debug)]
#[command(author = "skcd", version = "1.0", about = "SWE-Bench Sidecar Runner")]
struct CliArgs {
    /// Git directory name
    #[arg(long)]
    timeout: usize,

    /// Endpoint URL
    #[arg(long)]
    editor_url: String,

    /// Timeout in seconds
    #[arg(long)]
    input: PathBuf,

    /// Anthropic api key
    #[arg(long)]
    anthropic_api_key: String,

    /// The run id for the current run
    #[arg(long)]
    run_id: String,

    #[arg(long)]
    repo_name: String,

    /// Directory to dump all the logs into
    #[arg(long)]
    log_directory: String,
}

/// Define the SWEbenchInstance struct for serialization
#[derive(Debug, Serialize, Deserialize)]
struct SWEbenchInstance {
    repo: String,
    instance_id: String,
    base_commit: String,
    patch: String,
    test_patch: String,
    problem_statement: String,
    hints_text: String,
    created_at: String,
    version: String,
    #[serde(rename = "FAIL_TO_PASS")]
    fail_to_pass: String,
    #[serde(rename = "PASS_TO_PASS")]
    pass_to_pass: String,
    environment_setup_commit: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct InputParts {
    git_drname: String,
    instance: SWEbenchInstance,
}

fn default_index_dir() -> PathBuf {
    match directories::ProjectDirs::from("ai", "codestory", "sidecar") {
        Some(dirs) => dirs.data_dir().to_owned(),
        None => "codestory_sidecar".into(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args = CliArgs::parse();

    let editor_parsing = Arc::new(EditorParsing::default());
    let symbol_broker = Arc::new(SymbolTrackerInline::new(editor_parsing.clone()));
    let llm_broker = Arc::new(
        LLMBroker::new(LLMBrokerConfiguration::new(default_index_dir()))
            .await
            .expect("to initialize properly"),
    );
    let tool_broker = Arc::new(ToolBroker::new(
        llm_broker.clone(),
        Arc::new(CodeEditBroker::new()),
        symbol_broker.clone(),
        Arc::new(TSLanguageParsing::init()),
        ToolBrokerConfiguration::new(None, true),
        LLMProperties::new(
            LLMType::GeminiPro,
            LLMProvider::GoogleAIStudio,
            LLMProviderAPIKeys::GoogleAIStudio(GoogleAIStudioKey::new("".to_owned())),
        ),
    ));

    let symbol_tracker = Arc::new(SymbolTrackerInline::new(editor_parsing.clone()));

    let tool_box = Arc::new(ToolBox::new(tool_broker, symbol_broker, editor_parsing));

    let editor_url = args.editor_url.to_owned();
    let _timeout = args.timeout;
    let input_path = args.input;
    let run_id = args.run_id.to_owned();
    let repo_name = args.repo_name.to_owned();
    let anthropic_api_key = args.anthropic_api_key.to_owned();
    let log_directory = args.log_directory.to_owned();
    let input_content = tokio::fs::read(input_path).await.expect("path content");
    let input_parts: InputParts =
        serde_json::from_slice(&input_content).expect("Parse the serde json");

    let model_configuration = LLMProperties::new(
        LLMType::ClaudeSonnet,
        LLMProvider::Anthropic,
        LLMProviderAPIKeys::Anthropic(AnthropicAPIKey::new(anthropic_api_key)),
    );

    let session_id = format!(
        "{}-{}",
        input_parts.instance.instance_id,
        run_id.to_string()
    );

    println!("session_id:{}", &session_id);

    // Creates the unique path for the session
    let session_path = default_index_dir().join("session");
    // check if the plan_storage_path_exists
    if tokio::fs::metadata(&session_path).await.is_err() {
        tokio::fs::create_dir(&session_path)
            .await
            .expect("directory creation to not fail");
    }
    let session_path = session_path.join(session_id.to_owned());
    let storage_path = session_path
        .to_str()
        .expect("path conversion to work on all platforms")
        .to_owned();

    let initial_exchange_id = 0;

    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let cancellation_token = tokio_util::sync::CancellationToken::new();
    let message_properties = SymbolEventMessageProperties::new(
        SymbolEventRequestId::new(
            initial_exchange_id.to_string().to_owned(),
            run_id.to_string(),
        ),
        sender.clone(),
        editor_url,
        cancellation_token.clone(),
        model_configuration,
    );

    let selector = Selector::new(
        1.0,                         // exploitation_weight
        false,                       // use_average_reward
        1.0,                         // exploration_weight
        0.8,                         // depth_weight
        0.0,                         // depth_bonus_factor
        50.0,                        // high_value_threshold
        0.0,                         // low_value_threshold
        75.0,                        // very_high_value_threshold
        50.0,                        // high_value_leaf_bonus_constant
        20.0,                        // high_value_bad_children_bonus_constant
        5.0,                         // high_value_child_penalty_constant
        50.0,                        // finished_trajectory_penalty
        50.0,                        // expect_correction_bonus
        vec![ToolType::CodeEditing], // check_for_bad_child_actions
        100.0,                       // diversity_weight
        25.0,                        // duplicate_child_penalty_constant
        50.0,                        // duplicate_action_penalty_constant
    );

    // Instantiate the mcts tree over here and start the search
    let mut search_tree = SearchTree::new(
        3,                                      // max_expansions
        20,                                     // max_depth of the tree
        100,                                    // max_iterations
        Some(3),                                // max_finished_nodes
        None,                                   // reward_threshold
        Some(2),                                // min_finished_nodes
        input_parts.git_drname.to_owned(),      // root_directory
        repo_name,                              // repo_name
        input_parts.instance.problem_statement, // problem_statment
        selector,                               // selector
        vec![
            ToolType::ListFiles,
            ToolType::SearchFileContentWithRegex,
            ToolType::OpenFile,
            ToolType::CodeEditing,
            ToolType::AttemptCompletion,
            ToolType::RepoMapGeneration,
            ToolType::TerminalCommand,
            ToolType::TestRunner,
        ], // tools
        tool_box,                               // tool_box
        llm_broker,                             // llm_client
        log_directory,                          // log directory
    );

    // Run the search
    search_tree.run_search(message_properties).await;

    Ok(())
}
