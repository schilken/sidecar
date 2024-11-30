//! Client which generates the reward for the action we have taken

use async_trait::async_trait;
use quick_xml::de::from_str;
use std::sync::Arc;

use llm_client::{
    broker::LLMBroker,
    clients::types::{LLMClientCompletionRequest, LLMClientMessage},
};

use crate::agentic::{
    symbol::events::message_event::SymbolEventMessageProperties,
    tool::{
        errors::ToolError,
        input::ToolInput,
        output::ToolOutput,
        r#type::{Tool, ToolRewardScale},
    },
};

#[derive(Debug, Clone)]
pub struct RewardGenerationRequest {
    llm_messages: Vec<LLMClientMessage>,
    message_properties: SymbolEventMessageProperties,
}

impl RewardGenerationRequest {
    pub fn new(
        llm_messages: Vec<LLMClientMessage>,
        message_properties: SymbolEventMessageProperties,
    ) -> Self {
        Self {
            llm_messages,
            message_properties,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename = "reward")]
#[serde(rename_all = "lowercase")]
pub struct RewardGenerationResponse {
    explanation: String,
    feedback: Option<String>,
    value: i32,
}

impl RewardGenerationResponse {
    pub fn explanation(&self) -> &str {
        &self.explanation
    }

    pub fn feedback(&self) -> Option<String> {
        self.feedback.clone()
    }

    pub fn value(&self) -> i32 {
        self.value
    }
}

pub struct RewardClientGenerator {
    llm_client: Arc<LLMBroker>,
}

impl RewardClientGenerator {
    pub fn new(llm_client: Arc<LLMBroker>) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Tool for RewardClientGenerator {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        let context = input.is_reward_generation_request()?;
        let message_properties = context.message_properties.clone();
        let llm_properties = message_properties.llm_properties().clone();
        let request = LLMClientCompletionRequest::new(
            llm_properties.llm().clone(),
            context.llm_messages,
            0.2,
            None,
        );

        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let response = self
            .llm_client
            .stream_completion(
                llm_properties.api_key().clone(),
                request,
                llm_properties.provider().clone(),
                vec![
                    (
                        "root_id".to_owned(),
                        message_properties.root_request_id().to_owned(),
                    ),
                    ("event_type".to_owned(), "reward_generation".to_owned()),
                ]
                .into_iter()
                .collect(),
                sender,
            )
            .await;

        println!("reward_client::output::({:?})", &response);

        match response {
            Ok(response) => {
                let output = from_str::<RewardGenerationResponse>(&response)
                    .map_err(|_e| ToolError::SerdeConversionFailed)?;
                Ok(ToolOutput::RewardGeneration(output))
            }
            Err(e) => Err(ToolError::LLMClientError(e)),
        }
    }

    fn tool_description(&self) -> String {
        "".to_owned()
    }

    fn tool_input_format(&self) -> String {
        "".to_owned()
    }

    fn get_evaluation_criteria(&self, _trajectory_length: usize) -> Vec<String> {
        vec![]
    }

    fn get_reward_scale(&self, _trajectory_length: usize) -> Vec<ToolRewardScale> {
        vec![]
    }
}
