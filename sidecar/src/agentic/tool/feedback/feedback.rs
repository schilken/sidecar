//! Generates the feedback for the trajectory

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
pub struct FeedbackGenerationRequest {
    llm_messages: Vec<LLMClientMessage>,
    message_properties: SymbolEventMessageProperties,
}

impl FeedbackGenerationRequest {
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
#[serde(rename = "feedback")]
#[serde(rename_all = "lowercase")]
pub struct FeedbackGenerationResponse {
    analysis: String,
    feedback: String,
}

impl FeedbackGenerationResponse {
    pub fn analysis(&self) -> &str {
        &self.analysis
    }

    pub fn feedback(&self) -> &str {
        &self.feedback
    }
}

pub struct FeedbackClientGenerator {
    llm_client: Arc<LLMBroker>,
}

impl FeedbackClientGenerator {
    pub fn new(llm_client: Arc<LLMBroker>) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Tool for FeedbackClientGenerator {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        let context = input.is_feedback_generation_request()?;
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
                    ("event_type".to_owned(), "feedback_generation".to_owned()),
                ]
                .into_iter()
                .collect(),
                sender,
            )
            .await;

        println!("reward_client::output::({:?})", &response);

        match response {
            Ok(response) => {
                let output = from_str::<FeedbackGenerationResponse>(&response)
                    .map_err(|_e| ToolError::SerdeConversionFailed)?;
                Ok(ToolOutput::FeedbackGeneration(output))
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
