use crate::provider::{LLMProvider, LLMProviderAPIKeys};
use futures::StreamExt;

use super::types::{
    LLMClient, LLMClientCompletionRequest, LLMClientCompletionResponse,
    LLMClientCompletionStringRequest, LLMClientError, LLMType,
};
use async_trait::async_trait;
use eventsource_stream::Eventsource;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenRouterRequestMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenRouterRequest {
    model: String,
    temperature: f32,
    messages: Vec<OpenRouterRequestMessage>,
    stream: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OpenRouterResponseDelta {
    role: String,
    content: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenRouterResponseChoice {
    delta: OpenRouterResponseDelta,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OpenRouterResponse {
    model: String,
    choices: Vec<OpenRouterResponseChoice>,
}

impl OpenRouterRequest {
    pub fn from_chat_request(request: LLMClientCompletionRequest, model: String) -> Self {
        Self {
            model,
            temperature: request.temperature(),
            messages: request
                .messages()
                .into_iter()
                .map(|message| OpenRouterRequestMessage {
                    role: message.role().to_string(),
                    content: message.content().to_owned(),
                })
                .collect(),
            stream: true,
        }
    }
}

pub struct OpenRouterClient {
    client: reqwest::Client,
}

impl OpenRouterClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub fn model(&self, model: &LLMType) -> Option<String> {
        match model {
            LLMType::ClaudeHaiku => Some("anthropic/claude-3-haiku".to_owned()),
            LLMType::ClaudeSonnet => Some("anthropic/claude-3.5-sonnet:beta".to_owned()),
            LLMType::ClaudeOpus => Some("anthropic/claude-3-opus".to_owned()),
            LLMType::Gpt4 => Some("openai/gpt-4".to_owned()),
            LLMType::Gpt4O => Some("openai/gpt-4o".to_owned()),
            LLMType::DeepSeekCoderV2 => Some("deepseek/deepseek-coder".to_owned()),
            LLMType::Custom(name) => Some(name.to_owned()),
            _ => None,
        }
    }

    fn generate_auth_key(&self, api_key: LLMProviderAPIKeys) -> Result<String, LLMClientError> {
        match api_key {
            LLMProviderAPIKeys::OpenRouter(open_router) => Ok(open_router.api_key),
            _ => Err(LLMClientError::WrongAPIKeyType),
        }
    }
}

#[async_trait]
impl LLMClient for OpenRouterClient {
    fn client(&self) -> &LLMProvider {
        &LLMProvider::OpenRouter
    }

    async fn stream_completion(
        &self,
        api_key: LLMProviderAPIKeys,
        request: LLMClientCompletionRequest,
        sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        let base_url = "https://openrouter.ai/api/v1/chat/completions".to_owned();
        // pick this up from here, we need return type for the output we are getting form the stream
        let model = self
            .model(request.model())
            .ok_or(LLMClientError::WrongAPIKeyType)?;
        let auth_key = self.generate_auth_key(api_key)?;
        let request = OpenRouterRequest::from_chat_request(request, model.to_owned());
        let mut response_stream = self
            .client
            .post(base_url)
            .bearer_auth(auth_key)
            .header("HTTP-Referer", "https://aide.dev/")
            .header("X-Title", "aide")
            .json(&request)
            .send()
            .await?
            .bytes_stream()
            .eventsource();
        let mut buffered_stream = "".to_owned();
        while let Some(event) = response_stream.next().await {
            match event {
                Ok(event) => {
                    if &event.data == "[DONE]" {
                        continue;
                    }
                    let value = serde_json::from_str::<OpenRouterResponse>(&event.data)?;
                    buffered_stream = buffered_stream + &value.choices[0].delta.content;
                    sender.send(LLMClientCompletionResponse::new(
                        buffered_stream.to_owned(),
                        Some(value.choices[0].delta.content.to_owned()),
                        value.model,
                    ))?;
                }
                Err(e) => {
                    dbg!(e);
                }
            }
        }
        Ok(buffered_stream)
    }

    async fn completion(
        &self,
        _api_key: LLMProviderAPIKeys,
        _request: LLMClientCompletionRequest,
    ) -> Result<String, LLMClientError> {
        todo!()
    }

    async fn stream_prompt_completion(
        &self,
        _api_key: LLMProviderAPIKeys,
        _request: LLMClientCompletionStringRequest,
        _sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        todo!()
    }
}
