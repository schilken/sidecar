//! Ollama client here so we can send requests to it

use async_trait::async_trait;

use super::types::LLMClient;
use super::types::LLMClientCompletionRequest;
use super::types::LLMClientCompletionResponse;
use super::types::LLMClientError;

pub struct OllamaClient {
    pub client: reqwest::Client,
    pub base_url: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct OllamaResponse {
    model: String,
    response: String,
    done: bool,
}

#[derive(serde::Serialize)]
pub struct OllamaClientRequest {
    prompt: String,
    model: String,
    temperature: f32,
    stream: bool,
    raw: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
}

impl OllamaClientRequest {
    pub fn from_request(request: super::types::LLMClientCompletionRequest) -> Self {
        Self {
            prompt: request.prompt().to_owned(),
            model: request.model().to_owned(),
            temperature: request.temperature(),
            stream: true,
            raw: true,
            frequency_penalty: request.frequency_penalty(),
        }
    }
}

impl OllamaClient {
    pub fn new() -> Self {
        // ollama always runs on the following url:
        // http://localhost:11434/
        Self {
            client: reqwest::Client::new(),
            base_url: "http://localhost:11434".to_owned(),
        }
    }

    pub fn generation_endpoint(&self) -> String {
        format!("{}/api/generate", self.base_url)
    }
}

#[async_trait]
impl LLMClient for OllamaClient {
    async fn stream_completion(
        &self,
        request: LLMClientCompletionRequest,
        sender: tokio::sync::mpsc::UnboundedSender<LLMClientCompletionResponse>,
    ) -> Result<String, LLMClientError> {
        let ollama_request = OllamaClientRequest::from_request(request);
        let mut response = self
            .client
            .post(self.generation_endpoint())
            .json(&ollama_request)
            .send()
            .await?;

        let mut buffered_string = "".to_owned();
        while let Some(chunk) = response.chunk().await? {
            let value = serde_json::from_slice::<OllamaResponse>(chunk.to_vec().as_slice())?;
            buffered_string.push_str(&value.response);
            sender.send(LLMClientCompletionResponse::new(
                buffered_string.to_owned(),
                Some(value.response),
                value.model,
            ))?;
        }
        Ok(buffered_string)
    }

    async fn completion(
        &self,
        request: LLMClientCompletionRequest,
    ) -> Result<String, LLMClientError> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let result = self.stream_completion(request, sender).await?;
        Ok(result)
    }
}