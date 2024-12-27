use crate::{
    agentic::tool::{
        errors::ToolError,
        input::ToolInput,
        output::ToolOutput,
        r#type::{Tool, ToolRewardScale},
    },
    chunking::text_document::{Position, Range},
};
use async_trait::async_trait;
use gix::bstr::ByteSlice;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenFileRequestPartial {
    fs_file_path: String,
}

impl OpenFileRequestPartial {
    pub fn new(fs_file_path: String) -> Self {
        Self { fs_file_path }
    }

    pub fn fs_file_path(&self) -> &str {
        &self.fs_file_path
    }

    pub fn to_string(&self) -> String {
        format!(
            r#"<thinking>
...
</thinking>
<read_file>
<fs_file_path>
{}
</fs_file_path>
</read_file>"#,
            &self.fs_file_path
        )
    }

    pub fn to_json() -> serde_json::Value {
        serde_json::json!({
            "name": "read_file",
            "description": r#"Request to read the full contents of a file at the specified path.
Use this when you need to examine the full content of an existing file you do not know the contents of, for example to analyze code, review text files, or extract information from configuration files.
May not be suitable for other types of binary files, as it returns the raw content as a string"#,
            "input_schema": {
                "type": "object",
                "properties": {
                    "fs_file_path": {
                        "type": "string",
                        "description": "(required) The relative path of the file to read.",
                    },
                },
                "required": ["fs_file_path"],
            },
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenFileRequest {
    fs_file_path: String,
    editor_url: String,
}

impl OpenFileRequest {
    pub fn new(fs_file_path: String, editor_url: String) -> Self {
        Self {
            fs_file_path,
            editor_url,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpenFileResponse {
    fs_file_path: String,
    file_contents: String,
    exists: bool,
    // TODO(skcd): This might break
    language: String,
}

impl OpenFileResponse {
    pub fn to_string(&self) -> String {
        let fs_file_path = &self.fs_file_path;
        let content = &self.file_contents;
        let language = &self.language;
        format!(
            r#"<fs_file_path>
{fs_file_path}
</fs_file_path>
<content>
```{language}
{content}
</content>"#
        )
    }

    pub fn to_content(&self) -> String {
        self.file_contents.to_owned()
    }
}

impl OpenFileResponse {
    pub fn new(
        fs_file_path: String,
        file_contents: String,
        exists: bool,
        language: String,
    ) -> Self {
        Self {
            fs_file_path,
            file_contents,
            exists,
            language,
        }
    }

    pub fn contents(self) -> String {
        self.file_contents
    }

    pub fn contents_ref(&self) -> &str {
        &self.file_contents
    }

    pub fn fs_file_path(&self) -> &str {
        self.fs_file_path.as_str()
    }

    pub fn exists(&self) -> bool {
        self.exists
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn content_in_range(&self, range: &Range) -> Option<String> {
        if !self.exists {
            None
        } else {
            // we are calling it content in range and thats what it is exactly
            // if we do not have anything at the start of it, we leave it
            // but I am not sure if this is the best thing to do and do not know
            // of cases where we will be looking at exact text and not start_line..end_line
            // TODO(skcd): Make this more accurate when lines match up
            self.file_contents
                .lines()
                .enumerate()
                .filter(|(i, _line)| i >= &range.start_line())
                .take_while(|(i, _line)| i <= &range.end_line())
                .map(|(_i, line)| line.to_string())
                .collect::<Vec<String>>()
                .join("\n")
                .into()
        }
    }

    /// Grabs the content in a range using the byte offset
    pub fn content_in_ranges_exact(&self, range: &Range) -> Option<String> {
        if !self.exists {
            None
        } else {
            let bytes_len = self.file_contents.as_bytes().len();
            if range.start_byte() < bytes_len && range.end_byte() < bytes_len {
                self.file_contents.as_bytes()[range.start_byte()..range.end_byte()]
                    .to_str()
                    .map(|s| s.to_owned())
                    .ok()
            } else {
                None
            }
        }
    }

    /// Length of the file contents
    pub fn file_content_len(&self) -> usize {
        self.file_contents
            .lines()
            .into_iter()
            .collect::<Vec<_>>()
            .len()
    }

    pub fn full_range(&self) -> Range {
        let mut file_content_len = self.file_content_len();
        if file_content_len != 0 {
            file_content_len = file_content_len - 1;
        }
        Range::new(
            Position::new(0, 0, 0),
            Position::new(file_content_len, 0, 0),
        )
    }
}

pub struct LSPOpenFile {
    client: reqwest::Client,
}

impl LSPOpenFile {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Tool for LSPOpenFile {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        // we want to create a new file open request over here
        let context = input.is_file_open()?;
        // now we send it over to the editor
        let editor_endpoint = context.editor_url.to_owned() + "/file_open";
        let response = self
            .client
            .post(editor_endpoint)
            .body(serde_json::to_string(&context).map_err(|_e| ToolError::SerdeConversionFailed)?)
            .send()
            .await
            .map_err(|_e| ToolError::ErrorCommunicatingWithEditor)?;
        let response: OpenFileResponse = response
            .json()
            .await
            .map_err(|_e| ToolError::ErrorCommunicatingWithEditor)?;
        Ok(ToolOutput::FileOpen(response))
    }

    fn tool_description(&self) -> String {
        format!(
            r#"### read_file
Request to read the contents of a file at the specified path.
Use this when you need to examine the contents of an existing file you do not know the contents of, for example to analyze code, review text files, or extract information from configuration files.
This always takes ABSOLUTE paths as input.
May not be suitable for other types of binary files, as it returns the raw content as a string."#
        )
    }

    fn tool_input_format(&self) -> String {
        format!(
            r#"Parameters:
- fs_file_path: (required) The ABSOLUTE path of the file to read.

Usage:
<read_file>
<fs_file_path>
File path here
</fs_file_path>
</read_file>
"#
        )
    }

    fn get_evaluation_criteria(&self, _trajectory_length: usize) -> Vec<String> {
        vec![
            "Relevance of Requested Context: Ensure that the requested context is directly related to the problem and necessary for making progress.",
            "Avoiding Hallucinations: Verify that the agent is requesting context for code that actually exists in the codebase.",
            "Efficiency: Assess whether the agent is requesting an appropriate amount of context without overloading unnecessary information.",
            "Appropriateness of Action: Evaluate if requesting more context is logical at this point in the problem-solving process.",
        ].into_iter().map(|evaluation_criteria| evaluation_criteria.to_owned()).collect()
    }

    fn get_reward_scale(&self, _trajectory_length: usize) -> Vec<ToolRewardScale> {
        vec![
            ToolRewardScale::new(
                75,
                100,
                "The requested context is highly relevant, precise, and necessary for solving the problem; the agent avoids hallucinations.",
            ),
            ToolRewardScale::new(
                50,
                74,
                "The requested context is relevant and helpful, with minor issues in specificity or relevance.",
            ),
            ToolRewardScale::new(
                25,
                49,
                "The requested context is somewhat relevant but may include unnecessary information or lacks specificity.",
            ),
            ToolRewardScale::new(
                0,
                24,
                "The requested context has minimal relevance or includes excessive unnecessary information.",
            ),
            ToolRewardScale::new(
                -49,
                -1,
                "The requested context is irrelevant, demonstrates misunderstanding, or the agent is hallucinating code that doesn't exist.",
            ),
        ]
    }
}
