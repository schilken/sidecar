//! Searches the files in a given directory given a regex
//! Can be used by the agent to grep for this in the repository or in a sub-directory

use async_trait::async_trait;
use tokio::io::AsyncBufReadExt;
use tokio::{io::BufReader, process::Command};

use crate::agentic::tool::r#type::ToolRewardScale;
use crate::agentic::tool::{errors::ToolError, input::ToolInput, output::ToolOutput, r#type::Tool};
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// Magic number which came into existence to not break LLM context windows
/// This limits the number of results to 250 hits, if its more than that, the LLM
/// or the human needs to be more specific
const MAX_RESULTS: usize = 250;

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", content = "data")]
enum RipgrepEvent {
    #[serde(rename = "match")]
    Match {
        path: RipgrepPath,
        lines: RipgrepLines,
        line_number: usize,
    },
    #[serde(rename = "context")]
    Context {
        lines: RipgrepLines,
        line_number: usize,
    },
}

#[derive(Debug, serde::Deserialize)]
struct RipgrepPath {
    text: String,
}

#[derive(Debug, serde::Deserialize)]
struct RipgrepLines {
    text: String,
}

#[derive(Debug)]
struct SearchResult {
    file: String,
    line: usize,
    match_line: String,
    before_context: Vec<String>,
    after_context: Vec<String>,
}

impl SearchResult {
    fn format_results(results: Vec<Self>, directory_path: String) -> String {
        use std::collections::HashMap;

        let mut grouped_results: HashMap<PathBuf, Vec<&SearchResult>> = HashMap::new();
        let mut output = String::new();

        if results.len() >= MAX_RESULTS {
            output.push_str(&format!(
                "Showing first {} of {}+ results. Use a more specific search if necessary.\n\n",
                MAX_RESULTS, MAX_RESULTS
            ));
        } else {
            output.push_str(&format!(
                "Found {}.\n\n",
                if results.len() == 1 {
                    "1 result".to_string()
                } else {
                    format!("{} results", results.len())
                }
            ));
        }

        for result in results.iter().take(MAX_RESULTS) {
            let file_path = Path::new(&result.file);
            let directory_path = Path::new(&directory_path);
            let directory_path = directory_path.join(file_path);
            grouped_results
                .entry(directory_path)
                .or_default()
                .push(result);
        }

        for (file_path, file_results) in grouped_results {
            output.push_str(&format!(
                "{}\n│----\n",
                file_path
                    .as_os_str()
                    .to_str()
                    .expect("file_formatting to work unless something horrendou happens to the underlying OS")
            ));

            for (index, result) in file_results.iter().enumerate() {
                let all_lines = result
                    .before_context
                    .iter()
                    .chain(std::iter::once(&result.match_line))
                    .chain(result.after_context.iter());

                for line in all_lines {
                    output.push_str(&format!("│{}\n", line.trim_end()));
                }

                if index < file_results.len() - 1 {
                    output.push_str("│----\n");
                }
            }

            output.push_str("│----\n\n");
        }

        output.trim_end().to_string()
    }
}

#[derive(Debug, Clone)]
pub struct SearchFileContentWithRegexOutput {
    formatted_response: String,
}

impl SearchFileContentWithRegexOutput {
    pub fn response(&self) -> &str {
        &self.formatted_response
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchFileContentInputPartial {
    directory_path: String,
    regex_pattern: String,
    file_pattern: Option<String>,
}

impl SearchFileContentInputPartial {
    pub fn new(
        directory_path: String,
        regex_pattern: String,
        file_pattern: Option<String>,
    ) -> Self {
        Self {
            directory_path,
            regex_pattern,
            file_pattern,
        }
    }

    pub fn directory_path(&self) -> &str {
        &self.directory_path
    }

    pub fn regex_pattern(&self) -> &str {
        &self.regex_pattern
    }

    pub fn file_pattern(&self) -> Option<&str> {
        self.file_pattern.as_deref()
    }

    pub fn to_string(&self) -> String {
        format!(
            r#"<search_files>
<directory_path>
{}
</directory_path>
<regex_pattern>
{}
</regex_pattern>
<file_pattern>
{}
</file_pattern>
</search_files>"#,
            self.directory_path,
            self.regex_pattern,
            self.file_pattern
                .clone()
                .unwrap_or("not provided".to_owned())
        )
    }
}

#[derive(Debug, Clone)]
pub struct SearchFileContentInput {
    directory_path: String,
    regex_pattern: String,
    file_pattern: Option<String>,
    editor_url: String,
}

impl SearchFileContentInput {
    pub fn new(
        directory_path: String,
        regex_pattern: String,
        file_pattern: Option<String>,
        editor_url: String,
    ) -> Self {
        Self {
            directory_path,
            regex_pattern,
            file_pattern,
            editor_url,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct EditorRipGrepPath {
    rip_grep_path: String,
}

pub struct SearchFileContentClient {
    client: reqwest::Client,
}

impl SearchFileContentClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Tool for SearchFileContentClient {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        let context = input.is_search_file_content_with_regex()?;
        // first grab the rip-grep path from the editor
        let endpoint = context.editor_url.to_owned() + "/rip_grep_path";
        let response = self
            .client
            .post(endpoint)
            .send()
            .await
            .map_err(|_e| ToolError::ErrorCommunicatingWithEditor)?;
        let response: EditorRipGrepPath = response
            .json()
            .await
            .map_err(|_e| ToolError::SerdeConversionFailed)?;

        let binary_path = response.rip_grep_path;
        let regex_pattern = &context.regex_pattern;
        let file_pattern = &context.file_pattern.unwrap_or("*".to_owned());
        let args = vec![
            "--json",
            "-e",
            regex_pattern,
            "--glob",
            file_pattern,
            "--context",
            "1",
            &context.directory_path,
        ];

        println!("search_files::args::({:?})", args);

        let mut child = Command::new(binary_path)
            .args(&args)
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::IOError(e))?;

        // now we can read the output from the child line by line and parse it out properly
        let stdout = child.stdout.take();
        if let None = stdout {
            println!("stdout is empty over here");
            return Err(ToolError::OutputStreamNotPresent);
        }

        let stdout = stdout.expect("Failed to capture stdout");
        let reader = BufReader::new(stdout).lines();

        let mut output = String::new();
        let mut line_count = 0;
        let max_lines = MAX_RESULTS * 5;

        tokio::pin!(reader);

        while let Some(line) = reader.next_line().await? {
            if line_count >= max_lines {
                break;
            }
            output.push_str(&line);
            output.push('\n');
            line_count += 1;
        }

        let _status = child.wait().await?;
        // even if there were errors we still want to read from this
        // if !status.success() {
        //     return Err(ToolError::OutputStreamNotPresent);
        // }

        let mut results: Vec<SearchResult> = Vec::new();
        let mut current_result: Option<SearchResult> = None;

        for line in output.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let parsed: RipgrepEvent = match serde_json::from_str(line) {
                Ok(event) => event,
                Err(_err) => {
                    continue;
                }
            };

            match parsed {
                RipgrepEvent::Match {
                    path,
                    lines,
                    line_number,
                    ..
                } => {
                    if let Some(result) = current_result.take() {
                        results.push(result);
                    }
                    current_result = Some(SearchResult {
                        file: path.text,
                        line: line_number,
                        match_line: lines.text,
                        before_context: Vec::new(),
                        after_context: Vec::new(),
                    });
                }
                RipgrepEvent::Context {
                    lines, line_number, ..
                } => {
                    if let Some(ref mut result) = current_result {
                        if line_number < result.line {
                            result.before_context.push(lines.text);
                        } else {
                            result.after_context.push(lines.text);
                        }
                    }
                }
            }
        }

        if let Some(result) = current_result {
            results.push(result);
        }

        Ok(ToolOutput::search_file_content_with_regex(
            SearchFileContentWithRegexOutput {
                formatted_response: SearchResult::format_results(results, context.directory_path),
            },
        ))
    }

    fn tool_description(&self) -> String {
        format!(
            r#"### search_files
Request to perform a regex search across files in a specified directory, providing context-rich results.
This tool searches for patterns or specific content across multiple files, displaying each match with encapsulating context."#
        )
    }

    fn tool_input_format(&self) -> String {
        format!(
            r#"Parameters:
- directory_path: (required) The absolute path of the directory to search in. This directory will be recursively searched.
- regex_pattern: (required) The regular expression pattern to search for. Uses Rust regex syntax.
- file_pattern: (optional) Glob pattern to filter files (e.g., '*.ts' for TypeScript files). If not provided, it will search all files (*).

Usage:
<search_files>
<directory_path>
Directory path here
</directory_path>
<regex_pattern>
Your regex pattern here
</regex_pattern>
<file_pattern>
file pattern here (optional)
</file_pattern>
</search_files>
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
