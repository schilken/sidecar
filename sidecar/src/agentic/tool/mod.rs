//! Contains the tools which will be used by the agent
//! Some example tools are as follows:
//! Ask documentation: Asks the user for documentation
//! Ask User: Ask the user for more information
//! Code editing: Edits the code at a particular location
//! Search: Searchs for a keyword throughout the codebase (includes file or
//! something else)
//! Go-to-Definition: Goes to the defintiion for a function/class or symbol
//! Go-to-References: Goes to the references for a function/class or symbol
//! FS: Allows the file system to be written or read from
//! Folder outline: Shows the outline from the current working file or just
//! mirrors the files which were open recently, just an outline of the files which
//! are open in the editor (This is much better, and reflects how it looks in the editor)
//! Language server: Gets the diagnostics for the current file if required or over the workspace
//! Terminal: Use the terminal to run operations or something

pub mod base;
pub mod broker;
pub mod code_edit;
pub mod code_symbol;
pub mod errors;
pub mod filtering;
pub mod grep;
pub mod input;
pub mod lsp;
pub mod output;
pub mod rerank;