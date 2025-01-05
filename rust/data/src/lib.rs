//! TTT Data - Data loading and tokenization utilities
//!
//! This crate provides:
//! - `Tokenizer` - HuggingFace tokenizer wrapper
//! - `TextDataset` - Dataset for text generation
//! - `TextGenerationBatcher` - Batching for text generation training
//! - `PretokenizedDataset` - Pre-tokenized dataset for efficient loading

pub mod batcher;
pub mod dataset;
pub mod pretokenized;
pub mod tokenizer;

// Re-export commonly used items
pub use batcher::{TextGenerationBatch, TextGenerationBatcher, TrainingTextGenerationBatch};
pub use dataset::{TextDataset, TextGenerationItem};
pub use pretokenized::{PreTokenizedDataset, TokenBatcher, TokenizedItem, load_or_pretokenize};
pub use tokenizer::{Tokenizer, TokenizerTrait};
