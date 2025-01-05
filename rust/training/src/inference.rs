use std::sync::Arc;

use burn::{
    prelude::*,
    record::{DefaultRecorder, Recorder},
};
use ttt_data::{Tokenizer, TokenizerTrait};
use ttt_fused::FusedTttBackend;

use crate::{text_generation::TTTTextGenerationModel, training::TTTTrainingConfig};

pub struct TTTTextGenerator<B: FusedTttBackend> {
    model: TTTTextGenerationModel<B>,
    tokenizer: Arc<dyn TokenizerTrait>,
    device: B::Device,
}

impl<B: FusedTttBackend> TTTTextGenerator<B> {
    /// Load a trained model from artifacts directory with a specific tokenizer
    pub fn load_from_artifacts(
        artifact_dir: &str,
        device: B::Device,
        tokenizer: Tokenizer,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::text_generation::TTTTextGenerationConfig;

        let config: TTTTrainingConfig =
            TTTTrainingConfig::load(format!("{artifact_dir}/config.json"))?;

        let tokenizer = Arc::new(tokenizer);

        let model_config =
            TTTTextGenerationConfig::new(config.model_config.clone(), config.pad_token);

        let mut model = model_config.init(&config.model_config.ttt.layer_type, &device);

        let record =
            DefaultRecorder::new().load(format!("{artifact_dir}/model").into(), &device)?;
        model = model.load_record(record);

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn new(
        model: TTTTextGenerationModel<B>,
        tokenizer: Arc<dyn TokenizerTrait>,
        device: B::Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> String {
        let input_tokens = self.tokenizer.encode(prompt, true);
        let input_tokens_i32: Vec<i32> = input_tokens
            .iter()
            .map(|&x| x.try_into().expect("Int casting error"))
            .collect();
        let input_tensor =
            Tensor::<B, 1, Int>::from_ints(input_tokens_i32.as_slice(), &self.device)
                .reshape([1, input_tokens.len()]);

        let generated_tokens =
            self.model
                .generate(input_tensor, max_new_tokens, temperature, top_k);

        let generated_ids: Vec<usize> = generated_tokens
            .into_data()
            .as_slice::<i32>()
            .unwrap()
            .iter()
            .map(|&x| x.try_into().expect("Int casting error"))
            .collect();

        self.tokenizer.decode(&generated_ids, true)
    }

    /// Simple interactive generation session
    pub fn interactive_session(&self) {
        use std::io::{self, Write};

        println!("TTT Text Generator - Interactive Session");
        println!("Type 'quit' to exit, 'help' for commands");
        println!("Default settings: max_tokens=50, temperature=0.8, top_k=40");
        println!();

        let mut max_tokens = 50;
        let mut temperature = 0.8;
        let mut top_k = Some(40);

        loop {
            print!("> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            match input {
                "quit" | "exit" => {
                    println!("Goodbye!");
                    break;
                }
                "help" => {
                    println!("Commands:");
                    println!("  quit/exit - Exit the session");
                    println!("  help - Show this help");
                    println!("  set max_tokens <n> - Set max tokens to generate");
                    println!("  set temperature <f> - Set temperature (0.0-2.0)");
                    println!("  set top_k <n> - Set top-k sampling (or 'none')");
                    println!("  show settings - Show current settings");
                    println!("  Or just type a prompt to generate text");
                }
                s if s.starts_with("set ") => {
                    let parts: Vec<&str> = s.split_whitespace().collect();
                    if parts.len() == 3 {
                        match parts[1] {
                            "max_tokens" => {
                                if let Ok(val) = parts[2].parse::<usize>() {
                                    max_tokens = val;
                                    println!("Max tokens set to {val}");
                                } else {
                                    println!("Invalid value for max_tokens");
                                }
                            }
                            "temperature" => {
                                if let Ok(val) = parts[2].parse::<f32>() {
                                    temperature = val.clamp(0.0, 2.0);
                                    println!("Temperature set to {temperature}");
                                } else {
                                    println!("Invalid value for temperature");
                                }
                            }
                            "top_k" => {
                                if parts[2] == "none" {
                                    top_k = None;
                                    println!("Top-k disabled");
                                } else if let Ok(val) = parts[2].parse::<usize>() {
                                    top_k = Some(val);
                                    println!("Top-k set to {val}");
                                } else {
                                    println!("Invalid value for top_k");
                                }
                            }
                            _ => println!("Unknown setting: {}", parts[1]),
                        }
                    } else {
                        println!("Usage: set <setting> <value>");
                    }
                }
                "show settings" => {
                    println!("Current settings:");
                    println!("  max_tokens: {max_tokens}");
                    println!("  temperature: {temperature}");
                    println!("  top_k: {top_k:?}");
                }
                prompt if !prompt.is_empty() => {
                    println!("Generating...");
                    let start = std::time::Instant::now();

                    let generated = self.generate_text(prompt, max_tokens, temperature, top_k);

                    let duration = start.elapsed();
                    println!("\n--- Generated Text ---");
                    println!("{generated}");
                    println!("\n--- End (took {:.2}s) ---\n", duration.as_secs_f32());
                }
                _ => {}
            }
        }
    }
}

/// Quick text generation with default settings
pub fn generate<B: FusedTttBackend>(
    artifact_dir: &str,
    device: B::Device,
    prompt: &str,
    tokenizer: Tokenizer,
) -> Result<String, Box<dyn std::error::Error>> {
    let generator = TTTTextGenerator::<B>::load_from_artifacts(artifact_dir, device, tokenizer)?;
    Ok(generator.generate_text(prompt, 50, 0.8, Some(40)))
}

/// Quick interactive session
pub fn interactive<B: FusedTttBackend>(
    artifact_dir: &str,
    device: B::Device,
    tokenizer: Tokenizer,
) -> Result<(), Box<dyn std::error::Error>> {
    let generator = TTTTextGenerator::<B>::load_from_artifacts(artifact_dir, device, tokenizer)?;
    generator.interactive_session();
    Ok(())
}
