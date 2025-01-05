//! Configuration parsing for the training harness.

use std::path::Path;

use serde::{Deserialize, Serialize};
use ttt_config::TrainParams;

/// Top-level configuration loaded from TOML (raw, before merging).
#[derive(Debug, Clone, Deserialize)]
struct RawHarnessConfig {
    pub harness: HarnessSettings,
    #[serde(default = "empty_table")]
    pub defaults: toml::Value,
    #[serde(default)]
    pub runs: Vec<toml::Value>,
}

fn empty_table() -> toml::Value {
    toml::Value::Table(toml::map::Map::new())
}

/// Top-level configuration after merging defaults into runs.
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub harness: HarnessSettings,
    pub runs: Vec<RunConfig>,
}

/// Harness-level settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessSettings {
    /// Total available VRAM in GB.
    #[serde(default = "default_total_vram")]
    pub total_vram_gb: f64,
    /// Safety margin (e.g., 0.10 = 10% buffer).
    #[serde(default = "default_vram_margin")]
    pub vram_margin: f64,
    /// What to do on failure: "retry", "skip", or "abort".
    #[serde(default = "default_on_failure")]
    pub on_failure: FailurePolicy,
    /// Maximum retry attempts per run.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Path to the state file for persistence.
    #[serde(default = "default_state_file")]
    pub state_file: String,
    /// Runpod integration settings.
    #[serde(default)]
    pub runpod: RunpodSettings,
    /// `RUST_LOG` value to pass to child training processes.
    #[serde(default)]
    pub rust_log: Option<String>,
    /// Kill process if no progress update after the first one within this many seconds.
    #[serde(default)]
    pub hang_timeout_secs: Option<u64>,
    /// Kill process if no stdout/stderr activity at all within this many seconds.
    #[serde(default)]
    pub idle_timeout_secs: Option<u64>,
    /// Grace period (seconds) applied to all watchdogs when a new process starts,
    /// to let the GPU settle.
    #[serde(default = "default_settle_grace")]
    pub settle_grace_secs: u64,
}

/// What to do when a run fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FailurePolicy {
    /// Retry the run up to `max_retries` times.
    #[default]
    Retry,
    /// Skip the run and continue with others.
    Skip,
    /// Abort the entire harness.
    Abort,
}

/// Runpod shutdown settings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunpodSettings {
    /// Whether to auto-shutdown when all runs complete.
    #[serde(default)]
    pub enabled: bool,
}

/// Configuration for a single training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub name: String,
    #[serde(flatten)]
    pub params: TrainParams,
    #[serde(default)]
    pub out: Option<String>,
}

impl RunConfig {
    /// Build command-line arguments for `ttt train`.
    #[must_use]
    pub fn to_args(&self) -> Vec<String> {
        let mut args = vec!["train".to_string()];
        args.extend(self.params.to_cli_args());
        if let Some(ref out) = self.out {
            args.extend(["--out".into(), out.clone()]);
        }
        args
    }

    /// Get the output directory for this run.
    #[must_use]
    pub fn artifact_dir(&self) -> String {
        self.out
            .clone()
            .unwrap_or_else(|| format!("./artifacts/{}", self.name))
    }
}

fn default_total_vram() -> f64 {
    192.0
}

fn default_vram_margin() -> f64 {
    0.10
}

fn default_on_failure() -> FailurePolicy {
    FailurePolicy::Retry
}

fn default_max_retries() -> u32 {
    2
}

fn default_settle_grace() -> u64 {
    30
}

fn default_state_file() -> String {
    "./harness_state.json".to_string()
}

/// Merge two TOML tables, with `overlay` values taking precedence.
fn merge_toml(base: &toml::Value, overlay: &toml::Value) -> toml::Value {
    match (base, overlay) {
        (toml::Value::Table(base_map), toml::Value::Table(overlay_map)) => {
            let mut merged = base_map.clone();
            for (k, v) in overlay_map {
                merged.insert(
                    k.clone(),
                    if let Some(base_v) = base_map.get(k) {
                        merge_toml(base_v, v)
                    } else {
                        v.clone()
                    },
                );
            }
            toml::Value::Table(merged)
        }
        (_, overlay) => overlay.clone(),
    }
}

impl HarnessConfig {
    /// Load configuration from a TOML file, merging defaults into each run.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::Io(path.as_ref().display().to_string(), e))?;
        let raw: RawHarnessConfig =
            toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;

        // Get out_prefix from defaults if present
        let out_prefix = raw
            .defaults
            .get("out_prefix")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Merge defaults into each run
        let runs: Vec<RunConfig> = raw
            .runs
            .into_iter()
            .map(|run_value| {
                let mut merged = merge_toml(&raw.defaults, &run_value);
                // Apply out_prefix if out not specified
                if let toml::Value::Table(ref mut t) = merged {
                    t.remove("out_prefix"); // Don't pass to RunConfig
                    if !t.contains_key("out")
                        && let (Some(prefix), Some(name)) =
                            (&out_prefix, t.get("name").and_then(|n| n.as_str()))
                    {
                        t.insert(
                            "out".into(),
                            toml::Value::String(format!("{prefix}/{name}")),
                        );
                    }
                }
                merged
                    .try_into()
                    .map_err(|e: toml::de::Error| ConfigError::Parse(e.to_string()))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            harness: raw.harness,
            runs,
        })
    }

    /// Get the usable VRAM after applying the safety margin.
    #[must_use]
    pub fn usable_vram_gb(&self) -> f64 {
        self.harness.total_vram_gb * (1.0 - self.harness.vram_margin)
    }
}

/// Errors that can occur when loading configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file {0}: {1}")]
    Io(String, std::io::Error),
    #[error("failed to parse config: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use ttt_config::ModelSize;

    use super::*;

    fn parse_config(toml_str: &str) -> HarnessConfig {
        // Write to unique temp file since load() reads from file
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_harness.toml");
        std::fs::write(&path, toml_str).unwrap();
        HarnessConfig::load(&path).unwrap()
    }

    #[test]
    fn test_parse_minimal_config() {
        let config = parse_config(
            r#"
[harness]
total_vram_gb = 192

[[runs]]
name = "test-run"
size = "60m"
"#,
        );
        assert_eq!(config.harness.total_vram_gb, 192.0);
        assert_eq!(config.runs.len(), 1);
        assert_eq!(config.runs[0].name, "test-run");
    }

    #[test]
    fn test_defaults_applied() {
        let config = parse_config(
            r#"
[harness]
total_vram_gb = 192

[defaults]
tokenizer = "gpt2"
epochs = 20
batch = 32

[[runs]]
name = "run1"
size = "60m"

[[runs]]
name = "run2"
size = "125m"
batch = 8
"#,
        );
        assert_eq!(config.runs[0].params.tokenizer, "gpt2");
        assert_eq!(config.runs[0].params.train.epochs, 20);
        assert_eq!(config.runs[0].params.train.batch, 32);

        assert_eq!(config.runs[1].params.tokenizer, "gpt2");
        assert_eq!(config.runs[1].params.train.epochs, 20);
        assert_eq!(config.runs[1].params.train.batch, 8); // Overridden
    }

    #[test]
    fn test_to_args() {
        let run = RunConfig {
            name: "test".to_string(),
            params: TrainParams {
                size: ModelSize::M60,
                train: ttt_config::TrainConfig {
                    batch: 32,
                    ..Default::default()
                },
                ..Default::default()
            },
            out: Some("./out/test".to_string()),
        };
        let args = run.to_args();
        assert!(args.contains(&"train".to_string()));
        assert!(args.contains(&"--size".to_string()));
        assert!(args.contains(&"60m".to_string()));
        assert!(args.contains(&"--batch".to_string()));
        assert!(args.contains(&"32".to_string()));
    }
}
