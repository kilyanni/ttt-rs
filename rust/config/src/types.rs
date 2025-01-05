//! Model type enums, architecture definitions, and mix patterns.

use serde::{Deserialize, Serialize};

/// Inner model type for TTT layer.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[serde(rename_all = "kebab-case")]
pub enum InnerModel {
    #[default]
    Linear,
    LinearAdam,
    Mlp,
    Mlp2,
    Mlp3,
    Mlp4,
    FusedNaiveLinear,
    FusedNaiveMultiLinear,
    FusedTileLinear,
    FusedTileMultiLinear,
    D2dStreamingLinear,
    PtrStreamingLinear,
}

/// Position encoding type.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[serde(rename_all = "kebab-case")]
pub enum PosEncoding {
    #[default]
    Rope,
    RopeGlobal,
    None,
    Absolute,
}

/// Model size presets.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    #[cfg_attr(feature = "clap", value(name = "12m"))]
    #[serde(rename = "12m")]
    M12,
    #[default]
    #[cfg_attr(feature = "clap", value(name = "60m"))]
    #[serde(rename = "60m")]
    M60,
    #[cfg_attr(feature = "clap", value(name = "125m"))]
    #[serde(rename = "125m")]
    M125,
    #[cfg_attr(feature = "clap", value(name = "350m"))]
    #[serde(rename = "350m")]
    M350,
    #[cfg_attr(feature = "clap", value(name = "760m"))]
    #[serde(rename = "760m")]
    M760,
    #[cfg_attr(feature = "clap", value(name = "125m-h32"))]
    #[serde(rename = "125m-h32")]
    M125H32,
    #[cfg_attr(feature = "clap", value(name = "1b"))]
    #[serde(rename = "1b")]
    B1,
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::M12 => write!(f, "12m"),
            Self::M60 => write!(f, "60m"),
            Self::M125 => write!(f, "125m"),
            Self::M125H32 => write!(f, "125m-h32"),
            Self::M350 => write!(f, "350m"),
            Self::M760 => write!(f, "760m"),
            Self::B1 => write!(f, "1b"),
        }
    }
}

impl std::fmt::Display for InnerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear => write!(f, "linear"),
            Self::LinearAdam => write!(f, "linear-adam"),
            Self::Mlp => write!(f, "mlp"),
            Self::Mlp2 => write!(f, "mlp2"),
            Self::Mlp3 => write!(f, "mlp3"),
            Self::Mlp4 => write!(f, "mlp4"),
            Self::FusedNaiveLinear => write!(f, "fused-naive-linear"),
            Self::FusedNaiveMultiLinear => write!(f, "fused-naive-multi-linear"),
            Self::FusedTileLinear => write!(f, "fused-tile-linear"),
            Self::FusedTileMultiLinear => write!(f, "fused-tile-multi-linear"),
            Self::D2dStreamingLinear => write!(f, "d2d-streaming-linear"),
            Self::PtrStreamingLinear => write!(f, "ptr-streaming-linear"),
        }
    }
}

impl std::str::FromStr for InnerModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "linear-adam" => Ok(Self::LinearAdam),
            "mlp" => Ok(Self::Mlp),
            "mlp2" => Ok(Self::Mlp2),
            "mlp3" => Ok(Self::Mlp3),
            "mlp4" => Ok(Self::Mlp4),
            "fused-naive" | "fused-naive-linear" | "fused" | "fused-linear" => {
                Ok(Self::FusedNaiveLinear)
            }
            "fused-naive-multi" | "fused-naive-multi-linear" => Ok(Self::FusedNaiveMultiLinear),
            "fused-tile" | "fused-tile-linear" => Ok(Self::FusedTileLinear),
            "fused-tile-multi" | "fused-tile-multi-linear" => Ok(Self::FusedTileMultiLinear),
            "d2d-streaming" | "d2d-streaming-linear" => Ok(Self::D2dStreamingLinear),
            "ptr-streaming" | "ptr-streaming-linear" => Ok(Self::PtrStreamingLinear),
            _ => Err(format!(
                "unknown layer type '{s}'. Use: linear, linear-adam, mlp, mlp2, mlp3, mlp4, \
                 fused-naive, fused-naive-multi, fused-tile, fused-tile-multi"
            )),
        }
    }
}

/// A layer type mixing pattern that cycles over layers.
///
/// Examples:
/// - `"linear"` → all layers use Linear
/// - `"linear,mlp"` → alternates: Linear, MLP, Linear, MLP, ...
/// - `"4*linear,mlp"` → 4 Linear then 1 MLP, repeated cyclically
/// - `"6*linear,6*fused-tile"` → first half linear, second half fused (for 12-layer model)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixPattern(pub Vec<InnerModel>);

impl MixPattern {
    /// A uniform pattern (all layers use the same type).
    pub fn uniform(inner: InnerModel) -> Self {
        Self(vec![inner])
    }

    /// Get the inner model type for a given layer index (cycles the pattern).
    pub fn get(&self, layer_idx: usize) -> InnerModel {
        self.0[layer_idx % self.0.len()]
    }

    /// Returns true if all layers use the same type.
    pub fn is_uniform(&self) -> bool {
        self.0.iter().all(|t| *t == self.0[0])
    }
}

impl Default for MixPattern {
    fn default() -> Self {
        Self::uniform(InnerModel::default())
    }
}

impl std::str::FromStr for MixPattern {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut pattern = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((count_str, type_str)) = part.split_once('*') {
                let count: usize = count_str
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid multiplier '{count_str}' in '{part}'"))?;
                let inner: InnerModel = type_str.trim().parse()?;
                pattern.extend(std::iter::repeat_n(inner, count));
            } else {
                pattern.push(part.parse()?);
            }
        }
        if pattern.is_empty() {
            return Err("empty mix pattern".to_string());
        }
        Ok(Self(pattern))
    }
}

impl std::fmt::Display for MixPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut i = 0;
        let mut first = true;
        while i < self.0.len() {
            let ty = self.0[i];
            let mut count = 1;
            while i + count < self.0.len() && self.0[i + count] == ty {
                count += 1;
            }
            if !first {
                write!(f, ",")?;
            }
            if count > 1 {
                write!(f, "{count}*{ty}")?;
            } else {
                write!(f, "{ty}")?;
            }
            i += count;
            first = false;
        }
        Ok(())
    }
}

impl Serialize for MixPattern {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for MixPattern {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for PosEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rope => write!(f, "rope"),
            Self::RopeGlobal => write!(f, "rope-global"),
            Self::None => write!(f, "none"),
            Self::Absolute => write!(f, "absolute"),
        }
    }
}

/// Data type selection.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum DType {
    #[default]
    F32,
    F16,
    BF16,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
        }
    }
}

/// Model architecture (derived from ModelSize + vocab_size).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
pub struct ModelArch {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

impl ModelArch {
    pub fn from_size(size: ModelSize, vocab_size: usize) -> Self {
        match size {
            ModelSize::M12 => Self {
                hidden_size: 256,
                num_hidden_layers: 6,
                num_heads: 4,
                intermediate_size: 512,
                vocab_size,
            },
            ModelSize::M60 => Self {
                hidden_size: 512,
                num_hidden_layers: 6,
                num_heads: 8,
                intermediate_size: 768,
                vocab_size,
            },
            ModelSize::M125 => Self {
                hidden_size: 768,
                num_hidden_layers: 12,
                num_heads: 12,
                intermediate_size: 2048,
                vocab_size,
            },
            ModelSize::M125H32 => Self {
                hidden_size: 768,
                num_hidden_layers: 12,
                num_heads: 24, // head_dim = 768/24 = 32
                intermediate_size: 2048,
                vocab_size,
            },
            ModelSize::M350 => Self {
                hidden_size: 1024,
                num_hidden_layers: 24,
                num_heads: 16,
                intermediate_size: 2736,
                vocab_size,
            },
            ModelSize::M760 => Self {
                hidden_size: 1536,
                num_hidden_layers: 24,
                num_heads: 16,
                intermediate_size: 4096,
                vocab_size,
            },
            ModelSize::B1 => Self {
                hidden_size: 2048,
                num_hidden_layers: 24,
                num_heads: 32,
                intermediate_size: 5504,
                vocab_size,
            },
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}
