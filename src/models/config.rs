//! Configuration models for dpogen.
//!
//! All I^R (resolvable ignorance) is parameterized here.
//! The user resolves these unknowns at runtime via config file.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level configuration for dpogen.
///
/// I^R resolved: All configurable parameters are explicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// OpenRouter API configuration
    pub openrouter: OpenRouterConfig,

    /// Worker pool configuration (generation)
    pub workers: PoolConfig,

    /// Judge pool configuration (evaluation)
    pub judges: PoolConfig,

    /// Generation settings
    pub generation: GenerationConfig,

    /// Output settings
    pub output: OutputConfig,
}

/// OpenRouter API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    /// API key (can also be set via OPENROUTER_API_KEY env var)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Base URL for OpenRouter API
    #[serde(default = "default_base_url")]
    pub base_url: String,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Maximum retries on failure
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_base_url() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

fn default_timeout() -> u64 {
    180
}

fn default_max_retries() -> u32 {
    3
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: default_base_url(),
            timeout_secs: default_timeout(),
            max_retries: default_max_retries(),
        }
    }
}

/// Pool configuration (workers or judges).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Number of concurrent workers in the pool
    pub size: usize,

    /// Models to use in this pool
    pub models: Vec<ModelSpec>,
}

/// Specification for a model.
///
/// K_i: Model ID must be a valid OpenRouter model identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// OpenRouter model ID (e.g., "deepseek/deepseek-r1")
    pub id: String,

    /// Human-readable label
    #[serde(default)]
    pub label: Option<String>,

    /// Selection weight (higher = more likely to be selected)
    #[serde(default = "default_weight")]
    pub weight: u32,

    /// Input price per 1M tokens (USD)
    #[serde(default)]
    pub input_price_per_1m: f64,

    /// Output price per 1M tokens (USD)
    #[serde(default)]
    pub output_price_per_1m: f64,

    /// Maximum tokens for this model
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for this model
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

fn default_weight() -> u32 {
    1
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_temperature() -> f64 {
    0.7
}

/// Generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Path to system prompt file
    pub system_prompt: PathBuf,

    /// Path to judge prompt file
    pub judge_prompt: PathBuf,

    /// Score threshold for approval (0.0 - 1.0)
    #[serde(default = "default_threshold")]
    pub approval_threshold: f64,

    /// Number of responses to generate per problem (for DPO)
    #[serde(default = "default_responses_per_problem")]
    pub responses_per_problem: usize,
}

fn default_threshold() -> f64 {
    0.85
}

fn default_responses_per_problem() -> usize {
    1
}

/// Output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output file path
    pub path: PathBuf,

    /// Checkpoint directory for resume support
    #[serde(default)]
    pub checkpoint_dir: Option<PathBuf>,

    /// Whether to include rejected samples in output
    #[serde(default)]
    pub include_rejected: bool,

    /// Whether to track costs
    #[serde(default = "default_true")]
    pub track_costs: bool,
}

fn default_true() -> bool {
    true
}

impl Config {
    /// Load configuration from a TOML file.
    ///
    /// B_i(file exists) → Result
    /// B_i(file is valid TOML) → Result
    pub fn from_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::FileRead {
            path: path.to_owned(),
            source: e,
        })?;

        toml::from_str(&content).map_err(|e| ConfigError::Parse {
            path: path.to_owned(),
            source: e,
        })
    }

    /// Resolve API key from config or environment.
    ///
    /// B_i(api key available) → Result
    pub fn resolve_api_key(&self) -> Result<String, ConfigError> {
        if let Some(key) = &self.openrouter.api_key {
            return Ok(key.clone());
        }

        std::env::var("OPENROUTER_API_KEY").map_err(|_| ConfigError::MissingApiKey)
    }
}

/// Configuration errors.
///
/// Epistemic origin:
/// - B_i falsified: File not found, parse error
/// - I^B materialized: Missing required values
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read config file {path}: {source}")]
    FileRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Failed to parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },

    #[error("Missing API key: set OPENROUTER_API_KEY env var or api_key in config")]
    MissingApiKey,
}
