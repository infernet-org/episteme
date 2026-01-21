//! Configuration models for episteme.
//!
//! All I^R (resolvable ignorance) is parameterized here.
//! The user resolves these unknowns at runtime via config file.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level configuration for episteme.
///
/// I^R resolved: All configurable parameters are explicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// OpenRouter API configuration (primary endpoint, backward compatible)
    pub openrouter: OpenRouterConfig,

    /// Additional endpoints for on-prem or other aggregators (optional)
    #[serde(default)]
    pub endpoints: HashMap<String, EndpointConfig>,

    /// Worker pool configuration (generation)
    pub workers: PoolConfig,

    /// Judge pool configuration (evaluation)
    pub judges: PoolConfig,

    /// Generation settings
    pub generation: GenerationConfig,

    /// Output settings
    pub output: OutputConfig,
}

/// OpenRouter API configuration (primary endpoint).
///
/// K_i: OpenRouter is the default and primary endpoint.
/// B_i: Backward compatible with existing configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    /// API key (can also be set via OPENROUTER_API_KEY env var)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Environment variable name for API key
    #[serde(default = "default_openrouter_api_key_env")]
    pub api_key_env: String,

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

fn default_openrouter_api_key_env() -> String {
    "OPENROUTER_API_KEY".to_string()
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
            api_key_env: default_openrouter_api_key_env(),
            base_url: default_base_url(),
            timeout_secs: default_timeout(),
            max_retries: default_max_retries(),
        }
    }
}

/// Configuration for an additional LLM endpoint.
///
/// Supports on-prem (vLLM, TGI, Ollama, llama.cpp) and other aggregators
/// (Together AI, Fireworks, Groq).
///
/// K_i: All endpoints must be OpenAI-compatible (chat completions API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Base URL for the API (e.g., "http://localhost:11434/v1")
    pub base_url: String,

    /// API key (optional, can be omitted for local endpoints)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Environment variable name for API key
    #[serde(default)]
    pub api_key_env: Option<String>,

    /// Custom headers to include in requests
    /// Values can contain ${ENV_VAR} for environment variable expansion
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// Request timeout in seconds (default: 180)
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Maximum retries on failure (default: 3)
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: None,
            api_key_env: None,
            headers: HashMap::new(),
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

    /// Ensemble configuration (for judges only)
    #[serde(default)]
    pub ensemble: EnsembleConfig,
}

/// Ensemble judging configuration.
///
/// Epistemic foundation:
/// - B_i(single judge) → B_i(HIGH) via consensus from multiple judges
/// - I_i(judge bias) → K_i(disagreement) via explicit measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Enable ensemble judging (multiple judges per sample)
    #[serde(default)]
    pub enabled: bool,

    /// Number of judges per sample (default: 3)
    #[serde(default = "default_num_judges")]
    pub num_judges: usize,

    /// Aggregation strategy for combining scores
    #[serde(default)]
    pub strategy: AggregationStrategy,

    /// Standard deviation threshold for "low confidence" flag
    /// Samples with score_std_dev >= this are marked as "no consensus"
    #[serde(default = "default_disagreement_threshold")]
    pub disagreement_threshold: f64,

    /// Enable hierarchical mode: cheap judge first, ensemble only if uncertain
    #[serde(default)]
    pub hierarchical: bool,

    /// Score range considered "uncertain" in hierarchical mode
    /// Scores in [low, high] trigger full ensemble evaluation
    #[serde(default = "default_uncertain_range")]
    pub uncertain_range: (f64, f64),
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_judges: default_num_judges(),
            strategy: AggregationStrategy::default(),
            disagreement_threshold: default_disagreement_threshold(),
            hierarchical: false,
            uncertain_range: default_uncertain_range(),
        }
    }
}

fn default_num_judges() -> usize {
    3
}

fn default_disagreement_threshold() -> f64 {
    0.15
}

fn default_uncertain_range() -> (f64, f64) {
    (0.4, 0.7)
}

/// Strategy for aggregating multiple judge scores.
///
/// K_i: Median is most robust to outliers (one bad judge doesn't skew result).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AggregationStrategy {
    /// Median score (default, robust to outliers)
    #[default]
    Median,
    /// Average of all scores
    Average,
    /// Weighted average using model weights
    WeightedAverage,
}

/// Specification for a model.
///
/// K_i: Model ID format depends on the endpoint.
/// - OpenRouter: "provider/model" (e.g., "deepseek/deepseek-r1")
/// - vLLM/Ollama: model name as loaded (e.g., "llama3.3:70b")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Endpoint name (default: "openrouter")
    /// References [openrouter] or [endpoints.<name>] in config
    #[serde(default = "default_endpoint")]
    pub endpoint: String,

    /// Model ID (e.g., "deepseek/deepseek-r1" for OpenRouter, "llama3:70b" for Ollama)
    pub id: String,

    /// Human-readable label
    #[serde(default)]
    pub label: Option<String>,

    /// Selection weight (higher = more likely to be selected)
    #[serde(default = "default_weight")]
    pub weight: u32,

    /// Input price per 1M tokens (USD) - set to 0 for on-prem
    #[serde(default)]
    pub input_price_per_1m: f64,

    /// Output price per 1M tokens (USD) - set to 0 for on-prem
    #[serde(default)]
    pub output_price_per_1m: f64,

    /// Maximum tokens for this model
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for this model
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

fn default_endpoint() -> String {
    "openrouter".to_string()
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

    /// Resolve API key from config or environment for OpenRouter.
    ///
    /// B_i(api key available) → Result
    pub fn resolve_api_key(&self) -> Result<String, ConfigError> {
        // First check explicit api_key in config
        if let Some(key) = &self.openrouter.api_key {
            return Ok(expand_env_vars(key));
        }

        // Then check environment variable
        std::env::var(&self.openrouter.api_key_env).map_err(|_| ConfigError::MissingApiKey {
            endpoint: "openrouter".to_string(),
            env_var: self.openrouter.api_key_env.clone(),
        })
    }

    /// Resolve API key for a specific endpoint.
    ///
    /// B_i(api key available) → Result
    pub fn resolve_endpoint_api_key(
        &self,
        endpoint_name: &str,
    ) -> Result<Option<String>, ConfigError> {
        if endpoint_name == "openrouter" {
            return Ok(Some(self.resolve_api_key()?));
        }

        let endpoint = self
            .endpoints
            .get(endpoint_name)
            .ok_or_else(|| ConfigError::EndpointNotFound(endpoint_name.to_string()))?;

        // Check explicit api_key
        if let Some(key) = &endpoint.api_key {
            return Ok(Some(expand_env_vars(key)));
        }

        // Check environment variable
        if let Some(env_var) = &endpoint.api_key_env {
            match std::env::var(env_var) {
                Ok(key) => return Ok(Some(key)),
                Err(_) => {
                    return Err(ConfigError::MissingApiKey {
                        endpoint: endpoint_name.to_string(),
                        env_var: env_var.clone(),
                    });
                }
            }
        }

        // No API key configured (valid for local endpoints)
        Ok(None)
    }

    /// Get all unique endpoint names referenced by models.
    pub fn referenced_endpoints(&self) -> Vec<String> {
        let mut endpoints: Vec<String> = self
            .workers
            .models
            .iter()
            .chain(self.judges.models.iter())
            .map(|m| m.endpoint.clone())
            .collect();
        endpoints.sort();
        endpoints.dedup();
        endpoints
    }

    /// Validate that all referenced endpoints are configured.
    pub fn validate_endpoints(&self) -> Result<(), ConfigError> {
        for endpoint in self.referenced_endpoints() {
            if endpoint != "openrouter" && !self.endpoints.contains_key(&endpoint) {
                return Err(ConfigError::EndpointNotFound(endpoint));
            }
        }
        Ok(())
    }
}

/// Expand environment variables in a string.
///
/// Supports ${VAR_NAME} syntax.
/// If the variable is not set, the placeholder is left unchanged.
pub fn expand_env_vars(s: &str) -> String {
    let mut result = s.to_string();
    let re = regex::Regex::new(r"\$\{([^}]+)\}").unwrap();

    for cap in re.captures_iter(s) {
        let var_name = &cap[1];
        if let Ok(value) = std::env::var(var_name) {
            result = result.replace(&cap[0], &value);
        }
    }

    result
}

/// Expand environment variables in all headers.
pub fn expand_headers(headers: &HashMap<String, String>) -> HashMap<String, String> {
    headers
        .iter()
        .map(|(k, v)| (k.clone(), expand_env_vars(v)))
        .collect()
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

    #[error(
        "Missing API key for endpoint '{endpoint}': set {env_var} env var or api_key in config"
    )]
    MissingApiKey { endpoint: String, env_var: String },

    #[error("Endpoint not found: '{0}' (referenced by model but not configured in [endpoints.*])")]
    EndpointNotFound(String),
}
