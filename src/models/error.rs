//! Error types for dpogen.
//!
//! Epistemic taxonomy:
//! - B_i falsified: Expected failures (not found, invalid input)
//! - I^B materialized: Infrastructure failures (network, timeout)
//! - K_i violated: Internal invariant violations (bugs)

use thiserror::Error;

/// Top-level error type for dpogen.
#[derive(Debug, Error)]
pub enum DpogenError {
    // ═══════════════════════════════════════════════════════════════════
    // B_i FALSIFIED — Belief proven wrong (expected failures)
    // ═══════════════════════════════════════════════════════════════════

    #[error("Configuration error: {0}")]
    Config(#[from] super::ConfigError),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Problem not found: {0}")]
    ProblemNotFound(String),

    #[error("No samples generated for problem: {0}")]
    NoSamplesGenerated(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    // ═══════════════════════════════════════════════════════════════════
    // I^B MATERIALIZED — Bounded ignorance became known-bad
    // ═══════════════════════════════════════════════════════════════════

    #[error("OpenRouter API error: {0}")]
    OpenRouterApi(#[from] OpenRouterError),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Request timeout after {0:?}")]
    Timeout(std::time::Duration),

    #[error("Rate limited: retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: f64 },

    #[error("IO error: {context}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    // ═══════════════════════════════════════════════════════════════════
    // K_i VIOLATED — Invariant broken (bug, should not happen)
    // ═══════════════════════════════════════════════════════════════════

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Channel closed unexpectedly")]
    ChannelClosed,

    // ═══════════════════════════════════════════════════════════════════
    // I^B UNRESOLVABLE — Truly unknown failure
    // ═══════════════════════════════════════════════════════════════════

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// OpenRouter API specific errors.
#[derive(Debug, Error)]
pub enum OpenRouterError {
    #[error("Authentication failed: invalid API key")]
    AuthenticationFailed,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Rate limited by OpenRouter: {message}")]
    RateLimited {
        message: String,
        retry_after_secs: Option<f64>,
    },

    #[error("API error (status {status}): {message}")]
    ApiError {
        status: u16,
        message: String,
    },

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Request failed after {attempts} attempts: {last_error}")]
    MaxRetriesExceeded {
        attempts: u32,
        last_error: String,
    },
}

impl DpogenError {
    /// Create an IO error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_)
            | Self::RateLimited { .. }
            | Self::Network(_)
            | Self::OpenRouterApi(OpenRouterError::RateLimited { .. })
        )
    }

    /// Get retry delay hint in seconds, if applicable.
    pub fn retry_after(&self) -> Option<f64> {
        match self {
            Self::RateLimited { retry_after_secs } => Some(*retry_after_secs),
            Self::OpenRouterApi(OpenRouterError::RateLimited { retry_after_secs, .. }) => {
                *retry_after_secs
            }
            _ => None,
        }
    }
}

/// Result type alias for dpogen.
pub type Result<T> = std::result::Result<T, DpogenError>;
