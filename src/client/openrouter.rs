//! OpenRouter API client.
//!
//! Epistemic foundation:
//! - K_i: OpenRouter provides unified access to frontier models
//! - B_i: API will respond within timeout (might fail)
//! - B_i: Response will be valid JSON (might fail)
//! - I^B: Network availability unknowable → retry with backoff

use crate::client::RateLimiter;
use crate::models::{DpogenError, ModelSpec, OpenRouterError, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::debug;

/// Message in a chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Chat completion request payload.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

/// Chat completion response.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// OpenRouter API error response.
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

/// Response from a completion request.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Generated content
    pub content: String,
    /// Model used (may differ from requested)
    pub model: String,
    /// Input tokens
    pub input_tokens: u32,
    /// Output tokens
    pub output_tokens: u32,
    /// Total tokens
    pub total_tokens: u32,
    /// Estimated cost in USD
    pub cost_usd: f64,
    /// Request duration
    pub duration: Duration,
}

/// OpenRouter API client.
///
/// Features:
/// - Automatic rate limit handling with adaptive backoff
/// - Response header parsing for proactive throttling
/// - Cost tracking
/// - Retry with exponential backoff
pub struct OpenRouterClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    timeout: Duration,
    max_retries: u32,
    rate_limiter: Arc<RateLimiter>,
    // Cost tracking
    total_input_tokens: AtomicU64,
    total_output_tokens: AtomicU64,
    total_cost_micros: AtomicU64, // Store as microdollars for atomic ops
}

impl OpenRouterClient {
    /// Create a new OpenRouter client.
    pub fn new(
        api_key: String,
        base_url: Option<String>,
        timeout_secs: Option<u64>,
        max_retries: Option<u32>,
        rate_limiter: Option<Arc<RateLimiter>>,
    ) -> Result<Self> {
        let timeout = Duration::from_secs(timeout_secs.unwrap_or(180));

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(DpogenError::Network)?;

        Ok(Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
            timeout,
            max_retries: max_retries.unwrap_or(3),
            rate_limiter: rate_limiter.unwrap_or_else(|| Arc::new(RateLimiter::new())),
            total_input_tokens: AtomicU64::new(0),
            total_output_tokens: AtomicU64::new(0),
            total_cost_micros: AtomicU64::new(0),
        })
    }

    /// Get the rate limiter.
    pub fn rate_limiter(&self) -> &Arc<RateLimiter> {
        &self.rate_limiter
    }

    /// Build headers for a request.
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "HTTP-Referer",
            HeaderValue::from_static("https://github.com/infernet-org/dpogen"),
        );
        headers.insert("X-Title", HeaderValue::from_static("dpogen"));
        headers
    }

    /// Calculate cost for a request.
    fn calculate_cost(&self, model_spec: &ModelSpec, input_tokens: u32, output_tokens: u32) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * model_spec.input_price_per_1m;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * model_spec.output_price_per_1m;
        input_cost + output_cost
    }

    /// Complete a chat request.
    ///
    /// B_i(API available) → Result
    /// B_i(valid response) → Result
    /// I^B(rate limits) → adaptive backoff
    pub async fn complete(
        &self,
        model: &ModelSpec,
        messages: Vec<Message>,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
    ) -> Result<CompletionResponse> {
        let start = Instant::now();
        let max_tokens = max_tokens.unwrap_or(model.max_tokens);
        let temperature = temperature.unwrap_or(model.temperature);

        let request = ChatCompletionRequest {
            model: model.id.clone(),
            messages,
            max_tokens,
            temperature,
            stop: None,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let mut last_error: Option<DpogenError> = None;

        for attempt in 0..self.max_retries {
            // Wait if rate limited
            self.rate_limiter.wait_if_needed(&model.id).await;

            let response = self
                .client
                .post(&url)
                .headers(self.headers())
                .json(&request)
                .send()
                .await;

            let response = match response {
                Ok(r) => r,
                Err(e) => {
                    if e.is_timeout() {
                        last_error = Some(DpogenError::Timeout(self.timeout));
                    } else {
                        last_error = Some(DpogenError::Network(e));
                    }
                    if attempt < self.max_retries - 1 {
                        let backoff = Duration::from_secs(2u64.pow(attempt));
                        debug!(
                            attempt = attempt,
                            backoff_secs = backoff.as_secs(),
                            "Retrying after network error"
                        );
                        tokio::time::sleep(backoff).await;
                    }
                    continue;
                }
            };

            let status = response.status().as_u16();
            let headers = response.headers().clone();

            // Update rate limiter from headers
            self.rate_limiter
                .record_request(&model.id, status, &headers);

            // Handle rate limiting
            if status == 429 {
                let retry_after = headers
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);

                last_error = Some(DpogenError::RateLimited {
                    retry_after_secs: retry_after,
                });

                if attempt < self.max_retries - 1 {
                    debug!(
                        attempt = attempt,
                        retry_after_secs = retry_after,
                        "Rate limited, waiting"
                    );
                    tokio::time::sleep(Duration::from_secs_f64(retry_after)).await;
                }
                continue;
            }

            // Handle other errors
            if !response.status().is_success() {
                let error_body = response.text().await.unwrap_or_default();
                let error =
                    if let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(&error_body) {
                        if status == 401 {
                            OpenRouterError::AuthenticationFailed
                        } else if status == 404 {
                            OpenRouterError::ModelNotFound(model.id.clone())
                        } else {
                            OpenRouterError::ApiError {
                                status,
                                message: api_error.error.message,
                            }
                        }
                    } else {
                        OpenRouterError::ApiError {
                            status,
                            message: error_body,
                        }
                    };

                last_error = Some(DpogenError::OpenRouterApi(error));

                // Don't retry auth errors or not found
                if status == 401 || status == 404 {
                    break;
                }

                if attempt < self.max_retries - 1 {
                    let backoff = Duration::from_secs(2u64.pow(attempt));
                    tokio::time::sleep(backoff).await;
                }
                continue;
            }

            // Parse successful response
            let body: ChatCompletionResponse = response
                .json()
                .await
                .map_err(|e| DpogenError::ParseError(format!("Failed to parse response: {e}")))?;

            let content = body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| DpogenError::ParseError("No choices in response".to_string()))?;

            let usage = body.usage.unwrap_or(ChatUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            });

            let cost = self.calculate_cost(model, usage.prompt_tokens, usage.completion_tokens);

            // Update tracking
            self.total_input_tokens
                .fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
            self.total_output_tokens
                .fetch_add(usage.completion_tokens as u64, Ordering::Relaxed);
            self.total_cost_micros
                .fetch_add((cost * 1_000_000.0) as u64, Ordering::Relaxed);

            return Ok(CompletionResponse {
                content,
                model: body.model.unwrap_or_else(|| model.id.clone()),
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cost_usd: cost,
                duration: start.elapsed(),
            });
        }

        // All retries exhausted
        Err(last_error.unwrap_or_else(|| {
            DpogenError::OpenRouterApi(OpenRouterError::MaxRetriesExceeded {
                attempts: self.max_retries,
                last_error: "Unknown error".to_string(),
            })
        }))
    }

    /// Complete with system and user prompts (convenience method).
    pub async fn complete_with_system(
        &self,
        model: &ModelSpec,
        system_prompt: &str,
        user_prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
    ) -> Result<CompletionResponse> {
        let messages = vec![Message::system(system_prompt), Message::user(user_prompt)];
        self.complete(model, messages, max_tokens, temperature)
            .await
    }

    /// Get total cost tracked.
    pub fn total_cost_usd(&self) -> f64 {
        self.total_cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Get total tokens tracked.
    pub fn total_tokens(&self) -> (u64, u64) {
        (
            self.total_input_tokens.load(Ordering::Relaxed),
            self.total_output_tokens.load(Ordering::Relaxed),
        )
    }

    /// Reset cost tracking.
    pub fn reset_tracking(&self) {
        self.total_input_tokens.store(0, Ordering::Relaxed);
        self.total_output_tokens.store(0, Ordering::Relaxed);
        self.total_cost_micros.store(0, Ordering::Relaxed);
    }
}
