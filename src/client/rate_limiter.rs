//! Adaptive rate limiter for OpenRouter API.
//!
//! Epistemic foundation:
//! - K_i: OpenRouter returns rate limit headers on each response
//! - K_i: 429 errors require exponential backoff
//! - B_i: Rate limits vary by model and account tier
//! - I^B: Exact rate limits unknowable until headers received

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Rate limit state for a single model.
#[derive(Debug)]
pub struct ModelRateLimitState {
    /// Limit on requests per interval
    pub limit_requests: Option<u32>,
    /// Limit on tokens per interval
    pub limit_tokens: Option<u32>,
    /// Remaining requests in current interval
    pub remaining_requests: Option<u32>,
    /// Remaining tokens in current interval
    pub remaining_tokens: Option<u32>,
    /// When the request limit resets
    pub reset_requests_at: Option<Instant>,
    /// When the token limit resets
    pub reset_tokens_at: Option<Instant>,
    /// Consecutive 429 errors (for backoff)
    pub consecutive_429s: u32,
    /// Backoff until this time
    pub backoff_until: Option<Instant>,
    /// Last update time
    pub last_updated: Instant,
}

impl Default for ModelRateLimitState {
    fn default() -> Self {
        Self {
            limit_requests: None,
            limit_tokens: None,
            remaining_requests: None,
            remaining_tokens: None,
            reset_requests_at: None,
            reset_tokens_at: None,
            consecutive_429s: 0,
            backoff_until: None,
            last_updated: Instant::now(),
        }
    }
}

impl ModelRateLimitState {
    /// Check if we can proceed with a request.
    pub fn can_proceed(&self) -> bool {
        let now = Instant::now();

        // Check backoff
        if self.backoff_until.is_some_and(|t| now < t) {
            return false;
        }

        // Check request limit: blocked if remaining is 0 and reset time hasn't passed
        if self.remaining_requests == Some(0) && self.reset_requests_at.is_some_and(|t| now < t) {
            return false;
        }

        // Check token limit: blocked if remaining is 0 and reset time hasn't passed
        if self.remaining_tokens == Some(0) && self.reset_tokens_at.is_some_and(|t| now < t) {
            return false;
        }

        true
    }

    /// Get time to wait before proceeding.
    pub fn wait_time(&self) -> Duration {
        let now = Instant::now();
        let mut max_wait = Duration::ZERO;

        // Check backoff
        if let Some(backoff_until) = self.backoff_until.filter(|&t| t > now) {
            max_wait = max_wait.max(backoff_until - now);
        }

        // Check request reset: wait if remaining is 0 and reset is in the future
        if self.remaining_requests == Some(0) && self.reset_requests_at.is_some_and(|t| t > now) {
            let reset_at = self.reset_requests_at.unwrap(); // safe: is_some_and passed
            max_wait = max_wait.max(reset_at - now);
        }

        // Check token reset: wait if remaining is 0 and reset is in the future
        if self.remaining_tokens == Some(0) && self.reset_tokens_at.is_some_and(|t| t > now) {
            let reset_at = self.reset_tokens_at.unwrap(); // safe: is_some_and passed
            max_wait = max_wait.max(reset_at - now);
        }

        max_wait
    }

    /// Record a 429 error and calculate backoff.
    pub fn record_429(&mut self) {
        self.consecutive_429s += 1;
        let backoff_secs = (2.0_f64).powi(self.consecutive_429s as i32).min(60.0);
        self.backoff_until = Some(Instant::now() + Duration::from_secs_f64(backoff_secs));
        warn!(
            consecutive_429s = self.consecutive_429s,
            backoff_secs = backoff_secs,
            "Rate limited (429), backing off"
        );
    }

    /// Record a successful request.
    pub fn record_success(&mut self) {
        // Reset consecutive 429s after success
        if self.consecutive_429s > 0 {
            self.consecutive_429s = 0;
            self.backoff_until = None;
        }
    }

    /// Update state from response headers.
    pub fn update_from_headers(&mut self, headers: &reqwest::header::HeaderMap) {
        let now = Instant::now();

        // Helper to parse header value
        let parse_header = |headers: &reqwest::header::HeaderMap, key: &str| -> Option<String> {
            headers.get(key)?.to_str().ok().map(|s| s.to_string())
        };

        // Parse rate limit headers
        if let Some(s) = parse_header(headers, "x-ratelimit-limit-requests") {
            self.limit_requests = s.parse().ok();
        }
        if let Some(s) = parse_header(headers, "x-ratelimit-limit-tokens") {
            self.limit_tokens = s.parse().ok();
        }
        if let Some(s) = parse_header(headers, "x-ratelimit-remaining-requests") {
            self.remaining_requests = s.parse().ok();
        }
        if let Some(s) = parse_header(headers, "x-ratelimit-remaining-tokens") {
            self.remaining_tokens = s.parse().ok();
        }
        if let Some(secs) =
            parse_header(headers, "x-ratelimit-reset-requests").and_then(|s| s.parse::<f64>().ok())
        {
            self.reset_requests_at = Some(now + Duration::from_secs_f64(secs));
        }
        if let Some(secs) =
            parse_header(headers, "x-ratelimit-reset-tokens").and_then(|s| s.parse::<f64>().ok())
        {
            self.reset_tokens_at = Some(now + Duration::from_secs_f64(secs));
        }

        self.last_updated = now;
    }
}

/// Adaptive rate limiter for OpenRouter API.
///
/// Tracks per-model rate limits and provides backoff on 429s.
#[derive(Debug)]
pub struct RateLimiter {
    /// Per-model rate limit state
    states: DashMap<String, ModelRateLimitState>,
    /// Global stats
    total_requests: AtomicU64,
    total_429s: AtomicU64,
    total_wait_ms: AtomicU64,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new() -> Self {
        Self {
            states: DashMap::new(),
            total_requests: AtomicU64::new(0),
            total_429s: AtomicU64::new(0),
            total_wait_ms: AtomicU64::new(0),
        }
    }

    /// Wait if necessary before making a request to the given model.
    ///
    /// Returns the duration waited.
    pub async fn wait_if_needed(&self, model: &str) -> Duration {
        let wait_time = {
            let state = self.states.entry(model.to_string()).or_default();
            state.wait_time()
        };

        if wait_time > Duration::ZERO {
            debug!(
                model = model,
                wait_ms = wait_time.as_millis(),
                "Waiting for rate limit"
            );
            self.total_wait_ms
                .fetch_add(wait_time.as_millis() as u64, Ordering::Relaxed);
            tokio::time::sleep(wait_time).await;
        }

        wait_time
    }

    /// Check if we can proceed (non-blocking).
    pub fn can_proceed(&self, model: &str) -> bool {
        self.states
            .entry(model.to_string())
            .or_default()
            .can_proceed()
    }

    /// Record a request result.
    pub fn record_request(&self, model: &str, status: u16, headers: &reqwest::header::HeaderMap) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let mut state = self.states.entry(model.to_string()).or_default();

        // Update from headers regardless of status
        state.update_from_headers(headers);

        if status == 429 {
            self.total_429s.fetch_add(1, Ordering::Relaxed);
            state.record_429();
        } else if status < 400 {
            state.record_success();
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> RateLimiterStats {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_429s = self.total_429s.load(Ordering::Relaxed);
        let total_wait_ms = self.total_wait_ms.load(Ordering::Relaxed);

        RateLimiterStats {
            total_requests,
            total_429s,
            total_wait_secs: total_wait_ms as f64 / 1000.0,
            rate_limit_ratio: if total_requests > 0 {
                total_429s as f64 / total_requests as f64
            } else {
                0.0
            },
            models_tracked: self.states.len(),
        }
    }
}

/// Rate limiter statistics.
#[derive(Debug, Clone)]
pub struct RateLimiterStats {
    pub total_requests: u64,
    pub total_429s: u64,
    pub total_wait_secs: f64,
    pub rate_limit_ratio: f64,
    pub models_tracked: usize,
}
