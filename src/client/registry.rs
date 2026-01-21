//! Endpoint registry for managing multiple LLM endpoints.
//!
//! Epistemic foundation:
//! - K_i: All endpoints are OpenAI-compatible
//! - K_i: OpenRouter is the default endpoint
//! - B_i: Endpoints are reachable (verified via health checks)
//! - I^R: User configures which endpoints to use

use crate::client::{HealthCheckResult, HealthStatus, LLMClient, RateLimiter};
use crate::models::{Config, ConfigError, expand_headers};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// Registry of configured LLM endpoints.
///
/// Provides access to LLMClient instances by endpoint name.
/// "openrouter" is always the default endpoint and is automatically created.
///
/// Thread-safe: All clients are wrapped in `Arc<LLMClient>` for shared access.
///
/// # Example
///
/// ```ignore
/// use episteme::{Config, EndpointRegistry};
///
/// let config = Config::from_file("config.toml")?;
/// let registry = EndpointRegistry::from_config(&config)?;
///
/// // Get the default OpenRouter client
/// let client = registry.openrouter();
///
/// // Or get a specific endpoint
/// if let Some(local) = registry.get("ollama") {
///     // Use local client
/// }
///
/// // Run health checks on all endpoints
/// let results = registry.health_check_all().await;
/// ```
pub struct EndpointRegistry {
    endpoints: HashMap<String, Arc<LLMClient>>,
}

impl EndpointRegistry {
    /// Build registry from configuration.
    ///
    /// Creates LLMClient instances for OpenRouter and all configured endpoints.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::MissingApiKey` if an endpoint requires an API key
    /// but none is configured or found in environment variables.
    ///
    /// Returns `ConfigError::EndpointNotFound` if client creation fails.
    pub fn from_config(config: &Config) -> Result<Self, ConfigError> {
        let mut endpoints = HashMap::new();

        // Create shared rate limiter for all endpoints
        // (each endpoint gets its own instance for isolation)

        // Build OpenRouter client (always present)
        let openrouter_api_key = config.resolve_api_key()?;
        let openrouter_client = LLMClient::new(
            "openrouter".to_string(),
            Some(openrouter_api_key),
            config.openrouter.base_url.clone(),
            HashMap::new(), // OpenRouter doesn't need custom headers
            config.openrouter.timeout_secs,
            config.openrouter.max_retries,
            Some(Arc::new(RateLimiter::new())),
        )
        .map_err(|e| {
            ConfigError::EndpointNotFound(format!("Failed to create OpenRouter client: {e}"))
        })?;

        endpoints.insert("openrouter".to_string(), Arc::new(openrouter_client));

        // Build additional endpoint clients
        for (name, endpoint_config) in &config.endpoints {
            let api_key = config.resolve_endpoint_api_key(name)?;
            let headers = expand_headers(&endpoint_config.headers);

            let client = LLMClient::new(
                name.clone(),
                api_key,
                endpoint_config.base_url.clone(),
                headers,
                endpoint_config.timeout_secs,
                endpoint_config.max_retries,
                Some(Arc::new(RateLimiter::new())),
            )
            .map_err(|e| {
                ConfigError::EndpointNotFound(format!("Failed to create '{name}' client: {e}"))
            })?;

            endpoints.insert(name.clone(), Arc::new(client));
        }

        Ok(Self { endpoints })
    }

    /// Get client by endpoint name.
    ///
    /// Returns None if endpoint is not configured.
    pub fn get(&self, name: &str) -> Option<&Arc<LLMClient>> {
        self.endpoints.get(name)
    }

    /// Get the default OpenRouter client.
    pub fn openrouter(&self) -> &Arc<LLMClient> {
        self.endpoints
            .get("openrouter")
            .expect("OpenRouter client should always exist")
    }

    /// Get all endpoint names.
    pub fn endpoint_names(&self) -> Vec<&str> {
        self.endpoints.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of configured endpoints.
    pub fn len(&self) -> usize {
        self.endpoints.len()
    }

    /// Check if registry is empty (should never be true - OpenRouter is always present).
    pub fn is_empty(&self) -> bool {
        self.endpoints.is_empty()
    }

    /// Run health checks on all endpoints.
    ///
    /// Returns a list of health check results.
    pub async fn health_check_all(&self) -> Vec<HealthCheckResult> {
        let mut results = Vec::new();

        for (name, client) in &self.endpoints {
            let result = client.health_check().await;
            results.push(result);

            // Log result
            match &results.last().unwrap().status {
                HealthStatus::Healthy => {
                    if let Some(latency) = results.last().unwrap().latency_ms {
                        info!(endpoint = %name, latency_ms = latency, "Endpoint healthy");
                    }
                }
                HealthStatus::Unhealthy => {
                    warn!(
                        endpoint = %name,
                        error = results.last().unwrap().error.as_deref().unwrap_or("unknown"),
                        "Endpoint unhealthy"
                    );
                }
                HealthStatus::Unreachable => {
                    warn!(
                        endpoint = %name,
                        error = results.last().unwrap().error.as_deref().unwrap_or("unknown"),
                        "Endpoint unreachable"
                    );
                }
            }
        }

        results
    }

    /// Run health checks and return summary.
    ///
    /// Returns (healthy_count, total_count, unhealthy_endpoints).
    pub async fn health_check_summary(&self) -> (usize, usize, Vec<String>) {
        let results = self.health_check_all().await;
        let total = results.len();
        let healthy = results
            .iter()
            .filter(|r| r.status == HealthStatus::Healthy)
            .count();
        let unhealthy: Vec<String> = results
            .iter()
            .filter(|r| r.status != HealthStatus::Healthy)
            .map(|r| r.endpoint.clone())
            .collect();

        (healthy, total, unhealthy)
    }

    /// Validate that all referenced endpoints exist in the registry.
    pub fn validate_model_endpoints(&self, config: &Config) -> Result<(), ConfigError> {
        for endpoint_name in config.referenced_endpoints() {
            if !self.endpoints.contains_key(&endpoint_name) {
                return Err(ConfigError::EndpointNotFound(endpoint_name));
            }
        }
        Ok(())
    }
}
