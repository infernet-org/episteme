//! Worker pool for sample generation.
//!
//! Epistemic foundation:
//! - K_i: Workers generate samples by calling OpenRouter
//! - K_i: Multiple workers can run in parallel
//! - B_i: Each generation may succeed or fail → Result
//! - I^R: Model selection strategy is configurable

use crate::client::OpenRouterClient;
use crate::models::{DpogenError, ModelSpec, Problem, Result, Sample};
use chrono::Utc;
use regex::Regex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::warn;
use uuid::Uuid;

/// Worker pool for parallel sample generation.
#[allow(dead_code)]
pub struct WorkerPool {
    /// OpenRouter client (shared)
    client: Arc<OpenRouterClient>,
    /// Available models with weights
    models: Vec<ModelSpec>,
    /// System prompt for generation
    system_prompt: String,
    /// Pool size (max concurrent workers)
    pool_size: usize,
    /// Semaphore for concurrency control
    semaphore: Arc<Semaphore>,
    /// Round-robin model index
    model_index: AtomicUsize,
    /// Total weights for weighted selection
    total_weight: u32,
}

impl WorkerPool {
    /// Create a new worker pool.
    pub fn new(
        client: Arc<OpenRouterClient>,
        models: Vec<ModelSpec>,
        system_prompt: String,
        pool_size: usize,
    ) -> Self {
        let total_weight: u32 = models.iter().map(|m| m.weight).sum();
        Self {
            client,
            models,
            system_prompt,
            pool_size,
            semaphore: Arc::new(Semaphore::new(pool_size)),
            model_index: AtomicUsize::new(0),
            total_weight,
        }
    }

    /// Select a model using weighted round-robin.
    fn select_model(&self) -> &ModelSpec {
        if self.models.len() == 1 {
            return &self.models[0];
        }

        // Build weighted index
        let idx = self.model_index.fetch_add(1, Ordering::Relaxed);
        let mut cumulative = 0u32;
        let target = (idx as u32) % self.total_weight;

        for model in &self.models {
            cumulative += model.weight;
            if target < cumulative {
                return model;
            }
        }

        // Fallback to first model
        &self.models[0]
    }

    /// Extract answer from generated content.
    fn extract_answer(content: &str) -> Option<String> {
        // Try <answer> tags first
        let re = Regex::new(r"<answer>(.*?)</answer>").ok()?;
        if let Some(captures) = re.captures(content) {
            return captures.get(1).map(|m| m.as_str().trim().to_string());
        }

        // Try ## Answer section
        let re = Regex::new(r"##\s*Answer\s*\n(.*?)(?:\n##|$)").ok()?;
        if let Some(captures) = re.captures(content) {
            return captures.get(1).map(|m| m.as_str().trim().to_string());
        }

        // Return last paragraph as fallback
        content
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .last()
            .map(|p| p.trim().chars().take(500).collect())
    }

    /// Generate a single sample.
    ///
    /// B_i(generation succeeds) → Result<Sample>
    pub async fn generate(&self, problem: &Problem) -> Result<Sample> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| DpogenError::Internal("Semaphore closed".to_string()))?;

        let model = self.select_model();
        let start = Instant::now();

        let user_prompt = format!(
            "Solve the following problem:\n\n{}\n\nProvide your reasoning and final answer.",
            problem.input
        );

        let response = self
            .client
            .complete_with_system(model, &self.system_prompt, &user_prompt, None, None)
            .await?;

        let elapsed = start.elapsed();
        let answer = Self::extract_answer(&response.content);

        Ok(Sample {
            id: Uuid::new_v4().to_string(),
            problem_id: problem.id.clone(),
            input: problem.input.clone(),
            output: response.content,
            answer,
            model: response.model,
            generated_at: Utc::now(),
            generation_time_ms: elapsed.as_millis() as u64,
            tokens_in: response.input_tokens,
            tokens_out: response.output_tokens,
            cost_usd: response.cost_usd,
            metadata: problem.metadata.clone(),
        })
    }

    /// Generate samples for multiple problems in parallel.
    ///
    /// Returns (successful_samples, failed_problem_ids)
    pub async fn generate_batch(
        &self,
        problems: Vec<Problem>,
    ) -> (Vec<Sample>, Vec<String>) {
        let mut handles = Vec::with_capacity(problems.len());

        for problem in problems {
            let pool = self.clone_for_task();
            let handle = tokio::spawn(async move {
                let result = pool.generate(&problem).await;
                (problem.id.clone(), result)
            });
            handles.push(handle);
        }

        let mut samples = Vec::new();
        let mut failed = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((_problem_id, Ok(sample))) => samples.push(sample),
                Ok((problem_id, Err(e))) => {
                    warn!(problem_id = %problem_id, error = %e, "Generation failed");
                    failed.push(problem_id);
                }
                Err(e) => {
                    warn!(error = %e, "Task panicked");
                }
            }
        }

        (samples, failed)
    }

    /// Create a lightweight clone for spawning tasks.
    fn clone_for_task(&self) -> WorkerPoolHandle {
        WorkerPoolHandle {
            client: Arc::clone(&self.client),
            models: self.models.clone(),
            system_prompt: self.system_prompt.clone(),
            semaphore: Arc::clone(&self.semaphore),
            model_index: AtomicUsize::new(self.model_index.load(Ordering::Relaxed)),
            total_weight: self.total_weight,
        }
    }
}

/// Lightweight handle for spawned tasks.
struct WorkerPoolHandle {
    client: Arc<OpenRouterClient>,
    models: Vec<ModelSpec>,
    system_prompt: String,
    semaphore: Arc<Semaphore>,
    model_index: AtomicUsize,
    total_weight: u32,
}

impl WorkerPoolHandle {
    fn select_model(&self) -> &ModelSpec {
        if self.models.len() == 1 {
            return &self.models[0];
        }

        let idx = self.model_index.fetch_add(1, Ordering::Relaxed);
        let mut cumulative = 0u32;
        let target = (idx as u32) % self.total_weight;

        for model in &self.models {
            cumulative += model.weight;
            if target < cumulative {
                return model;
            }
        }

        &self.models[0]
    }

    async fn generate(&self, problem: &Problem) -> Result<Sample> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| DpogenError::Internal("Semaphore closed".to_string()))?;

        let model = self.select_model();
        let start = Instant::now();

        let user_prompt = format!(
            "Solve the following problem:\n\n{}\n\nProvide your reasoning and final answer.",
            problem.input
        );

        let response = self
            .client
            .complete_with_system(model, &self.system_prompt, &user_prompt, None, None)
            .await?;

        let elapsed = start.elapsed();
        let answer = WorkerPool::extract_answer(&response.content);

        Ok(Sample {
            id: Uuid::new_v4().to_string(),
            problem_id: problem.id.clone(),
            input: problem.input.clone(),
            output: response.content,
            answer,
            model: response.model,
            generated_at: Utc::now(),
            generation_time_ms: elapsed.as_millis() as u64,
            tokens_in: response.input_tokens,
            tokens_out: response.output_tokens,
            cost_usd: response.cost_usd,
            metadata: problem.metadata.clone(),
        })
    }
}
