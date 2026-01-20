//! Judge pool for sample evaluation.
//!
//! Epistemic foundation:
//! - K_i: Judges evaluate samples using LLM-based criteria
//! - K_i: Score is 0.0-1.0, verdict is binary (approve/reject)
//! - B_i: Each evaluation may succeed or fail → Result
//! - I^R: Judge prompt and threshold are configurable

use crate::client::OpenRouterClient;
use crate::models::{EpistemeError, JudgeResult, ModelSpec, Result, Sample, Verdict};
use chrono::Utc;
use regex::Regex;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{debug, warn};

/// Judge pool for parallel sample evaluation.
#[allow(dead_code)]
pub struct JudgePool {
    /// OpenRouter client (shared)
    client: Arc<OpenRouterClient>,
    /// Available models for judging
    models: Vec<ModelSpec>,
    /// Judge prompt template
    judge_prompt: String,
    /// Pool size (max concurrent judges)
    pool_size: usize,
    /// Semaphore for concurrency control
    semaphore: Arc<Semaphore>,
    /// Round-robin model index
    model_index: AtomicUsize,
    /// Total weights
    total_weight: u32,
    /// Approval threshold (0.0 - 1.0)
    approval_threshold: f64,
}

impl JudgePool {
    /// Create a new judge pool.
    pub fn new(
        client: Arc<OpenRouterClient>,
        models: Vec<ModelSpec>,
        judge_prompt: String,
        pool_size: usize,
        approval_threshold: f64,
    ) -> Self {
        let total_weight: u32 = models.iter().map(|m| m.weight).sum();
        Self {
            client,
            models,
            judge_prompt,
            pool_size,
            semaphore: Arc::new(Semaphore::new(pool_size)),
            model_index: AtomicUsize::new(0),
            total_weight,
            approval_threshold,
        }
    }

    /// Select a model using weighted round-robin.
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

    /// Build user prompt for judging a sample.
    fn build_judge_user_prompt(&self, sample: &Sample) -> String {
        format!(
            r#"## Sample to Evaluate

**Problem:**
{}

**Generated Response:**
{}

**Extracted Answer:**
{}

Please evaluate this sample according to the criteria and provide:
1. A score from 0.0 to 1.0
2. Your reasoning
3. A verdict: APPROVE or REJECT"#,
            sample.input,
            sample.output,
            sample.answer.as_deref().unwrap_or("(not extracted)")
        )
    }

    /// Parse score from judge response.
    fn parse_score(content: &str) -> Option<f64> {
        // Helper to extract score from captures
        let extract_score = |re: &Regex, content: &str| -> Option<f64> {
            re.captures(content)?
                .get(1)?
                .as_str()
                .parse::<f64>()
                .ok()
                .map(|s| s.clamp(0.0, 1.0))
        };

        // Pattern: "Score: 0.85" or "score: 0.85"
        if let Some(score) = Regex::new(r"[Ss]core[:\s]+(\d+\.?\d*)")
            .ok()
            .and_then(|re| extract_score(&re, content))
        {
            return Some(score);
        }

        // Pattern: "**Score:** 0.85"
        if let Some(score) = Regex::new(r"\*\*[Ss]core\*\*[:\s]+(\d+\.?\d*)")
            .ok()
            .and_then(|re| extract_score(&re, content))
        {
            return Some(score);
        }

        // Pattern: "Overall score: 0.85"
        if let Some(score) = Regex::new(r"[Oo]verall\s+[Ss]core[:\s]+(\d+\.?\d*)")
            .ok()
            .and_then(|re| extract_score(&re, content))
        {
            return Some(score);
        }

        // Fallback: look for any decimal between 0 and 1
        let re = Regex::new(r"(0\.\d+|1\.0)").ok()?;
        re.captures_iter(content).find_map(|captures| {
            captures
                .get(1)?
                .as_str()
                .parse::<f64>()
                .ok()
                .filter(|&s| (0.0..=1.0).contains(&s))
        })
    }

    /// Parse verdict from judge response.
    fn parse_verdict(content: &str, score: f64, threshold: f64) -> Verdict {
        let content_upper = content.to_uppercase();

        // Look for explicit verdict
        if content_upper.contains("APPROVE") && !content_upper.contains("NOT APPROVE") {
            return Verdict::Approve;
        }
        if content_upper.contains("REJECT") {
            return Verdict::Reject;
        }

        // Fall back to score-based verdict
        if score >= threshold {
            Verdict::Approve
        } else {
            Verdict::Reject
        }
    }

    /// Evaluate a single sample.
    ///
    /// B_i(evaluation succeeds) → Result<JudgeResult>
    pub async fn evaluate(&self, sample: &Sample) -> Result<JudgeResult> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| EpistemeError::Internal("Semaphore closed".to_string()))?;

        let model = self.select_model();
        let start = Instant::now();

        let user_prompt = self.build_judge_user_prompt(sample);

        let response = self
            .client
            .complete_with_system(model, &self.judge_prompt, &user_prompt, None, Some(0.3))
            .await?;

        let elapsed = start.elapsed();

        // Parse score and verdict
        let score = Self::parse_score(&response.content).unwrap_or(0.5);
        let verdict = Self::parse_verdict(&response.content, score, self.approval_threshold);

        debug!(
            sample_id = %sample.id,
            score = score,
            verdict = ?verdict,
            "Sample evaluated"
        );

        Ok(JudgeResult {
            sample_id: sample.id.clone(),
            score,
            verdict,
            reasoning: response.content,
            judge_model: response.model,
            judged_at: Utc::now(),
            judge_time_ms: elapsed.as_millis() as u64,
            judge_cost_usd: response.cost_usd,
        })
    }

    /// Evaluate multiple samples in parallel.
    ///
    /// Returns (successful_results, failed_sample_ids)
    pub async fn evaluate_batch(
        &self,
        samples: Vec<Sample>,
    ) -> (Vec<(Sample, JudgeResult)>, Vec<String>) {
        let mut handles = Vec::with_capacity(samples.len());

        for sample in samples {
            let pool = self.clone_for_task();
            let handle = tokio::spawn(async move {
                let result = pool.evaluate(&sample).await;
                (sample, result)
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        let mut failed = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((sample, Ok(judge_result))) => {
                    results.push((sample, judge_result));
                }
                Ok((sample, Err(e))) => {
                    warn!(sample_id = %sample.id, error = %e, "Evaluation failed");
                    failed.push(sample.id);
                }
                Err(e) => {
                    warn!(error = %e, "Task panicked");
                }
            }
        }

        (results, failed)
    }

    /// Create a lightweight clone for spawning tasks.
    fn clone_for_task(&self) -> JudgePoolHandle {
        JudgePoolHandle {
            client: Arc::clone(&self.client),
            models: self.models.clone(),
            judge_prompt: self.judge_prompt.clone(),
            semaphore: Arc::clone(&self.semaphore),
            model_index: AtomicUsize::new(self.model_index.load(Ordering::Relaxed)),
            total_weight: self.total_weight,
            approval_threshold: self.approval_threshold,
        }
    }
}

/// Lightweight handle for spawned tasks.
struct JudgePoolHandle {
    client: Arc<OpenRouterClient>,
    models: Vec<ModelSpec>,
    judge_prompt: String,
    semaphore: Arc<Semaphore>,
    model_index: AtomicUsize,
    total_weight: u32,
    approval_threshold: f64,
}

impl JudgePoolHandle {
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

    async fn evaluate(&self, sample: &Sample) -> Result<JudgeResult> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| EpistemeError::Internal("Semaphore closed".to_string()))?;

        let model = self.select_model();
        let start = Instant::now();

        let user_prompt = format!(
            r#"## Sample to Evaluate

**Problem:**
{}

**Generated Response:**
{}

**Extracted Answer:**
{}

Please evaluate this sample according to the criteria and provide:
1. A score from 0.0 to 1.0
2. Your reasoning
3. A verdict: APPROVE or REJECT"#,
            sample.input,
            sample.output,
            sample.answer.as_deref().unwrap_or("(not extracted)")
        );

        let response = self
            .client
            .complete_with_system(model, &self.judge_prompt, &user_prompt, None, Some(0.3))
            .await?;

        let elapsed = start.elapsed();

        let score = JudgePool::parse_score(&response.content).unwrap_or(0.5);
        let verdict = JudgePool::parse_verdict(&response.content, score, self.approval_threshold);

        Ok(JudgeResult {
            sample_id: sample.id.clone(),
            score,
            verdict,
            reasoning: response.content,
            judge_model: response.model,
            judged_at: Utc::now(),
            judge_time_ms: elapsed.as_millis() as u64,
            judge_cost_usd: response.cost_usd,
        })
    }
}
