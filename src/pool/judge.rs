//! Judge pool for sample evaluation with ensemble support.
//!
//! Epistemic foundation:
//! - K_i: Judges evaluate samples using LLM-based criteria
//! - K_i: Score is 0.0-1.0, verdict is binary (approve/reject)
//! - B_i: Single judge score → B_i(quality) with unknown bias
//! - B_i(HIGH): Ensemble consensus → higher confidence via multiple judges
//! - I^R: Judge prompt, threshold, and ensemble config are configurable

use crate::client::EndpointRegistry;
use crate::models::{
    AggregationStrategy, Confidence, EnsembleConfig, EnsembleJudgeResult, EpistemeError,
    JudgeResult, ModelSpec, Result, Sample, Verdict,
};
use chrono::Utc;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

/// Judge pool for parallel sample evaluation with optional ensemble mode.
///
/// Supports three evaluation modes:
/// 1. Single judge (default): One model evaluates each sample
/// 2. Ensemble: Multiple judges evaluate each sample, scores aggregated
/// 3. Hierarchical: Cheap judge first, ensemble only if uncertain
pub struct JudgePool {
    /// Endpoint registry (provides access to all configured LLM clients)
    registry: Arc<EndpointRegistry>,
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
    /// Total weights for weighted selection
    total_weight: u32,
    /// Approval threshold (0.0 - 1.0)
    approval_threshold: f64,
    /// Ensemble configuration
    ensemble_config: EnsembleConfig,
}

impl JudgePool {
    /// Create a new judge pool.
    pub fn new(
        registry: Arc<EndpointRegistry>,
        models: Vec<ModelSpec>,
        judge_prompt: String,
        pool_size: usize,
        approval_threshold: f64,
        ensemble_config: EnsembleConfig,
    ) -> Self {
        let total_weight: u32 = models.iter().map(|m| m.weight).sum();
        Self {
            registry,
            models,
            judge_prompt,
            pool_size,
            semaphore: Arc::new(Semaphore::new(pool_size)),
            model_index: AtomicUsize::new(0),
            total_weight,
            approval_threshold,
            ensemble_config,
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

    /// Find the cheapest model by total price (input + output per 1M tokens).
    fn cheapest_model(&self) -> &ModelSpec {
        self.models
            .iter()
            .min_by(|a, b| {
                let cost_a = a.input_price_per_1m + a.output_price_per_1m;
                let cost_b = b.input_price_per_1m + b.output_price_per_1m;
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&self.models[0])
    }

    /// Build a map of model ID to weight for weighted averaging.
    fn model_weight_map(&self) -> HashMap<String, u32> {
        self.models
            .iter()
            .map(|m| (m.id.clone(), m.weight))
            .collect()
    }

    /// Build user prompt for judging a sample.
    fn build_judge_user_prompt(sample: &Sample) -> String {
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

    /// Aggregate scores using the configured strategy.
    ///
    /// Returns (aggregated_score, standard_deviation).
    fn aggregate_scores(
        results: &[JudgeResult],
        strategy: AggregationStrategy,
        model_weights: &HashMap<String, u32>,
    ) -> (f64, f64) {
        if results.is_empty() {
            return (0.5, 0.0);
        }

        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();

        let aggregated = match strategy {
            AggregationStrategy::Median => {
                let mut sorted = scores.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len().is_multiple_of(2) && sorted.len() > 1 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
            AggregationStrategy::Average => scores.iter().sum::<f64>() / scores.len() as f64,
            AggregationStrategy::WeightedAverage => {
                let total_weight: f64 = results
                    .iter()
                    .map(|r| *model_weights.get(&r.judge_model).unwrap_or(&1) as f64)
                    .sum();
                if total_weight == 0.0 {
                    scores.iter().sum::<f64>() / scores.len() as f64
                } else {
                    results
                        .iter()
                        .map(|r| {
                            let w = *model_weights.get(&r.judge_model).unwrap_or(&1) as f64;
                            r.score * w
                        })
                        .sum::<f64>()
                        / total_weight
                }
            }
        };

        // Compute standard deviation
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        (aggregated, std_dev)
    }

    /// Compute majority vote verdict from multiple judge results.
    fn majority_vote_verdict(results: &[JudgeResult]) -> Verdict {
        let approve_count = results
            .iter()
            .filter(|r| r.verdict == Verdict::Approve)
            .count();
        if approve_count > results.len() / 2 {
            Verdict::Approve
        } else {
            Verdict::Reject
        }
    }

    /// Determine confidence level based on judge agreement.
    ///
    /// - High: Low disagreement AND unanimous verdict
    /// - Medium: Low disagreement AND majority verdict
    /// - Low: High disagreement (no consensus) - valuable signal
    fn compute_confidence(
        results: &[JudgeResult],
        std_dev: f64,
        threshold: f64,
        final_verdict: Verdict,
    ) -> Confidence {
        if std_dev >= threshold {
            return Confidence::Low; // No consensus - valuable for downstream filtering
        }

        // Check if all judges agree on verdict
        let unanimous = results.iter().all(|r| r.verdict == final_verdict);

        if unanimous {
            Confidence::High
        } else {
            Confidence::Medium
        }
    }

    /// Evaluate a single sample with a specific model.
    async fn evaluate_with_model(&self, sample: &Sample, model: &ModelSpec) -> Result<JudgeResult> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| EpistemeError::Internal("Semaphore closed".to_string()))?;

        let start = Instant::now();
        let user_prompt = Self::build_judge_user_prompt(sample);

        // Get the appropriate client for this model's endpoint
        let client = self.registry.get(&model.endpoint).ok_or_else(|| {
            EpistemeError::Internal(format!(
                "Endpoint '{}' not found in registry",
                model.endpoint
            ))
        })?;

        let response = client
            .complete_with_system(model, &self.judge_prompt, &user_prompt, None, Some(0.3))
            .await?;

        let elapsed = start.elapsed();

        let score = Self::parse_score(&response.content).unwrap_or(0.5);
        let verdict = Self::parse_verdict(&response.content, score, self.approval_threshold);

        debug!(
            sample_id = %sample.id,
            model = %model.id,
            score = score,
            verdict = ?verdict,
            "Single judge evaluated"
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

    /// Evaluate a single sample (single judge mode).
    async fn evaluate_single(&self, sample: &Sample) -> Result<JudgeResult> {
        let model = self.select_model();
        self.evaluate_with_model(sample, model).await
    }

    /// Evaluate with full ensemble (multiple judges).
    async fn evaluate_ensemble_internal(&self, sample: &Sample) -> Result<EnsembleJudgeResult> {
        let start = Instant::now();
        let num_judges = self.ensemble_config.num_judges;

        // Run N judges in parallel using different models for diversity
        let mut handles = Vec::with_capacity(num_judges);
        for _ in 0..num_judges {
            let registry = Arc::clone(&self.registry);
            let model = self.select_model().clone();
            let judge_prompt = self.judge_prompt.clone();
            let sample_clone = sample.clone();
            let semaphore = Arc::clone(&self.semaphore);
            let threshold = self.approval_threshold;

            let handle = tokio::spawn(async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .map_err(|_| EpistemeError::Internal("Semaphore closed".to_string()))?;

                let inner_start = Instant::now();
                let user_prompt = Self::build_judge_user_prompt(&sample_clone);

                // Get the appropriate client for this model's endpoint
                let client = registry.get(&model.endpoint).ok_or_else(|| {
                    EpistemeError::Internal(format!(
                        "Endpoint '{}' not found in registry",
                        model.endpoint
                    ))
                })?;

                let response = client
                    .complete_with_system(&model, &judge_prompt, &user_prompt, None, Some(0.3))
                    .await?;

                let elapsed = inner_start.elapsed();
                let score = Self::parse_score(&response.content).unwrap_or(0.5);
                let verdict = Self::parse_verdict(&response.content, score, threshold);

                Ok::<JudgeResult, EpistemeError>(JudgeResult {
                    sample_id: sample_clone.id.clone(),
                    score,
                    verdict,
                    reasoning: response.content,
                    judge_model: response.model,
                    judged_at: Utc::now(),
                    judge_time_ms: elapsed.as_millis() as u64,
                    judge_cost_usd: response.cost_usd,
                })
            });
            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(r)) => results.push(r),
                Ok(Err(e)) => warn!(error = %e, "Ensemble judge failed"),
                Err(e) => warn!(error = %e, "Ensemble task panicked"),
            }
        }

        if results.is_empty() {
            return Err(EpistemeError::Internal(
                "All ensemble judges failed".to_string(),
            ));
        }

        // Aggregate scores
        let model_weights = self.model_weight_map();
        let (score, std_dev) =
            Self::aggregate_scores(&results, self.ensemble_config.strategy, &model_weights);

        // Majority vote for verdict
        let verdict = Self::majority_vote_verdict(&results);

        // Compute confidence
        let confidence = Self::compute_confidence(
            &results,
            std_dev,
            self.ensemble_config.disagreement_threshold,
            verdict,
        );

        let total_cost: f64 = results.iter().map(|r| r.judge_cost_usd).sum();
        let judge_models: String = results
            .iter()
            .map(|r| r.judge_model.clone())
            .collect::<Vec<_>>()
            .join(",");

        let reasoning = results
            .first()
            .map(|r| r.reasoning.clone())
            .unwrap_or_default();

        info!(
            sample_id = %sample.id,
            score = score,
            std_dev = std_dev,
            verdict = ?verdict,
            confidence = ?confidence,
            num_judges = results.len(),
            "Ensemble evaluation complete"
        );

        Ok(EnsembleJudgeResult {
            sample_id: sample.id.clone(),
            score,
            verdict,
            confidence,
            score_std_dev: std_dev,
            individual_results: results,
            reasoning,
            judge_models,
            judged_at: Utc::now(),
            judge_time_ms: start.elapsed().as_millis() as u64,
            judge_cost_usd: total_cost,
        })
    }

    /// Evaluate with hierarchical strategy: cheap judge first, ensemble if uncertain.
    async fn evaluate_hierarchical_internal(&self, sample: &Sample) -> Result<EnsembleJudgeResult> {
        // Run cheapest judge first
        let cheapest = self.cheapest_model();
        let first_result = self.evaluate_with_model(sample, cheapest).await?;

        let (low, high) = self.ensemble_config.uncertain_range;

        // If score is outside uncertain range, return single result with high confidence
        if first_result.score < low || first_result.score > high {
            debug!(
                sample_id = %sample.id,
                score = first_result.score,
                "Hierarchical: confident after first judge"
            );
            return Ok(first_result.into());
        }

        // Score is uncertain, run full ensemble
        info!(
            sample_id = %sample.id,
            initial_score = first_result.score,
            "Hierarchical: score in uncertain range [{}, {}], escalating to ensemble",
            low,
            high
        );

        self.evaluate_ensemble_internal(sample).await
    }

    /// Evaluate a sample using the configured strategy.
    ///
    /// Returns EnsembleJudgeResult for consistent API regardless of mode.
    /// - Single mode: Wraps JudgeResult as EnsembleJudgeResult
    /// - Ensemble mode: Full multi-judge evaluation
    /// - Hierarchical mode: Cheap first, ensemble if uncertain
    pub async fn evaluate(&self, sample: &Sample) -> Result<EnsembleJudgeResult> {
        if !self.ensemble_config.enabled {
            // Single judge mode - wrap result
            return self.evaluate_single(sample).await.map(Into::into);
        }

        if self.ensemble_config.hierarchical {
            self.evaluate_hierarchical_internal(sample).await
        } else {
            self.evaluate_ensemble_internal(sample).await
        }
    }

    /// Evaluate multiple samples in parallel.
    ///
    /// Returns (successful_results, failed_sample_ids)
    pub async fn evaluate_batch(
        &self,
        samples: Vec<Sample>,
    ) -> (Vec<(Sample, EnsembleJudgeResult)>, Vec<String>) {
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

    /// Create a clone of this pool for spawning tasks.
    fn clone_for_task(&self) -> Self {
        Self {
            registry: Arc::clone(&self.registry),
            models: self.models.clone(),
            judge_prompt: self.judge_prompt.clone(),
            pool_size: self.pool_size,
            semaphore: Arc::clone(&self.semaphore),
            model_index: AtomicUsize::new(self.model_index.load(Ordering::Relaxed)),
            total_weight: self.total_weight,
            approval_threshold: self.approval_threshold,
            ensemble_config: self.ensemble_config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_scores_median() {
        let results = vec![
            make_judge_result(0.9, "model_a"),
            make_judge_result(0.85, "model_b"),
            make_judge_result(0.3, "model_c"), // outlier
        ];
        let weights = HashMap::new();
        let (score, std_dev) =
            JudgePool::aggregate_scores(&results, AggregationStrategy::Median, &weights);

        // Median of [0.3, 0.85, 0.9] = 0.85
        assert!((score - 0.85).abs() < 0.001);
        assert!(std_dev > 0.2); // High disagreement due to outlier
    }

    #[test]
    fn test_aggregate_scores_average() {
        let results = vec![
            make_judge_result(0.9, "model_a"),
            make_judge_result(0.8, "model_b"),
            make_judge_result(0.7, "model_c"),
        ];
        let weights = HashMap::new();
        let (score, _) =
            JudgePool::aggregate_scores(&results, AggregationStrategy::Average, &weights);

        // Average of [0.9, 0.8, 0.7] = 0.8
        assert!((score - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_majority_vote_verdict() {
        let results = vec![
            make_judge_result_with_verdict(0.9, Verdict::Approve),
            make_judge_result_with_verdict(0.8, Verdict::Approve),
            make_judge_result_with_verdict(0.3, Verdict::Reject),
        ];

        assert_eq!(JudgePool::majority_vote_verdict(&results), Verdict::Approve);
    }

    #[test]
    fn test_compute_confidence_high() {
        let results = vec![
            make_judge_result_with_verdict(0.9, Verdict::Approve),
            make_judge_result_with_verdict(0.88, Verdict::Approve),
            make_judge_result_with_verdict(0.91, Verdict::Approve),
        ];

        // Low std_dev (~0.015), unanimous verdict
        let confidence = JudgePool::compute_confidence(&results, 0.015, 0.15, Verdict::Approve);
        assert_eq!(confidence, Confidence::High);
    }

    #[test]
    fn test_compute_confidence_low() {
        let results = vec![
            make_judge_result_with_verdict(0.9, Verdict::Approve),
            make_judge_result_with_verdict(0.5, Verdict::Reject),
            make_judge_result_with_verdict(0.3, Verdict::Reject),
        ];

        // High std_dev due to score spread
        let confidence = JudgePool::compute_confidence(&results, 0.25, 0.15, Verdict::Reject);
        assert_eq!(confidence, Confidence::Low);
    }

    fn make_judge_result(score: f64, model: &str) -> JudgeResult {
        JudgeResult {
            sample_id: "test".to_string(),
            score,
            verdict: if score >= 0.7 {
                Verdict::Approve
            } else {
                Verdict::Reject
            },
            reasoning: "test".to_string(),
            judge_model: model.to_string(),
            judged_at: Utc::now(),
            judge_time_ms: 100,
            judge_cost_usd: 0.001,
        }
    }

    fn make_judge_result_with_verdict(score: f64, verdict: Verdict) -> JudgeResult {
        JudgeResult {
            sample_id: "test".to_string(),
            score,
            verdict,
            reasoning: "test".to_string(),
            judge_model: "test_model".to_string(),
            judged_at: Utc::now(),
            judge_time_ms: 100,
            judge_cost_usd: 0.001,
        }
    }
}
