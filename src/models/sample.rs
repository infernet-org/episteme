//! Sample and result types for episteme.
//!
//! K_i: These types represent the core data flow through the pipeline.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Input problem for generation.
///
/// K_i: Every problem has an ID and input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Problem {
    /// Unique identifier for this problem
    pub id: String,

    /// The problem/prompt text
    pub input: String,

    /// Optional metadata (passed through to output)
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

/// Generated sample from a worker.
///
/// K_i: A sample is the result of successfully generating a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Unique identifier for this sample
    pub id: String,

    /// ID of the source problem
    pub problem_id: String,

    /// Original input/prompt
    pub input: String,

    /// Generated output/response
    pub output: String,

    /// Extracted answer (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,

    /// Model used for generation
    pub model: String,

    /// Generation timestamp
    pub generated_at: DateTime<Utc>,

    /// Generation time in milliseconds
    pub generation_time_ms: u64,

    /// Input tokens used
    pub tokens_in: u32,

    /// Output tokens generated
    pub tokens_out: u32,

    /// Generation cost in USD
    pub cost_usd: f64,

    /// Optional metadata from problem
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

/// Verdict from the judge.
///
/// K_i: A verdict is either approve or reject. Binary decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Sample meets quality threshold
    Approve,
    /// Sample does not meet quality threshold
    Reject,
}

/// Confidence level from ensemble judging.
///
/// Epistemic foundation:
/// - High: Low disagreement + unanimous verdict → B_i(HIGH) approaches K_i
/// - Medium: Low disagreement + majority verdict → B_i(MED)
/// - Low: High disagreement (no consensus) → B_i(LOW), valuable as explicit uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// Low disagreement, unanimous verdict
    High,
    /// Low disagreement, majority verdict
    Medium,
    /// High disagreement (no consensus) - valuable signal for downstream filtering
    Low,
}

impl Default for Confidence {
    fn default() -> Self {
        Self::High
    }
}

/// Result from the judge pool.
///
/// K_i: Every judged sample has a score and verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeResult {
    /// ID of the sample that was judged
    pub sample_id: String,

    /// Quality score (0.0 - 1.0)
    pub score: f64,

    /// Binary verdict
    pub verdict: Verdict,

    /// Judge's reasoning/explanation
    pub reasoning: String,

    /// Model used for judging
    pub judge_model: String,

    /// Judge timestamp
    pub judged_at: DateTime<Utc>,

    /// Judging time in milliseconds
    pub judge_time_ms: u64,

    /// Judging cost in USD
    pub judge_cost_usd: f64,
}

/// Result from ensemble judging (multiple judges per sample).
///
/// Epistemic foundation:
/// - Aggregates multiple B_i(score) into higher-confidence result
/// - Makes disagreement explicit via score_std_dev
/// - Confidence level indicates strength of consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleJudgeResult {
    /// ID of the sample that was judged
    pub sample_id: String,

    /// Aggregated score (median by default)
    pub score: f64,

    /// Consensus verdict (majority vote)
    pub verdict: Verdict,

    /// Confidence level based on judge agreement
    pub confidence: Confidence,

    /// Standard deviation of scores (disagreement measure)
    pub score_std_dev: f64,

    /// Individual judge results (for traceability)
    pub individual_results: Vec<JudgeResult>,

    /// Combined reasoning (from first/primary judge)
    pub reasoning: String,

    /// Models used for judging (comma-separated)
    pub judge_models: String,

    /// Timestamp of final result
    pub judged_at: DateTime<Utc>,

    /// Total judging time in milliseconds
    pub judge_time_ms: u64,

    /// Total cost of all judge calls
    pub judge_cost_usd: f64,
}

impl From<JudgeResult> for EnsembleJudgeResult {
    /// Wrap a single JudgeResult as an ensemble result.
    ///
    /// Used when ensemble is disabled to maintain consistent API.
    fn from(r: JudgeResult) -> Self {
        Self {
            sample_id: r.sample_id.clone(),
            score: r.score,
            verdict: r.verdict,
            confidence: Confidence::High, // Single judge assumed confident
            score_std_dev: 0.0,
            individual_results: vec![r.clone()],
            reasoning: r.reasoning.clone(),
            judge_models: r.judge_model.clone(),
            judged_at: r.judged_at,
            judge_time_ms: r.judge_time_ms,
            judge_cost_usd: r.judge_cost_usd,
        }
    }
}

impl EnsembleJudgeResult {
    /// Get individual scores as a vector.
    pub fn individual_scores(&self) -> Vec<f64> {
        self.individual_results.iter().map(|r| r.score).collect()
    }

    /// Get the number of judges used.
    pub fn num_judges(&self) -> usize {
        self.individual_results.len()
    }
}

/// Quality flags for epistemic analysis of generated samples.
///
/// K_i: These flags capture quality signals beyond the score.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityFlags {
    /// Output appears truncated (cut off mid-sentence/word)
    pub truncated: bool,

    /// Contains structured answer tags (<answer>, ## Answer, etc.)
    pub has_answer_tags: bool,

    /// Contains reasoning/thinking steps
    pub has_reasoning: bool,

    /// Contains self-correction patterns ("Wait", "Actually", "Let me reconsider")
    pub self_correction: bool,

    /// Length of reasoning content in characters
    pub reasoning_length: usize,
}

impl QualityFlags {
    /// Analyze output content and compute quality flags.
    pub fn from_output(output: &str) -> Self {
        let trimmed = output.trim();

        // Check for truncation: ends mid-word, mid-sentence, or with incomplete patterns
        let truncated = {
            let last_char = trimmed.chars().last().unwrap_or('.');
            let ends_incomplete = !matches!(
                last_char,
                '.' | '!' | '?' | ')' | ']' | '}' | '"' | '\'' | '>' | '*'
            );
            let ends_with_incomplete_pattern = trimmed.ends_with("...")
                || trimmed.ends_with(" -")
                || trimmed.ends_with(" =")
                || trimmed.ends_with("Wait")
                || trimmed.ends_with("Actually");
            ends_incomplete || ends_with_incomplete_pattern
        };

        // Check for answer tags
        let has_answer_tags = output.contains("<answer>")
            || output.contains("</answer>")
            || output.contains("## Answer")
            || output.contains("**Answer**")
            || output.contains("\\boxed{");

        // Check for reasoning indicators
        let has_reasoning = output.contains("## OBSERVE")
            || output.contains("### OBSERVE")
            || output.contains("K_i")
            || output.contains("B_i")
            || output.contains("<reasoning>")
            || output.contains("Step 1")
            || output.contains("First,")
            || output.contains("Let's think")
            || output.contains("Let me");

        // Check for self-correction patterns
        let self_correction = output.contains("Wait")
            || output.contains("Actually")
            || output.contains("Let me reconsider")
            || output.contains("I made a mistake")
            || output.contains("Correction:")
            || output.contains("On second thought");

        // Measure reasoning length (approximate: content before answer)
        let reasoning_length = if let Some(pos) = output.find("<answer>") {
            pos
        } else if let Some(pos) = output.find("## Answer") {
            pos
        } else {
            output.len()
        };

        Self {
            truncated,
            has_answer_tags,
            has_reasoning,
            self_correction,
            reasoning_length,
        }
    }
}

/// SFT output sample (approved sample for supervised fine-tuning).
///
/// K_i: SFT sample is an approved sample with full metadata for training and analysis.
/// Enhanced with epistemic metadata for quality analysis and traceability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftSample {
    // === Core Fields ===
    /// Unique identifier
    pub id: String,

    /// Input prompt
    pub input: String,

    /// Generated output
    pub output: String,

    /// Extracted answer (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,

    /// Model used for generation
    pub model: String,

    /// Quality score from judge (0.0 - 1.0)
    pub score: f64,

    // === P0: Traceability (HIGH priority) ===
    /// Source problem ID for tracing back to input
    pub problem_id: String,

    /// Input tokens used for generation
    pub tokens_in: u32,

    /// Output tokens generated
    pub tokens_out: u32,

    /// Judge's reasoning/explanation for the score
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_reasoning: Option<String>,

    // === P1: Efficiency & Provenance (MEDIUM priority) ===
    /// Generation time in milliseconds
    pub generation_time_ms: u64,

    /// Model(s) used for judging (comma-separated if ensemble)
    pub judge_model: String,

    /// Explicit verdict from judge
    pub verdict: Verdict,

    // === P2: Quality Signals (Epistemic) ===
    /// Quality flags for epistemic analysis
    pub quality_flags: QualityFlags,

    // === P3: Ensemble Judging Metadata ===
    /// Confidence level from ensemble judging
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_confidence: Option<Confidence>,

    /// Score standard deviation (disagreement measure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_std_dev: Option<f64>,

    /// Number of judges used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_judges: Option<usize>,

    /// Individual judge scores (for analysis)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub individual_scores: Option<Vec<f64>>,

    // === Cost & Metadata ===
    /// Total cost (generation + judging) in USD
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,

    /// Optional metadata from problem
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

impl SftSample {
    /// Create an SFT sample from a sample and ensemble judge result.
    ///
    /// K_i: Preserves all metadata from generation and judging phases.
    /// This enables full traceability and epistemic analysis.
    pub fn from_judged(sample: Sample, judge: EnsembleJudgeResult, include_cost: bool) -> Self {
        // Compute quality flags from output
        let quality_flags = QualityFlags::from_output(&sample.output);

        // Extract ensemble metadata before moving judge
        let is_ensemble = judge.num_judges() > 1;
        let num_judges = judge.num_judges();
        let individual_scores = judge.individual_scores();

        Self {
            // Core fields
            id: sample.id,
            input: sample.input.clone(),
            output: sample.output,
            answer: sample.answer,
            model: sample.model,
            score: judge.score,

            // P0: Traceability
            problem_id: sample.problem_id,
            tokens_in: sample.tokens_in,
            tokens_out: sample.tokens_out,
            judge_reasoning: Some(judge.reasoning),

            // P1: Efficiency & Provenance
            generation_time_ms: sample.generation_time_ms,
            judge_model: judge.judge_models,
            verdict: judge.verdict,

            // P2: Quality Signals
            quality_flags,

            // P3: Ensemble Metadata (only if ensemble was used)
            judge_confidence: if is_ensemble {
                Some(judge.confidence)
            } else {
                None
            },
            score_std_dev: if is_ensemble {
                Some(judge.score_std_dev)
            } else {
                None
            },
            num_judges: if is_ensemble { Some(num_judges) } else { None },
            individual_scores: if is_ensemble {
                Some(individual_scores)
            } else {
                None
            },

            // Cost & Metadata
            cost_usd: if include_cost {
                Some(sample.cost_usd + judge.judge_cost_usd)
            } else {
                None
            },
            metadata: sample.metadata,
        }
    }
}

/// DPO preference pair (chosen vs rejected for preference learning).
///
/// K_i: A DPO pair has the same input with chosen (high score) and rejected (low score) outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoPair {
    /// Unique identifier for this pair
    pub id: String,

    /// Problem ID (source)
    pub problem_id: String,

    /// Input prompt (same for both)
    pub input: String,

    /// Chosen response (higher score)
    pub chosen: String,

    /// Rejected response (lower score)
    pub rejected: String,

    /// Score of chosen response
    pub chosen_score: f64,

    /// Score of rejected response
    pub rejected_score: f64,

    /// Model used for chosen response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_model: Option<String>,

    /// Model used for rejected response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_model: Option<String>,

    /// Optional metadata
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

impl DpoPair {
    /// Create a DPO pair from two judged samples.
    ///
    /// K_i: Chosen has higher score than rejected.
    pub fn from_samples(
        problem_id: String,
        input: String,
        chosen_sample: &Sample,
        chosen_judge: &EnsembleJudgeResult,
        rejected_sample: &Sample,
        rejected_judge: &EnsembleJudgeResult,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            problem_id,
            input,
            chosen: chosen_sample.output.clone(),
            rejected: rejected_sample.output.clone(),
            chosen_score: chosen_judge.score,
            rejected_score: rejected_judge.score,
            chosen_model: Some(chosen_sample.model.clone()),
            rejected_model: Some(rejected_sample.model.clone()),
            metadata,
        }
    }
}

/// Statistics for a generation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunStats {
    /// Total problems processed
    pub total_problems: usize,

    /// Total samples generated
    pub total_generated: usize,

    /// Total samples judged
    pub total_judged: usize,

    /// Total samples approved
    pub total_approved: usize,

    /// Total samples rejected
    pub total_rejected: usize,

    /// Total generation cost (USD)
    pub generation_cost_usd: f64,

    /// Total judging cost (USD)
    pub judge_cost_usd: f64,

    /// Total runtime in seconds
    pub runtime_secs: f64,

    /// Samples per hour throughput
    pub throughput_per_hour: f64,

    /// Approval rate (0.0 - 1.0)
    pub approval_rate: f64,
}

impl RunStats {
    /// Calculate derived stats.
    pub fn finalize(&mut self) {
        if self.total_judged > 0 {
            self.approval_rate = self.total_approved as f64 / self.total_judged as f64;
        }
        if self.runtime_secs > 0.0 {
            self.throughput_per_hour = self.total_approved as f64 / self.runtime_secs * 3600.0;
        }
    }
}
