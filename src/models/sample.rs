//! Sample and result types for dpogen.
//!
//! K_i: These types represent the core data flow through the pipeline.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

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

/// SFT output sample (approved sample for supervised fine-tuning).
///
/// K_i: SFT sample is an approved sample with all metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftSample {
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

    /// Quality score from judge
    pub score: f64,

    /// Total cost (generation + judging)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,

    /// Optional metadata
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

impl SftSample {
    /// Create an SFT sample from a sample and judge result.
    pub fn from_judged(sample: Sample, judge: JudgeResult, include_cost: bool) -> Self {
        Self {
            id: sample.id,
            input: sample.input,
            output: sample.output,
            answer: sample.answer,
            model: sample.model,
            score: judge.score,
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
        chosen_judge: &JudgeResult,
        rejected_sample: &Sample,
        rejected_judge: &JudgeResult,
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
