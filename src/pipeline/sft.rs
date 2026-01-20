//! SFT (Supervised Fine-Tuning) data generation pipeline.
//!
//! Pipeline flow:
//! Problems → Worker Pool → Samples → Judge Pool → Approved Samples → JSONL

use crate::checkpoint::CheckpointManager;
use crate::client::OpenRouterClient;
use crate::models::{Config, DpogenError, Problem, Result, RunStats, SftSample, Verdict};
use crate::pool::{JudgePool, WorkerPool};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// SFT pipeline for generating supervised fine-tuning data.
#[allow(dead_code)]
pub struct SftPipeline {
    worker_pool: WorkerPool,
    judge_pool: JudgePool,
    config: Config,
    client: Arc<OpenRouterClient>,
}

impl SftPipeline {
    /// Create a new SFT pipeline from configuration.
    pub fn new(config: Config, client: Arc<OpenRouterClient>) -> Result<Self> {
        // Load prompts
        let system_prompt = std::fs::read_to_string(&config.generation.system_prompt)
            .map_err(|e| DpogenError::io("reading system prompt", e))?;

        let judge_prompt = std::fs::read_to_string(&config.generation.judge_prompt)
            .map_err(|e| DpogenError::io("reading judge prompt", e))?;

        let worker_pool = WorkerPool::new(
            Arc::clone(&client),
            config.workers.models.clone(),
            system_prompt,
            config.workers.size,
        );

        let judge_pool = JudgePool::new(
            Arc::clone(&client),
            config.judges.models.clone(),
            judge_prompt,
            config.judges.size,
            config.generation.approval_threshold,
        );

        Ok(Self {
            worker_pool,
            judge_pool,
            config,
            client,
        })
    }

    /// Load problems from a JSONL file.
    pub fn load_problems(path: &Path) -> Result<Vec<Problem>> {
        let file = File::open(path).map_err(|e| DpogenError::io("opening problems file", e))?;
        let reader = BufReader::new(file);
        let mut problems = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| DpogenError::io("reading problems file", e))?;
            if line.trim().is_empty() {
                continue;
            }
            let problem: Problem = serde_json::from_str(&line).map_err(|e| {
                DpogenError::ParseError(format!("Line {}: {}", line_num + 1, e))
            })?;
            problems.push(problem);
        }

        info!(count = problems.len(), "Loaded problems");
        Ok(problems)
    }

    /// Run the SFT pipeline.
    pub async fn run(&self, problems: Vec<Problem>, output_path: &Path) -> Result<RunStats> {
        let start = Instant::now();
        let total = problems.len();

        info!(
            total_problems = total,
            workers = self.config.workers.size,
            judges = self.config.judges.size,
            "Starting SFT pipeline"
        );

        // Setup progress bar
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        // Open output file
        let output_file =
            File::create(output_path).map_err(|e| DpogenError::io("creating output file", e))?;
        let mut writer = BufWriter::new(output_file);

        let mut stats = RunStats::default();
        stats.total_problems = total;

        // Process in batches for better resource utilization
        let batch_size = (self.config.workers.size * 2).max(10);
        let mut approved_count = 0;
        let mut rejected_count = 0;

        for batch_start in (0..problems.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(problems.len());
            let batch: Vec<Problem> = problems[batch_start..batch_end].to_vec();

            // Generate samples
            let (samples, failed_gen) = self.worker_pool.generate_batch(batch).await;
            stats.total_generated += samples.len();

            if !failed_gen.is_empty() {
                warn!(count = failed_gen.len(), "Some problems failed generation");
            }

            // Judge samples
            let (judged, failed_judge) = self.judge_pool.evaluate_batch(samples).await;
            stats.total_judged += judged.len();

            if !failed_judge.is_empty() {
                warn!(count = failed_judge.len(), "Some samples failed evaluation");
            }

            // Write approved samples
            for (sample, judge_result) in judged {
                stats.generation_cost_usd += sample.cost_usd;
                stats.judge_cost_usd += judge_result.judge_cost_usd;

                match judge_result.verdict {
                    Verdict::Approve => {
                        approved_count += 1;
                        stats.total_approved += 1;

                        let sft_sample = SftSample::from_judged(
                            sample,
                            judge_result,
                            self.config.output.track_costs,
                        );

                        let json = serde_json::to_string(&sft_sample).map_err(|e| {
                            DpogenError::Internal(format!("Failed to serialize sample: {}", e))
                        })?;

                        writeln!(writer, "{}", json)
                            .map_err(|e| DpogenError::io("writing output", e))?;
                    }
                    Verdict::Reject => {
                        rejected_count += 1;
                        stats.total_rejected += 1;

                        // Optionally write rejected samples
                        if self.config.output.include_rejected {
                            let sft_sample = SftSample::from_judged(
                                sample,
                                judge_result,
                                self.config.output.track_costs,
                            );

                            let json = serde_json::to_string(&sft_sample).map_err(|e| {
                                DpogenError::Internal(format!("Failed to serialize sample: {}", e))
                            })?;

                            writeln!(writer, "{}", json)
                                .map_err(|e| DpogenError::io("writing output", e))?;
                        }
                    }
                }
            }

            // Flush periodically
            writer
                .flush()
                .map_err(|e| DpogenError::io("flushing output", e))?;

            // Update progress
            pb.set_position(batch_end as u64);
            pb.set_message(format!(
                "approved: {}, rejected: {}",
                approved_count, rejected_count
            ));
        }

        // Finalize
        writer
            .flush()
            .map_err(|e| DpogenError::io("flushing output", e))?;
        pb.finish_with_message(format!(
            "Done! {} approved, {} rejected",
            approved_count, rejected_count
        ));

        stats.runtime_secs = start.elapsed().as_secs_f64();
        stats.finalize();

        info!(
            approved = stats.total_approved,
            rejected = stats.total_rejected,
            approval_rate = format!("{:.1}%", stats.approval_rate * 100.0),
            throughput = format!("{:.0}/hr", stats.throughput_per_hour),
            cost = format!("${:.4}", stats.generation_cost_usd + stats.judge_cost_usd),
            "SFT pipeline complete"
        );

        Ok(stats)
    }

    /// Run the SFT pipeline with optional checkpoint support.
    pub async fn run_with_checkpoint(
        &self,
        problems: Vec<Problem>,
        output_path: &Path,
        checkpoint: Option<CheckpointManager>,
    ) -> Result<RunStats> {
        // If no checkpoint, use the regular run method
        let mut checkpoint = match checkpoint {
            Some(c) => c,
            None => return self.run(problems, output_path).await,
        };

        let start = Instant::now();
        let total = problems.len();

        // Filter to only pending problems
        let pending_problems = checkpoint.filter_pending(problems);
        let pending_count = pending_problems.len();

        if pending_count == 0 {
            info!("All problems already processed, nothing to do");
            let state = checkpoint.state().unwrap();
            return Ok(state.to_run_stats(0.0));
        }

        info!(
            total_problems = total,
            pending = pending_count,
            already_done = total - pending_count,
            workers = self.config.workers.size,
            judges = self.config.judges.size,
            "Resuming SFT pipeline"
        );

        // Setup progress bar
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        // Set initial progress from checkpoint
        let already_done = total - pending_count;
        pb.set_position(already_done as u64);

        // Open output file in append mode for resume
        let output_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .map_err(|e| DpogenError::io("opening output file", e))?;
        let mut writer = BufWriter::new(output_file);

        // Process in batches
        let batch_size = (self.config.workers.size * 2).max(10);
        let mut approved_count = 0;
        let mut rejected_count = 0;

        for batch_start in (0..pending_problems.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(pending_problems.len());
            let batch: Vec<Problem> = pending_problems[batch_start..batch_end].to_vec();

            // Generate samples
            let (samples, failed_gen) = self.worker_pool.generate_batch(batch).await;

            // Mark failed generations
            for problem_id in &failed_gen {
                checkpoint.mark_failed(problem_id)?;
            }

            if !failed_gen.is_empty() {
                warn!(count = failed_gen.len(), "Some problems failed generation");
            }

            // Judge samples
            let (judged, failed_judge) = self.judge_pool.evaluate_batch(samples).await;

            if !failed_judge.is_empty() {
                warn!(count = failed_judge.len(), "Some samples failed evaluation");
            }

            // Write approved samples and update checkpoint
            for (sample, judge_result) in judged {
                // Extract original problem ID (remove any suffix)
                let problem_id = sample.problem_id.split('_').next().unwrap_or(&sample.problem_id).to_string();

                match judge_result.verdict {
                    Verdict::Approve => {
                        approved_count += 1;

                        let sft_sample = SftSample::from_judged(
                            sample.clone(),
                            judge_result.clone(),
                            self.config.output.track_costs,
                        );

                        let json = serde_json::to_string(&sft_sample).map_err(|e| {
                            DpogenError::Internal(format!("Failed to serialize sample: {}", e))
                        })?;

                        writeln!(writer, "{}", json)
                            .map_err(|e| DpogenError::io("writing output", e))?;

                        checkpoint.mark_judged(
                            &problem_id,
                            judge_result.score,
                            Verdict::Approve,
                            judge_result.judge_cost_usd,
                        )?;
                    }
                    Verdict::Reject => {
                        rejected_count += 1;

                        // Optionally write rejected samples
                        if self.config.output.include_rejected {
                            let sft_sample = SftSample::from_judged(
                                sample.clone(),
                                judge_result.clone(),
                                self.config.output.track_costs,
                            );

                            let json = serde_json::to_string(&sft_sample).map_err(|e| {
                                DpogenError::Internal(format!("Failed to serialize sample: {}", e))
                            })?;

                            writeln!(writer, "{}", json)
                                .map_err(|e| DpogenError::io("writing output", e))?;
                        }

                        checkpoint.mark_judged(
                            &problem_id,
                            judge_result.score,
                            Verdict::Reject,
                            judge_result.judge_cost_usd,
                        )?;
                    }
                }
            }

            // Flush periodically
            writer
                .flush()
                .map_err(|e| DpogenError::io("flushing output", e))?;

            // Update progress
            pb.set_position((already_done + batch_end) as u64);
            pb.set_message(format!(
                "approved: {}, rejected: {}",
                approved_count, rejected_count
            ));
        }

        // Finalize
        writer
            .flush()
            .map_err(|e| DpogenError::io("flushing output", e))?;
        pb.finish_with_message(format!(
            "Done! {} approved, {} rejected",
            approved_count, rejected_count
        ));

        let runtime = start.elapsed().as_secs_f64();
        let stats = checkpoint.state().unwrap().to_run_stats(runtime);

        info!(
            approved = stats.total_approved,
            rejected = stats.total_rejected,
            approval_rate = format!("{:.1}%", stats.approval_rate * 100.0),
            throughput = format!("{:.0}/hr", stats.throughput_per_hour),
            cost = format!("${:.4}", stats.generation_cost_usd + stats.judge_cost_usd),
            "SFT pipeline complete"
        );

        Ok(stats)
    }
}
