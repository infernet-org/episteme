//! DPO (Direct Preference Optimization) data generation pipeline.
//!
//! Pipeline flow:
//! Problems → Worker Pool (N responses each) → Judge Pool → Pair (best vs worst) → JSONL
//!
//! K_i: DPO requires preference pairs (chosen, rejected) from the same input.
//! Strategy: Generate N responses per problem, judge all, pair highest vs lowest score.

use crate::checkpoint::CheckpointManager;
use crate::client::OpenRouterClient;
use crate::models::{
    Config, DpoPair, EnsembleJudgeResult, EpistemeError, Problem, Result, RunStats, Sample, Verdict,
};
use crate::pool::{JudgePool, WorkerPool};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// DPO pipeline for generating preference pairs.
#[allow(dead_code)]
pub struct DpoPipeline {
    worker_pool: WorkerPool,
    judge_pool: JudgePool,
    config: Config,
    client: Arc<OpenRouterClient>,
    responses_per_problem: usize,
}

impl DpoPipeline {
    /// Create a new DPO pipeline from configuration.
    pub fn new(config: Config, client: Arc<OpenRouterClient>) -> Result<Self> {
        // Load prompts
        let system_prompt = std::fs::read_to_string(&config.generation.system_prompt)
            .map_err(|e| EpistemeError::io("reading system prompt", e))?;

        let judge_prompt = std::fs::read_to_string(&config.generation.judge_prompt)
            .map_err(|e| EpistemeError::io("reading judge prompt", e))?;

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
            config.judges.ensemble.clone(),
        );

        let responses_per_problem = config.generation.responses_per_problem.max(2);

        Ok(Self {
            worker_pool,
            judge_pool,
            config,
            client,
            responses_per_problem,
        })
    }

    /// Load problems from a JSONL file.
    pub fn load_problems(path: &Path) -> Result<Vec<Problem>> {
        let file = File::open(path).map_err(|e| EpistemeError::io("opening problems file", e))?;
        let reader = BufReader::new(file);
        let mut problems = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| EpistemeError::io("reading problems file", e))?;
            if line.trim().is_empty() {
                continue;
            }
            let problem: Problem = serde_json::from_str(&line)
                .map_err(|e| EpistemeError::ParseError(format!("Line {}: {}", line_num + 1, e)))?;
            problems.push(problem);
        }

        info!(count = problems.len(), "Loaded problems");
        Ok(problems)
    }

    /// Create a DPO pair from judged samples.
    ///
    /// K_i: Chosen has higher score than rejected.
    fn create_pair(
        problem: &Problem,
        judged_samples: &[(Sample, EnsembleJudgeResult)],
    ) -> Option<DpoPair> {
        if judged_samples.len() < 2 {
            return None;
        }

        // Sort by score descending
        let mut sorted: Vec<_> = judged_samples.iter().collect();
        sorted.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap());

        let (chosen_sample, chosen_judge) = sorted.first()?;
        let (rejected_sample, rejected_judge) = sorted.last()?;

        // Only create pair if there's meaningful score difference
        let score_diff = chosen_judge.score - rejected_judge.score;
        if score_diff < 0.1 {
            debug!(
                problem_id = %problem.id,
                score_diff = score_diff,
                "Score difference too small for DPO pair"
            );
            return None;
        }

        Some(DpoPair::from_samples(
            problem.id.clone(),
            problem.input.clone(),
            chosen_sample,
            chosen_judge,
            rejected_sample,
            rejected_judge,
            problem.metadata.clone(),
        ))
    }

    /// Run the DPO pipeline.
    pub async fn run(&self, problems: Vec<Problem>, output_path: &Path) -> Result<RunStats> {
        let start = Instant::now();
        let total = problems.len();
        let total_generations = total * self.responses_per_problem;

        info!(
            total_problems = total,
            responses_per_problem = self.responses_per_problem,
            total_generations = total_generations,
            workers = self.config.workers.size,
            judges = self.config.judges.size,
            "Starting DPO pipeline"
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
            File::create(output_path).map_err(|e| EpistemeError::io("creating output file", e))?;
        let mut writer = BufWriter::new(output_file);

        let mut stats = RunStats {
            total_problems: total,
            ..Default::default()
        };

        let mut pairs_created = 0;
        let mut pairs_skipped = 0;

        // Process each problem
        for (idx, problem) in problems.into_iter().enumerate() {
            // Generate N responses for this problem
            let generation_tasks: Vec<Problem> = (0..self.responses_per_problem)
                .map(|i| Problem {
                    id: format!("{}_{}", problem.id, i),
                    input: problem.input.clone(),
                    metadata: problem.metadata.clone(),
                })
                .collect();

            let (samples, _failed_gen) = self.worker_pool.generate_batch(generation_tasks).await;
            stats.total_generated += samples.len();

            if samples.len() < 2 {
                warn!(
                    problem_id = %problem.id,
                    generated = samples.len(),
                    "Not enough samples generated for DPO pair"
                );
                pairs_skipped += 1;
                pb.set_position((idx + 1) as u64);
                continue;
            }

            // Judge all samples
            let (judged, _failed_judge) = self.judge_pool.evaluate_batch(samples).await;
            stats.total_judged += judged.len();

            // Track costs
            for (sample, judge_result) in &judged {
                stats.generation_cost_usd += sample.cost_usd;
                stats.judge_cost_usd += judge_result.judge_cost_usd;
            }

            // Create DPO pair
            if let Some(pair) = Self::create_pair(&problem, &judged) {
                pairs_created += 1;
                stats.total_approved += 1;

                let json = serde_json::to_string(&pair).map_err(|e| {
                    EpistemeError::Internal(format!("Failed to serialize pair: {e}"))
                })?;

                writeln!(writer, "{json}").map_err(|e| EpistemeError::io("writing output", e))?;
            } else {
                pairs_skipped += 1;
                stats.total_rejected += 1;
            }

            // Flush periodically
            if (idx + 1) % 10 == 0 {
                writer
                    .flush()
                    .map_err(|e| EpistemeError::io("flushing output", e))?;
            }

            // Update progress
            pb.set_position((idx + 1) as u64);
            pb.set_message(format!("pairs: {pairs_created}, skipped: {pairs_skipped}"));
        }

        // Finalize
        writer
            .flush()
            .map_err(|e| EpistemeError::io("flushing output", e))?;
        pb.finish_with_message(format!(
            "Done! {pairs_created} pairs created, {pairs_skipped} skipped"
        ));

        stats.runtime_secs = start.elapsed().as_secs_f64();
        stats.finalize();

        info!(
            pairs_created = pairs_created,
            pairs_skipped = pairs_skipped,
            total_generated = stats.total_generated,
            total_judged = stats.total_judged,
            throughput = format!("{:.0} pairs/hr", stats.throughput_per_hour),
            cost = format!("${:.4}", stats.generation_cost_usd + stats.judge_cost_usd),
            "DPO pipeline complete"
        );

        Ok(stats)
    }

    /// Run the DPO pipeline with optional checkpoint support.
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

        let total_generations = pending_count * self.responses_per_problem;

        info!(
            total_problems = total,
            pending = pending_count,
            already_done = total - pending_count,
            responses_per_problem = self.responses_per_problem,
            total_generations = total_generations,
            workers = self.config.workers.size,
            judges = self.config.judges.size,
            "Resuming DPO pipeline"
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
            .map_err(|e| EpistemeError::io("opening output file", e))?;
        let mut writer = BufWriter::new(output_file);

        let mut pairs_created = 0;
        let mut pairs_skipped = 0;

        // Process each pending problem
        for (idx, problem) in pending_problems.into_iter().enumerate() {
            // Generate N responses for this problem
            let generation_tasks: Vec<Problem> = (0..self.responses_per_problem)
                .map(|i| Problem {
                    id: format!("{}_{}", problem.id, i),
                    input: problem.input.clone(),
                    metadata: problem.metadata.clone(),
                })
                .collect();

            let (samples, _failed_gen) = self.worker_pool.generate_batch(generation_tasks).await;

            if samples.len() < 2 {
                warn!(
                    problem_id = %problem.id,
                    generated = samples.len(),
                    "Not enough samples generated for DPO pair"
                );
                pairs_skipped += 1;
                checkpoint.mark_failed(&problem.id)?;
                pb.set_position((already_done + idx + 1) as u64);
                continue;
            }

            // Calculate generation cost
            let gen_cost: f64 = samples.iter().map(|s| s.cost_usd).sum();

            // Mark as generated (use first model for tracking)
            let model = samples
                .first()
                .map(|s| s.model.as_str())
                .unwrap_or("unknown");
            checkpoint.mark_generated(&problem.id, model, gen_cost)?;

            // Judge all samples
            let (judged, _failed_judge) = self.judge_pool.evaluate_batch(samples).await;

            // Calculate judge cost
            let judge_cost: f64 = judged.iter().map(|(_, jr)| jr.judge_cost_usd).sum();

            // Create DPO pair
            if let Some(pair) = Self::create_pair(&problem, &judged) {
                pairs_created += 1;

                let json = serde_json::to_string(&pair).map_err(|e| {
                    EpistemeError::Internal(format!("Failed to serialize pair: {e}"))
                })?;

                writeln!(writer, "{json}").map_err(|e| EpistemeError::io("writing output", e))?;

                // Get the score from the chosen response
                let chosen_score = judged
                    .iter()
                    .max_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap())
                    .map(|(_, jr)| jr.score)
                    .unwrap_or(0.0);

                checkpoint.mark_judged(&problem.id, chosen_score, Verdict::Approve, judge_cost)?;
            } else {
                pairs_skipped += 1;
                checkpoint.mark_judged(&problem.id, 0.0, Verdict::Reject, judge_cost)?;
            }

            // Flush periodically
            if (idx + 1) % 10 == 0 {
                writer
                    .flush()
                    .map_err(|e| EpistemeError::io("flushing output", e))?;
            }

            // Update progress
            pb.set_position((already_done + idx + 1) as u64);
            pb.set_message(format!("pairs: {pairs_created}, skipped: {pairs_skipped}"));
        }

        // Finalize
        writer
            .flush()
            .map_err(|e| EpistemeError::io("flushing output", e))?;
        pb.finish_with_message(format!(
            "Done! {pairs_created} pairs created, {pairs_skipped} skipped"
        ));

        let runtime = start.elapsed().as_secs_f64();
        let stats = checkpoint.state().unwrap().to_run_stats(runtime);

        info!(
            pairs_created = pairs_created,
            pairs_skipped = pairs_skipped,
            total_generated = stats.total_generated,
            total_judged = stats.total_judged,
            throughput = format!("{:.0} pairs/hr", stats.throughput_per_hour),
            cost = format!("${:.4}", stats.generation_cost_usd + stats.judge_cost_usd),
            "DPO pipeline complete"
        );

        Ok(stats)
    }
}
