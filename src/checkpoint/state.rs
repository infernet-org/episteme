//! Checkpoint state management for resumable pipelines.
//!
//! Epistemic foundation:
//! - K_i: Checkpoints track which problems have been processed
//! - K_i: State is persisted to disk atomically (write-then-rename)
//! - B_i: Checkpoint file may not exist → Option
//! - I^B: Crash during write → backup file provides recovery

use crate::models::{DpogenError, Problem, Result, RunStats, Verdict};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Status of a problem in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProblemStatus {
    /// Not yet processed
    Pending,
    /// Generation complete, awaiting judgment
    Generated,
    /// Fully processed and approved
    Approved,
    /// Fully processed and rejected
    Rejected,
    /// Failed during processing
    Failed,
}

/// Checkpoint entry for a single problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemCheckpoint {
    /// Problem ID
    pub id: String,
    /// Current status
    pub status: ProblemStatus,
    /// Score if judged
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Model used for generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Generation cost
    #[serde(default)]
    pub gen_cost: f64,
    /// Judge cost
    #[serde(default)]
    pub judge_cost: f64,
    /// Timestamp of last update
    pub updated_at: DateTime<Utc>,
}

/// Checkpoint state for a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    /// Pipeline type (sft or dpo)
    pub pipeline: String,
    /// Total problems to process
    pub total_problems: usize,
    /// Problem statuses
    pub problems: HashMap<String, ProblemCheckpoint>,
    /// Statistics
    pub stats: CheckpointStats,
    /// When the run started
    pub started_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
}

/// Statistics tracked in checkpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckpointStats {
    pub pending: usize,
    pub generated: usize,
    pub approved: usize,
    pub rejected: usize,
    pub failed: usize,
    pub generation_cost: f64,
    pub judge_cost: f64,
}

impl CheckpointState {
    /// Create a new checkpoint state.
    pub fn new(pipeline: &str, problems: &[Problem]) -> Self {
        let now = Utc::now();
        let mut state = Self {
            pipeline: pipeline.to_string(),
            total_problems: problems.len(),
            problems: HashMap::with_capacity(problems.len()),
            stats: CheckpointStats {
                pending: problems.len(),
                ..Default::default()
            },
            started_at: now,
            updated_at: now,
        };

        for problem in problems {
            state.problems.insert(
                problem.id.clone(),
                ProblemCheckpoint {
                    id: problem.id.clone(),
                    status: ProblemStatus::Pending,
                    score: None,
                    model: None,
                    gen_cost: 0.0,
                    judge_cost: 0.0,
                    updated_at: now,
                },
            );
        }

        state
    }

    /// Get pending problem IDs.
    pub fn pending_ids(&self) -> Vec<String> {
        self.problems
            .iter()
            .filter(|(_, cp)| cp.status == ProblemStatus::Pending)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Mark a problem as generated.
    pub fn mark_generated(&mut self, problem_id: &str, model: &str, cost: f64) {
        if let Some(cp) = self.problems.get_mut(problem_id) {
            if cp.status == ProblemStatus::Pending {
                self.stats.pending -= 1;
                self.stats.generated += 1;
            }
            cp.status = ProblemStatus::Generated;
            cp.model = Some(model.to_string());
            cp.gen_cost = cost;
            cp.updated_at = Utc::now();
            self.stats.generation_cost += cost;
        }
        self.updated_at = Utc::now();
    }

    /// Mark a problem as judged (approved or rejected).
    pub fn mark_judged(&mut self, problem_id: &str, score: f64, verdict: Verdict, judge_cost: f64) {
        if let Some(cp) = self.problems.get_mut(problem_id) {
            if cp.status == ProblemStatus::Generated {
                self.stats.generated -= 1;
            }
            cp.status = match verdict {
                Verdict::Approve => {
                    self.stats.approved += 1;
                    ProblemStatus::Approved
                }
                Verdict::Reject => {
                    self.stats.rejected += 1;
                    ProblemStatus::Rejected
                }
            };
            cp.score = Some(score);
            cp.judge_cost = judge_cost;
            cp.updated_at = Utc::now();
            self.stats.judge_cost += judge_cost;
        }
        self.updated_at = Utc::now();
    }

    /// Mark a problem as failed.
    pub fn mark_failed(&mut self, problem_id: &str) {
        if let Some(cp) = self.problems.get_mut(problem_id) {
            match cp.status {
                ProblemStatus::Pending => self.stats.pending -= 1,
                ProblemStatus::Generated => self.stats.generated -= 1,
                _ => {}
            }
            cp.status = ProblemStatus::Failed;
            cp.updated_at = Utc::now();
            self.stats.failed += 1;
        }
        self.updated_at = Utc::now();
    }

    /// Check if all problems are processed.
    pub fn is_complete(&self) -> bool {
        self.stats.pending == 0 && self.stats.generated == 0
    }

    /// Get progress percentage.
    pub fn progress_percent(&self) -> f64 {
        if self.total_problems == 0 {
            return 100.0;
        }
        let processed = self.stats.approved + self.stats.rejected + self.stats.failed;
        (processed as f64 / self.total_problems as f64) * 100.0
    }

    /// Convert to RunStats.
    pub fn to_run_stats(&self, runtime_secs: f64) -> RunStats {
        let mut stats = RunStats {
            total_problems: self.total_problems,
            total_generated: self.stats.approved + self.stats.rejected + self.stats.generated,
            total_judged: self.stats.approved + self.stats.rejected,
            total_approved: self.stats.approved,
            total_rejected: self.stats.rejected,
            generation_cost_usd: self.stats.generation_cost,
            judge_cost_usd: self.stats.judge_cost,
            runtime_secs,
            ..Default::default()
        };
        stats.finalize();
        stats
    }
}

/// Checkpoint manager for persisting and loading checkpoint state.
pub struct CheckpointManager {
    /// Directory for checkpoint files
    dir: PathBuf,
    /// Path to main checkpoint file
    checkpoint_path: PathBuf,
    /// Path to backup file
    backup_path: PathBuf,
    /// Current state
    state: Option<CheckpointState>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).map_err(|e| DpogenError::io("creating checkpoint dir", e))?;

        Ok(Self {
            dir: dir.to_path_buf(),
            checkpoint_path: dir.join("checkpoint.json"),
            backup_path: dir.join("checkpoint.backup.json"),
            state: None,
        })
    }

    /// Check if a checkpoint exists.
    pub fn exists(&self) -> bool {
        self.checkpoint_path.exists()
    }

    /// Initialize a new checkpoint or load existing one.
    pub fn init_or_load(&mut self, pipeline: &str, problems: &[Problem]) -> Result<&CheckpointState> {
        if self.exists() {
            self.load()?;
            info!(
                pending = self.state.as_ref().unwrap().stats.pending,
                approved = self.state.as_ref().unwrap().stats.approved,
                "Resuming from checkpoint"
            );
        } else {
            self.state = Some(CheckpointState::new(pipeline, problems));
            self.save()?;
            info!(total = problems.len(), "Created new checkpoint");
        }
        Ok(self.state.as_ref().unwrap())
    }

    /// Load checkpoint from disk.
    pub fn load(&mut self) -> Result<&CheckpointState> {
        let file = File::open(&self.checkpoint_path)
            .map_err(|e| DpogenError::io("opening checkpoint", e))?;
        let reader = BufReader::new(file);
        let state: CheckpointState = serde_json::from_reader(reader)
            .map_err(|e| DpogenError::ParseError(format!("Invalid checkpoint: {}", e)))?;
        
        self.state = Some(state);
        Ok(self.state.as_ref().unwrap())
    }

    /// Save checkpoint to disk (atomic write).
    pub fn save(&self) -> Result<()> {
        let state = self.state.as_ref().ok_or_else(|| {
            DpogenError::Internal("No checkpoint state to save".to_string())
        })?;

        // Backup existing checkpoint
        if self.checkpoint_path.exists() {
            fs::copy(&self.checkpoint_path, &self.backup_path)
                .map_err(|e| DpogenError::io("backing up checkpoint", e))?;
        }

        // Write to temp file
        let temp_path = self.dir.join("checkpoint.tmp.json");
        let file = File::create(&temp_path)
            .map_err(|e| DpogenError::io("creating temp checkpoint", e))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, state)
            .map_err(|e| DpogenError::Internal(format!("Serializing checkpoint: {}", e)))?;

        // Atomic rename
        fs::rename(&temp_path, &self.checkpoint_path)
            .map_err(|e| DpogenError::io("renaming checkpoint", e))?;

        debug!("Checkpoint saved");
        Ok(())
    }

    /// Get mutable reference to state.
    pub fn state_mut(&mut self) -> Option<&mut CheckpointState> {
        self.state.as_mut()
    }

    /// Get reference to state.
    pub fn state(&self) -> Option<&CheckpointState> {
        self.state.as_ref()
    }

    /// Mark generated and save.
    pub fn mark_generated(&mut self, problem_id: &str, model: &str, cost: f64) -> Result<()> {
        if let Some(state) = &mut self.state {
            state.mark_generated(problem_id, model, cost);
        }
        self.save()
    }

    /// Mark judged and save.
    pub fn mark_judged(&mut self, problem_id: &str, score: f64, verdict: Verdict, judge_cost: f64) -> Result<()> {
        if let Some(state) = &mut self.state {
            state.mark_judged(problem_id, score, verdict, judge_cost);
        }
        self.save()
    }

    /// Mark failed and save.
    pub fn mark_failed(&mut self, problem_id: &str) -> Result<()> {
        if let Some(state) = &mut self.state {
            state.mark_failed(problem_id);
        }
        self.save()
    }

    /// Filter problems to only pending ones.
    pub fn filter_pending(&self, problems: Vec<Problem>) -> Vec<Problem> {
        let state = match &self.state {
            Some(s) => s,
            None => return problems,
        };

        let pending: HashSet<_> = state.pending_ids().into_iter().collect();
        problems
            .into_iter()
            .filter(|p| pending.contains(&p.id))
            .collect()
    }

    /// Get checkpoint directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}
