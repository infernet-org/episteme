//! Transaction support for atomic checkpoint + output operations.
//!
//! Epistemic foundation:
//! - K_i: Transaction ensures atomicity - all operations succeed or all fail
//! - K_i: Uses write-ahead pattern: record intent → execute → commit
//! - B_i: Partial failure → rollback to consistent state
//! - I^B: Crash during transaction → recovery via pending transaction file

use crate::models::{EpistemeError, Result, Verdict};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Reason for marking an item for retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryReason {
    /// Generation failed (API error, timeout, etc.)
    GenerationFailed,
    /// Judgment failed (API error, timeout, etc.)
    JudgmentFailed,
    /// Not enough samples generated for DPO pair
    InsufficientSamples,
    /// Transaction aborted before completion
    TransactionAborted,
}

/// A pending operation within a transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PendingOperation {
    /// Mark a problem as generated
    Generated {
        problem_id: String,
        model: String,
        cost: f64,
    },
    /// Mark a problem as judged with output
    Judged {
        problem_id: String,
        score: f64,
        verdict: Verdict,
        judge_cost: f64,
        output_json: String,
    },
    /// Mark a problem as failed
    Failed {
        problem_id: String,
        reason: RetryReason,
    },
}

/// Transaction state persisted to disk for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionState {
    /// Unique transaction ID
    pub id: String,
    /// Operations to be committed
    pub operations: Vec<PendingOperation>,
    /// Whether transaction was committed
    pub committed: bool,
    /// Timestamp when transaction started
    pub started_at: chrono::DateTime<chrono::Utc>,
}

/// Atomic transaction for batch operations.
///
/// Ensures that checkpoint updates and output writes happen atomically.
/// If a crash occurs mid-transaction, the pending transaction file allows recovery.
pub struct Transaction {
    /// Transaction ID
    id: String,
    /// Directory for transaction files
    dir: PathBuf,
    /// Path to pending transaction file
    pending_path: PathBuf,
    /// Accumulated operations
    operations: Vec<PendingOperation>,
    /// Output file path (for atomic writes)
    output_path: PathBuf,
    /// Whether the transaction has been committed or aborted
    finished: bool,
}

impl Transaction {
    /// Begin a new transaction.
    ///
    /// Creates a pending transaction file that can be used for recovery.
    pub fn begin(checkpoint_dir: &Path, output_path: &Path) -> Result<Self> {
        let id = format!("tx_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f"));
        let pending_path = checkpoint_dir.join(format!("{id}.pending.json"));

        let tx = Self {
            id: id.clone(),
            dir: checkpoint_dir.to_path_buf(),
            pending_path,
            operations: Vec::new(),
            output_path: output_path.to_path_buf(),
            finished: false,
        };

        // Write initial transaction state
        tx.persist_state(false)?;

        debug!(tx_id = %id, "Transaction started");
        Ok(tx)
    }

    /// Check for and recover from any pending transactions.
    ///
    /// Returns the operations that were pending if recovery was needed.
    pub fn recover(checkpoint_dir: &Path) -> Result<Option<Vec<PendingOperation>>> {
        let pattern = checkpoint_dir.join("tx_*.pending.json");
        let pattern_str = pattern.to_string_lossy();

        // Find pending transaction files
        let pending_files: Vec<_> = glob::glob(&pattern_str)
            .map_err(|e| EpistemeError::Internal(format!("Invalid glob pattern: {e}")))?
            .filter_map(|r| r.ok())
            .collect();

        if pending_files.is_empty() {
            return Ok(None);
        }

        // Process each pending transaction (should typically be just one)
        let mut recovered_ops = Vec::new();

        for pending_path in pending_files {
            warn!(path = %pending_path.display(), "Found pending transaction, recovering");

            let content = fs::read_to_string(&pending_path)
                .map_err(|e| EpistemeError::io("reading pending transaction", e))?;

            let state: TransactionState = serde_json::from_str(&content).map_err(|e| {
                EpistemeError::ParseError(format!("Invalid transaction state: {e}"))
            })?;

            if state.committed {
                // Transaction was committed but cleanup didn't happen
                debug!(tx_id = %state.id, "Transaction was committed, cleaning up");
            } else {
                // Transaction was not committed - operations need to be retried
                warn!(
                    tx_id = %state.id,
                    ops = state.operations.len(),
                    "Transaction was not committed, marking for retry"
                );
                recovered_ops.extend(state.operations);
            }

            // Clean up the pending file
            fs::remove_file(&pending_path)
                .map_err(|e| EpistemeError::io("removing pending transaction", e))?;
        }

        if recovered_ops.is_empty() {
            Ok(None)
        } else {
            Ok(Some(recovered_ops))
        }
    }

    /// Record a generation operation.
    pub fn record_generated(&mut self, problem_id: &str, model: &str, cost: f64) -> Result<()> {
        self.operations.push(PendingOperation::Generated {
            problem_id: problem_id.to_string(),
            model: model.to_string(),
            cost,
        });
        self.persist_state(false)
    }

    /// Record a judged operation with output.
    pub fn record_judged(
        &mut self,
        problem_id: &str,
        score: f64,
        verdict: Verdict,
        judge_cost: f64,
        output_json: String,
    ) -> Result<()> {
        self.operations.push(PendingOperation::Judged {
            problem_id: problem_id.to_string(),
            score,
            verdict,
            judge_cost,
            output_json,
        });
        self.persist_state(false)
    }

    /// Record a failed operation.
    pub fn record_failed(&mut self, problem_id: &str, reason: RetryReason) -> Result<()> {
        self.operations.push(PendingOperation::Failed {
            problem_id: problem_id.to_string(),
            reason,
        });
        self.persist_state(false)
    }

    /// Commit the transaction atomically.
    ///
    /// This writes all output lines to a temp file, then atomically:
    /// 1. Appends output to the main output file
    /// 2. Marks the transaction as committed
    /// 3. Cleans up the pending file
    pub fn commit(mut self) -> Result<Vec<PendingOperation>> {
        if self.finished {
            return Err(EpistemeError::Internal(
                "Transaction already finished".to_string(),
            ));
        }

        // Collect output lines
        let output_lines: Vec<&str> = self
            .operations
            .iter()
            .filter_map(|op| {
                if let PendingOperation::Judged { output_json, .. } = op {
                    Some(output_json.as_str())
                } else {
                    None
                }
            })
            .collect();

        // Write outputs to temp file first
        if !output_lines.is_empty() {
            let temp_output = self.dir.join(format!("{}.output.tmp", self.id));
            {
                let file = File::create(&temp_output)
                    .map_err(|e| EpistemeError::io("creating temp output", e))?;
                let mut writer = BufWriter::new(file);
                for line in &output_lines {
                    writeln!(writer, "{line}")
                        .map_err(|e| EpistemeError::io("writing temp output", e))?;
                }
                writer
                    .flush()
                    .map_err(|e| EpistemeError::io("flushing temp output", e))?;
            }

            // Append temp output to main output file
            let temp_content = fs::read_to_string(&temp_output)
                .map_err(|e| EpistemeError::io("reading temp output", e))?;

            let mut output_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.output_path)
                .map_err(|e| EpistemeError::io("opening output file", e))?;

            output_file
                .write_all(temp_content.as_bytes())
                .map_err(|e| EpistemeError::io("appending to output", e))?;
            output_file
                .sync_all()
                .map_err(|e| EpistemeError::io("syncing output", e))?;

            // Remove temp file
            fs::remove_file(&temp_output)
                .map_err(|e| EpistemeError::io("removing temp output", e))?;
        }

        // Mark as committed
        self.persist_state(true)?;

        // Clean up pending file
        if self.pending_path.exists() {
            fs::remove_file(&self.pending_path)
                .map_err(|e| EpistemeError::io("removing pending transaction", e))?;
        }

        self.finished = true;
        debug!(tx_id = %self.id, ops = self.operations.len(), "Transaction committed");

        Ok(std::mem::take(&mut self.operations))
    }

    /// Abort the transaction, discarding all pending operations.
    pub fn abort(mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }

        // Clean up pending file
        if self.pending_path.exists() {
            fs::remove_file(&self.pending_path)
                .map_err(|e| EpistemeError::io("removing pending transaction", e))?;
        }

        self.finished = true;
        debug!(tx_id = %self.id, "Transaction aborted");
        Ok(())
    }

    /// Get the pending operations.
    pub fn operations(&self) -> &[PendingOperation] {
        &self.operations
    }

    /// Persist current transaction state to disk.
    fn persist_state(&self, committed: bool) -> Result<()> {
        let state = TransactionState {
            id: self.id.clone(),
            operations: self.operations.clone(),
            committed,
            started_at: chrono::Utc::now(),
        };

        let content = serde_json::to_string_pretty(&state)
            .map_err(|e| EpistemeError::Internal(format!("Serializing transaction: {e}")))?;

        fs::write(&self.pending_path, content)
            .map_err(|e| EpistemeError::io("writing pending transaction", e))?;

        Ok(())
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // If transaction was not properly committed or aborted, log a warning
        if !self.finished {
            warn!(
                tx_id = %self.id,
                ops = self.operations.len(),
                "Transaction dropped without commit/abort - will be recovered on restart"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_transaction_commit() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_dir = temp_dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).unwrap();
        let output_path = temp_dir.path().join("output.jsonl");

        let mut tx = Transaction::begin(&checkpoint_dir, &output_path).unwrap();
        tx.record_judged(
            "problem_1",
            0.8,
            Verdict::Approve,
            0.001,
            r#"{"test": 1}"#.to_string(),
        )
        .unwrap();
        tx.record_judged(
            "problem_2",
            0.6,
            Verdict::Reject,
            0.001,
            r#"{"test": 2}"#.to_string(),
        )
        .unwrap();

        let ops = tx.commit().unwrap();
        assert_eq!(ops.len(), 2);

        // Check output file
        let output = fs::read_to_string(&output_path).unwrap();
        assert!(output.contains(r#"{"test": 1}"#));
        assert!(output.contains(r#"{"test": 2}"#));

        // Check no pending files
        let pending: Vec<_> = glob::glob(&checkpoint_dir.join("*.pending.json").to_string_lossy())
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_transaction_abort() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_dir = temp_dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).unwrap();
        let output_path = temp_dir.path().join("output.jsonl");

        let mut tx = Transaction::begin(&checkpoint_dir, &output_path).unwrap();
        tx.record_judged(
            "problem_1",
            0.8,
            Verdict::Approve,
            0.001,
            r#"{"test": 1}"#.to_string(),
        )
        .unwrap();

        tx.abort().unwrap();

        // Check no output file created
        assert!(!output_path.exists());

        // Check no pending files
        let pending: Vec<_> = glob::glob(&checkpoint_dir.join("*.pending.json").to_string_lossy())
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_transaction_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_dir = temp_dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).unwrap();
        // Note: output_path not needed for recovery test, only checkpoint_dir matters
        let _output_path = temp_dir.path().join("output.jsonl");

        // Create a pending transaction file manually (simulating crash)
        let state = TransactionState {
            id: "tx_test".to_string(),
            operations: vec![PendingOperation::Generated {
                problem_id: "p1".to_string(),
                model: "gpt-4".to_string(),
                cost: 0.01,
            }],
            committed: false,
            started_at: chrono::Utc::now(),
        };
        let pending_path = checkpoint_dir.join("tx_test.pending.json");
        fs::write(&pending_path, serde_json::to_string(&state).unwrap()).unwrap();

        // Recover
        let recovered = Transaction::recover(&checkpoint_dir).unwrap();
        assert!(recovered.is_some());
        let ops = recovered.unwrap();
        assert_eq!(ops.len(), 1);

        // Pending file should be cleaned up
        assert!(!pending_path.exists());
    }
}
