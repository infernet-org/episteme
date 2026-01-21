//! Checkpoint module for resumable pipeline execution.
//!
//! Provides:
//! - `CheckpointState`: State tracking for problems in a pipeline
//! - `CheckpointManager`: Persistence and loading of checkpoint state
//! - `Transaction`: Atomic batch operations with crash recovery

mod state;
mod transaction;

pub use state::*;
pub use transaction::*;
