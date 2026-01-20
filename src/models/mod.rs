//! Core data models for dpogen.
//!
//! Epistemic mapping:
//! - K_i (Knowledge): Concrete types with compile-time guarantees
//! - B_i (Beliefs): Wrapped in Result/Option
//! - I^R (Resolvable): Generics and config parameters
//! - I^B (Bounded): Error variants with fallback strategies

mod config;
mod sample;
mod error;

pub use config::*;
pub use sample::*;
pub use error::*;
