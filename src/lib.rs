//! episteme - Epistemic dataset generation for SFT/DPO training.
//!
//! ## Architecture
//!
//! episteme uses two logical pools:
//! - **Worker Pool**: Generates samples by calling LLM endpoints
//! - **Judge Pool**: Evaluates samples using LLM-based criteria
//!
//! ## Supported Endpoints
//!
//! - **Aggregators**: OpenRouter (default), Together AI, Fireworks, Groq
//! - **On-prem**: vLLM, TGI, Ollama, llama.cpp server
//!
//! All endpoints must be OpenAI-compatible (chat completions API).
//!
//! ## Pipelines
//!
//! - **SFT Pipeline**: Generate → Judge → Output approved samples
//! - **DPO Pipeline**: Generate N per problem → Judge all → Pair best/worst → Output pairs
//!
//! ## Epistemic Design
//!
//! - K_i (Knowledge): Compile-time enforced invariants (types, enums)
//! - B_i (Beliefs): Runtime fallible operations (Result, Option)
//! - I^R (Resolvable): User-configurable parameters
//! - I^B (Bounded): Network/API uncertainties (retry, backoff)

pub mod checkpoint;
pub mod client;
pub mod models;
pub mod pipeline;
pub mod pool;

// Re-exports for convenience
pub use checkpoint::{CheckpointManager, CheckpointState};
pub use client::{
    EndpointRegistry, HealthCheckResult, HealthStatus, LLMClient, OpenRouterClient, RateLimiter,
};
pub use models::{Config, EpistemeError, Problem, Result, Sample, Verdict};
pub use pipeline::{DpoPipeline, SftPipeline};
pub use pool::{JudgePool, WorkerPool};
