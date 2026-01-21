//! LLM client module.
//!
//! Provides generic OpenAI-compatible API client that works with:
//! - Aggregators: OpenRouter, Together AI, Fireworks, Groq
//! - On-prem: vLLM, TGI, Ollama, llama.cpp server

mod llm_client;
mod rate_limiter;
mod registry;

pub use llm_client::*;
pub use rate_limiter::*;
pub use registry::*;
