# episteme

> *"Make the implicit explicit. Verify before trusting. Compound always."*

[![CI](https://github.com/infernet-org/episteme/actions/workflows/test.yml/badge.svg)](https://github.com/infernet-org/episteme/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Epistemic dataset generation for RL training.

## Supported Endpoints

| Type | Examples | Status |
|------|----------|--------|
| **Aggregators** | OpenRouter (default), Together AI, Fireworks, Groq | Supported |
| **On-prem** | vLLM, TGI, Ollama, llama.cpp server | Supported |

All endpoints must be OpenAI-compatible (chat completions API).

## Epistemic Foundation

`episteme` is built on the principle that **training data quality depends on what we know, believe, and don't know** about each sample. The tool makes these epistemic states explicit:

| State | Symbol | Meaning | In episteme |
|-------|--------|---------|-------------|
| **Knowledge** | K_i | Verified fact | Score from judge, token counts, model used |
| **Belief** | B_i | Unverified assumption | Quality flags (has_reasoning, self_correction) |
| **Ignorance** | I_i | Known unknown | Missing answer extraction, truncation detection |

Every generated sample carries epistemic metadata that enables downstream filtering, analysis, and trust calibration.

## Overview

`episteme` is a Rust CLI tool for generating training datasets using frontier LLMs. It's **RL-agnostic** - the generated data can be used for any training method:

| Method | Data Format | episteme Support |
|--------|-------------|------------------|
| **SFT** | `(prompt, completion)` | `episteme sft` |
| **DPO/IPO** | `(prompt, chosen, rejected)` | `episteme dpo` |
| **RLHF/PPO** | `(prompt, completion, reward)` | SFT output (score = reward) |
| **GRPO** | `(prompt, completions[], scores[])` | DPO with N responses |
| **KTO** | `(prompt, completion, label)` | Threshold SFT scores |

## Architecture

```
                              EPISTEMIC DATA FLOW
                              
  ┌─────────────────────────────────────────────────────────────────────┐
  │                            episteme                                  │
  │                                                                      │
  │  ┌─────────────┐                          ┌─────────────────────┐   │
  │  │   Input     │  K_i: problem exists     │      Worker         │   │
  │  │  (JSONL)    │ ─────────────────────────│       Pool          │   │
  │  │             │  I_i: output quality     │     (tokio)         │   │
  │  └─────────────┘                          └──────────┬──────────┘   │
  │                                                      │              │
  │                                    samples + B_i(quality)           │
  │                                                      ▼              │
  │                                           ┌─────────────────────┐   │
  │                                           │       Judge         │   │
  │                                           │        Pool         │   │
  │                                           │      (tokio)        │   │
  │                                           └──────────┬──────────┘   │
  │                                                      │              │
  │                                      scored + K_i(verdict)          │
  │                                                      ▼              │
  │  ┌─────────────┐                          ┌─────────────────────┐   │
  │  │   Output    │  K_i: score, tokens      │      Curator        │   │
  │  │  (JSONL)    │ ◄────────────────────────│   (filter, pair)    │   │
  │  │             │  B_i: quality_flags      └─────────────────────┘   │
  │  └─────────────┘                                                    │
  └─────────────────────────────────────────────────────────────────────┘
```

### Epistemic Properties

- **K_i (Known)**: Token counts, generation time, model identity, judge score, verdict
- **B_i (Believed)**: Quality flags inferred from output patterns (truncation, reasoning)  
- **I_i (Unknown)**: Answer correctness (requires ground truth), long-term training impact

## Key Features

| Feature | Epistemic Basis | Benefit |
|---------|-----------------|---------|
| **Multi-model routing** | K_i(model capabilities) | Weighted selection based on known strengths |
| **LLM-as-judge** | B_i → K_i transition | Convert quality assumptions to verified scores |
| **Quality flags** | B_i(output analysis) | Surface patterns: truncation, reasoning, self-correction |
| **Checkpointing** | K_i(progress state) | Resume with known-good state |
| **Rate limiting** | K_i(API limits) | Adaptive backoff from observed 429s |
| **Cost tracking** | K_i(token usage) | Real-time cost visibility |
| **Ensemble judging** | B_i → K_i(HIGH) | Multi-judge consensus for higher confidence |
| **Multi-endpoint** | K_i(endpoint health) | Support for on-prem and multiple aggregators |

## Multi-Endpoint Support

episteme supports multiple LLM endpoints simultaneously:

```toml
# OpenRouter (primary, always available)
[openrouter]
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"

# On-prem endpoints (optional)
[endpoints.ollama]
base_url = "http://localhost:11434/v1"

[endpoints.vllm]
base_url = "http://gpu-server:8000/v1"
headers = { "X-Episteme-Auth" = "${VLLM_API_KEY}" }

# Other aggregators (optional)
[endpoints.together]
base_url = "https://api.together.xyz/v1"
api_key_env = "TOGETHER_API_KEY"

# Models reference their endpoint (default: "openrouter")
[workers]
models = [
    { id = "deepseek/deepseek-r1" },                    # OpenRouter (default)
    { endpoint = "ollama", id = "llama3.3:70b" },       # Local Ollama
    { endpoint = "vllm", id = "meta-llama/Llama-3-70B" }, # On-prem vLLM
]
```

### Health Checks

Before running pipelines, episteme verifies all endpoints are reachable:

```bash
# Check all endpoints
episteme health -c config.toml

# Skip health checks (for CI or offline testing)
episteme sft --skip-health-check -c config.toml -p problems.jsonl -o out.jsonl
```

### Environment Variable Expansion

Custom headers support `${VAR}` syntax for secrets:

```toml
[endpoints.vllm]
base_url = "http://gpu-server:8000/v1"
headers = { "X-Episteme-Auth" = "${VLLM_SECRET}" }  # Expanded at runtime
```

## Ensemble Judging

Single LLM judges are a known weakness - they can have blind spots, biases, or inconsistent scoring. Ensemble judging addresses this by using **multiple diverse judges** and aggregating their scores:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE JUDGING FLOW                        │
│                                                                 │
│  Sample ──► Judge 1 (GPT-4o)     ──► 0.90 ─┐                   │
│         ──► Judge 2 (Claude)     ──► 0.88 ─┼─► Median: 0.88    │
│         ──► Judge 3 (Gemini)     ──► 0.85 ─┘    σ: 0.025       │
│                                              Confidence: HIGH   │
└─────────────────────────────────────────────────────────────────┘
```

### Epistemic Benefits

| Single Judge | Ensemble |
|--------------|----------|
| B_i(score) - one model's opinion | K_i(score) - consensus from diverse models |
| Hidden bias | Bias exposed via disagreement |
| No confidence metric | Explicit confidence from variance |
| All-or-nothing | Hierarchical: cheap first, escalate if uncertain |

### Configuration

```toml
[judges]
size = 5
models = [
    { id = "openai/gpt-4o", label = "GPT-4o" },
    { id = "anthropic/claude-sonnet-4", label = "Sonnet" },
    { id = "google/gemini-2.0-flash-001", label = "Gemini" },
]

[judges.ensemble]
enabled = true              # Enable multi-judge evaluation
num_judges = 3              # Number of judges per sample
strategy = "median"         # median | average | weightedaverage
disagreement_threshold = 0.15  # std_dev >= this = low confidence
hierarchical = false        # Cheap judge first, ensemble if uncertain
uncertain_range = [0.4, 0.7]   # Score range triggering full ensemble
```

### Aggregation Strategies

| Strategy | Formula | Best For |
|----------|---------|----------|
| `median` | Middle value | Default, robust to outliers |
| `average` | Mean of scores | When all judges equally trusted |
| `weightedaverage` | Weighted by model weights | When some judges more reliable |

### Confidence Levels

| Level | Criteria | Interpretation |
|-------|----------|----------------|
| `high` | σ < threshold AND unanimous verdict | Strong consensus, trust score |
| `medium` | σ < threshold OR unanimous verdict | Moderate agreement |
| `low` | σ >= threshold AND split verdict | Judges disagree, review manually |

### Hierarchical Mode

For cost optimization, enable `hierarchical = true`:

1. **First**: Cheap/fast judge evaluates sample
2. **If score in `uncertain_range`**: Full ensemble evaluation
3. **Otherwise**: Accept cheap judge's verdict

This reduces costs while maintaining quality for edge cases.

## Installation

```bash
# From source
cargo install --path .

# Or build directly
cargo build --release
./target/release/episteme --help
```

## Quick Start

```bash
# Set your API key (K_i: authentication configured)
export OPENROUTER_API_KEY="sk-or-..."

# Generate SFT data
episteme sft \
  --config config/example.toml \
  --problems examples/problems.jsonl \
  --output sft_dataset.jsonl

# Generate DPO preference pairs
episteme dpo \
  --config config/example.toml \
  --problems examples/problems.jsonl \
  --output dpo_dataset.jsonl \
  --responses 3
```

## Configuration

```toml
# K_i: API configuration (verified on first request)
[openrouter]
timeout_secs = 180
max_retries = 3

# K_i: Worker configuration
[workers]
size = 10  # Concurrent generation workers
models = [
    { id = "deepseek/deepseek-r1", weight = 2 },
    { id = "anthropic/claude-sonnet-4", weight = 1 },
]

# K_i: Judge configuration  
[judges]
size = 5  # Concurrent judge workers
models = [
    { id = "openai/gpt-4o", temperature = 0.3 },
    { id = "anthropic/claude-sonnet-4", temperature = 0.3 },
    { id = "google/gemini-2.0-flash-001", temperature = 0.3 },
]

# Ensemble judging: B_i(single) → K_i(consensus)
[judges.ensemble]
enabled = true
num_judges = 3
strategy = "median"
disagreement_threshold = 0.15

# B_i: Generation parameters (tune based on observed quality)
[generation]
system_prompt = "prompts/system.md"
judge_prompt = "prompts/judge.md"
approval_threshold = 0.85    # Threshold for B_i → K_i(approved)
responses_per_problem = 3    # For DPO: responses to compare

[output]
path = "output/dataset.jsonl"
track_costs = true
```

See `config/example.toml` for a complete example.

## Input Format

Problems JSONL:
```jsonl
{"id": "prob_001", "input": "What is 2 + 2?"}
{"id": "prob_002", "input": "Explain the Pythagorean theorem."}
{"id": "prob_003", "input": "Write a function to reverse a string."}
```

## Output Schema

### SFT Output (Epistemic Sample)

Each sample carries full epistemic metadata:

```jsonl
{
  "id": "ae5f047c-3b8a-4984-8361-4eecd9b3d89d",
  "input": "What is 2 + 2?",
  "output": "The sum of 2 + 2 is 4...",
  "answer": "4",
  "model": "deepseek/deepseek-r1",
  "score": 0.88,
  "problem_id": "prob_001",
  "tokens_in": 1265,
  "tokens_out": 1572,
  "judge_reasoning": "The reasoning is thorough...",
  "generation_time_ms": 5336,
  "judge_model": "openai/gpt-4o,anthropic/claude-sonnet-4,google/gemini-2.0-flash-001",
  "verdict": "approve",
  "judge_confidence": "high",
  "score_std_dev": 0.025,
  "num_judges": 3,
  "individual_scores": [0.90, 0.88, 0.85],
  "quality_flags": {
    "truncated": false,
    "has_answer_tags": true,
    "has_reasoning": true,
    "self_correction": false,
    "reasoning_length": 1125
  },
  "cost_usd": 0.0024
}
```

### Field Reference (K_i: Verified Facts)

| Field | Type | Epistemic State | Description |
|-------|------|-----------------|-------------|
| `id` | string | K_i | Unique sample identifier |
| `input` | string | K_i | Original problem/prompt |
| `output` | string | K_i | Generated response |
| `model` | string | K_i | Model used for generation |
| `score` | float | K_i | Quality score from judge (0.0-1.0) |
| `problem_id` | string | K_i | Source problem ID for tracing |
| `tokens_in` | int | K_i | Input tokens consumed |
| `tokens_out` | int | K_i | Output tokens generated |
| `generation_time_ms` | int | K_i | Generation latency |
| `judge_model` | string | K_i | Model(s) used for judging (comma-separated if ensemble) |
| `verdict` | string | K_i | `"approve"` or `"reject"` |
| `judge_confidence` | string? | K_i | Ensemble confidence: `"high"`, `"medium"`, `"low"` |
| `score_std_dev` | float? | K_i | Standard deviation of judge scores (ensemble) |
| `num_judges` | int? | K_i | Number of judges used (ensemble) |
| `individual_scores` | float[]? | K_i | Individual judge scores (ensemble) |
| `cost_usd` | float? | K_i | Total cost (generation + judging) |

### Field Reference (B_i: Inferred Beliefs)

| Field | Type | Confidence | Description |
|-------|------|------------|-------------|
| `answer` | string? | MED | Extracted final answer (pattern-based) |
| `judge_reasoning` | string? | HIGH | Judge's explanation (LLM output) |
| `quality_flags` | object | MED | Heuristic quality signals |

### Quality Flags (B_i: Pattern-Based Inference)

| Flag | Type | Detection Method | Confidence |
|------|------|------------------|------------|
| `truncated` | bool | Last char not terminal punctuation | MED |
| `has_answer_tags` | bool | Contains `<answer>`, `\boxed{}`, etc. | HIGH |
| `has_reasoning` | bool | Contains "Step 1", "Let's think", etc. | HIGH |
| `self_correction` | bool | Contains "Wait", "Actually", etc. | HIGH |
| `reasoning_length` | int | Chars before answer tag | K_i |

### DPO Output

```jsonl
{
  "id": "dpo_001",
  "problem_id": "prob_001",
  "input": "What is 2 + 2?",
  "chosen": "The answer is 4 because...",
  "rejected": "2 + 2 = 5...",
  "chosen_score": 0.95,
  "rejected_score": 0.45,
  "chosen_model": "deepseek/deepseek-r1",
  "rejected_model": "anthropic/claude-sonnet-4"
}
```

## Epistemic Filtering

Use metadata to filter based on epistemic confidence:

```python
import json

def load_high_confidence_samples(path, min_score=0.9):
    """
    Load samples where:
    - K_i(score >= threshold) - verified quality
    - B_i(not truncated) - believed complete
    - B_i(has_reasoning) - believed to show work
    """
    samples = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            # K_i: score is verified by judge
            if sample["score"] < min_score:
                continue
            # B_i: quality flags are inferred
            flags = sample["quality_flags"]
            if flags["truncated"]:
                continue  # I_i: unknown how output would end
            if not flags["has_reasoning"]:
                continue  # B_i: may lack chain-of-thought
            samples.append(sample)
    return samples

def partition_by_confidence(path):
    """
    Partition samples by epistemic confidence.
    
    Returns:
        high_k: K_i(high score) AND B_i(good flags)
        med_b:  K_i(ok score) OR B_i(some flags missing)  
        low_i:  I_i(truncated) OR K_i(low score)
    """
    high_k, med_b, low_i = [], [], []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            flags = s["quality_flags"]
            
            if flags["truncated"]:
                low_i.append(s)  # I_i: incomplete
            elif s["score"] >= 0.9 and flags["has_reasoning"]:
                high_k.append(s)  # High confidence
            elif s["score"] >= 0.7:
                med_b.append(s)  # Medium confidence
            else:
                low_i.append(s)  # Low confidence
    return high_k, med_b, low_i
```

## Integration Examples

### TRL (Transformers RL)

```python
from datasets import load_dataset
from trl import DPOTrainer

# K_i: episteme output format matches TRL expectations
dataset = load_dataset("json", data_files="dpo_dataset.jsonl")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dataset["train"],
)
```

### Axolotl

```yaml
# B_i: field mapping should work (verify with your version)
datasets:
  - path: sft_dataset.jsonl
    type: completion
    field_map:
      prompt: input
      completion: output
```

### Cost Analysis

```python
def analyze_costs(path):
    """
    K_i: Analyze verified costs by model.
    """
    costs_by_model = {}
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            model = sample["model"]  # K_i: known
            cost = sample.get("cost_usd", 0)  # K_i: tracked
            if model not in costs_by_model:
                costs_by_model[model] = {"count": 0, "total_cost": 0}
            costs_by_model[model]["count"] += 1
            costs_by_model[model]["total_cost"] += cost
    return costs_by_model
```

## Epistemic Horizon

Some things are **knowable** (I^R) and some are **bounded** (I^B):

| Aspect | Status | Action |
|--------|--------|--------|
| Token count | K_i | Tracked precisely |
| Generation time | K_i | Measured |
| Judge score | K_i | Verified by judge model |
| Answer correctness | I^R | Resolvable with ground truth |
| Truncation | B_i | Inferred from patterns |
| Training impact | I^B | Bounded (unknowable until trained) |
| Future model behavior | I^B | Bounded (fundamentally uncertain) |

**Design principle**: episteme maximizes K_i and B_i while making I_i explicit. It does not claim to know the unknowable.

## Project Structure

```
episteme/
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Library exports
│   ├── client/           # OpenRouter client + rate limiter
│   │   └── rate_limiter.rs  # K_i: Per-model rate limit tracking
│   ├── models/           # Config, sample, error types
│   │   ├── sample.rs     # Epistemic sample schema
│   │   └── error.rs      # EpistemeError taxonomy
│   ├── pipeline/         # DPO + SFT generation pipelines
│   ├── pool/             # Worker + judge pools
│   │   └── judge.rs      # B_i → K_i: Score verification
│   └── checkpoint/       # Resume support (K_i persistence)
├── config/
│   └── example.toml      # Example configuration
├── prompts/
│   └── examples/         # Example prompts
├── test/
│   ├── config.toml       # Test configuration
│   └── problems.jsonl    # Test problems
└── Cargo.toml
```

## Environment Variables

| Variable | Purpose | Epistemic Note |
|----------|---------|----------------|
| `OPENROUTER_API_KEY` | API authentication | K_i after first successful request |
| `RUST_LOG` | Log verbosity | Set to `debug` for full K_i visibility |

## Related Projects

- **[EAE](https://github.com/infernet-org/eae)** - Epistemic Agentic Engineering framework (methodology source)
- **[eae-skills](https://github.com/infernet-org/eae-skills)** - OpenCode skills for epistemic reasoning

## License

Apache-2.0

---

**K_i: This tool generates training data. B_i: The data is high quality. I_i: Training outcomes depend on your model and method.**
