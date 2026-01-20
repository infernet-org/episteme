# dpogen

High-performance synthetic dataset generation for RL training via OpenRouter.

## Overview

`dpogen` is a Rust CLI tool for generating training datasets using frontier LLMs. It's **RL-agnostic** — the generated data can be used for:

| Method | Data Format | dpogen Support |
|--------|-------------|----------------|
| **SFT** | `(prompt, completion)` | ✅ `dpogen sft` |
| **DPO/IPO** | `(prompt, chosen, rejected)` | ✅ `dpogen dpo` |
| **RLHF/PPO** | `(prompt, completion, reward)` | ✅ Use SFT output (score = reward) |
| **GRPO** | `(prompt, completions[], scores[])` | ✅ Use DPO with N responses |
| **KTO** | `(prompt, completion, label)` | ✅ Threshold SFT scores |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         dpogen                                      │
│                                                                     │
│  ┌─────────────┐      problems       ┌─────────────┐                │
│  │   Input     │────────────────────►│   Worker    │                │
│  │  (JSONL)    │                     │   Pool      │                │
│  └─────────────┘                     │  (tokio)    │                │
│                                      └──────┬──────┘                │
│                                             │ samples               │
│                                             ▼                       │
│                                      ┌─────────────┐                │
│                                      │   Judge     │                │
│                                      │   Pool      │                │
│                                      │  (tokio)    │                │
│                                      └──────┬──────┘                │
│                                             │ scored                │
│                                             ▼                       │
│  ┌─────────────┐                     ┌─────────────┐                │
│  │   Output    │◄────────────────────│  Curator    │                │
│  │  (JSONL)    │      SFT/DPO        │  (filter,   │                │
│  └─────────────┘                     │   pair)     │                │
│                                      └─────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

**Key features:**
- **Multi-model routing**: Use different models for generation (weighted selection)
- **LLM-as-judge**: Automatic quality scoring with configurable judge models
- **Checkpointing**: Resume interrupted runs
- **Rate limiting**: Per-model rate limiting with backoff
- **Cost tracking**: Monitor API costs in real-time

## Installation

```bash
# From source
cargo install --path .

# Or build directly
cargo build --release
./target/release/dpogen --help
```

## Quick Start

```bash
# Set your API key
export OPENROUTER_API_KEY="sk-or-..."

# Generate SFT data
dpogen sft \
  --config config/example.toml \
  --problems examples/problems.jsonl \
  --output sft_dataset.jsonl

# Generate DPO preference pairs
dpogen dpo \
  --config config/example.toml \
  --problems examples/problems.jsonl \
  --output dpo_dataset.jsonl \
  --responses 3
```

## Usage

### SFT Generation

Generate high-quality completions with judge filtering:

```bash
dpogen sft \
  --config config.toml \
  --problems problems.jsonl \
  --output sft_dataset.jsonl
```

### DPO Generation

Generate preference pairs (chosen vs rejected):

```bash
dpogen dpo \
  --config config.toml \
  --problems problems.jsonl \
  --output dpo_dataset.jsonl \
  --responses 3  # Generate 3 responses per problem, pair best vs worst
```

### Other Commands

```bash
# Validate configuration
dpogen --config config.toml validate

# Show example configuration
dpogen example
```

## Configuration

See `config/example.toml` for a complete example.

```toml
[openrouter]
# API key (or set OPENROUTER_API_KEY env var)
timeout_secs = 180
max_retries = 3

[workers]
size = 10  # Concurrent generation workers
models = [
    { id = "deepseek/deepseek-r1", weight = 2 },
    { id = "anthropic/claude-sonnet-4", weight = 1 },
]

[judges]
size = 5  # Concurrent judge workers
models = [
    { id = "openai/gpt-4o", temperature = 0.3 },
]

[generation]
system_prompt = "prompts/system.md"    # Your system prompt
judge_prompt = "prompts/judge.md"      # Your judge prompt
approval_threshold = 0.85               # Minimum score to approve
responses_per_problem = 3               # For DPO: responses to compare

[output]
path = "output/dataset.jsonl"
track_costs = true
```

### Prompts

You provide your own prompts — dpogen doesn't assume any specific format. Example prompts are included in `prompts/examples/`:

- `system-reasoning.md` - Chain-of-thought reasoning prompt
- `judge-correctness.md` - Quality scoring prompt

Customize these for your use case (math, code, reasoning, etc.).

## Input Format

Problems JSONL:
```jsonl
{"id": "prob_001", "input": "What is 2 + 2?"}
{"id": "prob_002", "input": "Explain the Pythagorean theorem."}
{"id": "prob_003", "input": "Write a function to reverse a string."}
```

See `examples/problems.jsonl` for more examples across domains.

## Output Formats

### SFT Output

Each sample includes full metadata for traceability, quality analysis, and debugging:

```jsonl
{
  "id": "ae5f047c-3b8a-4984-8361-4eecd9b3d89d",
  "input": "What is 2 + 2?",
  "output": "The sum of 2 + 2 is 4...",
  "answer": "4",
  "model": "deepseek/deepseek-r1",
  "score": 0.92,
  "problem_id": "prob_001",
  "tokens_in": 1265,
  "tokens_out": 1572,
  "judge_reasoning": "The reasoning is thorough...",
  "generation_time_ms": 5336,
  "judge_model": "openai/gpt-4o",
  "verdict": "approve",
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

#### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `input` | string | Original problem/prompt |
| `output` | string | Generated response |
| `answer` | string? | Extracted final answer (if parseable) |
| `model` | string | Model used for generation |
| `score` | float | Quality score from judge (0.0-1.0) |
| `problem_id` | string | Source problem ID for tracing |
| `tokens_in` | int | Input tokens consumed |
| `tokens_out` | int | Output tokens generated |
| `judge_reasoning` | string? | Judge's explanation for the score |
| `generation_time_ms` | int | Generation latency in milliseconds |
| `judge_model` | string | Model used for judging |
| `verdict` | string | `"approve"` or `"reject"` |
| `quality_flags` | object | Epistemic quality signals (see below) |
| `cost_usd` | float? | Total cost (generation + judging) |

#### Quality Flags

| Flag | Type | Description |
|------|------|-------------|
| `truncated` | bool | Output appears cut off mid-sentence |
| `has_answer_tags` | bool | Contains `<answer>`, `\boxed{}`, etc. |
| `has_reasoning` | bool | Contains reasoning steps (EAE phases, "Step 1", etc.) |
| `self_correction` | bool | Contains "Wait", "Actually", self-correction patterns |
| `reasoning_length` | int | Character count of reasoning content |

**Use for**: SFT training, RLHF (score as reward), KTO (threshold score), quality filtering

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

**Use for**: DPO, IPO, any preference-based method

## Examples

### Using with TRL (Transformers RL)

```python
from datasets import load_dataset
from trl import DPOTrainer

# Load dpogen output
dataset = load_dataset("json", data_files="dpo_dataset.jsonl")

# Train with TRL
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dataset["train"],
    # dpogen format matches TRL expectations
)
```

### Using with Axolotl

```yaml
# axolotl config
datasets:
  - path: sft_dataset.jsonl
    type: completion
    field_map:
      prompt: input
      completion: output
```

### Filtering with Quality Flags

Use the metadata to filter high-quality samples:

```python
import json

def load_high_quality_samples(path, min_score=0.9):
    """Load only high-quality, non-truncated samples."""
    samples = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            # Filter by score and quality flags
            if (sample["score"] >= min_score 
                and not sample["quality_flags"]["truncated"]
                and sample["quality_flags"]["has_reasoning"]):
                samples.append(sample)
    return samples

# Filter for samples with self-correction (shows deeper reasoning)
def get_self_correcting_samples(path):
    samples = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            if sample["quality_flags"]["self_correction"]:
                samples.append(sample)
    return samples
```

### Cost Analysis

```python
import json

def analyze_costs(path):
    """Analyze generation costs by model."""
    costs_by_model = {}
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            model = sample["model"]
            cost = sample.get("cost_usd", 0)
            if model not in costs_by_model:
                costs_by_model[model] = {"count": 0, "total_cost": 0}
            costs_by_model[model]["count"] += 1
            costs_by_model[model]["total_cost"] += cost
    return costs_by_model
```

## Environment Variables

- `OPENROUTER_API_KEY`: API key for OpenRouter (required if not in config)
- `RUST_LOG`: Set log level (e.g., `RUST_LOG=debug`)

## Project Structure

```
dpogen/
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Library exports
│   ├── client/           # OpenRouter client + rate limiter
│   ├── models/           # Config, sample, error types
│   ├── pipeline/         # DPO + SFT generation pipelines
│   ├── pool/             # Worker + judge pools
│   └── checkpoint/       # Resume support
├── config/
│   └── example.toml      # Example configuration
├── prompts/
│   └── examples/         # Example prompts
├── examples/
│   └── problems.jsonl    # Sample problems
└── Cargo.toml
```

## License

Apache-2.0
