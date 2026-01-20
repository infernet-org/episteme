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

```jsonl
{
  "id": "sft_001",
  "input": "What is 2 + 2?",
  "output": "The sum of 2 + 2 is 4...",
  "model": "deepseek/deepseek-r1",
  "score": 0.92
}
```

**Use for**: SFT training, RLHF (score as reward), KTO (threshold score)

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
  "margin": 0.50
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
