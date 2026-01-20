//! episteme CLI - Epistemic dataset generation for SFT/DPO training.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use episteme::{CheckpointManager, Config, DpoPipeline, OpenRouterClient, SftPipeline};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "episteme")]
#[command(author = "Infernet <dev@infernet.org>")]
#[command(version)]
#[command(about = "Epistemic dataset generation for SFT/DPO training via OpenRouter")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to configuration file
    #[arg(short, long, global = true, default_value = "config.toml")]
    config: PathBuf,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate SFT (Supervised Fine-Tuning) dataset
    Sft {
        /// Path to input problems JSONL file
        #[arg(short, long)]
        problems: PathBuf,

        /// Path to output JSONL file
        #[arg(short, long)]
        output: PathBuf,

        /// Checkpoint directory for resumable runs
        #[arg(long)]
        checkpoint: Option<PathBuf>,
    },

    /// Generate DPO (Direct Preference Optimization) dataset
    Dpo {
        /// Path to input problems JSONL file
        #[arg(short, long)]
        problems: PathBuf,

        /// Path to output JSONL file
        #[arg(short, long)]
        output: PathBuf,

        /// Number of responses to generate per problem
        #[arg(short, long, default_value = "3")]
        responses: usize,

        /// Checkpoint directory for resumable runs
        #[arg(long)]
        checkpoint: Option<PathBuf>,
    },

    /// Validate configuration file
    Validate,

    /// Show example configuration
    Example,
}

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
}

fn print_example_config() {
    let example = r#"# episteme configuration file

[openrouter]
# API key (can also use OPENROUTER_API_KEY env var)
# api_key = "sk-..."
base_url = "https://openrouter.ai/api/v1"
timeout_secs = 180
max_retries = 3

[workers]
size = 10
models = [
    { id = "deepseek/deepseek-r1", weight = 2, input_price_per_1m = 0.70, output_price_per_1m = 2.50 },
    { id = "anthropic/claude-sonnet-4", weight = 1, input_price_per_1m = 3.0, output_price_per_1m = 15.0 },
]

[judges]
size = 5
models = [
    { id = "openai/gpt-4o", weight = 1, input_price_per_1m = 2.5, output_price_per_1m = 10.0, temperature = 0.3 },
]

[generation]
system_prompt = "prompts/system.md"
judge_prompt = "prompts/judge.md"
approval_threshold = 0.85
responses_per_problem = 3  # For DPO

[output]
path = "output/dataset.jsonl"
# checkpoint_dir = "checkpoints/"
include_rejected = false
track_costs = true
"#;
    println!("{example}");
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    match cli.command {
        Commands::Example => {
            print_example_config();
            return Ok(());
        }

        Commands::Validate => {
            let config = Config::from_file(&cli.config)
                .with_context(|| format!("Failed to load config from {:?}", cli.config))?;

            // Try to resolve API key
            config
                .resolve_api_key()
                .context("Failed to resolve API key")?;

            info!("Configuration is valid");
            info!(
                "  Workers: {} with {} models",
                config.workers.size,
                config.workers.models.len()
            );
            info!(
                "  Judges: {} with {} models",
                config.judges.size,
                config.judges.models.len()
            );
            info!(
                "  Approval threshold: {:.0}%",
                config.generation.approval_threshold * 100.0
            );
            return Ok(());
        }

        Commands::Sft {
            problems,
            output,
            checkpoint,
        } => {
            let config = Config::from_file(&cli.config)
                .with_context(|| format!("Failed to load config from {:?}", cli.config))?;

            let api_key = config
                .resolve_api_key()
                .context("Failed to resolve API key")?;

            let client = Arc::new(OpenRouterClient::new(
                api_key,
                Some(config.openrouter.base_url.clone()),
                Some(config.openrouter.timeout_secs),
                Some(config.openrouter.max_retries),
                None,
            )?);

            let pipeline = SftPipeline::new(config, client)?;
            let problems_data = SftPipeline::load_problems(&problems)?;

            // Setup checkpoint manager if requested
            let checkpoint_mgr = if let Some(checkpoint_dir) = checkpoint {
                let mut mgr = CheckpointManager::new(&checkpoint_dir)
                    .context("Failed to create checkpoint manager")?;
                mgr.init_or_load("sft", &problems_data)
                    .context("Failed to initialize checkpoint")?;
                Some(mgr)
            } else {
                None
            };

            let stats = pipeline
                .run_with_checkpoint(problems_data, &output, checkpoint_mgr)
                .await?;

            println!("\n=== SFT Generation Complete ===");
            println!("Problems:    {}", stats.total_problems);
            println!("Generated:   {}", stats.total_generated);
            println!("Approved:    {}", stats.total_approved);
            println!("Rejected:    {}", stats.total_rejected);
            println!("Approval:    {:.1}%", stats.approval_rate * 100.0);
            println!("Throughput:  {:.0}/hr", stats.throughput_per_hour);
            println!("Gen cost:    ${:.4}", stats.generation_cost_usd);
            println!("Judge cost:  ${:.4}", stats.judge_cost_usd);
            println!(
                "Total cost:  ${:.4}",
                stats.generation_cost_usd + stats.judge_cost_usd
            );
            println!("Runtime:     {:.1}s", stats.runtime_secs);
            println!("Output:      {output:?}");
        }

        Commands::Dpo {
            problems,
            output,
            responses,
            checkpoint,
        } => {
            let mut config = Config::from_file(&cli.config)
                .with_context(|| format!("Failed to load config from {:?}", cli.config))?;

            // Override responses_per_problem from CLI
            config.generation.responses_per_problem = responses;

            let api_key = config
                .resolve_api_key()
                .context("Failed to resolve API key")?;

            let client = Arc::new(OpenRouterClient::new(
                api_key,
                Some(config.openrouter.base_url.clone()),
                Some(config.openrouter.timeout_secs),
                Some(config.openrouter.max_retries),
                None,
            )?);

            let pipeline = DpoPipeline::new(config, client)?;
            let problems_data = DpoPipeline::load_problems(&problems)?;

            // Setup checkpoint manager if requested
            let checkpoint_mgr = if let Some(checkpoint_dir) = checkpoint {
                let mut mgr = CheckpointManager::new(&checkpoint_dir)
                    .context("Failed to create checkpoint manager")?;
                mgr.init_or_load("dpo", &problems_data)
                    .context("Failed to initialize checkpoint")?;
                Some(mgr)
            } else {
                None
            };

            let stats = pipeline
                .run_with_checkpoint(problems_data, &output, checkpoint_mgr)
                .await?;

            println!("\n=== DPO Generation Complete ===");
            println!("Problems:    {}", stats.total_problems);
            println!("Generated:   {}", stats.total_generated);
            println!("Judged:      {}", stats.total_judged);
            println!("Pairs:       {}", stats.total_approved);
            println!("Skipped:     {}", stats.total_rejected);
            println!("Throughput:  {:.0} pairs/hr", stats.throughput_per_hour);
            println!("Gen cost:    ${:.4}", stats.generation_cost_usd);
            println!("Judge cost:  ${:.4}", stats.judge_cost_usd);
            println!(
                "Total cost:  ${:.4}",
                stats.generation_cost_usd + stats.judge_cost_usd
            );
            println!("Runtime:     {:.1}s", stats.runtime_secs);
            println!("Output:      {output:?}");
        }
    }

    Ok(())
}
