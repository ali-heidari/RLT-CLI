use clap::{Parser, Subcommand, ValueEnum};
use std::fs;
use std::sync::Arc;
use tokio;

#[derive(Parser, Debug)]
#[command(author, version, about = "AIXKER-RLT CLI for training and exporting reinforcement learning models.", long_about = None)]
struct Cli {
    /// Set the global logging level
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,

    /// Path to a configuration file
    #[arg(long, value_name = "FILE")]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start training a model with customizable hyperparameters
    Train(TrainArgs),
    /// Export a trained model to a supported format
    Export(ExportArgs),
}

#[derive(clap::Args, Debug)]
struct TrainArgs {
    /// Path to the training dataset or environment configuration
    #[arg(long, value_name = "PATH", default_value = "./data/train")]
    dataset: String,

    /// Number of training epochs
    #[arg(long, default_value_t = 10)]
    epochs: u32,

    /// Batch size for training
    #[arg(long, default_value_t = 32)]
    batch_size: u32,

    /// Initial learning rate
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f32,

    /// Enable dry-run mode without actually executing training
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Name of the model checkpoint file to use or write
    #[arg(long, default_value = "model.json", value_name = "NAME")]
    model_name: String,
}

#[derive(clap::Args, Debug)]
struct ExportArgs {
    /// Path to the trained model checkpoint
    #[arg(long, value_name = "PATH", default_value = "./checkpoints/latest.json")]
    checkpoint: String,

    /// Output path for the exported model
    #[arg(long, value_name = "PATH", default_value = "./exported/model.json")]
    output: String,

    /// Export format for the model
    #[arg(long, value_enum, default_value_t = ExportFormat::Json)]
    format: ExportFormat,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ExportFormat {
    Json,
    // Add other formats as needed
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    println!("AIXKER-RLT CLI invoked with log level: {:?}", cli.log_level);
    if let Some(config_path) = &cli.config {
        println!("Using config file: {}", config_path);
    }

    match cli.command {
        Commands::Train(args) => {
            println!("Starting training with the following settings:");
            println!("  dataset: {}", args.dataset);
            println!("  epochs: {}", args.epochs);
            println!("  batch size: {}", args.batch_size);
            println!("  learning rate: {}", args.learning_rate);
            println!("  model name: {}", args.model_name);
            println!("  dry run: {}", args.dry_run);

            if args.dry_run {
                println!("Dry run enabled. No training will be performed.");
            } else {
                // Load config or use defaults
let model_name = args.model_name.clone();
                let config = if let Some(path) = &cli.config {
                    let config_str = fs::read_to_string(path)?;
                    let mut config: aion_rlt::configurations::Configurations = serde_json::from_str(&config_str)?;
                    config.model_name = model_name.clone();
                    Arc::new(config)
                } else {
                    // Default config
                    Arc::new(aion_rlt::configurations::Configurations {
                        interval_secs: 10, // u64
                        batch_size: args.batch_size,
                        total_batches: args.epochs as usize * 100, // Approximate
                        input_number: 8,
                        output_number: 3,
                        hidden_layers: 16,
                        reply_capacity: 4096,
                        model_name,
                        log_interval: 64,
                        mode: aion_rlt::RunningMode::Training,
                    })
                };

                aion_rlt::initialize(config.clone());

                // Start training node with dummy closures
                aion_rlt::node::Node::start(
                    |lowest_state| vec![0.0; 8], // dummy input
                    |features, action, reward| (0.0, true), // dummy reward
                    aion_rlt::RunningMode::Training,
                    &config.model_name,
                ).await;

                // In a real implementation, you'd run the training loop here
                // For now, just indicate training started
                println!("Training node started. Training logic would run here.");
            }
        }
        Commands::Export(args) => {
            println!("Exporting model with the following settings:");
            println!("  checkpoint: {}", args.checkpoint);
            println!("  output: {}", args.output);
            println!("  format: {:?}", args.format);

            // Note: Model export not implemented in public API yet
            // For now, just copy the checkpoint file
            fs::copy(&args.checkpoint, &args.output)?;
            println!("Model exported to {}", args.output);
        }
    }

    Ok(())
}
