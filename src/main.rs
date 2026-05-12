mod cli;

use clap::Parser;
use cli::{Cli, Commands};
use std::fs;
use std::sync::Arc;
use tokio;
use toml;

fn cli_option_present(name: &str) -> bool {
    std::env::args().any(|arg| arg == name || arg.starts_with(&format!("{}=", name)))
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
                let mut config = if let Some(path) = &cli.config {
                    let config_str = fs::read_to_string(path)?;
                    toml::from_str(&config_str)?
                } else {
                    // Default config
                    aion_rlt::configurations::Configurations {
                        interval_secs: 10, // u64
                        batch_size: 32,
                        total_batches: 1000,
                        input_number: 8,
                        output_number: 3,
                        hidden_layers: 16,
                        reply_capacity: 4096,
                        model_name: "model.json".to_string(),
                        log_interval: 64,
                        mode: aion_rlt::RunningMode::Training,
                    }
                };

                // Override with CLI args when present
                if args.batch_size != 0 {
                    config.batch_size = args.batch_size;
                }
                if args.epochs != 0 {
                    config.total_batches = args.epochs as usize * 100;
                }
                if !args.model_name.is_empty() {
                    config.model_name = args.model_name.clone();
                }

                let config = Arc::new(config);

                aion_rlt::initialize(config.clone());

                // Start training node with dummy closures
                aion_rlt::node::Node::start(
                    |_lowest_state| vec![0.0; 8],              // dummy input
                    |_features, _action, _reward| (0.0, true), // dummy reward
                    aion_rlt::RunningMode::Training,
                    &config.model_name,
                )
                .await;

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
        Commands::Infer(args) => {
            println!("Performing inference with the following settings:");
            println!("  model name: {}", args.model_name);
            if let Some(features) = &args.features {
                println!("  features: {}", features);
            } else {
                println!("  features: using dummy values");
            }

            // Load config or use defaults
            let mut config = if let Some(path) = &cli.config {
                let config_str = fs::read_to_string(path)?;
                toml::from_str(&config_str)?
            } else {
                // Default config
                aion_rlt::configurations::Configurations {
                    interval_secs: 10,
                    batch_size: 32,
                    total_batches: 1000,
                    input_number: 8,
                    output_number: 3,
                    hidden_layers: 16,
                    reply_capacity: 4096,
                    model_name: "model.json".to_string(),
                    log_interval: 64,
                    mode: aion_rlt::RunningMode::Infer,
                }
            };

            // Override with CLI args when the user actually passed them
            if cli_option_present("--model-name") {
                config.model_name = args.model_name.clone();
            }

            if cli_option_present("--model-name") {
                config.model_name = args.model_name.clone();
            }

            let config = Arc::new(config);
            aion_rlt::initialize(config.clone());

            // Start inference node
            aion_rlt::node::Node::start(
                |_lowest_state| {
                    if let Some(features_str) = &args.features {
                        features_str.split(',').map(|s| s.trim().parse().unwrap_or(0.0)).collect()
                    } else {
                        vec![0.0; 8] // dummy
                    }
                },
                |_features, _action, _reward| (0.0, true), // dummy reward
                aion_rlt::RunningMode::Infer,
                &config.model_name,
            )
            .await;

            println!("Inference completed.");
        }
    }

    Ok(())
}
