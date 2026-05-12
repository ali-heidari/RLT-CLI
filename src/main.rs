mod cli;

use clap::Parser;
use cli::commands::{InferArgs, TrainArgs};
use cli::{Cli, Commands};
use std::fs;
use std::sync::Arc;
use tokio;
use toml;

fn cli_option_present(name: &str) -> bool {
    std::env::args().any(|arg| arg == name || arg.starts_with(&format!("{}=", name)))
}

fn load_config(
    config_path: Option<&String>,
    default_mode: aion_rlt::RunningMode,
) -> Result<aion_rlt::configurations::Configurations, Box<dyn std::error::Error>> {
    if let Some(path) = config_path {
        let config_str = fs::read_to_string(path)?;
        let mut config: aion_rlt::configurations::Configurations = toml::from_str(&config_str)?;
        config.mode = default_mode;
        Ok(config)
    } else {
        Ok(aion_rlt::configurations::Configurations {
            interval_secs: 10,
            batch_size: 32,
            total_batches: 1000,
            input_number: 8,
            output_number: 3,
            hidden_layers: 16,
            reply_capacity: 4096,
            model_name: "model.json".to_string(),
            log_interval: 64,
            mode: default_mode,
        })
    }
}

fn override_train_config(
    config: &mut aion_rlt::configurations::Configurations,
    args: &TrainArgs,
) {
    if args.batch_size != 0 {
        config.batch_size = args.batch_size;
    }
    if args.epochs != 0 {
        config.total_batches = args.epochs as usize * 100;
    }
    if !args.model_name.is_empty() {
        config.model_name = args.model_name.clone();
    }
}

fn override_infer_config(
    config: &mut aion_rlt::configurations::Configurations,
    args: &InferArgs,
) {
    if cli_option_present("--model-name") {
        config.model_name = args.model_name.clone();
    }
}

async fn run_train(
    config_path: Option<&String>,
    args: TrainArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting training with the following settings:");
    println!("  dataset: {}", args.dataset);
    println!("  epochs: {}", args.epochs);
    println!("  batch size: {}", args.batch_size);
    println!("  learning rate: {}", args.learning_rate);
    println!("  model name: {}", args.model_name);
    println!("  dry run: {}", args.dry_run);

    if args.dry_run {
        println!("Dry run enabled. No training will be performed.");
        return Ok(());
    }

    let mut config = load_config(config_path, aion_rlt::RunningMode::Training)?;
    override_train_config(&mut config, &args);
    let config = Arc::new(config);

    aion_rlt::initialize(config.clone());
    aion_rlt::node::Node::start(
        |_lowest_state| vec![0.0; 8],              // dummy input
        |_features, _action, _reward| (0.0, true), // dummy reward
        aion_rlt::RunningMode::Training,
        &config.model_name,
    )
    .await;

    println!("Training node started. Training logic would run here.");
    Ok(())
}

fn parse_features(features: &Option<String>) -> Vec<f32> {
    if let Some(features_str) = features {
        features_str
            .split(',')
            .map(|s| s.trim().parse().unwrap_or(0.0))
            .collect()
    } else {
        vec![0.0; 8]
    }
}

async fn run_infer(
    config_path: Option<&String>,
    args: InferArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Performing inference with the following settings:");
    println!("  model name: {}", args.model_name);
    if let Some(features) = &args.features {
        println!("  features: {}", features);
    } else {
        println!("  features: using dummy values");
    }

    let mut config = load_config(config_path, aion_rlt::RunningMode::Infer)?;
    override_infer_config(&mut config, &args);
    let config = Arc::new(config);

    aion_rlt::initialize(config.clone());
    aion_rlt::node::Node::start(
        |_lowest_state| parse_features(&args.features),
        |_features, _action, _reward| (0.0, true),
        aion_rlt::RunningMode::Infer,
        &config.model_name,
    )
    .await;

    println!("Inference completed.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config_path = cli.config.clone();

    println!("AIXKER-RLT CLI invoked with log level: {:?}", cli.log_level);
    if let Some(config_path) = &config_path {
        println!("Using config file: {}", config_path);
    }

    match cli.command {
        Commands::Train(args) => run_train(config_path.as_ref(), args).await?,
        Commands::Export(args) => {
            println!("Exporting model with the following settings:");
            println!("  checkpoint: {}", args.checkpoint);
            println!("  output: {}", args.output);
            println!("  format: {:?}", args.format);
            fs::copy(&args.checkpoint, &args.output)?;
            println!("Model exported to {}", args.output);
        }
        Commands::Infer(args) => run_infer(config_path.as_ref(), args).await?,
    }

    Ok(())
}
