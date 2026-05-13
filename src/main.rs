mod cli;
mod providers;
mod reward_factory;

use clap::Parser;
use cli::commands::{InferArgs, TrainArgs};
use cli::{Cli, Commands};
use providers::csv_dataset::open_csv_dataset;
use providers::python_script::open_python_script;
use reward_factory::RewardFactory;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

fn load_config(
    config_path: Option<&String>,
    default_mode: aixker_rlt::RunningMode,
) -> Result<aixker_rlt::configurations::Configurations, Box<dyn std::error::Error>> {
    if let Some(path) = config_path {
        let config_str = fs::read_to_string(path)?;
        let value: toml::Value = toml::from_str(&config_str)?;
        let mut config: aixker_rlt::configurations::Configurations = value.clone().try_into()?;
        config.mode = default_mode;
        Ok(config)
    } else {
        Ok(aixker_rlt::configurations::Configurations {
            interval_secs: 10,
            batch_size: 32,
            total_batches: 1000,
            input_number: 8,
            output_number: 3,
            hidden_layers: 16,
            reply_capacity: 4096,
            model_name: String::new(),
            log_interval: 64,
            mode: default_mode,
        })
    }
}

fn load_dataset_path(
    config_path: Option<&String>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if let Some(path) = config_path {
        let config_str = fs::read_to_string(path)?;
        let value: toml::Value = toml::from_str(&config_str)?;
        Ok(value
            .get("dataset")
            .and_then(|v| v.as_str())
            .map(String::from))
    } else {
        Ok(None)
    }
}

fn load_reward_script_path(
    config_path: Option<&String>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if let Some(path) = config_path {
        let config_str = fs::read_to_string(path)?;
        let value: toml::Value = toml::from_str(&config_str)?;
        Ok(value
            .get("reward_script")
            .and_then(|v| v.as_str())
            .map(String::from))
    } else {
        Ok(None)
    }
}

fn override_train_config(
    config: &mut aixker_rlt::configurations::Configurations,
    args: &TrainArgs,
) {
    if args.batch_size != 0 {
        config.batch_size = args.batch_size;
    }
    if args.epochs != 0 {
        config.total_batches = args.epochs as usize * 100;
    }
    if let Some(model_name) = &args.model_name {
        config.model_name = model_name.clone();
    }
}

fn override_infer_config(
    config: &mut aixker_rlt::configurations::Configurations,
    args: &InferArgs,
) {
    if let Some(model_name) = &args.model_name {
        config.model_name = model_name.clone();
    }
}

async fn run_train(
    config_path: Option<&String>,
    args: TrainArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting training with the following settings:");
    println!(
        "  dataset: {}",
        args.dataset
            .as_deref()
            .unwrap_or("<from config or required>")
    );
    println!("  epochs: {}", args.epochs);
    println!("  batch size: {}", args.batch_size);
    println!("  learning rate: {}", args.learning_rate);
    println!(
        "  model name: {}",
        args.model_name.as_deref().unwrap_or("<from config>")
    );
    println!("  dry run: {}", args.dry_run);

    if args.dry_run {
        println!("Dry run enabled. No training will be performed.");
        return Ok(());
    }

    let mut config = load_config(config_path, aixker_rlt::RunningMode::Training)?;
    override_train_config(&mut config, &args);
    if config.model_name.is_empty() {
        return Err("model_name must be set in config.toml or via --model-name".into());
    }

    let config_dataset = load_dataset_path(config_path)?;
    let config_reward_script = load_reward_script_path(config_path)?;
    let dataset_path = args
        .dataset
        .clone()
        .or(config_dataset)
        .ok_or("dataset must be set in config.toml or via --dataset")?;

    let reward_script_path = args.reward_script.clone().or(config_reward_script);
    println!(
        "  reward script: {}",
        reward_script_path
            .as_deref()
            .unwrap_or("<disabled>")
    );

    let reward_factory = if let Some(script_path) = reward_script_path {
        let path = PathBuf::from(&script_path);
        if !path.exists() {
            return Err(format!(
                "reward script path does not exist: {}",
                path.display()
            )
            .into());
        }
        if !path.is_file() {
            return Err(format!(
                "reward script path is not a file: {}",
                path.display()
            )
            .into());
        }
        RewardFactory::new(Some(path))
    } else {
        RewardFactory::new(None)
    };

    // Determine which data provider to use based on file extension
    let is_python_script = dataset_path.ends_with(".py");
    
    if is_python_script {
        println!("Using Python script data provider from {}", dataset_path);
        let mut provider = open_python_script(&dataset_path)?;
        
        // Get first sample to validate
        if let Some(first_row_result) = provider.next() {
            let first_row = first_row_result?;
            println!("First sample loaded with {} features", first_row.len());
        }

        let cloned_provider = Arc::new(std::sync::Mutex::new(provider));
        let config = Arc::new(config);
        let input_number = config.input_number as usize;

        aixker_rlt::initialize(config.clone());
        aixker_rlt::node::Node::start(
            move |_lowest_state| {
                if let Some(row_result) = cloned_provider.lock().unwrap().next() {
                    match row_result {
                        Ok(row) => row.into_iter().map(|x| x as f32).collect(),
                        Err(_) => vec![0.0; input_number],
                    }
                } else {
                    vec![0.0; input_number]
                }
            },
            move |features, action, reward| reward_factory.evaluate(features, action, reward),
            aixker_rlt::RunningMode::Training,
            &config.model_name,
        )
        .await;
    } else {
        let dataset = open_csv_dataset(&dataset_path)?;
        let cloned_dataset = Arc::new(std::sync::Mutex::new(dataset));
        println!("Opened CSV dataset stream from {}", dataset_path);

        if let Some(first_row) = cloned_dataset.lock().unwrap().next() {
            let row = first_row?;
            println!("First CSV row loaded with {} fields", row.len());
        }

        let config = Arc::new(config);
        let input_number = config.input_number as usize;

        aixker_rlt::initialize(config.clone());
        aixker_rlt::node::Node::start(
            move |_lowest_state| {
                if let Some(row_result) = cloned_dataset.lock().unwrap().next() {
                    match row_result {
                        Ok(row) => row.into_iter().map(|x| x as f32).collect(),
                        Err(_) => vec![0.0; input_number],
                    }
                } else {
                    vec![0.0; input_number]
                }
            },
            move |features, action, reward| reward_factory.evaluate(features, action, reward),
            aixker_rlt::RunningMode::Training,
            &config.model_name,
        )
        .await;
    }

    println!("Training node started. Training logic would run here.");
    Ok(())
}

async fn run_infer(
    config_path: Option<&String>,
    args: InferArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Performing inference with the following settings:");
    
    let config_dataset = load_dataset_path(config_path)?;
    let dataset_path = args
        .dataset
        .clone()
        .or(config_dataset)
        .ok_or("dataset must be set in config.toml or via --dataset")?;
    
    println!("  dataset: {}", dataset_path);
    println!(
        "  model name: {}",
        args.model_name.as_deref().unwrap_or("<from config>")
    );

    let mut config = load_config(config_path, aixker_rlt::RunningMode::Infer)?;
    override_infer_config(&mut config, &args);
    if config.model_name.is_empty() {
        return Err("model_name must be set in config.toml or via --model-name".into());
    }
    let config = Arc::new(config);

    // Determine which data provider to use based on file extension
    let is_python_script = dataset_path.ends_with(".py");

    aixker_rlt::initialize(config.clone());

    if is_python_script {
        println!("Using Python script data provider from {}", dataset_path);
        let provider = open_python_script(&dataset_path)?;
        let cloned_provider = Arc::new(std::sync::Mutex::new(provider));
        let input_number = config.input_number as usize;

        aixker_rlt::node::Node::start(
            move |_lowest_state| {
                if let Some(row_result) = cloned_provider.lock().unwrap().next() {
                    match row_result {
                        Ok(row) => row.into_iter().map(|x| x as f32).collect(),
                        Err(_) => vec![0.0; input_number],
                    }
                } else {
                    vec![0.0; input_number]
                }
            },
            |_features, _action, _reward| (0.0, true),
            aixker_rlt::RunningMode::Infer,
            &config.model_name,
        )
        .await;
    } else {
        let provider = open_csv_dataset(&dataset_path)?;
        let cloned_provider = Arc::new(std::sync::Mutex::new(provider));
        let input_number = config.input_number as usize;

        println!("Using CSV data provider from {}", dataset_path);

        aixker_rlt::node::Node::start(
            move |_lowest_state| {
                if let Some(row_result) = cloned_provider.lock().unwrap().next() {
                    match row_result {
                        Ok(row) => row.into_iter().map(|x| x as f32).collect(),
                        Err(_) => vec![0.0; input_number],
                    }
                } else {
                    vec![0.0; input_number]
                }
            },
            |_features, _action, _reward| (0.0, true),
            aixker_rlt::RunningMode::Infer,
            &config.model_name,
        )
        .await;
    }

    println!("Inference completed.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config_path = cli.config.clone().or_else(|| Some("Config.toml".to_string()));

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
