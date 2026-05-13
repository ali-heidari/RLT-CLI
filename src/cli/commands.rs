use clap::{Subcommand, ValueEnum};

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start training a model with customizable hyperparameters
    Train(TrainArgs),
    /// Perform inference to get actions from a trained model
    Infer(InferArgs),
    /// Export a trained model to a supported format
    Export(ExportArgs),
}

#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Path to the training CSV dataset
    #[arg(long, value_name = "PATH")]
    pub dataset: Option<String>,

    /// Number of training epochs
    #[arg(long, default_value_t = 10)]
    pub epochs: u32,

    /// Batch size for training
    #[arg(long, default_value_t = 32)]
    pub batch_size: u32,

    /// Initial learning rate
    #[arg(long, default_value_t = 0.001)]
    pub learning_rate: f32,

    /// Enable dry-run mode without actually executing training
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Name of the model checkpoint file to use or write
    #[arg(long, value_name = "NAME")]
    pub model_name: Option<String>,

    /// Path to a Python reward script that computes reward and success values
    #[arg(long, value_name = "PATH")]
    pub reward_script: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct InferArgs {
    /// Name of the model checkpoint file to load for inference
    #[arg(long, value_name = "NAME")]
    pub model_name: Option<String>,

    /// Input features as comma-separated floats (e.g., "1.0,2.0,3.0")
    #[arg(long, value_name = "FEATURES")]
    pub features: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct ExportArgs {
    /// Path to the trained model checkpoint
    #[arg(long, value_name = "PATH", default_value = "./checkpoints/latest.json")]
    pub checkpoint: String,

    /// Output path for the exported model
    #[arg(long, value_name = "PATH", default_value = "./exported/model.json")]
    pub output: String,

    /// Export format for the model
    #[arg(long, value_enum, default_value_t = ExportFormat::Json)]
    pub format: ExportFormat,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ExportFormat {
    Json,
    // Add other formats as needed
}
