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

    /// Compute backend to run on (overrides the config file; defaults to cpu)
    #[arg(long, value_enum)]
    pub backend: Option<Backend>,
}

#[derive(clap::Args, Debug)]
pub struct InferArgs {
    /// Path to the data source for inference
    #[arg(long, value_name = "PATH")]
    pub dataset: Option<String>,

    /// Name of the model checkpoint file to load for inference
    #[arg(long, value_name = "NAME")]
    pub model_name: Option<String>,

    /// Compute backend to run on (overrides the config file; defaults to cpu)
    #[arg(long, value_enum)]
    pub backend: Option<Backend>,
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

/// Compute backend selection for training and inference.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum Backend {
    /// Pure-Rust ndarray implementation (default)
    Cpu,
    /// wgpu compute shaders; falls back to CPU when no adapter is found
    Gpu,
}

impl From<Backend> for aixker_rlt::ComputeBackend {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Cpu => aixker_rlt::ComputeBackend::Cpu,
            Backend::Gpu => aixker_rlt::ComputeBackend::Gpu,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum LogLevel {
    /// No log output at all
    Silent,
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
