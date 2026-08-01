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

    /// Number of training epochs; one epoch is 100 training batches [default: 10]
    #[arg(long)]
    pub epochs: Option<u32>,

    /// Batch size for training [default: 32]
    #[arg(long)]
    pub batch_size: Option<u32>,

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

    /// Skip the first CSV row instead of parsing it as data [default: false]
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub has_header: Option<bool>,

    /// Field separator for CSV datasets [default: ,]
    #[arg(long, value_name = "CHAR")]
    pub delimiter: Option<char>,
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

    /// Skip the first CSV row instead of parsing it as data [default: false]
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    pub has_header: Option<bool>,

    /// Field separator for CSV datasets [default: ,]
    #[arg(long, value_name = "CHAR")]
    pub delimiter: Option<char>,
}

#[derive(clap::Args, Debug)]
pub struct ExportArgs {
    /// Path to the trained model checkpoint
    /// [default: the configured model under ./models]
    #[arg(long, value_name = "PATH")]
    pub checkpoint: Option<String>,

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
    /// Copy the checkpoint file as-is. Checkpoints are already JSON, so no
    /// conversion is performed.
    Json,
    // Real conversion targets (safetensors, onnx) are not implemented yet.
}
