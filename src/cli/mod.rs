use clap::Parser;

pub mod commands;

pub use commands::{Commands, LogLevel};

#[derive(Parser, Debug)]
#[command(author, version, about = "AIXKER-RLT CLI for training and exporting reinforcement learning models.", long_about = None)]
pub struct Cli {
    /// Set the global logging level
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,

    /// Path to a configuration file
    #[arg(long, value_name = "FILE")]
    pub config: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}
