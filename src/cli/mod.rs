use clap::Parser;

pub mod commands;

pub use commands::{Commands, LogLevel};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Aixker-RLT CLI — train, evaluate, and export reinforcement learning policies.",
    long_about = None
)]
pub struct Cli {
    /// Set the global logging level
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    pub log_level: LogLevel,

    /// Verbose log output (debug level); overrides --log-level
    #[arg(short = 'v', long, conflicts_with_all = ["silent", "errors_only"])]
    pub verbose: bool,

    /// Suppress all log output; overrides --log-level
    #[arg(short = 'q', long, visible_alias = "quiet", conflicts_with_all = ["verbose", "errors_only"])]
    pub silent: bool,

    /// Only print error logs; overrides --log-level
    #[arg(long, conflicts_with_all = ["verbose", "silent"])]
    pub errors_only: bool,

    /// Path to a configuration file
    #[arg(long, value_name = "FILE")]
    pub config: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}
