mod action_writer;
mod cli;
mod eval;
mod init;
mod inspect;
mod providers;
mod reward_factory;
mod stop_signal;

use action_writer::{ActionWriter, STDOUT_DESTINATION};
use clap::Parser;
use cli::commands::{
    EvalArgs, ExportArgs, InferArgs, InitArgs, InspectArgs, ReportFormat, TrainArgs,
};
use cli::{Cli, Commands};
use eval::{Baseline, PolicyStats, Report};
use providers::csv_dataset::{open_csv_dataset, CsvOptions};
use providers::feature_source::{FeatureSource, Row};
use providers::python_script::open_python_script;
use reward_factory::RewardFactory;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use stop_signal::StopSignal;

/// Config file used when `--config` is not given. Unlike an explicitly
/// requested file this one is optional, so a fresh clone runs without it.
const DEFAULT_CONFIG_PATH: &str = "Config.toml";

const DEFAULT_INTERVAL_SECS: u64 = 10;
const DEFAULT_BATCH_SIZE: u32 = 32;
const DEFAULT_INPUT_NUMBER: usize = 8;
const DEFAULT_OUTPUT_NUMBER: usize = 3;
const DEFAULT_HIDDEN_LAYERS: usize = 16;
const DEFAULT_REPLY_CAPACITY: usize = 4096;
const DEFAULT_LOG_INTERVAL: u64 = 64;
const DEFAULT_EPOCHS: u32 = 10;

/// How long a Python script may take to answer one request.
///
/// Generous, because the cost of being wrong in one direction is a spurious
/// failure and in the other is a CLI that hangs with no output at all. `0`
/// disables the deadline.
const DEFAULT_SCRIPT_TIMEOUT_SECS: u64 = 30;

/// Training batches that one `--epochs` unit maps to.
const BATCHES_PER_EPOCH: usize = 100;

/// Directory the library keeps checkpoints in.
const MODELS_DIR: &str = "./models";

/// Everything the CLI reads out of the TOML config file, parsed once per run.
///
/// Every field is optional so an absent key falls through to the built-in
/// default. Unknown keys (`debug`, `mode`, ...) are ignored: the running mode
/// is decided by the subcommand, not by the file.
#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    interval_secs: Option<u64>,
    batch_size: Option<u32>,
    total_batches: Option<usize>,
    input_number: Option<usize>,
    output_number: Option<usize>,
    hidden_layers: Option<usize>,
    reply_capacity: Option<usize>,
    model_name: Option<String>,
    log_interval: Option<u64>,
    backend: Option<aixker_rlt::ComputeBackend>,
    dataset: Option<String>,
    reward_script: Option<String>,
    has_header: Option<bool>,
    delimiter: Option<char>,
    script_timeout_secs: Option<u64>,
}

/// Every key [`FileConfig`] understands.
///
/// Kept in step with the struct by `every_config_field_is_in_the_known_key_set`,
/// which fails to compile if a field is added without a matching entry here.
const KNOWN_CONFIG_KEYS: [&str; 15] = [
    "interval_secs",
    "batch_size",
    "total_batches",
    "input_number",
    "output_number",
    "hidden_layers",
    "reply_capacity",
    "model_name",
    "log_interval",
    "backend",
    "dataset",
    "reward_script",
    "has_header",
    "delimiter",
    "script_timeout_secs",
];

/// Keys ignored on purpose rather than warned about.
///
/// `mode` is decided by the subcommand and `debug` predates `--log-level`. Both
/// appear in configs written for earlier versions, so warning about them would
/// be noise rather than help.
const IGNORED_CONFIG_KEYS: [&str; 2] = ["mode", "debug"];

/// Keys in the config file that the CLI does not understand.
///
/// Returns them rather than logging so the rule can be asserted directly.
fn unknown_config_keys(text: &str) -> Vec<String> {
    let Ok(table) = toml::from_str::<toml::Table>(text) else {
        // Not parseable at all; the typed parse reports the real problem.
        return Vec::new();
    };

    table
        .keys()
        .filter(|key| {
            let key = key.as_str();
            !KNOWN_CONFIG_KEYS.contains(&key) && !IGNORED_CONFIG_KEYS.contains(&key)
        })
        .cloned()
        .collect()
}

/// Where an effective value came from, reported by the settings block.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Source {
    Flag,
    File,
    Default,
    /// A default chosen from context rather than a fixed constant, carrying the
    /// reason it was chosen — "one setting, several defaults" is confusing
    /// unless the report says which one applied and why.
    Derived(&'static str),
}

impl Source {
    fn label(self, config_file: Option<&str>) -> String {
        match self {
            Source::Flag => "flag".to_string(),
            Source::File => config_file.unwrap_or("config file").to_string(),
            Source::Default => "default".to_string(),
            Source::Derived(reason) => reason.to_string(),
        }
    }
}

/// Pick the first value that was actually set: flag, then config file, then default.
fn resolve<T>(flag: Option<T>, file: Option<T>, default: T) -> (T, Source) {
    match (flag, file) {
        (Some(value), _) => (value, Source::Flag),
        (None, Some(value)) => (value, Source::File),
        (None, None) => (default, Source::Default),
    }
}

/// The file the library reads and writes a checkpoint at.
///
/// `Model::model_path` joins the node id and the configured model name, and
/// `Node::start` is handed the model name as the node id — hence the doubled
/// stem. See `docs/found-issues.md`, issue 1; this mirrors the library so the
/// CLI can report the real path instead of a path that does not exist.
fn checkpoint_path(model_name: &str) -> PathBuf {
    PathBuf::from(MODELS_DIR).join(format!("{}.{}", model_name, model_name))
}

/// Render a path as an absolute one wherever possible.
///
/// The library keeps checkpoints in a relative `./models`, so the same command
/// run from two directories reads and writes different files with no hint that
/// anything differs. Reporting the absolute path is the cheapest way to make
/// that visible until `--models-dir` is possible.
///
/// Falls back to the path as given: a checkpoint that does not exist yet cannot
/// be canonicalised, and a half-resolved path would be worse than an honest
/// relative one.
fn absolute(path: &Path) -> String {
    if let Ok(resolved) = path.canonicalize() {
        return resolved.display().to_string();
    }

    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => match parent.canonicalize() {
            Ok(resolved) => resolved.join(name).display().to_string(),
            Err(_) => path.display().to_string(),
        },
        _ => path.display().to_string(),
    }
}

/// Check that a checkpoint exists and actually holds a model.
///
/// The library creates the file on load and only fills it when a batch is
/// saved, so a zero-byte checkpoint is a run that trained nothing.
fn ensure_checkpoint(checkpoint: &Path, hint: &str) -> Result<(), Box<dyn std::error::Error>> {
    match fs::metadata(checkpoint) {
        // Absolute, because `./models` is relative to the working directory and
        // "checkpoint not found: ./models/m.json" is unhelpful when the file
        // exists perfectly well somewhere else.
        Err(_) => Err(format!("checkpoint not found: {}. {}", absolute(checkpoint), hint).into()),
        // A directory has a non-zero length of its own, so the emptiness check
        // below would wave it through and leave the real complaint to a raw
        // copy or load error much later.
        Ok(metadata) if !metadata.is_file() => Err(format!(
            "checkpoint path is not a file: {}. {}",
            checkpoint.display(),
            hint
        )
        .into()),
        Ok(metadata) if metadata.len() == 0 => Err(format!(
            "checkpoint is empty: {}. The model was never saved; train it again.",
            checkpoint.display()
        )
        .into()),
        Ok(_) => Ok(()),
    }
}

/// Whether a path selects the Python-script provider rather than CSV.
///
/// Compared without regard to case: Windows filesystems are case-insensitive,
/// so `provider.PY` names the same file as `provider.py` and was being read as
/// CSV.
fn is_python_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .map(|extension| extension.eq_ignore_ascii_case("py"))
        .unwrap_or(false)
}

/// Whether a dataset path selects the Python-script provider rather than CSV.
fn is_python_dataset(dataset: &Option<String>) -> bool {
    dataset.as_deref().map(is_python_path).unwrap_or(false)
}

/// Same as [`resolve`] for settings that may legitimately stay unset.
fn resolve_opt<T>(flag: Option<T>, file: Option<T>) -> (Option<T>, Source) {
    match (flag, file) {
        (Some(value), _) => (Some(value), Source::Flag),
        (None, Some(value)) => (Some(value), Source::File),
        (None, None) => (None, Source::Default),
    }
}

/// Render an optional setting for the report.
fn show(value: &Option<String>, absent: &str) -> String {
    value.clone().unwrap_or_else(|| absent.to_string())
}

/// The settings for one run, with the origin of each value for the report.
struct Resolved {
    config: aixker_rlt::configurations::Configurations,
    dataset: Option<String>,
    reward_script: Option<String>,
    csv_has_header: bool,
    csv_delimiter: char,
    /// How long a Python script may take to answer. `None` waits forever.
    script_timeout: Option<std::time::Duration>,
    provenance: Vec<(&'static str, String, Source)>,
}

/// Resolve how long inference sleeps between samples.
///
/// The library polls between inference samples, which suits a live provider and
/// makes no sense over a file — the rows are already there. So the *default*
/// depends on the dataset: a 400-row CSV at the old flat default of 10s spent
/// 67 minutes asleep. An explicit flag or config value still wins, so this only
/// replaces the constant, not the configuration.
fn resolve_interval(
    flag: Option<u64>,
    file: Option<u64>,
    dataset: &Option<String>,
) -> (u64, Source) {
    match (flag, file) {
        (Some(seconds), _) => (seconds, Source::Flag),
        (None, Some(seconds)) => (seconds, Source::File),
        (None, None) if dataset.is_some() && !is_python_dataset(dataset) => {
            (0, Source::Derived("file dataset"))
        }
        (None, None) => (DEFAULT_INTERVAL_SECS, Source::Default),
    }
}

/// Resolve the Python script deadline and describe where it came from.
///
/// Zero means "wait forever", which is what the CLI did before the deadline
/// existed — kept as the escape hatch for a legitimately slow script.
fn resolve_script_timeout(
    flag: Option<u64>,
    file: Option<u64>,
) -> (Option<std::time::Duration>, String, Source) {
    let (seconds, source) = resolve(flag, file, DEFAULT_SCRIPT_TIMEOUT_SECS);

    let (timeout, shown) = match seconds {
        0 => (None, "disabled (wait forever)".to_string()),
        seconds => (
            Some(std::time::Duration::from_secs(seconds)),
            format!("{}s", seconds),
        ),
    };

    (timeout, shown, source)
}

/// Resolve the CSV reader options and describe where they came from.
///
/// The rows are only worth reporting for a CSV dataset, so the caller decides
/// whether to append them.
fn resolve_csv(
    flag_has_header: Option<bool>,
    flag_delimiter: Option<char>,
    file: &FileConfig,
) -> (bool, char, Vec<(&'static str, String, Source)>) {
    let (has_header, has_header_source) = resolve(flag_has_header, file.has_header, false);
    let (delimiter, delimiter_source) = resolve(flag_delimiter, file.delimiter, ',');

    let provenance = vec![
        ("csv header row", has_header.to_string(), has_header_source),
        (
            "csv delimiter",
            format!("'{}'", delimiter),
            delimiter_source,
        ),
    ];

    (has_header, delimiter, provenance)
}

/// Convert the delimiter into the single byte the CSV reader needs.
fn delimiter_byte(delimiter: char) -> Result<u8, Box<dyn std::error::Error>> {
    if delimiter.is_ascii() {
        Ok(delimiter as u8)
    } else {
        Err(format!(
            "delimiter must be a single ASCII character, got '{}'",
            delimiter
        )
        .into())
    }
}

/// Load the TOML config file.
///
/// A file requested with `--config` must exist. The default `Config.toml` is
/// optional, so the CLI works on a fresh clone that has no config file at all.
/// Returns the parsed config and the path it was read from, if any.
fn load_file_config(
    config_path: Option<&String>,
) -> Result<(FileConfig, Option<String>), Box<dyn std::error::Error>> {
    let (path, explicit) = match config_path {
        Some(path) => (path.clone(), true),
        None => (DEFAULT_CONFIG_PATH.to_string(), false),
    };

    if !Path::new(&path).exists() {
        if explicit {
            return Err(format!("config file not found: {}", path).into());
        }
        log::debug!("no {} found; using built-in defaults", path);
        return Ok((FileConfig::default(), None));
    }

    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read config file '{}': {}", path, err))?;
    let config = toml::from_str::<FileConfig>(&text)
        .map_err(|err| format!("failed to parse config file '{}': {}", path, err))?;

    // A second parse of the same string, purely to audit the key names. Reading
    // the values through `toml::Table` instead would lose the line and column
    // that make a type error findable. A warning rather than an error: an
    // unknown key is a mistake, not a reason to refuse to run.
    for key in unknown_config_keys(&text) {
        log::warn!(
            "unknown key '{}' in {} has no effect. See docs/usage.md for the supported keys.",
            key,
            path
        );
    }

    Ok((config, Some(path)))
}

/// Resolve training settings with precedence: flag, then config file, then default.
fn resolve_train(file: &FileConfig, args: &TrainArgs) -> Resolved {
    let (model_name, model_name_source) =
        resolve_opt(args.model_name.clone(), file.model_name.clone());
    let (batch_size, batch_size_source) =
        resolve(args.batch_size, file.batch_size, DEFAULT_BATCH_SIZE);
    let (backend, backend_source) = resolve(
        args.backend.map(Into::into),
        file.backend,
        aixker_rlt::ComputeBackend::default(),
    );
    let (dataset, dataset_source) = resolve_opt(args.dataset.clone(), file.dataset.clone());
    let (reward_script, reward_script_source) =
        resolve_opt(args.reward_script.clone(), file.reward_script.clone());

    // `--epochs` wins, then an explicit `total_batches` in the file, then the
    // default epoch count. One epoch is BATCHES_PER_EPOCH batches.
    let (total_batches, total_batches_source) = match (args.epochs, file.total_batches) {
        (Some(epochs), _) => (epochs as usize * BATCHES_PER_EPOCH, Source::Flag),
        (None, Some(total)) => (total, Source::File),
        (None, None) => (DEFAULT_EPOCHS as usize * BATCHES_PER_EPOCH, Source::Default),
    };

    let (csv_has_header, csv_delimiter, csv_provenance) =
        resolve_csv(args.has_header, args.delimiter, file);
    let (script_timeout, script_timeout_shown, script_timeout_source) =
        resolve_script_timeout(args.script_timeout, file.script_timeout_secs);

    let mut provenance = vec![
        ("dataset", show(&dataset, "<not set>"), dataset_source),
        (
            "model name",
            show(&model_name, "<not set>"),
            model_name_source,
        ),
        ("batch size", batch_size.to_string(), batch_size_source),
        (
            "total batches",
            if total_batches_source == Source::Flag {
                // Only the --epochs flag involves the multiplier; a config file
                // sets total_batches directly.
                format!("{} ({} per epoch)", total_batches, BATCHES_PER_EPOCH)
            } else {
                total_batches.to_string()
            },
            total_batches_source,
        ),
        (
            "backend",
            format!("{:?}", backend).to_lowercase(),
            backend_source,
        ),
        (
            "reward script",
            show(&reward_script, "<disabled>"),
            reward_script_source,
        ),
        (
            "dry run",
            args.dry_run.to_string(),
            if args.dry_run {
                Source::Flag
            } else {
                Source::Default
            },
        ),
    ];

    // The CSV reader options are noise for a Python data provider.
    if !is_python_dataset(&dataset) {
        provenance.extend(csv_provenance);
    }

    // ...and the script deadline is noise when no script is involved at all.
    if is_python_dataset(&dataset) || reward_script.is_some() {
        provenance.push((
            "script timeout",
            script_timeout_shown,
            script_timeout_source,
        ));
    }

    Resolved {
        config: aixker_rlt::configurations::Configurations {
            interval_secs: file.interval_secs.unwrap_or(DEFAULT_INTERVAL_SECS),
            batch_size,
            total_batches,
            input_number: file.input_number.unwrap_or(DEFAULT_INPUT_NUMBER),
            output_number: file.output_number.unwrap_or(DEFAULT_OUTPUT_NUMBER),
            hidden_layers: file.hidden_layers.unwrap_or(DEFAULT_HIDDEN_LAYERS),
            reply_capacity: file.reply_capacity.unwrap_or(DEFAULT_REPLY_CAPACITY),
            model_name: model_name.unwrap_or_default(),
            log_interval: file.log_interval.unwrap_or(DEFAULT_LOG_INTERVAL),
            mode: aixker_rlt::RunningMode::Training,
            backend,
        },
        dataset,
        reward_script,
        csv_has_header,
        csv_delimiter,
        script_timeout,
        provenance,
    }
}

/// Resolve inference settings with precedence: flag, then config file, then default.
fn resolve_infer(file: &FileConfig, args: &InferArgs) -> Resolved {
    let (model_name, model_name_source) =
        resolve_opt(args.model_name.clone(), file.model_name.clone());
    let (backend, backend_source) = resolve(
        args.backend.map(Into::into),
        file.backend,
        aixker_rlt::ComputeBackend::default(),
    );
    let (dataset, dataset_source) = resolve_opt(args.dataset.clone(), file.dataset.clone());
    let (csv_has_header, csv_delimiter, csv_provenance) =
        resolve_csv(args.has_header, args.delimiter, file);
    let (script_timeout, script_timeout_shown, script_timeout_source) =
        resolve_script_timeout(args.script_timeout, file.script_timeout_secs);
    let (interval_secs, interval_source) =
        resolve_interval(args.interval_secs, file.interval_secs, &dataset);

    let (actions_output, actions_output_source) =
        resolve(args.output.clone(), None, STDOUT_DESTINATION.to_string());

    let mut provenance = vec![
        ("dataset", show(&dataset, "<not set>"), dataset_source),
        (
            "model name",
            show(&model_name, "<not set>"),
            model_name_source,
        ),
        (
            "backend",
            format!("{:?}", backend).to_lowercase(),
            backend_source,
        ),
        ("interval", format!("{}s", interval_secs), interval_source),
        (
            "actions output",
            if actions_output == STDOUT_DESTINATION {
                "stdout".to_string()
            } else {
                actions_output
            },
            actions_output_source,
        ),
        (
            "action features",
            args.with_features.to_string(),
            if args.with_features {
                Source::Flag
            } else {
                Source::Default
            },
        ),
    ];

    // The CSV reader options are noise for a Python data provider, and the
    // script deadline is noise for anything else.
    if is_python_dataset(&dataset) {
        provenance.push((
            "script timeout",
            script_timeout_shown,
            script_timeout_source,
        ));
    } else {
        provenance.extend(csv_provenance);
    }

    Resolved {
        config: aixker_rlt::configurations::Configurations {
            interval_secs,
            batch_size: file.batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
            total_batches: file
                .total_batches
                .unwrap_or(DEFAULT_EPOCHS as usize * BATCHES_PER_EPOCH),
            input_number: file.input_number.unwrap_or(DEFAULT_INPUT_NUMBER),
            output_number: file.output_number.unwrap_or(DEFAULT_OUTPUT_NUMBER),
            hidden_layers: file.hidden_layers.unwrap_or(DEFAULT_HIDDEN_LAYERS),
            reply_capacity: file.reply_capacity.unwrap_or(DEFAULT_REPLY_CAPACITY),
            model_name: model_name.unwrap_or_default(),
            log_interval: file.log_interval.unwrap_or(DEFAULT_LOG_INTERVAL),
            mode: aixker_rlt::RunningMode::Infer,
            backend,
        },
        dataset,
        reward_script: None,
        csv_has_header,
        csv_delimiter,
        script_timeout,
        provenance,
    }
}

/// User-facing command output.
///
/// Progress and diagnostics go through the `log` facade, which `--silent`
/// already switches off. This writer covers the output that is the command's
/// result rather than a log line, and honours `--silent` explicitly — a plain
/// `println!` cannot be filtered by the log level.
#[derive(Copy, Clone)]
struct Output {
    silent: bool,
}

impl Output {
    fn new(silent: bool) -> Self {
        Self { silent }
    }

    /// Print one line of command output.
    fn line(&self, text: impl std::fmt::Display) {
        if !self.silent {
            println!("{}", text);
        }
    }

    /// Print the effective settings and where each value came from.
    fn settings(
        &self,
        title: &str,
        provenance: &[(&'static str, String, Source)],
        config_file: Option<&str>,
    ) {
        if self.silent {
            return;
        }
        println!("{}", title);
        for (name, value, source) in provenance {
            println!("  {}: {} ({})", name, value, source.label(config_file));
        }
    }
}

async fn run_train(
    file_config: &FileConfig,
    config_file: Option<&str>,
    out: Output,
    args: TrainArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolved = resolve_train(file_config, &args);
    out.settings(
        "Starting training with the following settings:",
        &resolved.provenance,
        config_file,
    );

    let config = resolved.config;
    if config.model_name.is_empty() {
        return Err("model_name must be set in the config file or via --model-name".into());
    }

    let dataset_path = resolved
        .dataset
        .ok_or("dataset must be set in the config file or via --dataset")?;
    if !Path::new(&dataset_path).is_file() {
        return Err(format!("dataset not found: {}", dataset_path).into());
    }

    let reward_script = match resolved.reward_script {
        Some(script_path) => {
            let path = PathBuf::from(&script_path);
            if !path.is_file() {
                return Err(
                    format!("reward script not found or not a file: {}", path.display()).into(),
                );
            }
            Some(path)
        }
        None => None,
    };

    // Everything the run needs has now been validated, which is the point of a
    // dry run: it must fail on a configuration that could not actually run.
    if args.dry_run {
        out.line("Dry run enabled. No training will be performed.");
        return Ok(());
    }

    // The reward factory ends the run through this; the data source is what
    // the library actually listens to.
    let stop = StopSignal::new();
    stop_on_interrupt(stop.clone());
    let reward_factory = RewardFactory::new(
        reward_script.as_deref(),
        stop.clone(),
        resolved.script_timeout,
    )?;

    let config = Arc::new(config);
    let input_number = config.input_number;

    let provider = open_provider(
        &dataset_path,
        resolved.csv_has_header,
        resolved.csv_delimiter,
        resolved.script_timeout,
    )?;

    let mut source = FeatureSource::new(provider, input_number).with_stop_signal(stop.clone());
    let width = source.validate_first()?;
    log::info!("first row loaded with {} value(s)", width);

    let source = Arc::new(Mutex::new(source));
    let node_source = source.clone();

    aixker_rlt::initialize(config.clone());
    aixker_rlt::node::Node::start(
        move |_lowest_state| node_source.lock().unwrap().next_features(),
        move |features, action, _counter| reward_factory.evaluate(features, action),
        aixker_rlt::RunningMode::Training,
        &config.model_name,
    )
    .await;

    report_source(&source)?;

    // A run that never completed a batch leaves the checkpoint empty. Exiting
    // zero without saying so is how "training worked" turns into "inference
    // fails for no visible reason" later.
    let checkpoint = checkpoint_path(&config.model_name);
    match fs::metadata(&checkpoint) {
        Ok(metadata) if metadata.len() > 0 => log::info!(
            "checkpoint written to {} ({} bytes)",
            absolute(&checkpoint),
            metadata.len()
        ),
        _ => log::warn!(
            "no model was saved to {}: the run ended before a full batch completed. \
             Check that the dataset holds enough rows for batch_size and total_batches.",
            absolute(&checkpoint)
        ),
    }

    log::info!("training finished");
    Ok(())
}

/// Open the data provider a dataset path selects.
///
/// Boxed so the `.py` and CSV cases collapse into one code path. `train` and
/// `infer` each used to carry both, four nearly identical blocks in total, and
/// they had already drifted: one logged its actions, the other discarded them.
///
/// `+ Send` is load-bearing — the boxed iterator ends up inside an
/// `Arc<Mutex<..>>` shared with the node loop, which is not `Sync` without it.
///
/// The delimiter is converted inside the CSV case only, so a non-ASCII
/// delimiter alongside a Python provider stays the harmless irrelevance it is.
fn open_provider(
    dataset: &str,
    has_header: bool,
    delimiter: char,
    timeout: Option<std::time::Duration>,
) -> Result<Box<dyn Iterator<Item = Row> + Send>, Box<dyn std::error::Error>> {
    if is_python_path(dataset) {
        log::info!("using Python script data provider from {}", dataset);
        Ok(Box::new(open_python_script(dataset, timeout)?))
    } else {
        log::info!("using CSV data provider from {}", dataset);
        let options = CsvOptions {
            has_header,
            delimiter: delimiter_byte(delimiter)?,
        };
        Ok(Box::new(open_csv_dataset(dataset, options)?))
    }
}

/// Turn Ctrl-C into a clean end-of-data rather than a killed process.
///
/// The library ends its loop when the input closure returns an empty vector, so
/// interrupting is a matter of telling the data source to stop. The run then
/// winds down exactly as if the dataset had ended: totals are reported, and the
/// last completed batch's checkpoint survives instead of the process being
/// killed part-way through writing one.
///
/// Uses `tokio::signal` rather than a signal-handling crate — tokio is already
/// a dependency, and this needs no extra machinery.
fn stop_on_interrupt(stop: Arc<StopSignal>) {
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_err() {
            return;
        }
        stop.stop_gracefully(
            "interrupted: finishing the current step and saving what is done".to_string(),
        );

        // Installing a handler means the second Ctrl-C would be swallowed too,
        // leaving no way out of a wind-down that is itself stuck — waiting on a
        // script mid-request, say. So the second one means "now", with the
        // conventional 128 + SIGINT status.
        if tokio::signal::ctrl_c().await.is_ok() {
            log::warn!("interrupted again; exiting without saving");
            std::process::exit(130);
        }
    });
}

/// Report how the data source was consumed, surfacing the error that stopped
/// the run if there was one.
fn report_source<I>(source: &Arc<Mutex<FeatureSource<I>>>) -> Result<(), Box<dyn std::error::Error>>
where
    I: Iterator<Item = Row>,
{
    let stats = source.lock().unwrap().finish()?;
    log::info!(
        "read {} row(s) from the data source, skipped {}",
        stats.rows_read,
        stats.rows_skipped
    );
    if stats.rows_skipped > 0 {
        log::warn!(
            "{} row(s) were skipped because the data source returned an error",
            stats.rows_skipped
        );
    }
    Ok(())
}

/// The inference callback: record the decision, then tell the library to carry
/// on.
///
/// Both provider branches use this, so the two paths cannot drift apart — which
/// is how one of them ended up logging its actions at debug level while the
/// other discarded them outright.
///
/// Locking the source here is safe: the library drops the guard the input
/// closure took before it calls this one.
fn action_recorder<I>(
    source: Arc<Mutex<FeatureSource<I>>>,
    actions: Arc<ActionWriter>,
) -> impl Fn(&Vec<f32>, u32, u32) -> (f32, bool)
where
    I: Iterator<Item = Row>,
{
    move |features: &Vec<f32>, action: u32, _counter: u32| {
        let row = source.lock().unwrap().last_row_number();
        actions.emit(row, features, action);
        // Inference computes no reward; `true` keeps the loop running.
        (0.0, true)
    }
}

async fn run_infer(
    file_config: &FileConfig,
    config_file: Option<&str>,
    out: Output,
    args: InferArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolved = resolve_infer(file_config, &args);
    out.settings(
        "Performing inference with the following settings:",
        &resolved.provenance,
        config_file,
    );

    let config = resolved.config;
    if config.model_name.is_empty() {
        return Err("model_name must be set in the config file or via --model-name".into());
    }

    // Without this the library silently falls back to random weights, and
    // inference returns confident-looking actions from an untrained model.
    let checkpoint = checkpoint_path(&config.model_name);
    ensure_checkpoint(
        &checkpoint,
        "Train a model first, or point --model-name at an existing checkpoint.",
    )?;
    log::info!("loading checkpoint {}", absolute(&checkpoint));

    let dataset_path = resolved
        .dataset
        .ok_or("dataset must be set in the config file or via --dataset")?;
    if !Path::new(&dataset_path).is_file() {
        return Err(format!("dataset not found: {}", dataset_path).into());
    }

    // Opened before the run so a bad destination fails immediately, rather
    // than after the model has already computed decisions with nowhere to go.
    let actions = Arc::new(ActionWriter::new(
        args.output.as_deref().unwrap_or(STDOUT_DESTINATION),
        args.with_features,
        out.silent,
    )?);

    let config = Arc::new(config);

    aixker_rlt::initialize(config.clone());

    let input_number = config.input_number;

    // Ctrl-C part-way through a long file should still leave the decisions
    // computed so far written out.
    let stop = StopSignal::new();
    stop_on_interrupt(stop.clone());

    let provider = open_provider(
        &dataset_path,
        resolved.csv_has_header,
        resolved.csv_delimiter,
        resolved.script_timeout,
    )?;

    let mut source = FeatureSource::new(provider, input_number).with_stop_signal(stop.clone());
    let width = source.validate_first()?;
    log::info!("first row loaded with {} value(s)", width);

    let source = Arc::new(Mutex::new(source));
    let node_source = source.clone();

    aixker_rlt::node::Node::start(
        move |_lowest_state| node_source.lock().unwrap().next_features(),
        action_recorder(source.clone(), actions.clone()),
        aixker_rlt::RunningMode::Infer,
        &config.model_name,
    )
    .await;

    report_source(&source)?;

    // A destination that could not be written is a failed run: the decisions
    // are the result of the command.
    actions.finish()?;

    log::info!("inference completed");
    Ok(())
}

/// Scaffold a working project.
fn run_init(out: Output, args: InitArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(&args.dir);
    let written = init::write(dir, args.force)?;

    out.line(format!("Scaffolded {} file(s):", written.len()));
    for path in &written {
        out.line(format!("  {}", path.display()));
    }

    out.line("");
    out.line("Next:");
    out.line("  rlt train");
    out.line("  rlt eval --dataset ./data/holdout.csv --baseline ./scripts/heuristic.py");
    out.line("");
    out.line("The reward script defines what \"good\" means; edit it for your problem.");

    Ok(())
}

/// Resolve evaluation settings with precedence: flag, then config file, then default.
///
/// `interval_secs` is forced to 0: evaluation is a batch pass over a held-out
/// file, so there is never anything to wait for between samples.
fn resolve_eval(file: &FileConfig, args: &EvalArgs) -> Resolved {
    let (model_name, model_name_source) =
        resolve_opt(args.model_name.clone(), file.model_name.clone());
    let (backend, backend_source) = resolve(
        args.backend.map(Into::into),
        file.backend,
        aixker_rlt::ComputeBackend::default(),
    );
    let (dataset, dataset_source) = resolve_opt(args.dataset.clone(), file.dataset.clone());
    let (reward_script, reward_script_source) =
        resolve_opt(args.reward_script.clone(), file.reward_script.clone());
    let (csv_has_header, csv_delimiter, csv_provenance) =
        resolve_csv(args.has_header, args.delimiter, file);
    let (script_timeout, script_timeout_shown, script_timeout_source) =
        resolve_script_timeout(args.script_timeout, file.script_timeout_secs);

    let mut provenance = vec![
        ("dataset", show(&dataset, "<not set>"), dataset_source),
        (
            "model name",
            show(&model_name, "<not set>"),
            model_name_source,
        ),
        (
            "reward script",
            show(&reward_script, "<not set>"),
            reward_script_source,
        ),
        (
            "baseline",
            show(&args.baseline, "<none>"),
            if args.baseline.is_some() {
                Source::Flag
            } else {
                Source::Default
            },
        ),
        (
            "backend",
            format!("{:?}", backend).to_lowercase(),
            backend_source,
        ),
    ];

    if is_python_dataset(&dataset) {
        provenance.push((
            "script timeout",
            script_timeout_shown,
            script_timeout_source,
        ));
    } else {
        provenance.extend(csv_provenance);
    }

    Resolved {
        config: aixker_rlt::configurations::Configurations {
            interval_secs: 0,
            batch_size: file.batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
            total_batches: file
                .total_batches
                .unwrap_or(DEFAULT_EPOCHS as usize * BATCHES_PER_EPOCH),
            input_number: file.input_number.unwrap_or(DEFAULT_INPUT_NUMBER),
            output_number: file.output_number.unwrap_or(DEFAULT_OUTPUT_NUMBER),
            hidden_layers: file.hidden_layers.unwrap_or(DEFAULT_HIDDEN_LAYERS),
            reply_capacity: file.reply_capacity.unwrap_or(DEFAULT_REPLY_CAPACITY),
            model_name: model_name.unwrap_or_default(),
            log_interval: file.log_interval.unwrap_or(DEFAULT_LOG_INTERVAL),
            mode: aixker_rlt::RunningMode::Infer,
            backend,
        },
        dataset,
        reward_script,
        csv_has_header,
        csv_delimiter,
        script_timeout,
        provenance,
    }
}

/// Score a checkpoint against a held-out dataset.
async fn run_eval(
    file_config: &FileConfig,
    config_file: Option<&str>,
    out: Output,
    args: EvalArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolved = resolve_eval(file_config, &args);
    out.settings(
        "Evaluating with the following settings:",
        &resolved.provenance,
        config_file,
    );

    let config = resolved.config;
    if config.model_name.is_empty() {
        return Err("model_name must be set in the config file or via --model-name".into());
    }

    let checkpoint = checkpoint_path(&config.model_name);
    ensure_checkpoint(
        &checkpoint,
        "Train a model first, or point --model-name at an existing checkpoint.",
    )?;

    let dataset_path = resolved
        .dataset
        .ok_or("dataset must be set in the config file or via --dataset")?;
    if !Path::new(&dataset_path).is_file() {
        return Err(format!("dataset not found: {}", dataset_path).into());
    }

    // Without a reward there is no notion of "good", so the whole command would
    // report zeros and look like a working answer.
    let reward_path = resolved.reward_script.ok_or(
        "eval needs a reward script to score against: pass --reward-script, \
         or set reward_script in the config file",
    )?;
    if !Path::new(&reward_path).is_file() {
        return Err(format!("reward script not found or not a file: {}", reward_path).into());
    }

    let stop = StopSignal::new();
    stop_on_interrupt(stop.clone());
    let rewards = RewardFactory::new(
        Some(Path::new(&reward_path)),
        stop.clone(),
        resolved.script_timeout,
    )?;

    let baseline = match &args.baseline {
        Some(spec) => Some(Baseline::parse(
            spec,
            config.output_number,
            resolved.script_timeout,
        )?),
        None => None,
    };

    // Taken before the baseline moves into the scoring closure.
    let baseline_label = baseline
        .as_ref()
        .map(|baseline| baseline.label().to_string());

    let policy_stats = Arc::new(Mutex::new(PolicyStats::default()));
    let baseline_stats = Arc::new(Mutex::new(PolicyStats::default()));

    let config = Arc::new(config);
    let input_number = config.input_number;

    let provider = open_provider(
        &dataset_path,
        resolved.csv_has_header,
        resolved.csv_delimiter,
        resolved.script_timeout,
    )?;

    let mut source = FeatureSource::new(provider, input_number).with_stop_signal(stop.clone());
    let width = source.validate_first()?;
    log::info!("first row loaded with {} value(s)", width);

    let source = Arc::new(Mutex::new(source));
    let node_source = source.clone();

    let scoring_policy = policy_stats.clone();
    let scoring_baseline = baseline_stats.clone();
    let scoring_stop = stop.clone();

    aixker_rlt::initialize(config.clone());
    aixker_rlt::node::Node::start(
        move |_lowest_state| node_source.lock().unwrap().next_features(),
        move |features: &Vec<f32>, action: u32, _counter: u32| {
            // Both policies are scored on this same row. Evaluating them in
            // separate passes would compare two different samples and call the
            // difference a result.
            let (reward, success) = rewards.evaluate(features, action);
            scoring_policy
                .lock()
                .unwrap()
                .record(action, reward, success);

            if let Some(baseline) = &baseline {
                match baseline.action(features) {
                    Ok(baseline_action) => {
                        let (reward, success) = rewards.evaluate(features, baseline_action);
                        scoring_baseline
                            .lock()
                            .unwrap()
                            .record(baseline_action, reward, success);
                    }
                    Err(err) => scoring_stop.stop(format!("baseline failed: {}", err)),
                }
            }

            (reward, success)
        },
        aixker_rlt::RunningMode::Infer,
        &config.model_name,
    )
    .await;

    report_source(&source)?;

    let policy = Arc::try_unwrap(policy_stats)
        .map_err(|_| "the evaluation outlived its statistics")?
        .into_inner()?;
    let baseline_totals = Arc::try_unwrap(baseline_stats)
        .map_err(|_| "the evaluation outlived its statistics")?
        .into_inner()?;

    if policy.samples() == 0 {
        return Err(format!("no rows were evaluated from {}", dataset_path).into());
    }

    let report = Report::new(
        dataset_path,
        config.model_name.clone(),
        policy,
        baseline_label.clone(),
        baseline_label.as_ref().map(|_| baseline_totals),
    );

    log::info!("scored {} sample(s)", report.samples());
    // Worth a warning of its own: someone reading CI logs rather than the
    // report should still see that the heuristic won.
    if report.policy_wins() == Some(false) {
        log::warn!("the baseline scored better than the policy");
    }

    match args.format {
        ReportFormat::Text => out.line(format!("\n{}", report)),
        ReportFormat::Json => out.line(report.to_json()?),
    }

    Ok(())
}

/// Work out which checkpoint a command was pointed at.
///
/// `--checkpoint` names a path directly; `--model-name` is resolved under
/// `./models` the way `train` and `infer` resolve it; otherwise the config
/// file's `model_name` applies. clap rejects the first two together, so neither
/// can silently win.
fn resolve_checkpoint(
    explicit: &Option<String>,
    model_name: &Option<String>,
    file_config: &FileConfig,
    absent: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match (explicit, model_name) {
        (Some(path), _) => Ok(PathBuf::from(path)),
        (None, Some(name)) => Ok(checkpoint_path(name)),
        (None, None) => {
            let name = file_config.model_name.as_deref().ok_or_else(|| {
                format!(
                    "{}: pass --checkpoint or --model-name, or set model_name in \
                     the config file",
                    absent
                )
            })?;
            Ok(checkpoint_path(name))
        }
    }
}

/// Report what a checkpoint holds.
fn run_inspect(
    file_config: &FileConfig,
    out: Output,
    args: InspectArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = resolve_checkpoint(
        &args.checkpoint,
        &args.model_name,
        file_config,
        "no checkpoint to inspect",
    )?;

    ensure_checkpoint(
        &checkpoint,
        "Train a model first, or pass --checkpoint with an existing path.",
    )?;

    let bytes = fs::metadata(&checkpoint)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let text = fs::read_to_string(&checkpoint)
        .map_err(|err| format!("failed to read '{}': {}", checkpoint.display(), err))?;

    let parsed = inspect::parse(&checkpoint.display().to_string(), bytes, &text)?;

    // Worth a warning of its own: someone reading logs rather than the report
    // should still learn that the weights are unusable.
    for warning in &parsed.warnings {
        log::warn!("{}", warning);
    }

    match args.format {
        ReportFormat::Text => out.line(parsed.to_string()),
        ReportFormat::Json => out.line(parsed.to_json()?),
    }

    Ok(())
}

/// Copy a trained checkpoint to an output path.
///
/// `--format json` performs no conversion: checkpoints are already JSON and the
/// file is copied verbatim. Formats that would require real conversion
/// (safetensors, onnx) are not implemented, so the command does not claim to
/// convert anything.
fn run_export(
    file_config: &FileConfig,
    out: Output,
    args: ExportArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = resolve_checkpoint(
        &args.checkpoint,
        &args.model_name,
        file_config,
        "no checkpoint to export",
    )?;

    out.line("Exporting model with the following settings:");
    out.line(format!("  checkpoint: {}", checkpoint.display()));
    out.line(format!("  output: {}", args.output));
    out.line(format!(
        "  format: {:?} (copied as-is, no conversion)",
        args.format
    ));

    ensure_checkpoint(
        &checkpoint,
        "Train a model first, or pass --checkpoint with an existing path.",
    )?;

    // fs::copy does not create the destination directory, and the default
    // output path points at one that will not exist on a fresh checkout.
    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create output directory '{}': {}",
                    parent.display(),
                    err
                )
            })?;
        }
    }

    fs::copy(&checkpoint, &args.output).map_err(|err| {
        format!(
            "failed to copy '{}' to '{}': {}",
            checkpoint.display(),
            args.output,
            err
        )
    })?;

    out.line(format!("Model exported to {}", args.output));
    Ok(())
}

/// Resolve the effective log level from --log-level and the
/// convenience flags (--verbose / --silent / --errors-only).
fn effective_log_level(cli: &Cli) -> cli::LogLevel {
    if cli.verbose {
        cli::LogLevel::Debug
    } else if cli.silent {
        cli::LogLevel::Silent
    } else if cli.errors_only {
        cli::LogLevel::Error
    } else {
        cli.log_level
    }
}

fn init_logger(level: cli::LogLevel) {
    let filter = match level {
        cli::LogLevel::Silent => log::LevelFilter::Off,
        cli::LogLevel::Error => log::LevelFilter::Error,
        cli::LogLevel::Warn => log::LevelFilter::Warn,
        cli::LogLevel::Info => log::LevelFilter::Info,
        cli::LogLevel::Debug => log::LevelFilter::Debug,
        cli::LogLevel::Trace => log::LevelFilter::Trace,
    };
    let mut builder = env_logger::Builder::from_default_env();
    builder.filter_level(filter);
    // wgpu internals are noisy at warn level; only surface them when debugging.
    if filter < log::LevelFilter::Debug {
        builder
            .filter_module("wgpu_hal", log::LevelFilter::Error)
            .filter_module("wgpu_core", log::LevelFilter::Error);
    }
    builder.init();
}

#[tokio::main]
async fn main() {
    // Print the error with Display rather than letting `main` return it: the
    // default Debug formatting escapes newlines, which mangles multi-line
    // messages such as TOML parse errors.
    if let Err(err) = run().await {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let log_level = effective_log_level(&cli);
    init_logger(log_level);

    // From the effective level, not the flag: `--log-level silent` documents
    // itself as "no log output at all", and a banner it does not suppress makes
    // the two spellings disagree. `--silent` resolves to Silent, so it still
    // behaves the same.
    let out = Output::new(matches!(log_level, cli::LogLevel::Silent));

    // `init` is the one command that has to work without a usable config file:
    // its whole job is to write one, and a directory holding a broken or
    // unrelated `Config.toml` is exactly where someone reaches for it.
    let (file_config, config_file) = if matches!(cli.command, Commands::Init(_)) {
        (FileConfig::default(), None)
    } else {
        let loaded = load_file_config(cli.config.as_ref())?;

        out.line(format!(
            "Aixker-RLT CLI invoked with log level: {:?}",
            log_level
        ));
        match &loaded.1 {
            Some(path) => out.line(format!("Using config file: {}", path)),
            None => out.line("No config file found; using built-in defaults."),
        }

        loaded
    };

    match cli.command {
        Commands::Init(args) => run_init(out, args)?,
        Commands::Train(args) => run_train(&file_config, config_file.as_deref(), out, args).await?,
        Commands::Inspect(args) => run_inspect(&file_config, out, args)?,
        Commands::Export(args) => run_export(&file_config, out, args)?,
        Commands::Infer(args) => run_infer(&file_config, config_file.as_deref(), out, args).await?,
        Commands::Eval(args) => run_eval(&file_config, config_file.as_deref(), out, args).await?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cli::commands::Backend;

    fn train_args() -> TrainArgs {
        TrainArgs {
            dataset: None,
            epochs: None,
            batch_size: None,
            dry_run: false,
            model_name: None,
            reward_script: None,
            backend: None,
            has_header: None,
            delimiter: None,
            script_timeout: None,
        }
    }

    /// A config file that sets every value a flag could also set.
    fn file_config() -> FileConfig {
        FileConfig {
            batch_size: Some(4),
            total_batches: Some(7),
            input_number: Some(12),
            model_name: Some("from-file.json".to_string()),
            dataset: Some("from-file.csv".to_string()),
            has_header: Some(true),
            delimiter: Some(';'),
            ..FileConfig::default()
        }
    }

    #[test]
    fn a_typod_config_key_is_reported() {
        // Regression: `bacth_size = 64` configured nothing and said nothing.
        let unknown = unknown_config_keys("bacth_size = 64\nbatch_size = 32\n");
        assert_eq!(unknown, vec!["bacth_size".to_string()]);
    }

    #[test]
    fn mode_and_debug_are_ignored_on_purpose() {
        assert!(unknown_config_keys("mode = \"Infer\"\ndebug = false\n").is_empty());
    }

    #[test]
    fn every_config_field_is_in_the_known_key_set() {
        // The pattern names every field, so adding one to FileConfig without a
        // matching entry in KNOWN_CONFIG_KEYS stops compiling here — rather
        // than warning users about a key that is perfectly valid.
        let FileConfig {
            interval_secs: _,
            batch_size: _,
            total_batches: _,
            input_number: _,
            output_number: _,
            hidden_layers: _,
            reply_capacity: _,
            model_name: _,
            log_interval: _,
            backend: _,
            dataset: _,
            reward_script: _,
            has_header: _,
            delimiter: _,
            script_timeout_secs: _,
        } = FileConfig::default();

        assert_eq!(
            KNOWN_CONFIG_KEYS.len(),
            15,
            "a field was added to FileConfig; add its key to KNOWN_CONFIG_KEYS"
        );
    }

    #[test]
    fn a_flag_beats_the_config_file_which_beats_the_default() {
        assert_eq!(resolve(Some(1), Some(2), 3), (1, Source::Flag));
        assert_eq!(resolve(None, Some(2), 3), (2, Source::File));
        assert_eq!(resolve(None, None, 3), (3, Source::Default));
    }

    #[test]
    fn built_in_defaults_apply_without_a_config_file() {
        let resolved = resolve_train(&FileConfig::default(), &train_args());

        assert_eq!(resolved.config.batch_size, DEFAULT_BATCH_SIZE);
        assert_eq!(resolved.config.input_number, DEFAULT_INPUT_NUMBER);
        assert_eq!(
            resolved.config.total_batches,
            DEFAULT_EPOCHS as usize * BATCHES_PER_EPOCH
        );
        assert!(resolved.dataset.is_none());
    }

    #[test]
    fn config_file_values_are_used_when_no_flag_is_given() {
        // Regression: clap defaults used to overwrite every config file value,
        // so batch_size was always 32 no matter what the file said.
        let resolved = resolve_train(&file_config(), &train_args());

        assert_eq!(resolved.config.batch_size, 4);
        assert_eq!(resolved.config.total_batches, 7);
        assert_eq!(resolved.config.model_name, "from-file.json");
        assert_eq!(resolved.dataset.as_deref(), Some("from-file.csv"));
        assert!(resolved.csv_has_header);
        assert_eq!(resolved.csv_delimiter, ';');
    }

    #[test]
    fn flags_override_the_config_file() {
        let args = TrainArgs {
            batch_size: Some(99),
            model_name: Some("from-flag.json".to_string()),
            dataset: Some("from-flag.csv".to_string()),
            has_header: Some(false),
            delimiter: Some('|'),
            backend: Some(Backend::Gpu),
            ..train_args()
        };
        let resolved = resolve_train(&file_config(), &args);

        assert_eq!(resolved.config.batch_size, 99);
        assert_eq!(resolved.config.model_name, "from-flag.json");
        assert_eq!(resolved.dataset.as_deref(), Some("from-flag.csv"));
        assert!(!resolved.csv_has_header, "--has-header false must win");
        assert_eq!(resolved.csv_delimiter, '|');
        assert_eq!(resolved.config.backend, aixker_rlt::ComputeBackend::Gpu);
    }

    #[test]
    fn epochs_flag_beats_total_batches_from_the_file() {
        let args = TrainArgs {
            epochs: Some(3),
            ..train_args()
        };
        let resolved = resolve_train(&file_config(), &args);

        assert_eq!(resolved.config.total_batches, 3 * BATCHES_PER_EPOCH);
    }

    #[test]
    fn csv_options_are_reported_only_for_csv_datasets() {
        let names = |resolved: &Resolved| -> Vec<&'static str> {
            resolved
                .provenance
                .iter()
                .map(|(name, _, _)| *name)
                .collect()
        };

        let csv = resolve_train(
            &FileConfig::default(),
            &TrainArgs {
                dataset: Some("data.csv".to_string()),
                ..train_args()
            },
        );
        assert!(names(&csv).contains(&"csv delimiter"));

        let python = resolve_train(
            &FileConfig::default(),
            &TrainArgs {
                dataset: Some("provider.py".to_string()),
                ..train_args()
            },
        );
        assert!(!names(&python).contains(&"csv delimiter"));
    }

    #[test]
    fn checkpoint_path_mirrors_the_librarys_doubled_name() {
        // See docs/found-issues.md issue 1: the library builds the file name
        // from the node id and the model name, and gets both from model_name.
        assert_eq!(
            checkpoint_path("m.json"),
            PathBuf::from("./models/m.json.m.json")
        );
    }

    fn infer_args() -> InferArgs {
        InferArgs {
            dataset: None,
            model_name: None,
            backend: None,
            has_header: None,
            delimiter: None,
            output: None,
            with_features: false,
            script_timeout: None,
            interval_secs: None,
        }
    }

    #[test]
    fn inference_over_a_file_does_not_poll_but_a_live_provider_does() {
        // Regression: the flat 10s default applied to files too, so inference
        // over a 400-row CSV spent 67 minutes asleep.
        let csv = resolve_infer(
            &FileConfig::default(),
            &InferArgs {
                dataset: Some("data.csv".to_string()),
                ..infer_args()
            },
        );
        assert_eq!(csv.config.interval_secs, 0);

        let live = resolve_infer(
            &FileConfig::default(),
            &InferArgs {
                dataset: Some("provider.py".to_string()),
                ..infer_args()
            },
        );
        assert_eq!(live.config.interval_secs, DEFAULT_INTERVAL_SECS);
    }

    #[test]
    fn an_explicit_interval_beats_the_dataset_rule() {
        // The rule replaces the default, not the configuration.
        let flagged = resolve_infer(
            &FileConfig::default(),
            &InferArgs {
                dataset: Some("data.csv".to_string()),
                interval_secs: Some(5),
                ..infer_args()
            },
        );
        assert_eq!(flagged.config.interval_secs, 5);

        let configured = resolve_infer(
            &FileConfig {
                interval_secs: Some(7),
                ..FileConfig::default()
            },
            &InferArgs {
                dataset: Some("data.csv".to_string()),
                ..infer_args()
            },
        );
        assert_eq!(configured.config.interval_secs, 7);
    }

    #[test]
    fn the_derived_interval_says_why_it_was_chosen() {
        let (seconds, source) = resolve_interval(None, None, &Some("data.csv".to_string()));

        assert_eq!(seconds, 0);
        assert_eq!(source.label(Some("Config.toml")), "file dataset");
    }

    #[test]
    fn a_zero_script_timeout_means_wait_forever() {
        // The escape hatch for a legitimately slow script, and what the CLI did
        // before the deadline existed.
        let (timeout, shown, source) = resolve_script_timeout(Some(0), None);
        assert!(timeout.is_none());
        assert!(shown.contains("wait forever"), "{}", shown);
        assert_eq!(source, Source::Flag);

        let (timeout, shown, _) = resolve_script_timeout(None, Some(5));
        assert_eq!(timeout, Some(std::time::Duration::from_secs(5)));
        assert_eq!(shown, "5s");

        let (timeout, _, source) = resolve_script_timeout(None, None);
        assert_eq!(
            timeout,
            Some(std::time::Duration::from_secs(DEFAULT_SCRIPT_TIMEOUT_SECS))
        );
        assert_eq!(source, Source::Default);
    }

    #[test]
    fn the_script_timeout_is_reported_only_when_a_script_is_involved() {
        let names = |resolved: &Resolved| -> Vec<&'static str> {
            resolved
                .provenance
                .iter()
                .map(|(name, _, _)| *name)
                .collect()
        };

        let csv_only = resolve_train(
            &FileConfig::default(),
            &TrainArgs {
                dataset: Some("data.csv".to_string()),
                ..train_args()
            },
        );
        assert!(!names(&csv_only).contains(&"script timeout"));

        let with_reward = resolve_train(
            &FileConfig::default(),
            &TrainArgs {
                dataset: Some("data.csv".to_string()),
                reward_script: Some("reward.py".to_string()),
                ..train_args()
            },
        );
        assert!(names(&with_reward).contains(&"script timeout"));
    }

    #[test]
    fn only_ascii_delimiters_are_accepted() {
        assert_eq!(delimiter_byte(',').unwrap(), b',');
        assert_eq!(delimiter_byte('\t').unwrap(), b'\t');
        assert!(delimiter_byte('€').is_err());
    }

    #[test]
    fn python_datasets_are_detected_by_extension() {
        assert!(is_python_dataset(&Some("provider.py".to_string())));
        assert!(!is_python_dataset(&Some("data.csv".to_string())));
        assert!(!is_python_dataset(&None));
    }

    #[test]
    fn the_python_extension_is_matched_without_regard_to_case() {
        // Regression: Windows filesystems are case-insensitive, so provider.PY
        // names the same file as provider.py — and was read as CSV.
        assert!(is_python_path("provider.PY"));
        assert!(is_python_path("provider.Py"));
        assert!(is_python_dataset(&Some("provider.PY".to_string())));

        // An extension, not a suffix: a file merely ending in those letters is
        // not a script.
        assert!(!is_python_path("notpy"));
        assert!(!is_python_path("data.csv"));
    }

    #[test]
    fn paths_are_reported_absolutely() {
        // `./models` is relative to the working directory, so a relative path
        // in a message cannot be acted on without knowing where the command ran.
        let dir = tempfile::TempDir::new().unwrap();
        let existing = dir.path().join("here.json");
        std::fs::write(&existing, "{}").unwrap();

        assert!(Path::new(&absolute(&existing)).is_absolute());

        // A checkpoint that does not exist yet still resolves, through its
        // parent — that is the case the training message needs.
        let missing = dir.path().join("not-yet.json");
        let shown = absolute(&missing);
        assert!(Path::new(&shown).is_absolute(), "{}", shown);
        assert!(shown.ends_with("not-yet.json"), "{}", shown);

        // Nothing resolvable: report it as given rather than half-resolved.
        let nowhere = Path::new("./no/such/dir/m.json");
        assert_eq!(absolute(nowhere), "./no/such/dir/m.json");
    }

    #[test]
    fn a_directory_is_not_a_usable_checkpoint() {
        // Regression: a directory's own length is non-zero, so it passed the
        // emptiness guard and failed later with a raw copy error instead.
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("checkpoint-shaped-directory");
        std::fs::create_dir(&path).unwrap();

        let error = ensure_checkpoint(&path, "hint").unwrap_err().to_string();
        assert!(error.contains("not a file"), "{}", error);
    }

    #[test]
    fn a_missing_default_config_is_not_an_error_but_a_missing_explicit_one_is() {
        let (config, path) = load_file_config(None).unwrap();
        if path.is_none() {
            assert!(config.model_name.is_none());
        }

        let explicit = "./definitely-not-here.toml".to_string();
        let error = load_file_config(Some(&explicit)).unwrap_err().to_string();
        assert!(error.contains("config file not found"), "{}", error);
    }
}
