//! Where `infer` sends the decisions it computes.
//!
//! The node loop hands every decision to a `Fn` callback, so the destination
//! has to be writable through a shared reference — hence the mutexes. Records
//! are newline-delimited JSON: one object per sample, so the stream can be
//! piped into a consumer as it is produced rather than parsed as one document
//! at the end.

use serde::Serialize;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// The `--output` value that selects stdout.
pub const STDOUT_DESTINATION: &str = "-";

/// One decision, as written to the output.
///
/// `features` is omitted rather than written as `null` so a consumer can test
/// for the key instead of for the value.
#[derive(Serialize)]
struct ActionRecord<'a> {
    row: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    features: Option<&'a [f32]>,
    action: u32,
}

enum Sink {
    /// `--silent` with no explicit destination: the decisions have nowhere to
    /// go that would not break the "zero bytes on both streams" contract.
    Discard,
    /// Rust line-buffers stdout, so each record leaves the process as it is
    /// written and the output can feed a live consumer.
    Stdout,
    File(BufWriter<File>),
}

pub struct ActionWriter {
    /// `Mutex` because the node loop calls the callback through a `Fn`, while
    /// writing needs `&mut`.
    sink: Mutex<Sink>,
    with_features: bool,
    /// The first write failure. Reported by [`ActionWriter::finish`]: the
    /// callback the library calls returns `(f32, bool)` and has no way to
    /// propagate an error at the point it happens.
    failure: Mutex<Option<String>>,
}

impl ActionWriter {
    /// Open the destination. [`STDOUT_DESTINATION`] means stdout.
    pub fn new(
        destination: &str,
        with_features: bool,
        silent: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let sink = match destination {
            STDOUT_DESTINATION if silent => Sink::Discard,
            STDOUT_DESTINATION => Sink::Stdout,
            // A file the user explicitly asked for is the result of the
            // command, not chatter, so `--silent` does not suppress it.
            path => Sink::File(BufWriter::new(create_output_file(path)?)),
        };

        Ok(Self {
            sink: Mutex::new(sink),
            with_features,
            failure: Mutex::new(None),
        })
    }

    /// Write one decision.
    pub fn emit(&self, row: usize, features: &[f32], action: u32) {
        let line = match record_line(row, features, action, self.with_features) {
            Ok(line) => line,
            Err(err) => {
                return self.fail(format!(
                    "failed to serialize the action for row {}: {}",
                    row, err
                ))
            }
        };

        let mut sink = match self.sink.lock() {
            Ok(sink) => sink,
            Err(_) => return self.fail("the action writer is poisoned".to_string()),
        };

        let written = match &mut *sink {
            Sink::Discard => Ok(()),
            Sink::Stdout => writeln!(std::io::stdout(), "{}", line),
            Sink::File(file) => writeln!(file, "{}", line),
        };

        if let Err(err) = written {
            self.fail(format!(
                "failed to write the action for row {}: {}",
                row, err
            ));
        }
    }

    /// Flush the destination and report the first write failure, if any.
    ///
    /// A run whose decisions were only half written must not exit zero: the
    /// output is the whole point of the command.
    pub fn finish(&self) -> Result<(), Box<dyn Error>> {
        if let Ok(mut sink) = self.sink.lock() {
            if let Sink::File(file) = &mut *sink {
                if let Err(err) = file.flush() {
                    self.fail(format!("failed to flush the actions: {}", err));
                }
            }
        }

        match self.failure.lock() {
            Ok(failure) => match &*failure {
                Some(message) => Err(message.clone().into()),
                None => Ok(()),
            },
            Err(_) => Err("the action writer is poisoned".into()),
        }
    }

    /// Keep the first failure and drop the rest, so a broken destination
    /// produces one error line instead of one per sample.
    fn fail(&self, message: String) {
        let Ok(mut failure) = self.failure.lock() else {
            return;
        };
        if failure.is_none() {
            log::error!("{}", message);
            *failure = Some(message);
        }
    }
}

/// Create the output file, and any directory it needs.
///
/// `File::create` does not create parent directories, and `--output` is exactly
/// the flag someone points at a `runs/today/` that does not exist yet.
fn create_output_file(path: &str) -> Result<File, Box<dyn Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create output directory '{}': {}",
                    parent.display(),
                    err
                )
            })?;
        }
    }

    File::create(path)
        .map_err(|err| format!("failed to open '{}' for the actions: {}", path, err).into())
}

/// Render one record. Pure, so the output shape can be asserted directly.
fn record_line(
    row: usize,
    features: &[f32],
    action: u32,
    with_features: bool,
) -> Result<String, serde_json::Error> {
    serde_json::to_string(&ActionRecord {
        row,
        features: with_features.then_some(features),
        action,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_holds_the_row_and_the_action() {
        assert_eq!(
            record_line(1, &[0.5, 1.5], 2, false).unwrap(),
            r#"{"row":1,"action":2}"#
        );
    }

    #[test]
    fn with_features_includes_the_input_row() {
        assert_eq!(
            record_line(7, &[0.5, 1.5], 0, true).unwrap(),
            r#"{"row":7,"features":[0.5,1.5],"action":0}"#
        );
    }

    #[test]
    fn features_are_omitted_rather_than_null() {
        // A consumer should be able to test for the key, not for a null value.
        let line = record_line(1, &[0.5], 2, false).unwrap();
        assert!(!line.contains("features"), "{}", line);
        assert!(!line.contains("null"), "{}", line);
    }

    #[test]
    fn a_file_destination_gets_one_line_per_decision() {
        // Silent, because `--silent --output <path>` must still write the file:
        // "print nothing" is not "do nothing".
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("actions.jsonl");

        let writer = ActionWriter::new(path.to_str().unwrap(), false, true).unwrap();
        writer.emit(1, &[0.5], 2);
        writer.emit(2, &[0.6], 0);
        writer.finish().unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"row\":1,\"action\":2}\n{\"row\":2,\"action\":0}\n"
        );
    }

    #[test]
    fn a_silent_run_without_a_destination_writes_nothing_and_still_succeeds() {
        let writer = ActionWriter::new(STDOUT_DESTINATION, true, true).unwrap();
        writer.emit(1, &[0.5], 2);
        assert!(writer.finish().is_ok());
    }
}
