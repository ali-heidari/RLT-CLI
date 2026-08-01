//! A long-lived Python interpreter, shared by the data provider and the reward
//! factory.
//!
//! Both used to run `python3 script.py` once per sample, paying interpreter
//! startup (tens of milliseconds) on every single step. The worker starts the
//! interpreter once and then exchanges newline-delimited JSON over its pipes:
//! one request line in, one response line out.
//!
//! The script must therefore loop over stdin and **flush after every response**,
//! because Python block-buffers stdout when it is a pipe:
//!
//! ```python
//! import json, sys
//! for line in sys.stdin:
//!     request = json.loads(line)
//!     print(json.dumps(handle(request)), flush=True)
//! ```
//!
//! Responses are read on a separate thread. A blocked `read_line` cannot be
//! given a deadline, so a script that reads a request and never answers would
//! otherwise hang the CLI with no output at all.

use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

/// Interpreters tried, in order, when starting a script.
const INTERPRETERS: [&str; 2] = ["python3", "python"];

pub struct PythonWorker {
    script: String,
    child: Child,
    /// `Option` so it can be closed before the process is reaped.
    stdin: Option<ChildStdin>,
    /// Lines the reader thread has taken off the script's stdout.
    responses: Receiver<std::io::Result<String>>,
    /// How long to wait for one response. `None` waits forever.
    timeout: Option<Duration>,
    /// Why the worker can no longer be used, once that is true.
    dead: Option<String>,
}

impl PythonWorker {
    /// Start the interpreter for `script_path`. Done once per run.
    pub fn spawn(script_path: &str, timeout: Option<Duration>) -> Result<Self, Box<dyn Error>> {
        let mut child = spawn_interpreter(script_path)?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("could not open stdin for '{}'", script_path))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| format!("could not open stdout for '{}'", script_path))?;

        if let Some(stderr) = child.stderr.take() {
            spawn_stderr_drain(stderr, script_path.to_string());
        }

        log::debug!("started Python worker for {}", script_path);

        Ok(Self {
            script: script_path.to_string(),
            child,
            stdin: Some(stdin),
            responses: spawn_reader(stdout),
            timeout,
            dead: None,
        })
    }

    /// Send one request line and read one response line.
    ///
    /// The script exiting, answering nothing, or missing its deadline is an
    /// error rather than a hang: a worker that is not answering has to be
    /// reported, not waited on.
    pub fn request(&mut self, payload: &str) -> Result<String, Box<dyn Error>> {
        if let Some(reason) = &self.dead {
            return Err(reason.clone().into());
        }

        let sent = match self.stdin.as_mut() {
            Some(stdin) => writeln!(stdin, "{}", payload).and_then(|_| stdin.flush()),
            None => return Err(format!("Python worker for '{}' is closed", self.script).into()),
        };
        if let Err(err) = sent {
            return Err(self
                .die(&format!("failed to send a request: {}", err))
                .into());
        }

        let received = match self.timeout {
            Some(timeout) => self.responses.recv_timeout(timeout),
            None => self.responses.recv().map_err(RecvTimeoutError::from),
        };

        match received {
            Ok(Ok(line)) => Ok(line.trim().to_string()),
            Ok(Err(err)) => Err(self
                .die(&format!("failed to read a response: {}", err))
                .into()),
            Err(RecvTimeoutError::Timeout) => {
                let waited = self.timeout.unwrap_or_default();
                Err(self
                    .die(&format!(
                        "it stopped answering after {:?}. Raise --script-timeout if the \
                         script is legitimately slow, or set it to 0 to wait forever",
                        waited
                    ))
                    .into())
            }
            Err(RecvTimeoutError::Disconnected) => Err(self
                .die("the script closed its output without answering")
                .into()),
        }
    }

    /// Record why the worker can no longer be used, and build the message.
    ///
    /// A worker that missed a deadline stays dead even if the script recovers:
    /// a late answer would be paired with the *next* request, quietly
    /// mismatching every feature vector and reward after it.
    fn die(&mut self, problem: &str) -> String {
        let message = self.describe_death(problem);
        if self.dead.is_none() {
            self.dead = Some(message.clone());
        }
        message
    }

    /// Build an error message, adding the exit status when the process is gone.
    fn describe_death(&mut self, problem: &str) -> String {
        match self.child.try_wait() {
            Ok(Some(status)) => format!(
                "Python script '{}' exited ({}): {}. Check its stderr above; \
                 the script must loop over stdin and print one JSON line per \
                 request with flush=True.",
                self.script, status, problem
            ),
            _ => format!("Python script '{}': {}", self.script, problem),
        }
    }
}

impl Drop for PythonWorker {
    fn drop(&mut self) {
        // Closing stdin lets a script reading `for line in sys.stdin` finish on
        // its own; kill whatever is still running so no interpreter is leaked.
        // Killing it also closes the pipe, which ends the reader thread.
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Pull lines off the script's stdout until it stops producing them.
///
/// This thread is what lets [`PythonWorker::request`] wait with a deadline: a
/// `read_line` already blocked on a pipe cannot be interrupted.
fn spawn_reader(stdout: ChildStdout) -> Receiver<std::io::Result<String>> {
    let (sender, receiver) = mpsc::channel();

    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                // End of output. Dropping the sender is what reports it.
                Ok(0) => break,
                Ok(_) => {
                    if sender.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    let _ = sender.send(Err(err));
                    break;
                }
            }
        }
    });

    receiver
}

/// Forward the script's stderr to the log facade, one line at a time.
///
/// Inheriting the stream let a traceback — or a stray
/// `print(..., file=sys.stderr)` — bypass every filter the CLI configures,
/// including `--silent`, whose contract is zero bytes on both streams. Piping
/// it without reading would deadlock a script that writes more than a pipe
/// buffer, so the drain gets its own thread and runs for the worker's lifetime.
fn spawn_stderr_drain(stderr: ChildStderr, script: String) {
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            match line {
                Ok(line) => log::error!("{}: {}", script, line),
                // The pipe closed, or the script emitted something that is not
                // UTF-8. Either way there is nothing further to forward.
                Err(_) => break,
            }
        }
    });
}

/// Start the first interpreter on PATH that exists.
fn spawn_interpreter(script_path: &str) -> Result<Child, Box<dyn Error>> {
    let mut last_error = None;

    for interpreter in INTERPRETERS {
        let result = Command::new(interpreter)
            // -u forces unbuffered stdout. Without it a script that forgets
            // flush=True leaves its response in a pipe buffer and the CLI waits
            // for a line that has already been written.
            .arg("-u")
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Piped and drained by `spawn_stderr_drain`, so the script's
            // diagnostics obey the CLI's log level like everything else.
            .stderr(Stdio::piped())
            .spawn();

        match result {
            Ok(child) => return Ok(child),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => last_error = Some(err),
            Err(err) => {
                return Err(format!(
                    "failed to start '{}' for '{}': {}",
                    interpreter, script_path, err
                )
                .into())
            }
        }
    }

    Err(format!(
        "no Python interpreter found on PATH (tried {}): {}",
        INTERPRETERS.join(", "),
        last_error
            .map(|err| err.to_string())
            .unwrap_or_else(|| "not found".to_string())
    )
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a script to a temporary directory and start a worker on it.
    ///
    /// The directory is returned so it outlives the worker.
    fn worker(body: &str, timeout: Option<Duration>) -> (tempfile::TempDir, PythonWorker) {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("script.py");
        std::fs::write(&script, body).unwrap();

        let worker = PythonWorker::spawn(script.to_str().unwrap(), timeout).unwrap();
        (dir, worker)
    }

    #[test]
    fn a_responsive_script_is_answered_within_the_deadline() {
        let (_dir, mut worker) = worker(
            "import sys\nfor line in sys.stdin:\n    print('[1, 2]', flush=True)\n",
            Some(Duration::from_secs(10)),
        );

        assert_eq!(worker.request("{}").unwrap(), "[1, 2]");
    }

    #[test]
    fn a_script_that_never_answers_times_out() {
        // Regression: this blocked in read_line forever, with no output and no
        // way to tell it apart from a slow run.
        let (_dir, mut worker) = worker(
            "import sys, time\nfor line in sys.stdin:\n    time.sleep(60)\n",
            Some(Duration::from_millis(200)),
        );

        let error = worker.request("{}").unwrap_err().to_string();
        assert!(error.contains("stopped answering"), "{}", error);
    }

    #[test]
    fn a_worker_that_missed_its_deadline_stays_dead() {
        // A late answer would be paired with the next request, quietly
        // mismatching every feature vector and reward after it.
        let (_dir, mut worker) = worker(
            "import sys, time\n\
             for line in sys.stdin:\n\
             \x20   time.sleep(0.5)\n\
             \x20   print('[1, 2]', flush=True)\n",
            Some(Duration::from_millis(100)),
        );

        assert!(worker.request("{}").is_err());
        thread::sleep(Duration::from_millis(700));

        let error = worker.request("{}").unwrap_err().to_string();
        assert!(error.contains("stopped answering"), "{}", error);
    }
}
