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

use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Interpreters tried, in order, when starting a script.
const INTERPRETERS: [&str; 2] = ["python3", "python"];

pub struct PythonWorker {
    script: String,
    child: Child,
    /// `Option` so it can be closed before the process is reaped.
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl PythonWorker {
    /// Start the interpreter for `script_path`. Done once per run.
    pub fn spawn(script_path: &str) -> Result<Self, Box<dyn Error>> {
        let mut child = spawn_interpreter(script_path)?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("could not open stdin for '{}'", script_path))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| format!("could not open stdout for '{}'", script_path))?;

        log::debug!("started Python worker for {}", script_path);

        Ok(Self {
            script: script_path.to_string(),
            child,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
        })
    }

    /// Send one request line and read one response line.
    ///
    /// The script exiting, or answering nothing, is an error rather than a
    /// hang: a dead worker must be reported, not waited on.
    pub fn request(&mut self, payload: &str) -> Result<String, Box<dyn Error>> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| format!("Python worker for '{}' is closed", self.script))?;

        writeln!(stdin, "{}", payload)
            .and_then(|_| stdin.flush())
            .map_err(|err| self.describe_death(&format!("failed to send a request: {}", err)))?;

        let mut line = String::new();
        let read = self
            .stdout
            .read_line(&mut line)
            .map_err(|err| self.describe_death(&format!("failed to read a response: {}", err)))?;

        if read == 0 {
            return Err(self
                .describe_death("the script closed its output without answering")
                .into());
        }

        Ok(line.trim().to_string())
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
        self.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
            // Inherited so tracebacks reach the terminal. Piping stderr without
            // draining it would deadlock a script that writes a lot to it.
            .stderr(Stdio::inherit())
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
