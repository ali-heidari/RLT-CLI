use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Serialize)]
struct RewardRequest {
    features: Vec<f32>,
    action: u8,
}

#[derive(Deserialize)]
struct RewardResponse {
    reward: f32,
    success: bool,
}

pub struct RewardFactory {
    script_path: Option<PathBuf>,
}

impl RewardFactory {
    pub fn new(script_path: Option<PathBuf>) -> Self {
        Self { script_path }
    }

    pub fn evaluate<F, A, R>(&self, features: F, action: A, _prev_reward: R) -> (f32, bool)
    where
        F: AsRef<[f32]>,
        A: TryInto<u8>,
        R: Into<f64>,
        A::Error: std::fmt::Debug,
    {
        let action: u8 = action
            .try_into()
            .unwrap_or_else(|err| {
                eprintln!("Invalid action value {:?}, defaulting to 0", err);
                0
            });
        let _ = _prev_reward.into();

        if let Some(path) = &self.script_path {
            let request = RewardRequest {
                features: features.as_ref().to_vec(),
                action,
            };
            match self.run_python(path, request) {
                Ok(response) => (response.reward, response.success),
                Err(err) => {
                    eprintln!(
                        "Failed to run reward script '{}': {}",
                        path.display(), err
                    );
                    (0.0, false)
                }
            }
        } else {
            (0.0, true)
        }
    }

    fn run_python(
        &self,
        script_path: &PathBuf,
        request: RewardRequest,
    ) -> Result<RewardResponse, String> {
        let payload = serde_json::to_vec(&request)
            .map_err(|err| format!("Failed to serialize reward request: {}", err))?;

        let mut python = Command::new("python3");
        python
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match python.spawn() {
            Ok(child) => child,
            Err(_) => {
                let mut fallback = Command::new("python");
                fallback
                    .arg(script_path)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                fallback
                    .spawn()
                    .map_err(|err| format!("Python interpreter not found: {}", err))?
            }
        };

        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(&payload)
                .map_err(|err| format!("Failed to write to reward script stdin: {}", err))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|err| format!("Failed to wait for reward script: {}", err))?;

        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }

        serde_json::from_slice(&output.stdout)
            .map_err(|err| format!("Invalid JSON reward response: {}", err))
    }
}
