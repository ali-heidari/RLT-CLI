//! Reward computation, optionally delegated to a Python script.
//!
//! The script is started once and asked for a reward per environment step over
//! its stdin / stdout pipes; see [`crate::providers::python_worker`] for the
//! protocol. Each request line is a JSON object with `features` and `action`,
//! and each response line a JSON object with `reward` and `success`.

use crate::providers::python_worker::PythonWorker;
use crate::stop_signal::StopSignal;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Consecutive reward failures tolerated before the run is abandoned.
///
/// Mirrors the data provider's threshold. One bad answer should not kill a long
/// run, but a script that fails over and over trains every remaining step on a
/// fabricated zero reward and still writes a checkpoint that looks legitimate.
const MAX_CONSECUTIVE_ERRORS: usize = 10;

#[derive(Serialize)]
struct RewardRequest {
    features: Vec<f32>,
    /// `u32`, matching the action the library hands us.
    ///
    /// This was `u8`: an action that did not fit logged a warning and sent
    /// action 0 instead, so with `output_number > 255` every reward would have
    /// been computed for the wrong action. JSON has no opinion about integer
    /// width, so widening costs nothing and removes the failure.
    action: u32,
}

#[derive(Deserialize)]
struct RewardResponse {
    reward: f32,
    success: bool,
}

pub struct RewardFactory {
    /// `Mutex` because the node loop calls `evaluate` through a `Fn` closure,
    /// while talking to the worker needs `&mut`.
    worker: Option<Mutex<PythonWorker>>,
    /// How the run is ended once the script has stopped answering.
    stop: Arc<StopSignal>,
    /// Atomic for the same reason the worker is behind a `Mutex`: `evaluate`
    /// only ever gets `&self`.
    consecutive_errors: AtomicUsize,
}

impl RewardFactory {
    /// Build a factory, starting the reward script if one was configured.
    pub fn new(
        script_path: Option<&Path>,
        stop: Arc<StopSignal>,
        timeout: Option<Duration>,
    ) -> Result<Self, Box<dyn Error>> {
        let worker = match script_path {
            Some(path) => {
                let path = path
                    .to_str()
                    .ok_or_else(|| format!("reward script path is not valid UTF-8: {:?}", path))?;
                Some(Mutex::new(PythonWorker::spawn(path, timeout)?))
            }
            None => None,
        };

        Ok(Self {
            worker,
            stop,
            consecutive_errors: AtomicUsize::new(0),
        })
    }

    /// Score one step.
    pub fn evaluate<F>(&self, features: F, action: u32) -> (f32, bool)
    where
        F: AsRef<[f32]>,
    {
        let Some(worker) = &self.worker else {
            return (0.0, true);
        };

        let request = RewardRequest {
            features: features.as_ref().to_vec(),
            action,
        };

        match ask(worker, &request) {
            Ok(response) => {
                self.consecutive_errors.store(0, Ordering::Relaxed);
                (response.reward, response.success)
            }
            Err(err) => {
                log::error!("reward script failed: {}", err);

                // Returning (0.0, false) forever is how a run used to "succeed"
                // having trained entirely on rewards the script never produced.
                let failures = self.consecutive_errors.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= MAX_CONSECUTIVE_ERRORS {
                    self.stop.stop(format!(
                        "reward script failed {} times in a row, last error: {}",
                        failures, err
                    ));
                }

                (0.0, false)
            }
        }
    }
}

/// One request/response exchange with the reward script.
fn ask(worker: &Mutex<PythonWorker>, request: &RewardRequest) -> Result<RewardResponse, String> {
    let payload = serde_json::to_string(request)
        .map_err(|err| format!("failed to serialize the reward request: {}", err))?;

    let mut worker = worker
        .lock()
        .map_err(|_| "the reward worker is poisoned".to_string())?;

    let response = worker.request(&payload).map_err(|err| err.to_string())?;

    serde_json::from_str(&response).map_err(|err| {
        format!(
            "invalid JSON reward response '{}': {}. Expected {{\"reward\": <number>, \"success\": <bool>}}",
            response, err
        )
    })
}
