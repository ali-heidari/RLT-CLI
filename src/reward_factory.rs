//! Reward computation, optionally delegated to a Python script.
//!
//! The script is started once and asked for a reward per environment step over
//! its stdin / stdout pipes; see [`crate::providers::python_worker`] for the
//! protocol. Each request line is a JSON object with `features` and `action`,
//! and each response line a JSON object with `reward` and `success`.

use crate::providers::python_worker::PythonWorker;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::error::Error;
use std::path::Path;
use std::sync::Mutex;

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
    /// `Mutex` because the node loop calls `evaluate` through a `Fn` closure,
    /// while talking to the worker needs `&mut`.
    worker: Option<Mutex<PythonWorker>>,
}

impl RewardFactory {
    /// Build a factory, starting the reward script if one was configured.
    pub fn new(script_path: Option<&Path>) -> Result<Self, Box<dyn Error>> {
        let worker = match script_path {
            Some(path) => {
                let path = path
                    .to_str()
                    .ok_or_else(|| format!("reward script path is not valid UTF-8: {:?}", path))?;
                Some(Mutex::new(PythonWorker::spawn(path)?))
            }
            None => None,
        };

        Ok(Self { worker })
    }

    pub fn evaluate<F, A, R>(&self, features: F, action: A, _prev_reward: R) -> (f32, bool)
    where
        F: AsRef<[f32]>,
        A: TryInto<u8>,
        R: Into<f64>,
        A::Error: std::fmt::Debug,
    {
        let action: u8 = action.try_into().unwrap_or_else(|err| {
            log::warn!("invalid action value {:?}, defaulting to 0", err);
            0
        });
        let _ = _prev_reward.into();

        let Some(worker) = &self.worker else {
            return (0.0, true);
        };

        let request = RewardRequest {
            features: features.as_ref().to_vec(),
            action,
        };

        match ask(worker, &request) {
            Ok(response) => (response.reward, response.success),
            Err(err) => {
                log::error!("reward script failed: {}", err);
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
