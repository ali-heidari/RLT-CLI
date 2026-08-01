//! The one way a run is asked to end early.
//!
//! The library ends its loop only when the input closure returns an empty
//! vector, so anything else that needs to stop a run — a dead reward script
//! today, Ctrl-C once that lands — has to say so through the data source.
//! This is that channel: whoever notices the problem calls
//! [`StopSignal::stop`], and [`crate::providers::feature_source::FeatureSource`]
//! turns it into the empty vector the library understands.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A shared "stop the run" flag.
///
/// A stop is **fatal**: the reason becomes the run's error and the process
/// exits non-zero. Ctrl-C will want a graceful variant that lets the run wind
/// down and keep its checkpoint; that belongs here too when it lands.
#[derive(Default)]
pub struct StopSignal {
    stopped: AtomicBool,
    /// The first reason given. A failure that repeats per step — a worker whose
    /// pipe is closed, say — would otherwise overwrite itself with noise.
    reason: Mutex<Option<String>>,
}

impl StopSignal {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Ask the run to stop, keeping the first reason given.
    pub fn stop(&self, reason: String) {
        if let Ok(mut held) = self.reason.lock() {
            if held.is_none() {
                log::error!("stopping the run: {}", reason);
                *held = Some(reason);
            }
        }
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Whether the run has been asked to stop.
    ///
    /// Checked once per sample, so it reads the flag rather than taking the
    /// lock that [`StopSignal::reason`] needs.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Why the run was stopped, if it was.
    pub fn reason(&self) -> Option<String> {
        self.reason.lock().ok().and_then(|reason| reason.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_signal_is_not_stopped() {
        let signal = StopSignal::new();

        assert!(!signal.is_stopped());
        assert!(signal.reason().is_none());
    }

    #[test]
    fn the_first_reason_is_the_one_kept() {
        // A dead worker fails once per step; the first failure is the one that
        // explains the run, the rest are consequences of it.
        let signal = StopSignal::new();

        signal.stop("reward script died".to_string());
        signal.stop("reward script still dead".to_string());

        assert!(signal.is_stopped());
        assert_eq!(signal.reason().unwrap(), "reward script died");
    }
}
