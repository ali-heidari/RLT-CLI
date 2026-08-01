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
/// Stops come in two kinds. A **fatal** stop — a reward script that died —
/// becomes the run's error and exits non-zero. A **graceful** stop — Ctrl-C —
/// ends the run as though the data had simply run out, so the last completed
/// batch's checkpoint survives and the usual totals are reported.
#[derive(Default)]
pub struct StopSignal {
    stopped: AtomicBool,
    /// Whether the stop should be reported as a failure.
    fatal: AtomicBool,
    /// The first reason given. A failure that repeats per step — a worker whose
    /// pipe is closed, say — would otherwise overwrite itself with noise.
    reason: Mutex<Option<String>>,
}

impl StopSignal {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Ask the run to stop and report a failure, keeping the first reason.
    pub fn stop(&self, reason: String) {
        if let Ok(mut held) = self.reason.lock() {
            if held.is_none() {
                log::error!("stopping the run: {}", reason);
                *held = Some(reason);
                self.fatal.store(true, Ordering::SeqCst);
            }
        }
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Ask the run to wind down as though the data source had ended.
    ///
    /// Used for Ctrl-C: the work already done is worth keeping, so this is not
    /// a failure and the process still exits zero.
    pub fn stop_gracefully(&self, reason: String) {
        if let Ok(mut held) = self.reason.lock() {
            if held.is_none() {
                log::warn!("{}", reason);
                *held = Some(reason);
            }
        }
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Whether the stop should be reported as a failure.
    pub fn is_fatal(&self) -> bool {
        self.fatal.load(Ordering::SeqCst)
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
    fn a_graceful_stop_is_not_a_failure() {
        // Ctrl-C should end the run the way an exhausted dataset does, keeping
        // the work already done rather than reporting it as broken.
        let signal = StopSignal::new();
        signal.stop_gracefully("interrupted".to_string());

        assert!(signal.is_stopped());
        assert!(!signal.is_fatal());
        assert_eq!(signal.reason().unwrap(), "interrupted");
    }

    #[test]
    fn a_fatal_stop_is_marked_as_one() {
        let signal = StopSignal::new();
        signal.stop("the reward script died".to_string());

        assert!(signal.is_stopped());
        assert!(signal.is_fatal());
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
