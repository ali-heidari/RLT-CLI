//! Scoring a checkpoint against a held-out dataset, optionally next to a
//! baseline policy.
//!
//! The question this answers is "is this model any good?", and the only answer
//! an operations team acts on is "better than the heuristic we already run".
//! So a baseline is a first-class part of the report rather than an extra.
//!
//! Both policies are scored on the *same* rows in one pass. Evaluating them
//! separately would compare two different samples of the data and call the
//! difference a result.

use crate::providers::python_worker::PythonWorker;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Seed for the `random` baseline.
///
/// Fixed, so that two runs over the same dataset produce the same comparison. A
/// baseline that moved between runs would make every difference unreadable.
const RANDOM_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// What to compare the trained policy against.
enum Kind {
    /// Always the same action — "is your AI even needed?".
    Static(u32),
    /// Cycle through the actions in order.
    RoundRobin,
    /// Uniform choice, seeded so the run repeats.
    Random,
    /// A Python heuristic: the rule the team runs today.
    Script(Mutex<PythonWorker>),
}

/// Request sent to a baseline script.
#[derive(Serialize)]
struct BaselineRequest<'a> {
    features: &'a [f32],
}

/// Response expected from a baseline script.
#[derive(Deserialize)]
struct BaselineResponse {
    action: u32,
}

pub struct Baseline {
    kind: Kind,
    /// How many actions the model can choose between, used to keep a baseline
    /// inside the same action space as the policy.
    outputs: u32,
    /// Cursor for round-robin, PRNG state for random.
    state: AtomicU64,
    label: String,
}

impl Baseline {
    /// Parse a `--baseline` value.
    ///
    /// `static:N`, `round-robin`, `random`, or the path to a Python script.
    pub fn parse(
        spec: &str,
        outputs: usize,
        timeout: Option<Duration>,
    ) -> Result<Self, Box<dyn Error>> {
        let outputs = outputs as u32;

        let kind = if let Some(action) = spec.strip_prefix("static:") {
            let action: u32 = action
                .trim()
                .parse()
                .map_err(|_| format!("'{}' is not a whole number in '{}'", action, spec))?;
            if action >= outputs {
                return Err(format!(
                    "baseline action {} is outside the model's action space (0..{})",
                    action,
                    outputs - 1
                )
                .into());
            }
            Kind::Static(action)
        } else if spec == "round-robin" || spec == "roundrobin" {
            Kind::RoundRobin
        } else if spec == "random" {
            Kind::Random
        } else if crate::is_python_path(spec) {
            if !Path::new(spec).is_file() {
                return Err(format!("baseline script not found or not a file: {}", spec).into());
            }
            Kind::Script(Mutex::new(PythonWorker::spawn(spec, timeout)?))
        } else {
            return Err(format!(
                "unrecognised baseline '{}'. Expected 'static:N', 'round-robin', \
                 'random', or the path to a .py script",
                spec
            )
            .into());
        };

        // The field is a cursor for round-robin and PRNG state for random, so
        // it cannot share one starting value: seeding the cursor with
        // RANDOM_SEED made round-robin begin at `RANDOM_SEED % outputs`.
        let state = match kind {
            Kind::Random => RANDOM_SEED,
            _ => 0,
        };

        Ok(Self {
            kind,
            outputs,
            state: AtomicU64::new(state),
            label: spec.to_string(),
        })
    }

    /// The baseline's action for one sample.
    pub fn action(&self, features: &[f32]) -> Result<u32, Box<dyn Error>> {
        match &self.kind {
            Kind::Static(action) => Ok(*action),
            Kind::RoundRobin => {
                let step = self.state.fetch_add(1, Ordering::Relaxed);
                Ok((step % self.outputs as u64) as u32)
            }
            Kind::Random => Ok((self.next_random() % self.outputs as u64) as u32),
            Kind::Script(worker) => self.ask_script(worker, features),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// xorshift64*, so a run repeats without pulling in an RNG dependency.
    fn next_random(&self) -> u64 {
        let mut x = self.state.load(Ordering::Relaxed);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state.store(x, Ordering::Relaxed);
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn ask_script(
        &self,
        worker: &Mutex<PythonWorker>,
        features: &[f32],
    ) -> Result<u32, Box<dyn Error>> {
        let payload = serde_json::to_string(&BaselineRequest { features })?;

        let mut worker = worker
            .lock()
            .map_err(|_| "the baseline worker is poisoned".to_string())?;
        let response = worker.request(&payload)?;

        let parsed: BaselineResponse = serde_json::from_str(&response).map_err(|err| {
            format!(
                "invalid JSON baseline response '{}': {}. Expected {{\"action\": <number>}}",
                response, err
            )
        })?;

        if parsed.action >= self.outputs {
            return Err(format!(
                "baseline script returned action {}, outside the model's action space (0..{})",
                parsed.action,
                self.outputs - 1
            )
            .into());
        }

        Ok(parsed.action)
    }
}

/// Totals for one action.
#[derive(Default, Clone, Copy, Serialize)]
struct ActionStats {
    count: usize,
    #[serde(skip)]
    reward_total: f64,
    successes: usize,
}

/// What one policy did over the evaluated rows.
#[derive(Default)]
pub struct PolicyStats {
    samples: usize,
    reward_total: f64,
    successes: usize,
    per_action: BTreeMap<u32, ActionStats>,
}

impl PolicyStats {
    pub fn record(&mut self, action: u32, reward: f32, success: bool) {
        let reward = reward as f64;
        self.samples += 1;
        self.reward_total += reward;
        self.successes += usize::from(success);

        let entry = self.per_action.entry(action).or_default();
        entry.count += 1;
        entry.reward_total += reward;
        entry.successes += usize::from(success);
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn mean_reward(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        self.reward_total / self.samples as f64
    }

    /// Share of samples the reward script called a success, as a fraction.
    pub fn success_rate(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        self.successes as f64 / self.samples as f64
    }

    /// One line per action taken, in action order.
    fn action_lines(&self) -> Vec<String> {
        self.per_action
            .iter()
            .map(|(action, stats)| {
                let share = stats.count as f64 / self.samples.max(1) as f64;
                let mean = stats.reward_total / stats.count.max(1) as f64;
                format!(
                    "{}: {} ({:.1}%)  mean reward {:.4}  success {:.1}%",
                    action,
                    stats.count,
                    share * 100.0,
                    mean,
                    stats.successes as f64 / stats.count.max(1) as f64 * 100.0
                )
            })
            .collect()
    }
}

/// The machine-readable form of a report.
#[derive(Serialize)]
pub struct Summary {
    dataset: String,
    samples: usize,
    policy: PolicySummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline: Option<PolicySummary>,
    /// Policy mean reward minus baseline mean reward. Negative means the
    /// baseline won.
    #[serde(skip_serializing_if = "Option::is_none")]
    reward_gain: Option<f64>,
}

#[derive(Serialize)]
struct PolicySummary {
    label: String,
    mean_reward: f64,
    success_rate: f64,
    actions: BTreeMap<u32, ActionStats>,
}

impl PolicySummary {
    fn new(label: String, stats: &PolicyStats) -> Self {
        Self {
            label,
            mean_reward: stats.mean_reward(),
            success_rate: stats.success_rate(),
            actions: stats.per_action.clone(),
        }
    }
}

/// A finished evaluation, ready to print.
pub struct Report {
    dataset: String,
    policy_label: String,
    policy: PolicyStats,
    baseline_label: Option<String>,
    baseline: Option<PolicyStats>,
}

impl Report {
    pub fn new(
        dataset: String,
        policy_label: String,
        policy: PolicyStats,
        baseline_label: Option<String>,
        baseline: Option<PolicyStats>,
    ) -> Self {
        Self {
            dataset,
            policy_label,
            policy,
            baseline_label,
            baseline,
        }
    }

    pub fn samples(&self) -> usize {
        self.policy.samples()
    }

    /// Whether the policy beat the baseline on mean reward.
    ///
    /// `None` when there was no baseline to beat.
    pub fn policy_wins(&self) -> Option<bool> {
        self.baseline
            .as_ref()
            .map(|baseline| self.policy.mean_reward() > baseline.mean_reward())
    }

    pub fn summary(&self) -> Summary {
        Summary {
            dataset: self.dataset.clone(),
            samples: self.policy.samples(),
            policy: PolicySummary::new(self.policy_label.clone(), &self.policy),
            baseline: self
                .baseline
                .as_ref()
                .map(|stats| PolicySummary::new(self.baseline_label.clone().unwrap(), stats)),
            reward_gain: self
                .baseline
                .as_ref()
                .map(|stats| self.policy.mean_reward() - stats.mean_reward()),
        }
    }

    pub fn to_json(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string_pretty(&self.summary())?)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "Evaluated {} sample(s) from {}",
            self.policy.samples(),
            self.dataset
        )?;

        write_policy(formatter, "Policy", &self.policy_label, &self.policy)?;

        let (Some(baseline), Some(label)) = (&self.baseline, &self.baseline_label) else {
            return Ok(());
        };
        write_policy(formatter, "Baseline", label, baseline)?;

        let gain = self.policy.mean_reward() - baseline.mean_reward();
        let success_gain = (self.policy.success_rate() - baseline.success_rate()) * 100.0;

        writeln!(formatter)?;
        writeln!(
            formatter,
            "Difference: {:+.4} mean reward, {:+.1}pp success rate",
            gain, success_gain
        )?;
        writeln!(
            formatter,
            "{}",
            if gain > 0.0 {
                "The policy beats the baseline."
            } else if gain < 0.0 {
                "The baseline beats the policy."
            } else {
                "The policy and the baseline are level."
            }
        )
    }
}

fn write_policy(
    formatter: &mut fmt::Formatter<'_>,
    heading: &str,
    label: &str,
    stats: &PolicyStats,
) -> fmt::Result {
    writeln!(formatter)?;
    writeln!(formatter, "{}: {}", heading, label)?;
    writeln!(formatter, "  mean reward   {:.4}", stats.mean_reward())?;
    writeln!(
        formatter,
        "  success rate  {:.1}%",
        stats.success_rate() * 100.0
    )?;
    for (index, line) in stats.action_lines().iter().enumerate() {
        let prefix = if index == 0 {
            "  actions       "
        } else {
            "                "
        };
        writeln!(formatter, "{}{}", prefix, line)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Baseline` holds a `Mutex<PythonWorker>` and so cannot derive `Debug`,
    /// which `unwrap_err` would need.
    fn parse_error(spec: &str, outputs: usize) -> String {
        match Baseline::parse(spec, outputs, None) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected '{}' to be rejected", spec),
        }
    }

    fn stats(samples: &[(u32, f32, bool)]) -> PolicyStats {
        let mut stats = PolicyStats::default();
        for (action, reward, success) in samples {
            stats.record(*action, *reward, *success);
        }
        stats
    }

    #[test]
    fn a_static_baseline_always_picks_the_same_action() {
        let baseline = Baseline::parse("static:2", 3, None).unwrap();

        assert_eq!(baseline.action(&[0.1]).unwrap(), 2);
        assert_eq!(baseline.action(&[0.9]).unwrap(), 2);
    }

    #[test]
    fn a_static_baseline_outside_the_action_space_is_rejected() {
        // Otherwise the comparison would be against an action the model could
        // never have chosen.
        let error = parse_error("static:5", 3);
        assert!(
            error.contains("outside the model's action space"),
            "{}",
            error
        );
    }

    #[test]
    fn round_robin_cycles_through_every_action() {
        let baseline = Baseline::parse("round-robin", 3, None).unwrap();

        let actions: Vec<u32> = (0..7).map(|_| baseline.action(&[0.0]).unwrap()).collect();
        assert_eq!(actions, vec![0, 1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn the_random_baseline_repeats_across_runs() {
        // A baseline that moved between runs would make every difference
        // unreadable.
        let first: Vec<u32> = {
            let baseline = Baseline::parse("random", 3, None).unwrap();
            (0..10).map(|_| baseline.action(&[0.0]).unwrap()).collect()
        };
        let second: Vec<u32> = {
            let baseline = Baseline::parse("random", 3, None).unwrap();
            (0..10).map(|_| baseline.action(&[0.0]).unwrap()).collect()
        };

        assert_eq!(first, second);
        assert!(first.iter().all(|action| *action < 3));
    }

    #[test]
    fn an_unrecognised_baseline_lists_the_forms_that_work() {
        let error = parse_error("magic", 3);
        assert!(error.contains("static:N"), "{}", error);
        assert!(error.contains("round-robin"), "{}", error);
    }

    #[test]
    fn mean_reward_and_success_rate_are_averaged_over_the_samples() {
        let stats = stats(&[(0, 1.0, true), (1, 0.0, false), (1, 2.0, true)]);

        assert_eq!(stats.samples(), 3);
        assert!((stats.mean_reward() - 1.0).abs() < 1e-9);
        assert!((stats.success_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_evaluation_does_not_divide_by_zero() {
        let stats = PolicyStats::default();

        assert_eq!(stats.mean_reward(), 0.0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn the_report_says_which_policy_won() {
        let better = Report::new(
            "d.csv".to_string(),
            "model".to_string(),
            stats(&[(0, 1.0, true)]),
            Some("static:0".to_string()),
            Some(stats(&[(0, 0.5, true)])),
        );
        assert_eq!(better.policy_wins(), Some(true));
        assert!(better
            .to_string()
            .contains("The policy beats the baseline."));

        let worse = Report::new(
            "d.csv".to_string(),
            "model".to_string(),
            stats(&[(0, 0.1, false)]),
            Some("static:0".to_string()),
            Some(stats(&[(0, 0.5, true)])),
        );
        assert_eq!(worse.policy_wins(), Some(false));
        assert!(worse.to_string().contains("The baseline beats the policy."));
    }

    #[test]
    fn without_a_baseline_there_is_no_verdict() {
        let report = Report::new(
            "d.csv".to_string(),
            "model".to_string(),
            stats(&[(0, 1.0, true)]),
            None,
            None,
        );

        assert_eq!(report.policy_wins(), None);
        assert!(!report.to_string().contains("Difference"));
    }

    #[test]
    fn the_json_summary_carries_the_comparison() {
        let report = Report::new(
            "d.csv".to_string(),
            "model".to_string(),
            stats(&[(0, 1.0, true), (1, 1.0, true)]),
            Some("static:0".to_string()),
            Some(stats(&[(0, 0.25, false), (0, 0.25, true)])),
        );

        let json = report.to_json().unwrap();
        assert!(json.contains("\"reward_gain\": 0.75"), "{}", json);
        assert!(json.contains("\"samples\": 2"), "{}", json);
    }
}
