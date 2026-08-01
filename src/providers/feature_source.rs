//! The boundary between a raw data provider and the training / inference loop.
//!
//! Providers yield `Result<Vec<f64>, _>` rows. The node loop wants one
//! `Vec<f32>` per step and treats an empty vector as "stop". This type performs
//! that conversion, and is where bad data is caught instead of being quietly
//! turned into a row of zeros the model would train on.

use crate::stop_signal::StopSignal;
use std::error::Error;
use std::sync::Arc;

/// A row as produced by a data provider.
pub type Row = Result<Vec<f64>, Box<dyn Error>>;

/// Consecutive provider failures tolerated before a run is abandoned.
///
/// One unparseable line in a large CSV should not kill a long run, but a
/// provider that fails over and over is broken and the run is worthless.
const MAX_CONSECUTIVE_ERRORS: usize = 10;

/// How a completed run consumed its data source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceStats {
    pub rows_read: usize,
    pub rows_skipped: usize,
}

/// Adapts a provider into the feature vectors the node loop consumes.
pub struct FeatureSource<I> {
    rows: I,
    /// Expected feature width. Every row must match it.
    input_number: usize,
    /// Row read by [`FeatureSource::validate_first`], kept for the run.
    pending: Option<Vec<f64>>,
    rows_read: usize,
    rows_skipped: usize,
    consecutive_errors: usize,
    exhausted: bool,
    /// Set when the run must stop and report a failure.
    fatal: Option<String>,
    /// How something outside the data source ends the run. Optional: a source
    /// with nothing to listen to is perfectly valid.
    stop: Option<Arc<StopSignal>>,
}

impl<I> FeatureSource<I>
where
    I: Iterator<Item = Row>,
{
    pub fn new(rows: I, input_number: usize) -> Self {
        Self {
            rows,
            input_number,
            pending: None,
            rows_read: 0,
            rows_skipped: 0,
            consecutive_errors: 0,
            exhausted: false,
            fatal: None,
            stop: None,
        }
    }

    /// Listen to a signal that ends the run from outside the data source.
    pub fn with_stop_signal(mut self, stop: Arc<StopSignal>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Read the first row, check its width, and keep it for the run.
    ///
    /// The row is pushed back rather than dropped: it is real data, and a
    /// header-less CSV would otherwise silently lose its first line.
    /// A width mismatch is reported here so a misconfigured `input_number`
    /// fails before training starts rather than panicking inside the library.
    pub fn validate_first(&mut self) -> Result<usize, Box<dyn Error>> {
        let row = match self.rows.next() {
            None => return Err("data source produced no rows".into()),
            Some(Err(err)) => return Err(format!("failed to read the first row: {}", err).into()),
            Some(Ok(row)) => row,
        };

        let width = row.len();
        if width != self.input_number {
            return Err(self.width_message(width, 1).into());
        }

        self.pending = Some(row);
        Ok(width)
    }

    /// The next feature vector for the node loop.
    ///
    /// Returns an empty vector when the run should stop: either the data source
    /// is exhausted or a fatal error was recorded. Callers must then check
    /// [`FeatureSource::finish`].
    pub fn next_features(&mut self) -> Vec<f32> {
        if self.exhausted || self.fatal.is_some() {
            return Vec::new();
        }

        // Something outside the data source asked the run to end — a reward
        // script that stopped answering, for instance. Reporting it as fatal
        // here is what turns it into a non-zero exit with an explanation.
        if let Some(stop) = &self.stop {
            if stop.is_stopped() {
                self.fatal = Some(
                    stop.reason()
                        .unwrap_or_else(|| "the run was stopped".to_string()),
                );
                return Vec::new();
            }
        }

        if let Some(row) = self.pending.take() {
            return self.accept(row);
        }

        loop {
            match self.rows.next() {
                None => {
                    self.exhausted = true;
                    log::info!(
                        "data source exhausted after {} row(s); stopping",
                        self.rows_read
                    );
                    return Vec::new();
                }
                Some(Ok(row)) => {
                    if row.len() != self.input_number {
                        self.fatal = Some(self.width_message(row.len(), self.row_number()));
                        return Vec::new();
                    }
                    return self.accept(row);
                }
                Some(Err(err)) => {
                    self.rows_skipped += 1;
                    self.consecutive_errors += 1;
                    log::warn!("skipping row {}: {}", self.row_number() - 1, err);

                    if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        self.fatal = Some(format!(
                            "data source failed {} times in a row, last error: {}",
                            self.consecutive_errors, err
                        ));
                        return Vec::new();
                    }
                }
            }
        }
    }

    /// Statistics for the completed run, or the error that stopped it.
    pub fn finish(&self) -> Result<SourceStats, Box<dyn Error>> {
        if let Some(message) = &self.fatal {
            return Err(message.clone().into());
        }
        Ok(SourceStats {
            rows_read: self.rows_read,
            rows_skipped: self.rows_skipped,
        })
    }

    fn accept(&mut self, row: Vec<f64>) -> Vec<f32> {
        self.rows_read += 1;
        self.consecutive_errors = 0;
        row.into_iter().map(|value| value as f32).collect()
    }

    /// 1-based number of the row currently being handled.
    fn row_number(&self) -> usize {
        self.rows_read + self.rows_skipped + 1
    }

    /// 1-based number of the most recently accepted row in the data source.
    ///
    /// Reported alongside each inference decision, so a record can be traced
    /// back to the row that produced it even when rows in between were skipped.
    pub fn last_row_number(&self) -> usize {
        self.rows_read + self.rows_skipped
    }

    fn width_message(&self, got: usize, row_number: usize) -> String {
        format!(
            "feature width mismatch at row {}: input_number is {} but the row has {} value(s). \
             Set input_number = {} in the config file, or fix the data source.",
            row_number, self.input_number, got, got
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(values: &[f64]) -> Row {
        Ok(values.to_vec())
    }

    fn err(message: &str) -> Row {
        Err(message.into())
    }

    fn source(rows: Vec<Row>, input_number: usize) -> FeatureSource<std::vec::IntoIter<Row>> {
        FeatureSource::new(rows.into_iter(), input_number)
    }

    #[test]
    fn validated_first_row_is_still_delivered() {
        // Regression: the old code pulled a row to log its width, then dropped
        // it, so a header-less CSV silently lost its first line.
        let mut source = source(vec![ok(&[1.0, 2.0]), ok(&[3.0, 4.0])], 2);

        assert_eq!(source.validate_first().unwrap(), 2);
        assert_eq!(source.next_features(), vec![1.0, 2.0]);
        assert_eq!(source.next_features(), vec![3.0, 4.0]);
        assert_eq!(source.finish().unwrap().rows_read, 2);
    }

    #[test]
    fn exhaustion_returns_the_stop_signal_not_zeros() {
        // Regression: returning zeros here made training run forever on
        // fabricated all-zero samples.
        let mut source = source(vec![ok(&[1.0, 2.0])], 2);

        assert_eq!(source.next_features(), vec![1.0, 2.0]);
        assert!(source.next_features().is_empty());
        assert!(source.next_features().is_empty(), "stop must be sticky");

        let stats = source.finish().unwrap();
        assert_eq!(stats.rows_read, 1);
        assert_eq!(stats.rows_skipped, 0);
    }

    #[test]
    fn empty_source_is_rejected_upfront() {
        let mut source = source(vec![], 2);
        assert!(source
            .validate_first()
            .unwrap_err()
            .to_string()
            .contains("produced no rows"));
    }

    #[test]
    fn wrong_width_fails_before_the_run_starts() {
        let mut source = source(vec![ok(&[1.0, 2.0, 3.0])], 2);

        let message = source.validate_first().unwrap_err().to_string();
        assert!(message.contains("row 1"), "{}", message);
        assert!(message.contains("input_number is 2"), "{}", message);
        assert!(message.contains("has 3 value(s)"), "{}", message);
    }

    #[test]
    fn wrong_width_mid_stream_stops_the_run_and_reports_the_row() {
        let mut source = source(vec![ok(&[1.0, 2.0]), ok(&[3.0]), ok(&[4.0, 5.0])], 2);

        assert_eq!(source.next_features(), vec![1.0, 2.0]);
        assert!(source.next_features().is_empty());

        let message = source.finish().unwrap_err().to_string();
        assert!(message.contains("row 2"), "{}", message);
    }

    #[test]
    fn failing_rows_are_skipped_and_counted_not_zeroed() {
        // Regression: provider errors used to become vec![0.0; n] and were
        // trained on as if they were real measurements.
        let mut source = source(vec![ok(&[1.0, 2.0]), err("bad line"), ok(&[3.0, 4.0])], 2);

        assert_eq!(source.next_features(), vec![1.0, 2.0]);
        assert_eq!(source.next_features(), vec![3.0, 4.0]);
        assert!(source.next_features().is_empty());

        let stats = source.finish().unwrap();
        assert_eq!(stats.rows_read, 2);
        assert_eq!(stats.rows_skipped, 1);
    }

    #[test]
    fn a_persistently_failing_source_is_abandoned() {
        let rows = (0..MAX_CONSECUTIVE_ERRORS + 5)
            .map(|_| err("provider is down"))
            .collect();
        let mut source = source(rows, 2);

        assert!(source.next_features().is_empty());

        let message = source.finish().unwrap_err().to_string();
        assert!(
            message.contains(&format!("failed {} times in a row", MAX_CONSECUTIVE_ERRORS)),
            "{}",
            message
        );
    }

    #[test]
    fn a_stop_signal_ends_the_run_and_explains_why() {
        // The library only stops on an empty vector, so a failure anywhere else
        // — a reward script that died, say — has to arrive through here.
        let signal = StopSignal::new();
        let mut source =
            source(vec![ok(&[1.0, 2.0]), ok(&[3.0, 4.0])], 2).with_stop_signal(signal.clone());

        assert_eq!(source.next_features(), vec![1.0, 2.0]);
        signal.stop("reward script failed 10 times in a row".to_string());
        assert!(source.next_features().is_empty());

        let message = source.finish().unwrap_err().to_string();
        assert!(message.contains("failed 10 times in a row"), "{}", message);
    }

    #[test]
    fn the_last_row_number_counts_rows_that_were_skipped() {
        // Each inference record names the row it came from, so the number has
        // to follow the data source rather than count decisions.
        let mut source = source(vec![ok(&[1.0, 2.0]), err("bad line"), ok(&[3.0, 4.0])], 2);

        source.next_features();
        assert_eq!(source.last_row_number(), 1);
        source.next_features();
        assert_eq!(source.last_row_number(), 3);
    }

    #[test]
    fn a_good_row_resets_the_failure_streak() {
        let mut rows: Vec<Row> = (0..MAX_CONSECUTIVE_ERRORS - 1)
            .map(|_| err("blip"))
            .collect();
        rows.push(ok(&[1.0, 2.0]));
        rows.extend((0..MAX_CONSECUTIVE_ERRORS - 1).map(|_| err("blip")));
        rows.push(ok(&[3.0, 4.0]));
        let mut source = source(rows, 2);

        assert_eq!(source.next_features(), vec![1.0, 2.0]);
        assert_eq!(source.next_features(), vec![3.0, 4.0]);
        assert!(
            source.finish().is_ok(),
            "transient errors must not be fatal"
        );
    }
}
