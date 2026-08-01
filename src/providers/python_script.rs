//! Python-script data provider.
//!
//! The script is started once and asked for a sample per step over its stdin /
//! stdout pipes; see [`crate::providers::python_worker`] for the protocol. For
//! each request line the script prints one line holding either a JSON array of
//! numbers or comma-separated floats.

use crate::providers::python_worker::PythonWorker;
use std::error::Error;
use std::path::Path;
use std::time::Duration;

/// Request sent to ask for the next sample.
///
/// An empty JSON object today; it is an object rather than a bare newline so
/// fields can be added later without breaking scripts that ignore it.
const NEXT_SAMPLE_REQUEST: &str = "{}";

pub struct PythonScriptProvider {
    worker: PythonWorker,
}

impl PythonScriptProvider {
    pub fn new(script_path: &str, timeout: Option<Duration>) -> Result<Self, Box<dyn Error>> {
        let path = Path::new(script_path);
        if !path.exists() {
            return Err(format!("Python script not found: {}", script_path).into());
        }
        if !path.is_file() {
            return Err(format!("Python script path is not a file: {}", script_path).into());
        }

        Ok(PythonScriptProvider {
            worker: PythonWorker::spawn(script_path, timeout)?,
        })
    }

    fn next_sample(&mut self) -> Result<Vec<f64>, Box<dyn Error>> {
        let response = self.worker.request(NEXT_SAMPLE_REQUEST)?;
        parse_features(&response)
    }
}

/// Parse one response line: a JSON array of numbers, or comma-separated floats.
fn parse_features(response: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Err("the script returned an empty line".into());
    }

    if let Ok(serde_json::Value::Array(values)) = serde_json::from_str::<serde_json::Value>(trimmed)
    {
        return values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| format!("non-numeric value {} in the JSON array", value).into())
            })
            .collect();
    }

    trimmed
        .split(',')
        .map(|item| {
            let field = item.trim();
            field
                .parse::<f64>()
                .map_err(|_| format!("'{}' is not a number", field).into())
        })
        .collect()
}

impl Iterator for PythonScriptProvider {
    type Item = Result<Vec<f64>, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_sample())
    }
}

pub fn open_python_script(
    script_path: &str,
    timeout: Option<Duration>,
) -> Result<PythonScriptProvider, Box<dyn Error>> {
    PythonScriptProvider::new(script_path, timeout)
}

#[cfg(test)]
mod tests {
    use super::parse_features;

    #[test]
    fn parses_a_json_array() {
        assert_eq!(
            parse_features("[0.1, 2, -3.5]").unwrap(),
            vec![0.1, 2.0, -3.5]
        );
    }

    #[test]
    fn parses_comma_separated_floats() {
        assert_eq!(
            parse_features("0.1, 2, -3.5").unwrap(),
            vec![0.1, 2.0, -3.5]
        );
    }

    #[test]
    fn rejects_an_empty_response() {
        // A blank line means the script answered nothing; treating it as an
        // empty feature vector would look like a stop signal instead.
        assert!(parse_features("   ")
            .unwrap_err()
            .to_string()
            .contains("empty line"));
    }

    #[test]
    fn rejects_a_non_numeric_json_element() {
        let error = parse_features("[1, \"two\", 3]").unwrap_err().to_string();
        assert!(error.contains("non-numeric"), "{}", error);
    }

    #[test]
    fn rejects_a_non_numeric_csv_field() {
        let error = parse_features("1, two, 3").unwrap_err().to_string();
        assert!(error.contains("'two' is not a number"), "{}", error);
    }
}
