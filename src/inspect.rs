//! Reading a checkpoint back out.
//!
//! Checkpoints are JSON, but that only means they are *technically* readable:
//! four tensors of bare numbers, a node id, and the text of the last training
//! footstep. This turns that into shapes, a parameter count, and — the part
//! worth having — a note when the numbers themselves look wrong.
//!
//! Weight health is not decoration. A model whose biases sit in the thousands
//! has diverged, and the only symptom otherwise is that its decisions are poor,
//! which is indistinguishable from a model that merely learned badly.

use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::fmt;

/// Magnitude above which a weight is called suspicious.
///
/// Activations here are bounded and inputs are expected to be normalised, so
/// anything this large means the updates ran away rather than converged.
const SUSPICIOUS_MAGNITUDE: f64 = 1_000.0;

/// One stored tensor.
#[derive(Debug, Serialize)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Values that are NaN or infinite. Any at all is a broken checkpoint.
    pub non_finite: usize,
}

impl Tensor {
    /// Whether the magnitudes suggest the run diverged.
    fn suspicious(&self) -> bool {
        self.min.abs() > SUSPICIOUS_MAGNITUDE || self.max.abs() > SUSPICIOUS_MAGNITUDE
    }
}

/// A parsed checkpoint.
#[derive(Debug, Serialize)]
pub struct Checkpoint {
    pub path: String,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub tensors: Vec<Tensor>,
    pub parameters: usize,
    /// Input, hidden and output widths, read off the weight shapes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<[usize; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
    pub warnings: Vec<String>,
}

/// Read a checkpoint's structure out of its JSON.
///
/// Parsed permissively through `Value` rather than a typed struct: the layout
/// belongs to the library, and a checkpoint holding one unexpected field should
/// still report everything else it holds.
pub fn parse(path: &str, bytes: u64, text: &str) -> Result<Checkpoint, Box<dyn Error>> {
    let value: Value = serde_json::from_str(text)
        .map_err(|err| format!("'{}' is not valid JSON: {}", path, err))?;

    let object = value
        .as_object()
        .ok_or_else(|| format!("'{}' does not hold a JSON object", path))?;

    let mut tensors = Vec::new();
    for (name, field) in object {
        if let Some(tensor) = read_tensor(name, field) {
            tensors.push(tensor);
        }
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));

    if tensors.is_empty() {
        return Err(format!(
            "'{}' holds no tensors; it may not be a checkpoint at all",
            path
        )
        .into());
    }

    let parameters = tensors.iter().map(|tensor| tensor.count).sum();
    let architecture = architecture(&tensors);
    let warnings = warnings(&tensors);

    Ok(Checkpoint {
        path: path.to_string(),
        bytes,
        id: object.get("id").and_then(Value::as_str).map(str::to_string),
        tensors,
        parameters,
        architecture,
        snapshot: object
            .get("snapshot")
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .map(|text| text.trim().to_string()),
        warnings,
    })
}

/// Read one `{ "dim": [...], "data": [...] }` field, if that is what it is.
fn read_tensor(name: &str, field: &Value) -> Option<Tensor> {
    let object = field.as_object()?;
    let dims: Vec<usize> = object
        .get("dim")?
        .as_array()?
        .iter()
        .map(|value| value.as_u64().map(|dim| dim as usize))
        .collect::<Option<_>>()?;
    let data: Vec<f64> = object
        .get("data")?
        .as_array()?
        .iter()
        .map(Value::as_f64)
        .collect::<Option<_>>()?;

    let finite: Vec<f64> = data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let total: f64 = finite.iter().sum();

    Some(Tensor {
        name: name.to_string(),
        dims,
        count: data.len(),
        min: finite.iter().copied().fold(f64::INFINITY, f64::min),
        max: finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean: if finite.is_empty() {
            0.0
        } else {
            total / finite.len() as f64
        },
        non_finite: data.len() - finite.len(),
    })
}

/// Input, hidden and output widths, from the two weight matrices.
fn architecture(tensors: &[Tensor]) -> Option<[usize; 3]> {
    let find = |name: &str| tensors.iter().find(|tensor| tensor.name == name);
    let first = find("w1")?;
    let second = find("w2")?;

    match (first.dims.as_slice(), second.dims.as_slice()) {
        ([input, hidden], [second_hidden, output]) if hidden == second_hidden => {
            Some([*input, *hidden, *output])
        }
        _ => None,
    }
}

fn warnings(tensors: &[Tensor]) -> Vec<String> {
    let mut warnings = Vec::new();

    for tensor in tensors {
        if tensor.non_finite > 0 {
            warnings.push(format!(
                "{} holds {} non-finite value(s): this checkpoint is broken and \
                 inference from it is meaningless",
                tensor.name, tensor.non_finite
            ));
        } else if tensor.suspicious() {
            warnings.push(format!(
                "{} reaches {:.4e}: the run probably diverged rather than converged",
                tensor.name,
                tensor.min.abs().max(tensor.max.abs())
            ));
        }
    }

    warnings
}

impl Checkpoint {
    pub fn to_json(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

impl fmt::Display for Checkpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "Checkpoint: {} ({} bytes)",
            self.path, self.bytes
        )?;
        if let Some(id) = &self.id {
            writeln!(formatter, "Node id:    {}", id)?;
        }

        writeln!(formatter)?;
        writeln!(formatter, "Architecture")?;
        match self.architecture {
            Some([input, hidden, output]) => {
                writeln!(formatter, "  {} -> {} -> {}", input, hidden, output)?;
            }
            None => writeln!(formatter, "  <could not be read from the weight shapes>")?,
        }
        writeln!(formatter, "  {} parameters", self.parameters)?;

        writeln!(formatter)?;
        writeln!(formatter, "Tensors")?;
        for tensor in &self.tensors {
            writeln!(
                formatter,
                "  {:<4} {:<10} {:>6} values   min {:>12.4}  max {:>12.4}  mean {:>12.4}",
                tensor.name,
                format!("{:?}", tensor.dims),
                tensor.count,
                tensor.min,
                tensor.max,
                tensor.mean
            )?;
        }

        if !self.warnings.is_empty() {
            writeln!(formatter)?;
            writeln!(formatter, "Warnings")?;
            for warning in &self.warnings {
                writeln!(formatter, "  {}", warning)?;
            }
        }

        if let Some(snapshot) = &self.snapshot {
            writeln!(formatter)?;
            writeln!(formatter, "Last training snapshot")?;
            for line in snapshot.lines() {
                writeln!(formatter, "  {}", line)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint shaped like the ones the library writes.
    fn checkpoint(w2_data: &str) -> String {
        format!(
            r#"{{
                "w1": {{"v": 1, "dim": [2, 3], "data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}},
                "b1": {{"v": 1, "dim": [3], "data": [0.0, 0.1, -0.1]}},
                "w2": {{"v": 1, "dim": [3, 2], "data": {}}},
                "b2": {{"v": 1, "dim": [2], "data": [0.0, 0.0]}},
                "snapshot": "  [Logits]\t[0.0, -1.0]\n",
                "id": "m.json"
            }}"#,
            w2_data
        )
    }

    #[test]
    fn shapes_and_parameter_count_are_read_back() {
        let parsed = parse("m.json", 100, &checkpoint("[1,2,3,4,5,6]")).unwrap();

        assert_eq!(parsed.architecture, Some([2, 3, 2]));
        assert_eq!(parsed.parameters, 6 + 3 + 6 + 2);
        assert_eq!(parsed.id.as_deref(), Some("m.json"));
        assert_eq!(parsed.tensors.len(), 4);
        // Sorted, so the report does not reorder itself between runs.
        assert_eq!(parsed.tensors[0].name, "b1");
    }

    #[test]
    fn the_snapshot_is_carried_through() {
        let parsed = parse("m.json", 100, &checkpoint("[1,2,3,4,5,6]")).unwrap();

        assert!(parsed.snapshot.unwrap().contains("[Logits]"));
    }

    #[test]
    fn healthy_weights_raise_no_warnings() {
        let parsed = parse("m.json", 100, &checkpoint("[0.1,0.2,0.3,0.4,0.5,0.6]")).unwrap();

        assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
    }

    #[test]
    fn runaway_weights_are_called_out() {
        // The real symptom of divergence: a checkpoint that loads fine and
        // decides badly. Reading it back is the only way to see it.
        let parsed = parse(
            "m.json",
            100,
            &checkpoint("[895730.2, -758624.1, 0.0, 0.0, 0.0, 0.0]"),
        )
        .unwrap();

        assert_eq!(parsed.warnings.len(), 1);
        assert!(parsed.warnings[0].contains("w2"), "{:?}", parsed.warnings);
        assert!(
            parsed.warnings[0].contains("diverged"),
            "{:?}",
            parsed.warnings
        );
    }

    #[test]
    fn non_finite_weights_are_reported_as_broken() {
        let json = r#"{"w1": {"dim": [1, 2], "data": [1.0, null]}}"#;
        // `null` is not a number, so the tensor is not read at all; a checkpoint
        // of nothing but unreadable fields must say so rather than report zero
        // parameters as if that were fine.
        let error = parse("m.json", 10, json).unwrap_err().to_string();
        assert!(error.contains("holds no tensors"), "{}", error);
    }

    #[test]
    fn something_that_is_not_a_checkpoint_is_rejected() {
        let error = parse("m.json", 10, "{\"hello\": 1}")
            .unwrap_err()
            .to_string();
        assert!(error.contains("holds no tensors"), "{}", error);

        let error = parse("m.json", 10, "not json").unwrap_err().to_string();
        assert!(error.contains("not valid JSON"), "{}", error);
    }

    #[test]
    fn statistics_ignore_non_finite_values_rather_than_poisoning_the_mean() {
        let json = r#"{"w1": {"dim": [2], "data": [1.0, 3.0]}}"#;
        let parsed = parse("m.json", 10, json).unwrap();

        assert_eq!(parsed.tensors[0].mean, 2.0);
        assert_eq!(parsed.tensors[0].min, 1.0);
        assert_eq!(parsed.tensors[0].max, 3.0);
        assert_eq!(parsed.tensors[0].non_finite, 0);
    }
}
