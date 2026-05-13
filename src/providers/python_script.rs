use std::process::{Command, Stdio};

pub struct PythonScriptProvider {
    script_path: String,
}

impl PythonScriptProvider {
    pub fn new(script_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let path_obj = std::path::Path::new(script_path);
        if !path_obj.exists() {
            return Err(format!("Python script not found: {}", script_path).into());
        }
        if !path_obj.is_file() {
            return Err(format!("Python script path is not a file: {}", script_path).into());
        }
        Ok(PythonScriptProvider {
            script_path: script_path.to_string(),
        })
    }

    fn call_python(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut python = Command::new("python3");
        python
            .arg(&self.script_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = match python.output() {
            Ok(out) => out,
            Err(_) => {
                let mut fallback = Command::new("python");
                fallback
                    .arg(&self.script_path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                fallback.output()?
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(
                format!("Python script failed: {}", stderr.trim()).into()
            );
        }

        let stdout = String::from_utf8(output.stdout)?;
        let trimmed = stdout.trim();

        // Try to parse as JSON array first
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(arr) = json_value.as_array() {
                let features: Result<Vec<f64>, _> = arr
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| "Non-numeric value in JSON array".into())
                    })
                    .collect();
                return features.map_err(|e: String| e.into());
            }
        }

        // Fall back to comma-separated float parsing
        let features: Result<Vec<f64>, _> = trimmed
            .split(',')
            .map(|item| {
                item.trim()
                    .parse::<f64>()
                    .map_err(|_| "Invalid float value".into())
            })
            .collect();

        features.map_err(|e: Box<dyn std::error::Error>| e)
    }
}

impl Iterator for PythonScriptProvider {
    type Item = Result<Vec<f64>, Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.call_python())
    }
}

pub fn open_python_script(script_path: &str) -> Result<PythonScriptProvider, Box<dyn std::error::Error>> {
    PythonScriptProvider::new(script_path)
}
