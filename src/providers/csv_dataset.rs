//! CSV data provider.
//!
//! Parsing goes through the `csv` crate so quoted fields, embedded delimiters
//! and escapes are handled correctly. A field that is not a number is reported
//! as an error naming the line and column: silently substituting `0.0` turns a
//! typo into a column of fabricated training data.

use std::error::Error;
use std::fs::File;

/// How to read a CSV file.
#[derive(Debug, Clone, Copy)]
pub struct CsvOptions {
    /// Skip the first record instead of parsing it as data.
    pub has_header: bool,
    /// Field separator, e.g. `b','` or `b'\t'`.
    pub delimiter: u8,
}

impl Default for CsvOptions {
    fn default() -> Self {
        Self {
            has_header: false,
            delimiter: b',',
        }
    }
}

pub struct CsvDataset {
    records: csv::StringRecordsIntoIter<File>,
}

impl CsvDataset {
    pub fn new(dataset_path: &str, options: CsvOptions) -> Result<Self, Box<dyn Error>> {
        let reader = csv::ReaderBuilder::new()
            .has_headers(options.has_header)
            .delimiter(options.delimiter)
            // Row width is validated against `input_number` by FeatureSource,
            // which reports it better than the csv crate's own message.
            .flexible(true)
            .from_path(dataset_path)
            .map_err(|err| format!("failed to open '{}': {}", dataset_path, err))?;

        Ok(CsvDataset {
            records: reader.into_records(),
        })
    }
}

/// Parse one record into floats, naming the line and column on failure.
fn parse_record(record: &csv::StringRecord) -> Result<Vec<f64>, Box<dyn Error>> {
    let line = record.position().map(|p| p.line()).unwrap_or(0);

    record
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let column = index + 1;
            let value = field.trim();

            if value.is_empty() {
                // A missing measurement is not a measurement of zero.
                return Err(format!("line {}, column {}: empty field", line, column).into());
            }

            value.parse::<f64>().map_err(|_| {
                format!(
                    "line {}, column {}: '{}' is not a number",
                    line, column, value
                )
                .into()
            })
        })
        .collect()
}

impl Iterator for CsvDataset {
    type Item = Result<Vec<f64>, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.records.next()? {
                Ok(record) => {
                    // Skip blank lines rather than reporting them as errors.
                    if record.iter().all(|field| field.trim().is_empty()) {
                        continue;
                    }
                    return Some(parse_record(&record));
                }
                Err(err) => return Some(Err(format!("failed to read CSV: {}", err).into())),
            }
        }
    }
}

pub fn open_csv_dataset(
    dataset_path: &str,
    options: CsvOptions,
) -> Result<CsvDataset, Box<dyn Error>> {
    CsvDataset::new(dataset_path, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Read a CSV from a literal, returning one result per row.
    fn read(contents: &str, options: CsvOptions) -> Vec<Result<Vec<f64>, Box<dyn Error>>> {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(contents.as_bytes()).unwrap();

        CsvDataset::new(file.path().to_str().unwrap(), options)
            .unwrap()
            .collect()
    }

    #[test]
    fn parses_plain_rows() {
        let rows = read("1,2.5,-3\n4,5,6\n", CsvOptions::default());

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].as_ref().unwrap(), &vec![1.0, 2.5, -3.0]);
        assert_eq!(rows[1].as_ref().unwrap(), &vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn a_quoted_field_is_one_field() {
        // The old splitter turned this into four fields.
        let rows = read("\"1.5\",2,3\n", CsvOptions::default());
        assert_eq!(rows[0].as_ref().unwrap(), &vec![1.5, 2.0, 3.0]);
    }

    #[test]
    fn a_bad_field_is_an_error_not_a_zero() {
        // Regression: unparseable fields used to silently become 0.0, turning a
        // typo into a column of fabricated data.
        let rows = read("1,2,3\n4,n/a,6\n", CsvOptions::default());

        assert!(rows[0].is_ok());
        let message = rows[1].as_ref().unwrap_err().to_string();
        assert!(message.contains("line 2"), "{}", message);
        assert!(message.contains("column 2"), "{}", message);
        assert!(message.contains("'n/a' is not a number"), "{}", message);
    }

    #[test]
    fn an_empty_field_is_an_error() {
        // A missing measurement is not a measurement of zero.
        let rows = read("1,,3\n", CsvOptions::default());
        assert!(rows[0]
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("empty field"));
    }

    #[test]
    fn a_header_is_skipped_only_when_asked_for() {
        let csv = "cpu,mem,net\n1,2,3\n";

        let without = read(csv, CsvOptions::default());
        assert!(without[0].is_err(), "header must not parse as data");

        let with = read(
            csv,
            CsvOptions {
                has_header: true,
                ..CsvOptions::default()
            },
        );
        assert_eq!(with.len(), 1);
        assert_eq!(with[0].as_ref().unwrap(), &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn honours_a_custom_delimiter() {
        let rows = read(
            "1\t2\t3\n",
            CsvOptions {
                delimiter: b'\t',
                ..CsvOptions::default()
            },
        );
        assert_eq!(rows[0].as_ref().unwrap(), &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let rows = read("1,2,3\n\n\n4,5,6\n", CsvOptions::default());
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn ragged_rows_are_returned_as_is_for_the_width_check() {
        // Width is validated against input_number by FeatureSource, so the
        // reader must not reject a short row on its own.
        let rows = read("1,2,3\n4,5\n", CsvOptions::default());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].as_ref().unwrap(), &vec![4.0, 5.0]);
    }

    #[test]
    fn a_missing_file_is_reported_with_its_path() {
        let error = match CsvDataset::new("./does-not-exist.csv", CsvOptions::default()) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("opening a missing file should fail"),
        };
        assert!(error.contains("./does-not-exist.csv"), "{}", error);
    }
}
