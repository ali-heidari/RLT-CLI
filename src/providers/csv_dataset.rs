use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct CsvDataset {
    lines: io::Lines<BufReader<File>>,
}

impl CsvDataset {
    pub fn new(dataset_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(dataset_path)?;
        let reader = BufReader::new(file);
        Ok(CsvDataset {
            lines: reader.lines(),
        })
    }
}

impl Iterator for CsvDataset {
    type Item = Result<Vec<f64>, Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        for line_result in self.lines.by_ref() {
            match line_result {
                Ok(line) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let parsed = trimmed
                        .split(',')
                        .map(|item| item.trim().parse::<f64>().or_else(|_| Ok(0_f64)))
                        .collect::<Result<Vec<_>, _>>();

                    return Some(parsed);
                }
                Err(err) => return Some(Err(err.into())),
            }
        }

        None
    }
}

pub fn open_csv_dataset(dataset_path: &str) -> Result<CsvDataset, Box<dyn std::error::Error>> {
    CsvDataset::new(dataset_path)
}
