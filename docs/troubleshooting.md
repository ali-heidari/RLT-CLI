# Troubleshooting

This document covers common errors and issues you might encounter when using RLT-CLI, along with solutions and workarounds.

## Configuration Errors

### Invalid Configuration File

**Error:** `Error: Failed to load config: invalid TOML syntax`

**Cause:** The configuration file contains invalid TOML syntax.

**Solution:**
- Validate your TOML file using an online TOML validator
- Check for missing quotes, incorrect indentation, or invalid characters
- Ensure all required fields are present and correctly typed

### Missing Configuration File

**Error:** `Error: Config file not found: Config.toml`

**Cause:** The specified config file doesn't exist or the path is incorrect.

**Solution:**
- Verify the file path is correct
- Create a sample config file if it doesn't exist
- Use absolute paths or ensure relative paths are from the current working directory

## Dataset Errors

### Dataset File Not Found

**Error:** `Error: Dataset file not found: ./data/train.csv`

**Cause:** The CSV dataset file specified in config or arguments doesn't exist.

**Solution:**
- Check the file path in your config or command-line arguments
- Ensure the file exists and has read permissions
- Use absolute paths if relative paths are causing issues

### Invalid CSV Format

**Error:** `Error: Failed to parse CSV row: expected 8 features, got 5`

**Cause:** The CSV file doesn't match the expected format (8 features per row).

**Solution:**
- Verify your CSV has exactly 8 numeric columns
- Check for missing values or extra commas
- Ensure no header row if not expected, or remove it if present

### Shape Incompatibility Error

**Error:** `called \`Result::unwrap()\` on an \`Err\` value: ShapeError/IncompatibleShape: incompatible shapes`

**Cause:** The number of columns in your datasource CSV doesn't match the `input_number` specified in the configuration.

**Solution:**
- Ensure your CSV file has exactly the same number of columns as the `input_number` value in your config file
- Check that `input_number` in `Config.toml` matches your dataset's feature count
- Verify there are no extra columns, missing columns, or header rows in the CSV

### Empty Dataset

**Error:** `Error: Dataset is empty`

**Cause:** The CSV file contains no data rows.

**Solution:**
- Add data rows to your CSV file
- Check if the file is truncated or corrupted
- Verify the correct file is being loaded

## Model Errors

### Model Directory Not Writable

**Error:** `Error: Failed to create model directory: Permission denied`

**Cause:** The application doesn't have write permissions to create the models directory.

**Solution:**
- Run the application with appropriate permissions
- Change the model path to a writable location
- Create the directory manually: `mkdir -p models/`

### Model File Corruption

**Error:** `Error: Failed to load model: invalid JSON format`

**Cause:** The model checkpoint file is corrupted or not a valid JSON.

**Solution:**
- Delete the corrupted checkpoint and restart training
- Check file permissions and disk space
- Ensure no other processes are modifying the file simultaneously

### Feature Dimension Mismatch

**Error:** `Error: Feature dimension mismatch: expected 8, got 6`

**Cause:** The input features don't match the model's expected input size.

**Solution:**
- Verify your dataset has exactly 8 features per row
- Check the `input_number` in your config matches your data
- Ensure consistent feature extraction across training and inference

## Training Errors

### Buffer Underflow

**Error:** `Warning: Buffer underflow - not enough experiences for batch`

**Cause:** Training started before sufficient experiences were collected.

**Solution:**
- Increase the initial data collection period
- Reduce batch size or increase buffer capacity
- Wait for more experiences before starting training

### Diverging Loss

**Error:** `Warning: Training diverging - loss increasing`

**Cause:** The model is not learning and loss is getting worse.

**Solution:**
- Reduce learning rate
- Check reward calculation logic
- Verify feature normalization
- Restart training with different hyperparameters

### Memory Issues

**Error:** `Error: Out of memory`

**Cause:** Large datasets or models exceeding available RAM.

**Solution:**
- Use streaming dataset loading (iterator-based)
- Reduce batch size or model size
- Increase system memory or use a machine with more RAM

## Runtime Errors

### Tokio Runtime Issues

**Error:** `Error: Cannot start a runtime from within a runtime`

**Cause:** Attempting to start a Tokio runtime inside an existing one.

**Solution:**
- Ensure the application is the entry point for the runtime
- Check for nested async calls or library conflicts

### Serialization Errors

**Error:** `Error: Failed to serialize model: serde error`

**Cause:** Model weights contain invalid values (NaN, infinity).

**Solution:**
- Check for numerical instabilities in training
- Add gradient clipping
- Validate input data for extreme values

## Command-Line Errors

### Unknown Subcommand

**Error:** `error: Found argument 'invalid' which wasn't expected`

**Cause:** Invalid subcommand or flag used.

**Solution:**
- Use `cargo run -- --help` to see available commands
- Check command syntax: `train`, `infer`, `export`
- Verify flag names and formats

### Missing Required Arguments

**Error:** `error: the following required arguments were not provided: --dataset`

**Cause:** Required arguments not specified.

**Solution:**
- Provide all required arguments or use a config file
- Use `--help` with the subcommand for details

## Performance Issues

### Slow Training

**Cause:** Various factors can slow down training.

**Solutions:**
- Use GPU acceleration if available
- Optimize batch size (typically 32-128)
- Reduce model complexity (fewer hidden layers)
- Use faster storage for checkpoints

### High CPU Usage

**Cause:** Inefficient data loading or processing.

**Solutions:**
- Use streaming iterators for large datasets
- Optimize feature preprocessing
- Reduce logging frequency
- Profile the application for bottlenecks

## Getting Help

If you encounter an error not covered here:

1. Check the application logs for more detailed error messages
2. Verify your Rust and Cargo versions are up to date
3. Ensure all dependencies are correctly installed
4. Search existing issues or create a new one with full error details
5. Include your config file, command used, and system information when reporting issues