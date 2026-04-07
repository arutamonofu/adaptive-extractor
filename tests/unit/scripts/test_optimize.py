"""Unit tests for optimize CLI command.

Tests cover:
- Argument parsing
- Configuration loading
- Signature validation
- Parsed directory existence check
- Error handling
"""

from pathlib import Path

import pytest

# Note: clear_task_registry fixture is now in tests/conftest.py (autouse=True)


@pytest.mark.unit
class TestOptimizeArgumentParsing:
    """Tests for optimize command argument parsing."""

    def test_missing_config_argument(self):
        """Test that script fails without --config argument."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/optimize.py"],
            capture_output=True,
            text=True,
        )

        # Should exit with non-zero code
        assert result.returncode != 0
        # Should mention required argument
        assert "--config" in result.stderr or "required" in result.stderr.lower()

    def test_help_argument(self):
        """Test that --help works correctly."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/optimize.py", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--run-name" in result.stdout
        assert "--no-mlflow" in result.stdout


@pytest.mark.unit
class TestConfigValidation:
    """Tests for configuration file validation."""

    def test_config_file_not_found(self, tmp_path: Path):
        """Test that script fails when config file doesn't exist."""
        from aee.interface.cli.optimize import optimize_command

        result = optimize_command([
            "--config", str(tmp_path / "nonexistent.yaml"),
        ])

        assert result == 1

    def test_invalid_config_file(self, tmp_path: Path):
        """Test that script fails with invalid YAML config."""
        from aee.interface.cli.optimize import optimize_command

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [", encoding="utf-8")

        result = optimize_command([
            "--config", str(config_file),
        ])

        assert result == 1


@pytest.mark.unit
class TestSignatureValidation:
    """Tests for task signature validation."""

    def test_missing_signature_returns_error(self, tmp_path: Path):
        """Test that optimize command fails when task signature is missing."""
        from aee.interface.cli.optimize import optimize_command

        # Create minimal config with invalid task name and temporary instruction file
        config_file = tmp_path / "config.yaml"
        instruction_file = tmp_path / "config" / "initial_instructions" / "nanozymes_sota.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction for nanozymes extraction.")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: data/parsed
  ground_truth_dir: data/ground_truth
  splits_file: data/splits.json
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: invalid_task
  initial_instruction_file: {instruction_file}
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    provider: "ollama"
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
  teacher:
    provider: "ollama"
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
parsing:
  parser: marker
  overwrite: false
  marker:
    device: cpu
optimization:
  num_trials: 1
  train_split: 5
  total_load: 10
  random_seed: 42
  num_candidates: 2
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  verbose: false
  use_cache: false
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""", encoding="utf-8")

        # Should fail because task signature won't be found for invalid_task
        result = optimize_command([
            "--config", str(config_file),
        ])

        # Should fail with exit code 1 (signature validation or task not found)
        assert result == 1


@pytest.mark.unit
class TestParsedDirectoryCheck:
    """Tests for parsed directory existence validation."""

    def test_missing_parsed_dir_returns_error(self, tmp_path: Path):
        """Test that optimize command fails when parsed_dir doesn't exist."""
        from aee.interface.cli.optimize import optimize_command

        # Create config with non-existent parsed_dir and temporary instruction file
        config_file = tmp_path / "config.yaml"
        non_existent_dir = tmp_path / "nonexistent_parsed"
        instruction_file = tmp_path / "config" / "initial_instructions" / "nanozymes_sota.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction for nanozymes extraction.")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: {non_existent_dir}
  ground_truth_dir: data/ground_truth
  splits_file: data/splits.json
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: {instruction_file}
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    provider: "ollama"
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
  teacher:
    provider: "ollama"
    model: test-model
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 256
parsing:
  parser: marker
  overwrite: false
  marker:
    device: cpu
optimization:
  num_trials: 1
  train_split: 5
  total_load: 10
  random_seed: 42
  num_candidates: 2
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  verbose: false
  use_cache: false
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""", encoding="utf-8")

        # Should return exit code 1 for missing parsed_dir
        result = optimize_command([
            "--config", str(config_file),
        ])

        assert result == 1
