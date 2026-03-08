"""Unit tests for generate_manual_agent.py CLI command.

Tests cover:
- Argument parsing
- Configuration loading
- Signature validation
- Parsed directory existence check
- Splits JSON validation
- Error handling
"""

from pathlib import Path

import pytest

# Note: clear_task_registry fixture is now in tests/conftest.py (autouse=True)


@pytest.mark.unit
class TestGenerateManualArgumentParsing:
    """Tests for generate_manual_agent argument parsing."""

    def test_missing_config_argument(self):
        """Test that script fails without --config argument."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/generate_manual_agent.py"],
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
            ["python", "scripts/generate_manual_agent.py", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--output" in result.stdout


@pytest.mark.unit
class TestConfigValidation:
    """Tests for configuration file validation."""

    def test_config_file_not_found(self, tmp_path: Path):
        """Test that script fails when config file doesn't exist."""
        # Import main function from script
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

        import subprocess

        # Use subprocess to test sys.argv handling
        result = subprocess.run(
            ["python", "scripts/generate_manual_agent.py", "--config", str(tmp_path / "nonexistent.yaml")],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1

    def test_invalid_config_file(self, tmp_path: Path):
        """Test that script fails with invalid YAML config."""
        import subprocess

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [", encoding="utf-8")

        result = subprocess.run(
            ["python", "scripts/generate_manual_agent.py", "--config", str(config_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1


@pytest.mark.unit
class TestSignatureValidation:
    """Tests for task signature validation."""

    def test_missing_signature_returns_error(self, tmp_path: Path):
        """Test that script fails when task signature is missing."""
        import subprocess

        # Create minimal config with invalid task name
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
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
  initial_instruction_file: config/initial_instructions/nanozymes_sota.txt
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    use_ollama: true
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
    use_ollama: true
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
  parser: docling
  overwrite: false
  docling:
    device: cpu
    num_threads: 2
    do_ocr: false
    do_table_structure: false
    ocr_backend: onnxruntime
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

        result = subprocess.run(
            ["python", "scripts/generate_manual_agent.py", "--config", str(config_file)],
            capture_output=True,
            text=True,
        )

        # Should fail with exit code 1 (signature validation or task not found)
        assert result.returncode == 1


@pytest.mark.unit
class TestParsedDirectoryCheck:
    """Tests for parsed directory existence validation."""

    def test_missing_parsed_dir_returns_error(self, tmp_path: Path):
        """Test that script fails when parsed_dir doesn't exist."""
        import subprocess

        # Create config with non-existent parsed_dir
        config_file = tmp_path / "config.yaml"
        non_existent_dir = str(tmp_path / "nonexistent_parsed")
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
  initial_instruction_file: config/initial_instructions/nanozymes_sota.txt
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    use_ollama: true
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
    use_ollama: true
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
  parser: docling
  overwrite: false
  docling:
    device: cpu
    num_threads: 2
    do_ocr: false
    do_table_structure: false
    ocr_backend: onnxruntime
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

        result = subprocess.run(
            ["python", "scripts/generate_manual_agent.py", "--config", str(config_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1


@pytest.mark.unit
class TestSplitsJsonValidation:
    """Tests for splits JSON file validation."""

    def test_invalid_splits_json_returns_error(self, tmp_path: Path):
        """Test that script fails when splits file contains invalid JSON."""
        import subprocess

        # Create config
        config_file = tmp_path / "config.yaml"
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        splits_file = tmp_path / "splits.json"
        
        # Write invalid JSON
        splits_file.write_text("invalid json content {", encoding="utf-8")

        config_file.write_text(f"""
project:
  log_level: INFO
paths:
  pdf_dir: data/pdf
  parsed_dir: {parsed_dir}
  ground_truth_dir: data/ground_truth
  splits_file: {splits_file}
  agents_dir: data/agents
  extractions_dir: data/extractions
task:
  name: nanozymes
  initial_instruction_file: config/initial_instructions/nanozymes_sota.txt
  evaluation:
    compare_fields: [formula]
    float_tolerance: 0.1
llm:
  student:
    use_ollama: true
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
    use_ollama: true
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
  parser: docling
  overwrite: false
  docling:
    device: cpu
    num_threads: 2
    do_ocr: false
    do_table_structure: false
    ocr_backend: onnxruntime
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

        result = subprocess.run(
            ["python", "scripts/generate_manual_agent.py", "--config", str(config_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
