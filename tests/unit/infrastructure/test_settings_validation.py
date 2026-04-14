"""Tests for Settings validation - Ollama URLs and API keys."""

import os
from pathlib import Path

import pytest

from aee import Settings


@pytest.fixture(autouse=True)
def clear_env():
    """Clear relevant environment variables before each test."""
    env_vars_to_clear = [
        "OLLAMA_STUDENT_BASE_URL",
        "OLLAMA_TEACHER_BASE_URL",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
    ]

    # Save original values
    original_values = {}
    for var in env_vars_to_clear:
        original_values[var] = os.environ.get(var)

    # Clear all
    for var in env_vars_to_clear:
        os.environ.pop(var, None)

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        else:
            os.environ.pop(var, None)


@pytest.mark.unit
class TestConfigFileRequired:
    """Tests for configuration file requirement."""

    def test_missing_config_path_raises_value_error(self):
        """Test that calling load() without config_path raises ValueError."""
        with pytest.raises(ValueError, match="Configuration file path is required"):
            Settings.load(load_env_file=False)

    def test_nonexistent_config_file_raises_file_not_found_error(self, tmp_path: Path):
        """Test that loading a non-existent config file raises FileNotFoundError."""
        nonexistent_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match=f"Configuration file not found: {nonexistent_path}"):
            Settings.load(config_path=nonexistent_path, load_env_file=False)


@pytest.mark.unit
class TestOllamaUrlValidation:
    """Tests for Ollama URL validation."""

    def test_missing_ollama_student_url_raises_error(self, tmp_path: Path):
        """Test that missing OLLAMA_STUDENT_BASE_URL raises ValueError."""
        # Create initial instruction file
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        # Create a minimal config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
project:
  log_level: INFO
llm:
  student:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
  teacher:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "config/initial_instructions/test.txt"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""")

        # Set only teacher URL, student URL is missing
        os.environ["OLLAMA_TEACHER_BASE_URL"] = "http://localhost:11434"

        try:
            with pytest.raises(
                ValueError,
                match="OLLAMA_STUDENT_BASE_URL environment variable must be set",
            ):
                Settings.load(config_path=config_file, load_env_file=False)
        finally:
            os.environ.pop("OLLAMA_TEACHER_BASE_URL", None)

    def test_missing_ollama_teacher_url_raises_error(self, tmp_path: Path):
        """Test that missing OLLAMA_TEACHER_BASE_URL raises ValueError."""
        # Create initial instruction file
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
project:
  log_level: INFO
llm:
  student:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
  teacher:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "config/initial_instructions/test.txt"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""")

        # Set only student URL, teacher URL is missing
        os.environ["OLLAMA_STUDENT_BASE_URL"] = "http://localhost:11434"

        try:
            with pytest.raises(
                ValueError,
                match="OLLAMA_TEACHER_BASE_URL environment variable must be set",
            ):
                Settings.load(config_path=config_file, load_env_file=False)
        finally:
            os.environ.pop("OLLAMA_STUDENT_BASE_URL", None)

    def test_both_ollama_urls_set_succeeds(self, tmp_path: Path):
        """Test that setting both Ollama URLs succeeds."""
        # Create initial instruction file
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(f"""
project:
  log_level: INFO
llm:
  student:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
  teacher:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "{instruction_file}"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""")

        # Set both URLs
        os.environ["OLLAMA_STUDENT_BASE_URL"] = "http://localhost:11434"
        os.environ["OLLAMA_TEACHER_BASE_URL"] = "http://localhost:11435"

        try:
            settings = Settings.load(config_path=config_file, load_env_file=False)

            assert settings.llm.student.ollama.ollama_base_url == "http://localhost:11434"
            assert settings.llm.teacher.ollama.ollama_base_url == "http://localhost:11435"
        finally:
            os.environ.pop("OLLAMA_STUDENT_BASE_URL", None)
            os.environ.pop("OLLAMA_TEACHER_BASE_URL", None)

    def test_empty_ollama_url_raises_error(self, tmp_path: Path):
        """Test that empty Ollama URL raises ValueError."""
        # Create initial instruction file
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
project:
  log_level: INFO
llm:
  student:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
  teacher:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "config/initial_instructions/test.txt"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""")

        # Set empty student URL
        os.environ["OLLAMA_STUDENT_BASE_URL"] = ""
        os.environ["OLLAMA_TEACHER_BASE_URL"] = "http://localhost:11434"

        try:
            with pytest.raises(
                ValueError,
                match="OLLAMA_STUDENT_BASE_URL environment variable must be set",
            ):
                Settings.load(config_path=config_file, load_env_file=False)
        finally:
            os.environ.pop("OLLAMA_STUDENT_BASE_URL", None)
            os.environ.pop("OLLAMA_TEACHER_BASE_URL", None)


@pytest.mark.unit
class TestApiKeyValidation:
    """Tests for API key validation."""

    def test_api_without_api_key_raises_error(self, tmp_path: Path):
        """Test that API config without API key raises ValueError."""
        # Create initial instruction file
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
project:
  log_level: INFO
llm:
  student:
    provider: "api"
    model: "gpt-4"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
  teacher:
    provider: "api"
    model: "gpt-4"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "{instruction_file}"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""")

        # Create initial instruction file
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(f"""
project:
  log_level: INFO
llm:
  student:
    provider: "api"
    model: "gpt-4"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
  teacher:
    provider: "api"
    model: "gpt-4"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "{instruction_file}"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""")

        # Set Ollama URLs but no API keys
        os.environ["OLLAMA_STUDENT_BASE_URL"] = "http://localhost:11434"
        os.environ["OLLAMA_TEACHER_BASE_URL"] = "http://localhost:11434"

        try:
            with pytest.raises(
                ValueError,
                match="API key must be set",
            ):
                Settings.load(config_path=config_file, load_env_file=False)
        finally:
            os.environ.pop("OLLAMA_STUDENT_BASE_URL", None)
            os.environ.pop("OLLAMA_TEACHER_BASE_URL", None)

    def test_api_with_api_key_succeeds(self, tmp_path: Path):
        """Test that API config with API key succeeds."""
        # Create initial instruction file
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(f"""
project:
  log_level: INFO
llm:
  student:
    provider: "api"
    model: "gpt-4"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
  teacher:
    provider: "api"
    model: "gpt-4"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 4096
      num_predict: 1024
      stream: false
      repeat_penalty: 1.1
      repeat_last_n: 512
    api:
      max_tokens: 4096
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "{instruction_file}"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""")

        # Set Ollama URLs and API key
        os.environ["OLLAMA_STUDENT_BASE_URL"] = "http://localhost:11434"
        os.environ["OLLAMA_TEACHER_BASE_URL"] = "http://localhost:11434"
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            settings = Settings.load(config_path=config_file, load_env_file=False)

            assert settings.llm.student.provider == "api"
            assert settings.llm.student.api.api_key is not None
        finally:
            os.environ.pop("OLLAMA_STUDENT_BASE_URL", None)
            os.environ.pop("OLLAMA_TEACHER_BASE_URL", None)
            os.environ.pop("OPENAI_API_KEY", None)


def _minimal_yaml_snippet() -> str:
    """Return a minimal valid YAML config template for test helpers."""
    return """
project:
  log_level: INFO
llm:
  student:
    provider: "transformers"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
  teacher:
    provider: "transformers"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"
parsing:
  parser: "marker"
  overwrite: false
  marker:
    device: "cpu"
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 5
  num_trials: 10
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 5
  view_data_batch_size: 2
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: false
task:
  name: "test"
  initial_instruction_file: "{instruction_file}"
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
"""


@pytest.mark.unit
class TestProviderConfigRequirement:
    """Tests that provider-specific config sections are required."""

    def test_provider_ollama_without_ollama_config_raises(self, tmp_path: Path):
        """Test that provider='ollama' without ollama section raises ValueError."""
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(_minimal_yaml_snippet().format(
            instruction_file=str(instruction_file)
        ).replace(
            'provider: "transformers"',
            'provider: "ollama"'
        ))

        os.environ["OLLAMA_STUDENT_BASE_URL"] = "http://localhost:11434"
        os.environ["OLLAMA_TEACHER_BASE_URL"] = "http://localhost:11434"

        try:
            with pytest.raises(
                ValueError,
                match="ollama configuration is required when provider='ollama'",
            ):
                Settings.load(config_path=config_file, load_env_file=False)
        finally:
            os.environ.pop("OLLAMA_STUDENT_BASE_URL", None)
            os.environ.pop("OLLAMA_TEACHER_BASE_URL", None)

    def test_provider_api_without_api_config_raises(self, tmp_path: Path):
        """Test that provider='api' without api section raises ValueError."""
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(_minimal_yaml_snippet().format(
            instruction_file=str(instruction_file)
        ).replace(
            'provider: "transformers"',
            'provider: "api"'
        ))

        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            with pytest.raises(
                ValueError,
                match="api configuration is required when provider='api'",
            ):
                Settings.load(config_path=config_file, load_env_file=False)
        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_provider_transformers_without_ollama_api_config_succeeds(self, tmp_path: Path):
        """Test that provider='transformers' works without ollama/api sections."""
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.parent.mkdir(parents=True, exist_ok=True)
        instruction_file.write_text("Test instruction")

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(_minimal_yaml_snippet().format(
            instruction_file=str(instruction_file)
        ))

        # No Ollama URLs or API keys needed for transformers
        settings = Settings.load(config_path=config_file, load_env_file=False)

        assert settings.llm.student.provider == "transformers"
        assert settings.llm.student.ollama is None
        assert settings.llm.student.api is None
        assert settings.llm.teacher.provider == "transformers"
        assert settings.llm.teacher.ollama is None
        assert settings.llm.teacher.api is None
