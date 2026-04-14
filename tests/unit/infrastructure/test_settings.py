"""Tests for Settings configuration with OpenRouter support."""

import os
from pathlib import Path

import pytest

from aee import Settings


class TestApiConfig:
    """Tests for ApiConfig with base_url support."""

    def test_api_config_with_base_url(self):
        """Test ApiConfig accepts base_url field."""
        from aee.infrastructure.config import ApiConfig

        config = ApiConfig(
            api_key="test-key",
            max_tokens=4096,
            base_url="https://openrouter.ai/api/v1",
        )

        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.max_tokens == 4096
        assert config.api_key.get_secret_value() == "test-key"

    def test_api_config_without_base_url(self):
        """Test ApiConfig works without base_url (optional field)."""
        from aee.infrastructure.config import ApiConfig

        config = ApiConfig(
            api_key="test-key",
            max_tokens=4096,
        )

        assert config.base_url is None
        assert config.max_tokens == 4096


class TestSettingsOpenRouter:
    """Tests for Settings with OpenRouter API key support."""

    def test_openrouter_api_key_field_exists(self):
        """Test Settings class has openrouter_api_key field."""
        from aee import Settings

        # Check field exists in model fields
        assert "openrouter_api_key" in Settings.model_fields

    def test_settings_loads_openrouter_key_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test Settings loads OPENROUTER_API_KEY from environment."""
        # Clear all API key env vars first, then set only OPENROUTER_API_KEY
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345")

        # Create minimal config
        config_content = """
project:
  log_level: INFO
paths:
  pdf_dir: {tmp_path}/pdf
  parsed_dir: {tmp_path}/parsed
  ground_truth_dir: {tmp_path}/ground_truth
  splits_file: {tmp_path}/splits.json
  agents_dir: {tmp_path}/agents
  extractions_dir: {tmp_path}/extractions
task:
  name: test
  initial_instruction_file: {tmp_path}/instruction.txt
llm:
  student:
    provider: "api"
    model: "openrouter/qwen/qwen3.5-397b-a17b"
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 256
  teacher:
    provider: "api"
    model: "openrouter/qwen/qwen3.5-397b-a17b"
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 256
parsing:
  parser: marker
  overwrite: false
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 2
  num_trials: 1
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  random_seed: 42
  verbose: false
  use_cache: false
  max_errors: 5
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""".format(
            tmp_path=tmp_path
        )

        # Create instruction file
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("Test instruction")

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        # Load settings without loading .env file
        settings = Settings.load(config_path=config_file, load_env_file=False)

        # Verify OpenRouter API key is loaded
        assert settings.openrouter_api_key is not None
        assert settings.openrouter_api_key.get_secret_value() == "sk-or-test-key-12345"

    def test_settings_openrouter_applied_to_api_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test OpenRouter API key is applied to API config."""
        # Clear all API key env vars first, then set only OPENROUTER_API_KEY
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-openrouter-key")

        # Create minimal config with provider: "api" for teacher
        config_content = """
project:
  log_level: INFO
paths:
  pdf_dir: {tmp_path}/pdf
  parsed_dir: {tmp_path}/parsed
  ground_truth_dir: {tmp_path}/ground_truth
  splits_file: {tmp_path}/splits.json
  agents_dir: {tmp_path}/agents
  extractions_dir: {tmp_path}/extractions
task:
  name: test
  initial_instruction_file: {tmp_path}/instruction.txt
llm:
  student:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 256
  teacher:
    provider: "api"
    model: "openrouter/qwen/qwen3.5-397b-a17b"
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 4096
      base_url: "https://openrouter.ai/api/v1"
parsing:
  parser: marker
  overwrite: false
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 2
  num_trials: 1
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  random_seed: 42
  verbose: false
  use_cache: false
  max_errors: 5
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""".format(
            tmp_path=tmp_path
        )

        # Create instruction file
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("Test instruction")

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        # Load settings without loading .env file
        settings = Settings.load(config_path=config_file, load_env_file=False)

        # Verify teacher config has API key applied
        assert settings.llm.teacher.provider == "api"
        assert settings.llm.teacher.api.api_key is not None
        assert (
            settings.llm.teacher.api.api_key.get_secret_value()
            == "sk-or-openrouter-key"
        )
        # Verify base_url is preserved
        assert (
            settings.llm.teacher.api.base_url
            == "https://openrouter.ai/api/v1"
        )


class TestSettingsBaseURL:
    """Tests for base_url configuration in API settings."""

    def test_base_url_in_yaml_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test base_url is loaded from YAML config."""
        # Clear all API key env vars first, then set only OPENAI_API_KEY
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")

        config_content = """
project:
  log_level: INFO
paths:
  pdf_dir: {tmp_path}/pdf
  parsed_dir: {tmp_path}/parsed
  ground_truth_dir: {tmp_path}/ground_truth
  splits_file: {tmp_path}/splits.json
  agents_dir: {tmp_path}/agents
  extractions_dir: {tmp_path}/extractions
task:
  name: test
  initial_instruction_file: {tmp_path}/instruction.txt
llm:
  student:
    provider: "api"
    model: "openai/gpt-4"
    timeout: 60
    max_retries: 1
    temperature: 0.0
    rate_limit_delay: 0.0
    top_p: 0.1
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 4096
      base_url: "https://custom-api.example.com/v1"
  teacher:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 1
    temperature: 0.5
    rate_limit_delay: 0.0
    top_p: 0.9
    enable_cache: false
    ollama:
      num_ctx: 1024
      num_predict: 256
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 256
parsing:
  parser: marker
  overwrite: false
optimization:
  total_load: 10
  train_split: 5
  num_candidates: 2
  num_trials: 1
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 1
  view_data_batch_size: 1
  metric_threshold: 0.5
  init_temperature: 0.5
  random_seed: 42
  verbose: false
  use_cache: false
  max_errors: 5
extraction:
  enable_cache: false
cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100
circuit_breaker:
  failure_threshold: 3
  reset_timeout: 30.0
  half_open_max_calls: 1
""".format(
            tmp_path=tmp_path
        )

        # Create instruction file
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("Test instruction")

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        # Load settings without loading .env file
        settings = Settings.load(config_path=config_file, load_env_file=False)

        # Verify base_url is loaded (not treated as path)
        assert (
            settings.llm.student.api.base_url
            == "https://custom-api.example.com/v1"
        )
