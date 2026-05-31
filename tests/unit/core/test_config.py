from pathlib import Path

import pytest

from ae import Settings
from ae.core.config import ApiConfig


@pytest.mark.unit
class TestConfigSettings:
    """Consolidated unit tests for Settings configuration."""

    def test_api_config_fields(self):
        """Test ApiConfig loading and validation of base_url."""
        cfg = ApiConfig(max_tokens=256, base_url="https://test.api")
        assert cfg.base_url == "https://test.api"

        cfg_none = ApiConfig(max_tokens=256)
        assert cfg_none.base_url is None

    def test_openrouter_api_key_loading(self, tmp_path: Path, monkeypatch):
        """Test Settings loads OPENROUTER_API_KEY from environment."""
        # Clear all API key env vars first, then set only OPENROUTER_API_KEY
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345")

        # Create config file content
        config_content = f"""
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
  visual:
    enabled: false
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
"""
        config_file = tmp_path / "config_mono.yaml"
        config_file.write_text(config_content)

        # Split into modular files inside config directory
        config_dir = tmp_path / "config"
        from tests.conftest import _split_config
        _split_config(config_file, config_dir)

        # Load settings and verify openrouter_api_key resolved
        settings = Settings.load(config_path=config_dir, load_env_file=False)
        assert settings.openrouter_api_key is not None
        assert settings.openrouter_api_key.get_secret_value() == "sk-or-test-key-12345"

    def test_settings_validation_errors(self, tmp_path: Path):
        """Test configuration requirement and validation rules."""
        # Load with a file path instead of directory should raise ValueError
        temp_file = tmp_path / "file.yaml"
        temp_file.write_text("invalid")
        with pytest.raises(ValueError, match="Configuration path must be a directory"):
            Settings.load(config_path=temp_file)

        # Load nonexistent config file should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            Settings.load(config_path=tmp_path / "nonexistent")

