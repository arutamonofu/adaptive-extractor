"""Unit tests for HuggingFace token environment override in Settings."""

import os
from unittest.mock import patch

from aee.infrastructure.config.settings import Settings


class TestHuggingFaceTokenEnvOverride:
    """Test HUGGINGFACE_TOKEN env var injection into transformers config."""

    def test_hf_token_injected_for_student_transformers(self):
        """Test HUGGINGFACE_TOKEN is injected when student uses transformers."""
        config_data = {
            "project": {"log_level": "INFO"},
            "paths": {"pdf_dir": "data/pdfs", "parsed_dir": "data/parsed"},
            "llm": {
                "student": {
                    "model": "qwen/test-model",
                    "provider": "transformers",
                },
                "teacher": {
                    "model": "qwen/test-model",
                    "provider": "ollama",
                    "ollama": {
                        "ollama_base_url": "http://localhost:11434",
                    },
                },
            },
        }

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_my_secret_token"}):
            Settings._apply_env_overrides(config_data)

        student_transformers = config_data["llm"]["student"]["transformers"]
        assert student_transformers["hf_token"] == "hf_my_secret_token"

    def test_hf_token_injected_for_teacher_transformers(self):
        """Test HUGGINGFACE_TOKEN is injected when teacher uses transformers."""
        config_data = {
            "project": {"log_level": "INFO"},
            "paths": {"pdf_dir": "data/pdfs", "parsed_dir": "data/parsed"},
            "llm": {
                "student": {
                    "model": "qwen/test-model",
                    "provider": "ollama",
                    "ollama": {
                        "ollama_base_url": "http://localhost:11434",
                    },
                },
                "teacher": {
                    "model": "qwen/test-model",
                    "provider": "transformers",
                },
            },
        }

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_teacher_token"}):
            Settings._apply_env_overrides(config_data)

        teacher_transformers = config_data["llm"]["teacher"]["transformers"]
        assert teacher_transformers["hf_token"] == "hf_teacher_token"

    def test_hf_token_injected_for_both(self):
        """Test HUGGINGFACE_TOKEN is injected for both student and teacher transformers."""
        config_data = {
            "project": {"log_level": "INFO"},
            "paths": {"pdf_dir": "data/pdfs", "parsed_dir": "data/parsed"},
            "llm": {
                "student": {
                    "model": "qwen/test-model",
                    "provider": "transformers",
                },
                "teacher": {
                    "model": "qwen/test-model",
                    "provider": "transformers",
                },
            },
        }

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_shared_token"}):
            Settings._apply_env_overrides(config_data)

        assert config_data["llm"]["student"]["transformers"]["hf_token"] == "hf_shared_token"
        assert config_data["llm"]["teacher"]["transformers"]["hf_token"] == "hf_shared_token"

    def test_hf_token_not_injected_when_no_transformers(self):
        """Test HUGGINGFACE_TOKEN is not injected when provider is not transformers."""
        config_data = {
            "project": {"log_level": "INFO"},
            "paths": {"pdf_dir": "data/pdfs", "parsed_dir": "data/parsed"},
            "llm": {
                "student": {
                    "model": "qwen/test-model",
                    "provider": "ollama",
                    "ollama": {
                        "ollama_base_url": "http://localhost:11434",
                    },
                },
                "teacher": {
                    "model": "qwen/test-model",
                    "provider": "api",
                    "api": {"max_tokens": 4096},
                },
            },
        }

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_unused_token"}):
            Settings._apply_env_overrides(config_data)

        assert "transformers" not in config_data["llm"]["student"]
        assert "transformers" not in config_data["llm"]["teacher"]

    def test_hf_token_not_injected_when_env_missing(self):
        """Test nothing is injected when HUGGINGFACE_TOKEN is not set."""
        config_data = {
            "project": {"log_level": "INFO"},
            "paths": {"pdf_dir": "data/pdfs", "parsed_dir": "data/parsed"},
            "llm": {
                "student": {
                    "model": "qwen/test-model",
                    "provider": "transformers",
                },
                "teacher": {
                    "model": "qwen/test-model",
                    "provider": "transformers",
                },
            },
        }

        env_without_token = {k: v for k, v in os.environ.items() if k != "HUGGINGFACE_TOKEN"}
        with patch.dict(os.environ, env_without_token, clear=True):
            Settings._apply_env_overrides(config_data)

        # transformers sections should not have hf_token added
        student_transformers = config_data["llm"]["student"].get("transformers", {})
        teacher_transformers = config_data["llm"]["teacher"].get("transformers", {})
        assert "hf_token" not in student_transformers
        assert "hf_token" not in teacher_transformers

    def test_hf_token_strips_whitespace(self):
        """Test HUGGINGFACE_TOKEN is stripped before injection."""
        config_data = {
            "project": {"log_level": "INFO"},
            "paths": {"pdf_dir": "data/pdfs", "parsed_dir": "data/parsed"},
            "llm": {
                "student": {
                    "model": "qwen/test-model",
                    "provider": "transformers",
                },
            },
        }

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "  hf_token_with_spaces  "}):
            Settings._apply_env_overrides(config_data)

        assert config_data["llm"]["student"]["transformers"]["hf_token"] == "hf_token_with_spaces"
