"""Unit tests for Ingestion v2 configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ae.core.config.optimization_settings import (
    ApiClientConfig,
    CropConfig,
    ExecutionConfig,
    GeminiDefaultsConfig,
    GeminiParserConfig,
    IngestionConfig,
    MoleculeValidationConfig,
    StageModelConfig,
)


class TestIngestionConfigV2:
    def test_default_config_instantiation(self):
        """Test default config initialization with all explicit defaults."""
        cfg = IngestionConfig()
        assert cfg.execution.concurrency.max_documents == 2
        assert cfg.execution.concurrency.max_requests_per_document == 8
        assert cfg.execution.overwrite is False
        assert cfg.execution.from_stage is None
        assert cfg.execution.save_llm_history is False

        assert cfg.api_client.timeout == 300
        assert cfg.api_client.max_retries == 5
        assert cfg.api_client.retry_delay == 2.0

        assert cfg.stages.parse_text.model.model == "gemini-3.8-flash"
        assert cfg.stages.parse_text.model.thinking_level == "low"

        assert cfg.stages.localize.page_dpi == 150
        assert cfg.stages.localize.model.model == "gemini-3.8-flash"

        assert cfg.stages.crop.base_dpi == 300
        assert cfg.stages.crop.padding_ratio == 0.05
        assert cfg.stages.crop.upscale.enabled is True
        assert cfg.stages.crop.upscale.min_long_side_px == 2000
        assert cfg.stages.crop.upscale.max_dpi == 1200

        assert cfg.stages.parse_visuals.molecule.validation.validate_rdkit is True
        assert cfg.stages.parse_visuals.molecule.validation.max_retries == 1

        # Backward compatibility properties
        assert cfg.concurrency == 2
        assert cfg.overwrite is False
        assert cfg.engine == "gemini"
        assert cfg.gemini.crop.base_dpi == 300
        assert cfg.gemini.molecules.validate_rdkit is True

    def test_explicit_configuration_from_dict(self):
        """Test parsing explicit configuration from dict structure matching ingestion.yaml."""
        raw = {
            "execution": {
                "overwrite": True,
                "from_stage": "classify",
                "save_llm_history": True,
                "concurrency": {
                    "max_documents": 4,
                    "max_requests_per_document": 12,
                },
            },
            "api_client": {
                "timeout": 120,
                "max_retries": 3,
                "retry_delay": 1.5,
            },
            "stages": {
                "parse_text": {
                    "model": {
                        "provider": "gemini",
                        "model": "gemini-3.8-flash",
                        "temperature": 0.1,
                        "thinking_level": "low",
                    },
                },
                "classify": {
                    "model": {
                        "provider": "gemini",
                        "model": "gemini-3.8-flash",
                        "thinking_level": "low",
                    },
                },
                "localize": {
                    "page_dpi": 200,
                    "model": {
                        "provider": "gemini",
                        "model": "gemini-3.8-flash",
                        "thinking_level": "low",
                    },
                },
                "crop": {
                    "padding_ratio": 0.05,
                    "base_dpi": 400,
                    "upscale": {
                        "enabled": False,
                        "min_long_side_px": 1500,
                        "max_dpi": 800,
                    },
                },
                "decompose": {
                    "model": {
                        "provider": "gemini",
                        "model": "gemini-3.8-pro",
                        "thinking_level": "high",
                    },
                },
                "parse_visuals": {
                    "table": {
                        "model": {
                            "provider": "gemini",
                            "model": "gemini-3.8-flash",
                            "thinking_level": "medium",
                        },
                    },
                    "chart": {
                        "model": {
                            "provider": "gemini",
                            "model": "gemini-3.8-flash",
                            "thinking_level": "medium",
                        },
                    },
                    "molecule": {
                        "model": {
                            "provider": "gemini",
                            "model": "gemini-3.8-pro",
                            "thinking_level": "high",
                        },
                        "validation": {
                            "validate_rdkit": True,
                            "max_retries": 3,
                        },
                    },
                },
            },
        }

        cfg = IngestionConfig.model_validate(raw)
        assert cfg.execution.concurrency.max_documents == 4
        assert cfg.execution.concurrency.max_requests_per_document == 12
        assert cfg.execution.overwrite is True
        assert cfg.execution.from_stage == "classify"
        assert cfg.execution.save_llm_history is True

        assert cfg.api_client.timeout == 120
        assert cfg.api_client.max_retries == 3
        assert cfg.api_client.retry_delay == 1.5

        assert cfg.stages.parse_text.model.temperature == 0.1

        assert cfg.stages.localize.page_dpi == 200

        assert cfg.stages.crop.base_dpi == 400
        assert cfg.stages.crop.padding_ratio == 0.05
        assert cfg.stages.crop.upscale.enabled is False

        assert cfg.stages.decompose.model.model == "gemini-3.8-pro"
        assert cfg.stages.decompose.model.thinking_level == "high"

        assert cfg.stages.parse_visuals.molecule.model.model == "gemini-3.8-pro"
        assert cfg.stages.parse_visuals.molecule.validation.max_retries == 3

        # Stage resolution helper
        decomp_cfg = cfg.get_stage_config("decompose")
        assert decomp_cfg.model == "gemini-3.8-pro"
        assert decomp_cfg.thinking_level == "high"

        mol_cfg = cfg.get_stage_config("parse_visuals", sub_stage="molecule")
        assert mol_cfg.model == "gemini-3.8-pro"
        assert mol_cfg.thinking_level == "high"

    def test_legacy_structure_migration(self):
        """Test transparent backward compatibility migration from old gemini dict format."""
        raw = {
            "engine": "gemini",
            "concurrency": 4,
            "overwrite": True,
            "gemini": {
                "max_concurrent_requests": 6,
                "timeout": 180,
                "defaults": {
                    "model": "gemini-3-flash-preview",
                    "temperature": 0.0,
                },
                "stages": {
                    "parse_text": {
                        "model": "gemini-pro-transcribe",
                        "temperature": 0.2,
                    },
                    "localize": {
                        "page_dpi": 200,
                    },
                    "parse_visuals": {
                        "molecule": {
                            "thinking_level": "high",
                        },
                    },
                },
                "crop": {
                    "base_dpi": 450,
                },
                "molecules": {
                    "validate_rdkit": False,
                    "max_rdkit_retries": 0,
                },
            },
        }

        cfg = IngestionConfig.model_validate(raw)
        assert cfg.execution.concurrency.max_documents == 4
        assert cfg.execution.concurrency.max_requests_per_document == 6
        assert cfg.execution.overwrite is True
        assert cfg.api_client.timeout == 180

        assert cfg.stages.localize.page_dpi == 200
        assert cfg.stages.crop.base_dpi == 450
        assert cfg.stages.parse_visuals.molecule.validation.validate_rdkit is False

        t_cfg = cfg.get_stage_config("parse_text")
        assert t_cfg.model == "gemini-pro-transcribe"
        assert t_cfg.temperature == 0.2

        c_cfg = cfg.get_stage_config("classify")
        assert c_cfg.model == "gemini-3-flash-preview"

        mol_cfg = cfg.get_stage_config("parse_visuals", sub_stage="molecule")
        assert mol_cfg.thinking_level == "high"

    def test_invalid_types_raise_validation_error(self):
        """Test validation constraints on invalid fields."""
        with pytest.raises(ValidationError):
            IngestionConfig(
                execution=ExecutionConfig(
                    concurrency={"max_documents": -1}  # type: ignore[arg-type]
                )
            )

        with pytest.raises(ValidationError):
            CropConfig(padding_ratio="not_a_float")  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            MoleculeValidationConfig(max_retries=-5)

        with pytest.raises(ValidationError):
            StageModelConfig(model="")  # Empty model name must raise ValidationError

    def test_thinking_budget_configuration(self):
        """Test explicit thinking_budget settings per stage."""
        raw = {
            "stages": {
                "parse_text": {
                    "model": {
                        "model": "gemini-3.8-flash",
                        "thinking_budget": 0,
                    },
                },
                "parse_visuals": {
                    "chart": {
                        "model": {
                            "model": "gemini-3.8-flash",
                            "thinking_budget": 256,
                        },
                    },
                    "molecule": {
                        "model": {
                            "model": "gemini-3.8-flash",
                            "thinking_budget": 3072,
                        },
                    },
                },
            },
        }

        cfg = IngestionConfig.model_validate(raw)
        t_cfg = cfg.get_stage_config("parse_text")
        assert t_cfg.thinking_budget == 0

        chart_cfg = cfg.get_stage_config("parse_visuals", sub_stage="chart")
        assert chart_cfg.thinking_budget == 256

        mol_cfg = cfg.get_stage_config("parse_visuals", sub_stage="molecule")
        assert mol_cfg.thinking_budget == 3072

    def test_classify_render_page_max_size_preserved(self):
        """ClassifyStageConfig must preserve render_page_max_size from config."""
        cfg = IngestionConfig.model_validate({"stages": {"classify": {"render_page_max_size": 1024}}})
        assert cfg.stages.classify.render_page_max_size == 1024
