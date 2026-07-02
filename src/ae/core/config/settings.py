"""Configuration settings for Adaptive Extractor.

This is a facade settings module that aggregates configuration sections from:
- paths_settings.py (ProjectConfig, PathsConfig)
- llm_settings.py (LLMConfig, LLMInstanceConfig, etc.)
- optimization_settings.py (IngestionConfig, OptimizationConfig, etc.)

Loading settings is delegated to loader.py.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ae.core.config.paths_settings import ProjectConfig, PathsConfig
from ae.core.config.llm_settings import (
    OllamaConfig,
    OllamaStudentConfig,
    OllamaTeacherConfig,
    OpenRouterServiceProviderPreferences,
    ApiConfig,
    LLMInstanceConfig,
    LLMConfig,
    KNOWN_OPENROUTER_PROVIDERS,
)
from ae.core.config.optimization_settings import (
    MinerUParserConfig,
    ChartExtractionConfig,
    IngestionConfig,
    OptimizationConfig,
    ExtractionConfig,
    CacheConfig,
    CircuitBreakerConfig,
    REConfig,
)
from ae.core.config.loader import load_settings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Main application settings with environment variable support."""
    version: int = Field(
        ...,
        description="Configuration schema version"
    )
    project: ProjectConfig
    paths: PathsConfig
    llm: LLMConfig
    parsing: IngestionConfig
    optimization: OptimizationConfig
    extraction: ExtractionConfig
    cache: CacheConfig
    circuit_breaker: CircuitBreakerConfig
    re: Optional[REConfig] = None

    # Infrastructure settings from environment variables only
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking URI (from MLFLOW_TRACKING_URI env var)"
    )
    dspy_cache_dir: Optional[str] = Field(
        default=None,
        description="DSPy cache directory (from DSPY_CACHE_DIR env var)"
    )

    # API keys for non-Ollama providers (read from env vars)
    gemini_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Google Gemini API key (from GEMINI_API_KEY env var)"
    )
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key (from OPENAI_API_KEY env var)"
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key (from ANTHROPIC_API_KEY env var)"
    )
    openrouter_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenRouter API key (from OPENROUTER_API_KEY env var)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=()
    )

    @field_validator("mlflow_tracking_uri", mode="before")
    @classmethod
    def resolve_mlflow_path(cls, v: Optional[str]) -> Optional[str]:
        """Resolve relative MLflow SQLite path to absolute."""
        if not v:
            return None

        if v.startswith("sqlite:///") and not v.startswith("sqlite:////"):
            db_filename = v.replace("sqlite:///", "", 1)
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            db_path = project_root / db_filename
            return f"sqlite:///{db_path}"

        return v

    @field_validator("*", mode="before")
    @classmethod
    def validate_not_empty(cls, v: Any) -> Any:
        """Validate that string values are not empty strings."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    def __repr__(self) -> str:
        """Safe representation that hides sensitive fields."""
        safe_fields = {}
        sensitive_patterns = ("api_key", "secret", "password", "token", "key")

        for field_name, value in self.__dict__.items():
            if any(pattern in field_name.lower() for pattern in sensitive_patterns):
                safe_fields[field_name] = "***REDACTED***" if value is not None else None
            else:
                safe_fields[field_name] = value

        return f"{self.__class__.__name__}({safe_fields})"

    @classmethod
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        load_env_file: bool = True,
    ) -> "Settings":
        """Load settings from modular YAML configuration directory."""
        config_data = load_settings(config_path=config_path, load_env_file=load_env_file)
        return cls(**config_data)
