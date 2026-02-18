# src/aee/infrastructure/config/settings.py
"""Configuration settings for AutoEvoExtractor.

Configuration priority (highest to lowest):
1. Environment variables (.env file, processed by pydantic-settings)
2. YAML configuration files (config/default.yaml, config/*.yaml)
3. Internal defaults (defined in Field default_factory)

Security notes:
- API keys MUST be set via environment variables only (never in YAML)
- Infrastructure URLs (OLLAMA_*, MLFLOW_*) should be in .env for environment portability
- Application parameters belong in YAML files for version control
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    """Project-level configuration settings."""
    name: str = "autoevoextractor"
    log_level: str = Field(
        default="INFO",
        description="Logging level (from YAML or LOG_LEVEL env var)"
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if isinstance(v, str):
            v = v.upper()
            if v not in valid_levels:
                raise ValueError(
                    f"Invalid log level: {v}. Must be one of {valid_levels}"
                )
        return v


class PathsConfig(BaseModel):
    """File system paths configuration."""
    pdf_dir: Path = Field(
        default_factory=lambda: Path("data/pdf"),
        description="Directory containing PDF files to process"
    )
    parsed_dir: Path = Field(
        default_factory=lambda: Path("data/parsed"),
        description="Directory for parsed document outputs"
    )
    ground_truth_dir: Path = Field(
        default_factory=lambda: Path("data/ground_truth"),
        description="Directory containing ground truth data"
    )
    splits_file: Path = Field(
        ...,
        description="Path to JSON file with data splits (train/val/test)"
    )
    agents_dir: Path = Field(
        default_factory=lambda: Path("data/agents"),
        description="Directory for storing trained agents"
    )
    extractions_dir: Path = Field(
        default_factory=lambda: Path("data/extractions"),
        description="Directory for extraction outputs"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path("logs"),
        description="Directory for log files"
    )

    @field_validator("*", mode="before")
    @classmethod
    def cast_to_path(cls, v: Any) -> Path:
        """Cast input value to Path object."""
        return Path(v) if v else v


class OllamaConfig(BaseModel):
    """Ollama-specific configuration.

    Environment variables (set in .env, NOT in YAML):
        OLLAMA_STUDENT_BASE_URL: Ollama server URL for student model
        OLLAMA_TEACHER_BASE_URL: Ollama server URL for teacher model
        OLLAMA_BASE_URL: Fallback URL for both (if specific ones not set)

    YAML configuration (config/default.yaml):
        ollama_base_url, num_ctx, num_predict, repeat_penalty, stream: Model-specific parameters
    """
    ollama_base_url: str = Field(
        ...,
        description="Ollama base URL (from YAML or OLLAMA_*_BASE_URL env var)"
    )
    num_ctx: int = Field(
        ...,
        description="Context window size for Ollama model"
    )
    num_predict: int = Field(
        ...,
        description="Maximum number of tokens to predict"
    )
    repeat_penalty: float = Field(
        ...,
        description="Penalty for repeated tokens"
    )
    repeat_last_n: int = Field(
        ...,
        description="Number of tokens to consider for repeat penalty"
    )
    stream: bool = Field(
        ...,
        description="Enable streaming responses"
    )


class OllamaStudentConfig(OllamaConfig):
    """Ollama configuration for student model with dedicated env var."""
    # URL is overridden by load() method from OLLAMA_STUDENT_BASE_URL env var
    pass


class OllamaTeacherConfig(OllamaConfig):
    """Ollama configuration for teacher model with dedicated env var."""
    # URL is overridden by load() method from OLLAMA_TEACHER_BASE_URL env var
    pass


class NonOllamaConfig(BaseModel):
    """Non-Ollama LLM configuration.

    Environment variables:
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key
        GEMINI_API_KEY: Google Gemini API key

    YAML configuration (config/default.yaml):
        max_tokens: Maximum tokens for non-Ollama providers
    """
    # API key from environment - pydantic-settings will automatically
    # map field name to uppercase env var (e.g., api_key -> API_KEY)
    # For specific providers, use explicit env var names
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for non-Ollama providers (from environment)"
    )
    max_tokens: int = Field(
        ...,
        description="Maximum tokens for non-Ollama providers"
    )


class LLMInstanceConfig(BaseModel):
    """Configuration for a single LLM instance."""
    use_ollama: bool = Field(
        ...,
        description="Whether to use Ollama provider (True) or external API (False)"
    )
    model: str = Field(
        ...,
        description="Model name/identifier"
    )
    timeout: int = Field(
        ...,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        ...,
        description="Maximum number of retry attempts"
    )
    temperature: float = Field(
        ...,
        description="Sampling temperature for generation"
    )
    rate_limit_delay: float = Field(
        ...,
        description="Delay in seconds between API calls for rate limiting"
    )
    top_p: float = Field(
        ...,
        description="Nucleus sampling top-p parameter"
    )
    repeat_penalty: float = Field(
        ...,
        description="Penalty for repeated tokens"
    )
    repeat_last_n: int = Field(
        ...,
        description="Number of tokens to consider for repeat penalty"
    )
    enable_cache: bool = Field(
        ...,
        description="Enable LLM response caching"
    )

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    non_ollama: NonOllamaConfig = Field(default_factory=NonOllamaConfig)


class LLMConfig(BaseModel):
    """Configuration for LLM instances."""
    student: LLMInstanceConfig = Field(
        default_factory=lambda: LLMInstanceConfig(
            ollama=OllamaStudentConfig()
        )
    )
    teacher: LLMInstanceConfig = Field(
        default_factory=lambda: LLMInstanceConfig(
            ollama=OllamaTeacherConfig()
        )
    )


class DoclingConfig(BaseModel):
    """Docling parser configuration."""
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_threads: int = 4
    do_ocr: bool = True
    do_table_structure: bool = True


class MarkerConfig(BaseModel):
    """Marker parser configuration."""
    device: Literal["cpu", "cuda"] = "cpu"


class IngestionConfig(BaseModel):
    """Document ingestion configuration."""
    parser: Literal["docling", "marker"] = Field(
        ...,
        description="Document parser to use: 'docling' or 'marker'"
    )
    overwrite: bool = Field(
        ...,
        description="Overwrite existing parsed files"
    )

    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    marker: MarkerConfig = Field(default_factory=MarkerConfig)


class OptimizationConfig(BaseModel):
    """Optimization and training configuration."""
    total_load: int = Field(
        ...,
        description="Total number of samples to load for optimization"
    )
    train_split: int = Field(
        ...,
        description="Number of samples for training split"
    )
    num_candidates: int = Field(
        ...,
        description="Number of candidate instructions to generate"
    )
    num_trials: int = Field(
        ...,
        description="Number of optimization trials to run"
    )
    max_bootstrapped_demos: int = Field(
        ...,
        description="Maximum number of bootstrapped demonstrations"
    )
    max_labeled_demos: int = Field(
        ...,
        description="Maximum number of labeled demonstrations"
    )
    minibatch: bool = Field(
        ...,
        description="Use minibatch evaluation during optimization"
    )
    minibatch_size: int = Field(
        ...,
        description="Size of minibatch for evaluation"
    )
    view_data_batch_size: int = Field(
        ...,
        description="Batch size for viewing data samples"
    )
    metric_threshold: float = Field(
        ...,
        description="Threshold metric value for optimization stopping"
    )
    init_temperature: float = Field(
        ...,
        description="Initial temperature for candidate generation"
    )
    random_seed: int = Field(
        ...,
        description="Random seed for reproducibility"
    )
    use_cache: bool = Field(
        ...,
        description="Enable caching during optimization"
    )
    verbose: bool = Field(
        ...,
        description="Enable verbose logging during optimization"
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    float_tolerance: float = Field(
        ...,
        description="Tolerance for comparing floating-point values"
    )
    compare_fields: List[str] = Field(
        ...,
        description="List of field names to compare during evaluation"
    )


class TaskConfig(BaseModel):
    """Task-specific configuration."""
    name: str = Field(
        ...,
        description="Task name identifier"
    )
    initial_instruction_file: Optional[str] = Field(
        default=None,
        description="Path to initial instruction file (optional)"
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @field_validator("initial_instruction_file", mode="before")
    @classmethod
    def validate_initial_instruction_file(cls, v: Optional[str]) -> Optional[str]:
        """Validate that initial_instruction_file exists if provided."""
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Initial instruction file not found: {v}")
        return v


class ExtractionConfig(BaseModel):
    """Extraction configuration."""
    enable_cache: bool = Field(
        ...,
        description="Enable LLM response caching during extraction"
    )


class CacheConfig(BaseModel):
    """DSPy cache configuration."""
    disk_size_limit_bytes: int = Field(
        ...,
        description="Maximum disk cache size in bytes"
    )
    memory_max_entries: int = Field(
        ...,
        description="Maximum number of entries in memory cache"
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for LLM calls."""
    failure_threshold: int = Field(
        ...,
        description="Number of failures before opening circuit"
    )
    reset_timeout: float = Field(
        ...,
        description="Seconds to wait before attempting reset (half-open state)"
    )
    half_open_max_calls: int = Field(
        ...,
        description="Maximum test calls allowed in half-open state"
    )


class Settings(BaseSettings):
    """Main application settings with environment variable support.

    Configuration priority (highest to lowest):
        1. Environment variables from .env file (secrets, infrastructure)
        2. YAML configuration files (application parameters)
        3. Internal defaults (fallback values)

    Environment variables (set in .env):
        # API Keys (required for non-Ollama providers)
        GEMINI_API_KEY: Google Gemini API key
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key

        # Infrastructure URLs (environment-specific)
        OLLAMA_STUDENT_BASE_URL: Ollama server URL for student model
        OLLAMA_TEACHER_BASE_URL: Ollama server URL for teacher model
        OLLAMA_BASE_URL: Fallback Ollama server URL (if specific ones not set)
        MLFLOW_TRACKING_URI: MLflow tracking URI (e.g., sqlite:///mlflow.db)
        DSPY_CACHE_DIR: DSPy cache directory (default: ~/.dspy_cache)
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    YAML configuration (config/default.yaml):
        # All other application settings:
        # - LLM models, temperatures, timeouts
        # - File paths
        # - Optimization parameters
        # - Task-specific settings
    """
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    parsing: IngestionConfig = Field(default_factory=IngestionConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    # Infrastructure settings from environment variables only
    # These are read directly from env and not overridden by YAML
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking URI (from MLFLOW_TRACKING_URI env var)"
    )
    dspy_cache_dir: Optional[str] = Field(
        default_factory=lambda: os.getenv("DSPY_CACHE_DIR"),
        description="DSPy cache directory (from DSPY_CACHE_DIR env var)"
    )

    # API keys for non-Ollama providers (read from env vars)
    gemini_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="Google Gemini API key (from GEMINI_API_KEY env var)"
    )
    openai_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key (from OPENAI_API_KEY env var)"
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API key (from ANTHROPIC_API_KEY env var)"
    )

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Disabled to allow YAML fields to override settings.
        # Secrets are protected via SecretStr type and safe __repr__ implementation.
        protected_namespaces=()
    )

    @field_validator("mlflow_tracking_uri", mode="before")
    @classmethod
    def resolve_mlflow_path(cls, v: Optional[str]) -> Optional[str]:
        """Resolve relative MLflow SQLite path to absolute.

        Converts sqlite:///mlflow.db → sqlite:///absolute/path/to/project/mlflow.db

        This ensures the database is always created in the project root,
        regardless of the current working directory when running scripts.

        Args:
            v: Raw value from environment variable.

        Returns:
            Resolved SQLite URI with absolute path.
        """
        if not v:
            return None

        # Only process sqlite:/// relative paths (not absolute sqlite:////)
        if v.startswith("sqlite:///") and not v.startswith("sqlite:////"):
            db_filename = v.replace("sqlite:///", "", 1)
            # Calculate project root (5 levels up from this file)
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            db_path = project_root / db_filename
            return f"sqlite:///{db_path}"

        return v

    @field_validator("*", mode="before")
    @classmethod
    def validate_not_empty(cls, v: Any) -> Any:
        """Validate that string values are not empty strings.

        Empty strings from .env are converted to None for optional fields.
        """
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
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "Settings":
        """Load settings with the following priority:
        
        1. Default YAML (relative to this file)
        2. Custom YAML (if provided)
        3. Environment variables (handled by Pydantic)
        
        Args:
            config_path: Path to custom configuration YAML file.
            
        Returns:
            Settings: Loaded settings instance.
        """
        # Calculate base directory correctly (project root)
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        default_path = base_dir / "config" / "default.yaml"

        config_data = {}

        # Load default config if it exists
        if default_path.exists():
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded default configuration from {default_path}")
            except Exception as e:
                logger.warning(f"Failed to load default config from {default_path}: {e}")
                config_data = {}
        else:
            logger.warning(f"Default config not found at {default_path}. Using internal defaults.")

        # Load custom config if provided
        if config_path:
            custom_path = Path(config_path)
            # If path is not absolute, resolve it relative to config/ directory
            if not custom_path.is_absolute():
                custom_path = base_dir / "config" / custom_path
            if custom_path.exists():
                try:
                    with open(custom_path, "r", encoding="utf-8") as f:
                        custom_data = yaml.safe_load(f) or {}
                        cls._deep_update(config_data, custom_data)
                    logger.info(f"Loaded custom configuration from {custom_path}")
                except Exception as e:
                    logger.warning(f"Failed to load custom config from {custom_path}: {e}")
            else:
                logger.warning(f"Custom config {custom_path} not found. Using defaults.")

        # Set default task config if not present
        if "task" not in config_data:
            config_data["task"] = {
                "name": "nanozymes",
                "evaluation": {
                    "float_tolerance": 0.05,
                    "compare_fields": []
                }
            }

        # Resolve all paths relative to project root
        config_data = cls._resolve_paths(config_data, base_dir)

        # Apply environment variables that should override YAML config
        # This ensures .env takes priority over config/default.yaml
        cls._apply_env_overrides(config_data)

        return cls(**config_data)

    @classmethod
    def _apply_env_overrides(cls, config_data: dict) -> None:
        """Apply environment variable overrides to config data.

        Environment variables have higher priority than YAML configuration.
        This method modifies config_data in place.

        Env vars applied:
            - OLLAMA_STUDENT_BASE_URL: Student Ollama server URL
            - OLLAMA_TEACHER_BASE_URL: Teacher Ollama server URL
            - OLLAMA_BASE_URL: Fallback Ollama server URL
            - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Args:
            config_data: Configuration dictionary to update.
        """
        # Log level from environment (higher priority than YAML)
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            if "project" not in config_data:
                config_data["project"] = {}
            config_data["project"]["log_level"] = log_level.upper()

        # Ollama URLs from .env (infrastructure settings)
        ollama_student_url = os.getenv("OLLAMA_STUDENT_BASE_URL")
        ollama_teacher_url = os.getenv("OLLAMA_TEACHER_BASE_URL")
        ollama_fallback_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        if ollama_student_url:
            if "llm" not in config_data:
                config_data["llm"] = {}
            if "student" not in config_data["llm"]:
                config_data["llm"]["student"] = {}
            if "ollama" not in config_data["llm"]["student"]:
                config_data["llm"]["student"]["ollama"] = {}
            config_data["llm"]["student"]["ollama"]["ollama_base_url"] = ollama_student_url
        elif "llm" in config_data and "student" in config_data["llm"]:
            # Apply fallback if student URL not set
            if "ollama" not in config_data["llm"]["student"]:
                config_data["llm"]["student"]["ollama"] = {}
            config_data["llm"]["student"]["ollama"]["ollama_base_url"] = ollama_fallback_url

        if ollama_teacher_url:
            if "llm" not in config_data:
                config_data["llm"] = {}
            if "teacher" not in config_data["llm"]:
                config_data["llm"]["teacher"] = {}
            if "ollama" not in config_data["llm"]["teacher"]:
                config_data["llm"]["teacher"]["ollama"] = {}
            config_data["llm"]["teacher"]["ollama"]["ollama_base_url"] = ollama_teacher_url
        elif "llm" in config_data and "teacher" in config_data["llm"]:
            # Apply fallback if teacher URL not set
            if "ollama" not in config_data["llm"]["teacher"]:
                config_data["llm"]["teacher"]["ollama"] = {}
            config_data["llm"]["teacher"]["ollama"]["ollama_base_url"] = ollama_fallback_url

    @classmethod
    def _resolve_paths(cls, config_data: dict, base_dir: Path) -> dict:
        """Resolve path values relative to project root.

        Recursively processes the config dictionary and converts values
        that look like file paths (strings containing '/' or ending with
        common file extensions) to absolute paths.

        Args:
            config_data: Configuration dictionary.
            base_dir: Project root directory.

        Returns:
            Configuration dictionary with resolved paths.
        """
        import re

        def is_path_like(value: str) -> bool:
            """Check if a string looks like a file path."""
            if not isinstance(value, str):
                return False
            # Skip empty strings
            if not value:
                return False
            # Skip URLs (http://, https://, ftp://, etc.)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', value):
                return False
            # Check for path separators or file extensions
            return '/' in value or value.endswith(('.txt', '.yaml', '.yml', '.json', '.csv', '.md'))

        def resolve_value(value: Any) -> Any:
            """Resolve a single value if it's a path."""
            if is_path_like(value):
                path = Path(value)
                # Don't modify absolute paths
                if path.is_absolute():
                    return value
                # Resolve relative paths against project root
                resolved = base_dir / path
                return str(resolved)
            return value

        def process_dict(d: dict) -> dict:
            """Recursively process dictionary."""
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = process_dict(v)
                elif isinstance(v, list):
                    result[k] = [resolve_value(item) if isinstance(item, str) else item for item in v]
                elif isinstance(v, str):
                    result[k] = resolve_value(v)
                else:
                    result[k] = v
            return result

        return process_dict(config_data)

    @staticmethod
    def _deep_update(base_dict: dict, update_with: dict) -> None:
        """Recursively update a dictionary with another dictionary.
        
        Args:
            base_dict: The dictionary to update.
            update_with: The dictionary to update with.
        """
        for k, v in update_with.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                Settings._deep_update(base_dict[k], v)
            else:
                base_dict[k] = v


# Initialize settings
settings = Settings.load()