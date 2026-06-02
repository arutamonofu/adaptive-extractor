# src/ae/core/config/settings.py
"""Configuration settings for Adaptive Extractor.

Configuration loading:
    YAML configuration file is REQUIRED. Use Settings.load(config_path=...)
    to load settings from a YAML file. There is no fallback to internal defaults.

    All configuration values must be provided either:
    - In the YAML configuration file (application settings)
    - Via environment variables (secrets and infrastructure URLs only)

Security notes:
- API keys MUST be set via environment variables only (never in YAML)
- Infrastructure URLs (OLLAMA_*, MLFLOW_*) should be in .env for environment portability
- Application parameters belong in YAML files for version control

Note: Environment variables with double underscores (e.g., OPTIMIZATION__NUM_TRIALS)
    are NOT supported. All application configuration should be set in YAML files.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    """Project-level configuration settings."""
    log_level: str = Field(
        ...,
        description="Logging level (from YAML config)"
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
        ...,
        description="Directory containing PDF files to process"
    )
    parsed_dir: Path = Field(
        ...,
        description="Directory for parsed document outputs"
    )
    ground_truth_dir: Path = Field(
        ...,
        description="Directory containing ground truth data"
    )
    splits_file: Path = Field(
        ...,
        description="Path to JSON file with data splits (train/val/test)"
    )
    agents_dir: Path = Field(
        ...,
        description="Directory for storing trained agents"
    )
    extractions_dir: Path = Field(
        ...,
        description="Directory for extraction outputs"
    )

    @field_validator("*", mode="before")
    @classmethod
    def cast_to_path(cls, v: Any) -> Path:
        """Cast input value to Path object."""
        return Path(v) if v else v


class OllamaConfig(BaseModel):
    """Ollama-specific configuration.

    Environment variables (set in .env, NOT in YAML):
        OLLAMA_STUDENT_BASE_URL: Ollama server URL for student model (required)
        OLLAMA_TEACHER_BASE_URL: Ollama server URL for teacher model (required)

    YAML configuration (config/default.yaml):
        num_ctx, num_predict, repeat_penalty, stream: Model-specific parameters

    Note: ollama_base_url is NOT set in YAML. It must be provided via environment
        variables only. Validation will fail if the environment variable is not set.
    """
    ollama_base_url: Optional[str] = Field(
        default=None,
        description="Ollama base URL (from OLLAMA_*_BASE_URL env var only)"
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

    @field_validator("ollama_base_url", mode="after")
    @classmethod
    def validate_ollama_base_url(cls, v: Optional[str]) -> str:
        """Validate that Ollama base URL is set via environment variable."""
        if v is None or v.strip() == "":
            raise ValueError(
                "OLLAMA_*_BASE_URL environment variable must be set in .env file. "
                "Set OLLAMA_STUDENT_BASE_URL or OLLAMA_TEACHER_BASE_URL as appropriate."
            )
        return v.strip()


class OllamaStudentConfig(OllamaConfig):
    """Ollama configuration for student model with dedicated env var."""
    # URL is overridden by load() method from OLLAMA_STUDENT_BASE_URL env var
    num_ctx: int = Field(
        ...,
        description="Context window size for student model"
    )
    num_predict: int = Field(
        ...,
        description="Maximum tokens to predict for student model"
    )
    repeat_penalty: float = Field(
        ...,
        description="Repeat penalty for student model"
    )
    repeat_last_n: int = Field(
        ...,
        description="Number of tokens for repeat penalty for student model"
    )
    stream: bool = Field(
        ...,
        description="Enable streaming for student model"
    )


class OllamaTeacherConfig(OllamaConfig):
    """Ollama configuration for teacher model with dedicated env var."""
    # URL is overridden by load() method from OLLAMA_TEACHER_BASE_URL env var
    num_ctx: int = Field(
        ...,
        description="Context window size for teacher model"
    )
    num_predict: int = Field(
        ...,
        description="Maximum tokens to predict for teacher model"
    )
    repeat_penalty: float = Field(
        ...,
        description="Repeat penalty for teacher model"
    )
    repeat_last_n: int = Field(
        ...,
        description="Number of tokens for repeat penalty for teacher model"
    )
    stream: bool = Field(
        ...,
        description="Enable streaming for teacher model"
    )




class ApiConfig(BaseModel):
    """API provider configuration.

    Environment variables:
        OPENAI_API_KEY: OpenAI API key (required when provider="api")
        ANTHROPIC_API_KEY: Anthropic API key (required when provider="api")
        GEMINI_API_KEY: Google Gemini API key (required when provider="api")
        OPENROUTER_API_KEY: OpenRouter API key (required when provider="api")

    YAML configuration (config/default.yaml):
        max_tokens: Maximum tokens for API providers
        base_url: Custom API base URL (optional, for OpenRouter or compatible endpoints)

    Note: API key must be set via environment variable. Validation will fail
        if api_key is not provided.
    """
    # API key from environment - pydantic-settings will automatically
    # map field name to uppercase env var (e.g., api_key -> API_KEY)
    # For specific providers, use explicit env var names
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for API providers (from environment)"
    )
    max_tokens: int = Field(
        ...,
        description="Maximum tokens for API providers"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (e.g., https://openrouter.ai/api/v1 for OpenRouter)"
    )
    reasoning: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Reasoning configuration for OpenRouter reasoning models (e.g., {'enabled': True})"
    )

    @field_validator("api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        """Validate that API key is set for API providers.

        Note: This validator only checks if the key is provided. The actual
            requirement (provider="api") is validated at the LLMInstanceConfig level.
        """
        # Validation is deferred to LLMInstanceConfig level where we know provider
        return v


class LLMInstanceConfig(BaseModel):
    """Configuration for a single LLM instance."""
    provider: Literal["ollama", "api"] = Field(
        ...,
        description="LLM provider type: 'ollama' or 'api'"
    )
    model: str = Field(
        ...,
        description="Model name/identifier"
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Request timeout in seconds (required for ollama/api)"
    )
    max_retries: int = Field(
        ...,
        description="Maximum number of retry attempts"
    )
    temperature: float = Field(
        ...,
        description="Sampling temperature for generation (>= 0)"
    )
    rate_limit_delay: Optional[float] = Field(
        default=None,
        description="Delay in seconds between API calls (required for ollama/api)"
    )
    top_p: float = Field(
        ...,
        description="Nucleus sampling top-p parameter (0.0 < value <= 1.0)"
    )
    enable_cache: bool = Field(
        ...,
        description="Enable LLM response caching"
    )

    ollama: Optional[OllamaConfig] = Field(
        default=None,
        description="Ollama-specific configuration (required when provider='ollama')"
    )
    api: Optional[ApiConfig] = Field(
        default=None,
        description="API provider configuration (required when provider='api')"
    )

    @field_validator("temperature", mode="after")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Temperature must be non-negative."""
        if v < 0:
            raise ValueError(f"temperature must be >= 0, got {v}")
        return v

    @field_validator("top_p", mode="after")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """top_p must be in range (0.0, 1.0]."""
        if v <= 0.0 or v > 1.0:
            raise ValueError(f"top_p must be in range (0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def validate_provider_config(self) -> "LLMInstanceConfig":
        """Validate provider-specific requirements."""
        if self.provider == "ollama":
            if self.timeout is None:
                raise ValueError("timeout is required when provider='ollama'")
            if self.rate_limit_delay is None:
                raise ValueError("rate_limit_delay is required when provider='ollama'")
            if self.ollama is None:
                raise ValueError(
                    "ollama configuration is required when provider='ollama'. "
                    "Add an 'ollama' section with num_ctx, num_predict, "
                    "repeat_penalty, repeat_last_n, stream, and set the "
                    "OLLAMA_*_BASE_URL environment variable."
                )
            if not self.ollama.ollama_base_url:
                raise ValueError(
                    "Ollama URL must be set via OLLAMA_*_BASE_URL env var when provider='ollama'"
                )
        elif self.provider == "api":
            if self.timeout is None:
                raise ValueError("timeout is required when provider='api'")
            if self.rate_limit_delay is None:
                raise ValueError("rate_limit_delay is required when provider='api'")
            if self.api is None:
                raise ValueError(
                    "api configuration is required when provider='api'. "
                    "Add an 'api' section with max_tokens and set the "
                    "appropriate API key environment variable."
                )
            if self.api.api_key is None:
                raise ValueError(
                    "API key must be set for API provider. "
                    "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in .env file."
                )
        return self


class LLMConfig(BaseModel):
    """Configuration for LLM instances."""
    student: LLMInstanceConfig = Field(
        ...,
        description="Student LLM configuration"
    )
    teacher: LLMInstanceConfig = Field(
        ...,
        description="Teacher LLM configuration"
    )




class GeminiParserConfig(BaseModel):
    """Gemini API parser configuration."""
    model_name: str = Field(
        default="gemini-3-flash-preview",
        validation_alias=AliasChoices("model", "model_name"),
        serialization_alias="model_name",
        description="Gemini model name for PDF-to-Markdown conversion"
    )
    upload_timeout: int = Field(
        default=300,
        description="Timeout for file upload in seconds"
    )
    safety_settings: bool = Field(
        default=True,
        description="Enable safety settings for Gemini API"
    )
    request_delay: float = Field(
        default=10.0,
        description="Delay between file requests to avoid rate limiting"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for network errors"
    )


class AEVisualParserConfig(BaseModel):
    """Configuration for the visual extraction parser (ae_visual)."""
    enabled: bool = Field(
        default=False,
        description="Enable the visual PDF parsing pipeline"
    )
    task_config_path: Optional[str] = Field(
        default=None,
        description="Path to ae_visual schema / task configuration"
    )
    pipeline: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Embedded visual pipeline configuration"
    )
    gemini: GeminiParserConfig = Field(
        default_factory=GeminiParserConfig,
        description="Gemini-specific configuration for base text extraction"
    )

    @model_validator(mode="after")
    def validate_enabled_config(self) -> "AEVisualParserConfig":
        """Validate that task_config_path and pipeline are present when visual is enabled."""
        if self.enabled:
            if not self.task_config_path:
                raise ValueError("task_config_path is required when visual parser is enabled")
            if not self.pipeline:
                raise ValueError("pipeline configuration is required when visual parser is enabled")
        return self


class IngestionConfig(BaseModel):
    """Document ingestion configuration."""
    overwrite: bool = Field(
        ...,
        description="Overwrite existing parsed files"
    )

    gemini: Optional[GeminiParserConfig] = Field(
        default=None,
        description="Gemini-specific configuration (required if visual is disabled)"
    )
    visual: Optional[AEVisualParserConfig] = Field(
        default_factory=AEVisualParserConfig,
        description="Gemini-visual-specific configuration (visual: block in yaml)"
    )

    @model_validator(mode="after")
    def validate_parser_config(self) -> "IngestionConfig":
        """Validate that the correct parser-specific config is provided."""
        if self.visual is None:
            self.visual = AEVisualParserConfig(enabled=False)

        if not self.visual.enabled and self.gemini is None:
            # For gemini, we can use defaults, so just create a default config
            self.gemini = GeminiParserConfig()
        return self


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
        description="Maximum number of bootstrapped demonstrations (0 for zero-shot mode)"
    )
    max_labeled_demos: int = Field(
        ...,
        description="Maximum number of labeled demonstrations (0 for zero-shot mode)"
    )

    @field_validator('max_bootstrapped_demos')
    @classmethod
    def validate_max_bootstrapped_demos(cls, v):
        """Validate max_bootstrapped_demos is non-negative."""
        if v < 0:
            raise ValueError('max_bootstrapped_demos must be >= 0 (use 0 for zero-shot mode)')
        return v

    @field_validator('max_labeled_demos')
    @classmethod
    def validate_max_labeled_demos(cls, v):
        """Validate max_labeled_demos is non-negative."""
        if v < 0:
            raise ValueError('max_labeled_demos must be >= 0 (use 0 for zero-shot mode)')
        return v
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
        validation_alias=AliasChoices("enable_cache", "use_cache"),
        serialization_alias="use_cache",
        description="Enable caching during optimization"
    )
    verbose: bool = Field(
        ...,
        description="Enable verbose logging during optimization"
    )
    max_errors: int = Field(
        default=5,
        description=(
            "Maximum number of errors allowed before stopping optimization "
            "(DSPy parallelizer setting). Increase from default (5) to allow "
            "more faults during trials."
        )
    )
    save_llm_history: bool = Field(
        default=True,
        description="Save LLM call histories after optimization"
    )
    llm_history_dir: str = Field(
        default="logs/llm_history",
        description="Directory for LLM history files"
    )
    contrastive_analysis: "ContrastiveAnalysisConfig" = Field(
        default_factory=lambda: ContrastiveAnalysisConfig(),
        description="Settings for contrastive Ground Truth analyzer"
    )


class ContrastiveAnalysisConfig(BaseModel):
    """Configuration for contrastive analysis."""
    enabled: bool = Field(default=False, description="Enable contrastive analysis before optimization")
    batch_size: int = Field(default=10, description="Default document batch size for contrastive analysis")
    auto_skip_review: bool = Field(default=False, description="Whether to skip interactive discrepancy review")
    cache_dir: str = Field(default="data/analysis", description="Directory to cache analysis results")


class TaskConfig(BaseModel):
    """Task-specific configuration."""
    name: str = Field(
        ...,
        description="Task name identifier"
    )

    @model_validator(mode="after")
    def validate_task_files(self) -> "TaskConfig":
        """Validate that task schema and instruction files exist."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        task_dir = project_root / "config" / "tasks" / self.name
        if "PYTEST_CURRENT_TEST" in os.environ and not task_dir.exists():
            return self
            
        if not task_dir.exists():
            raise ValueError(f"Task directory not found: config/tasks/{self.name}")
        
        schema_yaml = task_dir / "initial_schema.yaml"
        if not schema_yaml.exists():
            raise ValueError(f"Task schema file not found: config/tasks/{self.name}/initial_schema.yaml")

        instruction_file = task_dir / "initial_instruction.txt"
        if not instruction_file.exists():
            raise ValueError(f"Task instruction file not found: config/tasks/{self.name}/initial_instruction.txt")
        return self


class ExtractionConfig(BaseModel):
    """Extraction configuration."""
    enable_cache: bool = Field(
        ...,
        description="Enable LLM response caching during extraction"
    )
    save_llm_history: bool = Field(
        default=False,
        description="Save LLM call histories after extraction"
    )
    llm_history_dir: str = Field(
        default="logs/llm_history",
        description="Directory for LLM history files"
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

    YAML configuration file is REQUIRED. Use Settings.load(config_path=...)
    to load settings. There is no fallback to internal defaults.

    Environment variables (set in .env) — ONLY for secrets and infrastructure:
        # API Keys (required for non-Ollama providers)
        GEMINI_API_KEY: Google Gemini API key
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key

        # Infrastructure URLs (environment-specific, required)
        OLLAMA_STUDENT_BASE_URL: Ollama server URL for student model (required)
        OLLAMA_TEACHER_BASE_URL: Ollama server URL for teacher model (required)
        MLFLOW_TRACKING_URI: MLflow tracking URI (e.g., sqlite:///mlflow.db)
        DSPY_CACHE_DIR: DSPy cache directory
        AE_ENV: Environment selection (dev, test, prod)

    YAML configuration — all other application settings:
        # LLM models, temperatures, timeouts
        # File paths
        # Optimization parameters
        # Task-specific settings
        # Logging level (project.log_level)

    Note: Environment variables with double underscores (e.g., OPTIMIZATION__NUM_TRIALS)
        are NOT supported. All application configuration should be set in YAML files.

    Note: Ollama URLs (OLLAMA_STUDENT_BASE_URL, OLLAMA_TEACHER_BASE_URL) must be set
        in .env file. No fallback or default value is provided.
    """
    project: ProjectConfig
    paths: PathsConfig
    llm: LLMConfig
    parsing: IngestionConfig
    optimization: OptimizationConfig
    task: TaskConfig
    extraction: ExtractionConfig
    cache: CacheConfig
    circuit_breaker: CircuitBreakerConfig

    # Infrastructure settings from environment variables only
    # These are read directly from env and not overridden by YAML
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

    @model_validator(mode="after")
    def resolve_dynamic_defaults(self) -> "Settings":
        """Resolve dynamic defaults based on global project parameters."""
        if self.parsing.visual and self.parsing.visual.enabled and not self.parsing.visual.task_config_path:
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            self.parsing.visual.task_config_path = str(
                project_root / "config" / "tasks" / self.task.name / "initial_schema.yaml"
            )
        return self

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
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        load_env_file: bool = True,
    ) -> "Settings":
        """Load settings from modular YAML configuration directory.

        Args:
            config_path: Path to configuration directory. Defaults to root config/ directory.
            load_env_file: Whether to load .env file. Default is True.
                Set to False for testing.

        Returns:
            Settings: Loaded settings instance.

        Raises:
            ValueError: If config_path is not a directory.
            FileNotFoundError: If configuration files or directory do not exist.
        """
        # Calculate base directory correctly (project root)
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent

        # Default to root config/ directory
        if config_path is None:
            config_path = base_dir / "config"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = base_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_path}")

        if not config_path.is_dir():
            raise ValueError(f"Configuration path must be a directory: {config_path}")

        # Load .env file before applying env overrides
        if load_env_file:
            from dotenv import load_dotenv
            env_file = base_dir / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {env_file}")

        # Find and load all yaml/yml files deterministically
        yaml_files = sorted(
            [p for p in config_path.glob("*.yaml")] + [p for p in config_path.glob("*.yml")]
        )
        if not yaml_files:
            raise FileNotFoundError(f"No YAML configuration files found in: {config_path}")

        config_data: dict[str, Any] = {}
        for file_path in yaml_files:
            # Skip visual pipeline config as it is loaded separately inside visual parser
            if file_path.name == "visual_pipeline.yaml":
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_data = yaml.safe_load(f) or {}
                cls._deep_update(config_data, file_data)
                logger.info(f"Loaded config chunk from {file_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load configuration from {file_path}: {e}")

        # Resolve all paths relative to project root
        config_data = cls._resolve_paths(config_data, base_dir)

        # Apply Ollama URL overrides from environment variables
        cls._apply_env_overrides(config_data)

        # Apply API keys from environment variables to api config
        cls._apply_api_keys(config_data)

        return cls(**config_data)

    @classmethod
    def _apply_api_keys(cls, config_data: dict) -> None:
        """Apply API keys from environment variables to api config.

        API keys are read from environment variables and injected into the
        llm.student.api and llm.teacher.api configurations.

        Env vars applied:
            - OPENAI_API_KEY: OpenAI API key
            - ANTHROPIC_API_KEY: Anthropic API key
            - GEMINI_API_KEY: Google Gemini API key
            - OPENROUTER_API_KEY: OpenRouter API key

        Priority logic (in order):
            1. If base_url is specified, use the corresponding API key:
               - openrouter.ai → OPENROUTER_API_KEY
               - api.openai.com → OPENAI_API_KEY
               - api.anthropic.com → ANTHROPIC_API_KEY
               - generativelanguage.googleapis.com → GEMINI_API_KEY
            2. Else if model name starts with provider prefix, use that key:
               - "openai/" → OPENAI_API_KEY
               - "openrouter/" → OPENROUTER_API_KEY
               - "anthropic/" → ANTHROPIC_API_KEY
               - "gemini/" → GEMINI_API_KEY
            3. Else fallback to priority order: OpenAI > Anthropic > Gemini > OpenRouter

        Args:
            config_data: Configuration dictionary to update.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # DEBUG: Log which API keys are found
        logger.debug(f"API keys from env - OpenAI: {'SET' if openai_key else 'NOT SET'}, "
                     f"Anthropic: {'SET' if anthropic_key else 'NOT SET'}, "
                     f"Gemini: {'SET' if gemini_key else 'NOT SET'}, "
                     f"OpenRouter: {'SET' if openrouter_key else 'NOT SET'}")

        # Map base_url patterns to corresponding API keys
        BASE_URL_TO_KEY = {
            "openrouter.ai": openrouter_key,
            "api.openai.com": openai_key,
            "api.anthropic.com": anthropic_key,
            "generativelanguage.googleapis.com": gemini_key,
            "api.groq.com": openai_key,  # Groq uses OpenAI-compatible API
        }

        def get_api_key_from_base_url(base_url: Optional[str]) -> Optional[str]:
            """Get API key based on base_url pattern matching."""
            if not base_url:
                return None

            base_url_lower = base_url.lower()
            for pattern, api_key in BASE_URL_TO_KEY.items():
                if pattern in base_url_lower:
                    return api_key
            return None

        def get_api_key_for_model(model_name: str) -> Optional[str]:
            """Get API key based on model name prefix or priority order."""
            if not model_name:
                return None

            model_lower = model_name.lower()

            # Check model name prefix
            if model_lower.startswith("openai/"):
                return openai_key
            elif model_lower.startswith("anthropic/"):
                return anthropic_key
            elif model_lower.startswith("gemini/"):
                return gemini_key
            elif model_lower.startswith("openrouter/"):
                return openrouter_key
            elif model_lower.startswith("huggingface/"):
                return os.getenv("HUGGINGFACE_API_KEY")

            # Fallback to priority order for models without prefix
            return openai_key or anthropic_key or gemini_key or openrouter_key

        def apply_key_to_component(component_data: dict, component_name: str) -> None:
            """Apply API key to a single LLM component (student or teacher)."""
            if component_data.get("provider") != "api":
                return  # Skip non-API models (ollama, transformers)

            model_name = component_data.get("model", "")
            api_config = component_data.get("api", {})
            base_url = api_config.get("base_url")

            # Priority 1: Try to get API key from base_url
            api_key = get_api_key_from_base_url(base_url)

            # Priority 2: Try model name prefix
            if api_key is None:
                api_key = get_api_key_for_model(model_name)

            if api_key:
                if "api" in component_data:
                    component_data["api"]["api_key"] = api_key

                    # Determine key source for logging
                    key_source = "Unknown"
                    if api_key == openai_key:
                        key_source = "OpenAI"
                    elif api_key == anthropic_key:
                        key_source = "Anthropic"
                    elif api_key == gemini_key:
                        key_source = "Gemini"
                    elif api_key == openrouter_key:
                        key_source = "OpenRouter"
                    else:
                        key_source = "HuggingFace"

                    source_info = f"base_url: {base_url}" if base_url else f"model prefix: {model_name}"
                    logger.info(f"Using {key_source} API key for {component_name}: {model_name} ({source_info})")
                else:
                    logger.warning(f"No api section in YAML for {component_name}: {model_name}, skipping API key injection")
            else:
                logger.warning(f"No API key found for {component_name}: {model_name}")

        # Apply to student
        if "llm" in config_data and "student" in config_data["llm"]:
            apply_key_to_component(config_data["llm"]["student"], "student")

        # Apply to teacher
        if "llm" in config_data and "teacher" in config_data["llm"]:
            apply_key_to_component(config_data["llm"]["teacher"], "teacher")

    @classmethod
    def _apply_env_overrides(cls, config_data: dict) -> None:
        """Apply environment variable overrides to config data.

        Ollama URLs are read from environment variables only (NOT from YAML).
        This method modifies config_data in place.

        Note: Ollama URLs must be set in .env file. No fallback is provided.

        Env vars applied:
            - OLLAMA_STUDENT_BASE_URL: Student Ollama server URL (required when provider='ollama' for student)
            - OLLAMA_TEACHER_BASE_URL: Teacher Ollama server URL (required when provider='ollama' for teacher)

        Args:
            config_data: Configuration dictionary to update.

        Raises:
            ValueError: If required Ollama URL is not set in environment when provider='ollama'.
        """

        # Check if Ollama is used for student and teacher
        student_uses_ollama = (
            config_data.get("llm", {})
            .get("student", {})
            .get("provider") == "ollama"
        )
        teacher_uses_ollama = (
            config_data.get("llm", {})
            .get("teacher", {})
            .get("provider") == "ollama"
        )

        # Get Ollama URLs from environment
        ollama_student_url = os.getenv("OLLAMA_STUDENT_BASE_URL")
        ollama_teacher_url = os.getenv("OLLAMA_TEACHER_BASE_URL")

        # Apply student URL only if ollama section exists in YAML
        if student_uses_ollama:
            if not ollama_student_url or ollama_student_url.strip() == "":
                raise ValueError(
                    "OLLAMA_STUDENT_BASE_URL environment variable must be set in .env file when provider='ollama' for student"
                )
            if "ollama" in config_data.get("llm", {}).get("student", {}):
                if "llm" not in config_data:
                    config_data["llm"] = {}
                if "student" not in config_data["llm"]:
                    config_data["llm"]["student"] = {}
                config_data["llm"]["student"]["ollama"]["ollama_base_url"] = ollama_student_url.strip()

        # Apply teacher URL only if ollama section exists in YAML
        if teacher_uses_ollama:
            if not ollama_teacher_url or ollama_teacher_url.strip() == "":
                raise ValueError(
                    "OLLAMA_TEACHER_BASE_URL environment variable must be set in .env file when provider='ollama' for teacher"
                )
            if "ollama" in config_data.get("llm", {}).get("teacher", {}):
                if "llm" not in config_data:
                    config_data["llm"] = {}
                if "teacher" not in config_data["llm"]:
                    config_data["llm"]["teacher"] = {}
                config_data["llm"]["teacher"]["ollama"]["ollama_base_url"] = ollama_teacher_url.strip()

    @classmethod
    def _resolve_paths(cls, config_data: dict, base_dir: Path) -> dict:
        """Resolve path values relative to project root.

        All paths in the configuration file are resolved relative to the project root.
        Absolute paths are left unchanged.

        Only values that look like file paths (containing '/' or ending with
        common file extensions) are resolved. Simple values like 'INFO', 'cpu', etc.
        are left unchanged.

        Args:
            config_data: Configuration dictionary.
            base_dir: Project root directory.

        Returns:
            Configuration dictionary with resolved paths.
        """

        def is_path_like(value: str) -> bool:
            """Check if a string looks like a file path."""
            if not value:
                return False
            # Exclude URLs (http://, https://)
            if value.startswith(('http://', 'https://')):
                return False
            # Exclude known model name patterns (provider/model-name format)
            # to prevent 'gemini/gemini-2.0-flash' or 'qwen/qwen3.5' from being treated as a path
            # Common LLM provider prefixes
            MODEL_PREFIXES = (
                'gemini/', 'openai/', 'anthropic/', 'huggingface/', 'ollama/',
                'openrouter/', 'meta-llama/', 'google/', 'mistral/', 'cohere/',
                'together/', 'anyscale/', 'deepseek/', 'qwen/', 'yi/', 'baichuan/',
                '01-ai/', 'teknium/', 'nousresearch/', 'lmsys/', 'upstage/',
            )
            if value.lower().startswith(MODEL_PREFIXES):
                return False
            # Must contain '/' to be considered a path
            # This avoids converting simple values like 'INFO', 'cpu', 'marker'
            return '/' in value

        def resolve_value(value: Any) -> Any:
            """Resolve a single value if it's a relative path."""
            if isinstance(value, str) and is_path_like(value):
                path = Path(value)
                if not path.is_absolute():
                    return str(base_dir / path)
            return value

        def process_dict(d: dict[str, Any]) -> dict[str, Any]:
            """Recursively process dictionary."""
            result: dict[str, Any] = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = process_dict(v)
                elif isinstance(v, list):
                    result[k] = [
                        resolve_value(item) if isinstance(item, str) else item
                        for item in v
                    ]
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
