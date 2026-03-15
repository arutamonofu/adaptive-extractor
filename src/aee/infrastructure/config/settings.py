# src/aee/infrastructure/config/settings.py
"""Configuration settings for AutoEvoExtractor.

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
from typing import Any, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
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


class NonOllamaConfig(BaseModel):
    """Non-Ollama LLM configuration.

    Environment variables:
        OPENAI_API_KEY: OpenAI API key (required when use_ollama=False)
        ANTHROPIC_API_KEY: Anthropic API key (required when use_ollama=False)
        GEMINI_API_KEY: Google Gemini API key (required when use_ollama=False)

    YAML configuration (config/default.yaml):
        max_tokens: Maximum tokens for non-Ollama providers

    Note: API key must be set via environment variable. Validation will fail
        if api_key is not provided.
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

    @field_validator("api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        """Validate that API key is set for non-Ollama providers.

        Note: This validator only checks if the key is provided. The actual
            requirement (use_ollama=False) is validated at the LLMInstanceConfig level.
        """
        # Validation is deferred to LLMInstanceConfig level where we know use_ollama
        return v


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

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)  # type: ignore[arg-type]
    non_ollama: NonOllamaConfig = Field(default_factory=NonOllamaConfig)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def validate_non_ollama_api_key(self) -> "LLMInstanceConfig":
        """Validate that API key is set when using non-Ollama providers."""
        if not self.use_ollama:
            if self.non_ollama.api_key is None:
                raise ValueError(
                    "API key must be set for non-Ollama providers. "
                    "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY in .env file."
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


class MarkerConfig(BaseModel):
    """Marker parser configuration.

    Note: Detailed Marker settings (~70 parameters) are now defined in code
    at src/aee/infrastructure/parsers/marker_config.py. This class is kept
    for backward compatibility and may be used for runtime overrides in the future.

    Currently, this class has no required fields. All settings are loaded from
    the marker_config module.
    """
    # Placeholder for potential future runtime overrides
    # All detailed settings are in marker_config.py
    pass


class GeminiParserConfig(BaseModel):
    """Gemini API parser configuration."""
    model_name: str = Field(
        default="gemini-3-flash-preview",
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


class IngestionConfig(BaseModel):
    """Document ingestion configuration."""
    parser: Literal["marker", "gemini"] = Field(
        ...,
        description="Document parser to use: 'marker' or 'gemini'"
    )
    overwrite: bool = Field(
        ...,
        description="Overwrite existing parsed files"
    )

    marker: Optional[MarkerConfig] = Field(
        default=None,
        description="Marker-specific configuration (required if parser is 'marker')"
    )
    gemini: Optional[GeminiParserConfig] = Field(
        default=None,
        description="Gemini-specific configuration (required if parser is 'gemini')"
    )

    @model_validator(mode="after")
    def validate_parser_config(self) -> "IngestionConfig":
        """Validate that the correct parser-specific config is provided."""
        if self.parser == "marker" and self.marker is None:
            # Marker settings are now in code (marker_config.py), so empty config is valid
            self.marker = MarkerConfig()
        if self.parser == "gemini" and self.gemini is None:
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
    max_errors: int = Field(
        default=5,
        description=(
            "Maximum number of errors allowed before stopping optimization "
            "(DSPy parallelizer setting). Increase from default (5) to allow "
            "more faults during trials."
        )
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
    enable_semantic_judge: bool = Field(
        default=True,
        description="Enable semantic judge for evaluation (default: True)"
    )


class TaskConfig(BaseModel):
    """Task-specific configuration."""
    name: str = Field(
        ...,
        description="Task name identifier"
    )
    initial_instruction_file: str = Field(
        ...,
        description="Path to initial instruction file for DSPy optimization"
    )
    evaluation: EvaluationConfig = Field(...)

    @field_validator("initial_instruction_file", mode="before")
    @classmethod
    def validate_initial_instruction_file(cls, v: str) -> str:
        """Validate that initial_instruction_file exists."""
        if not v or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("initial_instruction_file is required and cannot be empty")
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
        AEE_ENV: Environment selection (dev, test, prod)

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
        """Load settings from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file. Required.
            load_env_file: Whether to load .env file. Default is True.
                Set to False for testing.

        Returns:
            Settings: Loaded settings instance.

        Raises:
            ValueError: If config_path is not provided.
            FileNotFoundError: If config file does not exist.
        """
        if config_path is None:
            raise ValueError(
                "Configuration file path is required. "
                "Provide --config CLI argument or set AEE_ENV environment variable."
            )

        # Load .env file before applying env overrides
        if load_env_file:
            from dotenv import load_dotenv
            base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
            env_file = base_dir / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {env_file}")

        # Calculate base directory correctly (project root)
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent

        # Resolve config path
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = base_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load config file
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

        # Resolve all paths relative to project root
        config_data = cls._resolve_paths(config_data, base_dir)

        # Apply Ollama URL overrides from environment variables
        cls._apply_env_overrides(config_data)

        # Apply API keys from environment variables to non_ollama config
        cls._apply_api_keys(config_data)

        return cls(**config_data)

    @classmethod
    def _apply_api_keys(cls, config_data: dict) -> None:
        """Apply API keys from environment variables to non_ollama config.

        API keys are read from environment variables and injected into the
        llm.student.non_ollama and llm.teacher.non_ollama configurations.

        Env vars applied:
            - OPENAI_API_KEY: OpenAI API key
            - ANTHROPIC_API_KEY: Anthropic API key
            - GEMINI_API_KEY: Google Gemini API key

        Args:
            config_data: Configuration dictionary to update.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        # DEBUG: Log which API keys are found
        logger.debug(f"API keys from env - OpenAI: {'SET' if openai_key else 'NOT SET'}, "
                     f"Anthropic: {'SET' if anthropic_key else 'NOT SET'}, "
                     f"Gemini: {'SET' if gemini_key else 'NOT SET'}")

        # Use the first available API key (priority: OpenAI > Anthropic > Gemini)
        api_key = openai_key or anthropic_key or gemini_key

        # DEBUG: Log which API key will be used
        if api_key:
            key_source = "OpenAI" if api_key == openai_key else ("Anthropic" if api_key == anthropic_key else "Gemini")
            key_preview = f"{api_key[:8]}..." if api_key else "NONE"
            logger.info(f"Using {key_source} API key: {key_preview}")

        if api_key:
            # Apply to student if use_ollama is false
            if "llm" in config_data and "student" in config_data["llm"]:
                if not config_data["llm"]["student"].get("use_ollama", True):
                    if "non_ollama" not in config_data["llm"]["student"]:
                        config_data["llm"]["student"]["non_ollama"] = {}
                    # Store the key value - it will be wrapped in SecretStr by Pydantic
                    config_data["llm"]["student"]["non_ollama"]["api_key"] = api_key

            # Apply to teacher if use_ollama is false
            if "llm" in config_data and "teacher" in config_data["llm"]:
                if not config_data["llm"]["teacher"].get("use_ollama", True):
                    if "non_ollama" not in config_data["llm"]["teacher"]:
                        config_data["llm"]["teacher"]["non_ollama"] = {}
                    config_data["llm"]["teacher"]["non_ollama"]["api_key"] = api_key

    @classmethod
    def _apply_env_overrides(cls, config_data: dict) -> None:
        """Apply environment variable overrides to config data.

        Ollama URLs are read from environment variables only (NOT from YAML).
        This method modifies config_data in place.

        Note: Ollama URLs must be set in .env file. No fallback is provided.

        Env vars applied:
            - OLLAMA_STUDENT_BASE_URL: Student Ollama server URL (required when use_ollama=True for student)
            - OLLAMA_TEACHER_BASE_URL: Teacher Ollama server URL (required when use_ollama=True for teacher)

        Args:
            config_data: Configuration dictionary to update.

        Raises:
            ValueError: If required Ollama URL is not set in environment when use_ollama=True.
        """
        # Check if Ollama is used for student and teacher
        student_uses_ollama = (
            config_data.get("llm", {})
            .get("student", {})
            .get("use_ollama", False)
        )
        teacher_uses_ollama = (
            config_data.get("llm", {})
            .get("teacher", {})
            .get("use_ollama", False)
        )

        # Get Ollama URLs from environment
        ollama_student_url = os.getenv("OLLAMA_STUDENT_BASE_URL")
        ollama_teacher_url = os.getenv("OLLAMA_TEACHER_BASE_URL")

        # Validate and apply student URL only if use_ollama=True
        if student_uses_ollama:
            if not ollama_student_url or ollama_student_url.strip() == "":
                raise ValueError(
                    "OLLAMA_STUDENT_BASE_URL environment variable must be set in .env file when use_ollama=True for student"
                )
            if "llm" not in config_data:
                config_data["llm"] = {}
            if "student" not in config_data["llm"]:
                config_data["llm"]["student"] = {}
            if "ollama" not in config_data["llm"]["student"]:
                config_data["llm"]["student"]["ollama"] = {}
            config_data["llm"]["student"]["ollama"]["ollama_base_url"] = ollama_student_url.strip()

        # Validate and apply teacher URL only if use_ollama=True
        if teacher_uses_ollama:
            if not ollama_teacher_url or ollama_teacher_url.strip() == "":
                raise ValueError(
                    "OLLAMA_TEACHER_BASE_URL environment variable must be set in .env file when use_ollama=True for teacher"
                )
            if "llm" not in config_data:
                config_data["llm"] = {}
            if "teacher" not in config_data["llm"]:
                config_data["llm"]["teacher"] = {}
            if "ollama" not in config_data["llm"]["teacher"]:
                config_data["llm"]["teacher"]["ollama"] = {}
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
            # Exclude known model name patterns (provider/model-name format)
            # to prevent 'gemini/gemini-2.0-flash' from being treated as a path
            if value.startswith(('gemini/', 'openai/', 'anthropic/', 'huggingface/', 'ollama/')):
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
