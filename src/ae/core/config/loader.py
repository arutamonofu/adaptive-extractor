"""Loader utility for application settings."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

logger = logging.getLogger(__name__)


def _deep_update(base_dict: dict, update_with: dict) -> None:
    """Recursively update a dictionary with another dictionary.

    Args:
        base_dict: The dictionary to update.
        update_with: The dictionary to update with.
    """
    for k, v in update_with.items():
        if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
            _deep_update(base_dict[k], v)
        else:
            base_dict[k] = v


def _resolve_paths(config_data: dict, base_dir: Path) -> dict:
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
        MODEL_PREFIXES = (
            'gemini/', 'openai/', 'anthropic/', 'huggingface/', 'ollama/',
            'openrouter/', 'meta-llama/', 'google/', 'mistral/', 'cohere/',
            'together/', 'anyscale/', 'deepseek/', 'qwen/', 'yi/', 'baichuan/',
            '01-ai/', 'teknium/', 'nousresearch/', 'lmsys/', 'upstage/',
        )
        if value.lower().startswith(MODEL_PREFIXES):
            return False
        # Must contain '/' to be considered a path
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
            if k in ("model", "priority_order", "order"):
                result[k] = v
            elif isinstance(v, dict):
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


def _apply_env_overrides(config_data: dict) -> None:
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
    ingestor_uses_ollama = (
        config_data.get("llm", {})
        .get("ingestor", {})
        .get("provider") == "ollama"
    )

    ollama_student_url = os.getenv("OLLAMA_STUDENT_BASE_URL")
    ollama_teacher_url = os.getenv("OLLAMA_TEACHER_BASE_URL")
    ollama_ingestor_url = os.getenv("OLLAMA_INGESTOR_BASE_URL")

    if student_uses_ollama:
        if not ollama_student_url or ollama_student_url.strip() == "":
            raise ValueError(
                "OLLAMA_STUDENT_BASE_URL environment variable must be set in .env file when provider='ollama' for student"
            )
        if "llm" not in config_data:
            config_data["llm"] = {}
        if "student" not in config_data["llm"]:
            config_data["llm"]["student"] = {}
        if "ollama" not in config_data["llm"]["student"]:
            config_data["llm"]["student"]["ollama"] = {}
        config_data["llm"]["student"]["ollama"]["ollama_base_url"] = ollama_student_url.strip()

    if teacher_uses_ollama:
        if not ollama_teacher_url or ollama_teacher_url.strip() == "":
            raise ValueError(
                "OLLAMA_TEACHER_BASE_URL environment variable must be set in .env file when provider='ollama' for teacher"
            )
        if "llm" not in config_data:
            config_data["llm"] = {}
        if "teacher" not in config_data["llm"]:
            config_data["llm"]["teacher"] = {}
        if "ollama" not in config_data["llm"]["teacher"]:
            config_data["llm"]["teacher"]["ollama"] = {}
        config_data["llm"]["teacher"]["ollama"]["ollama_base_url"] = ollama_teacher_url.strip()

    if ingestor_uses_ollama:
        if not ollama_ingestor_url or ollama_ingestor_url.strip() == "":
            raise ValueError(
                "OLLAMA_INGESTOR_BASE_URL environment variable must be set in .env file when provider='ollama' for ingestor"
            )
        if "llm" not in config_data:
            config_data["llm"] = {}
        if "ingestor" not in config_data["llm"]:
            config_data["llm"]["ingestor"] = {}
        if "ollama" not in config_data["llm"]["ingestor"]:
            config_data["llm"]["ingestor"]["ollama"] = {}
        config_data["llm"]["ingestor"]["ollama"]["ollama_base_url"] = ollama_ingestor_url.strip()


def _apply_api_keys(config_data: dict) -> None:
    """Apply API keys from environment variables to api config.

    API keys are read from environment variables and injected into the
    llm.student.api and llm.teacher.api configurations.

    Env vars applied:
        - OPENAI_API_KEY: OpenAI API key
        - ANTHROPIC_API_KEY: Anthropic API key
        - GEMINI_API_KEY: Google Gemini API key
        - OPENROUTER_API_KEY: OpenRouter API key

    Args:
        config_data: Configuration dictionary to update.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    logger.debug(f"API keys from env - OpenAI: {'SET' if openai_key else 'NOT SET'}, "
                 f"Anthropic: {'SET' if anthropic_key else 'NOT SET'}, "
                 f"Gemini: {'SET' if gemini_key else 'NOT SET'}, "
                 f"OpenRouter: {'SET' if openrouter_key else 'NOT SET'}")

    BASE_URL_TO_KEY = {
        "openrouter.ai": openrouter_key,
        "api.openai.com": openai_key,
        "api.anthropic.com": anthropic_key,
        "generativelanguage.googleapis.com": gemini_key,
        "api.groq.com": openai_key,
    }

    def get_api_key_from_base_url(base_url: Optional[str]) -> Optional[str]:
        if not base_url:
            return None
        base_url_lower = base_url.lower()
        for pattern, api_key in BASE_URL_TO_KEY.items():
            if pattern in base_url_lower:
                return api_key
        return None

    def get_api_key_for_model(model_name: str) -> Optional[str]:
        if not model_name:
            return None
        model_lower = model_name.lower()
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
        return openai_key or anthropic_key or gemini_key or openrouter_key

    def apply_key_to_component(component_data: dict, component_name: str) -> None:
        if component_data.get("provider") != "api":
            return

        model_name = component_data.get("model", "")
        api_config = component_data.get("api", {})
        base_url = api_config.get("base_url")

        api_key = get_api_key_from_base_url(base_url)
        if api_key is None:
            api_key = get_api_key_for_model(model_name)

        if api_key:
            if "api" in component_data:
                component_data["api"]["api_key"] = api_key
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
                logger.debug(f"Using {key_source} API key for {component_name}: {model_name} ({source_info})")
            else:
                logger.warning(f"No api section in YAML for {component_name}: {model_name}, skipping API key injection")
        else:
            logger.warning(f"No API key found for {component_name}: {model_name}")

    if "llm" in config_data and "ingestor" in config_data["llm"]:
        apply_key_to_component(config_data["llm"]["ingestor"], "ingestor")
    if "llm" in config_data and "student" in config_data["llm"]:
        apply_key_to_component(config_data["llm"]["student"], "student")
    if "llm" in config_data and "teacher" in config_data["llm"]:
        apply_key_to_component(config_data["llm"]["teacher"], "teacher")


def load_settings(
    config_path: Optional[Union[str, Path]] = None,
    load_env_file: bool = True,
) -> dict[str, Any]:
    """Load settings data dictionary from modular YAML configuration directory.

    Args:
        config_path: Path to configuration directory. Defaults to root config/ directory.
        load_env_file: Whether to load .env file. Default is True.

    Returns:
        dict: Loaded and processed settings dict.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent

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

    if load_env_file:
        from dotenv import load_dotenv
        env_file = base_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")

    yaml_files = sorted(
        [p for p in config_path.glob("*.yaml")] + [p for p in config_path.glob("*.yml")]
    )
    if not yaml_files:
        raise FileNotFoundError(f"No YAML configuration files found in: {config_path}")

    config_data: dict[str, Any] = {}
    for file_path in yaml_files:
        if file_path.name in ("visual_pipeline.yaml", "schema.yaml"):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = yaml.safe_load(f) or {}
            _deep_update(config_data, file_data)
            logger.info(f"Loaded config chunk from {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {file_path}: {e}")

    config_data = _resolve_paths(config_data, base_dir)



    _apply_env_overrides(config_data)
    _apply_api_keys(config_data)

    # Configuration version check and migration
    version = config_data.get("version")
    if version is None:
        logger.warning("Configuration version is missing. Automatically upgrading to version 1 schema.")
        config_data["version"] = 1
    else:
        try:
            version = int(version)
        except ValueError:
            raise ValueError(f"Invalid configuration version: {version}. Version must be an integer.")

        if version < 1:
            logger.info(f"Upgrading configuration schema from version {version} to 1")
            config_data["version"] = 1
        elif version > 1:
            raise ValueError(
                f"Configuration version {version} is newer than supported version 1. "
                "Please downgrade configuration or upgrade the application."
            )

    return config_data
