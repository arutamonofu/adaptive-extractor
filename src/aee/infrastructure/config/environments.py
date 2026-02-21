"""Environment-specific configuration support.

This module provides utilities for loading environment-specific
configurations (dev, test, prod).

Configuration priority (highest to lowest):
    1. CLI argument --config (explicit file path)
    2. AEE_ENV environment variable (dev, test, prod)

Note: Configuration file is required. There is no fallback to default.yaml.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from .settings import Settings


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "dev"
    TESTING = "test"
    PRODUCTION = "prod"


def get_environment() -> Environment:
    """Get current environment from AEE_ENV variable.

    Returns:
        Current environment (defaults to DEVELOPMENT).
    """
    env_str = os.getenv("AEE_ENV", "dev").lower()

    try:
        return Environment(env_str)
    except ValueError:
        return Environment.DEVELOPMENT


def load_settings_for_environment(
    env: Optional[Environment] = None,
    custom_config: Optional[Path] = None,
) -> Settings:
    """Load settings for a specific environment.

    Configuration priority (highest to lowest):
        1. custom_config (explicit file path from CLI)
        2. env (AEE_ENV environment variable -> config/{env}.yaml)

    Args:
        env: Environment to load settings for (auto-detected if None).
        custom_config: Optional custom configuration file path.

    Returns:
        Loaded settings instance.

    Raises:
        ValueError: If no configuration source is specified.
        FileNotFoundError: If configuration file does not exist.
    """
    # CLI argument has highest priority
    if custom_config:
        return Settings.load(config_path=custom_config)

    # Use AEE_ENV or default to development
    if env is None:
        env = get_environment()

    # Load environment-specific config (no fallback)
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    env_config_path = base_dir / "config" / f"{env.value}.yaml"

    return Settings.load(config_path=env_config_path)


# Convenience functions
def load_dev_settings() -> Settings:
    """Load development settings."""
    return load_settings_for_environment(Environment.DEVELOPMENT)


def load_test_settings() -> Settings:
    """Load test settings."""
    return load_settings_for_environment(Environment.TESTING)


def load_prod_settings() -> Settings:
    """Load production settings."""
    return load_settings_for_environment(Environment.PRODUCTION)
