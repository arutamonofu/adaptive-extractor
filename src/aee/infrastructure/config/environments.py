"""Environment-specific configuration support.

This module provides utilities for loading environment-specific
configurations (dev, test, prod).
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
    """Get current environment from environment variable.

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

    Args:
        env: Environment to load settings for (auto-detected if None).
        custom_config: Optional custom configuration file.

    Returns:
        Loaded settings instance.
    """
    if env is None:
        env = get_environment()

    # Try to load environment-specific config
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    env_config_path = base_dir / "config" / f"{env.value}.yaml"

    if env_config_path.exists():
        return Settings.load(env_config_path)
    elif custom_config:
        return Settings.load(custom_config)
    else:
        return Settings.load()


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
