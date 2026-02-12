# src/aee/tasks/__init__.py
"""Task registry for AutoEvoExtractor."""

from aee.models.nanozymes import task_config as nanozymes_config

TASK_REGISTRY = {
    "nanozymes": nanozymes_config,
}

__all__ = ["TASK_REGISTRY"]