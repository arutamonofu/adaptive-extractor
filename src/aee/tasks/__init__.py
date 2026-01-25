# src/aee/tasks/__init__.py

from aee.tasks.nanozymes import task_config as nanozymes_config

TASK_REGISTRY = {
    "nanozymes": nanozymes_config,
}

__all__ = ["TASK_REGISTRY"]