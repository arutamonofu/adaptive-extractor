# src/aee/tasks/__init__.py

from aee.tasks.nanozymes import task_config as nanozymes_config

# Central registry of available extraction tasks
TASK_REGISTRY = {
    "nanozymes": nanozymes_config,
    # "catalysis": catalysis_config,
}