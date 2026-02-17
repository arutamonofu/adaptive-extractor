"""Task plugin system for AutoEvoExtractor.

This module provides the task plugin infrastructure including base classes,
registry, and utilities for managing extraction tasks.
"""

from .base import TaskDefinition
from .registry import (
    TaskRegistry,
    get_global_registry,
    get_task,
    register_task,
)

__all__ = [
    "TaskDefinition",
    "TaskRegistry",
    "get_global_registry",
    "get_task",
    "register_task",
]
