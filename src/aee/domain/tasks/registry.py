"""Task registry for managing task plugins.

The registry maintains a collection of registered tasks and provides
type-safe access to task definitions with validation.

Supports both classic TaskDefinition approach and new TaskConfig approach.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aee.domain.tasks.base import TaskDefinition
from aee.domain.tasks.config import TaskConfig
from aee.shared.exceptions import TaskNotFoundError, TaskValidationError

logger = logging.getLogger(__name__)

# Type alias for task registration
TaskType = Union[TaskDefinition, TaskConfig]


class TaskRegistry:
    """Central registry for task definitions.

    The registry provides a type-safe way to register and retrieve task
    definitions. All tasks are validated upon registration.

    Supports two approaches:
    1. Classic: Register TaskDefinition instances (e.g., NanozymeTask)
    2. Modern: Register TaskConfig from YAML or programmatically

    Example:
        ```python
        # Classic approach
        registry = TaskRegistry()
        registry.register(NanozymeTask(initial_instruction="..."))

        # Modern approach
        registry.register_from_yaml("tasks/nanozymes/task.yaml")

        # Get a task
        task = registry.get("nanozymes")

        # List all tasks
        tasks = registry.list_tasks()
        ```
    """

    def __init__(self) -> None:
        """Initialize empty task registry."""
        self._tasks: Dict[str, TaskDefinition] = {}
        self._configs: Dict[str, TaskConfig] = {}
        logger.debug("Task registry initialized")

    def register(
        self,
        task: TaskDefinition,
        validate: bool = True,
    ) -> None:
        """Register a TaskDefinition instance.

        Args:
            task: Task definition to register.
            validate: Whether to validate the task before registration (default True).

        Raises:
            TaskValidationError: If validation fails.
            ValueError: If task with same name already registered.
        """
        # Validate task if requested
        if validate:
            try:
                task.validate()
            except TaskValidationError as e:
                logger.error(f"Task validation failed for '{task.name}': {e}")
                raise

        # Check for duplicate task names
        if task.name in self._tasks or task.name in self._configs:
            raise ValueError(
                f"Task '{task.name}' is already registered. "
                f"Cannot register duplicate tasks."
            )

        # Register task
        self._tasks[task.name] = task
        logger.info(f"Registered task: '{task.name}' - {task.description}")

    def register_config(
        self,
        config: TaskConfig,
        validate: bool = True,
    ) -> None:
        """Register a TaskConfig.

        Args:
            config: Task configuration to register.
            validate: Whether to validate the config before registration (default True).

        Raises:
            ValueError: If validation fails or task with same name already registered.
        """
        # Validate config if requested
        if validate:
            errors = config.validate()
            if errors:
                error_msg = "\n".join(errors)
                logger.error(f"TaskConfig validation failed for '{config.name}':\n{error_msg}")
                raise ValueError(f"TaskConfig validation failed: {error_msg}")

        # Check for duplicate task names
        if config.name in self._tasks or config.name in self._configs:
            raise ValueError(
                f"Task '{config.name}' is already registered. "
                f"Cannot register duplicate tasks."
            )

        # Register config
        self._configs[config.name] = config
        logger.info(f"Registered task config: '{config.name}' - {config.description}")

    def register_from_yaml(
        self,
        yaml_path: str | Path,
        validate: bool = True,
    ) -> TaskConfig:
        """Load and register a task from YAML file.

        Args:
            yaml_path: Path to YAML manifest file.
            validate: Whether to validate the config before registration.

        Returns:
            Loaded TaskConfig instance.

        Raises:
            FileNotFoundError: If YAML file not found.
            ValueError: If validation fails or registration fails.
        """
        from .loader import load_task_from_yaml

        yaml_path = Path(yaml_path)
        config = load_task_from_yaml(yaml_path)
        self.register_config(config, validate=validate)

        return config

    def unregister(self, task_name: str) -> None:
        """Unregister a task definition.

        Args:
            task_name: Name of task to unregister.

        Raises:
            TaskNotFoundError: If task not found.
        """
        if task_name in self._tasks:
            del self._tasks[task_name]
            logger.info(f"Unregistered task: '{task_name}'")
        elif task_name in self._configs:
            del self._configs[task_name]
            logger.info(f"Unregistered task config: '{task_name}'")
        else:
            raise TaskNotFoundError(task_name)

    def get(self, task_name: str) -> TaskDefinition:
        """Get a registered task definition.

        For TaskConfig registrations, this will dynamically create
        and return a TaskDefinition wrapper.

        Args:
            task_name: Name of the task to retrieve.

        Returns:
            Task definition.

        Raises:
            TaskNotFoundError: If task not found.
        """
        # Check classic tasks first
        if task_name in self._tasks:
            return self._tasks[task_name]

        # Check configs and create wrapper
        if task_name in self._configs:
            return self._create_task_from_config(self._configs[task_name])

        # Task not found
        available = ", ".join(self.list_task_names())
        raise TaskNotFoundError(task_name)

    def get_config(self, task_name: str) -> TaskConfig:
        """Get a registered TaskConfig.

        Args:
            task_name: Name of the task config to retrieve.

        Returns:
            TaskConfig instance.

        Raises:
            TaskNotFoundError: If task config not found.
        """
        if task_name not in self._configs:
            raise TaskNotFoundError(task_name)

        return self._configs[task_name]

    def has(self, task_name: str) -> bool:
        """Check if a task is registered.

        Args:
            task_name: Name of the task to check.

        Returns:
            True if task is registered, False otherwise.
        """
        return task_name in self._tasks or task_name in self._configs

    def list_tasks(self) -> List[TaskDefinition]:
        """List all registered tasks as TaskDefinition instances.

        Returns:
            List of task definitions in registration order.
        """
        # Return classic tasks
        tasks = list(self._tasks.values())

        # Add configs as TaskDefinition wrappers
        for config in self._configs.values():
            tasks.append(self._create_task_from_config(config))

        return tasks

    def list_configs(self) -> List[TaskConfig]:
        """List all registered TaskConfigs.

        Returns:
            List of TaskConfig instances.
        """
        return list(self._configs.values())

    def list_task_names(self) -> List[str]:
        """List all registered task names.

        Returns:
            List of task names in registration order.
        """
        return list(self._tasks.keys()) + list(self._configs.keys())

    def count(self) -> int:
        """Count registered tasks.

        Returns:
            Number of registered tasks.
        """
        return len(self._tasks) + len(self._configs)

    def clear(self) -> None:
        """Clear all registered tasks and configs.

        Warning:
            This removes all tasks from the registry. Use with caution.
        """
        task_count = len(self._tasks)
        config_count = len(self._configs)
        self._tasks.clear()
        self._configs.clear()
        logger.warning(
            f"Cleared task registry ({task_count} tasks, {config_count} configs removed)"
        )

    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about a registered task.

        Args:
            task_name: Name of the task.

        Returns:
            Dictionary with task information.

        Raises:
            TaskNotFoundError: If task not found.
        """
        if task_name in self._configs:
            return self._configs[task_name].to_dict()

        task = self.get(task_name)
        return task.to_dict()

    def validate_all(self) -> Dict[str, Optional[Union[TaskValidationError, ValueError]]]:
        """Validate all registered tasks.

        Returns:
            Dictionary mapping task names to validation errors (None if valid).
        """
        results: Dict[str, Optional[Union[TaskValidationError, ValueError]]] = {}

        # Validate classic tasks
        for task_name, task in self._tasks.items():
            try:
                task.validate()
                results[task_name] = None
            except TaskValidationError as e:
                results[task_name] = e
                logger.error(f"Validation failed for task '{task_name}': {e}")

        # Validate configs
        for config_name, config in self._configs.items():
            errors = config.validate()
            if errors:
                error_msg = "\n".join(errors)
                results[config_name] = ValueError(
                    f"TaskConfig validation failed: {error_msg}"
                )
                logger.error(f"Validation failed for config '{config_name}': {error_msg}")
            else:
                results[config_name] = None

        return results

    def _create_task_from_config(self, config: TaskConfig) -> TaskDefinition:
        """Create a TaskDefinition wrapper from TaskConfig.

        This allows TaskConfig to be used interchangeably with TaskDefinition.

        Args:
            config: TaskConfig to wrap.

        Returns:
            TaskDefinition instance.
        """
        from .dynamic_wrapper import ConfigBackedTask

        return ConfigBackedTask(config)

    def __contains__(self, task_name: str) -> bool:
        """Support 'in' operator for checking task registration.

        Args:
            task_name: Name of the task to check.

        Returns:
            True if task is registered, False otherwise.
        """
        return task_name in self._tasks or task_name in self._configs

    def __len__(self) -> int:
        """Support len() for counting tasks.

        Returns:
            Number of registered tasks.
        """
        return len(self._tasks) + len(self._configs)

    def __repr__(self) -> str:
        """String representation of registry."""
        count = self.count()
        tasks = ", ".join(self.list_task_names())
        return f"<TaskRegistry: {count} tasks ({tasks})>"


# Global singleton registry instance
_global_registry: Optional[TaskRegistry] = None


def get_global_registry() -> TaskRegistry:
    """Get the global task registry singleton.

    Returns:
        Global task registry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TaskRegistry()
        logger.debug("Created global task registry")
    return _global_registry


def register_task(
    task: TaskDefinition | TaskConfig,
    validate: bool = True,
) -> None:
    """Register a task in the global registry.

    Convenience function for registering tasks in the global registry.
    Supports both TaskDefinition and TaskConfig.

    Args:
        task: Task definition or config to register.
        validate: Whether to validate the task before registration.
    """
    registry = get_global_registry()

    if isinstance(task, TaskConfig):
        registry.register_config(task, validate=validate)
    else:
        registry.register(task, validate=validate)


def get_task(task_name: str) -> TaskDefinition:
    """Get a task from the global registry.

    Convenience function for retrieving tasks from the global registry.

    Args:
        task_name: Name of the task to retrieve.

    Returns:
        Task definition.
    """
    registry = get_global_registry()
    return registry.get(task_name)


def load_and_register_task(yaml_path: str | Path) -> TaskConfig:
    """Load a task from YAML and register it.

    Convenience function for loading and registering YAML-based tasks.

    Args:
        yaml_path: Path to YAML manifest file.

    Returns:
        Loaded TaskConfig instance.
    """
    registry = get_global_registry()
    return registry.register_from_yaml(yaml_path)
