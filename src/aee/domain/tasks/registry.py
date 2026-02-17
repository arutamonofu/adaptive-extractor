"""Task registry for managing task plugins.

The registry maintains a collection of registered tasks and provides
type-safe access to task definitions with validation.
"""

import logging
from typing import Any, Dict, List, Optional

from aee.domain.tasks.base import TaskDefinition
from aee.shared.exceptions import TaskNotFoundError, TaskValidationError

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Central registry for task definitions.

    The registry provides a type-safe way to register and retrieve task
    definitions. All tasks are validated upon registration.

    Example:
        ```python
        registry = TaskRegistry()

        # Register a task
        registry.register(NanozymeTask())

        # Get a task
        task = registry.get("nanozymes")

        # List all tasks
        tasks = registry.list_tasks()
        ```
    """

    def __init__(self) -> None:
        """Initialize empty task registry."""
        self._tasks: Dict[str, TaskDefinition] = {}
        logger.debug("Task registry initialized")

    def register(self, task: TaskDefinition, validate: bool = True) -> None:
        """Register a task definition.

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
        if task.name in self._tasks:
            raise ValueError(
                f"Task '{task.name}' is already registered. "
                f"Cannot register duplicate tasks."
            )

        # Register task
        self._tasks[task.name] = task
        logger.info(f"Registered task: '{task.name}' - {task.description}")

    def unregister(self, task_name: str) -> None:
        """Unregister a task definition.

        Args:
            task_name: Name of task to unregister.

        Raises:
            TaskNotFoundError: If task not found.
        """
        if task_name not in self._tasks:
            raise TaskNotFoundError(task_name)

        del self._tasks[task_name]
        logger.info(f"Unregistered task: '{task_name}'")

    def get(self, task_name: str) -> TaskDefinition:
        """Get a registered task definition.

        Args:
            task_name: Name of the task to retrieve.

        Returns:
            Task definition.

        Raises:
            TaskNotFoundError: If task not found.
        """
        if task_name not in self._tasks:
            available = ", ".join(self.list_task_names())
            raise TaskNotFoundError(task_name)

        return self._tasks[task_name]

    def has(self, task_name: str) -> bool:
        """Check if a task is registered.

        Args:
            task_name: Name of the task to check.

        Returns:
            True if task is registered, False otherwise.
        """
        return task_name in self._tasks

    def list_tasks(self) -> List[TaskDefinition]:
        """List all registered tasks.

        Returns:
            List of task definitions in registration order.
        """
        return list(self._tasks.values())

    def list_task_names(self) -> List[str]:
        """List all registered task names.

        Returns:
            List of task names in registration order.
        """
        return list(self._tasks.keys())

    def count(self) -> int:
        """Count registered tasks.

        Returns:
            Number of registered tasks.
        """
        return len(self._tasks)

    def clear(self) -> None:
        """Clear all registered tasks.

        Warning:
            This removes all tasks from the registry. Use with caution.
        """
        count = len(self._tasks)
        self._tasks.clear()
        logger.warning(f"Cleared task registry ({count} tasks removed)")

    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about a registered task.

        Args:
            task_name: Name of the task.

        Returns:
            Dictionary with task information.

        Raises:
            TaskNotFoundError: If task not found.
        """
        task = self.get(task_name)
        return task.to_dict()

    def validate_all(self) -> Dict[str, Optional[TaskValidationError]]:
        """Validate all registered tasks.

        Returns:
            Dictionary mapping task names to validation errors (None if valid).
        """
        results: Dict[str, Optional[TaskValidationError]] = {}

        for task_name, task in self._tasks.items():
            try:
                task.validate()
                results[task_name] = None
            except TaskValidationError as e:
                results[task_name] = e
                logger.error(f"Validation failed for task '{task_name}': {e}")

        return results

    def __contains__(self, task_name: str) -> bool:
        """Support 'in' operator for checking task registration.

        Args:
            task_name: Name of the task to check.

        Returns:
            True if task is registered, False otherwise.
        """
        return task_name in self._tasks

    def __len__(self) -> int:
        """Support len() for counting tasks.

        Returns:
            Number of registered tasks.
        """
        return len(self._tasks)

    def __repr__(self) -> str:
        """String representation of registry."""
        count = len(self._tasks)
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


def register_task(task: TaskDefinition, validate: bool = True) -> None:
    """Register a task in the global registry.

    Convenience function for registering tasks in the global registry.

    Args:
        task: Task definition to register.
        validate: Whether to validate the task before registration.
    """
    registry = get_global_registry()
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
