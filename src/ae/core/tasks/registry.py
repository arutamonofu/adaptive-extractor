"""Task registry for managing task configurations.

The registry maintains a collection of registered task configs and provides
type-safe access to task definitions with validation.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ae.core.exceptions import TaskNotFoundError
from ae.core.tasks.config import TaskConfig
from ae.core.tasks.dynamic_models import create_all_models, create_row_converter
from ae.core.tasks.loader import load_task_from_yaml
from ae.core.tasks.signature import create_signature

if TYPE_CHECKING:
    from ae.core.config.settings import Settings

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Central registry for task configurations.

    The registry provides a type-safe way to register and retrieve task
    configurations. All tasks are validated upon registration.

    Example:
        ```python
        registry = TaskRegistry()
        registry.register_from_yaml("tasks/nanozymes/task.yaml")

        # Get a task config
        config = registry.get_config("nanozymes")

        # List all tasks
        tasks = registry.list_task_names()
        ```
    """

    def __init__(self) -> None:
        """Initialize empty task registry."""
        self._configs: Dict[str, TaskConfig] = {}
        self._cache: Dict[str, dict] = {}  # Cache for generated components
        logger.debug("Task registry initialized")

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
        if config.name in self._configs:
            raise ValueError(
                f"Task '{config.name}' is already registered. "
                f"Cannot register duplicate tasks."
            )

        # Register config
        self._configs[config.name] = config
        logger.info(f"Registered task config: '{config.name}'")

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
        yaml_path = Path(yaml_path)
        config = load_task_from_yaml(yaml_path)
        self.register_config(config, validate=validate)

        return config

    def unregister(self, task_name: str) -> None:
        """Unregister a task config.

        Args:
            task_name: Name of task to unregister.

        Raises:
            TaskNotFoundError: If task not found.
        """
        if task_name in self._configs:
            del self._configs[task_name]
            if task_name in self._cache:
                del self._cache[task_name]
            logger.info(f"Unregistered task config: '{task_name}'")
        else:
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

    def get_task(self, task_name: str) -> dict:
        """Get complete task components (config, models, signature, converter).

        Args:
            task_name: Name of the task.

        Returns:
            Dictionary with keys: config, experiment_model, output_model, signature, row_converter

        Raises:
            TaskNotFoundError: If task not found.
        """
        if task_name not in self._configs:
            raise TaskNotFoundError(task_name)

        # Return from cache if available
        if task_name in self._cache:
            return self._cache[task_name]

        # Generate components
        config = self._configs[task_name]
        experiment_model, output_model = create_all_models(config)
        signature = create_signature(config, experiment_model, output_model)
        row_converter = create_row_converter(config, experiment_model)

        # Cache components
        self._cache[task_name] = {
            "config": config,
            "experiment_model": experiment_model,
            "output_model": output_model,
            "signature": signature,
            "row_converter": row_converter,
        }

        return self._cache[task_name]

    def has(self, task_name: str) -> bool:
        """Check if a task is registered.

        Args:
            task_name: Name of the task to check.

        Returns:
            True if task is registered, False otherwise.
        """
        return task_name in self._configs

    def list_task_names(self) -> List[str]:
        """List all registered task names.

        Returns:
            List of task names in registration order.
        """
        return list(self._configs.keys())

    def count(self) -> int:
        """Count registered tasks.

        Returns:
            Number of registered tasks.
        """
        return len(self._configs)

    def clear(self) -> None:
        """Clear all registered tasks and cache.

        Warning:
            This removes all tasks from the registry. Use with caution.
        """
        config_count = len(self._configs)
        self._configs.clear()
        self._cache.clear()
        logger.warning(f"Cleared task registry ({config_count} configs removed)")

    def validate_all(self) -> Dict[str, Optional[ValueError]]:
        """Validate all registered tasks.

        Returns:
            Dictionary mapping task names to validation errors (None if valid).
        """
        results: Dict[str, Optional[ValueError]] = {}

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

    def __contains__(self, task_name: str) -> bool:
        """Support 'in' operator for checking task registration."""
        return task_name in self._configs

    def __len__(self) -> int:
        """Support len() for counting tasks."""
        return len(self._configs)

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


def register_config(
    config: TaskConfig,
    validate: bool = True,
) -> None:
    """Register a TaskConfig in the global registry.

    Args:
        config: Task configuration to register.
        validate: Whether to validate before registration.
    """
    registry = get_global_registry()
    registry.register_config(config, validate=validate)


def get_config(task_name: str) -> TaskConfig:
    """Get a TaskConfig from the global registry.

    Args:
        task_name: Name of the task.

    Returns:
        TaskConfig instance.
    """
    registry = get_global_registry()
    return registry.get_config(task_name)


def get_task(task_name: str) -> dict:
    """Get complete task components from the global registry.

    Args:
        task_name: Name of the task.

    Returns:
        Dictionary with config, experiment_model, output_model, signature, row_converter.
    """
    registry = get_global_registry()
    return registry.get_task(task_name)


def load_and_register_task(yaml_path: str | Path) -> TaskConfig:
    """Load a task from YAML and register it.

    Args:
        yaml_path: Path to YAML manifest file.

    Returns:
        Loaded TaskConfig instance.
    """
    registry = get_global_registry()
    return registry.register_from_yaml(yaml_path)


def load_task_with_instruction(
    task_name: str,
    settings: "Settings",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load task with initial instruction from task directory.

    Loads task configuration from tasks/{task_name}/initial_schema.yaml,
    applies initial_instruction.txt from the same directory, registers the task,
    and returns all task components along with instruction metadata.

    Args:
        task_name: Name of the task to load (used to find task folder).
        settings: Settings object containing task configuration.

    Returns:
        Tuple of (task_dict, instruction_metadata) where:
        - task_dict: Dictionary with keys config, experiment_model,
          output_model, signature, row_converter
        - instruction_metadata: Dictionary with keys instruction,
          instruction_length, instruction_hash
    """
    # Find project root and build YAML path
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    yaml_path = project_root / "config" / "tasks" / task_name / "initial_schema.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Task schema YAML config not found: {yaml_path}")

    # Load task from YAML
    task_config = load_task_from_yaml(yaml_path)

    # Set initial_instruction_file resolved path for internal DSPy compatibility
    task_config.initial_instruction_file = str(
        project_root / "config" / "tasks" / task_name / "initial_instruction.txt"
    )

    # Register task in global registry
    register_config(task_config)

    # Get complete task components from registry
    task = get_task(task_name)

    # Get instruction metadata
    instruction = task_config.get_instruction()
    instruction_hash = task_config.get_instruction_hash()

    instruction_metadata = {
        "instruction": instruction,
        "instruction_length": len(instruction),
        "instruction_hash": instruction_hash,
    }

    logger.info(
        f"Loaded task '{task_name}' from {yaml_path} "
        f"(instruction: {len(instruction)} chars, hash: {instruction_hash})"
    )

    return task, instruction_metadata
