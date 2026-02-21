"""Integration tests for task plugin system.

Tests cover:
- TaskConfig loading from YAML
- TaskRegistry with YAML-based tasks
- Dynamic model and signature generation
"""

import pytest

from aee.domain.tasks import TaskRegistry, get_global_registry, get_task, load_task_from_yaml, register_config


@pytest.fixture(autouse=True)
def setup_nanozyme_task():
    """Automatically register nanozyme task before each test."""
    registry = get_global_registry()
    if not registry.has("nanozymes"):
        yaml_path = "src/aee/domain/tasks/nanozymes/task.yaml"
        config = load_task_from_yaml(yaml_path)
        config.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
        register_config(config)
    yield
    # Cleanup after test
    if registry.has("nanozymes"):
        registry.unregister("nanozymes")


class TestTaskPlugins:
    """Tests for task plugin system."""

    def test_nanozyme_task_full_workflow(self):
        """Test complete nanozyme task workflow using YAML config."""
        # Get task components
        task = get_task("nanozymes")

        # Validate config
        task["config"].validate()

        # Verify models
        assert task["output_model"] is not None
        assert task["experiment_model"] is not None

        # Verify signature
        assert task["signature"] is not None

        # Verify converters
        assert task["row_converter"] is not None

        # Verify compare fields
        assert len(task["config"].compare_fields) > 0
        assert isinstance(task["config"].compare_fields, list)

    def test_task_registry_lifecycle(self):
        """Test task registry lifecycle with YAML-based task."""
        registry = TaskRegistry()

        # Initial state
        assert registry.count() == 0

        # Load and register task from YAML
        yaml_path = "src/aee/domain/tasks/nanozymes/task.yaml"
        config = load_task_from_yaml(yaml_path)
        config.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
        registry.register_config(config)

        # Verify registration
        assert registry.count() == 1
        assert "nanozymes" in registry.list_task_names()

        # Get task
        retrieved = registry.get_task("nanozymes")
        assert retrieved is not None

        # Check containment
        assert "nanozymes" in registry

        # Unregister
        registry.unregister("nanozymes")
        assert registry.count() == 0
        assert "nanozymes" not in registry

    def test_task_duplicate_registration_raises(self):
        """Test that duplicate registration raises error."""
        registry = TaskRegistry()

        # Load and register task from YAML
        yaml_path = "src/aee/domain/tasks/nanozymes/task.yaml"
        config = load_task_from_yaml(yaml_path)
        config.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
        registry.register_config(config)

        # Second registration - should fail
        with pytest.raises(ValueError, match="already registered"):
            registry.register_config(config)

    def test_task_not_found_raises(self):
        """Test that getting non-existent task raises error."""
        from aee.shared.exceptions import TaskNotFoundError

        registry = TaskRegistry()

        with pytest.raises(TaskNotFoundError):
            registry.get_task("nonexistent_task")

    def test_task_to_dict(self):
        """Test task serialization to dictionary."""
        task = get_task("nanozymes")

        config_dict = task["config"].to_dict()

        assert config_dict["name"] == "nanozymes"
        assert "experiment_fields" in config_dict
        assert "compare_fields" in config_dict
        assert "float_tolerance" in config_dict
        assert isinstance(config_dict["compare_fields"], list)
        assert len(config_dict["compare_fields"]) > 0

    def test_global_registry_singleton(self):
        """Test global registry is a singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()

        assert registry1 is registry2

        # Task should be registered from autouse fixture
        assert registry1.has("nanozymes")
        assert registry2.has("nanozymes")

        # Cleanup
        registry1.clear()
