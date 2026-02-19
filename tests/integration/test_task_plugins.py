"""Integration tests for task plugin system.

Tests cover:
- All tasks registered correctly
- TaskConfig validation
- compare_fields validation
"""

import pytest

from aee.domain.tasks import TaskRegistry, get_global_registry
from aee.domain.tasks.nanozymes import NanozymeTask


class TestTaskPlugins:
    """Tests for task plugin system."""

    def test_nanozyme_task_full_workflow(self):
        """Test complete nanozyme task workflow."""
        # Create task
        instruction = """Extract nanozyme experiments from scientific articles.
        
For each experiment, identify:
- Chemical formula
- Catalytic activity type
- Kinetic parameters (Km, Vmax)
- Reaction conditions (pH, temperature)
- Material properties (size, surface, crystal structure)
"""
        task = NanozymeTask(initial_instruction=instruction)
        
        # Validate
        task.validate()
        
        # Verify models
        assert task.output_model is not None
        assert task.experiment_model is not None
        
        # Verify signature
        assert task.signature is not None
        
        # Verify converters
        assert task.row_converter is not None
        
        # Verify compare fields
        assert len(task.compare_fields) > 0
        assert isinstance(task.compare_fields, list)

    def test_task_registry_lifecycle(self):
        """Test task registry lifecycle."""
        registry = TaskRegistry()
        
        # Initial state
        assert registry.count() == 0
        
        # Register task
        instruction = "Extract nanozyme experiments."
        task = NanozymeTask(initial_instruction=instruction)
        registry.register(task)
        
        # Verify registration
        assert registry.count() == 1
        assert "nanozymes" in registry.list_task_names()
        
        # Get task
        retrieved = registry.get("nanozymes")
        assert retrieved is task
        
        # Check containment
        assert "nanozymes" in registry
        
        # Unregister
        registry.unregister("nanozymes")
        assert registry.count() == 0
        assert "nanozymes" not in registry

    def test_task_duplicate_registration_raises(self):
        """Test that duplicate registration raises error."""
        registry = TaskRegistry()
        
        instruction = "Extract nanozyme experiments."
        task = NanozymeTask(initial_instruction=instruction)
        
        # First registration - should succeed
        registry.register(task)
        
        # Second registration - should fail
        with pytest.raises(ValueError, match="already registered"):
            registry.register(task)

    def test_task_not_found_raises(self):
        """Test that getting non-existent task raises error."""
        from aee.shared.exceptions import TaskNotFoundError
        
        registry = TaskRegistry()
        
        with pytest.raises(TaskNotFoundError):
            registry.get("nonexistent_task")

    def test_task_to_dict(self):
        """Test task serialization to dictionary."""
        instruction = "Extract nanozyme experiments."
        task = NanozymeTask(initial_instruction=instruction)
        
        task_dict = task.to_dict()
        
        assert task_dict["name"] == "nanozymes"
        assert "signature" in task_dict
        assert "output_model" in task_dict
        assert "experiment_model" in task_dict
        assert "compare_fields" in task_dict
        assert "float_tolerance" in task_dict
        assert isinstance(task_dict["compare_fields"], list)
        assert len(task_dict["compare_fields"]) > 0

    def test_global_registry_singleton(self):
        """Test global registry is a singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
        
        # Register in one, check in other
        instruction = "Extract nanozyme experiments."
        task = NanozymeTask(initial_instruction=instruction)
        registry1.register(task)
        
        assert registry2.has("nanozymes")
        
        # Cleanup
        registry1.clear()
