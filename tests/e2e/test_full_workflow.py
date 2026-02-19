"""End-to-End tests for AutoEvoExtractor refactored architecture.

Tests cover complete workflows:
- YAML task configuration loading
- Dynamic model generation
- Agent save/load with functional API
- Full extraction pipeline

Note: These tests use mock data to avoid actual LLM calls.
Mark with @pytest.mark.slow for optional execution.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from aee.domain.tasks import (
    ConfigBackedTask,
    load_task_complete,
    load_task_from_yaml,
)
from aee.domain.tasks.config import TaskConfig
from aee.infrastructure.storage.agents_fn import (
    AgentMetadata,
    load_agent,
    save_agent,
)
from aee.infrastructure.storage.ground_truth_fn import load_ground_truth
from aee.infrastructure.storage.splits_fn import load_all_splits


@pytest.mark.slow
class TestYamlTaskWorkflow:
    """E2E tests for YAML-based task configuration."""

    def test_load_nanozymes_task_from_yaml(self):
        """Test loading nanozymes task from YAML manifest."""
        yaml_path = Path("src/aee/domain/tasks/nanozymes/task.yaml")

        config = load_task_from_yaml(yaml_path)

        assert config.name == "nanozymes"
        assert len(config.experiment_fields) > 0
        assert len(config.compare_fields) > 0
        assert config.get_instruction() is not None

    def test_load_nanozymes_task_complete(self):
        """Test loading complete nanozymes task with models."""
        yaml_path = Path("src/aee/domain/tasks/nanozymes/task.yaml")

        task = load_task_complete(yaml_path)

        assert "config" in task
        assert "experiment_model" in task
        assert "output_model" in task
        assert "signature" in task
        assert "row_converter" in task

        # Test creating experiment
        exp = task["experiment_model"](
            formula="Fe3O4",
            activity="peroxidase",
        )
        assert exp.formula == "Fe3O4"
        assert exp.activity == "peroxidase"

    def test_config_backed_task_workflow(self):
        """Test ConfigBackedTask wrapper workflow."""
        yaml_path = Path("src/aee/domain/tasks/nanozymes/task.yaml")
        config = load_task_from_yaml(yaml_path)

        task = ConfigBackedTask(config)

        # Test all TaskDefinition interface methods
        assert task.name == "nanozymes"
        assert task.description is not None
        assert task.signature is not None
        assert task.output_model is not None
        assert task.experiment_model is not None
        assert task.row_converter is not None
        assert len(task.compare_fields) > 0

        # Test validation
        task.validate()  # Should not raise

    def test_task_config_validation(self):
        """Test TaskConfig validation with various scenarios."""
        # Valid config
        from aee.domain.tasks.config import FieldSpec

        config = TaskConfig(
            name="test",
            description="Test task",
            experiment_fields={
                "field1": FieldSpec(type=str, description="Field 1"),
            },
            compare_fields=["field1"],
            initial_instruction="Test instruction",
        )

        errors = config.validate()
        assert errors == []

        # Invalid: missing instruction
        config_no_instruction = TaskConfig(
            name="test",
            description="Test",
            experiment_fields={
                "field1": FieldSpec(type=str, description="Field 1"),
            },
            compare_fields=["field1"],
        )

        errors = config_no_instruction.validate()
        assert len(errors) > 0
        assert any("instruction" in e.lower() for e in errors)


@pytest.mark.slow
class TestAgentFunctionalApi:
    """E2E tests for functional agent storage API."""

    def test_agent_save_load_workflow(self, tmp_path: Path):
        """Test complete agent save/load workflow."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create mock agent
        agent = {
            "lm": {"model": "test-model", "type": "mock"},
            "traces": [],
            "settings": {"num_trials": 5},
        }

        # Save agent
        agent_path = save_agent(
            agent=agent,
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics={"f1": 0.85, "precision": 0.82, "recall": 0.88},
            config_snapshot={"num_trials": 5},
            model_version="test-v1",
        )

        assert agent_path.exists()
        assert agent_path.with_suffix(".meta.json").exists()

        # Load agent
        loaded_agent, metadata = load_agent(agent_path)

        assert loaded_agent["lm"]["model"] == "test-model"
        assert metadata.task_name == "nanozymes"
        assert metadata.metrics["f1"] == 0.85

    def test_agent_metadata_with_instruction(self, tmp_path: Path):
        """Test agent metadata with instruction tracking."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent = {"lm": {"model": "test"}}

        agent_path = save_agent(
            agent=agent,
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics={"f1": 0.9},
            config_snapshot={},
            initial_instruction_file="config/initial_instructions/nanozymes_sota.txt",
            instruction_hash="abc123def456",
        )

        _, metadata = load_agent(agent_path)

        assert metadata.initial_instruction_file is not None
        assert metadata.instruction_hash == "abc123def456"


@pytest.mark.slow
class TestGroundTruthWorkflow:
    """E2E tests for ground truth loading workflow."""

    def test_ground_truth_loading_workflow(self, tmp_path: Path):
        """Test complete ground truth loading workflow."""
        # Create test CSV
        csv_path = tmp_path / "gt.csv"
        csv_path.write_text(
            "filename,formula,activity,length,km_value\n"
            "paper1.pdf,Fe3O4,peroxidase,10,0.05\n"
            "paper2.pdf,CuO,oxidase,20,0.08\n"
            "paper3.pdf,ZnO,catalase,15,0.06\n",
            encoding="utf-8",
        )

        # Create simple converter
        def converter(row):
            formula = row.get("formula")
            if not formula:
                return None
            return {
                "formula": formula,
                "activity": row.get("activity", "unknown"),
            }

        gt_data = load_ground_truth(csv_path, converter)

        assert len(gt_data) == 3
        assert "paper1" in gt_data
        assert gt_data["paper1"][0]["formula"] == "Fe3O4"


@pytest.mark.slow
class TestSplitsWorkflow:
    """E2E tests for data splits workflow."""

    def test_splits_loading_workflow(self, tmp_path: Path):
        """Test complete splits loading workflow."""
        # Create test splits JSON
        split_path = tmp_path / "splits.json"
        split_path.write_text(
            '{"train": ["paper1", "paper2"], "val": ["paper3"], "test": ["paper4", "paper5"]}',
            encoding="utf-8",
        )

        # Load all splits
        splits = load_all_splits(split_path)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 2
        assert len(splits["val"]) == 1
        assert len(splits["test"]) == 2

        # Load specific split
        from aee.infrastructure.storage.splits_fn import load_split

        train_docs = load_split(split_path, "train")
        assert len(train_docs) == 2
        assert "paper1" in train_docs


@pytest.mark.slow
class TestIntegrationWorkflow:
    """Integration tests combining multiple components."""

    def test_task_and_agent_workflow(self, tmp_path: Path):
        """Test workflow combining task loading and agent management."""
        # Load task from YAML
        yaml_path = Path("src/aee/domain/tasks/nanozymes/task.yaml")
        task = load_task_complete(yaml_path)

        assert task["config"].name == "nanozymes"

        # Create and save mock agent
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Use simple config dict instead of full to_dict()
        config_dict = {
            "name": task["config"].name,
            "description": task["config"].description,
            "compare_fields": task["config"].compare_fields,
        }

        agent = {
            "lm": {"model": "test-model"},
            "traces": [],
            "task_config": config_dict,
        }

        agent_path = save_agent(
            agent=agent,
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics={"f1": 0.85},
            config_snapshot={},
        )

        # Load and verify
        loaded_agent, metadata = load_agent(agent_path)
        assert loaded_agent["task_config"]["name"] == "nanozymes"
        assert metadata.metrics["f1"] == 0.85

    def test_full_data_pipeline(self, tmp_path: Path):
        """Test complete data pipeline: GT + Splits + Task."""
        # Create test data
        csv_path = tmp_path / "gt.csv"
        csv_path.write_text(
            "filename,formula,activity\n"
            "doc1.pdf,Fe3O4,peroxidase\n"
            "doc2.pdf,CuO,oxidase\n",
            encoding="utf-8",
        )

        split_path = tmp_path / "splits.json"
        split_path.write_text(
            '{"train": ["doc1"], "test": ["doc2"]}',
            encoding="utf-8",
        )

        # Load task
        yaml_path = Path("src/aee/domain/tasks/nanozymes/task.yaml")
        config = load_task_from_yaml(yaml_path)

        # Load ground truth
        def converter(row):
            formula = row.get("formula")
            if not formula:
                return None
            return {"formula": formula, "activity": row.get("activity")}

        gt_data = load_ground_truth(csv_path, converter)

        # Load splits
        train_docs = load_split(split_path, "train")

        # Verify integration
        assert len(gt_data) == 2
        assert len(train_docs) == 1
        assert config.name == "nanozymes"

        # Check that train docs have ground truth
        for doc_id in train_docs:
            assert doc_id in gt_data


# Import for test compatibility
from aee.infrastructure.storage.splits_fn import load_split  # noqa: E402
