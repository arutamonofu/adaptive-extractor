"""Unit tests for functional storage API.

Tests cover:
- save_agent, load_agent functions
- load_ground_truth function
- load_split, load_all_splits functions
- Backward compatibility with class-based API
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from aee.infrastructure.storage.agents_fn import (
    AgentMetadata,
    delete_agent,
    get_agent_info,
    get_latest_agent,
    list_agents,
    load_agent,
    save_agent,
)
from aee.infrastructure.storage.ground_truth_fn import (
    load_ground_truth,
    validate_coverage,
)
from aee.infrastructure.storage.splits_fn import (
    create_random_split,
    load_all_splits,
    load_split,
    save_splits,
    validate_splits,
)
from aee.shared.exceptions import AgentNotFoundError, DataNotFoundError


class TestSaveLoadAgent:
    """Tests for save_agent and load_agent functions."""

    def test_save_and_load_agent(self, tmp_path: Path):
        """Test saving and loading an agent."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent = {"lm": {"model": "test-model"}, "traces": []}
        metrics = {"f1": 0.85, "precision": 0.82}

        agent_path = save_agent(
            agent=agent,
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics=metrics,
            config_snapshot={"num_trials": 5},
        )

        assert agent_path.exists()
        assert "nanozymes" in agent_path.name

        loaded_agent, metadata = load_agent(agent_path)

        assert loaded_agent["lm"]["model"] == "test-model"
        assert metadata.task_name == "nanozymes"
        assert metadata.metrics["f1"] == 0.85

    def test_save_agent_with_metadata(self, tmp_path: Path):
        """Test saving agent with explicit metadata."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent = {"lm": {"model": "test"}}
        metadata = AgentMetadata(
            task_name="test_task",
            created_at="2026-02-19T10:00:00",
            model_version="v1",
            metrics={"f1": 0.9},
            config_snapshot={},
        )

        agent_path = save_agent(
            agent=agent,
            task_name="test_task",
            agents_dir=agents_dir,
            metadata=metadata,
        )

        loaded_agent, loaded_meta = load_agent(agent_path)

        assert loaded_meta.task_name == "test_task"
        assert loaded_meta.metrics["f1"] == 0.9

    def test_load_nonexistent_agent(self, tmp_path: Path):
        """Test loading nonexistent agent raises error."""
        with pytest.raises(AgentNotFoundError):
            load_agent(tmp_path / "nonexistent.json")

    def test_list_agents(self, tmp_path: Path):
        """Test listing agents."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Save multiple agents
        for i in range(3):
            save_agent(
                agent={"lm": {"model": "test"}},
                task_name="nanozymes",
                agents_dir=agents_dir,
                metrics={},
                config_snapshot={},
            )

        agents = list_agents(agents_dir, task_name="nanozymes")
        assert len(agents) == 3

    def test_get_latest_agent(self, tmp_path: Path):
        """Test getting latest agent."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        path1 = save_agent(
            agent={"lm": {"model": "test1"}},
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics={},
            config_snapshot={},
        )

        path2 = save_agent(
            agent={"lm": {"model": "test2"}},
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics={},
            config_snapshot={},
        )

        latest = get_latest_agent(agents_dir, "nanozymes")
        assert latest == path2

    def test_delete_agent(self, tmp_path: Path):
        """Test deleting an agent."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent_path = save_agent(
            agent={"lm": {"model": "test"}},
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics={},
            config_snapshot={},
        )

        assert agent_path.exists()
        assert agent_path.with_suffix(".meta.json").exists()

        delete_agent(agent_path)

        assert not agent_path.exists()
        assert not agent_path.with_suffix(".meta.json").exists()

    def test_get_agent_info(self, tmp_path: Path):
        """Test getting agent info."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent_path = save_agent(
            agent={"lm": {"model": "test"}},
            task_name="nanozymes",
            agents_dir=agents_dir,
            metrics={"f1": 0.85},
            config_snapshot={},
        )

        info = get_agent_info(agent_path)

        assert info["task_name"] == "nanozymes"
        assert info["metrics"]["f1"] == 0.85
        assert "path" in info


class TestLoadGroundTruth:
    """Tests for load_ground_truth function."""

    def test_load_ground_truth_success(self, tmp_path: Path):
        """Test successful GT loading."""
        csv_path = tmp_path / "gt.csv"
        csv_path.write_text(
            "filename,formula,activity,length\n"
            "paper1.pdf,Fe3O4,peroxidase,10\n"
            "paper2.pdf,CuO,oxidase,20\n",
            encoding="utf-8",
        )

        def converter(row):
            return {
                "formula": row.get("formula"),
                "activity": row.get("activity"),
            }

        gt_data = load_ground_truth(csv_path, converter)

        assert len(gt_data) == 2
        assert "paper1" in gt_data
        assert gt_data["paper1"][0]["formula"] == "Fe3O4"

    def test_load_nonexistent_csv(self, tmp_path: Path):
        """Test loading nonexistent CSV raises error."""
        with pytest.raises(DataNotFoundError):
            load_ground_truth(
                tmp_path / "nonexistent.csv",
                lambda row: None,
            )

    def test_validate_coverage(self):
        """Test GT coverage validation."""
        gt_data = {
            "paper1": [{"formula": "Fe3O4"}],
            "paper2": [{"formula": "CuO"}],
        }

        available_docs = {"paper1", "paper2", "paper3"}

        coverage = validate_coverage(gt_data, available_docs)

        assert coverage["covered_documents"] == 2
        assert coverage["total_documents"] == 3
        assert len(coverage["missing_documents"]) == 1


class TestLoadSplits:
    """Tests for load_split and load_all_splits functions."""

    def test_load_all_splits(self, tmp_path: Path):
        """Test loading all splits."""
        split_path = tmp_path / "splits.json"
        split_path.write_text(
            '{"train": ["doc1", "doc2"], "test": ["doc3"]}',
            encoding="utf-8",
        )

        splits = load_all_splits(split_path)

        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) == 2
        assert len(splits["test"]) == 1

    def test_load_split(self, tmp_path: Path):
        """Test loading specific split."""
        split_path = tmp_path / "splits.json"
        split_path.write_text(
            '{"train": ["doc1", "doc2"], "test": ["doc3"]}',
            encoding="utf-8",
        )

        train_docs = load_split(split_path, "train")

        assert len(train_docs) == 2
        assert "doc1" in train_docs

    def test_load_nonexistent_split_file(self, tmp_path: Path):
        """Test loading nonexistent split file."""
        with pytest.raises(DataNotFoundError):
            load_all_splits(tmp_path / "nonexistent.json")

    def test_save_splits(self, tmp_path: Path):
        """Test saving splits."""
        output_path = tmp_path / "output.json"

        splits = {
            "train": {"doc1", "doc2"},
            "test": {"doc3"},
        }

        saved_path = save_splits(splits, output_path)

        assert saved_path.exists()

        # Verify content
        loaded = load_all_splits(saved_path)
        assert len(loaded["train"]) == 2
        assert len(loaded["test"]) == 1

    def test_create_random_split(self):
        """Test creating random split."""
        documents = [f"doc{i}" for i in range(10)]

        splits = create_random_split(
            documents=documents,
            train_ratio=0.8,
            seed=42,
        )

        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) == 8
        assert len(splits["test"]) == 2

        # Verify reproducibility
        splits2 = create_random_split(
            documents=documents,
            train_ratio=0.8,
            seed=42,
        )
        assert splits["train"] == splits2["train"]

    def test_validate_splits(self):
        """Test splits validation."""
        splits = {
            "train": {"doc1", "doc2"},
            "test": {"doc3", "doc4"},
        }

        available_docs = {"doc1", "doc2", "doc3"}

        validation = validate_splits(splits, available_docs)

        assert "train" in validation
        assert "test" in validation
        assert validation["train"]["valid"] is True
        assert validation["test"]["valid"] is False
        assert "doc4" in validation["test"]["missing"]


class TestBackwardCompatibility:
    """Tests for backward compatibility with class-based API."""

    def test_agent_repository_delegates(self, tmp_path: Path):
        """Test AgentRepository delegates to functional API."""
        from aee.infrastructure.storage.agents import AgentRepository

        repo = AgentRepository(tmp_path / "agents")

        agent = {"lm": {"model": "test"}}
        metadata = AgentMetadata(
            task_name="test",
            created_at="2026-02-19T10:00:00",
            model_version="v1",
            metrics={},
            config_snapshot={},
        )

        path = repo.save(agent, "test", metadata)
        loaded_agent, loaded_meta = repo.load(path)

        assert loaded_agent["lm"]["model"] == "test"
        assert loaded_meta.task_name == "test"

    def test_ground_truth_repository_delegates(self, tmp_path: Path):
        """Test GroundTruthRepository delegates to functional API."""
        from aee.infrastructure.storage.ground_truth import GroundTruthRepository

        csv_path = tmp_path / "gt.csv"
        csv_path.write_text(
            "filename,formula\n"
            "paper1.pdf,Fe3O4\n",
            encoding="utf-8",
        )

        repo = GroundTruthRepository()

        def converter(row):
            return {"formula": row.get("formula")}

        gt_data = repo.load(csv_path, converter)

        assert len(gt_data) == 1

    def test_data_split_repository_delegates(self, tmp_path: Path):
        """Test DataSplitRepository delegates to functional API."""
        from aee.infrastructure.storage.splits import DataSplitRepository

        split_path = tmp_path / "splits.json"
        split_path.write_text(
            '{"train": ["doc1", "doc2"]}',
            encoding="utf-8",
        )

        repo = DataSplitRepository()
        train_docs = repo.load_split(split_path, "train")

        assert len(train_docs) == 2
