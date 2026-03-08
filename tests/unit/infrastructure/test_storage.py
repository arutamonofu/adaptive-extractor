"""Unit tests for infrastructure storage layer.

Tests cover:
- AgentRepository: save/load agents with metadata
- GroundTruthRepository: load GT from CSV
- DataSplitRepository: load splits from JSON
- Agent listing and management
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from aee.infrastructure.storage.agents import AgentMetadata, AgentRepository
from aee.infrastructure.storage.ground_truth import GroundTruthRepository
from aee.infrastructure.storage.splits import DataSplitRepository
from aee.shared.exceptions import (
    AgentNotFoundError,
    DataNotFoundError,
    DataValidationError,
    InvalidDataFormatError,
)


@pytest.mark.unit
class TestAgentRepository:
    """Tests for AgentRepository."""

    def test_save_and_load_agent(
        self,
        tmp_agents_dir: Path,
        sample_agent_dict: dict,
        sample_agent_metadata: dict,
    ):
        """Test saving and loading an agent with metadata."""
        repo = AgentRepository(tmp_agents_dir)
        
        metadata = AgentMetadata(**sample_agent_metadata)
        
        # Save agent
        agent_path = repo.save(
            agent=sample_agent_dict,
            task_name="nanozymes",
            metadata=metadata,
        )
        
        # Verify file exists
        assert agent_path.exists()
        assert agent_path.suffix == ".json"
        
        # Load agent
        loaded_agent, loaded_metadata = repo.load(agent_path)
        
        # Verify agent data
        assert loaded_agent["lm"]["model"] == sample_agent_dict["lm"]["model"]
        
        # Verify metadata
        assert loaded_metadata.task_name == "nanozymes"
        assert loaded_metadata.metrics["f1"] == 0.85

    def test_save_agent_custom_filename(
        self,
        tmp_agents_dir: Path,
        sample_agent_dict: dict,
        sample_agent_metadata: dict,
    ):
        """Test saving agent with custom filename."""
        repo = AgentRepository(tmp_agents_dir)
        metadata = AgentMetadata(**sample_agent_metadata)
        
        custom_filename = "custom_agent.json"
        agent_path = repo.save(
            agent=sample_agent_dict,
            task_name="nanozymes",
            metadata=metadata,
            filename=custom_filename,
        )
        
        assert agent_path.name == custom_filename
        assert agent_path.exists()

    def test_load_nonexistent_agent(self, tmp_agents_dir: Path):
        """Test loading agent that doesn't exist."""
        repo = AgentRepository(tmp_agents_dir)
        nonexistent_path = tmp_agents_dir / "nonexistent.json"
        
        with pytest.raises(AgentNotFoundError):
            repo.load(nonexistent_path)

    def test_list_agents(self, tmp_agents_dir: Path, sample_agent_dict: dict):
        """Test listing agents by task."""
        repo = AgentRepository(tmp_agents_dir)
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        
        # Save multiple agents
        repo.save(sample_agent_dict, "nanozymes", metadata)
        repo.save(sample_agent_dict, "nanozymes", metadata)
        repo.save(sample_agent_dict, "catalysts", metadata)
        
        # List all agents
        all_agents = repo.list_agents()
        assert len(all_agents) == 3
        
        # List by task
        nanozyme_agents = repo.list_agents(task_name="nanozymes")
        assert len(nanozyme_agents) == 2
        
        catalyst_agents = repo.list_agents(task_name="catalysts")
        assert len(catalyst_agents) == 1

    def test_get_latest_agent(self, tmp_agents_dir: Path, sample_agent_dict: dict):
        """Test getting the latest agent for a task."""
        repo = AgentRepository(tmp_agents_dir)

        # Use explicit timestamps to avoid time.sleep()
        metadata1 = AgentMetadata(
            task_name="nanozymes",
            created_at="2026-01-01T10:00:00",
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        _ = repo.save(sample_agent_dict, "nanozymes", metadata1)

        metadata2 = AgentMetadata(
            task_name="nanozymes",
            created_at="2026-01-01T11:00:00",  # Later timestamp
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        path2 = repo.save(sample_agent_dict, "nanozymes", metadata2)

        # Get latest
        latest = repo.get_latest("nanozymes")

        assert latest is not None
        # Latest should be the second one (by creation time)
        assert latest == path2

    def test_delete_agent(self, tmp_agents_dir: Path, sample_agent_dict: dict):
        """Test deleting an agent."""
        repo = AgentRepository(tmp_agents_dir)
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        
        agent_path = repo.save(sample_agent_dict, "nanozymes", metadata)
        metadata_path = agent_path.with_suffix(".meta.json")
        
        # Verify files exist
        assert agent_path.exists()
        assert metadata_path.exists()
        
        # Delete agent
        repo.delete(agent_path)
        
        # Verify files deleted
        assert not agent_path.exists()
        assert not metadata_path.exists()

    def test_get_agent_info(
        self,
        tmp_agents_dir: Path,
        sample_agent_dict: dict,
        sample_agent_metadata: dict,
    ):
        """Test getting agent info without full load."""
        repo = AgentRepository(tmp_agents_dir)
        metadata = AgentMetadata(**sample_agent_metadata)
        
        agent_path = repo.save(sample_agent_dict, "nanozymes", metadata)
        
        info = repo.get_agent_info(agent_path)
        
        assert info["task_name"] == "nanozymes"
        assert info["metrics"]["f1"] == 0.85
        assert "path" in info
        assert "created_at" in info


@pytest.mark.unit
class TestGroundTruthRepository:
    """Tests for GroundTruthRepository."""

    def test_load_ground_truth_success(self, sample_gt_csv: Path, row_converter):
        """Test successful GT loading from CSV."""
        repo = GroundTruthRepository()

        gt_data = repo.load(
            csv_path=sample_gt_csv,
            row_converter=row_converter,
        )
        
        # Verify structure
        assert isinstance(gt_data, dict)
        assert len(gt_data) == 3  # 3 documents in sample
        
        # Verify experiments
        assert "paper1" in gt_data
        assert len(gt_data["paper1"]) == 1
        assert gt_data["paper1"][0].formula == "Fe3O4"

    def test_load_nonexistent_csv(self, tmp_path: Path, row_converter):
        """Test loading GT from nonexistent file."""
        repo = GroundTruthRepository()
        nonexistent_path = tmp_path / "nonexistent.csv"
        
        with pytest.raises(DataNotFoundError):
            repo.load(
                csv_path=nonexistent_path,
                row_converter=row_converter,
            )

    def test_load_empty_csv(self, tmp_path: Path, row_converter):
        """Test loading empty CSV file."""
        repo = GroundTruthRepository()
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("", encoding="utf-8")
        
        with pytest.raises(InvalidDataFormatError, match="empty"):
            repo.load(
                csv_path=empty_csv,
                row_converter=row_converter,
            )

    def test_load_csv_missing_id_column(self, tmp_path: Path, row_converter):
        """Test loading CSV without valid ID column."""
        repo = GroundTruthRepository()
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text(
            "formula,activity,length\n"
            "Fe3O4,peroxidase,10\n",
            encoding="utf-8",
        )
        
        with pytest.raises(DataValidationError, match="ID column"):
            repo.load(
                csv_path=bad_csv,
                row_converter=row_converter,
            )

    def test_normalize_document_key(self):
        """Test document key normalization."""
        repo = GroundTruthRepository()
        
        # Test extension removal (only .pdf and .txt)
        assert repo._normalize_document_key("paper1.pdf") == "paper1"
        assert repo._normalize_document_key("paper1.PDF") == "paper1"
        # .json is NOT removed by this method (it's for parsed files)
        assert repo._normalize_document_key("paper1.json") == "paper1.json"
        
        # Test case normalization
        assert repo._normalize_document_key("Paper1.PDF") == "paper1"

    def test_validate_coverage(self, sample_gt_csv: Path, row_converter):
        """Test GT coverage validation."""
        repo = GroundTruthRepository()
        
        gt_data = repo.load(
            csv_path=sample_gt_csv,
            row_converter=row_converter,
        )
        
        available_docs = ["paper1", "paper2", "paper3", "paper4", "paper5"]
        
        coverage = repo.validate_coverage(gt_data, available_docs)
        
        assert "covered_documents" in coverage
        assert "coverage_percentage" in coverage
        assert "missing_documents" in coverage
        assert coverage["covered_documents"] == 3
        assert len(coverage["missing_documents"]) == 2


@pytest.mark.unit
class TestDataSplitRepository:
    """Tests for DataSplitRepository."""

    def test_load_split_success(self, sample_splits_json: Path):
        """Test successful split loading."""
        repo = DataSplitRepository()
        
        train_files = repo.load_split(
            split_path=sample_splits_json,
            split_name="train",
        )
        
        assert len(train_files) == 2
        assert "paper1" in train_files
        assert "paper2" in train_files

    def test_load_nonexistent_split_file(self, tmp_path: Path):
        """Test loading from nonexistent split file."""
        repo = DataSplitRepository()
        nonexistent_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(DataNotFoundError):
            repo.load_split(
                split_path=nonexistent_path,
                split_name="train",
            )

    def test_load_invalid_split_name(self, sample_splits_json: Path):
        """Test loading invalid split name."""
        repo = DataSplitRepository()
        
        result = repo.load_split(
            split_path=sample_splits_json,
            split_name="nonexistent",
        )
        
        # Should return empty set with warning
        assert result == set()

    def test_load_all_splits(self, sample_splits_json: Path):
        """Test loading all splits."""
        repo = DataSplitRepository()
        
        splits = repo.load_all_splits(sample_splits_json)
        
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 2
        assert len(splits["val"]) == 1
        assert len(splits["test"]) == 2  # Fixed: sample_splits.json has 2 test files

    def test_save_splits(self, tmp_path: Path):
        """Test saving splits to JSON."""
        repo = DataSplitRepository()
        output_path = tmp_path / "output_splits.json"
        
        splits = {
            "train": ["doc1", "doc2"],
            "test": ["doc3"],
        }
        
        saved_path = repo.save_splits(splits=splits, output_path=output_path)
        
        assert saved_path.exists()
        
        # Verify content
        with open(saved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert loaded["train"] == ["doc1", "doc2"]
        assert loaded["test"] == ["doc3"]

    def test_validate_splits(self, sample_splits_json: Path):
        """Test splits validation."""
        repo = DataSplitRepository()
        
        splits = repo.load_all_splits(sample_splits_json)
        available_docs = ["paper1", "paper2", "paper3"]  # Missing paper4, paper5
        
        validation = repo.validate_splits(splits, available_docs)
        
        assert "train" in validation
        assert "test" in validation
        
        # Test split should have 2 missing files (paper4, paper5)
        assert len(validation["test"]["missing"]) == 2

    def test_create_random_split(self):
        """Test creating random train/test split."""
        repo = DataSplitRepository()
        
        documents = [f"doc{i}" for i in range(10)]
        
        splits = repo.create_random_split(
            documents=documents,
            train_ratio=0.8,
            seed=42,
        )
        
        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) == 8
        assert len(splits["test"]) == 2
        
        # Verify reproducibility with seed
        splits2 = repo.create_random_split(
            documents=documents,
            train_ratio=0.8,
            seed=42,
        )
        assert splits["train"] == splits2["train"]
        assert splits["test"] == splits2["test"]


@pytest.mark.unit
class TestAgentMetadata:
    """Tests for AgentMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating AgentMetadata."""
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at="2026-02-19T10:00:00",
            model_version="test-model-v1",
            metrics={"f1": 0.85},
            config_snapshot={"num_trials": 10},
        )
        
        assert metadata.task_name == "nanozymes"
        assert metadata.metrics["f1"] == 0.85
        assert metadata.git_commit is None  # Optional field

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        from dataclasses import asdict
        
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at="2026-02-19T10:00:00",
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        
        metadata_dict = asdict(metadata)
        
        assert metadata_dict["task_name"] == "nanozymes"
        assert metadata_dict["model_version"] == "test"
