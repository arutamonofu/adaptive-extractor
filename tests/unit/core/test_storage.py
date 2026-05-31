import json

import pytest

from ae.core.storage import (
    AgentMetadata,
    AgentRepository,
    GroundTruthRepository,
    create_random_split,
    load_agent,
    load_ground_truth,
    load_split,
    save_agent,
    save_splits,
)


@pytest.mark.unit
class TestStorage:
    """Consolidated unit tests for storage repositories and functional API."""

    def test_agent_storage(self, tmp_path, sample_agent_dict, sample_agent_metadata):
        """Test functional and class-based agent saving, loading, listing, and deletion."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Functional save
        metadata = AgentMetadata(**sample_agent_metadata)
        agent_path = save_agent(sample_agent_dict, "nanozymes", agents_dir, metadata=metadata)

        # Functional load
        loaded_agent, loaded_meta = load_agent(agent_path)
        assert loaded_agent["lm"]["model"] == "test-model"
        assert loaded_meta.task_name == "nanozymes"

        # Repository-based list and delete
        repo = AgentRepository(agents_dir=agents_dir)
        agents = repo.list_agents(task_name="nanozymes")
        assert len(agents) == 1
        assert agents[0] == agent_path

        repo.delete(agent_path)
        assert len(repo.list_agents(task_name="nanozymes")) == 0

    def test_ground_truth_storage(self, tmp_path, sample_gt_csv, row_converter):
        """Test loading ground truth and document key normalization."""
        # Functional load
        gt_data = load_ground_truth(sample_gt_csv, row_converter)
        assert "paper1" in gt_data
        assert gt_data["paper1"][0].formula == "Fe3O4"

        # Repository load
        repo = GroundTruthRepository()
        assert repo._normalize_document_key("paper1.pdf") == "paper1"
        assert repo._normalize_document_key("paper2.PDF") == "paper2"

    def test_data_splits(self, tmp_path, sample_splits_json):
        """Test loading, saving, creating, and validating train/val/test data splits."""
        # Load functional
        splits = load_split(sample_splits_json, "train")
        assert splits == {"paper1", "paper2"}

        # Save and create random split
        out_splits_file = tmp_path / "new_splits.json"
        all_ids = ["paper1", "paper2", "paper3", "paper4", "paper5"]

        splits_dict = create_random_split(all_ids, train_ratio=0.6)
        save_splits(splits_dict, out_splits_file)

        assert out_splits_file.exists()
        loaded = json.loads(out_splits_file.read_text())
        assert len(loaded["train"]) == 3
        assert len(loaded["test"]) == 2

    def test_backward_compatibility(self, tmp_path):
        """Verify class-based repository wrappers delegate to the functional API."""
        repo = AgentRepository(agents_dir=tmp_path)
        assert repo.agents_dir == tmp_path

