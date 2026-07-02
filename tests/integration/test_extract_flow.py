"""Integration tests for extraction pipeline.

Tests cover:
- Batch extraction with mock agent
- Extraction output format
- Handling empty agents

Note: These tests use mock data to avoid actual LLM calls.
"""

import json
from pathlib import Path

import pytest


@pytest.mark.integration
class TestExtractFlow:
    """Integration tests for extraction pipeline."""

    @pytest.fixture
    def extraction_test_setup(self, tmp_path: Path):
        """Setup test environment for extraction tests."""
        # Create directories
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        extractions_dir = tmp_path / "extractions"
        extractions_dir.mkdir()

        # Create parsed documents as .md files
        for i in range(1, 4):
            doc_path = parsed_dir / f"paper{i}.md"
            doc_path.write_text(
                f"Sample scientific content about nanozymes from paper {i}. "
                f"Fe3O4 nanoparticles show peroxidase activity.",
                encoding="utf-8",
            )

        # Create mock agent
        agent_data = {
            "lm": {"model": "test-model", "type": "mock"},
            "traces": [],
            "settings": {"num_trials": 5},
        }
        agent_path = agents_dir / "nanozymes_test.json"
        agent_path.write_text(json.dumps(agent_data), encoding="utf-8")

        # Create agent metadata
        meta_data = {
            "task_name": "nanozymes",
            "created_at": "2026-02-19T10:00:00",
            "model_version": "test-v1",
            "metrics": {"f1": 0.85},
            "config_snapshot": {},
        }
        meta_path = agent_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta_data), encoding="utf-8")

        return {
            "tmp_path": tmp_path,
            "parsed_dir": parsed_dir,
            "agents_dir": agents_dir,
            "extractions_dir": extractions_dir,
            "agent_path": agent_path,
        }

    def test_extraction_output_format(self, experiment_model, output_model):
        """Test extraction output format is valid JSON."""
        # Create mock extraction result
        experiments = [
            experiment_model(
                formula="Fe3O4",
                activity="peroxidase",
                length=10.0,
                km_value=0.05,
                vmax_value=100.0,
                ph=7.0,
                temperature=25.0,
            ),
            experiment_model(
                formula="CuO",
                activity="oxidase",
                length=20.0,
                km_value=0.08,
                vmax_value=150.0,
                ph=7.5,
                temperature=30.0,
            ),
        ]

        output = output_model(experiments=experiments)

        # Verify serialization
        output_dict = output.model_dump()
        assert "experiments" in output_dict
        assert len(output_dict["experiments"]) == 2
        assert output_dict["experiments"][0]["formula"] == "Fe3O4"

        # Verify JSON serialization
        json_str = output.model_dump_json(indent=2)
        assert isinstance(json_str, str)

        # Verify deserialization
        loaded = output_model.model_validate_json(json_str)
        assert len(loaded.experiments) == 2
        assert loaded.experiments[0].formula == "Fe3O4"

    def test_extraction_with_empty_agent(self, extraction_test_setup):
        """Test extraction handles empty/minimal agent gracefully."""
        from ae.core.storage import AgentRepository

        repo = AgentRepository(agents_dir=extraction_test_setup["agents_dir"])

        # Load agent
        agent, metadata = repo.load(extraction_test_setup["agent_path"])

        # Verify agent structure
        assert isinstance(agent, dict)
        assert "lm" in agent
        assert metadata.model_version == "test-v1"

    def test_document_loading_for_extraction(self, extraction_test_setup):
        """Test loading documents for extraction."""
        from ae.core.storage import DocumentRepository

        repo = DocumentRepository(ingestion_dir=extraction_test_setup["parsed_dir"])

        # Load all documents
        docs = repo.load_all()

        # Verify loading
        assert len(docs) == 3
        assert "paper1" in docs
        assert "paper2" in docs
        assert "paper3" in docs

        # Verify document content
        text = docs["paper1"]
        assert "nanozymes" in text.lower()


@pytest.mark.integration
class TestAgentStateRestoration:
    """Integration tests for agent state restoration."""

    def test_agent_restoration_with_flat_dspy_format(
        self, tmp_path: Path, nanozyme_task: dict
    ):
        """Test agent restoration from flat DSPy format (lm, traces, settings)."""
        from ae.core.storage import AgentRepository
        from ae.extraction.manager import AgentManager

        # Create test directories
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create agent with flat DSPy format
        agent_path = agents_dir / "nanozymes_test.json"
        agent_path.write_text(
            '{"lm": {"model": "test-model", "type": "mock"}, "traces": [], "settings": {"num_trials": 5}}',
            encoding="utf-8",
        )

        # Create metadata
        meta_path = agent_path.with_suffix(".meta.json")
        meta_path.write_text(
            '{"created_at": "2026-01-01T00:00:00", '
            '"model_version": "test", "metrics": {"f1": 0.85}, "config_snapshot": {}}',
            encoding="utf-8",
        )

        # Load agent via AgentManager
        repo = AgentRepository(agents_dir=agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Should restore without raising error
        agent = manager.load_agent_as_object(agent_path, nanozyme_task)

        # Verify agent is callable
        assert agent is not None
        assert hasattr(agent, "prog")

    def test_agent_restoration_with_nested_prog_format(
        self, tmp_path: Path, nanozyme_task: dict
    ):
        """Test agent restoration from nested format (prog: {...})."""
        from ae.core.storage import AgentRepository
        from ae.extraction.manager import AgentManager

        # Create test directories
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create agent with nested prog format
        agent_path = agents_dir / "nanozymes_nested.json"
        import json
        with open(agent_path, "w", encoding="utf-8") as f:
            json.dump({"prog": {"predict": {"demos": []}, "some_attribute": "test_value"}}, f)

        # Create metadata
        meta_path = agent_path.with_suffix(".meta.json")
        meta_path.write_text(
            '{"created_at": "2026-01-01T00:00:00", '
            '"model_version": "test", "metrics": {"f1": 0.85}, "config_snapshot": {}}',
            encoding="utf-8",
        )

        # Load agent via AgentManager
        repo = AgentRepository(agents_dir=agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Should restore without raising error
        agent = manager.load_agent_as_object(agent_path, nanozyme_task)

        # Verify agent is callable
        assert agent is not None
        assert hasattr(agent, "prog")

    def test_agent_restoration_fails_with_invalid_format(
        self, tmp_path: Path, nanozyme_task: dict
    ):
        """Test that agent restoration fails with clear error for invalid format."""
        from ae.core.exceptions import UseCaseExecutionError
        from ae.core.storage import AgentRepository
        from ae.extraction.manager import AgentManager

        # Create test directories
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create agent with invalid format (no recognized keys)
        agent_path = agents_dir / "nanozymes_invalid.json"
        agent_path.write_text(
            '{"invalid_key": "invalid_value"}',
            encoding="utf-8",
        )

        # Create metadata
        meta_path = agent_path.with_suffix(".meta.json")
        meta_path.write_text(
            '{"created_at": "2026-01-01T00:00:00", '
            '"model_version": "test", "metrics": {"f1": 0.85}, "config_snapshot": {}}',
            encoding="utf-8",
        )

        # Load agent via AgentManager
        repo = AgentRepository(agents_dir=agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Should raise UseCaseExecutionError
        import pytest
        with pytest.raises(UseCaseExecutionError, match="Agent state format not recognized"):
            manager.load_agent_as_object(agent_path, nanozyme_task)
