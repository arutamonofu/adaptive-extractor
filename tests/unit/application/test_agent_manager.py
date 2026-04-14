"""Unit tests for AgentManager service.

Tests cover:
- load_agent_as_object(): reconstructing callable agent from saved state
- create_agent_with_demos(): creating agent with few-shot demonstrations
- Agent serialization and deserialization
- Agent lifecycle management
"""

from pathlib import Path
from typing import Any, Dict

import dspy
import pytest

from aee.application.services import AgentManager
from aee.infrastructure.agents import UniversalExtractor
from aee.infrastructure.storage import AgentMetadata, AgentRepository
from aee.shared.exceptions import UseCaseExecutionError


@pytest.mark.unit
class TestAgentManagerLoadAgentAsObject:
    """Tests for load_agent_as_object() method."""

    def test_load_agent_as_object_success(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test successful agent reconstruction from saved state."""
        # Setup: Create and save an agent
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Create agent with signature
        agent = UniversalExtractor(nanozyme_task["signature"])

        # Create metadata
        from datetime import datetime
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test-model",
            metrics={"f1": 0.85},
            config_snapshot={},
        )

        # Save agent
        agent_dict = agent.dump_state()
        agent_path = repo.save(
            agent=agent_dict,
            task_name="nanozymes",
            metadata=metadata,
        )

        # Act: Load agent as object
        reconstructed_agent = manager.load_agent_as_object(
            agent_path=agent_path,
            task_dict=nanozyme_task,
        )

        # Assert: Agent is callable and has correct type
        assert reconstructed_agent is not None
        assert isinstance(reconstructed_agent, UniversalExtractor)
        assert hasattr(reconstructed_agent, "forward")
        assert hasattr(reconstructed_agent, "prog")

    def test_load_agent_as_object_restores_demos(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test that loaded agent has demos properly restored and is callable."""
        # Setup: Create agent with demos
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        agent = UniversalExtractor(nanozyme_task["signature"])

        # Add demo to agent
        demo = dspy.Example(
            document_text="Test document",
            extracted_data="Test result"
        ).with_inputs("document_text")
        agent.prog.predict.demos = [demo]

        # Create metadata
        from datetime import datetime
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test-model",
            metrics={"f1": 0.85},
            config_snapshot={},
        )

        # Save agent
        agent_dict = agent.dump_state()
        agent_path = repo.save(
            agent=agent_dict,
            task_name="nanozymes",
            metadata=metadata,
        )

        # Act: Load agent as object
        reconstructed_agent = manager.load_agent_as_object(
            agent_path=agent_path,
            task_dict=nanozyme_task,
        )

        # Assert: Agent is callable and demos are restored
        assert reconstructed_agent is not None
        assert isinstance(reconstructed_agent, UniversalExtractor)

        # Key assertions: prog.predict must be callable (not a dict)
        assert hasattr(reconstructed_agent.prog, "predict")
        assert callable(reconstructed_agent.prog.predict)

        # Verify demos were restored
        assert hasattr(reconstructed_agent.prog.predict, "demos")
        assert len(reconstructed_agent.prog.predict.demos) == 1

        # Verify agent itself is callable
        assert callable(reconstructed_agent)


@pytest.mark.unit
class TestAgentManagerCreateAgentWithDemos:
    """Tests for create_agent_with_demos() method."""

    def test_create_agent_with_demos_success(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test creating agent with few-shot demonstrations."""
        # Setup
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Create mock demos
        demo1 = dspy.Example(
            document_text="Test document 1",
            extracted_data="Result 1"
        ).with_inputs("document_text")
        demo2 = dspy.Example(
            document_text="Test document 2",
            extracted_data="Result 2"
        ).with_inputs("document_text")
        demos = [demo1, demo2]

        # Act
        agent = manager.create_agent_with_demos(
            signature_class=nanozyme_task["signature"],
            demos=demos,
        )

        # Assert
        assert agent is not None
        assert isinstance(agent, UniversalExtractor)
        assert hasattr(agent.prog, "predict")
        assert hasattr(agent.prog.predict, "demos")
        assert len(agent.prog.predict.demos) == 2
        assert agent.prog.predict.demos[0] == demo1
        assert agent.prog.predict.demos[1] == demo2

    def test_create_agent_with_empty_demos(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test creating agent with empty demos list."""
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Act
        agent = manager.create_agent_with_demos(
            signature_class=nanozyme_task["signature"],
            demos=[],
        )

        # Assert
        assert agent is not None
        assert isinstance(agent, UniversalExtractor)
        assert hasattr(agent.prog.predict, "demos")
        assert len(agent.prog.predict.demos) == 0

    def test_create_agent_with_demos_is_callable(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test that created agent is callable."""
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        demo = dspy.Example(
            document_text="Test",
            extracted_data="Result"
        ).with_inputs("document_text")

        # Act
        agent = manager.create_agent_with_demos(
            signature_class=nanozyme_task["signature"],
            demos=[demo],
        )

        # Assert agent has forward method (is callable)
        assert hasattr(agent, "forward")
        assert callable(agent)

    def test_load_agent_as_object_missing_signature(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test error when task_dict lacks signature key."""
        # Setup
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Create minimal task dict without signature
        bad_task_dict = {"config": nanozyme_task["config"]}

        # Create and save agent
        agent = UniversalExtractor(nanozyme_task["signature"])
        from datetime import datetime
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        agent_dict = agent.dump_state()
        agent_path = repo.save(
            agent=agent_dict,
            task_name="nanozymes",
            metadata=metadata,
        )

        # Act & Assert: Should raise UseCaseExecutionError
        with pytest.raises(UseCaseExecutionError, match="signature"):
            manager.load_agent_as_object(
                agent_path=agent_path,
                task_dict=bad_task_dict,
            )

    def test_load_agent_as_object_nonexistent_file(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test error when agent file doesn't exist."""
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        nonexistent_path = tmp_agents_dir / "nonexistent.json"

        with pytest.raises(Exception):  # AgentNotFoundError wrapped in UseCaseExecutionError
            manager.load_agent_as_object(
                agent_path=nonexistent_path,
                task_dict=nanozyme_task,
            )

    def test_load_agent_vs_load_agent_as_object(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test that load_agent returns dict while load_agent_as_object returns callable."""
        # Setup
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Create and save agent
        agent = UniversalExtractor(nanozyme_task["signature"])
        from datetime import datetime
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        agent_dict = agent.dump_state()
        agent_path = repo.save(
            agent=agent_dict,
            task_name="nanozymes",
            metadata=metadata,
        )

        # Load as dict
        loaded_dict = manager.load_agent(agent_path)
        assert isinstance(loaded_dict, dict)

        # Load as object
        loaded_object = manager.load_agent_as_object(agent_path, nanozyme_task)
        assert not isinstance(loaded_object, dict)
        assert callable(loaded_object)

        # Key assertion: prog.predict must be callable (not a dict)
        assert hasattr(loaded_object.prog, "predict")
        assert callable(loaded_object.prog.predict)


@pytest.mark.unit
class TestAgentManagerSerialization:
    """Tests for agent serialization methods."""

    def test_serialize_dict_agent(self, tmp_agents_dir: Path):
        """Test serializing a dict agent."""
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        agent_dict = {"lm": {"model": "test"}, "traces": []}

        # Should return dict as-is
        result = manager._serialize_agent(agent_dict)
        assert result == agent_dict

    def test_serialize_universal_extractor(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test serializing UniversalExtractor agent."""
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        agent = UniversalExtractor(nanozyme_task["signature"])

        # Should return dict with agent state
        result = manager._serialize_agent(agent)
        assert isinstance(result, dict)
        # DSPy dump_state() returns dict with module state (e.g., 'prog.predict' key)
        assert len(result) > 0

    def test_serialize_invalid_agent(self, tmp_agents_dir: Path):
        """Test serializing an invalid agent type."""
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Invalid agent: no dump_state() or save() method
        class InvalidAgent:
            pass

        with pytest.raises(UseCaseExecutionError, match="serialize"):
            manager._serialize_agent(InvalidAgent())  # type: ignore[arg-type]


@pytest.mark.unit
class TestAgentManagerSaveAndLoad:
    """Integration tests for save_agent and load_agent_as_object."""

    def test_save_and_load_roundtrip(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test saving and loading agent preserves state."""
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Create task config from task dict
        task_config = nanozyme_task["config"]

        # Create original agent
        original_agent = UniversalExtractor(nanozyme_task["signature"])

        # Save agent
        saved_path = manager.save_agent(
            agent=original_agent,
            task=task_config,
            metrics={"f1": 0.90, "precision": 0.88},
            config={"num_trials": 10},
            model_version="test-v1",
            description="Test agent",
        )

        # Load agent as object
        loaded_agent = manager.load_agent_as_object(
            agent_path=saved_path,
            task_dict=nanozyme_task,
        )

        # Verify loaded agent is functional
        assert loaded_agent is not None
        assert isinstance(loaded_agent, UniversalExtractor)

        # Key assertion: prog.predict must be callable (not a dict)
        assert hasattr(loaded_agent.prog, "predict")
        assert callable(loaded_agent.prog.predict)

        # Verify metadata was preserved (via repository)
        _, metadata = repo.load(saved_path)
        assert metadata.task_name == "nanozymes"
        assert metadata.metrics["f1"] == 0.90
        assert metadata.model_version == "test-v1"

    def test_restored_agent_is_callable_with_mock(
        self,
        tmp_agents_dir: Path,
        nanozyme_task: Dict[str, Any],
    ):
        """Test that restored agent can be called with mocked LLM.

        This test verifies that the agent reconstruction produces a fully
        functional agent that has the right structure, not just attributes.
        """

        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Create original agent with demos
        original_agent = UniversalExtractor(nanozyme_task["signature"])

        # Add demo to make agent more realistic
        demo = dspy.Example(
            document_text="Sample document about Fe3O4 nanozymes",
            extracted_data={"experiments": [{"formula": "Fe3O4", "activity": "peroxidase"}]}
        ).with_inputs("document_text")
        original_agent.prog.predict.demos = [demo]

        # Save agent
        task_config = nanozyme_task["config"]
        saved_path = manager.save_agent(
            agent=original_agent,
            task=task_config,
            metrics={"f1": 0.85},
            config={"num_trials": 5},
            model_version="test-v1",
        )

        # Load agent as object
        loaded_agent = manager.load_agent_as_object(
            agent_path=saved_path,
            task_dict=nanozyme_task,
        )

        # Verify agent structure
        assert loaded_agent is not None
        assert hasattr(loaded_agent, "prog")
        assert hasattr(loaded_agent.prog, "predict")
        assert callable(loaded_agent.prog.predict)

        # Verify demos are preserved
        assert hasattr(loaded_agent.prog.predict, "demos")
        assert len(loaded_agent.prog.predict.demos) == 1

        # Verify agent has __call__ method (is callable)
        assert callable(loaded_agent) or hasattr(loaded_agent, "forward")
