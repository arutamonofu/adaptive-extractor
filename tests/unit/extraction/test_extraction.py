from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import dspy
import pytest

from ae.core.exceptions import UseCaseExecutionError
from ae.core.storage import AgentMetadata, AgentRepository
from ae.extraction.agent import UniversalExtractor
from ae.extraction.manager import AgentManager


@pytest.mark.unit
class TestAgentManager:
    """Consolidated unit tests for AgentManager service and serialization."""

    def test_load_agent_as_object_and_demos(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # 1. Setup: Create agent with demos
        agent = UniversalExtractor(nanozyme_task["signature"])
        demo = dspy.Example(document_text="Test doc", extracted_data="Test res").with_inputs("document_text")
        agent.prog.predict.demos = [demo]

        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test-model",
            metrics={"f1": 0.85},
            config_snapshot={},
        )

        agent_dict = agent.dump_state()
        agent_path = repo.save(agent=agent_dict, task_name="nanozymes", metadata=metadata)

        # 2. Reconstruct agent as object
        reconstructed = manager.load_agent_as_object(agent_path=agent_path, task_dict=nanozyme_task)
        assert reconstructed is not None
        assert isinstance(reconstructed, UniversalExtractor)
        assert callable(reconstructed)
        assert len(reconstructed.prog.predict.demos) == 1


    def test_create_agent_with_demos(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        demo = dspy.Example(document_text="Demo doc", extracted_data="Demo res").with_inputs("document_text")
        agent = manager.create_agent_with_demos(signature_class=nanozyme_task["signature"], demos=[demo])

        assert isinstance(agent, UniversalExtractor)
        assert len(agent.prog.predict.demos) == 1
        assert agent.prog.predict.demos[0].document_text == "Demo doc"

    def test_load_agent_as_object_missing_signature(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        agent = UniversalExtractor(nanozyme_task["signature"])
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test",
            metrics={},
            config_snapshot={},
        )
        agent_path = repo.save(agent=agent.dump_state(), task_name="nanozymes", metadata=metadata)

        bad_task_dict = {"config": nanozyme_task["config"]}
        with pytest.raises(UseCaseExecutionError, match="signature"):
            manager.load_agent_as_object(agent_path=agent_path, task_dict=bad_task_dict)

    def test_serialization(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Dict serialization (passthrough)
        agent_dict = {"lm": {"model": "test"}, "traces": []}
        assert manager._serialize_agent(agent_dict) == agent_dict

        # UniversalExtractor serialization
        agent = UniversalExtractor(nanozyme_task["signature"])
        serialized = manager._serialize_agent(agent)
        assert isinstance(serialized, dict)
        assert len(serialized) > 0

        # Invalid agent serialization
        class InvalidAgent:
            pass
        with pytest.raises(UseCaseExecutionError, match="serialize"):
            manager._serialize_agent(InvalidAgent())  # type: ignore[arg-type]

    def test_save_and_load_roundtrip(self, tmp_agents_dir: Path, nanozyme_task: Dict[str, Any]):
        repo = AgentRepository(agents_dir=tmp_agents_dir)
        manager = AgentManager(agent_repo=repo)

        original_agent = UniversalExtractor(nanozyme_task["signature"])
        saved_path = manager.save_agent(
            agent=original_agent,
            task=nanozyme_task["config"],
            metrics={"f1": 0.90},
            config={"num_trials": 10},
            model_version="test-v1",
        )

        # Load as dict
        loaded_dict = manager.load_agent(saved_path)
        assert isinstance(loaded_dict, dict)

        # Load as object
        loaded_object = manager.load_agent_as_object(saved_path, nanozyme_task)
        assert isinstance(loaded_object, UniversalExtractor)
        assert callable(loaded_object)
