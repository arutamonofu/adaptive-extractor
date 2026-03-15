"""Integration tests for optimization pipeline.

Tests cover:
- OptimizeAgentUseCase execution flow
- Agent state restoration after optimization
- Error handling during optimization

Note: These tests use mock data to avoid actual LLM calls and MIPROv2 optimization.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Note: clear_task_registry fixture is now in tests/conftest.py (autouse=True)


@pytest.mark.integration
@pytest.mark.slow
class TestOptimizeAgentUseCase:
    """Integration tests for OptimizeAgentUseCase."""

    @pytest.fixture
    def optimization_test_setup(self, tmp_path: Path):
        """Setup test environment for optimization tests."""
        # Create directories
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        ground_truth_dir = tmp_path / "ground_truth"
        ground_truth_dir.mkdir()

        # Create parsed documents
        for i in range(1, 4):
            doc_path = parsed_dir / f"paper{i}.md"
            doc_path.write_text(
                f"Sample scientific content about nanozymes from paper {i}.",
                encoding="utf-8",
            )

        # Create ground truth CSV
        gt_path = ground_truth_dir / "nanozymes.csv"
        header = (
            "filename,formula,activity,length,km,vmax,ph,temperature,"
            "substrate,cofactor,method,selectivity,stability,reference\n"
        )
        gt_path.write_text(
            header
            + "paper1,Fe3O4,peroxidase,10.0,0.05,100.0,7.0,25.0,TMB,None,UV-Vis,high,stable,Ref1\n"
            + "paper2,CuO,oxidase,20.0,0.08,150.0,7.5,30.0,ABTS,None,UV-Vis,medium,stable,Ref2\n"
            + "paper3,Au,catalase,15.0,0.06,120.0,7.2,28.0,H2O2,None,UV-Vis,high,unstable,Ref3\n",
            encoding="utf-8",
        )

        # Create data splits JSON
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(
            '{"train": ["paper1", "paper2"], "val": ["paper3"]}',
            encoding="utf-8",
        )

        return {
            "tmp_path": tmp_path,
            "parsed_dir": parsed_dir,
            "agents_dir": agents_dir,
            "ground_truth_dir": ground_truth_dir,
            "gt_path": gt_path,
            "splits_path": splits_path,
        }

    def test_use_case_executes_with_mocked_optimization(
        self, tmp_path: Path, optimization_test_setup
    ):
        """Test OptimizeAgentUseCase executes successfully with mocked optimization."""
        from aee.application.services import AgentManager
        from aee.application.use_cases import OptimizeAgentRequest, OptimizeAgentUseCase
        from aee.domain.tasks import load_task_from_yaml, register_config
        from aee.infrastructure.storage import AgentRepository, GroundTruthRepository

        # Load task config and set instruction file
        task_config = load_task_from_yaml("config/tasks/nanozymes.yaml")
        task_config.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
        register_config(task_config)

        # Get registered task
        from aee.domain.tasks import get_task
        registered_task = get_task("nanozymes")

        # Create directories
        agents_dir = optimization_test_setup["agents_dir"]

        # Create agent repository
        agent_repo = AgentRepository(agents_dir=agents_dir)
        agent_manager = AgentManager(agent_repo=agent_repo)

        # Create mock ground truth repo that returns valid data
        mock_gt_repo = MagicMock(spec=GroundTruthRepository)
        mock_gt_repo.load.return_value = {
            "paper1": [{"formula": "Fe3O4", "activity": "peroxidase"}],
            "paper2": [{"formula": "CuO", "activity": "oxidase"}],
            "paper3": [{"formula": "Au", "activity": "catalase"}],
        }

        # Create mock dataset builder
        mock_dataset_builder = MagicMock()
        mock_dataset_builder.build_from_split.side_effect = [
            [MagicMock(document_text="doc1", extracted_data=MagicMock())],  # train
            [MagicMock(document_text="doc2", extracted_data=MagicMock())],  # val
        ]

        # Create use case with mocked dependencies
        use_case = OptimizeAgentUseCase(
            dataset_builder=mock_dataset_builder,
            agent_manager=agent_manager,
            gt_repo=mock_gt_repo,
            tracker=None,
            enable_preflight_check=False,
        )

        # Create mock LMs
        mock_student_lm = MagicMock()
        mock_student_lm.model = "test-model"

        mock_teacher_lm = MagicMock()
        mock_teacher_lm.model = "test-teacher-model"

        # Create request
        request = OptimizeAgentRequest(
            task=registered_task["config"],
            signature_class=registered_task["signature"],
            gt_path=Path("fake/path.csv"),
            split_path=Path("fake/splits.json"),
            student_lm=mock_student_lm,
            teacher_lm=mock_teacher_lm,
            num_trials=1,
            train_limit=2,
            val_limit=1,
            model_version="test-v1",
            description="Test optimization",
            seed=42,
            num_candidates=1,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            minibatch=False,
            minibatch_size=1,
            view_data_batch_size=1,
            metric_threshold=0.5,
            init_temperature=0.5,
            verbose=False,
            initial_instruction_file="config/initial_instructions/nanozymes_sota.txt",
            instruction_hash="test_hash",
        )

        # Mock MIPROv2 to avoid actual optimization
        with patch("aee.application.use_cases.optimize_agent.MIPROv2") as mock_mipro:
            # Create mock optimized agent
            mock_optimized_agent = MagicMock()
            mock_optimized_agent.dump_state.return_value = {
                "lm": {"model": "test-model"},
                "traces": [],
                "settings": {},
            }
            mock_mipro.return_value.compile.return_value = mock_optimized_agent

            # Execute optimization
            response = use_case.execute(request)

            # Verify response
            assert response.success is True
            assert response.agent_path is not None
            assert response.agent_path.exists()
            assert response.final_metrics is not None
            assert "f1" in response.final_metrics

    def test_use_case_handles_empty_validation_set(
        self, tmp_path: Path, optimization_test_setup
    ):
        """Test OptimizeAgentUseCase handles empty validation set gracefully."""
        from aee.application.services import AgentManager, DatasetBuilder, DataValidator
        from aee.application.use_cases import OptimizeAgentRequest, OptimizeAgentUseCase
        from aee.domain.tasks import load_task_from_yaml, register_config
        from aee.infrastructure.storage import AgentRepository, DocumentRepository, GroundTruthRepository

        # Register task first
        task_config = load_task_from_yaml("config/tasks/nanozymes.yaml")
        task_config.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
        register_config(task_config)

        # Get task
        from aee.domain.tasks import get_task
        task = get_task("nanozymes")

        # Create empty validation split
        splits_path = optimization_test_setup["splits_path"]
        splits_path.write_text(
            '{"train": ["paper1", "paper2"], "val": []}',
            encoding="utf-8",
        )

        # Create repositories
        doc_repo = DocumentRepository(parsed_dir=optimization_test_setup["parsed_dir"])
        gt_repo = GroundTruthRepository()
        agent_repo = AgentRepository(agents_dir=optimization_test_setup["agents_dir"])

        # Create services
        dataset_builder = DatasetBuilder(document_repo=doc_repo, gt_repo=gt_repo)
        agent_manager = AgentManager(agent_repo=agent_repo)
        validator = DataValidator(gt_repo=gt_repo)

        # Create use case
        use_case = OptimizeAgentUseCase(
            dataset_builder=dataset_builder,
            agent_manager=agent_manager,
            gt_repo=gt_repo,
            tracker=None,
            validator=validator,
            enable_preflight_check=True,
        )

        # Create mock LMs
        mock_student_lm = MagicMock()
        mock_student_lm.model = "test-model"

        # Create request
        request = OptimizeAgentRequest(
            task=task["config"],
            signature_class=task["signature"],
            gt_path=optimization_test_setup["gt_path"],
            split_path=splits_path,
            student_lm=mock_student_lm,
            num_trials=1,
            train_limit=2,
            val_limit=0,
            model_version="test-v1",
            seed=42,
            num_candidates=1,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            minibatch=False,
            minibatch_size=1,
            view_data_batch_size=1,
            metric_threshold=0.5,
            init_temperature=0.5,
            verbose=False,
            initial_instruction_file="config/initial_instructions/nanozymes_sota.txt",
            instruction_hash="test_hash",
        )

        # Execute optimization - should fail with validation set error
        response = use_case.execute(request)

        # Verify response indicates failure
        assert response.success is False
        assert response.error_message is not None
        assert "Validation set is empty" in response.error_message or "empty" in response.error_message.lower()


@pytest.mark.integration
@pytest.mark.slow
class TestAgentStateRestoration:
    """Tests for agent state restoration after optimization."""

    @pytest.fixture(autouse=True)
    def setup_task(self):
        """Setup task for agent restoration tests."""
        from aee.domain.tasks import load_task_from_yaml, register_config

        # Register task first
        task_config = load_task_from_yaml("config/tasks/nanozymes.yaml")
        task_config.initial_instruction_file = "config/initial_instructions/nanozymes_sota.txt"
        register_config(task_config)

        yield

    def test_restored_agent_is_callable(self, tmp_path: Path):
        """Test that restored agent can be called for inference."""
        from aee.application.services import AgentManager
        from aee.domain.tasks import get_task
        from aee.infrastructure.storage import AgentRepository

        # Create test directories
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create agent with flat DSPy format
        agent_path = agents_dir / "nanozymes_test.json"
        agent_path.write_text(
            '{"lm": {"model": "test-model", "type": "mock"}, "traces": [], "settings": {}}',
            encoding="utf-8",
        )

        # Create metadata
        meta_path = agent_path.with_suffix(".meta.json")
        meta_path.write_text(
            '{"task_name": "nanozymes", "created_at": "2026-01-01T00:00:00", '
            '"model_version": "test", "metrics": {"f1": 0.85}, "config_snapshot": {}}',
            encoding="utf-8",
        )

        # Load agent via AgentManager
        repo = AgentRepository(agents_dir=agents_dir)
        manager = AgentManager(agent_repo=repo)

        # Get task for signature
        task = get_task("nanozymes")

        # Restore agent
        agent = manager.load_agent_as_object(agent_path, task)

        # Verify agent is callable (has __call__ method via UniversalExtractor)
        assert agent is not None
        assert hasattr(agent, "__call__") or hasattr(agent, "forward")
        assert hasattr(agent, "prog")
