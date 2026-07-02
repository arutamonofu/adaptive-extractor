from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ae import DatasetBuilder
from ae.core.exceptions import DataValidationError, UseCaseExecutionError
from ae.optimization.tracking import ExperimentTracker

# --- DatasetBuilder Fixtures & Tests ---

@pytest.fixture
def sample_task_config(tmp_path: Path) -> dict:
    output_model = MagicMock()
    output_model.return_value = MagicMock()
    return {
        "config": MagicMock(
            name="test_task",
            compare_fields=["formula", "activity"],
            float_tolerance=0.05,
        ),
        "output_model": output_model,
    }


@pytest.fixture
def sample_gt_data() -> dict:
    return {
        "doc1": [MagicMock(formula="Fe3O4", activity="peroxidase")],
        "doc2": [MagicMock(formula="CuO", activity="oxidase")],
        "doc3": [MagicMock(formula="ZnO", activity="catalase")],
    }


@pytest.fixture
def sample_documents() -> dict:
    return {
        "doc1": "Sample document about Fe3O4 nanozymes with peroxidase activity.",
        "doc2": "Study of CuO nanoparticles showing oxidase behavior.",
        "doc3": "Research on ZnO catalytic properties.",
    }


@pytest.fixture
def dataset_builder(sample_documents: dict):
    mock_doc_repo = MagicMock()
    mock_doc_repo.load_all.return_value = sample_documents
    mock_gt_repo = MagicMock()
    mock_split_repo = MagicMock()
    mock_split_repo.load_split.return_value = {"doc1", "doc2", "doc3"}
    return DatasetBuilder(
        document_repo=mock_doc_repo,
        gt_repo=mock_gt_repo,
        split_repo=mock_split_repo,
    )


@pytest.mark.unit
class TestDatasetBuilder:
    """Consolidated unit tests for DatasetBuilder."""

    def test_init_repos(self):
        mock_doc_repo = MagicMock()
        builder = DatasetBuilder(document_repo=mock_doc_repo)
        assert builder.document_repo is mock_doc_repo
        assert builder.gt_repo is not None

    def test_build_from_ids_success_and_limit(self, dataset_builder, sample_task_config, sample_gt_data):
        # Successful build
        dataset = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc2"],
            gt_data=sample_gt_data,
        )
        assert len(dataset) == 2
        assert all(hasattr(ex, 'document_text') for ex in dataset)

        # Build with limit and seed reproducibility
        dataset_lim1 = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc2", "doc3"],
            gt_data=sample_gt_data,
            limit=2,
            seed=42,
        )
        dataset_lim2 = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc2", "doc3"],
            gt_data=sample_gt_data,
            limit=2,
            seed=42,
        )
        assert len(dataset_lim1) == 2
        assert [ex.document_text for ex in dataset_lim1] == [ex.document_text for ex in dataset_lim2]

    def test_build_from_split_success_and_error(self, dataset_builder, sample_task_config, sample_gt_data, tmp_path: Path):
        gt_path = tmp_path / "gt.csv"
        gt_path.write_text("filename,formula,activity\ndoc1.pdf,Fe3O4,peroxidase")
        split_path = tmp_path / "splits.json"
        split_path.write_text('{"train": ["doc1", "doc2"]}')

        dataset = dataset_builder.build_from_split(
            task=sample_task_config,
            gt_path=gt_path,
            split_path=split_path,
            split_name="train",
            gt_data=sample_gt_data,
        )
        assert len(dataset) > 0

        # Handle split loading error
        dataset_builder.split_repo.load_split.side_effect = FileNotFoundError("Split not found")
        with pytest.raises(UseCaseExecutionError, match="build_from_split"):
            dataset_builder.build_from_split(
                task=sample_task_config,
                gt_path=gt_path,
                split_path=tmp_path / "nonexistent.json",
                split_name="train",
                gt_data=sample_gt_data,
            )

    def test_validate_inputs(self, dataset_builder, sample_task_config, sample_gt_data):
        with pytest.raises(DataValidationError, match="document_ids cannot be empty"):
            dataset_builder.build_from_ids(task=sample_task_config, document_ids=[], gt_data=sample_gt_data)

        with pytest.raises(DataValidationError, match="gt_data cannot be empty"):
            dataset_builder.build_from_ids(task=sample_task_config, document_ids=["doc1"], gt_data={})

        with pytest.raises(DataValidationError, match="limit must be a positive integer"):
            dataset_builder.build_from_ids(task=sample_task_config, document_ids=["doc1"], gt_data=sample_gt_data, limit=0)

    def test_get_statistics(self, dataset_builder):
        mock_ex1 = MagicMock()
        mock_ex1.document_text = "Text1"
        mock_ex1.extracted_data.experiments = [MagicMock()]

        stats = dataset_builder.get_dataset_statistics([mock_ex1])
        assert stats["total_examples"] == 1
        assert stats["total_experiments"] == 1
        assert stats["avg_text_length"] == 5


# --- ExperimentTracker Helpers & Tests ---

def create_mock_mlflow():
    mock = MagicMock()
    mock.set_experiment.return_value = MagicMock(experiment_id="test-123")
    mock.start_run.return_value = MagicMock(info=MagicMock(run_id="run-123"))
    mock.dspy = MagicMock()
    return mock


@pytest.fixture
def mlflow_mock():
    mock = create_mock_mlflow()
    with patch.dict("sys.modules", {"mlflow": mock}):
        yield mock


@pytest.mark.unit
class TestExperimentTracker:
    """Consolidated unit tests for ExperimentTracker."""

    def test_initialization(self, mlflow_mock):
        tracker = ExperimentTracker(experiment_name="test_exp", tracking_uri="sqlite:///test.db")
        assert tracker.experiment_name == "test_exp"
        assert tracker.enabled is True
        assert tracker.experiment_id is None
        mlflow_mock.set_tracking_uri.assert_not_called()

        tracker.start_run()
        assert tracker.experiment_id == "test-123"
        mlflow_mock.set_tracking_uri.assert_called_once_with("sqlite:///test.db")

        # Disabled mode
        tracker_disabled = ExperimentTracker(experiment_name="test_exp", enabled=False)
        assert tracker_disabled.enabled is False

    def test_run_lifecycle_and_logging(self, mlflow_mock, tmp_path: Path):
        tracker = ExperimentTracker(experiment_name="test")

        # Logging before run start should not call mlflow
        tracker.log_params({"key": "val"})
        mlflow_mock.log_params.assert_not_called()

        # Start run and verify loggers
        tracker.start_run(run_name="run_name")
        assert tracker.is_active is True
        assert tracker.run_id == "run-123"

        tracker.log_params({"num_trials": 10})
        mlflow_mock.log_params.assert_called_with({"num_trials": "10"})

        tracker.log_metrics({"f1": 0.85}, step=1)
        mlflow_mock.log_metrics.assert_called_with({"f1": 0.85}, step=1)

        # Log artifact
        art_path = tmp_path / "art.txt"
        art_path.write_text("Hello")
        tracker.log_artifact(art_path)
        mlflow_mock.log_artifact.assert_called_with(str(art_path))

        # Log optimization results
        tracker.log_optimization_results(
            metrics={"f1": 0.85},
            config={"num_trials": 10},
            agent_path=art_path,
            task_name="nanozymes",
        )
        assert mlflow_mock.set_tags.called

        # End run
        tracker.end_run()
        mlflow_mock.end_run.assert_called_once()
        assert tracker.is_active is False

    def test_context_manager(self, mlflow_mock):
        tracker = ExperimentTracker(experiment_name="test")
        with tracker.start_run(run_name="ctx_run"):
            assert tracker.is_active is True
        mlflow_mock.end_run.assert_called()

    def test_dspy_autolog(self, mlflow_mock):
        tracker = ExperimentTracker(experiment_name="test")
        tracker.enable_dspy_autolog()
        mlflow_mock.dspy.autolog.assert_called()
        assert tracker._dspy_autolog_enabled is True

        tracker.disable_dspy_autolog()
        mlflow_mock.dspy.autolog.assert_called_with(disable=True)
        assert tracker._dspy_autolog_enabled is False

    def test_log_dspy_model(self, mlflow_mock):
        tracker = ExperimentTracker(experiment_name="test")
        tracker.start_run(run_name="dspy_run")

        mock_dspy_model = MagicMock()

        # Test successful logging
        tracker.log_dspy_model(mock_dspy_model, name="dspy_model", save_program=True)
        mlflow_mock.dspy.log_model.assert_called_with(
            dspy_model=mock_dspy_model,
            name="dspy_model",
            signature=None,
            input_example=None,
            use_dspy_model_save=True,
            save_program=True,
        )

        # Test fallback when exception is raised during log_model
        mlflow_mock.dspy.log_model.side_effect = Exception("save_program error")
        with patch.object(tracker, "_log_dspy_model_fallback") as mock_fallback:
            tracker.log_dspy_model(mock_dspy_model, name="dspy_model")
            mock_fallback.assert_called_once_with(mock_dspy_model)


@pytest.mark.unit
class TestCheckpointingMIPROv2:
    """Unit tests for CheckpointingMIPROv2."""

    def test_save_checkpoint(self, tmp_path, monkeypatch):
        from ae.optimization.mipro import CheckpointingMIPROv2
        
        # Override data directory to use a temp path
        monkeypatch.setattr("ae.optimization.mipro.Path", lambda *args: Path(tmp_path, *args))

        # Mock the student model/program
        mock_program = MagicMock()
        mock_program.dump_state.return_value = {"key": "val"}

        # Instantiate teleprompter
        teleprompter = CheckpointingMIPROv2(
            metric=lambda x, y: 1.0,
            task_name="test_task",
            prompt_model=MagicMock(),
            task_model=MagicMock(),
        )
        
        # Save checkpoint
        teleprompter._save_checkpoint(mock_program, 0.95, params={"param1": 1})
        
        # Verify it saved
        checkpoint_path = tmp_path / "data" / "processed" / "agents" / "test_task_checkpoint.json"
        assert checkpoint_path.exists()
        
        import json
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
            
        assert data["score"] == 0.95
        assert data["program"] == {"key": "val"}
        assert data["params"] == {"param1": 1}

    @patch("ae.optimization.mipro._import_optuna")
    @patch("dspy.teleprompt.utils.eval_candidate_program")
    @patch("dspy.teleprompt.utils.save_candidate_program")
    def test_optimize_prompt_parameters_graceful_exit(self, mock_save, mock_eval, mock_import_optuna, tmp_path, monkeypatch):
        from ae.optimization.mipro import CheckpointingMIPROv2
        import threading
        
        # Set up optuna mock
        mock_optuna = MagicMock()
        mock_import_optuna.return_value = mock_optuna
        
        # Mock study.optimize to invoke our objective function once
        def mock_optimize(objective, n_trials):
            trial = MagicMock()
            trial.number = 0
            objective(trial)
            
        mock_study = MagicMock()
        mock_study.optimize.side_effect = mock_optimize
        mock_optuna.create_study.return_value = mock_study
        
        # Mock other dependencies
        mock_eval.return_value = MagicMock(score=0.5)
        
        # Configure cancel_event to trigger cancellation
        cancel_event = threading.Event()
        
        teleprompter = CheckpointingMIPROv2(
            metric=lambda x, y: 1.0,
            task_name="test_task",
            cancel_event=cancel_event,
            prompt_model=MagicMock(),
            task_model=MagicMock(),
        )
        monkeypatch.setattr("ae.optimization.mipro.Path", lambda *args: Path(tmp_path, *args))
        
        # Setup mock program
        mock_program = MagicMock()
        mock_predictors = MagicMock()
        mock_program.predictors.return_value = [mock_predictors]
        mock_program.deepcopy.return_value = mock_program
        
        # Trigger cancellation
        cancel_event.set()
        
        res = teleprompter._optimize_prompt_parameters(
            program=mock_program,
            instruction_candidates={},
            demo_candidates=None,
            evaluate=MagicMock(),
            valset=[],
            num_trials=5,
            minibatch=False,
            minibatch_size=5,
            minibatch_full_eval_steps=5,
            seed=42
        )
        
        assert res is not None


@pytest.mark.unit
class TestOptimizeAgentDegradedMode:
    """Unit tests for OptimizeAgentUseCase degraded mode fallbacks."""

    @patch("ae.optimization.use_case.CheckpointingMIPROv2")
    @patch("ae.optimization.use_case.save_optimization_history")
    def test_run_optimization_teacher_fails_falls_back_to_student(
        self, mock_save, mock_mipro_class, tmp_path
    ):
        from ae.optimization.use_case import OptimizeAgentUseCase, OptimizeAgentRequest
        
        # Configure CheckpointingMIPROv2 mocks to fail on first attempt, succeed on second (student fallback)
        mock_mipro_instance_1 = MagicMock()
        mock_mipro_instance_1.compile.side_effect = Exception("Teacher API Error 500")
        
        mock_mipro_instance_2 = MagicMock()
        mock_optimized_agent = MagicMock()
        mock_mipro_instance_2.compile.return_value = mock_optimized_agent
        
        mock_mipro_class.side_effect = [mock_mipro_instance_1, mock_mipro_instance_2]
        
        # Mock other fields
        mock_builder = MagicMock()
        mock_manager = MagicMock()
        use_case = OptimizeAgentUseCase(dataset_builder=mock_builder, agent_manager=mock_manager)
        
        mock_task = MagicMock()
        mock_task.config.name = "test_task"
        
        mock_student_lm = MagicMock()
        mock_student_lm.model = "student-model"
        mock_teacher_lm = MagicMock()
        mock_teacher_lm.model = "teacher-model"
        
        request = OptimizeAgentRequest(
            task=mock_task,
            signature_class=MagicMock(),
            gt_path=Path("gt.csv"),
            split_path=Path("split.json"),
            student_lm=mock_student_lm,
            teacher_lm=mock_teacher_lm,
            num_trials=2,
            seed=42,
            num_candidates=2,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            minibatch=False,
            minibatch_size=2,
            view_data_batch_size=2,
            metric_threshold=0.9,
            init_temperature=0.7,
            max_errors=3,
        )
        
        base_agent = MagicMock()
        trainset = []
        valset = []
        metric = MagicMock()
        
        res = use_case._run_optimization(
            base_agent=base_agent,
            trainset=trainset,
            valset=valset,
            metric=metric,
            request=request,
        )
        
        # The first call failed, the second compile succeeded, returning mock_optimized_agent
        assert res == mock_optimized_agent
        # Check that CheckpointingMIPROv2 was instantiated twice (first with teacher, second with student as teacher)
        assert mock_mipro_class.call_count == 2
        # First call has prompt_model=teacher
        first_call_args = mock_mipro_class.call_args_list[0]
        # Second call should have prompt_model=student
        second_call_args = mock_mipro_class.call_args_list[1]
        assert second_call_args[1]["prompt_model"] is not request.teacher_lm

    @patch("ae.optimization.use_case.CheckpointingMIPROv2")
    @patch("ae.optimization.use_case.save_optimization_history")
    def test_run_optimization_all_fail_falls_back_to_zero_shot(
        self, mock_save, mock_mipro_class, tmp_path
    ):
        from ae.optimization.use_case import OptimizeAgentUseCase, OptimizeAgentRequest
        
        # Configure CheckpointingMIPROv2 mocks to fail on first and second, succeed on third (zero shot)
        mock_mipro_instance_1 = MagicMock()
        mock_mipro_instance_1.compile.side_effect = Exception("Teacher API Error 500")
        
        mock_mipro_instance_2 = MagicMock()
        mock_mipro_instance_2.compile.side_effect = Exception("Student API Error 500")
        
        mock_mipro_instance_3 = MagicMock()
        mock_optimized_agent = MagicMock()
        mock_mipro_instance_3.compile.return_value = mock_optimized_agent
        
        mock_mipro_class.side_effect = [
            mock_mipro_instance_1,
            mock_mipro_instance_2,
            mock_mipro_instance_3,
        ]
        
        # Mock other fields
        mock_builder = MagicMock()
        mock_manager = MagicMock()
        use_case = OptimizeAgentUseCase(dataset_builder=mock_builder, agent_manager=mock_manager)
        
        mock_task = MagicMock()
        mock_task.config.name = "test_task"
        
        mock_student_lm = MagicMock()
        mock_student_lm.model = "student-model"
        mock_teacher_lm = MagicMock()
        mock_teacher_lm.model = "teacher-model"
        
        request = OptimizeAgentRequest(
            task=mock_task,
            signature_class=MagicMock(),
            gt_path=Path("gt.csv"),
            split_path=Path("split.json"),
            student_lm=mock_student_lm,
            teacher_lm=mock_teacher_lm,
            num_trials=2,
            seed=42,
            num_candidates=2,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            minibatch=False,
            minibatch_size=2,
            view_data_batch_size=2,
            metric_threshold=0.9,
            init_temperature=0.7,
            max_errors=3,
        )
        
        base_agent = MagicMock()
        trainset = []
        valset = []
        metric = MagicMock()
        
        res = use_case._run_optimization(
            base_agent=base_agent,
            trainset=trainset,
            valset=valset,
            metric=metric,
            request=request,
        )
        
        assert res == mock_optimized_agent
        # Check that CheckpointingMIPROv2 was instantiated three times
        assert mock_mipro_class.call_count == 3
        
        # Third call has max_bootstrapped_demos=0
        third_call_args = mock_mipro_class.call_args_list[2]
        assert third_call_args[1]["max_bootstrapped_demos"] == 0



