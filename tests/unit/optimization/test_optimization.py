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
