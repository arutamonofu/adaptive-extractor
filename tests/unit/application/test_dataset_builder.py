"""Unit tests for DatasetBuilder service.

Tests cover:
- build_from_split: Building datasets from splits
- build_from_ids: Building datasets from specific IDs
- Input validation
- Dataset statistics
- Error handling
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aee import DatasetBuilder
from aee.infrastructure.storage import DocumentRepository
from aee.shared.exceptions import DataValidationError, UseCaseExecutionError


@pytest.fixture
def sample_task_config(tmp_path: Path) -> dict:
    """Create sample task config for testing."""
    instruction_file = tmp_path / "instruction.txt"
    instruction_file.write_text("Test instruction")

    # Create mock output model
    output_model = MagicMock()
    output_model.return_value = MagicMock()

    # Return dict as expected by the code
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
    """Create sample ground truth data."""
    return {
        "doc1": [MagicMock(formula="Fe3O4", activity="peroxidase")],
        "doc2": [MagicMock(formula="CuO", activity="oxidase")],
        "doc3": [MagicMock(formula="ZnO", activity="catalase")],
    }


@pytest.fixture
def sample_documents() -> dict:
    """Create sample documents."""
    return {
        "doc1": "Sample document about Fe3O4 nanozymes with peroxidase activity.",
        "doc2": "Study of CuO nanoparticles showing oxidase behavior.",
        "doc3": "Research on ZnO catalytic properties.",
        "doc4": "Document without ground truth.",
    }


@pytest.fixture
def dataset_builder(sample_documents: dict):
    """Create DatasetBuilder with mocked repositories."""
    # Mock document repository
    mock_doc_repo = MagicMock()
    mock_doc_repo.load_all.return_value = sample_documents

    # Mock GT repository
    mock_gt_repo = MagicMock()

    # Mock split repository
    mock_split_repo = MagicMock()
    mock_split_repo.load_split.return_value = {"doc1", "doc2", "doc3"}

    return DatasetBuilder(
        document_repo=mock_doc_repo,
        gt_repo=mock_gt_repo,
        split_repo=mock_split_repo,
    )


@pytest.mark.unit
class TestDatasetBuilderInit:
    """Tests for DatasetBuilder initialization."""

    def test_init_with_default_repos(self):
        """Test initialization with default repositories."""
        mock_doc_repo = MagicMock()
        builder = DatasetBuilder(document_repo=mock_doc_repo)

        assert builder.document_repo is mock_doc_repo
        assert builder.gt_repo is not None
        assert builder.split_repo is not None

    def test_init_with_custom_repos(self):
        """Test initialization with custom repositories."""
        mock_doc_repo = MagicMock()
        mock_gt_repo = MagicMock()
        mock_split_repo = MagicMock()

        builder = DatasetBuilder(
            document_repo=mock_doc_repo,
            gt_repo=mock_gt_repo,
            split_repo=mock_split_repo,
        )

        assert builder.document_repo is mock_doc_repo
        assert builder.gt_repo is mock_gt_repo
        assert builder.split_repo is mock_split_repo


@pytest.mark.unit
class TestBuildFromIds:
    """Tests for build_from_ids method."""

    def test_build_from_ids_success(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test successful dataset building from IDs."""
        dataset = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc2"],
            gt_data=sample_gt_data,
        )

        assert len(dataset) == 2
        assert all(hasattr(ex, 'document_text') for ex in dataset)
        assert all(hasattr(ex, 'extracted_data') for ex in dataset)

    def test_build_from_ids_with_limit(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test dataset building with limit."""
        dataset = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc2", "doc3"],
            gt_data=sample_gt_data,
            limit=2,
            seed=42,
        )

        assert len(dataset) == 2

    def test_build_from_ids_limit_reproducible(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test that limit with seed is reproducible."""
        dataset1 = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc2", "doc3"],
            gt_data=sample_gt_data,
            limit=2,
            seed=42,
        )

        dataset2 = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc2", "doc3"],
            gt_data=sample_gt_data,
            limit=2,
            seed=42,
        )

        assert len(dataset1) == len(dataset2)
        # Same seed should give same selection
        assert [ex.document_text for ex in dataset1] == [ex.document_text for ex in dataset2]

    def test_build_from_ids_no_matching_gt(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test building dataset when no IDs match GT."""
        dataset = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc99", "doc100"],  # Not in GT
            gt_data=sample_gt_data,
        )

        assert len(dataset) == 0

    def test_build_from_ids_missing_document(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test building dataset when document is missing."""
        # doc99 is in GT but not in documents
        sample_gt_data["doc99"] = [MagicMock()]

        dataset = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc99"],
            gt_data=sample_gt_data,
        )

        # Should have doc1 only
        assert len(dataset) == 1

    def test_build_from_ids_empty_document(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test building dataset with empty documents."""
        # Add empty document
        sample_gt_data["doc_empty"] = [MagicMock()]
        dataset_builder.document_repo.load_all.return_value["doc_empty"] = ""

        dataset = dataset_builder.build_from_ids(
            task=sample_task_config,
            document_ids=["doc1", "doc_empty"],
            gt_data=sample_gt_data,
        )

        # Should have doc1 only
        assert len(dataset) == 1


@pytest.mark.unit
class TestBuildFromSplit:
    """Tests for build_from_split method."""

    def test_build_from_split_success(
        self, dataset_builder, sample_task_config, sample_gt_data, tmp_path: Path
    ):
        """Test successful dataset building from split."""
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

    def test_build_from_split_with_limit(
        self, dataset_builder, sample_task_config, sample_gt_data, tmp_path: Path
    ):
        """Test building from split with limit."""
        gt_path = tmp_path / "gt.csv"
        split_path = tmp_path / "splits.json"
        split_path.write_text('{"train": ["doc1", "doc2", "doc3"]}')

        dataset = dataset_builder.build_from_split(
            task=sample_task_config,
            gt_path=gt_path,
            split_path=split_path,
            split_name="train",
            gt_data=sample_gt_data,
            limit=2,
        )

        assert len(dataset) == 2

    def test_build_from_split_error_handling(
        self, dataset_builder, sample_task_config, sample_gt_data, tmp_path: Path
    ):
        """Test error handling when split doesn't exist."""
        gt_path = tmp_path / "gt.csv"
        split_path = tmp_path / "nonexistent.json"

        dataset_builder.split_repo.load_split.side_effect = FileNotFoundError("Split not found")

        with pytest.raises(UseCaseExecutionError, match="build_from_split"):
            dataset_builder.build_from_split(
                task=sample_task_config,
                gt_path=gt_path,
                split_path=split_path,
                split_name="train",
                gt_data=sample_gt_data,
            )


@pytest.mark.unit
class TestValidateInputs:
    """Tests for input validation."""

    def test_validate_empty_document_ids_raises(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test that empty document_ids raises error."""
        with pytest.raises(DataValidationError, match="document_ids cannot be empty"):
            dataset_builder.build_from_ids(
                task=sample_task_config,
                document_ids=[],
                gt_data=sample_gt_data,
            )

    def test_validate_invalid_document_ids_type_raises(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test that non-list document_ids raises error."""
        with pytest.raises(DataValidationError, match="document_ids must be a list"):
            dataset_builder.build_from_ids(
                task=sample_task_config,
                document_ids="doc1",  # type: ignore[arg-type]
                gt_data=sample_gt_data,
            )

    def test_validate_empty_gt_data_raises(
        self, dataset_builder, sample_task_config
    ):
        """Test that empty gt_data raises error."""
        with pytest.raises(DataValidationError, match="gt_data cannot be empty"):
            dataset_builder.build_from_ids(
                task=sample_task_config,
                document_ids=["doc1"],
                gt_data={},
            )

    def test_validate_invalid_gt_data_type_raises(
        self, dataset_builder, sample_task_config
    ):
        """Test that non-dict gt_data raises error."""
        with pytest.raises(DataValidationError, match="gt_data must be a dictionary"):
            dataset_builder.build_from_ids(
                task=sample_task_config,
                document_ids=["doc1"],
                gt_data=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_validate_invalid_limit_raises(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test that invalid limit raises error."""
        with pytest.raises(DataValidationError, match="limit must be a positive integer"):
            dataset_builder.build_from_ids(
                task=sample_task_config,
                document_ids=["doc1"],
                gt_data=sample_gt_data,
                limit=0,
            )

    def test_validate_invalid_seed_raises(
        self, dataset_builder, sample_task_config, sample_gt_data
    ):
        """Test that invalid seed raises error."""
        with pytest.raises(DataValidationError, match="seed must be an integer"):
            dataset_builder.build_from_ids(
                task=sample_task_config,
                document_ids=["doc1"],
                gt_data=sample_gt_data,
                seed="42",  # type: ignore[arg-type]
            )


@pytest.mark.unit
class TestGetDatasetStatistics:
    """Tests for get_dataset_statistics method."""

    def test_get_statistics_empty_dataset(self, dataset_builder):
        """Test statistics for empty dataset."""
        stats = dataset_builder.get_dataset_statistics([])

        assert stats["total_examples"] == 0
        assert stats["avg_text_length"] == 0
        assert stats["avg_experiments_per_example"] == 0

    def test_get_statistics_success(self, dataset_builder):
        """Test statistics for valid dataset."""
        # Create mock examples
        mock_ex1 = MagicMock()
        mock_ex1.document_text = "Short text"
        mock_ex1.extracted_data.experiments = [MagicMock(), MagicMock()]

        mock_ex2 = MagicMock()
        mock_ex2.document_text = "Much longer text here"
        mock_ex2.extracted_data.experiments = [MagicMock()]

        stats = dataset_builder.get_dataset_statistics([mock_ex1, mock_ex2])

        assert stats["total_examples"] == 2
        assert stats["total_experiments"] == 3
        assert stats["avg_experiments_per_example"] == 1.5
        assert stats["avg_text_length"] > 0


@pytest.mark.unit
class TestDatasetBuilderIntegration:
    """Integration tests for DatasetBuilder with real data."""

    def test_full_workflow(self, tmp_path: Path):
        """Test complete dataset building workflow."""
        # Create documents
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        (parsed_dir / "doc1.md").write_text("Document 1 about Fe3O4.")
        (parsed_dir / "doc2.md").write_text("Document 2 about CuO.")

        # Create builder with real document repo
        doc_repo = DocumentRepository(parsed_dir=parsed_dir)

        builder = DatasetBuilder(document_repo=doc_repo)

        # Create mock GT data
        gt_data = {
            "doc1": [MagicMock(formula="Fe3O4")],
            "doc2": [MagicMock(formula="CuO")],
        }

        # Create task config as dict
        output_model = MagicMock()
        output_model.return_value = MagicMock()

        task = {
            "config": MagicMock(
                name="test",
                compare_fields=["formula"],
                float_tolerance=0.05,
            ),
            "output_model": output_model,
        }

        # Build dataset
        dataset = builder.build_from_ids(
            task=task,
            document_ids=["doc1", "doc2"],
            gt_data=gt_data,
        )

        assert len(dataset) == 2
        assert "Fe3O4" in dataset[0].document_text or "CuO" in dataset[1].document_text
