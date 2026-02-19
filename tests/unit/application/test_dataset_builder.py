"""Unit tests for DatasetBuilder service.

Tests cover:
- Building datasets from splits
- Building datasets with limits
- Handling empty datasets
- Dataset statistics
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import dspy
import pytest

from aee.application.services.dataset_builder import DatasetBuilder
from aee.domain.entities import ProcessedDocument, DocumentMetadata
from aee.domain.tasks.base import TaskDefinition
from aee.domain.tasks.nanozymes import NanozymeExtractionOutput, NanozymeExperiment
from aee.infrastructure.storage.documents import DocumentRepository
from aee.shared.exceptions import DataValidationError, UseCaseExecutionError


class MockTaskDefinition(TaskDefinition):
    """Mock task definition for testing."""

    def __init__(self):
        self._name = "test_task"
        self._description = "Test task for dataset builder"
        self._compare_fields = ["formula", "activity"]
        self._float_tol = 0.05

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def signature(self):
        # Mock signature - not used in these tests
        return None

    @property
    def output_model(self):
        return NanozymeExtractionOutput

    @property
    def experiment_model(self):
        return NanozymeExperiment

    @property
    def row_converter(self):
        # Mock converter - not used in these tests
        return None

    @property
    def compare_fields(self) -> List[str]:
        return self._compare_fields

    @property
    def float_tolerance(self) -> float:
        return self._float_tol


class TestDatasetBuilderInit:
    """Tests for DatasetBuilder initialization."""

    def test_init_with_defaults(self, tmp_path: Path):
        """Test initialization with default repositories."""
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=parsed_dir),
        )
        
        assert builder.document_repo is not None
        assert builder.gt_repo is not None
        assert builder.split_repo is not None

    def test_init_with_custom_repos(
        self,
        tmp_path: Path,
    ):
        """Test initialization with custom repositories."""
        from aee.infrastructure.storage.ground_truth import GroundTruthRepository
        from aee.infrastructure.storage.splits import DataSplitRepository
        
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=parsed_dir),
            gt_repo=GroundTruthRepository(),
            split_repo=DataSplitRepository(),
        )
        
        assert isinstance(builder.document_repo, DocumentRepository)
        assert isinstance(builder.gt_repo, GroundTruthRepository)
        assert isinstance(builder.split_repo, DataSplitRepository)


class TestBuildFromIds:
    """Tests for build_from_ids method."""

    @pytest.fixture
    def mock_gt_data(self) -> Dict[str, List[NanozymeExperiment]]:
        """Create mock ground truth data."""
        return {
            "doc1": [
                NanozymeExperiment(formula="Fe3O4", activity="peroxidase"),
            ],
            "doc2": [
                NanozymeExperiment(formula="CuO", activity="oxidase"),
                NanozymeExperiment(formula="ZnO", activity="catalase"),
            ],
            "doc3": [
                NanozymeExperiment(formula="Au", activity="peroxidase"),
            ],
        }

    @pytest.fixture
    def builder_with_docs(self, tmp_path: Path, mock_gt_data) -> DatasetBuilder:
        """Create builder with mock documents."""
        # Create parsed documents
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        
        for doc_id in mock_gt_data.keys():
            doc_path = parsed_dir / f"{doc_id}.json"
            doc_data = {
                "text_content": f"Sample content for {doc_id}",
                "metadata": {
                    "source_path": f"/path/to/{doc_id}.pdf",
                    "filename": f"{doc_id}.pdf",
                    "page_count": 10,
                },
                "tables": [],
                "images": [],
            }
            doc_path.write_text(json.dumps(doc_data), encoding="utf-8")
        
        return DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=parsed_dir),
        )

    def test_build_from_ids_success(
        self,
        builder_with_docs: DatasetBuilder,
        mock_gt_data: Dict[str, List[NanozymeExperiment]],
    ):
        """Test successful dataset building from IDs."""
        task = MockTaskDefinition()
        
        dataset = builder_with_docs.build_from_ids(
            task=task,
            document_ids=["doc1", "doc2"],
            gt_data=mock_gt_data,
        )
        
        assert len(dataset) == 2
        assert isinstance(dataset[0], dspy.Example)
        assert hasattr(dataset[0], "document_text")
        assert hasattr(dataset[0], "extracted_data")

    def test_build_from_ids_with_limit(
        self,
        builder_with_docs: DatasetBuilder,
        mock_gt_data: Dict[str, List[NanozymeExperiment]],
    ):
        """Test dataset building with limit."""
        task = MockTaskDefinition()
        
        dataset = builder_with_docs.build_from_ids(
            task=task,
            document_ids=["doc1", "doc2", "doc3"],
            gt_data=mock_gt_data,
            limit=2,
            seed=42,
        )
        
        assert len(dataset) == 2

    def test_build_from_ids_empty_result(
        self,
        builder_with_docs: DatasetBuilder,
        mock_gt_data: Dict[str, List[NanozymeExperiment]],
    ):
        """Test dataset building with no matching documents."""
        task = MockTaskDefinition()
        
        dataset = builder_with_docs.build_from_ids(
            task=task,
            document_ids=["nonexistent1", "nonexistent2"],
            gt_data=mock_gt_data,
        )
        
        assert len(dataset) == 0

    def test_build_from_ids_missing_gt_data(
        self,
        builder_with_docs: DatasetBuilder,
        mock_gt_data: Dict[str, List[NanozymeExperiment]],
    ):
        """Test dataset building when GT missing for some docs."""
        task = MockTaskDefinition()
        
        # Request doc1 (has GT) and doc4 (no GT)
        dataset = builder_with_docs.build_from_ids(
            task=task,
            document_ids=["doc1", "doc4"],
            gt_data=mock_gt_data,
        )
        
        # Should only include doc1
        assert len(dataset) == 1

    def test_build_from_ids_invalid_inputs(
        self,
        builder_with_docs: DatasetBuilder,
        mock_gt_data: Dict[str, List[NanozymeExperiment]],
    ):
        """Test dataset building with invalid inputs."""
        task = MockTaskDefinition()
        
        # Empty document_ids
        with pytest.raises(DataValidationError):
            builder_with_docs.build_from_ids(
                task=task,
                document_ids=[],
                gt_data=mock_gt_data,
            )
        
        # Invalid limit
        with pytest.raises(DataValidationError):
            builder_with_docs.build_from_ids(
                task=task,
                document_ids=["doc1"],
                gt_data=mock_gt_data,
                limit=-1,
            )
        
        # Invalid seed type
        with pytest.raises(DataValidationError):
            builder_with_docs.build_from_ids(
                task=task,
                document_ids=["doc1"],
                gt_data=mock_gt_data,
                seed="not_an_int",
            )


class TestBuildFromSplit:
    """Tests for build_from_split method."""

    @pytest.fixture
    def setup_split_test(self, tmp_path: Path):
        """Setup test environment for split-based building."""
        # Create directories
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        gt_path = tmp_path / "gt.csv"
        split_path = tmp_path / "splits.json"
        
        # Create parsed documents
        for i in range(1, 4):
            doc_path = parsed_dir / f"doc{i}.json"
            doc_data = {
                "text_content": f"Sample content for doc{i}",
                "metadata": {
                    "source_path": f"/path/to/doc{i}.pdf",
                    "filename": f"doc{i}.pdf",
                    "page_count": 10,
                },
                "tables": [],
                "images": [],
            }
            doc_path.write_text(json.dumps(doc_data), encoding="utf-8")
        
        # Create GT CSV
        gt_path.write_text(
            "filename,formula,activity\n"
            "doc1.pdf,Fe3O4,peroxidase\n"
            "doc2.pdf,CuO,oxidase\n"
            "doc3.pdf,ZnO,catalase\n",
            encoding="utf-8",
        )
        
        # Create splits JSON
        split_data = {
            "train": ["doc1", "doc2"],
            "test": ["doc3"],
        }
        split_path.write_text(json.dumps(split_data), encoding="utf-8")
        
        return {
            "tmp_path": tmp_path,
            "parsed_dir": parsed_dir,
            "gt_path": gt_path,
            "split_path": split_path,
        }

    def test_build_from_split_success(
        self,
        setup_split_test: Dict[str, Any],
    ):
        """Test successful dataset building from split."""
        from aee.domain.tasks.nanozymes import row_to_nanozyme
        
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=setup_split_test["parsed_dir"]),
        )
        task = MockTaskDefinition()
        
        # Note: build_from_split internally loads GT using the task's row_converter
        # But our MockTaskDefinition has row_converter=None, so we need to test differently
        # Let's test build_from_ids instead for full control
        
        # Load GT manually
        from aee.infrastructure.storage.ground_truth import GroundTruthRepository
        gt_repo = GroundTruthRepository()
        gt_data = gt_repo.load(
            csv_path=setup_split_test["gt_path"],
            row_converter=row_to_nanozyme,
        )
        
        # Load split manually
        from aee.infrastructure.storage.splits import DataSplitRepository
        split_repo = DataSplitRepository()
        split_ids = list(split_repo.load_split(
            split_path=setup_split_test["split_path"],
            split_name="train",
            normalize_keys=True,
        ))
        
        # Build from IDs
        dataset = builder.build_from_ids(
            task=task,
            document_ids=split_ids,
            gt_data=gt_data,
        )
        
        assert len(dataset) == 2

    def test_build_from_split_with_limit(
        self,
        setup_split_test: Dict[str, Any],
    ):
        """Test dataset building from split with limit."""
        from aee.domain.tasks.nanozymes import row_to_nanozyme
        from aee.infrastructure.storage.ground_truth import GroundTruthRepository
        from aee.infrastructure.storage.splits import DataSplitRepository
        
        # Load GT manually
        gt_repo = GroundTruthRepository()
        gt_data = gt_repo.load(
            csv_path=setup_split_test["gt_path"],
            row_converter=row_to_nanozyme,
        )
        
        # Load split manually
        split_repo = DataSplitRepository()
        split_ids = list(split_repo.load_split(
            split_path=setup_split_test["split_path"],
            split_name="train",
            normalize_keys=True,
        ))
        
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=setup_split_test["parsed_dir"]),
        )
        task = MockTaskDefinition()
        
        dataset = builder.build_from_ids(
            task=task,
            document_ids=split_ids,
            gt_data=gt_data,
            limit=1,
        )
        
        assert len(dataset) == 1

    def test_build_from_split_nonexistent_file(
        self,
        tmp_path: Path,
    ):
        """Test dataset building with nonexistent GT file."""
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=tmp_path / "parsed"),
        )
        task = MockTaskDefinition()
        
        with pytest.raises(UseCaseExecutionError):
            builder.build_from_split(
                task=task,
                gt_path=tmp_path / "nonexistent.csv",
                split_path=tmp_path / "splits.json",
                split_name="train",
            )


class TestDatasetStatistics:
    """Tests for dataset statistics."""

    def test_get_statistics_empty_dataset(self, tmp_path: Path):
        """Test statistics for empty dataset."""
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=tmp_path / "parsed"),
        )
        
        stats = builder.get_dataset_statistics([])
        
        assert stats["total_examples"] == 0
        assert stats["avg_text_length"] == 0
        assert stats["avg_experiments_per_example"] == 0

    def test_get_statistics_nonempty_dataset(
        self,
        tmp_path: Path,
    ):
        """Test statistics for non-empty dataset."""
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=tmp_path / "parsed"),
        )
        
        # Create mock dataset
        dataset = [
            dspy.Example(
                document_text="Short text",
                extracted_data=NanozymeExtractionOutput(
                    experiments=[
                        NanozymeExperiment(formula="Fe3O4", activity="peroxidase"),
                    ]
                ),
            ).with_inputs("document_text"),
            dspy.Example(
                document_text="Longer text with more content",
                extracted_data=NanozymeExtractionOutput(
                    experiments=[
                        NanozymeExperiment(formula="CuO", activity="oxidase"),
                        NanozymeExperiment(formula="ZnO", activity="catalase"),
                    ]
                ),
            ).with_inputs("document_text"),
        ]
        
        stats = builder.get_dataset_statistics(dataset)
        
        assert stats["total_examples"] == 2
        assert stats["total_experiments"] == 3
        assert stats["avg_experiments_per_example"] == 1.5
        assert stats["avg_text_length"] > 0
