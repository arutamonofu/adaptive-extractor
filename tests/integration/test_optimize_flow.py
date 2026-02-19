"""Integration tests for optimization pipeline.

Tests cover:
- Full optimization cycle (GT → Dataset → Agent)
- Pre-flight validation
- Agent metadata saving
- Optimization with different configs

Note: These tests use mock LLM to avoid actual API calls.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

from aee.application.services import DatasetBuilder
from aee.application.services.dataset_builder import DatasetBuilder
from aee.domain.evaluation import ExperimentMatcher, TaskMetric
from aee.domain.tasks.nanozymes import (
    NanozymeExperiment,
    NanozymeExtractionOutput,
    row_to_nanozyme,
)
from aee.infrastructure.storage.agents import AgentMetadata, AgentRepository
from aee.infrastructure.storage.documents import DocumentRepository
from aee.infrastructure.storage.ground_truth import GroundTruthRepository
from aee.infrastructure.storage.splits import DataSplitRepository


class TestOptimizeFlow:
    """Integration tests for optimization pipeline."""

    @pytest.fixture
    def optimization_test_setup(self, tmp_path: Path):
        """Setup test environment for optimization tests."""
        # Create directories
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        gt_path = tmp_path / "gt.csv"
        split_path = tmp_path / "splits.json"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        # Create parsed documents
        for i in range(1, 4):
            doc_path = parsed_dir / f"paper{i}_parsed.json"
            doc_data = {
                "text_content": f"Sample scientific content about nanozymes from paper {i}. "
                               f"Fe3O4 nanoparticles show peroxidase activity with Km={0.05*i} mM.",
                "metadata": {
                    "source_path": f"/path/to/paper{i}.pdf",
                    "filename": f"paper{i}.pdf",
                    "page_count": 10,
                },
                "tables": [],
                "images": [],
            }
            doc_path.write_text(json.dumps(doc_data), encoding="utf-8")
        
        # Create GT CSV
        gt_path.write_text(
            "filename,formula,activity,length,km_value,vmax_value,ph,temperature\n"
            "paper1.pdf,Fe3O4,peroxidase,10,0.05,100,7.0,25.0\n"
            "paper2.pdf,CuO,oxidase,20,0.08,150,7.5,30.0\n"
            "paper3.pdf,ZnO,catalase,15,0.06,120,6.8,28.0\n",
            encoding="utf-8",
        )
        
        # Create splits JSON
        split_data = {
            "train": ["paper1", "paper2"],
            "val": ["paper3"],
        }
        split_path.write_text(json.dumps(split_data), encoding="utf-8")
        
        return {
            "tmp_path": tmp_path,
            "parsed_dir": parsed_dir,
            "gt_path": gt_path,
            "split_path": split_path,
            "agents_dir": agents_dir,
        }

    def test_dataset_builder_integration(self, optimization_test_setup):
        """Test dataset builder with real repositories."""
        from aee.domain.tasks.nanozymes import NanozymeTask
        
        # Load instruction (use fallback for tests)
        instruction = """Extract nanozyme experiments from scientific articles."""
        
        task = NanozymeTask(initial_instruction=instruction)
        
        # Build dataset
        builder = DatasetBuilder(
            document_repo=DocumentRepository(parsed_dir=optimization_test_setup["parsed_dir"]),
        )
        
        dataset = builder.build_from_split(
            task=task,
            gt_path=optimization_test_setup["gt_path"],
            split_path=optimization_test_setup["split_path"],
            split_name="train",
        )
        
        # Verify dataset
        assert len(dataset) == 2
        assert hasattr(dataset[0], "document_text")
        assert hasattr(dataset[0], "extracted_data")
        assert isinstance(dataset[0].extracted_data, NanozymeExtractionOutput)

    def test_ground_truth_loading_integration(self, optimization_test_setup):
        """Test ground truth loading with row converter."""
        gt_repo = GroundTruthRepository()
        
        gt_data = gt_repo.load(
            csv_path=optimization_test_setup["gt_path"],
            row_converter=row_to_nanozyme,
        )
        
        # Verify structure
        assert isinstance(gt_data, dict)
        assert len(gt_data) == 3
        assert "paper1" in gt_data
        assert len(gt_data["paper1"]) == 1
        assert gt_data["paper1"][0].formula == "Fe3O4"

    def test_splits_loading_integration(self, optimization_test_setup):
        """Test splits loading."""
        split_repo = DataSplitRepository()
        
        train_ids = split_repo.load_split(
            split_path=optimization_test_setup["split_path"],
            split_name="train",
            normalize_keys=True,
        )
        
        val_ids = split_repo.load_split(
            split_path=optimization_test_setup["split_path"],
            split_name="val",
            normalize_keys=True,
        )
        
        assert len(train_ids) == 2
        assert len(val_ids) == 1
        assert "paper1" in train_ids
        assert "paper2" in train_ids
        assert "paper3" in val_ids

    def test_agent_repository_integration(
        self,
        optimization_test_setup,
    ):
        """Test agent save/load with metadata."""
        repo = AgentRepository(agents_dir=optimization_test_setup["agents_dir"])
        
        # Create mock agent
        mock_agent = {
            "lm": {"model": "test-model", "type": "mock"},
            "traces": [],
            "settings": {"num_trials": 5},
        }
        
        metadata = AgentMetadata(
            task_name="nanozymes",
            created_at=datetime.now().isoformat(),
            model_version="test-v1",
            metrics={"f1": 0.85, "precision": 0.82, "recall": 0.88},
            config_snapshot={"num_trials": 5},
            initial_instruction_file="nanozymes_sota.txt",
            instruction_hash="abc123def456",
        )
        
        # Save agent
        agent_path = repo.save(
            agent=mock_agent,
            task_name="nanozymes",
            metadata=metadata,
        )
        
        # Verify save
        assert agent_path.exists()
        assert "nanozymes" in agent_path.name
        
        # Load agent
        loaded_agent, loaded_meta = repo.load(agent_path)
        
        # Verify load
        assert loaded_agent["lm"]["model"] == "test-model"
        assert loaded_meta.task_name == "nanozymes"
        assert loaded_meta.metrics["f1"] == 0.85
        assert loaded_meta.instruction_hash == "abc123def456"


class TestEvaluationIntegration:
    """Integration tests for evaluation components."""

    def test_task_metric_integration(self):
        """Test TaskMetric with real experiments."""
        task_config = {
            "compare_fields": ["formula", "activity", "length"],
        }
        
        metric = TaskMetric(task_config=task_config, float_tolerance=0.10)
        
        # Create mock DSPy objects
        import dspy
        
        pred_experiments = [
            NanozymeExperiment(formula="Fe3O4", activity="peroxidase", length=10.0),
            NanozymeExperiment(formula="CuO", activity="oxidase", length=20.0),
        ]
        
        gt_experiments = [
            NanozymeExperiment(formula="Fe3O4", activity="peroxidase", length=10.0),
            NanozymeExperiment(formula="ZnO", activity="catalase", length=15.0),
        ]
        
        # Create DSPy Example and Prediction
        example = dspy.Example(
            extracted_data=NanozymeExtractionOutput(experiments=gt_experiments)
        ).with_inputs("document_text")
        
        prediction = dspy.Prediction(
            extracted_data=NanozymeExtractionOutput(experiments=pred_experiments)
        )
        
        # Calculate metric
        score = metric(example, prediction)
        
        # Verify score is in valid range
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_experiment_matcher_report(self):
        """Test ExperimentMatcher detailed report."""
        matcher = ExperimentMatcher(
            fields_to_compare=["formula", "activity", "length"],
            float_tolerance=0.05,
        )
        
        preds = [
            NanozymeExperiment(formula="Fe3O4", activity="peroxidase", length=10.0),
        ]
        gts = [
            NanozymeExperiment(formula="Fe3O4", activity="peroxidase", length=10.0),
            NanozymeExperiment(formula="CuO", activity="oxidase", length=20.0),
        ]
        
        report = matcher.get_detailed_report(preds, gts)
        
        # Verify report structure
        assert "f1" in report
        assert "precision" in report
        assert "recall" in report
        assert "fields" in report
        assert "counts" in report
        
        # Verify field-level scores
        assert "formula" in report["fields"]
        assert "activity" in report["fields"]
        assert "length" in report["fields"]
