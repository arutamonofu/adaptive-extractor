from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pytest
import yaml

from ae.core.schema import FieldSpec, SchemaConfig
from ae.extraction.cli import extract_command
from ae.extraction.use_case import BatchExtractionRequest
from ae.optimization.cli import optimize_command
from ae.ingestion.cli import parse_command


def write_minimal_config(config_dir: Path, parsed_dir: Path, instruction_file: Path, task_name="nanozymes", pdf_dir=None):
    schema_yaml = config_dir.parent / "schema.yaml"
    schema_yaml.write_text(
        "name: " + task_name + "\n"
        "compare_fields:\n"
        "  - field1\n"
        "float_tolerance: 0.1\n"
        "fields:\n"
        "  field1:\n"
        "    type: str\n"
        "    description: Test field\n",
        encoding="utf-8"
    )

    gt_file = config_dir.parent / "ground_truth.csv"
    gt_file.write_text("filename,field1\npaper1.pdf,value1\n", encoding="utf-8")

    config_data = {
        "project": {"log_level": "INFO"},
        "paths": {
            "pdf_dir": str(pdf_dir) if pdf_dir else "data/pdf",
            "ingestion_dir": str(parsed_dir),
            "ground_truth_file": str(gt_file),
            "schema_file": str(schema_yaml),
            "baseline_prompt_file": str(instruction_file),
            "generated_prompt_file": str(config_dir.parent / "generated_instruction.txt"),
            "chart_prompt_file": str(config_dir.parent / "chart_instruction.txt"),
            "splits_file": "data/splits.json",
            "agents_dir": "data/processed/agents",
            "extracted_dir": "data/extractions",
        },
        "llm": {
            "student": {
                "provider": "ollama",
                "model": "test-model",
                "timeout": 60,
                "max_retries": 1,
                "temperature": 0.0,
                "rate_limit_delay": 0.0,
                "top_p": 0.1,
                "enable_cache": False,
                "ollama": {
                    "num_ctx": 1024,
                    "num_predict": 256,
                    "repeat_penalty": 1.0,
                    "repeat_last_n": 64,
                    "stream": False,
                },
                "api": {"max_tokens": 256},
            },
            "teacher": {
                "provider": "ollama",
                "model": "test-model",
                "timeout": 60,
                "max_retries": 1,
                "temperature": 0.5,
                "rate_limit_delay": 0.0,
                "top_p": 0.9,
                "enable_cache": False,
                "ollama": {
                    "num_ctx": 1024,
                    "num_predict": 256,
                    "repeat_penalty": 1.0,
                    "repeat_last_n": 64,
                    "stream": False,
                },
                "api": {"max_tokens": 256},
            },
        },
        "parsing": {
            "visual": {"enabled": False},
            "overwrite": False,
        },
        "optimization": {
            "num_trials": 1,
            "train_split": 5,
            "total_load": 10,
            "random_seed": 42,
            "num_candidates": 2,
            "max_bootstrapped_demos": 1,
            "max_labeled_demos": 1,
            "minibatch": False,
            "minibatch_size": 1,
            "view_data_batch_size": 1,
            "metric_threshold": 0.5,
            "init_temperature": 0.5,
            "verbose": False,
            "use_cache": False,
            "max_errors": 5,
        },
        "extraction": {"enable_cache": False},
        "cache": {
            "disk_size_limit_bytes": 1000000,
            "memory_max_entries": 100,
        },
        "circuit_breaker": {
            "failure_threshold": 3,
            "reset_timeout": 30.0,
            "half_open_max_calls": 1,
        },
    }
    config_file = config_dir.parent / "temp_config.yaml"
    config_file.write_text(yaml.dump(config_data), encoding="utf-8")
    from tests.conftest import _split_config
    _split_config(config_file, config_dir)


@pytest.mark.unit
class TestExtractCLI:
    """In-process unit tests for extract CLI command."""

    def test_argument_parsing(self):
        # Missing agent (which is required)
        with pytest.raises(SystemExit) as exc:
            extract_command([])
        assert exc.value.code != 0

        # Help works
        with pytest.raises(SystemExit) as exc:
            extract_command(["--help"])
        assert exc.value.code == 0

    def test_llm_setup_and_agent_not_found(self, tmp_path: Path):
        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Extract nanozymes")
        config_dir = tmp_path / "config"
        write_minimal_config(config_dir, tmp_path / "parsed", instruction)

        with patch("ae.core.llm.provider.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "test"}}]}
            result = extract_command([
                "--config", str(config_dir),
                "--agent", str(tmp_path / "nonexistent_agent.json"),
            ])
            # Fails with 1 because agent is not found
            assert result == 1

    def test_batch_prediction_request_includes_task_dict(self):
        task_config = SchemaConfig(
            name="test",
            experiment_fields={"field1": FieldSpec(type=str, description="Test field")},
            compare_fields=["field1"],
            float_tolerance=0.1,
        )
        task_dict = {
            "config": task_config,
            "signature": MagicMock(),
            "output_model": MagicMock(),
            "row_converter": MagicMock(),
        }
        request = BatchExtractionRequest(
            task=task_config,
            task_dict=task_dict,
            agent_path=Path("test.json"),
            document_ids=["doc1"],
            output_dir=Path("output"),
        )
        assert request.task_dict == task_dict

    def test_invalid_task_signature_error(self, tmp_path: Path):
        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        write_minimal_config(config_dir, tmp_path / "parsed", instruction, task_name="invalid_task")

        # Overwrite schema.yaml to make it invalid (missing compare_fields and fields)
        schema_yaml = config_dir.parent / "schema.yaml"
        schema_yaml.write_text("name: invalid_task\n")

        agent_file = tmp_path / "agent.json"
        agent_file.write_text('{"lm": "test"}')
        agent_file.with_suffix(".meta.json").write_text(
            '{"created_at": "2026-01-01T00:00:00", "model_version": "test", "metrics": {}, "config_snapshot": {}}'
        )

        result = extract_command(["--config", str(config_dir), "--agent", str(agent_file)])
        assert result == 1

    def test_missing_parsed_dir_returns_zero(self, tmp_path: Path):
        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        non_existent_parsed = tmp_path / "nonexistent_parsed"
        write_minimal_config(config_dir, non_existent_parsed, instruction)

        agent_file = tmp_path / "agent.json"
        agent_file.write_text('{"lm": "test"}')
        agent_file.with_suffix(".meta.json").write_text(
            '{"task_name": "nanozymes", "created_at": "2026-01-01T00:00:00", "model_version": "test", "metrics": {}, "config_snapshot": {}}'  # noqa: E501
        )

        with patch("ae.core.llm.setup_student") as mock_setup:
            mock_lm = MagicMock(spec=dspy.LM)
            mock_lm.model = "test-model"
            mock_setup.return_value = mock_lm

            result = extract_command(["--config", str(config_dir), "--agent", str(agent_file)])
            # Should exit with 0 since there are no parsed documents to process
            assert result == 0

    def test_successful_extraction_flow(self, tmp_path: Path, monkeypatch):
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        doc_path = parsed_dir / "paper1.md"
        doc_path.write_text("# Fe3O4\nNanoparticles with peroxidase activity size 10.5 nm.")

        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        write_minimal_config(config_dir, parsed_dir, instruction)

        agent_file = tmp_path / "agent.json"
        agent_file.write_text('{"lm": "test"}')
        agent_file.with_suffix(".meta.json").write_text(
            '{"task_name": "nanozymes", "created_at": "2026-01-01T00:00:00", "model_version": "test", "metrics": {}, "config_snapshot": {}}'  # noqa: E501
        )

        # Mocks
        mock_lm = MagicMock(spec=dspy.LM)
        mock_lm.model = "test-model"
        monkeypatch.setattr("ae.core.llm.setup_student", lambda *args, **kwargs: mock_lm)

        monkeypatch.setattr(
            "ae.extraction.manager.AgentManager.load_agent_as_object",
            lambda self, agent_path, task: MagicMock()
        )

        mock_response = MagicMock()
        mock_response.success = True
        mock_response.extractions_saved = 1
        mock_response.total_documents = 1
        mock_response.failed_documents = 0
        mock_response.output_dir = tmp_path / "extractions"
        monkeypatch.setattr(
            "ae.extraction.use_case.BatchExtractionUseCase.execute",
            lambda self, request: mock_response
        )

        result = extract_command(["--config", str(config_dir), "--agent", str(agent_file)])
        assert result == 0

    def test_partial_success_extraction_flow(self, tmp_path: Path, monkeypatch):
        # Setup documents and config
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        (parsed_dir / "doc1.md").write_text("Text 1")
        (parsed_dir / "doc2.md").write_text("Text 2")

        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        write_minimal_config(config_dir, parsed_dir, instruction)

        agent_file = tmp_path / "agent.json"
        agent_file.write_text('{"lm": "test"}')
        agent_file.with_suffix(".meta.json").write_text(
            '{"task_name": "nanozymes", "created_at": "2026-01-01T00:00:00", "model_version": "test", "metrics": {}, "config_snapshot": {}}'
        )

        # Mocks
        mock_lm = MagicMock(spec=dspy.LM)
        mock_lm.model = "test-model"
        monkeypatch.setattr("ae.core.llm.setup_student", lambda *args, **kwargs: mock_lm)

        monkeypatch.setattr(
            "ae.extraction.manager.AgentManager.load_agent_as_object",
            lambda self, agent_path, task: MagicMock()
        )

        mock_response = MagicMock()
        mock_response.success = True
        mock_response.extractions_saved = 1
        mock_response.total_documents = 2
        mock_response.failed_documents = 1
        mock_response.output_dir = tmp_path / "extractions"
        monkeypatch.setattr(
            "ae.extraction.use_case.BatchExtractionUseCase.execute",
            lambda self, request: mock_response
        )

        result = extract_command(["--config", str(config_dir), "--agent", str(agent_file)])
        assert result == 2

    def test_aborted_extraction_flow(self, tmp_path: Path, monkeypatch):
        # Setup documents and config
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        (parsed_dir / "doc1.md").write_text("Text 1")
        (parsed_dir / "doc2.md").write_text("Text 2")

        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        write_minimal_config(config_dir, parsed_dir, instruction)

        agent_file = tmp_path / "agent.json"
        agent_file.write_text('{"lm": "test"}')
        agent_file.with_suffix(".meta.json").write_text(
            '{"task_name": "nanozymes", "created_at": "2026-01-01T00:00:00", "model_version": "test", "metrics": {}, "config_snapshot": {}}'
        )

        # Mocks
        mock_lm = MagicMock(spec=dspy.LM)
        mock_lm.model = "test-model"
        monkeypatch.setattr("ae.core.llm.setup_student", lambda *args, **kwargs: mock_lm)

        monkeypatch.setattr(
            "ae.extraction.manager.AgentManager.load_agent_as_object",
            lambda self, agent_path, task: MagicMock()
        )

        mock_response = MagicMock()
        mock_response.success = False
        mock_response.aborted = True
        mock_response.extractions_saved = 1
        mock_response.total_documents = 2
        mock_response.failed_documents = 1
        mock_response.error_message = "Circuit breaker open"
        monkeypatch.setattr(
            "ae.extraction.use_case.BatchExtractionUseCase.execute",
            lambda self, request: mock_response
        )

        result = extract_command(["--config", str(config_dir), "--agent", str(agent_file)])
        assert result == 2


@pytest.mark.unit
class TestOptimizeCLI:
    """In-process unit tests for optimize CLI command."""

    def test_argument_parsing(self):
        # Help works
        with pytest.raises(SystemExit) as exc:
            optimize_command(["--help"])
        assert exc.value.code == 0

    def test_config_file_not_found_or_invalid(self, tmp_path: Path):
        # Not found
        result = optimize_command(["--config", str(tmp_path / "nonexistent")])
        assert result == 1

        # Invalid (empty/nonexistent dir)
        result2 = optimize_command(["--config", str(tmp_path / "invalid")])
        assert result2 == 1

    def test_invalid_task_signature_error(self, tmp_path: Path):
        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        write_minimal_config(config_dir, tmp_path / "parsed", instruction, task_name="invalid_task")

        result = optimize_command(["--config", str(config_dir)])
        assert result == 1

    def test_missing_parsed_dir_error(self, tmp_path: Path):
        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        write_minimal_config(config_dir, tmp_path / "nonexistent_parsed", instruction)

        result = optimize_command(["--config", str(config_dir)])
        assert result == 1


@pytest.mark.unit
class TestIngestCLI:
    """In-process unit tests for ingest CLI command."""

    def test_argument_parsing(self):
        with pytest.raises(SystemExit) as exc:
            parse_command(["--help"])
        assert exc.value.code == 0

    def test_aborted_ingestion_flow(self, tmp_path: Path, monkeypatch):
        instruction = tmp_path / "instruction.txt"
        instruction.write_text("Test instruction")
        config_dir = tmp_path / "config"
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()
        
        # Create dummy PDF files in resolved pdf_dir
        pdf_dir = config_dir / "data" / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        (pdf_dir / "paper1.pdf").write_bytes(b"%PDF-1.4")
        (pdf_dir / "paper2.pdf").write_bytes(b"%PDF-1.4")

        write_minimal_config(config_dir, parsed_dir, instruction, pdf_dir=pdf_dir)

        # Mock the execution of the ingestion use case
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.aborted = True
        mock_response.documents_parsed = 1
        mock_response.documents_skipped = 0
        mock_response.total_documents = 2
        mock_response.failed_documents = 1
        mock_response.error_message = "Circuit breaker open"
        
        monkeypatch.setattr(
            "ae.ingestion.use_case.ParseDocumentsUseCase.execute",
            lambda self, request: mock_response
        )

        result = parse_command(["--config", str(config_dir)])
        assert result == 2
