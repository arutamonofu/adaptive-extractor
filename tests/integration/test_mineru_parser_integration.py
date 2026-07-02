import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ae import Settings
from ae.core.config import IngestionConfig, MinerUParserConfig, ChartExtractionConfig
from ae.core.storage import DocumentRepository
from ae.ingestion.parsers.mineru.parser import MinerUParser
from ae.ingestion.parsers import get_parser
from ae.ingestion.use_case import (
    ParseDocumentsRequest,
    ParseDocumentsUseCase,
)


@pytest.mark.integration
class TestMinerUParserIntegration:
    """Integration tests for MinerUParser."""

    @patch.dict(os.environ, {"MINERU_API_TOKEN": "test_token"})
    @patch("ae.ingestion.parsers.mineru.parser.MinerUClient")
    @patch("ae.ingestion.parsers.mineru.parser.get_model_client")
    @patch("ae.ingestion.parsers.mineru.parser.extract_single_chart")
    def test_mineru_parser_with_document_repository(
        self,
        mock_extract_single_chart,
        mock_get_model_client,
        mock_client_class,
        tmp_path: Path,
    ):
        """Test MinerU parser saving to document repository."""
        # Setup paths
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()

        # Create mock PDF
        pdf_path = tmp_path / "test_paper.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")

        # Setup MinerU mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Setup side effect for parse_pdf: write output files to mock output dir
        def mock_parse_pdf(pdf, out_dir):
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "full.md").write_text("# Test Paper\n\nAbstract: Test content.", encoding="utf-8")
            (out_path / "content_list.json").write_text("[]", encoding="utf-8")
            return {"state": "done"}

        mock_client.parse_pdf.side_effect = mock_parse_pdf

        # Create parser and repository
        config = IngestionConfig(
            overwrite=True,
            mineru=MinerUParserConfig(
                api_url="https://mineru.net/api/v4",
                poll_interval=1,
                poll_timeout=10
            ),
            chart_extraction=ChartExtractionConfig(
                enabled=False
            )
        )
        parser = MinerUParser(config)
        repo = DocumentRepository(ingestion_dir=parsed_dir)

        # Parse and save
        markdown = parser.parse(pdf_path)
        output_path = parsed_dir / "test_paper.md"
        repo.save(markdown, output_path)

        # Verify saved content
        assert output_path.exists()
        saved_content = output_path.read_text(encoding="utf-8")
        assert saved_content == "# Test Paper\n\nAbstract: Test content."

    @patch.dict(os.environ, {"MINERU_API_TOKEN": "test_token"})
    @patch("ae.ingestion.parsers.mineru.parser.MinerUClient")
    @patch("ae.ingestion.parsers.mineru.parser.get_model_client")
    @patch("ae.ingestion.parsers.mineru.parser.extract_single_chart")
    def test_parse_documents_use_case_with_mineru(
        self,
        mock_extract_single_chart,
        mock_get_model_client,
        mock_client_class,
        tmp_path: Path,
    ):
        """Test ParseDocumentsUseCase with MinerU parser."""
        # Setup paths
        pdf_dir = tmp_path / "pdf"
        pdf_dir.mkdir()
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()

        # Create mock PDFs
        pdf1 = pdf_dir / "paper1.pdf"
        pdf1.write_bytes(b"%PDF-content-1")
        pdf2 = pdf_dir / "paper2.pdf"
        pdf2.write_bytes(b"%PDF-content-2")

        # Setup MinerU mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Different content for each file
        call_count = [0]

        def mock_parse_pdf(pdf, out_dir):
            call_count[0] += 1
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "full.md").write_text(f"# Paper {call_count[0]}\n\nContent {call_count[0]}.", encoding="utf-8")
            (out_path / "content_list.json").write_text("[]", encoding="utf-8")
            return {"state": "done"}

        mock_client.parse_pdf.side_effect = mock_parse_pdf

        # Create use case
        doc_repo = DocumentRepository(ingestion_dir=parsed_dir)
        use_case = ParseDocumentsUseCase(document_repo=doc_repo)

        config = IngestionConfig(
            overwrite=True,
            mineru=MinerUParserConfig(
                api_url="https://mineru.net/api/v4",
                poll_interval=1,
                poll_timeout=10
            ),
            chart_extraction=ChartExtractionConfig(
                enabled=True
            )
        )

        # Create request
        request = ParseDocumentsRequest(
            input_paths=[pdf1, pdf2],
            output_dir=parsed_dir,
            parser_name="mineru",
            parser_config=config,
            overwrite=True,
        )

        # Execute
        response = use_case.execute(request)

        # Verify results
        assert response.success is True
        assert response.documents_parsed == 2
        assert response.total_documents == 2
        assert response.failed_documents == 0

        # Verify files created
        assert (parsed_dir / "paper1.md").exists()
        assert (parsed_dir / "paper2.md").exists()


@pytest.mark.integration
class TestMinerUConfigLoading:
    """Test configuration loading with MinerU parser."""

    @patch.dict(
        os.environ,
        {
            "OLLAMA_STUDENT_BASE_URL": "http://localhost:11434",
            "OLLAMA_TEACHER_BASE_URL": "http://localhost:11434",
        },
    )
    def test_load_mineru_config_from_yaml(self, tmp_path: Path):
        """Test loading MinerU config from YAML file."""
        # Create required directories and files
        (tmp_path / "data").mkdir()
        
        task_dir = tmp_path / "config" / "tasks" / "test"
        task_dir.mkdir(parents=True, exist_ok=True)
        instruction_file = task_dir / "generated_instruction.txt"
        instruction_file.write_text("test instruction")
        (task_dir / "schema.yaml").write_text("name: test\ncompare_fields:\n  - formula\nfloat_tolerance: 0.05\nfields:\n  formula:\n    type: str\n    description: inorganic formula\n")

        # Create minimal YAML config
        config_path = tmp_path / "mineru_test.yaml"
        config_path.write_text(
            f"""
project:
  log_level: "INFO"

paths:
  pdf_dir: "data/pdf"
  ingestion_dir: "data/interim/ingestion"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/processed/agents"
  extracted_dir: "data/extractions"

llm:
  student:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 512

  teacher:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 512

parsing:
  overwrite: false
  concurrency: 4
  mineru:
    api_url: "https://mineru.net/api/v4"
    model_version: "vlm"
    poll_interval: 3
    poll_timeout: 600
  chart_extraction:
    enabled: true

optimization:
  total_load: 3
  train_split: 3
  num_candidates: 3
  num_trials: 3
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 10
  view_data_batch_size: 3
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: true

task:
  name: "test"

extraction:
  enable_cache: false

cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100

circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""",
            encoding="utf-8",
        )

        # Split and load settings from config directory
        config_dir = tmp_path / "config"
        from tests.conftest import _split_config
        _split_config(config_path, config_dir)
        
        settings = Settings.load(config_path=config_dir, load_env_file=False)

        # Verify parsing config
        assert settings.parsing.mineru.api_url == "https://mineru.net/api/v4"
        assert settings.parsing.mineru.poll_timeout == 600
        assert settings.parsing.chart_extraction.enabled is True

    @patch.dict(
        os.environ,
        {
            "OLLAMA_STUDENT_BASE_URL": "http://localhost:11434",
            "OLLAMA_TEACHER_BASE_URL": "http://localhost:11434",
        },
    )
    def test_get_parser_from_settings(self, tmp_path: Path):
        """Test getting parser instance from loaded settings."""
        # Create required directories and files
        (tmp_path / "data").mkdir()
        
        task_dir = tmp_path / "config" / "tasks" / "test"
        task_dir.mkdir(parents=True, exist_ok=True)
        instruction_file = task_dir / "generated_instruction.txt"
        instruction_file.write_text("test instruction")
        (task_dir / "schema.yaml").write_text("name: test\ncompare_fields:\n  - formula\nfloat_tolerance: 0.05\nfields:\n  formula:\n    type: str\n    description: inorganic formula\n")

        # Create YAML config (same as above)
        config_path = tmp_path / "mineru_test.yaml"
        config_path.write_text(
            f"""
project:
  log_level: "INFO"

paths:
  pdf_dir: "data/pdf"
  ingestion_dir: "data/interim/ingestion"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/processed/agents"
  extracted_dir: "data/extractions"

llm:
  student:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 512

  teacher:
    provider: "ollama"
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    api:
      max_tokens: 512

parsing:
  overwrite: false
  concurrency: 4
  mineru:
    api_url: "https://mineru.net/api/v4"
    model_version: "vlm"
    poll_interval: 3
    poll_timeout: 600
  chart_extraction:
    provider: "gemini"
    model: "gemini-3.5-flash"
    temperature: 0.0
    max_output_tokens: 20000
    thinking_level: "high"

optimization:
  total_load: 3
  train_split: 3
  num_candidates: 3
  num_trials: 3
  max_bootstrapped_demos: 1
  max_labeled_demos: 1
  minibatch: false
  minibatch_size: 10
  view_data_batch_size: 3
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: true

task:
  name: "test"

extraction:
  enable_cache: false

cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100

circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""",
            encoding="utf-8",
        )

        # Split and load settings from config directory
        config_dir = tmp_path / "config"
        from tests.conftest import _split_config
        _split_config(config_path, config_dir)
        
        settings = Settings.load(config_path=config_dir, load_env_file=False)

        # Get parser
        with patch.dict(os.environ, {"MINERU_API_TOKEN": "test_token"}):
            with patch("ae.ingestion.parsers.mineru.parser.MinerUClient"):
                parser = get_parser("mineru", settings.parsing)

        assert isinstance(parser, MinerUParser)
        assert parser.cfg.mineru.api_url == "https://mineru.net/api/v4"
