"""Integration tests for Gemini PDF parser.

Tests cover:
- Integration with ParseDocumentsUseCase
- Configuration loading from YAML
- End-to-end parsing flow with mocked Gemini client
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ae import Settings
from ae.core.config import GeminiParserConfig
from ae.core.storage import DocumentRepository
from ae.ingestion.parsers import GeminiParser, get_parser
from ae.ingestion.pipeline import (
    ParseDocumentsRequest,
    ParseDocumentsUseCase,
)


@pytest.mark.integration
class TestGeminiParserIntegration:
    """Integration tests for GeminiParser."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_gemini_parser_with_document_repository(
        self,
        mock_client_class,
        tmp_path: Path,
    ):
        """Test Gemini parser saving to document repository."""
        # Setup paths
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()

        # Create mock PDF
        pdf_path = tmp_path / "test_paper.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")

        # Setup Gemini mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "ACTIVE"
        mock_uploaded_file.name = "files/test-file"
        mock_client.files.upload.return_value = mock_uploaded_file

        mock_chunk = MagicMock()
        mock_chunk.text = "# Test Paper\n\nAbstract: Test content."
        mock_client.models.generate_content_stream.return_value = [mock_chunk]

        # Create parser and repository
        config = GeminiParserConfig()
        parser = GeminiParser(config)
        repo = DocumentRepository(parsed_dir=parsed_dir)

        # Parse and save
        markdown = parser.parse(pdf_path)
        output_path = parsed_dir / "test_paper.md"
        repo.save(markdown, output_path)

        # Verify saved content
        assert output_path.exists()
        saved_content = output_path.read_text(encoding="utf-8")
        assert saved_content == "# Test Paper\n\nAbstract: Test content."

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_documents_use_case_with_gemini(
        self,
        mock_client_class,
        tmp_path: Path,
    ):
        """Test ParseDocumentsUseCase with Gemini parser."""
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

        # Setup Gemini mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "ACTIVE"
        mock_uploaded_file.name = "files/test-file"
        mock_client.files.upload.return_value = mock_uploaded_file

        # Different content for each file
        call_count = [0]

        def get_chunk():
            call_count[0] += 1
            chunk = MagicMock()
            chunk.text = f"# Paper {call_count[0]}\n\nContent {call_count[0]}."
            return [chunk]

        mock_client.models.generate_content_stream.side_effect = lambda **kwargs: (
            get_chunk()
        )

        # Create use case
        doc_repo = DocumentRepository(parsed_dir=parsed_dir)
        use_case = ParseDocumentsUseCase(document_repo=doc_repo)

        # Create request
        request = ParseDocumentsRequest(
            input_paths=[pdf1, pdf2],
            output_dir=parsed_dir,
            parser_name="gemini",
            parser_config=GeminiParserConfig(request_delay=0.0),
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
class TestGeminiConfigLoading:
    """Test configuration loading with Gemini parser."""

    @patch.dict(
        os.environ,
        {
            "OLLAMA_STUDENT_BASE_URL": "http://localhost:11434",
            "OLLAMA_TEACHER_BASE_URL": "http://localhost:11434",
        },
    )
    def test_load_gemini_config_from_yaml(self, tmp_path: Path):
        """Test loading Gemini config from YAML file."""
        # Create required directories and files
        (tmp_path / "data").mkdir()
        
        task_dir = tmp_path / "config" / "tasks" / "test"
        task_dir.mkdir(parents=True, exist_ok=True)
        instruction_file = task_dir / "initial_instruction.txt"
        instruction_file.write_text("test instruction")
        (task_dir / "initial_schema.yaml").write_text("name: test\ncompare_fields:\n  - formula\nfloat_tolerance: 0.05\nfields:\n  formula:\n    type: str\n    description: inorganic formula\n")

        # Create minimal YAML config
        config_path = tmp_path / "gemini_test.yaml"
        config_path.write_text(
            f"""
project:
  log_level: "INFO"

paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"

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
  visual:
    enabled: false
  overwrite: false
  gemini:
    model_name: "gemini-2.0-flash"
    upload_timeout: 600
    safety_settings: true

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
        assert settings.parsing.visual.enabled is False
        assert settings.parsing.gemini is not None
        assert settings.parsing.gemini.model_name == "gemini-2.0-flash"
        assert settings.parsing.gemini.upload_timeout == 600
        assert settings.parsing.gemini.safety_settings is True

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
        instruction_file = task_dir / "initial_instruction.txt"
        instruction_file.write_text("test instruction")
        (task_dir / "initial_schema.yaml").write_text("name: test\ncompare_fields:\n  - formula\nfloat_tolerance: 0.05\nfields:\n  formula:\n    type: str\n    description: inorganic formula\n")

        # Create YAML config (same as above)
        config_path = tmp_path / "gemini_test.yaml"
        config_path.write_text(
            f"""
project:
  log_level: "INFO"

paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"

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
  visual:
    enabled: false
  overwrite: false
  gemini:
    model_name: "gemini-2.0-flash"
    upload_timeout: 600
    safety_settings: true

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
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            parser_name = "gemini_visual" if settings.parsing.visual.enabled else "gemini"
            parser = get_parser(parser_name, settings.parsing.gemini)

        assert isinstance(parser, GeminiParser)
        assert parser.cfg.model_name == "gemini-2.0-flash"
