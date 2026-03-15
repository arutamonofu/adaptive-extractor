"""Integration tests for Marker PDF parser.

Tests cover:
- Integration with ParseDocumentsUseCase
- Configuration loading from YAML
- End-to-end parsing flow with mocked PdfConverter
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aee.application.use_cases.parse_documents import (
    ParseDocumentsRequest,
    ParseDocumentsUseCase,
)
from aee.infrastructure.config.settings import MarkerConfig, Settings
from aee.infrastructure.parsers import MarkerParser, get_parser
from aee.infrastructure.storage import DocumentRepository


@pytest.mark.integration
class TestMarkerParserIntegration:
    """Integration tests for MarkerParser."""

    @patch("aee.infrastructure.parsers.parsers.create_model_dict")
    @patch("aee.infrastructure.parsers.parsers.ConfigParser")
    @patch("aee.infrastructure.parsers.parsers.PdfConverter")
    def test_marker_parser_with_document_repository(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
        tmp_path: Path,
    ):
        """Test Marker parser saving to document repository."""
        # Setup paths
        parsed_dir = tmp_path / "parsed"
        parsed_dir.mkdir()

        # Create mock PDF
        pdf_path = tmp_path / "test_paper.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")

        # Setup mocks
        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser

        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.markdown = "# Test Paper\n\nAbstract: Test content."
        mock_converter.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        # Create parser and repository
        config = MarkerConfig()
        parser = MarkerParser(config)
        repo = DocumentRepository(parsed_dir=parsed_dir)

        # Parse and save
        markdown = parser.parse(pdf_path)
        output_path = parsed_dir / "test_paper.md"
        repo.save(markdown, output_path)

        # Verify saved content
        assert output_path.exists()
        saved_content = output_path.read_text(encoding="utf-8")
        assert saved_content == "# Test Paper\n\nAbstract: Test content."

    @patch("aee.infrastructure.parsers.parsers.create_model_dict")
    @patch("aee.infrastructure.parsers.parsers.ConfigParser")
    @patch("aee.infrastructure.parsers.parsers.PdfConverter")
    def test_parse_documents_use_case_with_marker(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
        tmp_path: Path,
    ):
        """Test ParseDocumentsUseCase with Marker parser."""
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

        # Setup mocks
        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser

        mock_converter = MagicMock()
        # Different content for each file
        call_count = [0]

        def convert_side_effect(path):
            call_count[0] += 1
            mock_result = MagicMock()
            mock_result.markdown = f"# Paper {call_count[0]}\n\nContent {call_count[0]}."
            return mock_result

        mock_converter.side_effect = convert_side_effect
        mock_converter_class.return_value = mock_converter

        # Create use case
        doc_repo = DocumentRepository(parsed_dir=parsed_dir)
        use_case = ParseDocumentsUseCase(document_repo=doc_repo)

        # Create request
        request = ParseDocumentsRequest(
            input_paths=[pdf1, pdf2],
            output_dir=parsed_dir,
            parser_name="marker",
            parser_config=MarkerConfig(),
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

        # Verify content
        content1 = (parsed_dir / "paper1.md").read_text(encoding="utf-8")
        content2 = (parsed_dir / "paper2.md").read_text(encoding="utf-8")
        assert content1 == "# Paper 1\n\nContent 1."
        assert content2 == "# Paper 2\n\nContent 2."


@pytest.mark.integration
class TestMarkerConfigLoading:
    """Test configuration loading with Marker parser."""

    def test_load_marker_config_from_yaml(self, tmp_path: Path):
        """Test loading Marker config from YAML file (minimal config)."""
        # Create required directories and files
        (tmp_path / "data").mkdir()
        (tmp_path / "config" / "initial_instructions").mkdir(parents=True)
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.write_text("test instruction")

        # Create minimal YAML config (no marker section)
        config_path = tmp_path / "marker_test.yaml"
        config_path.write_text(f"""
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
    use_ollama: true
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 512

  teacher:
    use_ollama: true
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 512

parsing:
  parser: "marker"
  overwrite: false
  # Note: marker section is now optional, settings are in marker_config.py

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
  initial_instruction_file: "{instruction_file}"
  evaluation:
    compare_fields:
      - "formula"
    float_tolerance: 0.05

extraction:
  enable_cache: false

cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100

circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""", encoding="utf-8")

        # Load settings
        settings = Settings.load(config_path=config_path, load_env_file=False)

        # Verify parsing config
        assert settings.parsing.parser == "marker"
        # Marker config is auto-created (empty but valid)
        assert settings.parsing.marker is not None

    def test_get_parser_from_settings(self, tmp_path: Path):
        """Test getting parser instance from loaded settings."""
        # Create required directories and files
        (tmp_path / "data").mkdir()
        (tmp_path / "config" / "initial_instructions").mkdir(parents=True)
        instruction_file = tmp_path / "config" / "initial_instructions" / "test.txt"
        instruction_file.write_text("test instruction")

        # Create YAML config (same as above)
        config_path = tmp_path / "marker_test.yaml"
        config_path.write_text(f"""
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
    use_ollama: true
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.0
    rate_limit_delay: 1.0
    top_p: 0.1
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 512

  teacher:
    use_ollama: true
    model: "test-model"
    timeout: 60
    max_retries: 3
    temperature: 0.5
    rate_limit_delay: 1.0
    top_p: 0.9
    repeat_penalty: 1.0
    repeat_last_n: 64
    enable_cache: true
    ollama:
      num_ctx: 1024
      num_predict: 512
      repeat_penalty: 1.0
      repeat_last_n: 64
      stream: false
    non_ollama:
      max_tokens: 512

parsing:
  parser: "marker"
  overwrite: false

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
  initial_instruction_file: "{instruction_file}"
  evaluation:
    compare_fields:
      - "formula"
    float_tolerance: 0.05

extraction:
  enable_cache: false

cache:
  disk_size_limit_bytes: 1000000
  memory_max_entries: 100

circuit_breaker:
  failure_threshold: 5
  reset_timeout: 30.0
  half_open_max_calls: 1
""", encoding="utf-8")

        # Load settings
        settings = Settings.load(config_path=config_path, load_env_file=False)

        # Get parser (mock the actual Marker initialization)
        with patch("aee.infrastructure.parsers.parsers.create_model_dict"):
            with patch("aee.infrastructure.parsers.parsers.ConfigParser"):
                with patch("aee.infrastructure.parsers.parsers.PdfConverter"):
                    parser = get_parser(settings.parsing.parser, settings.parsing.marker)

        assert isinstance(parser, MarkerParser)
