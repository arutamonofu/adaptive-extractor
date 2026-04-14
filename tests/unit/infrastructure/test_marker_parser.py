"""Unit tests for Marker PDF parser.

Tests cover:
- MarkerParser initialization with new config
- MarkerParser.parse() method (mocked)
- Configuration loading from marker_config module
- Error handling

Note: These tests are skipped when marker is not importable
(e.g., when using transformers >= 5.0 which removes transformers.onnx).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("marker.converters.pdf")

from aee.infrastructure.config import MarkerConfig
from aee.infrastructure.parsers import MarkerParser, get_parser


@pytest.mark.unit
class TestMarkerParserInitialization:
    """Tests for MarkerParser initialization."""

    def test_init_without_config_raises_error(self):
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="Configuration object is required"):
            MarkerParser(None)

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_init_with_valid_config(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
    ):
        """Test successful initialization with valid config."""
        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser

        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter

        config = MarkerConfig()
        parser = MarkerParser(config)

        assert parser.cfg == config
        assert parser.converter is not None

        # Verify PdfConverter was called with correct arguments
        mock_converter_class.assert_called_once()
        call_kwargs = mock_converter_class.call_args[1]
        assert "artifact_dict" in call_kwargs
        assert "config" in call_kwargs
        assert "processor_list" in call_kwargs
        assert "renderer" in call_kwargs
        assert "llm_service" in call_kwargs

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_init_uses_marker_config_module(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
    ):
        """Test that initialization uses settings from marker_config module."""
        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser
        mock_converter_class.return_value = MagicMock()

        config = MarkerConfig()
        MarkerParser(config)

        # Verify ConfigParser was called (meaning config_dict was generated)
        mock_config_parser_class.assert_called_once()

        # Verify the config dict contains expected settings
        config_dict_arg = mock_config_parser_class.call_args[0][0]
        assert config_dict_arg.get("use_llm") is True
        assert config_dict_arg.get("force_ocr") is True
        assert config_dict_arg.get("ollama_model") == "qwen2.5vl:72b"

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_init_uses_cuda_device(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
    ):
        """Test that device is set correctly based on CUDA availability."""
        import torch
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"

        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser
        mock_converter_class.return_value = MagicMock()

        config = MarkerConfig()
        MarkerParser(config)

        # Verify create_model_dict was called with correct device
        mock_model_dict.assert_called_once_with(device=expected_device)

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_init_uses_custom_processors(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
    ):
        """Test that custom processor list is used."""
        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser
        mock_converter_class.return_value = MagicMock()

        config = MarkerConfig()
        MarkerParser(config)

        # Verify processor_list was passed
        call_kwargs = mock_converter_class.call_args[1]
        processor_list = call_kwargs.get("processor_list")
        assert processor_list is not None
        assert len(processor_list) > 0

        # Verify LLM processors are included
        processor_strings = [str(p) for p in processor_list]
        assert any("LLMTableProcessor" in p for p in processor_strings)
        assert any("LLMEquationProcessor" in p for p in processor_strings)


@pytest.mark.unit
class TestMarkerParserParse:
    """Tests for MarkerParser.parse() method."""

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_parse_success(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
        tmp_path: Path,
    ):
        """Test successful PDF parsing."""
        # Create mock PDF file
        pdf_path = tmp_path / "test.pdf"
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
        mock_result.markdown = "# Test Markdown\n\nContent here."
        mock_converter.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        # Create parser and parse
        config = MarkerConfig()
        parser = MarkerParser(config)
        result = parser.parse(pdf_path)

        # Verify result
        assert result == "# Test Markdown\n\nContent here."

        # Verify converter was called
        mock_converter.assert_called_once_with(str(pdf_path))

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_parse_with_text_fallback(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
        tmp_path: Path,
    ):
        """Test parsing with fallback to text attribute."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")

        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser

        mock_converter = MagicMock()
        mock_result = MagicMock()
        # No markdown attribute, only text
        del mock_result.markdown
        mock_result.text = "# Text Fallback"
        mock_converter.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        config = MarkerConfig()
        parser = MarkerParser(config)
        result = parser.parse(pdf_path)

        assert result == "# Text Fallback"

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_parse_with_str_fallback(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
        tmp_path: Path,
    ):
        """Test parsing with fallback to str() conversion."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")

        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser

        mock_converter = MagicMock()
        mock_result = MagicMock()
        # No markdown or text attribute
        del mock_result.markdown
        del mock_result.text
        mock_result.__str__ = lambda self: "String fallback"
        mock_converter.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        config = MarkerConfig()
        parser = MarkerParser(config)
        result = parser.parse(pdf_path)

        assert result == "String fallback"

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_parse_error_raises_exception(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
        tmp_path: Path,
    ):
        """Test that parsing errors are raised."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")

        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser

        mock_converter = MagicMock()
        mock_converter.side_effect = Exception("Conversion failed")
        mock_converter_class.return_value = mock_converter

        config = MarkerConfig()
        parser = MarkerParser(config)

        with pytest.raises(Exception, match="Conversion failed"):
            parser.parse(pdf_path)


@pytest.mark.unit
class TestGetParserMarker:
    """Tests for get_parser() factory function with Marker."""

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_get_parser_marker(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
    ):
        """Test getting Marker parser."""
        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser
        mock_converter_class.return_value = MagicMock()

        config = MarkerConfig()
        parser = get_parser("marker", config)

        assert isinstance(parser, MarkerParser)

    @patch("marker.models.create_model_dict")
    @patch("marker.config.parser.ConfigParser")
    @patch("marker.converters.pdf.PdfConverter")
    def test_get_parser_marker_case_insensitive(
        self,
        mock_converter_class,
        mock_config_parser_class,
        mock_model_dict,
    ):
        """Test that parser name is case-insensitive."""
        mock_model_dict.return_value = {}
        mock_config_parser = MagicMock()
        mock_config_parser.generate_config_dict.return_value = {}
        mock_config_parser.get_renderer.return_value = MagicMock()
        mock_config_parser.get_llm_service.return_value = MagicMock()
        mock_config_parser_class.return_value = mock_config_parser
        mock_converter_class.return_value = MagicMock()

        config = MarkerConfig()

        parser1 = get_parser("MARKER", config)
        parser2 = get_parser("Marker", config)

        assert isinstance(parser1, MarkerParser)
        assert isinstance(parser2, MarkerParser)

    def test_get_parser_marker_without_config_raises_error(self):
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="requires MarkerConfig"):
            get_parser("marker", None)

    def test_get_parser_marker_with_wrong_config_raises_error(self):
        """Test that wrong config type raises ValueError."""
        from aee.infrastructure.config import GeminiParserConfig

        gemini_config = GeminiParserConfig()

        with pytest.raises(ValueError, match="requires MarkerConfig"):
            get_parser("marker", gemini_config)
