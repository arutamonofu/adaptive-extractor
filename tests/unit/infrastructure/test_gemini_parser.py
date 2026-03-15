"""Unit tests for Gemini PDF parser.

Tests cover:
- GeminiParserConfig validation
- GeminiParser initialization
- GeminiParser.parse() method (mocked)
- get_parser() factory function with gemini parser
- Error handling and cleanup
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aee.infrastructure.config.settings import GeminiParserConfig, IngestionConfig
from aee.infrastructure.parsers import GeminiParser, get_parser
from aee.infrastructure.parsers.parsers import GEMINI_PDF_TO_MD_PROMPT


@pytest.mark.unit
class TestGeminiParserConfig:
    """Tests for GeminiParserConfig."""

    def test_create_config_with_defaults(self):
        """Test creating config with default values."""
        config = GeminiParserConfig()
        
        assert config.model_name == "gemini-3-flash-preview"
        assert config.upload_timeout == 300
        assert config.safety_settings is True

    def test_create_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = GeminiParserConfig(
            model_name="gemini-2.0-flash",
            upload_timeout=600,
            safety_settings=False,
        )
        
        assert config.model_name == "gemini-2.0-flash"
        assert config.upload_timeout == 600
        assert config.safety_settings is False


@pytest.mark.unit
class TestIngestionConfigWithGemini:
    """Tests for IngestionConfig with Gemini parser."""

    def test_ingestion_config_gemini_auto_create(self):
        """Test that Gemini config is auto-created when parser is 'gemini'."""
        config = IngestionConfig(
            parser="gemini",
            overwrite=False,
        )
        
        assert config.gemini is not None
        assert config.gemini.model_name == "gemini-3-flash-preview"

    def test_ingestion_config_gemini_with_explicit_config(self):
        """Test IngestionConfig with explicit Gemini config."""
        config = IngestionConfig(
            parser="gemini",
            overwrite=True,
            gemini=GeminiParserConfig(
                model_name="gemini-2.0-flash",
                upload_timeout=600,
            ),
        )
        
        assert config.gemini.model_name == "gemini-2.0-flash"
        assert config.gemini.upload_timeout == 600

    def test_ingestion_config_marker_still_works(self):
        """Test that Marker parser config still works."""
        from aee.infrastructure.config.settings import MarkerConfig
        
        config = IngestionConfig(
            parser="marker",
            overwrite=False,
            marker=MarkerConfig(device="cpu"),
        )
        
        assert config.parser == "marker"
        assert config.marker.device == "cpu"

    def test_ingestion_config_marker_requires_config(self):
        """Test that Marker parser requires config."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError, match="marker.*configuration is missing"):
            IngestionConfig(
                parser="marker",
                overwrite=False,
            )


@pytest.mark.unit
class TestGeminiParserInitialization:
    """Tests for GeminiParser initialization."""

    def test_init_without_config_raises_error(self):
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="Configuration object is required"):
            GeminiParser(None)

    @patch.dict(os.environ, {}, clear=False)
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        # Ensure API key is not set
        os.environ.pop("GEMINI_API_KEY", None)
        
        config = GeminiParserConfig()
        
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable"):
            GeminiParser(config)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_init_with_valid_config(self, mock_client_class):
        """Test successful initialization with valid config."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        config = GeminiParserConfig()
        
        parser = GeminiParser(config)
        
        assert parser.cfg == config
        assert parser.api_key == "test_key"
        assert parser.client is not None


@pytest.mark.unit
class TestGeminiParserParse:
    """Tests for GeminiParser.parse() method."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_success(self, mock_client_class, tmp_path: Path):
        """Test successful PDF parsing."""
        # Create mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")
        
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock uploaded file
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "ACTIVE"
        mock_uploaded_file.name = "files/test-file"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        # Mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.text = "# Test Markdown\n\nContent here."
        mock_stream = [mock_chunk]
        mock_client.models.generate_content_stream.return_value = mock_stream
        
        # Create parser and parse
        config = GeminiParserConfig()
        parser = GeminiParser(config)
        result = parser.parse(pdf_path)
        
        # Verify result
        assert result == "# Test Markdown\n\nContent here."
        
        # Verify API calls
        mock_client.files.upload.assert_called_once()
        mock_client.models.generate_content_stream.assert_called_once()
        mock_client.files.delete.assert_called_once_with(name="files/test-file")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_with_file_processing_wait(self, mock_client_class, tmp_path: Path):
        """Test that parser waits for file processing."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock uploaded file that starts in PROCESSING state
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "PROCESSING"
        mock_uploaded_file.name = "files/test-file"
        
        # Mock file.get to return PROCESSING first, then ACTIVE
        mock_processing_file = MagicMock()
        mock_processing_file.state.name = "PROCESSING"
        mock_active_file = MagicMock()
        mock_active_file.state.name = "ACTIVE"
        
        mock_client.files.upload.return_value = mock_uploaded_file
        mock_client.files.get.side_effect = [mock_processing_file, mock_active_file]
        
        mock_chunk = MagicMock()
        mock_chunk.text = "Content"
        mock_client.models.generate_content_stream.return_value = [mock_chunk]
        
        config = GeminiParserConfig()
        parser = GeminiParser(config)
        result = parser.parse(pdf_path)
        
        assert result == "Content"
        # Should call get twice: once for PROCESSING, once for ACTIVE
        assert mock_client.files.get.call_count >= 1

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_file_processing_failed(self, mock_client_class, tmp_path: Path):
        """Test handling of failed file processing."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "FAILED"
        mock_uploaded_file.name = "files/test-file"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        config = GeminiParserConfig()
        parser = GeminiParser(config)
        
        with pytest.raises(RuntimeError, match="Failed to process file"):
            parser.parse(pdf_path)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_empty_response(self, mock_client_class, tmp_path: Path):
        """Test handling of empty response."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "ACTIVE"
        mock_uploaded_file.name = "files/test-file"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        # Empty stream
        mock_client.models.generate_content_stream.return_value = []
        
        config = GeminiParserConfig()
        parser = GeminiParser(config)
        result = parser.parse(pdf_path)
        
        assert result == ""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_cleanup_on_error(self, mock_client_class, tmp_path: Path):
        """Test that uploaded file is cleaned up on error."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "ACTIVE"
        mock_uploaded_file.name = "files/test-file"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        # Raise error during generation
        mock_client.models.generate_content_stream.side_effect = Exception("API error")
        
        config = GeminiParserConfig()
        parser = GeminiParser(config)
        
        with pytest.raises(Exception, match="API error"):
            parser.parse(pdf_path)
        
        # Verify cleanup was called
        mock_client.files.delete.assert_called_once_with(name="files/test-file")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_safety_settings_enabled(self, mock_client_class, tmp_path: Path):
        """Test that safety settings are applied when enabled."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "ACTIVE"
        mock_uploaded_file.name = "files/test-file"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        mock_chunk = MagicMock()
        mock_chunk.text = "Content"
        mock_client.models.generate_content_stream.return_value = [mock_chunk]
        
        config = GeminiParserConfig(safety_settings=True)
        parser = GeminiParser(config)
        parser.parse(pdf_path)
        
        # Verify safety settings were passed
        call_args = mock_client.models.generate_content_stream.call_args
        config_arg = call_args[1]["config"]
        assert len(config_arg.safety_settings) == 4

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_parse_uses_prompt_constant(self, mock_client_class, tmp_path: Path):
        """Test that the conversion prompt is used."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded_file
        mock_client.models.generate_content_stream.return_value = []
        
        config = GeminiParserConfig()
        parser = GeminiParser(config)
        parser.parse(pdf_path)
        
        # Verify prompt was passed
        call_args = mock_client.models.generate_content_stream.call_args
        contents = call_args[1]["contents"]
        assert GEMINI_PDF_TO_MD_PROMPT in contents


@pytest.mark.unit
class TestGetParserFactory:
    """Tests for get_parser() factory function."""

    def test_get_parser_gemini(self):
        """Test getting Gemini parser."""
        config = GeminiParserConfig()
        parser = get_parser("gemini", config)
        
        assert isinstance(parser, GeminiParser)

    def test_get_parser_gemini_case_insensitive(self):
        """Test that parser name is case-insensitive."""
        config = GeminiParserConfig()
        
        parser1 = get_parser("GEMINI", config)
        parser2 = get_parser("Gemini", config)
        
        assert isinstance(parser1, GeminiParser)
        assert isinstance(parser2, GeminiParser)

    def test_get_parser_gemini_without_config_raises_error(self):
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="requires GeminiParserConfig"):
            get_parser("gemini", None)

    def test_get_parser_gemini_with_wrong_config_raises_error(self):
        """Test that wrong config type raises ValueError."""
        from aee.infrastructure.config.settings import MarkerConfig
        
        marker_config = MarkerConfig(device="cpu")
        
        with pytest.raises(ValueError, match="requires GeminiParserConfig"):
            get_parser("gemini", marker_config)

    def test_get_parser_unknown_parser_raises_error(self):
        """Test that unknown parser raises ValueError."""
        with pytest.raises(ValueError, match="Unknown parser: unknown"):
            get_parser("unknown", None)

    def test_get_parser_marker_still_works(self):
        """Test that Marker parser still works."""
        from aee.infrastructure.config.settings import MarkerConfig
        
        config = MarkerConfig(device="cpu")
        parser = get_parser("marker", config)
        
        assert parser.__class__.__name__ == "MarkerParser"
