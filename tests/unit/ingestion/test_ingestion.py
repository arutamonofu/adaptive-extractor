import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ae.core.config import (
    AEVisualParserConfig,
    GeminiParserConfig,
    IngestionConfig,
)
from ae.ingestion.gemini_parser import GeminiParser
from ae.ingestion.parsers import get_parser
from ae.ingestion.visual_parser import AEVisualParser


@pytest.mark.unit
class TestGeminiParser:
    """Unit tests for Gemini parser configuration and mocked parsing."""

    def test_config_validation(self):
        config = GeminiParserConfig(model_name="gemini-2.0-flash", upload_timeout=600, safety_settings=False)
        assert config.model_name == "gemini-2.0-flash"
        assert config.upload_timeout == 600
        assert config.safety_settings is False

        ing_config = IngestionConfig(gemini=config, overwrite=False)
        assert ing_config.gemini.model_name == "gemini-2.0-flash"
        assert ing_config.visual.enabled is False

    @patch.dict(os.environ, {}, clear=False)
    def test_init_without_api_key_raises_error(self):
        os.environ.pop("GEMINI_API_KEY", None)
        config = GeminiParserConfig()
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable"):
            GeminiParser(config)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    @patch("time.sleep")
    def test_parse_success_and_wait_loop(self, mock_sleep, mock_client_class, tmp_path: Path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake-content")

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock uploaded file starting in PROCESSING then ACTIVE
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.state.name = "PROCESSING"
        mock_uploaded_file.name = "files/test-file"

        mock_active_file = MagicMock()
        mock_active_file.state.name = "ACTIVE"
        mock_active_file.name = "files/test-file"

        mock_client.files.upload.return_value = mock_uploaded_file
        mock_client.files.get.side_effect = [mock_uploaded_file, mock_active_file]

        # Mock generate content
        mock_chunk = MagicMock()
        mock_chunk.text = "# Content"
        mock_client.models.generate_content_stream.return_value = [mock_chunk]

        parser = GeminiParser(GeminiParserConfig())
        result = parser.parse(pdf_path)

        assert result == "# Content"
        mock_client.files.upload.assert_called_once()
        mock_client.files.delete.assert_called_once_with(name="files/test-file")
        assert mock_sleep.call_count >= 1

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    def test_get_parser_factory(self, mock_client_class):
        config = GeminiParserConfig()
        parser = get_parser("gemini", config)
        assert isinstance(parser, GeminiParser)


@pytest.mark.unit
class TestAEVisualParser:
    """Unit tests for AEVisualParser configuration and enrichment pipeline."""

    def test_visual_config(self):
        config = AEVisualParserConfig(
            enabled=True,
            task_config_path="config/tasks/nanozymes/initial_schema.yaml",
            pipeline={
                "models": {
                    "manifest": {"provider": "gemini", "model": "gemini-3-flash-preview"},
                    "bbox": {"provider": "gemini", "model": "gemini-3-flash-preview"},
                    "chart_extraction": {"provider": "gemini", "model": "gemini-3-flash-preview"}
                },
                "runtime": {"dpi": 400}
            },
            gemini=GeminiParserConfig(model_name="gemini-2.0-flash")
        )
        assert config.task_config_path == "config/tasks/nanozymes/initial_schema.yaml"
        assert config.gemini.model_name == "gemini-2.0-flash"

        ing_config = IngestionConfig(visual=config, overwrite=False)
        assert ing_config.visual.enabled is True


    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("google.genai.Client")
    @patch("ae.ingestion.visual_parser.run_visual_pipeline")
    def test_visual_parse_success(self, mock_run_pipeline, mock_client):
        config = AEVisualParserConfig(
            enabled=True,
            task_config_path="config/tasks/nanozymes/initial_schema.yaml",
            pipeline={
                "models": {
                    "manifest": {"provider": "gemini", "model": "gemini-3-flash-preview"},
                    "bbox": {"provider": "gemini", "model": "gemini-3-flash-preview"},
                    "chart_extraction": {"provider": "gemini", "model": "gemini-3-flash-preview"}
                },
                "runtime": {"dpi": 400}
            }
        )
        parser = AEVisualParser(config)

        # Mock the base text parser
        parser.base_parser.parse = MagicMock(return_value="Initial Markdown with <!-- AE_VISUAL_ANCHOR: main_fig_1 -->")

        # Mock run_visual_pipeline to return our enriched markdown
        mock_run_pipeline.return_value = "Enriched Visual Markdown"

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "article.pdf"
            pdf_path.write_text("dummy pdf", encoding="utf-8")

            result = parser.parse(pdf_path)

        assert result == "Enriched Visual Markdown"
        mock_run_pipeline.assert_called_once()

    def test_filter_tables_single_panel_robustness(self):
        from ae.ingestion.visual_pipeline.stages.extract_chart_tables import _filter_tables_by_panel_metadata

        # Case 1: Exactly 1 allowed panel, and LLM output has no panel.
        # It should auto-assign it.
        target = {
            "panels": [
                {"panel": "a", "num": True, "rel": "direct"}
            ]
        }
        parsed = {
            "tables": [
                {
                    "panel": None,
                    "columns": ["col1"],
                    "rows": [["val1"]]
                }
            ],
            "warnings": []
        }
        result = _filter_tables_by_panel_metadata(parsed, target)
        assert len(result["tables"]) == 1
        assert result["tables"][0]["panel"] == "a"
        assert any("auto_assigned_table_to_single_panel" in w for w in result["warnings"])

        # Case 2: Disallowed panel mismatch with multiple allowed panels
        target_multi = {
            "panels": [
                {"panel": "a", "num": True, "rel": "direct"},
                {"panel": "b", "num": True, "rel": "direct"}
            ]
        }
        parsed_multi = {
            "tables": [
                {
                    "panel": None,
                    "columns": ["col1"],
                    "rows": [["val1"]]
                }
            ],
            "warnings": []
        }
        result_multi = _filter_tables_by_panel_metadata(parsed_multi, target_multi)
        # Should be dropped because multiple allowed panels exist and none matches null
        assert len(result_multi["tables"]) == 0
        assert any("dropped_table_disallowed_panel" in w for w in result_multi["warnings"])
