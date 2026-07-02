import os
import io
import json
import zipfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import responses

from ae.core.config import IngestionConfig, MinerUParserConfig, ChartExtractionConfig
from ae.ingestion.parsers.mineru.client import MinerUClient
from ae.ingestion.parsers.mineru.parser import MinerUParser, find_mineru_outputs
from ae.ingestion.parsers import get_parser
from ae.ingestion.parsers.mineru.visual.stages.insert_visual_tables import replace_image_tags


@pytest.mark.unit
class TestMinerUParserConfig:
    """Unit tests for MinerU parser configuration."""

    def test_config_validation(self):
        config = IngestionConfig(
            overwrite=False,
            concurrency=2,
            mineru=MinerUParserConfig(
                api_url="https://test.mineru.net/api/v4",
                model_version="vlm",
                poll_interval=1,
                poll_timeout=10,
            ),
            chart_extraction=ChartExtractionConfig(
                enabled=True,
            )
        )
        assert config.mineru.api_url == "https://test.mineru.net/api/v4"
        assert config.mineru.poll_timeout == 10
        assert config.chart_extraction.enabled is True


@pytest.mark.unit
class TestMinerUClient:
    """Unit tests for MinerUClient using responses mock."""

    @patch.dict(os.environ, {"MINERU_API_TOKEN": "test_token"})
    @responses.activate
    def test_parse_pdf_flow_success(self, tmp_path: Path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        # Mock 1: Request upload URL
        responses.add(
            responses.POST,
            "https://mineru.net/api/v4/file-urls/batch",
            json={
                "code": 0,
                "msg": "success",
                "data": {
                    "batch_id": "batch_123",
                    "file_urls": ["https://mineru.net/upload/test.pdf"]
                }
            },
            status=200
        )

        # Mock 2: PUT upload file
        responses.add(
            responses.PUT,
            "https://mineru.net/upload/test.pdf",
            status=200
        )

        # Mock 3: Poll status (returns done)
        responses.add(
            responses.GET,
            "https://mineru.net/api/v4/extract-results/batch/batch_123",
            json={
                "code": 0,
                "msg": "success",
                "data": {
                    "extract_result": [
                        {
                            "state": "done",
                            "full_zip_url": "https://mineru.net/download/result.zip"
                        }
                    ]
                }
            },
            status=200
        )

        # Create dummy ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr("full.md", "Hello world! ![](images/test_chart.jpg)")
            zip_file.writestr(
                "content_list.json",
                json.dumps([
                    {
                        "type": "chart",
                        "img_path": "images/test_chart.jpg",
                        "chart_caption": "A nice chart"
                    }
                ])
            )
            zip_file.writestr("images/test_chart.jpg", b"fake-image-bytes")

        # Mock 4: Download ZIP
        responses.add(
            responses.GET,
            "https://mineru.net/download/result.zip",
            body=zip_buffer.getvalue(),
            status=200
        )

        config = MinerUParserConfig(
            api_url="https://mineru.net/api/v4",
            poll_interval=1,
            poll_timeout=10
        )
        client = MinerUClient(config)

        output_dir = tmp_path / "mineru_output"
        result = client.parse_pdf(str(pdf_path), str(output_dir))

        assert result["state"] == "done"
        assert (output_dir / "full.md").exists()
        assert (output_dir / "content_list.json").exists()
        assert (output_dir / "images" / "test_chart.jpg").exists()


@pytest.mark.unit
class TestMinerUParser:
    """Unit tests for MinerUParser and image tag replacement."""

    def test_find_mineru_outputs(self, tmp_path: Path):
        # Setup mock MinerU directory structure
        mineru_dir = tmp_path / "mineru_run"
        mineru_dir.mkdir()
        (mineru_dir / "full.md").write_text("Hello", encoding="utf-8")
        (mineru_dir / "content_list.json").write_text("[]", encoding="utf-8")
        (mineru_dir / "images").mkdir()
        (mineru_dir / "images" / "img.jpg").write_text("data", encoding="utf-8")

        md_file, json_file, images_dir = find_mineru_outputs(mineru_dir)
        assert md_file.name == "full.md"
        assert json_file.name == "content_list.json"
        assert images_dir.name == "images"

    def test_replace_image_tags(self):
        markdown = "Some text.\n![](images/test_chart.jpg)\nOther text."
        results = {
            "images/test_chart.jpg": {
                "status": "success",
                "tables": [
                    {
                        "columns": ["X", "Y"],
                        "rows": [["1", "2"], ["3", "4"]]
                    }
                ]
            }
        }
        warnings = []
        replaced = replace_image_tags(markdown, results, warnings)
        
        expected_table = "| X | Y |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        assert expected_table in replaced
        assert "![](images/test_chart.jpg)" not in replaced

    def test_replace_image_tags_with_caption_status_warnings(self):
        markdown = "Some text.\n![Figure 1: Comparison](images/test_chart.jpg)\nOther text."
        results = {
            "images/test_chart.jpg": {
                "status": "partial",
                "tables": [
                    {
                        "panel": "a",
                        "chart_type": "bar",
                        "series_name": "SeriesA",
                        "columns": ["X", "Y"],
                        "rows": [["1", "2", "3"]]
                    }
                ],
                "warnings": ["unreadable labels"]
            }
        }
        warnings = []
        replaced = replace_image_tags(markdown, results, warnings)
        
        assert "**Figure Caption:** Figure 1: Comparison" in replaced
        assert "Panel a, Series: SeriesA, Type: bar" in replaced
        assert "| X | Y |" in replaced
        assert "test_chart.jpg:row_1_column_count_mismatch" in warnings
        
        # Verify removed elements are not present in output markdown
        assert "*Source Image:" not in replaced
        assert "Extraction Status" not in replaced
        assert "Warnings / Ambiguities" not in replaced
        assert "unreadable labels" not in replaced
        assert "Data Formatting Warnings" not in replaced

    @patch.dict(os.environ, {"MINERU_API_TOKEN": "test_token"})
    @patch("ae.ingestion.parsers.mineru.parser.find_project_root")
    @patch("ae.ingestion.parsers.mineru.parser.MinerUClient")
    @patch("ae.ingestion.parsers.mineru.parser.get_model_client")
    @patch("ae.ingestion.parsers.mineru.parser.extract_single_chart")
    def test_mineru_parser_end_to_end(
        self,
        mock_extract_single_chart,
        mock_get_model_client,
        mock_client_class,
        mock_find_project_root,
        tmp_path: Path
    ):
        mock_find_project_root.return_value = tmp_path
        pdf_path = tmp_path / "document.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        # Mock the MinerU client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock VLM extraction result
        mock_extract_single_chart.return_value = {
            "status": "success",
            "tables": [
                {
                    "columns": ["ColA", "ColB"],
                    "rows": [["val1", "val2"]]
                }
            ],
            "warnings": []
        }

        # Mock parser config
        config = IngestionConfig(
            overwrite=True,
            mineru=MinerUParserConfig(
                api_url="https://mineru.net/api/v4",
                poll_interval=1,
                poll_timeout=10
            ),
            chart_extraction=ChartExtractionConfig(
                provider="gemini",
                model="gemini-3.5-flash"
            )
        )

        parser = MinerUParser(config)

        # Setup side effect for parse_pdf: write output files to mock output dir
        def mock_parse_pdf(pdf, out_dir):
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "full.md").write_text("Text\n![](images/chart1.jpg)", encoding="utf-8")
            (out_path / "content_list.json").write_text(
                json.dumps([{"type": "chart", "img_path": "images/chart1.jpg"}]),
                encoding="utf-8"
            )
            (out_path / "images").mkdir(exist_ok=True)
            (out_path / "images" / "chart1.jpg").write_text("fake image bytes", encoding="utf-8")
            return {"state": "done"}

        mock_client.parse_pdf.side_effect = mock_parse_pdf

        # Run parser
        result_md = parser.parse(pdf_path)

        assert "| ColA | ColB |" in result_md
        assert "![](images/chart1.jpg)" not in result_md
        mock_client.parse_pdf.assert_called_once()
        mock_extract_single_chart.assert_called_once()
