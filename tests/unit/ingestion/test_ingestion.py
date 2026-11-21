"""Unit tests for document ingestion use case and parser registry (Ingestion v2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ae.core.config.optimization_settings import (
    GeminiDefaultsConfig,
    GeminiParserConfig,
    IngestionConfig,
)
from ae.core.storage import DocumentRepository
from ae.ingestion.parsers import get_parser
from ae.ingestion.parsers.gemini.parser import GeminiParser
from ae.ingestion.use_case import (
    IngestDocumentsRequest,
    IngestDocumentsUseCase,
)


@pytest.mark.unit
class TestParserRegistry:
    def test_get_gemini_parser(self):
        config = IngestionConfig(
            gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
        )
        with patch.object(GeminiParser, "_resolve_api_key", return_value="fake_key"):
            parser = get_parser("gemini", config)
            assert isinstance(parser, GeminiParser)

    def test_unknown_parser_raises_value_error(self):
        config = IngestionConfig(
            gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
        )
        with pytest.raises(ValueError, match="Unknown parser: mineru"):
            get_parser("mineru", config)


@pytest.mark.unit
class TestIngestDocumentsUseCase:
    @pytest.fixture
    def mock_doc_repo(self, tmp_path: Path):
        return DocumentRepository(ingestion_dir=tmp_path)

    def test_ingest_documents_success(self, tmp_path: Path, mock_doc_repo: DocumentRepository):
        pdf1 = tmp_path / "paper1.pdf"
        pdf2 = tmp_path / "paper2.pdf"
        pdf1.write_bytes(b"%PDF-1.4 dummy 1")
        pdf2.write_bytes(b"%PDF-1.4 dummy 2")

        request = IngestDocumentsRequest(
            input_paths=[pdf1, pdf2],
            output_dir=tmp_path / "output",
            parser_name="gemini",
            overwrite=False,
            concurrency=2,
        )

        mock_parser = MagicMock()
        mock_parser.parse.side_effect = lambda path: f"# Enriched {path.stem}"

        with patch("ae.ingestion.use_case.get_parser", return_value=mock_parser):
            use_case = IngestDocumentsUseCase(document_repo=mock_doc_repo)
            response = use_case.execute(request)

            assert response.success is True
            assert response.documents_parsed == 2
            assert response.total_documents == 2
            assert response.failed_documents == 0

            # Verify files were saved
            out1 = request.output_dir / "paper1" / "paper1.md"
            out2 = request.output_dir / "paper2" / "paper2.md"
            assert out1.exists()
            assert out2.exists()
            assert out1.read_text(encoding="utf-8") == "# Enriched paper1"

    def test_ingest_documents_skips_existing(
        self, tmp_path: Path, mock_doc_repo: DocumentRepository
    ):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        out_dir = tmp_path / "output"
        out_file = out_dir / "paper" / "paper.md"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("# Existing", encoding="utf-8")

        request = IngestDocumentsRequest(
            input_paths=[pdf],
            output_dir=out_dir,
            parser_name="gemini",
            overwrite=False,
        )

        mock_parser = MagicMock()
        with patch("ae.ingestion.use_case.get_parser", return_value=mock_parser):
            use_case = IngestDocumentsUseCase(document_repo=mock_doc_repo)
            response = use_case.execute(request)

            assert response.success is True
            assert response.documents_parsed == 0
            assert response.documents_skipped == 1
            mock_parser.parse.assert_not_called()

    def test_ingest_documents_overwrite_flag(
        self, tmp_path: Path, mock_doc_repo: DocumentRepository
    ):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        out_dir = tmp_path / "output"
        out_file = out_dir / "paper" / "paper.md"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("# Old", encoding="utf-8")

        request = IngestDocumentsRequest(
            input_paths=[pdf],
            output_dir=out_dir,
            parser_name="gemini",
            overwrite=True,
        )

        mock_parser = MagicMock()
        mock_parser.parse.return_value = "# New Overwritten"
        with patch("ae.ingestion.use_case.get_parser", return_value=mock_parser):
            use_case = IngestDocumentsUseCase(document_repo=mock_doc_repo)
            response = use_case.execute(request)

            assert response.success is True
            assert response.documents_parsed == 1
            assert out_file.read_text(encoding="utf-8") == "# New Overwritten"
