import pytest
from pathlib import Path
from ae.core.storage.documents import DocumentRepository


@pytest.mark.unit
class TestDocumentRepositoryLayouts:
    """Test DocumentRepository handling of flat, nested, and mixed directory layouts."""

    def test_flat_layout(self, tmp_path: Path):
        """Test document discovery and retrieval in flat directory."""
        ingestion_dir = tmp_path / "ingestion"
        ingestion_dir.mkdir()

        (ingestion_dir / "paper1.md").write_text("Content of paper 1", encoding="utf-8")
        (ingestion_dir / "paper2.md").write_text("Content of paper 2", encoding="utf-8")

        repo = DocumentRepository(ingestion_dir=ingestion_dir)

        keys = repo.list_document_keys()
        assert keys == ["paper1", "paper2"]

        assert repo.exists("paper1")
        assert repo.exists("paper2")
        assert not repo.exists("paper3")

        assert repo.get("paper1") == "Content of paper 1"
        assert repo.get("PAPER1") == "Content of paper 1"
        assert repo.get("nonexistent") is None

        docs = repo.load_all()
        assert len(docs) == 2
        assert docs["paper1"] == "Content of paper 1"
        assert docs["paper2"] == "Content of paper 2"

    def test_nested_layout_with_artifacts(self, tmp_path: Path):
        """Test document discovery and retrieval in nested directory (multi-stage ingestion output)."""
        ingestion_dir = tmp_path / "ingestion"
        ingestion_dir.mkdir()

        doc_dir = ingestion_dir / "antibiotics12071220_pages_4-5"
        doc_dir.mkdir()

        # Primary doc file
        primary_file = doc_dir / "antibiotics12071220_pages_4-5.md"
        primary_file.write_text("Full processed markdown with visuals", encoding="utf-8")

        # Stage artifacts (must not be discovered as documents)
        stage1 = doc_dir / "01_parse_text"
        stage1.mkdir()
        (stage1 / "text.md").write_text("Raw text", encoding="utf-8")

        stage2 = doc_dir / "02_manifest_visuals"
        stage2.mkdir()
        (stage2 / "manifest_enriched_text.md").write_text("Enriched text", encoding="utf-8")

        repo = DocumentRepository(ingestion_dir=ingestion_dir)

        keys = repo.list_document_keys()
        assert keys == ["antibiotics12071220_pages_4-5"]

        assert repo.exists("antibiotics12071220_pages_4-5")
        assert repo.exists("ANTIBIOTICS12071220_pages_4-5")
        assert not repo.exists("text")
        assert not repo.exists("manifest_enriched_text")

        text = repo.get("antibiotics12071220_pages_4-5")
        assert text == "Full processed markdown with visuals"

        docs = repo.load_all()
        assert len(docs) == 1
        assert "antibiotics12071220_pages_4-5" in docs
        assert docs["antibiotics12071220_pages_4-5"] == "Full processed markdown with visuals"

    def test_mixed_layout(self, tmp_path: Path):
        """Test directory containing both flat .md files and nested doc folders."""
        ingestion_dir = tmp_path / "ingestion"
        ingestion_dir.mkdir()

        # Flat doc
        (ingestion_dir / "flat_doc.md").write_text("Flat doc content", encoding="utf-8")

        # Nested doc
        nested_dir = ingestion_dir / "nested_doc"
        nested_dir.mkdir()
        (nested_dir / "nested_doc.md").write_text("Nested doc content", encoding="utf-8")

        repo = DocumentRepository(ingestion_dir=ingestion_dir)

        keys = repo.list_document_keys()
        assert sorted(keys) == ["flat_doc", "nested_doc"]

        assert repo.exists("flat_doc")
        assert repo.exists("nested_doc")

        assert repo.get("flat_doc") == "Flat doc content"
        assert repo.get("nested_doc") == "Nested doc content"

        docs = repo.load_all()
        assert len(docs) == 2
        assert docs["flat_doc"] == "Flat doc content"
        assert docs["nested_doc"] == "Nested doc content"

    def test_suffixes_and_normalization(self, tmp_path: Path):
        """Test that suffixes like _parsed are supported in nested folders."""
        ingestion_dir = tmp_path / "ingestion"
        ingestion_dir.mkdir()

        nested_dir = ingestion_dir / "sample_doc"
        nested_dir.mkdir()
        (nested_dir / "sample_doc_parsed.md").write_text("Parsed content", encoding="utf-8")

        repo = DocumentRepository(ingestion_dir=ingestion_dir)

        keys = repo.list_document_keys()
        assert "sample_doc" in keys

        assert repo.exists("sample_doc")
        assert repo.get("sample_doc") == "Parsed content"

    def test_get_canonical_manifest_text(self, tmp_path: Path):
        """Prefer 02_manifest_visuals/manifest_enriched_text.md when present and non-empty, else fallback."""
        ingestion_dir = tmp_path / "ingestion"
        doc_dir = ingestion_dir / "doc_a"
        manifest_dir = doc_dir / "02_manifest_visuals"
        manifest_dir.mkdir(parents=True)

        (doc_dir / "doc_a.md").write_text("Unpacked in-place text", encoding="utf-8")
        manifest_path = manifest_dir / "manifest_enriched_text.md"
        manifest_path.write_text("Canonical manifest text", encoding="utf-8")

        repo = DocumentRepository(ingestion_dir=ingestion_dir)
        assert repo.get_canonical_manifest_text("doc_a") == "Canonical manifest text"

        # Empty manifest falls back to main .md
        manifest_path.write_text("", encoding="utf-8")
        assert repo.get_canonical_manifest_text("doc_a") == "Unpacked in-place text"

        # Missing manifest falls back to main .md
        manifest_path.unlink()
        assert repo.get_canonical_manifest_text("doc_a") == "Unpacked in-place text"
        assert repo.get_canonical_manifest_text("missing_doc") is None
