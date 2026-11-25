"""Unit tests for Contour 1 (EXP-01A) hardening fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pymupdf as fitz
import pytest
from PIL import Image as PILImage

from ae.core.config.optimization_settings import (
    ApiClientConfig,
    ExecutionConfig,
    GeminiDefaultsConfig,
    GeminiParserConfig,
    IngestionConfig,
)
from ae.core.exceptions import GeminiMaxTokensError
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import BatchItemResult, GeminiClient
from ae.ingestion.parsers.gemini.parser import GeminiParser
from ae.ingestion.parsers.gemini.schemas import (
    Manifest,
    ManifestVisual,
    PlaceholderItem,
)
from ae.ingestion.parsers.gemini.stages.manifest_visuals import (
    _read_prompt_template as _read_manifest_prompt,
    build_manifest_visuals_batch_spec,
    get_required_pages,
    reconcile_manifest,
    run_manifest_visuals,
)
from ae.ingestion.parsers.gemini.stages.parse_text import (
    assemble_parse_text_chunks,
    atomic_write_text,
    build_parse_text_batch_specs,
    run_chunked_parse_text,
    split_pdf_into_chunks,
    strip_outer_code_fences,
)
from ae.ingestion.parsers.gemini.stages.silent_error_guard import (
    _is_likely_reference_or_boilerplate_page,
    detect_manifest_silent_error,
    detect_parse_text_silent_error,
)


def _make_pdf(path: Path, pages_text: list[str]) -> Path:
    doc = fitz.open()
    for text in pages_text:
        p = doc.new_page(width=400, height=600)
        if text:
            for idx, line in enumerate(text.splitlines()):
                p.insert_text(fitz.Point(40, 50 + idx * 16), line)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# 1. client.py fixes
# ---------------------------------------------------------------------------


def test_client_validate_response_max_tokens_nonempty_raises():
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "MAX_TOKENS"
    mock_candidate.content.parts = [MagicMock()]
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None

    with pytest.raises(GeminiMaxTokensError):
        client._validate_response(mock_resp, stage_name="parse_text")


def test_client_normalize_contents_closes_pil_file_handle(tmp_path: Path):
    img_path = tmp_path / "sample.png"
    PILImage.new("RGB", (32, 32), color=(255, 0, 0)).save(img_path, format="PNG")

    norm = GeminiClient._normalize_contents(["Header", img_path])
    assert norm[0] == "Header"
    assert isinstance(norm[1], PILImage.Image)
    assert getattr(norm[1], "fp", None) is None
    assert norm[1].size == (32, 32)


def test_client_passes_timeout_ms_and_falls_back(monkeypatch):
    captured: list[dict] = []

    def _fake_client_with_http(*, api_key: str, http_options=None):
        captured.append({"api_key": api_key, "http_options": http_options})
        return MagicMock()

    import google.genai as real_genai

    monkeypatch.setattr(real_genai, "Client", _fake_client_with_http)
    cfg = IngestionConfig(api_client=ApiClientConfig(timeout=120))
    GeminiClient(cfg.gemini, api_key="k1")
    assert "http_options" in captured[0]
    http_opts = captured[0]["http_options"]
    timeout_val = getattr(http_opts, "timeout", None)
    if timeout_val is None and isinstance(http_opts, dict):
        timeout_val = http_opts.get("timeout")
    assert timeout_val == 120_000

    # Fallback when genai.Client rejects http_options (e.g. legacy signature)
    def _legacy_client(api_key: str):
        return MagicMock(api_key=api_key)

    monkeypatch.setattr(real_genai, "Client", _legacy_client)
    client2 = GeminiClient(cfg.gemini, api_key="k2")
    assert client2._client.api_key == "k2"


# ---------------------------------------------------------------------------
# 2. silent_error_guard.py fixes
# ---------------------------------------------------------------------------


def test_guard_skips_silent_omission_for_valid_short_visual_placeholder():
    raw_pdf = " ".join(["CellData"] * 40)  # >= 150 chars
    assert len(raw_pdf) >= 150

    assert detect_parse_text_silent_error('<visual id="Table_1" />', pdf_page_text=raw_pdf) is None
    assert (
        detect_parse_text_silent_error("<!-- omitted -->", pdf_page_text=raw_pdf)
        is not None
    )


def test_guard_mixed_conclusions_plus_references_not_skipped():
    mixed_lines = [
        "### 4. Conclusions",
        "Compound 15 demonstrated strong activity across all tested Gram-positive strains.",
        "The pharmacokinetic profile supports further lead optimization in animal models.",
        "Additional selectivity assays confirmed minimal cytotoxicity against Vero cells.",
        "Overall, the benzimidazole scaffold provides a promising starting point.",
        "Further structural modifications at C5 are currently underway in our laboratory.",
        "### References",
        "[1] Smith J. J. Med. Chem. 2020, 63, 100-110.",
        "[2] Doe A. Eur. J. Med. Chem. 2021, 210, 112-120.",
    ]
    mixed_text = "\n".join(mixed_lines)
    assert _is_likely_reference_or_boilerplate_page(mixed_text) is False
    err = detect_parse_text_silent_error("<!-- omitted -->", pdf_page_text=mixed_text)
    assert err is not None and "silent_omission" in err

    pure_refs = "\n".join(
        [
            "References",
            "[1] Smith J. J. Med. Chem. 2020, 63, 100-110.",
            "[2] Doe A. Eur. J. Med. Chem. 2021, 210, 112-120.",
            "[3] Roe B. Bioorg. Med. Chem. 2022, 55, 116-125.",
            "[4] Lee C. ACS Omega 2023, 8, 500-510.",
            "[5] Kim D. Nature 2024, 625, 10-20.",
        ]
    )
    assert _is_likely_reference_or_boilerplate_page(pure_refs) is True


def test_guard_detects_malformed_visual_tags():
    assert "malformed_visual_tag" in (
        detect_parse_text_silent_error('Text\n<visual id="Figure 1" />') or ""
    )
    assert "malformed_visual_tag" in (
        detect_parse_text_silent_error('Text\n<visual type="table" id="Table_1" />') or ""
    )
    assert "malformed_visual_tag" in (
        detect_parse_text_silent_error('Text\n<visual id="Table_1"') or ""
    )
    assert "stray_visual_close_tag" in (
        detect_parse_text_silent_error("Text\n</visual>") or ""
    )


# ---------------------------------------------------------------------------
# 3. parse_text.py & parser.py fixes
# ---------------------------------------------------------------------------


def test_strip_outer_code_fences_handles_markdown_and_inner_blocks():
    wrapped = "```markdown\n# Title\n\nBody text.\n```"
    assert strip_outer_code_fences(wrapped) == "# Title\n\nBody text."

    two_blocks = "```\nblock1\n```\nmiddle text\n```\nblock2\n```"
    assert strip_outer_code_fences(two_blocks) == two_blocks

    nested_md = "```markdown\n# Code\n```python\nx = 1\n```\nDone.\n```"
    assert strip_outer_code_fences(nested_md) == "# Code\n```python\nx = 1\n```\nDone."


def test_parse_text_does_not_accept_longer_broken_retry(tmp_path: Path):
    pdf_path = _make_pdf(
        tmp_path / "doc.pdf",
        ["Experimental details: compound 12 was synthesized in 85% yield and recrystallized from EtOH.\n" * 3],
    )
    artifacts = ArtifactManager("doc", base_dir=tmp_path)
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))

    mock_client = MagicMock()
    mock_client.make_pdf_part.return_value = "pdf_part"
    collapsed_long_retry = "\n".join(["Repeated collapse line for testing."] * 8)

    calls = {"n": 0}

    def _gen(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "<!-- omitted -->"
        return collapsed_long_retry

    mock_client.generate_text.side_effect = _gen
    md, _ = run_chunked_parse_text(mock_client, pdf_path, artifacts, cfg)
    assert calls["n"] == 2
    assert "Repeated collapse line" not in md
    assert "<!-- omitted -->" in md


def test_parse_text_retries_on_max_tokens_and_never_caches_truncated(tmp_path: Path):
    pdf_path = _make_pdf(tmp_path / "max_tok.pdf", ["Page 1 scientific text."])
    artifacts = ArtifactManager("max_tok", base_dir=tmp_path)
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))

    # Case 1: first call hits MAX_TOKENS, retry succeeds
    mock_client = MagicMock()
    mock_client.make_pdf_part.return_value = "pdf_part"
    calls = {"n": 0}

    def _recover(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise GeminiMaxTokensError(stage="parse_text")
        return "Recovered page 1 text."

    mock_client.generate_text.side_effect = _recover
    md, _ = run_chunked_parse_text(mock_client, pdf_path, artifacts, cfg)
    assert "Recovered page 1 text." in md
    assert (artifacts.chunks_dir / "chunk_1_1.md").exists()

    # Case 2: both calls hit MAX_TOKENS -> raises and does NOT cache
    artifacts2 = ArtifactManager("max_tok_fail", base_dir=tmp_path)
    mock_client2 = MagicMock()
    mock_client2.make_pdf_part.return_value = "pdf_part"
    mock_client2.generate_text.side_effect = GeminiMaxTokensError(stage="parse_text")
    with pytest.raises(GeminiMaxTokensError):
        run_chunked_parse_text(mock_client2, pdf_path, artifacts2, cfg)
    assert not (artifacts2.chunks_dir / "chunk_1_1.md").exists()


def test_lazy_chunk_splitting_skips_cached_pages(tmp_path: Path):
    pdf_path = _make_pdf(tmp_path / "lazy.pdf", ["Page 1", "Page 2"])
    artifacts = ArtifactManager("lazy", base_dir=tmp_path)
    (artifacts.chunks_dir / "chunk_1_1.md").write_text("Cached page 1", encoding="utf-8")

    temp_dir = tmp_path / "tmp_split"
    chunks = split_pdf_into_chunks(pdf_path, temp_dir, chunk_pages=1, chunks_dir=artifacts.chunks_dir)
    assert len(chunks) == 2
    assert not (temp_dir / "chunk_1_1.pdf").exists()
    assert (temp_dir / "chunk_2_2.pdf").exists()

    # When both are cached, temp_dir is not even created
    (artifacts.chunks_dir / "chunk_2_2.md").write_text("Cached page 2", encoding="utf-8")
    temp_dir2 = tmp_path / "tmp_split2"
    split_pdf_into_chunks(pdf_path, temp_dir2, chunk_pages=1, chunks_dir=artifacts.chunks_dir)
    assert not temp_dir2.exists()


def test_assemble_preserves_internal_page_breaks_when_chunk_pages_gt_1(tmp_path: Path):
    artifacts = ArtifactManager("multi_chunk", base_dir=tmp_path)
    chunks = [(1, 2, tmp_path / "c1.pdf"), (3, 4, tmp_path / "c2.pdf")]
    chunk_results = [
        '<!-- page_break -->\nPage 1\n\n<!-- page_break -->\n\nPage 2 <visual id="Fig_1" />\n<!-- page_break -->',
        "Page 3\n\n<!-- page_break -->\n\nPage 4",
    ]
    clean_md, phs = assemble_parse_text_chunks(chunks, chunk_results, artifacts)
    assert clean_md.count("<!-- page_break -->") == 3
    assert len(phs) == 1
    assert phs[0].page == 2


def test_atomic_write_text_and_batch_inline_limit_and_retry_validation(tmp_path: Path, monkeypatch):
    target = tmp_path / "cache" / "chunk_1_1.md"
    atomic_write_text(target, "Atomic content")
    assert target.read_text(encoding="utf-8") == "Atomic content"
    assert list(target.parent.glob("*.tmp")) == []

    # Test build_parse_text_batch_specs falls back to upload_file when > INLINE_PDF_MAX_BYTES
    pdf_path = _make_pdf(tmp_path / "big_batch.pdf", ["Batch text " * 20])
    artifacts = ArtifactManager("big_batch", base_dir=tmp_path)
    cfg = IngestionConfig(
        ingestion_dir=tmp_path / "ing",
        execution=ExecutionConfig(use_batch_api=True, silent_error_guard=True),
    )
    monkeypatch.setattr("ae.ingestion.parsers.gemini.stages.parse_text.INLINE_PDF_MAX_BYTES", 1)

    mock_client = MagicMock()
    fake_handle = MagicMock(name="uploaded_pdf_handle")
    mock_client.upload_file.return_value = fake_handle
    mock_client.get_all_stage_usage.return_value = {}
    mock_client.get_scope_counters.return_value = {}

    _, specs, _ = build_parse_text_batch_specs(mock_client, pdf_path, artifacts, cfg)
    assert len(specs) == 1
    assert specs[0].contents[0] is fake_handle
    assert specs[0].uploaded_handles == [fake_handle]

    # Test parse_batch cleans up uploaded handles and re-validates silent-error retry
    monkeypatch.setattr(GeminiParser, "_resolve_api_key", staticmethod(lambda: "dummy"))
    parser = GeminiParser(cfg)
    parser._client = mock_client

    broken_retry = "\n".join(["Repeated garbage line."] * 8)
    mock_client.generate_text.return_value = broken_retry

    def _batch_side_effect(batch_specs, **kwargs):
        out = {}
        for s in batch_specs:
            if s.stage_name == "parse_text":
                out[s.key] = BatchItemResult(key=s.key, text="<!-- omitted -->")
            else:
                out[s.key] = BatchItemResult(key=s.key, parsed=Manifest(visuals=[], discarded_ids=[]))
        return out

    mock_client.run_batch_requests.side_effect = _batch_side_effect
    parser.parse_batch([pdf_path])

    mock_client.delete_file.assert_called_with(fake_handle)
    cached = (ArtifactManager("big_batch", base_dir=cfg.ingestion_dir).chunks_dir / "chunk_1_1.md").read_text(
        encoding="utf-8"
    )
    # Broken retry must NOT overwrite original chunk_md
    assert "Repeated garbage line." not in cached


# ---------------------------------------------------------------------------
# 4. manifest_visuals.py & manifest_visuals.txt fixes
# ---------------------------------------------------------------------------


def test_reconcile_manifest_fuzzy_dedup_and_page_bounds():
    placeholders = [
        PlaceholderItem(id="Fig._10", caption="Fig. 10", page=7),
        PlaceholderItem(id="Table_1", caption="Table 1", page=3),
    ]
    raw = Manifest(
        visuals=[
            ManifestVisual(id="Fig.-10", type="molecular_structure", pages=[7, 6, 7, 99, 0], description="D"),
            ManifestVisual(id="Table_1", type="table", pages=[3], description="T1"),
            ManifestVisual(id="Table_1", type="table", pages=[4, 3], description="T1 dup"),
        ],
        discarded_ids=[],
    )
    rec = reconcile_manifest(raw, placeholders, total_pages=10)
    assert [v.id for v in rec.visuals] == ["Fig._10", "Table_1"]
    assert rec.visuals[0].pages == [6, 7]
    assert rec.visuals[1].pages == [3, 4]


def test_get_required_pages_case_insensitive_and_batch_spec_shared(tmp_path: Path):
    phs = [
        PlaceholderItem(id="unlabeled_p4_1", caption="", page=4),
        PlaceholderItem(id="Table_1", caption="T1", page=1),
    ]
    assert get_required_pages(phs) == {1, 3, 4}

    artifacts = ArtifactManager("req_pages", base_dir=tmp_path)
    for p in (1, 3, 4):
        (artifacts.pages_dir / f"p{p}.png").write_bytes(b"png")
    cfg = IngestionConfig()
    spec = build_manifest_visuals_batch_spec(phs, artifacts, cfg)
    assert spec is not None
    assert "=== Page 3 ===" in spec.contents
    assert "=== Page 4 ===" in spec.contents


def test_stage2_manifest_silent_error_guard_retries_at_temp_05(tmp_path: Path):
    artifacts = ArtifactManager("stage2_guard", base_dir=tmp_path)
    (artifacts.pages_dir / "p1.png").write_bytes(b"png1")
    phs = [
        PlaceholderItem(id="Table_1", caption="Table 1", page=1),
        PlaceholderItem(id="Figure_1", caption="Fig 1", page=1),
    ]
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))
    mock_client = MagicMock()

    temps: list[float] = []

    def _gen(*args, **kwargs):
        temps.append(kwargs["temperature"])
        if len(temps) == 1:
            return Manifest(visuals=[], discarded_ids=[])
        return Manifest(
            visuals=[
                ManifestVisual(id="Table_1", type="table", pages=[1], description="MIC"),
                ManifestVisual(id="Figure_1", type="chart", pages=[1], description="Plot"),
            ],
            discarded_ids=[],
        )

    mock_client.generate_structured.side_effect = _gen
    manifest = run_manifest_visuals(mock_client, phs, artifacts, cfg, total_pages=1)
    assert temps == [0.0, 0.5]
    assert len(manifest.visuals) == 2
    mock_client.record_event.assert_called_with("silent_error_retries")

    # Direct check of detect_manifest_silent_error for high unexpected drop
    partial_raw = Manifest(
        visuals=[ManifestVisual(id="Ghost_1", type="table", pages=[1], description="X")],
        discarded_ids=[],
    )
    partial_rec = reconcile_manifest(partial_raw, phs)
    assert detect_manifest_silent_error(partial_raw, partial_rec, phs) is not None


def test_manifest_prompt_aligned_ids_and_dup_docs():
    prompt = _read_manifest_prompt()
    assert "Fig._10" in prompt
    assert "Fig.-10" not in prompt
    assert "*_dupN" in prompt
