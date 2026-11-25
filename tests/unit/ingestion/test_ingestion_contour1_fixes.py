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
    assert get_required_pages(phs) == {1, 3, 4, 5}
    assert get_required_pages(phs, total_pages=4) == {1, 3, 4}

    cont_phs = [
        PlaceholderItem(id="Table_1_cont", caption="Table 1 cont.", page=4),
        PlaceholderItem(id="Table_2_continued", caption="Table 2 continued", page=6),
        PlaceholderItem(id="Control_1", caption="Control", page=8),
    ]
    assert get_required_pages(cont_phs) == {3, 4, 5, 6, 8}

    artifacts = ArtifactManager("req_pages", base_dir=tmp_path)
    for p in (1, 3, 4, 5):
        (artifacts.pages_dir / f"p{p}.png").write_bytes(b"png")
    cfg = IngestionConfig()
    spec = build_manifest_visuals_batch_spec(phs, artifacts, cfg, total_pages=5)
    assert spec is not None
    assert "=== Page 3 ===" in spec.contents
    assert "=== Page 4 ===" in spec.contents
    assert "=== Page 5 ===" in spec.contents


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


# ---------------------------------------------------------------------------
# 5. EXP-01A hardening follow-up tests (render, guard, artifacts, batch, yaml)
# ---------------------------------------------------------------------------


def test_render_self_closing_visual_without_caption_map_and_oracle_notes(tmp_path: Path):
    from ae.ingestion.parsers.gemini.schemas import ChemicalEntity, MolecularParseResult
    from ae.ingestion.parsers.gemini.stages.render_markdown import (
        assemble_enriched_markdown,
        assemble_manifest_markdown,
        render_parse_body,
    )

    raw_md = 'Text before.\n\n<visual id="Unlabeled_p1_1" />\n\nText after.'
    manifest = Manifest(
        visuals=[
            ManifestVisual(
                id="Unlabeled_p1_1",
                type="molecular_structure",
                pages=[1],
                description="Structure",
            )
        ],
        discarded_ids=[],
    )
    artifacts = ArtifactManager("self_close_doc", base_dir=tmp_path)

    # Should not crash on match.group("inner_content") when caption_map omits placeholder
    manifest_md = assemble_manifest_markdown(raw_md, manifest, {}, artifacts)
    assert '<visual id="Unlabeled_p1_1" type="molecular_structure">' in manifest_md

    enriched_md = assemble_enriched_markdown(raw_md, manifest, {}, {})
    assert '<visual id="Unlabeled_p1_1" type="molecular_structure">' in enriched_md

    # Surface compact HTML comment for mismatch_flagged or overridden entities
    mol_res = MolecularParseResult(
        entities=[
            ChemicalEntity(
                label="15",
                smiles="CCO",
                provenance_status="mismatch_flagged",
                oracle_notes="Formula mismatch",
            ),
            ChemicalEntity(
                label="37",
                smiles="CCC",
                provenance_status="verified_exact",
                oracle_notes="overridden from prose IUPAC heading (stereo may be lost)",
            ),
            ChemicalEntity(label="1", smiles="C", provenance_status="verified_exact"),
        ]
    )
    rendered = render_parse_body(mol_res)
    assert "| **15** | `CCO` |" in rendered
    assert "| **37** | `CCC` |" in rendered
    assert "<!-- oracle_notes:" in rendered
    assert "15: mismatch_flagged (Formula mismatch)" in rendered
    assert "37: verified_exact (overridden from prose IUPAC heading" in rendered


def test_parse_text_case_insensitive_unlabeled_and_empty_retry_and_no_broken_cache(tmp_path: Path):
    from ae.core.exceptions import GeminiEmptyResponseError
    from ae.ingestion.parsers.gemini.stages.parse_text import parse_placeholders_from_markdown

    _, phs = parse_placeholders_from_markdown(
        '<visual id="unlabeled_a" />\n<visual id="UNLABELED_b" />',
        start_page=3,
    )
    assert [p.id for p in phs] == ["Unlabeled_p3_1", "Unlabeled_p3_2"]

    # Empty chunk response routes through validate_and_retry_chunk_markdown and recovers
    pdf_path = _make_pdf(tmp_path / "empty_retry.pdf", ["Page 1 scientific text."])
    artifacts = ArtifactManager("empty_retry", base_dir=tmp_path)
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))
    mock_client = MagicMock()
    mock_client.make_pdf_part.return_value = "pdf_part"
    temps: list[float] = []

    def _gen_empty_then_ok(contents, **kwargs):
        temps.append(kwargs["temperature"])
        if len(temps) == 1:
            return "   "
        return "Recovered text on retry."

    mock_client.generate_text.side_effect = _gen_empty_then_ok
    md, _ = run_chunked_parse_text(mock_client, pdf_path, artifacts, cfg)
    assert temps == [0.0, 0.5]
    assert "Recovered text on retry." in md
    assert (artifacts.chunks_dir / "chunk_1_1.md").exists()

    # Also recovers when GeminiEmptyResponseError is raised initially
    artifacts_exc = ArtifactManager("empty_exc_retry", base_dir=tmp_path)
    mock_client_exc = MagicMock()
    mock_client_exc.make_pdf_part.return_value = "pdf_part"
    mock_client_exc.generate_text.side_effect = [
        GeminiEmptyResponseError(stage="parse_text"),
        "Recovered from empty exception.",
    ]
    md_exc, _ = run_chunked_parse_text(mock_client_exc, pdf_path, artifacts_exc, cfg)
    assert "Recovered from empty exception." in md_exc

    # Unresolved malformed/unclosed visual tag after retry must NOT be cached
    artifacts_bad = ArtifactManager("bad_tag_nocache", base_dir=tmp_path)
    mock_client_bad = MagicMock()
    mock_client_bad.make_pdf_part.return_value = "pdf_part"
    mock_client_bad.generate_text.side_effect = [
        'Text\n<visual id="Fig_1">unclosed caption',
        'Retry\n<visual id="Bad Space" />',
    ]
    md_bad, _ = run_chunked_parse_text(mock_client_bad, pdf_path, artifacts_bad, cfg)
    assert "unclosed caption" in md_bad
    assert not (artifacts_bad.chunks_dir / "chunk_1_1.md").exists()


def test_artifacts_invalidate_from_classify_alias(tmp_path: Path):
    artifacts = ArtifactManager("alias_doc", base_dir=tmp_path)
    artifacts.save_text(artifacts.text_md_path, "Stage 1 text")
    artifacts.save_json(artifacts.placeholders_json_path, [])
    artifacts.save_json(artifacts.manifest_json_path, {"visuals": [], "discarded_ids": []})
    artifacts.save_json(artifacts.bboxes_json_path, {"pages": []})

    artifacts.invalidate_from_stage("classify")
    assert artifacts.has_parse_text_artifact() is True
    assert artifacts.has_manifest_artifact() is False
    assert artifacts.has_localize_artifact() is False


def test_silent_error_guard_stage1_stack_boilerplate_completeness_and_scan(tmp_path: Path):
    # Stack-based nested & out-of-order visual tags
    assert "stray_visual_close_tag" in (
        detect_parse_text_silent_error('</visual>\n<visual id="Fig_1">cap') or ""
    )
    assert "malformed_visual_tag" in (
        detect_parse_text_silent_error('<visual id="Fig_1">\n<visual id="Fig_2">x</visual>\n</visual>')
        or ""
    )
    assert "unclosed_visual_tag" in (
        detect_parse_text_silent_error('<visual id="Fig_1">unclosed') or ""
    )

    # Narrowed boilerplate detection: prose line starting with "references" is not skipped
    prose_with_refs_word = "\n".join(
        [
            "references to previous benzimidazole derivatives showed weak potency.",
            "Compound 15 exhibited MIC = 1.5 ug/mL against MRSA.",
            "Further SAR analysis confirmed the role of the C5 substituent.",
        ]
    )
    assert _is_likely_reference_or_boilerplate_page(prose_with_refs_word) is False

    # Mixed page with 4 content lines before References heading is not skipped
    mixed_early = "\n".join(
        [
            "Line 1 of discussion on compound 12.",
            "Line 2 of discussion on selectivity.",
            "Line 3 of discussion on toxicity.",
            "Line 4 concluding remarks.",
            "References",
            "[1] Doe J. 2024.",
        ]
    )
    assert _is_likely_reference_or_boilerplate_page(mixed_early) is False

    # Soft completeness check (< 0.15 ratio on >= 500 char non-boilerplate page without <visual>)
    dense_pdf = "Compound 15 showed strong antibacterial activity against S. aureus. " * 12
    assert len(dense_pdf) >= 500
    short_md = "Compound 15 showed activity."
    assert len(short_md) / len(dense_pdf) < 0.15
    assert "silent_omission" in (
        detect_parse_text_silent_error(short_md, pdf_page_text=dense_pdf) or ""
    )

    # Scan page without text layer (has_visual_content=True)
    assert "silent_omission" in (
        detect_parse_text_silent_error("<!-- omitted -->", pdf_page_text="", has_visual_content=True)
        or ""
    )
    assert (
        detect_parse_text_silent_error("<!-- omitted -->", pdf_page_text="", has_visual_content=False)
        is None
    )


def test_silent_error_guard_stage2_all_discarded_and_tightened_drop_and_warning(caplog):
    phs = [
        PlaceholderItem(id="Fig_1", caption="Photo 1", page=1),
        PlaceholderItem(id="Fig_2", caption="Photo 2", page=2),
    ]
    all_discarded = Manifest(visuals=[], discarded_ids=["Fig_1", "Fig_2"])
    rec_all_disc = reconcile_manifest(all_discarded, phs)
    assert detect_manifest_silent_error(all_discarded, rec_all_disc, phs) is None

    # Tightened high_unexpected_drop: 3/10 unaccounted (30%) triggers
    phs_10 = [PlaceholderItem(id=f"Fig_{i}", caption=f"F{i}", page=1) for i in range(10)]
    raw_drop = Manifest(
        visuals=[ManifestVisual(id=f"Fig_{i}", type="chart", pages=[1], description="C") for i in range(7)],
        discarded_ids=[],
    )
    with caplog.at_level("WARNING"):
        rec_drop = reconcile_manifest(raw_drop, phs_10)
    assert "Force-discarded 3 unaccounted placeholder(s)" in caplog.text
    err = detect_manifest_silent_error(raw_drop, rec_drop, phs_10)
    assert err is not None and "high_unexpected_drop" in err


def test_manifest_visuals_dup_pages_max_tokens_retry_error_reduction_and_sys_prompt(tmp_path: Path):
    from ae.ingestion.parsers.gemini.stages.manifest_visuals import validate_and_retry_manifest

    # 1. *_dupN includes p.page - 1
    dup_phs = [PlaceholderItem(id="Table_1_dup2", caption="Table 1 (cont.)", page=4)]
    assert get_required_pages(dup_phs) == {3, 4}

    # 2. validate_and_retry_manifest rejects retry that introduces unmatched errors
    artifacts = ArtifactManager("retry_err_red", base_dir=tmp_path)
    (artifacts.pages_dir / "p1.png").write_bytes(b"png1")
    phs = [
        PlaceholderItem(id="Table_1", caption="T1", page=1),
        PlaceholderItem(id="Table_2", caption="T2", page=1),
        PlaceholderItem(id="Table_3", caption="T3", page=1),
    ]
    orig_raw = Manifest(
        visuals=[ManifestVisual(id="Table_1", type="table", pages=[1], description="T1")],
        discarded_ids=[],
    )
    worse_retry_raw = Manifest(
        visuals=[
            ManifestVisual(id="Table_1", type="table", pages=[1], description="T1"),
            ManifestVisual(id="Table_2", type="table", pages=[1], description="T2"),
            ManifestVisual(id="Ghost_1", type="table", pages=[1], description="G1"),
            ManifestVisual(id="Ghost_2", type="table", pages=[1], description="G2"),
            ManifestVisual(id="Ghost_3", type="table", pages=[1], description="G3"),
        ],
        discarded_ids=[],
    )
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))
    mock_client = MagicMock()
    mock_client.generate_structured.return_value = worse_retry_raw
    chosen = validate_and_retry_manifest(mock_client, orig_raw, phs, artifacts, cfg, total_pages=1)
    assert [v.id for v in chosen.visuals] == ["Table_1"]

    # 3. run_manifest_visuals handles GeminiMaxTokensError and passes system_instruction
    artifacts_mt = ArtifactManager("manifest_max_tok", base_dir=tmp_path)
    (artifacts_mt.pages_dir / "p1.png").write_bytes(b"png1")
    calls: list[dict] = []

    def _gen_mt(contents, **kwargs):
        calls.append({"contents": contents, **kwargs})
        if len(calls) == 1:
            raise GeminiMaxTokensError(stage="classify")
        return Manifest(
            visuals=[ManifestVisual(id="Table_1", type="table", pages=[1], description="T1")],
            discarded_ids=[],
        )

    mock_client_mt = MagicMock()
    mock_client_mt.generate_structured.side_effect = _gen_mt
    res_mt = run_manifest_visuals(
        mock_client_mt,
        [PlaceholderItem(id="Table_1", caption="T1", page=1)],
        artifacts_mt,
        cfg,
        total_pages=1,
    )
    assert [v.id for v in res_mt.visuals] == ["Table_1"]
    assert [c["temperature"] for c in calls] == [0.0, 0.5]
    assert "system_instruction" in calls[0] and "Filtering" in calls[0]["system_instruction"]
    assert all("Filtering" not in str(item) for item in calls[0]["contents"])

    # Batch spec also puts static prompt into system_instruction
    artifacts_bs = ArtifactManager("batch_sys", base_dir=tmp_path)
    (artifacts_bs.pages_dir / "p1.png").write_bytes(b"png1")
    spec = build_manifest_visuals_batch_spec(
        [PlaceholderItem(id="Table_1", caption="T1", page=1)], artifacts_bs, cfg
    )
    assert spec is not None
    assert spec.system_instruction and "Filtering" in spec.system_instruction
    assert all("Filtering" not in str(item) for item in spec.contents)


def test_batch_error_type_and_parser_direct_max_tokens_retry_and_no_empty_cache(
    tmp_path: Path, monkeypatch
):
    from ae.core.exceptions import GeminiEmptyResponseError

    # 1. client.py stores error_type on BatchItemResult for MAX_TOKENS
    client = GeminiClient(GeminiParserConfig(), api_key="dummy")
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "MAX_TOKENS"
    mock_candidate.content.parts = [MagicMock()]
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None
    mock_resp.usage_metadata = None
    mock_ir = MagicMock(error=None, response=mock_resp)
    mock_job = MagicMock(
        state="JOB_STATE_SUCCEEDED",
        dest=MagicMock(inlined_responses=[mock_ir], file_name=None),
    )
    client._client = MagicMock()
    client._client.batches.create.return_value = mock_job

    from ae.ingestion.parsers.gemini.client import BatchRequestSpec

    batch_out = client.run_batch_requests(
        [BatchRequestSpec(key="k1", contents=["hello"], stage_name="parse_text")]
    )
    assert batch_out["k1"].error_type == "GeminiMaxTokensError"

    # 2. parser.parse_batch retries GeminiMaxTokensError directly at temperature=0.5 (no temp=0.0 sync call)
    pdf_path = _make_pdf(tmp_path / "batch_mt.pdf", ["Page 1 scientific text."])
    cfg = IngestionConfig(
        ingestion_dir=tmp_path / "ing1",
        execution=ExecutionConfig(use_batch_api=True, silent_error_guard=True),
    )
    monkeypatch.setattr(GeminiParser, "_resolve_api_key", staticmethod(lambda: "dummy"))
    parser = GeminiParser(cfg)
    mock_client = MagicMock()
    mock_client.make_pdf_part.return_value = "pdf_part"
    mock_client.get_all_stage_usage.return_value = {}
    mock_client.get_scope_counters.return_value = {}

    sync_temps: list[float] = []
    mock_client.generate_text.side_effect = lambda *a, **kw: (
        sync_temps.append(kw["temperature"]) or "Recovered directly at temp 0.5."
    )
    mock_client.run_batch_requests.side_effect = lambda specs, **kw: {
        s.key: (
            BatchItemResult(key=s.key, error="MAX_TOKENS", error_type="GeminiMaxTokensError")
            if s.stage_name == "parse_text"
            else BatchItemResult(key=s.key, parsed=Manifest(visuals=[], discarded_ids=[]))
        )
        for s in specs
    }
    parser._client = mock_client
    out = parser.parse_batch([pdf_path])
    assert sync_temps == [0.5]
    assert "Recovered directly at temp 0.5." in out[pdf_path.resolve()]

    # 3. parser.parse_batch with use_guard=False must NOT cache empty response
    pdf_empty = _make_pdf(tmp_path / "batch_empty_noguard.pdf", ["Page 1 text."])
    cfg_noguard = IngestionConfig(
        ingestion_dir=tmp_path / "ing2",
        execution=ExecutionConfig(use_batch_api=True, silent_error_guard=False),
    )
    parser2 = GeminiParser(cfg_noguard)
    mock_client2 = MagicMock()
    mock_client2.make_pdf_part.return_value = "pdf_part"
    mock_client2.generate_text.return_value = "   "
    mock_client2.run_batch_requests.return_value = {
        "batch_empty_noguard__chunk_1_1": BatchItemResult(key="batch_empty_noguard__chunk_1_1", text="")
    }
    parser2._client = mock_client2
    with pytest.raises(GeminiEmptyResponseError):
        parser2.parse_batch([pdf_empty])
    arts2 = ArtifactManager("batch_empty_noguard", base_dir=cfg_noguard.ingestion_dir)
    assert not (arts2.chunks_dir / "chunk_1_1.md").exists()


def test_ingestion_yaml_classify_thinking_level_low():
    import yaml

    yaml_path = Path(__file__).resolve().parents[3] / "config" / "ingestion.yaml"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg = IngestionConfig.model_validate(raw["ingestion"])
    assert cfg.stages.classify.model.thinking_level == "low"
    assert raw["ingestion"]["stages"]["classify"]["render_page_max_size"] == 768
    assert cfg.stages.localize.model.max_output_tokens == 16384
    assert cfg.stages.decompose.model.max_output_tokens == 32768


def test_parse_text_repair_id_constraints_dup_collision_empty_pages_and_max_tokens(
    tmp_path: Path,
):
    from ae.core.exceptions import GeminiEmptyResponseError
    from ae.ingestion.parsers.gemini.stages.parse_text import (
        _extract_pdf_pages_metadata,
        parse_placeholders_from_markdown,
        repair_simple_visual_tags,
    )

    # 1. Narrowed visual ID regex rejects '.' and '..' and non-ASCII
    _, dot_phs = parse_placeholders_from_markdown(
        '<visual id="." />\n<visual id=".." />\n<visual id="Табл_1" />\n<visual id="Fig_1.a" />'
    )
    assert [p.id for p in dot_phs] == ["Fig_1.a"]

    # 2. Collision prevention for _dupN when Fig_1_dup2 already exists
    md_col = (
        '<visual id="Fig_1_dup2" />\n'
        '<visual id="Fig_1" />\n'
        '<visual id="Fig_1" />\n'
    )
    clean_col, phs_col = parse_placeholders_from_markdown(md_col)
    assert [p.id for p in phs_col] == ["Fig_1_dup2", "Fig_1", "Fig_1_dup3"]
    assert '<visual id="Fig_1_dup3" />' in clean_col

    # 3. Empty page marker preserves physical page indexing via <!-- page_break -->
    arts_empty = ArtifactManager("empty_page_align", base_dir=tmp_path)
    chunks = [
        (1, 1, tmp_path / "c1.pdf"),
        (2, 2, tmp_path / "c2.pdf"),
        (3, 3, tmp_path / "c3.pdf"),
    ]
    clean_aligned, _ = assemble_parse_text_chunks(
        chunks, ["Page 1 prose.", "<!-- page_break -->", "Page 3 prose."], arts_empty
    )
    pages_split = clean_aligned.split("<!-- page_break -->")
    assert len(pages_split) == 3
    assert "<!-- omitted -->" in pages_split[1]

    # All-empty chunks still raise GeminiEmptyResponseError
    with pytest.raises(GeminiEmptyResponseError):
        assemble_parse_text_chunks(
            [(1, 1, tmp_path / "c1.pdf")], ["<!-- page_break -->"], arts_empty
        )

    # 4. Deterministic repair fixes simple malformed tags (extra attrs, casing, truncated EOL)
    repaired = repair_simple_visual_tags(
        'Text\n<VISUAL type="table" id="Table_1" />\n<visual id="Fig_2"\n'
    )
    assert detect_parse_text_silent_error(repaired) is None
    assert '<visual id="Table_1" />' in repaired
    assert '<visual id="Fig_2" />' in repaired

    # Unresolved visual tag error does NOT persist text.md / placeholders.json
    pdf_path = _make_pdf(tmp_path / "unres.pdf", ["Page 1 scientific text."])
    arts_unres = ArtifactManager("unres_no_persist", base_dir=tmp_path)
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))
    mock_client_unres = MagicMock()
    mock_client_unres.make_pdf_part.return_value = "pdf_part"
    mock_client_unres.generate_text.side_effect = [
        'Text\n<visual id="Fig_1">unclosed caption',
        'Retry\n<visual id="Bad Space" />',
    ]
    run_chunked_parse_text(mock_client_unres, pdf_path, arts_unres, cfg)
    assert arts_unres.has_parse_text_artifact() is False

    # 5. Initial GeminiEmptyResponseError followed by retry GeminiMaxTokensError must NOT mask MAX_TOKENS
    arts_mt = ArtifactManager("empty_then_max_tok", base_dir=tmp_path)
    mock_client_mt = MagicMock()
    mock_client_mt.make_pdf_part.return_value = "pdf_part"
    mock_client_mt.generate_text.side_effect = [
        GeminiEmptyResponseError(stage="parse_text"),
        GeminiMaxTokensError(stage="parse_text"),
    ]
    with pytest.raises(GeminiMaxTokensError):
        run_chunked_parse_text(mock_client_mt, pdf_path, arts_mt, cfg)

    # 6. Single-pass metadata extraction
    texts, flags = _extract_pdf_pages_metadata(pdf_path)
    assert 1 in texts and "Page 1 scientific text." in texts[1]
    assert flags[1] is False


def test_silent_error_guard_prose_table_substantive_after_refs_and_1_drop():
    # 1. Dropped prose on a table page is caught even with valid <visual>
    dense_prose = (
        "Compound 15 demonstrated strong antibacterial activity against MRSA strains in vitro. " * 12
    )
    err_table_prose = detect_parse_text_silent_error(
        '<visual id="Table_1">Table 1. MIC values.</visual>',
        pdf_page_text=dense_prose,
        has_visual_content=True,
    )
    assert err_table_prose is not None and "silent_omission" in err_table_prose

    # Pure table cell dump without prose sentences is NOT flagged
    pure_table_dump = "\n".join(["1a | 0.25 | 4.0 | 8.0"] * 40)
    assert (
        detect_parse_text_silent_error(
            '<visual id="Table_1">Table 1.</visual>',
            pdf_page_text=pure_table_dump,
            has_visual_content=True,
        )
        is None
    )

    # 2. Substantive section after References/Acknowledgments on same page is NOT boilerplate
    refs_then_methods = "\n".join(
        [
            "### References",
            "[1] Smith J. J. Med. Chem. 2020, 63, 100-110.",
            "[2] Doe A. Eur. J. Med. Chem. 2021, 210, 112-120.",
            "### Methods",
            "General synthesis procedure for benzimidazole derivatives.",
            "All reactions were carried out under inert nitrogen atmosphere.",
        ]
    )
    assert _is_likely_reference_or_boilerplate_page(refs_then_methods) is False

    # 3. Stage 2 manifest silent error triggers even on 1 unaccounted out of 10 placeholders
    phs_10 = [PlaceholderItem(id=f"Fig_{i}", caption=f"F{i}", page=1) for i in range(10)]
    raw_1_drop = Manifest(
        visuals=[ManifestVisual(id=f"Fig_{i}", type="chart", pages=[1], description="C") for i in range(9)],
        discarded_ids=[],
    )
    rec_1_drop = reconcile_manifest(raw_1_drop, phs_10)
    err_1_drop = detect_manifest_silent_error(raw_1_drop, rec_1_drop, phs_10)
    assert err_1_drop is not None and "high_unexpected_drop: 1/10" in err_1_drop


def test_manifest_merged_from_reconcile_render_and_max_tokens_retry_fail(tmp_path: Path):
    from ae.ingestion.parsers.gemini.stages.manifest_visuals import validate_and_retry_manifest
    from ae.ingestion.parsers.gemini.stages.render_markdown import (
        assemble_enriched_markdown,
        assemble_manifest_markdown,
    )

    # 1. merged_from reconciliation and clean removal of continuation tag/caption
    phs = [
        PlaceholderItem(id="Table_1", caption="Table 1. MIC data.", page=3),
        PlaceholderItem(id="Table_1_dup2", caption="Table 1 (continued)", page=4),
        PlaceholderItem(id="Fig_1", caption="Photo of crystals", page=5),
    ]
    raw = Manifest(
        visuals=[
            ManifestVisual(
                id="Table_1",
                type="table",
                pages=[3, 4],
                description="MIC data for 1a-1f.",
                merged_from=["Table_1_dup2"],
            )
        ],
        discarded_ids=["Fig_1"],
    )
    rec = reconcile_manifest(raw, phs, total_pages=6)
    assert rec.visuals[0].merged_from == ["Table_1_dup2"]
    assert rec.discarded_ids == ["Fig_1"]
    assert detect_manifest_silent_error(raw, rec, phs) is None

    # 1b. If model omits merged placeholder's page from pages, reconcile_manifest must union it
    raw_omit_page = Manifest(
        visuals=[
            ManifestVisual(
                id="Table_1",
                type="table",
                pages=[3],
                description="MIC data for 1a-1f.",
                merged_from=["Table_1_dup2"],
            )
        ],
        discarded_ids=["Fig_1"],
    )
    rec_omit_page = reconcile_manifest(raw_omit_page, phs, total_pages=6)
    assert rec_omit_page.visuals[0].pages == [3, 4]

    raw_md = (
        "Intro text.\n\n"
        '<visual id="Table_1">\nTable 1. MIC data.\n</visual>\n\n'
        '<visual id="Table_1_dup2">\nTable 1 (continued)\n</visual>\n\n'
        '<visual id="Fig_1">\nPhoto of crystals\n</visual>'
    )
    artifacts = ArtifactManager("merged_render", base_dir=tmp_path)
    manifest_md = assemble_manifest_markdown(
        raw_md, rec, {p.id: p.caption for p in phs}, artifacts
    )
    assert "Table 1 (continued)" not in manifest_md
    assert "Table_1_dup2" not in manifest_md
    assert "Photo of crystals" in manifest_md

    enriched_md = assemble_enriched_markdown(
        raw_md, rec, {}, {p.id: p.caption for p in phs}
    )
    assert "Table 1 (continued)" not in enriched_md

    # 2. validate_and_retry_manifest raises GeminiMaxTokensError when retry after MAX_TOKENS still fails validation
    (artifacts.pages_dir / "p3.png").write_bytes(b"png")
    (artifacts.pages_dir / "p4.png").write_bytes(b"png")
    (artifacts.pages_dir / "p5.png").write_bytes(b"png")
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))
    mock_client = MagicMock()
    mock_client.generate_structured.return_value = Manifest(visuals=[], discarded_ids=[])
    with pytest.raises(GeminiMaxTokensError):
        validate_and_retry_manifest(
            mock_client,
            None,
            phs,
            artifacts,
            cfg,
            total_pages=6,
            initial_max_tokens_exc=GeminiMaxTokensError(stage="classify"),
        )


def test_parse_batch_cleans_up_on_prep_failure_and_no_second_render(tmp_path: Path, monkeypatch):
    pdf1 = _make_pdf(tmp_path / "doc1.pdf", ["Doc 1 page 1"])
    pdf2 = _make_pdf(tmp_path / "doc2.pdf", ["Doc 2 page 1"])
    cfg = IngestionConfig(
        ingestion_dir=tmp_path / "ing",
        execution=ExecutionConfig(use_batch_api=True, use_inline_pdf=False),
    )
    monkeypatch.setattr(GeminiParser, "_resolve_api_key", staticmethod(lambda: "dummy"))
    parser = GeminiParser(cfg)
    mock_client = MagicMock()
    handle1 = MagicMock(name="h1")
    mock_client.upload_file.side_effect = [handle1, RuntimeError("upload boom")]
    parser._client = mock_client

    with pytest.raises(RuntimeError, match="upload boom"):
        parser.parse_batch([pdf1, pdf2])

    mock_client.delete_file.assert_called_once_with(handle1)
    assert not (cfg.ingestion_dir / "doc1" / "tmp_chunks").exists()
    assert not (cfg.ingestion_dir / "doc2" / "tmp_chunks").exists()

    # Verify parse() invokes run_render_pages_with_visuals only once (for placeholders, not manifest)
    cfg_sync = IngestionConfig(ingestion_dir=tmp_path / "ing_sync")
    parser_sync = GeminiParser(cfg_sync)
    parser_sync._client = MagicMock()
    monkeypatch.setattr(
        "ae.ingestion.parsers.gemini.parser.run_parse_text",
        lambda *a, **kw: ('<visual id="Table_1">T1</visual>', [PlaceholderItem(id="Table_1", caption="T1", page=1)]),
    )
    render_calls = {"n": 0}
    monkeypatch.setattr(
        "ae.ingestion.parsers.gemini.parser.run_render_pages_with_visuals",
        lambda *a, **kw: render_calls.__setitem__("n", render_calls["n"] + 1) or {1},
    )
    monkeypatch.setattr(
        "ae.ingestion.parsers.gemini.parser.run_manifest_visuals",
        lambda *a, **kw: Manifest(visuals=[ManifestVisual(id="Table_1", type="table", pages=[1, 2], description="T1")]),
    )
    monkeypatch.setattr(
        "ae.ingestion.parsers.gemini.parser.run_render_manifest",
        lambda *a, **kw: "final md",
    )
    out = parser_sync.parse(pdf1)
    assert out == "final md"
    assert render_calls["n"] == 1
