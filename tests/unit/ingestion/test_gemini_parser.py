"""Unit tests for Ingestion v2 Gemini pipeline (Stages 1 through 8)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pymupdf as fitz
import pytest
from PIL import Image as PILImage

from ae.core.config.optimization_settings import (
    GeminiDefaultsConfig,
    GeminiParserConfig,
    IngestionConfig,
)
from ae.core.exceptions import (
    GeminiEmptyResponseError,
    GeminiMaxTokensError,
    GeminiRecitationError,
    GeminiSafetyError,
)
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.parser import GeminiParser
from ae.ingestion.parsers.gemini.schemas import (
    ChartParseResult,
    ChemicalEntity,
    Manifest,
    ManifestVisual,
    MolecularParseResult,
    PageLocalization,
    PlaceholderItem,
    TableParseResult,
    VisualBox,
)
from ae.ingestion.parsers.gemini.stages.manifest_visuals import (
    reconcile_manifest,
    run_classify,
    run_render_pages,
)
from ae.ingestion.parsers.gemini.stages.parse_visuals import (
    compose_parse_prompt,
    parse_single_visual,
    run_parse_visuals,
)
from ae.ingestion.parsers.gemini.stages.rdkit_validation import (
    SmilesCorrectionItem,
    SmilesCorrectionResponse,
    validate_and_canonicalize,
    validate_and_correct_entities,
)
from ae.ingestion.parsers.gemini.stages.render_markdown import (
    assemble_enriched_markdown,
    render_placeholder_block,
)
from ae.ingestion.parsers.gemini.stages.segment_and_crop_visuals_elements import (
    compute_crop_rect_and_dpi,
    run_crop,
    run_localize_visuals,
    validate_box,
)
from ae.ingestion.parsers.gemini.stages.parse_text import (
    compose_parse_text_prompt,
    parse_placeholders_from_markdown,
    run_parse_text,
    split_pdf_into_chunks,
)


@pytest.fixture
def mock_config():
    return IngestionConfig(
        gemini=GeminiParserConfig(
            defaults=GeminiDefaultsConfig(
                model="gemini-3.8-flash",
            )
        )
    )


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """Create a minimal 2-page synthetic PDF using PyMuPDF."""
    pdf_path = tmp_path / "test_doc.pdf"
    doc = fitz.open()

    # Page 1: standard letter / A4 size (600 x 800 pt)
    page1 = doc.new_page(width=600, height=800)
    page1.draw_rect(fitz.Rect(50, 50, 250, 250), color=(0, 0, 1), fill=(0.9, 0.9, 1))
    page1.insert_text(fitz.Point(60, 100), "Figure 1 Chart on Page 1")

    # Page 2
    page2 = doc.new_page(width=600, height=800)
    page2.draw_rect(fitz.Rect(50, 50, 550, 400), color=(1, 0, 0), fill=(1, 0.9, 0.9))
    page2.insert_text(fitz.Point(60, 100), "Table 1 on Page 2")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ===================================================================
# 1. Transcribe tests
# ===================================================================


def test_transcribe_placeholder_parsing_and_deduplication():
    raw_md = (
        "# Introduction\n\n"
        "Here is the first figure:\n\n"
        '<visual id="Figure_1">\n'
        "**Figure 1.** Comparison of reaction rates.\n"
        "</visual>\n\n"
        "And another block without caption:\n\n"
        '<visual id="Unlabeled_p1_1" />\n\n'
        "Synthesis of intermediate compound.\n\n"
        "And a duplicate ID tag:\n\n"
        '<visual id="Figure_1">\n'
        "**Figure 1.** Second part of the comparison.\n"
        "</visual>\n"
    )

    clean_md, placeholders = parse_placeholders_from_markdown(raw_md)

    assert len(placeholders) == 3
    assert placeholders[0].id == "Figure_1"
    assert "Comparison of reaction rates" in placeholders[0].caption

    assert placeholders[1].id == "Unlabeled_p1_1"
    assert placeholders[1].caption == ""

    # Duplicate should have received _dup2 suffix
    assert placeholders[2].id == "Figure_1_dup2"
    assert '<visual id="Figure_1_dup2">' in clean_md


def test_transcribe_complex_and_multiline_captions():
    raw_md = (
        "Text before.\n\n"
        '<visual id="Table_S1">\n'
        "**Table S1.** Multi-line title.\n"
        "Footnote *a*: determined at 25 °C.\n"
        "Footnote *b*: average of 3 runs.\n"
        "</visual>\n\n"
        '<visual id="Scheme_1" />\n\n'
        "Text after."
    )
    clean_md, placeholders = parse_placeholders_from_markdown(raw_md)
    assert len(placeholders) == 2
    assert placeholders[0].id == "Table_S1"
    assert "Multi-line title" in placeholders[0].caption
    assert "Footnote *b*: average of 3 runs." in placeholders[0].caption
    assert placeholders[1].id == "Scheme_1"
    assert placeholders[1].caption == ""


# ===================================================================
# 2. Classify tests
# ===================================================================


def test_classify_manifest_reconciliation():
    expected = [
        PlaceholderItem(id="Figure_1", caption="Caption 1", page=1),
        PlaceholderItem(id="Table_1", caption="Caption table", page=2),
        PlaceholderItem(id="Figure_2", caption="Caption omitted", page=3),
    ]

    model_output = Manifest(
        visuals=[
            ManifestVisual(
                id="Figure_1",
                type="chart",
                pages=[1],
                description="Bar plot",
            ),
            ManifestVisual(
                id="Table_1",
                type="table",
                contains_types=["molecular_structure"],
                pages=[2],
                description="Data table",
            ),
            ManifestVisual(
                id="Unknown_Figure_99",
                type="chart",
                pages=[1],
                description="Ghost",
            ),
        ],
        discarded_ids=[],
    )

    reconciled = reconcile_manifest(model_output, expected)

    vis_map = {v.id: v for v in reconciled.visuals}
    assert "Unknown_Figure_99" not in vis_map
    assert "Figure_1" in vis_map
    assert "Table_1" in vis_map
    assert "Figure_2" in reconciled.discarded_ids


# ===================================================================
# 3. Localize tests
# ===================================================================


def test_localize_box_validation():
    valid = VisualBox(id="Figure_1", box_2d=[100, 200, 500, 800], found=True)
    assert validate_box(valid).found is True

    # ymin >= ymax -> invalid
    inv1 = VisualBox(id="Figure_2", box_2d=[600, 200, 500, 800], found=True)
    assert validate_box(inv1).found is False

    # out of bounds (> 1000)
    inv2 = VisualBox(id="Figure_3", box_2d=[0, 0, 1200, 800], found=True)
    assert validate_box(inv2).found is False

    # malformed length
    inv3 = VisualBox(id="Figure_4", box_2d=[0, 0, 500], found=True)
    assert validate_box(inv3).found is False


# ===================================================================
# 4. Crop tests
# ===================================================================


def test_crop_math_and_dpi_scaling(mock_config):
    crop_cfg = mock_config.gemini.crop
    page_rect = fitz.Rect(0, 0, 600, 800)  # width=600, height=800

    # box_2d: 0..1000 -> [ymin, xmin, ymax, xmax] = [100, 100, 300, 300]
    # y0=80, x0=60, y1=240, x1=180 -> width=120pt, height=160pt
    clip, dpi = compute_crop_rect_and_dpi([100, 100, 300, 300], page_rect, crop_cfg)

    # Padding added (0.05 * 600 = 30pt, 0.05 * 800 = 40pt)
    assert clip.x0 == 60 - 30
    assert clip.y0 == 80 - 40
    assert clip.x1 == 180 + 30
    assert clip.y1 == 240 + 40

    # Long side pt = 160 + 80 = 240 pt
    # min_long_side_px = 2000 -> 2000 * 72 / 240 = 600 DPI
    assert dpi == 600

    # Test upscale disabled
    mock_config.gemini.crop.upscale.enabled = False
    _, base_dpi = compute_crop_rect_and_dpi(
        [100, 100, 300, 300], page_rect, mock_config.gemini.crop
    )
    assert base_dpi == 300


def test_crop_synthetic_pdf(tmp_path: Path, synthetic_pdf: Path, mock_config):
    artifacts = ArtifactManager("test_doc", base_dir=tmp_path)
    manifest = Manifest(
        visuals=[
            ManifestVisual(
                id="Figure_1",
                type="chart",
                pages=[1],
                description="Chart",
            ),
        ],
    )
    locs = {
        1: PageLocalization(
            page=1,
            boxes=[VisualBox(id="Figure_1", box_2d=[100, 100, 400, 500], found=True)],
        ),
    }

    crops = run_crop(synthetic_pdf, manifest, locs, artifacts, mock_config)
    assert "Figure_1" in crops
    assert len(crops["Figure_1"]) == 1
    assert crops["Figure_1"][0].exists()


# ===================================================================
# 5. Extract tests
# ===================================================================


def test_parse_prompt_composition():
    # Table visual
    table_vis = ManifestVisual(
        id="Table_1",
        type="table",
        pages=[1, 2],
        description="Bioactivity assay",
    )
    prompt, schema = compose_parse_prompt(table_vis, caption="Table of compounds", crops_count=2)
    assert schema == TableParseResult
    assert "Bioactivity assay" in prompt
    assert "Table of compounds" in prompt
    assert "sequential page fragments of the SAME multi-page table" in prompt

    # Molecule visual
    mol_vis = ManifestVisual(
        id="Figure_2",
        type="molecular_structure",
        pages=[1],
        description="Synthesis scheme",
    )
    prompt_mol, schema_mol = compose_parse_prompt(mol_vis, caption="Scheme 1", crops_count=1)
    assert schema_mol == MolecularParseResult
    assert "Markush" in prompt_mol


def test_parse_handles_client_failure(tmp_path: Path, mock_config):
    mock_client = MagicMock()
    mock_client.generate_structured.side_effect = RuntimeError("API quota exceeded")

    vis = ManifestVisual(
        id="Table_1",
        type="table",
        pages=[1],
        description="Table",
    )
    fake_crop = tmp_path / "crop.png"
    fake_crop.write_bytes(b"PNG")

    res = parse_single_visual(mock_client, vis, [fake_crop], "Caption", mock_config)
    assert res.status == "failed"


def test_parse_single_visual_molecular_rdkit_validation(tmp_path: Path, mock_config):
    mock_client = MagicMock()
    mock_client.generate_structured.return_value = MolecularParseResult(
        entities=[ChemicalEntity(label="1", smiles="c1ccccc1", role="target_product")]
    )

    vis = ManifestVisual(
        id="Scheme_1",
        type="molecular_structure",
        pages=[1],
        description="Scheme with molecules",
    )
    fake_crop = tmp_path / "crop.png"
    fake_crop.write_bytes(b"PNG")

    res = parse_single_visual(mock_client, vis, [fake_crop], "Caption", mock_config)
    assert isinstance(res, MolecularParseResult)
    assert isinstance(res.entities, list)
    assert len(res.entities) == 1
    assert res.entities[0].smiles_valid is True
    assert res.entities[0].smiles == "c1ccccc1"


def test_run_parse_visuals_preserves_context_on_retry(tmp_path: Path, mock_config, monkeypatch):
    artifacts = ArtifactManager(tmp_path, "doc_retry_ctx")
    artifacts.save_text(
        artifacts.text_md_path,
        "Intro paragraph mentioning Table 11 results.\n\n[[VISUAL_PLACEHOLDER: Table_11]]\n\nFollowing discussion.",
    )
    artifacts.save_json(
        artifacts.placeholders_json_path,
        [{"id": "Table_11", "caption": "Table 11. MIC values"}],
    )

    vis = ManifestVisual(
        id="Table_11",
        type="table",
        pages=[1],
        description="MIC values table",
    )
    manifest = Manifest(doc_id="doc_retry_ctx", total_pages=1, visuals=[vis])
    crop = tmp_path / "Table_11.png"
    crop.write_bytes(b"PNG")

    seen_contexts: list[str] = []

    def _mock_parse_single(client, visual, crop_paths, caption, config, context=""):
        seen_contexts.append(context)
        if len(seen_contexts) == 1:
            raise RuntimeError("transient error")
        return TableParseResult(status="success", content="<table><tr><td>ok</td></tr></table>")

    monkeypatch.setattr("ae.ingestion.parsers.gemini.stages.parse_visuals.parse_single_visual", _mock_parse_single)
    monkeypatch.setattr("ae.ingestion.parsers.gemini.stages.parse_visuals.time.sleep", lambda _: None)

    results = run_parse_visuals(MagicMock(), manifest, {"Table_11": [crop]}, artifacts, mock_config)

    assert len(seen_contexts) == 2
    assert seen_contexts[0] != ""
    assert seen_contexts[1] == seen_contexts[0]
    assert results["Table_11"].status == "success"


# ===================================================================
# 6. RDKit validation tests
# ===================================================================


def test_rdkit_validation_and_canonicalization():
    valid, can, err = validate_and_canonicalize("c1ccccc1")
    assert valid is True
    assert can == "c1ccccc1"
    assert err is None

    # Invalid SMILES
    valid2, orig, err2 = validate_and_canonicalize("InvalidC@@@Bond")
    assert valid2 is False
    assert err2 is not None


def test_rdkit_self_correction_loop(mock_config):
    mock_client = MagicMock()
    # Mock self-correction returning valid canonical SMILES
    mock_client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="1a", corrected_smiles="CC(=O)O")],
    )

    entities = [
        ChemicalEntity(label="1a", smiles="InvalidSMILES@@", role="target_product"),
        ChemicalEntity(label="1b", smiles="CCO", role="starting_material"),
    ]

    updated, replacements = validate_and_correct_entities(
        entities,
        crop_paths=[],
        client=mock_client,
        config=mock_config,
    )

    assert updated[0].smiles_valid is True
    assert updated[0].smiles == "CC(=O)O"
    assert "InvalidSMILES@@" in replacements


def test_rdkit_correction_exhausted(mock_config):
    mock_client = MagicMock()
    # Mock keeps returning invalid SMILES
    mock_client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="1a", corrected_smiles="StillBad!!!")],
    )

    entities = [
        ChemicalEntity(label="1a", smiles="BadSmiles#", role="target_product"),
    ]

    updated, _ = validate_and_correct_entities(
        entities,
        crop_paths=[],
        client=mock_client,
        config=mock_config,
    )

    # Entity remains in output with smiles_valid=False
    assert len(updated) == 1
    assert updated[0].smiles_valid is False
    assert updated[0].rdkit_error is not None


# ===================================================================
# 7. Render Markdown tests
# ===================================================================


def test_render_placeholder_blocks():
    # 1. Discarded placeholder -> keep caption only
    disc_block = render_placeholder_block("Figure_1", "**Figure 1.** Microscopy photo.", [], {})
    assert disc_block == "**Figure 1.** Microscopy photo."
    assert "<visual" not in disc_block

    # 2. Multi-panel block
    manifest_visuals = [
        ManifestVisual(
            id="Figure_2",
            type="chart",
            pages=[1],
            description="Bar plot",
        ),
    ]
    extractions = {
        "Figure_2": ChartParseResult(
            status="success",
            content="| Category | Value |\n| :--- | :--- |\n| A | 10 |\n| B | 20 |",
        ),
    }

    rendered = render_placeholder_block(
        "Figure_2", "**Figure 2.** Overview.", manifest_visuals, extractions
    )
    assert '<visual id="Figure_2" type="chart">' in rendered
    assert "| Category | Value |" in rendered
    assert "</visual>" in rendered


def test_assemble_enriched_markdown_failed_and_not_localized():
    raw_md = (
        "Some text before.\n\n"
        '<visual id="Table_1">\n'
        "**Table 1.** Bioassay data.\n"
        "</visual>\n\n"
        "Text after."
    )
    manifest = Manifest(
        visuals=[
            ManifestVisual(
                id="Table_1",
                type="table",
                pages=[1],
                description="Data",
            ),
        ],
    )
    # Failed extraction
    extractions = {
        "Table_1": TableParseResult(status="failed", content=""),
    }
    caption_map = {"Table_1": "**Table 1.** Bioassay data."}

    out_md = assemble_enriched_markdown(raw_md, manifest, extractions, caption_map)
    assert "*[Parsing failed]*" in out_md
    assert "**Table 1.** Bioassay data." in out_md
    assert "Some text before." in out_md
    assert "Text after." in out_md


def test_assemble_enriched_markdown_strictly_preserves_adjacent_text():
    raw_md = (
        "Paragraph 1 right before visual.\n"
        '<visual id="Fig_X">**Fig X.** Caption</visual>\n'
        "Paragraph 2 immediately following without blank line."
    )
    manifest = Manifest(
        visuals=[],
        discarded_ids=["Fig_X"],
    )
    # Discarded object returns caption only
    out_md = assemble_enriched_markdown(raw_md, manifest, {}, {})
    assert "Paragraph 1 right before visual." in out_md
    assert "**Fig X.** Caption" in out_md
    assert "Paragraph 2 immediately following without blank line." in out_md


# ===================================================================
# 8. Artifacts and Resumption tests
# ===================================================================


def test_artifacts_invalidation_and_resumption(tmp_path: Path, mock_config):
    artifacts = ArtifactManager("doc_1", base_dir=tmp_path)
    assert artifacts.rendered_pages_with_visuals_dir.name == "rendered_pages_with_visuals"
    assert artifacts.manifest_enriched_text_path.name == "manifest_enriched_text.md"
    assert artifacts.summary_json_path.parent == artifacts.debug_dir

    # Simulate artifacts
    artifacts.save_text(artifacts.text_md_path, "Text")
    artifacts.save_json(artifacts.placeholders_json_path, [])
    artifacts.save_json(artifacts.manifest_json_path, {"placeholders": []})
    artifacts.save_json(artifacts.bboxes_json_path, {"pages": []})
    (artifacts.crops_dir / "obj1__p1.png").write_bytes(b"PNG")
    artifacts.save_json(artifacts.parsed_visuals_dir / "obj1.json", {"status": "success"})
    artifacts.save_text(artifacts.document_md_path, "Enriched")
    artifacts.save_json(artifacts.summary_json_path, {})

    assert artifacts.has_parse_text_artifact() is True
    assert artifacts.has_manifest_artifact() is True
    assert artifacts.has_localize_artifact() is True
    assert artifacts.has_parsed_for_object("obj1") is True
    assert artifacts.has_render_artifact() is True

    # Invalidate from 'parse_visuals' -> parse_text, manifest, segment_and_crop stay,
    # parse is deleted.
    artifacts.invalidate_from_stage("parse_visuals")

    assert artifacts.has_parse_text_artifact() is True
    assert artifacts.has_manifest_artifact() is True
    assert artifacts.has_localize_artifact() is True
    assert (artifacts.crops_dir / "obj1__p1.png").exists() is True
    assert artifacts.has_parsed_for_object("obj1") is False

    # Invalidate from 'segment_and_crop_visuals_elements' ->
    # parse_text and manifest stay, segment_and_crop is deleted.
    artifacts.invalidate_from_stage("segment_and_crop_visuals_elements")

    assert artifacts.has_parse_text_artifact() is True
    assert artifacts.has_manifest_artifact() is True
    assert artifacts.has_localize_artifact() is False
    assert (artifacts.crops_dir / "obj1__p1.png").exists() is False


def test_gemini_parser_resumes_without_api_calls(tmp_path: Path, synthetic_pdf: Path, monkeypatch):
    """When all artifacts exist, parse() should load them without any API calls."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key_123")

    config = IngestionConfig(
        ingestion_dir=tmp_path,
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash")),
    )
    parser = GeminiParser(config)

    # Pre-populate all artifacts
    artifacts = ArtifactManager(synthetic_pdf.stem, base_dir=tmp_path)
    artifacts.save_text(artifacts.text_md_path, "01 text")
    artifacts.save_json(artifacts.placeholders_json_path, [])
    artifacts.save_json(artifacts.manifest_json_path, {"placeholders": []})
    artifacts.save_json(artifacts.bboxes_json_path, {"pages": []})
    artifacts.save_text(artifacts.document_md_path, "# Fully Enriched Markdown")
    artifacts.save_json(artifacts.summary_json_path, {})

    # Mock client methods to ensure they are NOT called
    mock_client = MagicMock()
    parser._client = mock_client

    result = parser.parse(synthetic_pdf)
    assert "01 text" in result
    assert "<!-- document_id: test_doc -->" in result
    mock_client.upload_file.assert_not_called()
    mock_client.generate_text.assert_not_called()
    mock_client.generate_structured.assert_not_called()


# ===================================================================
# 9. Gemini Client & Pipeline Error Handling tests
# ===================================================================


def test_client_validate_response_recitation_raises():
    """Verify GeminiRecitationError is raised when finish_reason is RECITATION."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "FinishReason.RECITATION"
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None

    with pytest.raises(GeminiRecitationError) as exc_info:
        client._validate_response(mock_resp, stage_name="transcribe")

    assert exc_info.value.finish_reason == "RECITATION"
    assert exc_info.value.stage == "transcribe"


def test_client_validate_response_safety_raises():
    """Verify GeminiSafetyError is raised when finish_reason is SAFETY."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "SAFETY"
    mock_candidate.safety_ratings = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}]
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None

    with pytest.raises(GeminiSafetyError) as exc_info:
        client._validate_response(mock_resp, stage_name="extract")

    assert exc_info.value.finish_reason == "SAFETY"
    assert exc_info.value.stage == "extract"


def test_client_validate_response_prompt_block_raises():
    """Verify GeminiSafetyError is raised when prompt_feedback blocks generation."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_feedback = MagicMock()
    mock_feedback.block_reason = "SAFETY"
    mock_resp.prompt_feedback = mock_feedback

    with pytest.raises(GeminiSafetyError) as exc_info:
        client._validate_response(mock_resp, stage_name="classify")

    assert exc_info.value.block_reason == "SAFETY"


def test_client_validate_response_empty_candidates_raises():
    """Verify GeminiEmptyResponseError is raised when candidates list is empty."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_resp.candidates = []
    mock_resp.prompt_feedback = None

    with pytest.raises(GeminiEmptyResponseError):
        client._validate_response(mock_resp, stage_name="transcribe")


def test_client_validate_response_max_tokens_empty_raises():
    """Verify GeminiMaxTokensError is raised when MAX_TOKENS has empty parts."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "MAX_TOKENS"
    mock_candidate.content.parts = []
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None

    with pytest.raises(GeminiMaxTokensError):
        client._validate_response(mock_resp, stage_name="transcribe")


def test_client_generate_text_raises_on_recitation():
    """generate_text should raise GeminiRecitationError instead of returning empty string."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "RECITATION"
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None
    mock_resp.usage_metadata = None
    mock_resp.text = None

    client._call_with_retry = MagicMock(return_value=mock_resp)

    with pytest.raises(GeminiRecitationError):
        client.generate_text(["dummy_content"], stage_name="transcribe")


def test_client_generate_text_raises_on_empty_text():
    """generate_text should raise GeminiEmptyResponseError when response text is whitespace."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_candidate.content.parts = [MagicMock()]
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None
    mock_resp.usage_metadata = None
    mock_resp.text = "   \n\t  "

    client._call_with_retry = MagicMock(return_value=mock_resp)

    with pytest.raises(GeminiEmptyResponseError):
        client.generate_text(["dummy_content"], stage_name="transcribe")


def test_client_generate_structured_raises_on_safety():
    """generate_structured should raise GeminiSafetyError instead of fallback parsing."""
    client = GeminiClient(GeminiParserConfig(), api_key="test_key")

    mock_resp = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "SAFETY"
    mock_resp.candidates = [mock_candidate]
    mock_resp.prompt_feedback = None
    mock_resp.usage_metadata = None
    mock_resp.parsed = None
    mock_resp.text = "{}"

    client._call_with_retry = MagicMock(return_value=mock_resp)

    with pytest.raises(GeminiSafetyError):
        client.generate_structured(["dummy_content"], Manifest, stage_name="classify")


def test_run_parse_text_does_not_save_artifacts_on_error(tmp_path: Path, mock_config):
    """When client raises an error during text parsing, no artifacts should be saved."""
    artifacts = ArtifactManager("doc_err", base_dir=tmp_path)
    mock_client = MagicMock()
    mock_client.generate_text.side_effect = GeminiRecitationError(stage="parse_text")

    with pytest.raises(GeminiRecitationError):
        run_parse_text(mock_client, "fake_handle", artifacts, mock_config)

    assert artifacts.text_md_path.exists() is False
    assert artifacts.placeholders_json_path.exists() is False
    assert artifacts.has_parse_text_artifact() is False


def test_run_parse_text_raises_on_empty_markdown(tmp_path: Path, mock_config):
    """If model outputs whitespace markdown, run_parse_text raises GeminiEmptyResponseError."""
    artifacts = ArtifactManager("doc_empty", base_dir=tmp_path)
    mock_client = MagicMock()
    mock_client.generate_text.return_value = "   \n\n  "

    with pytest.raises(GeminiEmptyResponseError):
        run_parse_text(mock_client, "fake_handle", artifacts, mock_config)

    assert artifacts.has_parse_text_artifact() is False


def test_artifacts_manager_rejects_zero_byte_artifacts(tmp_path: Path):
    """ArtifactManager readiness checks should return False for 0-byte files."""
    artifacts = ArtifactManager("doc_zero", base_dir=tmp_path)

    # Create empty 0-byte files
    artifacts.text_md_path.touch()
    artifacts.placeholders_json_path.touch()
    assert artifacts.has_parse_text_artifact() is False

    # Create empty manifest
    artifacts.manifest_json_path.touch()
    assert artifacts.has_manifest_artifact() is False

    # Create empty bboxes
    artifacts.bboxes_json_path.touch()
    assert artifacts.has_localize_artifact() is False

    # Create empty render
    artifacts.document_md_path.touch()
    artifacts.summary_json_path.touch()
    assert artifacts.has_render_artifact() is False


# ===================================================================
# 10. Page Chunking tests
# ===================================================================


def test_split_pdf_into_chunks(tmp_path: Path):
    """Test splitting a multi-page PDF into smaller chunk PDFs."""
    pdf_path = tmp_path / "multi_doc.pdf"
    doc = fitz.open()
    for i in range(7):
        p = doc.new_page(width=400, height=600)
        p.insert_text(fitz.Point(50, 50), f"Page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()

    temp_dir = tmp_path / "temp_chunks"
    chunks = split_pdf_into_chunks(pdf_path, chunk_pages=3, temp_dir=temp_dir)

    assert len(chunks) == 3
    # Chunk 1: pages 1 to 3
    assert chunks[0][0] == 1
    assert chunks[0][1] == 3
    c1_doc = fitz.open(chunks[0][2])
    assert len(c1_doc) == 3
    c1_doc.close()

    # Chunk 2: pages 4 to 6
    assert chunks[1][0] == 4
    assert chunks[1][1] == 6
    c2_doc = fitz.open(chunks[1][2])
    assert len(c2_doc) == 3
    c2_doc.close()

    # Chunk 3: page 7
    assert chunks[2][0] == 7
    assert chunks[2][1] == 7
    c3_doc = fitz.open(chunks[2][2])
    assert len(c3_doc) == 1
    c3_doc.close()


def test_compose_parse_text_prompt():
    """compose_parse_text_prompt returns the base prompt with boundary rules."""
    prompt = compose_parse_text_prompt()
    assert "Boundary continuity" in prompt
    assert "Transcribe text strictly within the physical page boundaries" in prompt
    assert "page_break" not in prompt


def test_run_chunked_parse_text_flow(tmp_path: Path):
    """Test full chunked text parsing execution across multiple 1-page chunks."""
    pdf_path = tmp_path / "test_4p.pdf"
    doc = fitz.open()
    for i in range(4):
        p = doc.new_page(width=400, height=600)
        p.insert_text(fitz.Point(50, 50), f"Page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()

    artifacts = ArtifactManager("test_4p", base_dir=tmp_path)
    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )

    mock_client = MagicMock()
    mock_client.make_pdf_part.side_effect = lambda p: f"handle_{p.stem}"
    mock_client.upload_file.side_effect = lambda p: f"handle_{p.stem}"

    def _mock_generate(contents, **kwargs):
        handle = contents[0]
        if "chunk_1_1" in handle:
            return '# Section 1\n\n<visual id="Figure_1">Figure 1</visual>'
        elif "chunk_4_4" in handle:
            return '# Section 2\n\n<visual id="Table_1">Table 1</visual>'
        return f"Text for {handle}"

    mock_client.generate_text.side_effect = _mock_generate

    clean_md, placeholders = run_parse_text(
        client=mock_client,
        file_handle="unused_handle",
        artifacts=artifacts,
        config=config,
        pdf_path=pdf_path,
    )

    assert mock_client.generate_text.call_count == 4
    mock_client.upload_file.assert_not_called()
    # Verify static system_instruction was provided and page context is in contents
    call_kwargs = mock_client.generate_text.call_args_list[0].kwargs
    assert "system_instruction" in call_kwargs
    assert "Boundary continuity" in call_kwargs["system_instruction"]
    assert "### Chunk Context" not in call_kwargs["system_instruction"]
    assert any(
        "### Chunk Context" in str(c)
        for c in mock_client.generate_text.call_args_list[0].args[0]
    )

    assert len(placeholders) == 2
    assert [p.id for p in placeholders] == ["Figure_1", "Table_1"]
    assert placeholders[0].page == 1
    assert placeholders[1].page == 4
    assert "# Section 1" in clean_md
    assert "# Section 2" in clean_md
    assert clean_md.count("<!-- page_break -->") == 3
    assert artifacts.text_md_path.exists() is True
    assert artifacts.placeholders_json_path.exists() is True
    assert (artifacts.chunks_dir / "chunk_1_1.md").exists() is True
    assert (artifacts.chunks_dir / "chunk_4_4.md").exists() is True


def test_run_chunked_parse_text_reuses_cached_chunks(tmp_path: Path):
    """Test that existing chunk cache files in 01_chunks are reused without API calls."""
    pdf_path = tmp_path / "test_cached_chunks.pdf"
    doc = fitz.open()
    for i in range(2):
        p = doc.new_page(width=400, height=600)
        p.insert_text(fitz.Point(50, 50), f"Page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()

    artifacts = ArtifactManager("test_cached_chunks", base_dir=tmp_path)
    # Pre-populate chunk 1
    (artifacts.chunks_dir / "chunk_1_1.md").write_text(
        '# Chunk 1 from cache\n<visual id="Fig_1"></visual>'
    )

    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )

    mock_client = MagicMock()
    mock_client.upload_file.return_value = "fake_handle"
    mock_client.generate_text.return_value = '# Chunk 2 generated\n<visual id="Tab_1"></visual>'

    clean_md, placeholders = run_parse_text(
        client=mock_client,
        file_handle="unused_handle",
        artifacts=artifacts,
        config=config,
        pdf_path=pdf_path,
    )

    # generate_text should only be called ONCE (for chunk 2)
    assert mock_client.generate_text.call_count == 1
    assert "# Chunk 1 from cache" in clean_md
    assert "# Chunk 2 generated" in clean_md
    assert len(placeholders) == 2
    assert placeholders[0].page == 1
    assert placeholders[1].page == 2


def test_run_chunked_parse_text_preserves_page_breaks_between_chunks(tmp_path: Path):
    """Test that page boundaries between 1-page chunks receive <!-- page_break --> without duplication."""
    pdf_path = tmp_path / "test_4p_breaks.pdf"
    doc = fitz.open()
    for i in range(4):
        p = doc.new_page(width=400, height=600)
        p.insert_text(fitz.Point(50, 50), f"Page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()

    artifacts = ArtifactManager("test_4p_breaks", base_dir=tmp_path)
    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )

    mock_client = MagicMock()
    mock_client.make_pdf_part.side_effect = lambda p: f"handle_{p.stem}"
    mock_client.upload_file.side_effect = lambda p: f"handle_{p.stem}"

    def _mock_generate(contents, **kwargs):
        handle = contents[0]
        if "chunk_1_1" in handle:
            return 'Page 1 content\n\n<visual id="Fig_1">Figure 1</visual>'
        elif "chunk_4_4" in handle:
            return 'Page 4 content\n\n<visual id="Fig_2">Figure 2</visual>'
        return f"Content for {handle}"

    mock_client.generate_text.side_effect = _mock_generate

    clean_md, placeholders = run_parse_text(
        client=mock_client,
        file_handle="unused_handle",
        artifacts=artifacts,
        config=config,
        pdf_path=pdf_path,
    )

    # In a 4-page document across 4 1-page chunks, there must be exactly 3 page breaks
    assert clean_md.count("<!-- page_break -->") == 3
    assert len(placeholders) == 2
    assert placeholders[0].id == "Fig_1"
    assert placeholders[0].page == 1
    assert placeholders[1].id == "Fig_2"
    assert placeholders[1].page == 4


def test_strip_chunk_edge_page_breaks():
    """Verify that _strip_chunk_edge_page_breaks removes leading/trailing breaks but keeps internal ones."""
    from ae.ingestion.parsers.gemini.stages.parse_text import _strip_chunk_edge_page_breaks

    # 1. Clean chunk with internal break
    text = "Page 1 text\n\n<!-- page_break -->\n\nPage 2 text"
    assert _strip_chunk_edge_page_breaks(text) == text

    # 2. Leading break stripped
    text_leading = "<!-- page_break -->\n\nPage 1 text\n\n<!-- page_break -->\n\nPage 2 text"
    assert _strip_chunk_edge_page_breaks(text_leading) == text

    # 3. Trailing standalone break stripped
    text_trailing = "Page 1 text\n\n<!-- page_break -->\n\nPage 2 text\n\n<!-- page_break -->"
    assert _strip_chunk_edge_page_breaks(text_trailing) == text

    # 4. Trailing inline break stripped (as seen in Line 54)
    text_inline_trailing = "Page 1 text\n\n<!-- page_break -->\n\nPage 2 text.<!-- page_break -->"
    assert _strip_chunk_edge_page_breaks(text_inline_trailing) == "Page 1 text\n\n<!-- page_break -->\n\nPage 2 text."

    # 5. Multiple leading/trailing breaks stripped
    text_multi = "<!-- page_break -->\n<!-- page_break -->\nPage 1\n<!-- page_break -->\n<!-- page_break -->"
    assert _strip_chunk_edge_page_breaks(text_multi) == "Page 1"


def test_run_chunked_parse_text_strips_edge_page_breaks(tmp_path: Path):
    """Test that chunks with accidental edge page breaks are cleanly sanitized without duplicates."""
    pdf_path = tmp_path / "test_2p_edge_breaks.pdf"
    doc = fitz.open()
    for i in range(2):
        p = doc.new_page(width=400, height=600)
        p.insert_text(fitz.Point(50, 50), f"Page {i + 1}")
    doc.save(str(pdf_path))
    doc.close()

    artifacts = ArtifactManager("test_2p_edge_breaks", base_dir=tmp_path)
    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )

    mock_client = MagicMock()
    mock_client.make_pdf_part.side_effect = lambda p: f"handle_{p.stem}"
    mock_client.upload_file.side_effect = lambda p: f"handle_{p.stem}"

    # Chunk 1 ends with an accidental trailing page break
    # Chunk 2 starts with an accidental leading page break
    def _mock_generate(contents, **kwargs):
        handle = str(contents[0])
        if "chunk_1_1" in handle:
            return "Page 1 content\n\n<!-- page_break -->"
        elif "chunk_2_2" in handle:
            return "<!-- page_break -->\n\nPage 2 content."
        return f"Content for {handle}"

    mock_client.generate_text.side_effect = _mock_generate

    clean_md, placeholders = run_parse_text(
        client=mock_client,
        file_handle="unused_handle",
        artifacts=artifacts,
        config=config,
        pdf_path=pdf_path,
    )

    # In a 2-page document across 2 1-page chunks, there must be exactly 1 page break
    assert clean_md.count("<!-- page_break -->") == 1
    assert clean_md == "Page 1 content\n\n<!-- page_break -->\n\nPage 2 content."


def test_compose_parse_text_prompt_chunk_rules():
    """Verify that compose_parse_text_prompt provides consistent boundary continuity rules."""
    prompt = compose_parse_text_prompt(start_page=3, end_page=3)
    assert "Boundary continuity" in prompt
    assert "page_break" not in prompt


def test_client_passes_system_instruction():
    """Verify GeminiClient._build_config passes system_instruction to GenerateContentConfig."""
    cfg = GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    client = GeminiClient(cfg, api_key="dummy_key")
    config = client._build_config(system_instruction="You are an expert transcriber.")
    assert config.system_instruction == "You are an expert transcriber."



# ===================================================================
# 11. Ingestion Token Optimization Tests
# ===================================================================


def test_client_thinking_config_mutual_exclusion():
    """Verify GeminiClient._build_config enforces mutual exclusion for thinking settings."""
    cfg = GeminiParserConfig(defaults=GeminiDefaultsConfig(thinking_budget=0))
    client = GeminiClient(cfg, api_key="dummy_key")

    # 1. Default thinking_budget=0 propagates to ThinkingConfig(thinking_budget=0)
    built_cfg = client._build_config()
    tc = built_cfg.thinking_config
    assert getattr(tc, "thinking_budget", None) == 0

    # 2. Explicit thinking_budget=3072 overrides
    built_3072 = client._build_config(thinking_budget=3072)
    assert getattr(built_3072.thinking_config, "thinking_budget", None) == 3072

    # 3. Explicit thinking_level="none" maps to thinking_budget=0
    built_none = client._build_config(thinking_budget=None, thinking_level="none")
    assert getattr(built_none.thinking_config, "thinking_budget", None) == 0

    # 4. Explicit thinking_level="high" sets thinking_level="high"
    built_high = client._build_config(thinking_budget=None, thinking_level="high")
    assert str(getattr(built_high.thinking_config, "thinking_level", "")).lower().endswith("high")


def test_transcribe_page_break_detection():
    """Verify parse_placeholders_from_markdown calculates page numbers from page break markers."""
    raw_md = (
        "# Title on Page 1\n\n"
        '<visual id="Figure_1">\nFigure 1 caption\n</visual>\n\n'
        "Some body text.\n\n"
        "<!-- page_break -->\n\n"
        "# Section on Page 2\n\n"
        '<visual id="Table_1">\nTable 1 caption\n</visual>\n\n'
        "<!-- page_break -->\n\n"
        "# Section on Page 3\n\n"
        '<visual id="Unlabeled_p3_1" />\n'
    )

    clean_md, placeholders = parse_placeholders_from_markdown(raw_md, start_page=1, end_page=3)

    assert len(placeholders) == 3
    assert placeholders[0].id == "Figure_1"
    assert placeholders[0].page == 1

    assert placeholders[1].id == "Table_1"
    assert placeholders[1].page == 2

    assert placeholders[2].id == "Unlabeled_p3_1"
    assert placeholders[2].page == 3


def test_render_pages_dynamic_scale(tmp_path: Path):
    """Verify run_render_pages dynamically scales pages so max dimension <= 768 px (1 tile)."""
    # Create an A4 PDF page (595.32 x 841.92 pt)
    pdf_path = tmp_path / "a4_test.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(str(pdf_path))
    doc.close()

    artifacts = ArtifactManager("a4_test", base_dir=tmp_path)
    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )
    placeholders = [PlaceholderItem(id="Fig_1", caption="", page=1)]

    rendered_pages = run_render_pages(pdf_path, placeholders, artifacts, config)
    assert 1 in rendered_pages

    out_img_path = artifacts.pages_dir / "p1.png"
    assert out_img_path.exists()

    with PILImage.open(out_img_path) as img:
        # Max dimension must be <= 768 px
        assert max(img.width, img.height) <= 768
        # For A4 (595x842 pt), scaled to max_dim=768 gives exactly 768 in height
        assert img.height == 768
        assert img.width <= 544


def test_classify_targeted_page_preview(tmp_path: Path):
    """Verify run_classify passes preview images for all placeholder pages (1, 2, 3) and skips non-placeholder pages (4)."""
    artifacts = ArtifactManager("test_classify_preview", base_dir=tmp_path)
    # Pre-render dummy page images 1, 2, 3, 4
    for p in (1, 2, 3, 4):
        img_path = artifacts.pages_dir / f"p{p}.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_image_bytes")

    placeholders = [
        PlaceholderItem(id="Table_1", caption="Table caption", page=1),
        PlaceholderItem(id="Figure_2", caption="Figure 2 caption", page=2),
        PlaceholderItem(id="Unlabeled_p3_1", caption="", page=3),
    ]

    mock_client = MagicMock()
    mock_manifest = Manifest(
        visuals=[
            ManifestVisual(
                id="Table_1",
                type="table",
                pages=[1],
                description="T",
            ),
            ManifestVisual(
                id="Figure_2",
                type="chart",
                pages=[2],
                description="F",
            ),
        ],
        discarded_ids=["Unlabeled_p3_1"],
    )
    mock_client.generate_structured.return_value = mock_manifest

    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )
    config.gemini.stages.classify.thinking_level = "low"
    config.gemini.stages.classify.thinking_budget = None
    manifest = run_classify(mock_client, placeholders, artifacts, config)
    assert manifest is not None

    assert mock_client.generate_structured.call_count == 1
    call_args = mock_client.generate_structured.call_args
    contents = call_args[0][0]

    # Contents should have pages 1, 2, 3 (all placeholder pages)
    assert "=== Page 1 ===" in contents
    assert (artifacts.pages_dir / "p1.png") in contents
    assert "=== Page 2 ===" in contents
    assert (artifacts.pages_dir / "p2.png") in contents
    assert "=== Page 3 ===" in contents
    assert (artifacts.pages_dir / "p3.png") in contents

    # Page 4 has no placeholders and should NOT be in contents
    assert "=== Page 4 ===" not in contents
    assert (artifacts.pages_dir / "p4.png") not in contents

    # thinking_level must be passed from stage config ("low")
    assert call_args[1].get("thinking_level") == "low"


def test_reconcile_manifest_preserves_page_invariants():
    """Verify reconcile_manifest retains multi-page table spans and aligns with placeholder page."""
    placeholders = [
        PlaceholderItem(id="Table_1", caption="Table caption", page=3),
    ]
    # Model returns table spanning pages [3, 4]
    model_output = Manifest(
        visuals=[
            ManifestVisual(
                id="Table_1",
                type="table",
                pages=[3, 4],
                description="Multi-page table",
            ),
        ]
    )

    reconciled = reconcile_manifest(model_output, placeholders)
    tbl = reconciled.visuals[0]
    assert tbl.pages == [3, 4]

    # If model only returned page [4], page 3 from placeholder is unioned in
    model_output_p4 = Manifest(
        visuals=[
            ManifestVisual(
                id="Table_1",
                type="table",
                pages=[4],
                description="Multi-page table",
            ),
        ]
    )
    reconciled_p4 = reconcile_manifest(model_output_p4, placeholders)
    tbl_p4 = reconciled_p4.visuals[0]
    assert tbl_p4.pages == [3, 4]


def test_reconcile_manifest_multipage_figure_continuation():
    """Verify reconcile_manifest handles multi-page figure/scheme continuations with uncaptioned previous page."""
    placeholders = [
        PlaceholderItem(id="Unlabeled_p6_1", caption="", page=6),
        PlaceholderItem(
            id="Fig.-10",
            caption="Fig.-10: Hybridization of Pyrazoline with Benzimidazole",
            page=7,
        ),
    ]

    # Model merged compounds 19–23 on p. 6 and 24–33 on p. 7 into Fig.-10 spanning [6, 7]
    model_output = Manifest(
        visuals=[
            ManifestVisual(
                id="Fig.-10",
                type="molecular_structure",
                pages=[6, 7],
                description="Chemical structures of derivatives 19–33 with Ar substituent table.",
            ),
        ],
        discarded_ids=["Unlabeled_p6_1"],
    )

    reconciled = reconcile_manifest(model_output, placeholders)
    assert len(reconciled.visuals) == 1
    fig = reconciled.visuals[0]
    assert fig.id == "Fig.-10"
    assert fig.pages == [6, 7]
    assert fig.type == "molecular_structure"
    assert "Unlabeled_p6_1" in reconciled.discarded_ids


def test_manifest_prompt_contains_multipage_figure_instructions():
    """Verify manifest prompt template instructs the model on multi-page chemical schemes and figures."""
    from ae.ingestion.parsers.gemini.stages.manifest_visuals import _read_prompt_template

    prompt = _read_prompt_template()
    assert "Multi-page continuations & split figures" in prompt
    assert "Figure & scheme continuations" in prompt
    assert "Shared R-group tables & legends" in prompt
    assert "Standalone uncaptioned structures" in prompt


def test_rdkit_correction_passes_thinking_level():
    """Verify validate_and_correct_entities forwards configured thinking_level to generate_structured."""
    mock_client = MagicMock()
    mock_client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="1", corrected_smiles="CC")],
    )

    entities = [
        ChemicalEntity(label="1", smiles="InvalidSmiles@@", role="target_product"),
    ]
    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )

    validate_and_correct_entities(entities, [], mock_client, config)

    assert mock_client.generate_structured.call_count == 1
    call_kwargs = mock_client.generate_structured.call_args[1]
    assert call_kwargs.get("thinking_level") == "low"
    assert call_kwargs.get("stage_name") == "rdkit_correction"


def test_artifacts_manager_per_page_bboxes(tmp_path: Path):
    """Test saving and loading per-page bboxes, empty marker, and invalidation."""
    artifacts = ArtifactManager("test_bboxes_doc", base_dir=tmp_path)

    assert artifacts.visual_bboxes_dir == artifacts.visuals_geom_dir / "bboxes"
    assert artifacts.has_localize_artifact() is False

    # Empty marker test
    artifacts.mark_visual_bboxes_empty()
    assert artifacts.has_localize_artifact() is True
    assert artifacts.load_all_page_bboxes() == {}

    # Reset (also test backward-compatible alias "segment_and_crop_visuals_atoms")
    artifacts.invalidate_from_stage("segment_and_crop_visuals_atoms")
    assert artifacts.has_localize_artifact() is False

    # Also test primary stage name
    artifacts.mark_visual_bboxes_empty()
    assert artifacts.has_localize_artifact() is True
    artifacts.invalidate_from_stage("segment_and_crop_visuals_elements")
    assert artifacts.has_localize_artifact() is False

    # Save page 1 and page 2
    loc1 = PageLocalization(page=1, boxes=[VisualBox(id="V1", box_2d=[10, 10, 100, 100], found=True)])
    loc2 = PageLocalization(page=2, boxes=[VisualBox(id="V2", box_2d=[20, 20, 200, 200], found=True)])

    artifacts.save_page_bboxes(1, loc1)
    assert artifacts.has_localize_artifact() is True
    assert artifacts.load_page_bboxes(1) == loc1
    assert artifacts.load_page_bboxes(2) is None

    artifacts.save_page_bboxes(2, loc2)
    all_locs = artifacts.load_all_page_bboxes()
    assert len(all_locs) == 2
    assert all_locs[1] == loc1
    assert all_locs[2] == loc2


def test_legacy_bboxes_json_fallback(tmp_path: Path):
    """Test backward compatibility fallback when monolithic bboxes.json is present."""
    artifacts = ArtifactManager("test_legacy_doc", base_dir=tmp_path)

    # Legacy format: visuals/bboxes.json
    artifacts.save_json(
        artifacts.bboxes_json_path,
        {
            "2": {
                "page": 2,
                "boxes": [{"id": "Scheme_1", "box_2d": [10, 20, 30, 40], "found": True}],
            }
        },
    )

    assert artifacts.has_localize_artifact() is True
    locs = artifacts.load_all_page_bboxes()
    assert 2 in locs
    assert locs[2].boxes[0].id == "Scheme_1"


def test_localize_visuals_per_page_resumption(tmp_path: Path, mock_config):
    """Test that run_localize_visuals skips already cached pages and only queries missing ones."""
    artifacts = ArtifactManager("test_resume_doc", base_dir=tmp_path)

    # Create dummy rendered pages
    (artifacts.pages_dir / "p1.png").write_bytes(b"dummy_p1")
    (artifacts.pages_dir / "p2.png").write_bytes(b"dummy_p2")

    # Pre-cache page 1
    cached_p1 = PageLocalization(
        page=1,
        boxes=[VisualBox(id="Fig1", box_2d=[50, 50, 500, 500], found=True)],
    )
    artifacts.save_page_bboxes(1, cached_p1)

    manifest = Manifest(
        visuals=[
            ManifestVisual(id="Fig1", type="chart", pages=[1], description="Chart 1"),
            ManifestVisual(id="Fig2", type="molecular_structure", pages=[2], description="Mol 2"),
        ]
    )

    mock_client = MagicMock()
    mock_client.generate_structured.return_value = PageLocalization(
        page=2,
        boxes=[VisualBox(id="Fig2", box_2d=[100, 100, 600, 600], found=True)],
    )

    results = run_localize_visuals(mock_client, manifest, artifacts, mock_config)

    # Client should only have been called ONCE (for page 2, not page 1)
    assert mock_client.generate_structured.call_count == 1
    assert len(results) == 2
    assert results[1] == cached_p1
    assert results[2].boxes[0].id == "Fig2"

    # Both page bboxes should now exist on disk
    assert artifacts.page_bboxes_path(1).exists()
    assert artifacts.page_bboxes_path(2).exists()


def test_compose_parse_text_prompt_injects_page_context():
    """Verify compose_parse_text_prompt includes page numbers when provided."""
    base = compose_parse_text_prompt()
    assert "Document Page Context" not in base

    single_page_prompt = compose_parse_text_prompt(start_page=4, end_page=4)
    assert "Document Page Context: You are transcribing physical page 4" in single_page_prompt
    assert "Unlabeled_p4_1" in single_page_prompt

    range_prompt = compose_parse_text_prompt(start_page=2, end_page=3)
    assert "Document Page Context: You are transcribing physical pages 2 to 3" in range_prompt
    assert "Unlabeled_p2_1" in range_prompt


def test_parse_placeholders_normalizes_unlabeled_ids_by_page():
    """Verify parse_placeholders_from_markdown normalizes Unlabeled_* placeholders to Unlabeled_p{page}_{k}."""
    raw_md_p4 = (
        "Some text on page 4.\n\n"
        '<visual id="Unlabeled_p1_1" />\n\n'
        '<visual id="Fig.-5">\nFig.-5: Structure\n</visual>\n\n'
        '<visual id="Unlabeled_something" />\n'
    )
    clean_p4, phs_p4 = parse_placeholders_from_markdown(raw_md_p4, start_page=4)
    assert len(phs_p4) == 3
    assert phs_p4[0].id == "Unlabeled_p4_1"
    assert phs_p4[0].page == 4
    assert '<visual id="Unlabeled_p4_1" />' in clean_p4

    assert phs_p4[1].id == "Fig.-5"
    assert phs_p4[1].page == 4

    assert phs_p4[2].id == "Unlabeled_p4_2"
    assert phs_p4[2].page == 4
    assert '<visual id="Unlabeled_p4_2" />' in clean_p4

    # Now verify multi-chunk scenario with shared unlabeled_counts
    id_counts = {}
    unlabeled_counts = {}

    clean_p4_shared, phs_p4_shared = parse_placeholders_from_markdown(
        '<visual id="Unlabeled_p1_1" />',
        start_page=4,
        id_counts=id_counts,
        unlabeled_counts=unlabeled_counts,
    )
    assert phs_p4_shared[0].id == "Unlabeled_p4_1"
    assert '<visual id="Unlabeled_p4_1" />' in clean_p4_shared

    clean_p6_shared, phs_p6_shared = parse_placeholders_from_markdown(
        '<visual id="Unlabeled_p1_1" />',
        start_page=6,
        id_counts=id_counts,
        unlabeled_counts=unlabeled_counts,
    )
    assert phs_p6_shared[0].id == "Unlabeled_p6_1"
    assert '<visual id="Unlabeled_p6_1" />' in clean_p6_shared

