"""Unit tests for EXP-01 ingestion optimizations and semantic OCSR oracle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pymupdf as fitz

from ae.core.config.optimization_settings import (
    ExecutionConfig,
    IngestionConfig,
)
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import BatchItemResult, GeminiClient
from ae.ingestion.parsers.gemini.parser import GeminiParser
from ae.ingestion.parsers.gemini.schemas import (
    ChemicalEntity,
    ElementLocalizationResult,
    ElementManifestItem,
    Manifest,
    ManifestVisual,
    MolecularParseResult,
    PageLocalization,
    PipelineSummary,
    StageUsage,
    TableParseResult,
    VisualBox,
    VisualDecomposition,
)
from ae.ingestion.parsers.gemini.stages.context import extract_local_text_context
from ae.ingestion.parsers.gemini.stages.ocsr_oracle import (
    formulas_compatible,
    verify_entities_with_oracle,
    verify_markush_example,
)
from ae.ingestion.parsers.gemini.stages.parse_text import run_parse_text
from ae.ingestion.parsers.gemini.stages.parse_visuals import parse_single_visual
from ae.ingestion.parsers.gemini.stages.rdkit_validation import (
    SmilesCorrectionItem,
    SmilesCorrectionResponse,
    validate_and_correct_entities,
)
from ae.ingestion.parsers.gemini.stages.segment_and_crop_visuals_elements import (
    extract_element_localization_from_decomposition,
)
from ae.ingestion.service import VisualParsingService


def test_merged_t3_t4_skips_localize_elements_when_boxes_valid(tmp_path: Path, monkeypatch):
    cfg = IngestionConfig(ingestion_dir=tmp_path / "ing")
    doc_id = "merged_doc"
    artifacts = ArtifactManager(doc_id, base_dir=cfg.ingestion_dir)
    artifacts.document_md_path.write_text(
        '<visual id="Scheme_1" type="molecular_structure">\nScheme 1.\n> desc\n</visual>',
        encoding="utf-8",
    )
    artifacts.save_page_bboxes(
        1,
        PageLocalization(page=1, boxes=[VisualBox(id="Scheme_1", found=True, box_2d=[0, 0, 500, 500])]),
    )
    crop = artifacts.crops_dir / "Scheme_1__p1.png"
    crop.write_bytes(b"fake_png")

    decomp = VisualDecomposition(
        mode="composite",
        composition_template='<element id="mol_1" />',
        elements=[
            ElementManifestItem(
                element_id="mol_1",
                type="molecular_structure",
                label="1",
                description="Compound 1",
                box_2d=[10, 20, 400, 500],
            )
        ],
    )
    loc_called = {"count": 0}

    def _fail_if_called(*args, **kwargs):
        loc_called["count"] += 1
        return ElementLocalizationResult(boxes=[])

    monkeypatch.setattr("ae.ingestion.service.decompose_visual", lambda *a, **kw: decomp)
    monkeypatch.setattr("ae.ingestion.service.localize_elements", _fail_if_called)
    monkeypatch.setattr(
        "ae.ingestion.service.parse_single_visual",
        lambda *a, **kw: MolecularParseResult(entities=[ChemicalEntity(label="1", smiles="CCO")]),
    )

    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4")
    mock_page = MagicMock()
    mock_page.rect = fitz.Rect(0, 0, 600, 800)
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    monkeypatch.setattr("ae.ingestion.service.fitz.open", lambda *a, **kw: mock_doc)

    service = VisualParsingService(config=cfg, pdf_dir=tmp_path, ingestion_dir=cfg.ingestion_dir)
    service._client = MagicMock()
    res = service.parse_visuals(doc_id, ["Scheme_1"])

    assert res["Scheme_1"].status == "success"
    assert loc_called["count"] == 0
    assert (artifacts.element_bboxes_dir / "Scheme_1.json").exists()


def test_merged_t3_t4_falls_back_when_box_missing():
    decomp = VisualDecomposition(
        mode="composite",
        composition_template='<element id="m1" /> + <element id="m2" />',
        elements=[
            ElementManifestItem(
                element_id="m1", type="molecular_structure", label="1", description="ok", box_2d=[10, 10, 200, 200]
            ),
            ElementManifestItem(
                element_id="m2", type="molecular_structure", label="2", description="missing", box_2d=None
            ),
        ],
    )
    loc_res, missing = extract_element_localization_from_decomposition(decomp)
    assert len(loc_res.boxes) == 1
    assert loc_res.boxes[0].id == "m1"
    assert [e.element_id for e in missing] == ["m2"]


def test_parse_text_files_api_fallback_when_inline_disabled(tmp_path: Path):
    pdf_path = tmp_path / "fallback.pdf"
    doc = fitz.open()
    doc.new_page(width=400, height=600).insert_text(fitz.Point(50, 50), "Hello world")
    doc.save(str(pdf_path))
    doc.close()

    artifacts = ArtifactManager("fallback", base_dir=tmp_path)
    cfg = IngestionConfig(execution=ExecutionConfig(use_inline_pdf=False))
    mock_client = MagicMock()
    mock_client.upload_file.return_value = "uploaded_handle"
    mock_client.generate_text.return_value = "Hello world transcribed"

    md, _ = run_parse_text(mock_client, None, artifacts, cfg, pdf_path=pdf_path)
    assert "Hello world transcribed" in md
    mock_client.upload_file.assert_called_once()
    mock_client.delete_file.assert_called_once_with("uploaded_handle")


def test_parse_text_silent_error_guard_retries_at_temp_05(tmp_path: Path):
    pdf_path = tmp_path / "silent.pdf"
    doc = fitz.open()
    p = doc.new_page(width=400, height=600)
    p.insert_text(fitz.Point(50, 50), "Experimental section: Compound 15 showed potent antibacterial")
    p.insert_text(fitz.Point(50, 80), "activity against S. aureus with MIC = 3.125 ug/mL and contained")
    p.insert_text(fitz.Point(50, 110), "a 2-chloro-6-fluorophenyl ring at C2 of the benzimidazole scaffold.")
    doc.save(str(pdf_path))
    doc.close()

    artifacts = ArtifactManager("silent", base_dir=tmp_path)
    cfg = IngestionConfig(execution=ExecutionConfig(silent_error_guard=True))
    mock_client = MagicMock()
    mock_client.make_pdf_part.return_value = "inline_part"

    temps: list[float] = []

    def _gen(contents, **kwargs):
        temps.append(kwargs["temperature"])
        if len(temps) == 1:
            return "<!-- omitted -->"
        return "Experimental section: Compound 15 showed potent antibacterial activity."

    mock_client.generate_text.side_effect = _gen
    md, _ = run_parse_text(mock_client, None, artifacts, cfg, pdf_path=pdf_path)
    assert temps == [0.0, 0.5]
    assert "Compound 15" in md
    mock_client.record_event.assert_called_with("silent_error_retries")


def test_table_adaptive_thinking_and_silent_error_retry(tmp_path: Path):
    cfg = IngestionConfig()
    vis = ManifestVisual(id="Table_1", type="table", pages=[1], description="MIC table")
    crop1 = tmp_path / "t1.png"
    crop2 = tmp_path / "t2.png"
    crop1.write_bytes(b"png1")
    crop2.write_bytes(b"png2")

    mock_client = MagicMock()
    calls: list[dict] = []

    def _gen(contents, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return TableParseResult(status="success", table_body="")
        return TableParseResult(
            status="success",
            table_format="markdown",
            headers=["Comp", "MIC"],
            table_body="| Comp | MIC |\n|---|---|\n| 1a | 4 |",
            footnotes=["a: MRSA N315"],
        )

    mock_client.generate_structured.side_effect = _gen
    res = parse_single_visual(mock_client, vis, [crop1], "Table 1", cfg)
    assert len(calls) == 2
    assert calls[0]["thinking_level"] == "low"
    assert calls[0]["temperature"] == 0.0
    assert calls[1]["temperature"] == 0.5
    assert "MRSA N315" in res.content

    # Multi-page table keeps configured 'medium' thinking level
    calls.clear()
    mock_client.generate_structured.side_effect = lambda *a, **kw: (
        calls.append(dict(kw))
        or TableParseResult(status="success", table_body="| A |\n|---|\n| 1 |")
    )
    parse_single_visual(mock_client, vis, [crop1, crop2], "Table 1", cfg)
    assert calls[0]["thinking_level"] == "medium"


def test_context_n_plus_minus_1_and_in_text_callouts():
    pages = [
        "### 1. Intro\nBackground info.",
        "### 2. Chemistry\nDiscussion of derivatives 37–39.",
        '<visual id="Scheme_1" type="molecular_structure">\nScheme 1. Synthesis of 37–39.\n</visual>',
        "### 3. Biology\nMIC assays.",
        "### 4. Experimental\n\n#### 4.5.25. (2-(3-(4-fluorophenyl)-1-phenyl-1H-pyrazol-4-yl)-1H-benzo[d]imidazole-1-yl)(pyridin-4-yl)methanone (37)\n"
        "White powder, yield 76%. ESI-MS: 458.50 (C29H19FN4O, [M+H]+).",
    ]
    doc_md = "\n\n<!-- page_break -->\n\n".join(pages)
    ctx = extract_local_text_context(
        doc_md,
        "Scheme_1",
        caption_map={"Scheme_1": "Scheme 1. Synthesis of 37–39."},
    )
    assert "Discussion of derivatives 37–39." in ctx
    assert "MIC assays." in ctx
    assert "Background info." not in ctx
    assert "In-text callouts:" in ctx
    assert "pyridin-4-yl" in ctx


def test_stages_usage_scoped_and_merged_without_overwrite(tmp_path: Path, monkeypatch):
    cfg = IngestionConfig(ingestion_dir=tmp_path / "ing")
    client = GeminiClient(cfg.gemini, api_key="dummy")

    with client.doc_scope("doc_a"):
        client.record_usage("parse_text", prompt_tokens=100, cached_tokens=40, candidates_tokens=20, reasoning_tokens=5, total_tokens=125)
    with client.doc_scope("doc_b"):
        client.record_usage("parse_text", prompt_tokens=999, candidates_tokens=99, total_tokens=1098)

    assert client.get_all_stage_usage("doc_a")["parse_text"].prompt_tokens == 100
    assert client.get_all_stage_usage("doc_a")["parse_text"].cached_tokens == 40
    assert client.get_all_stage_usage("doc_b")["parse_text"].prompt_tokens == 999

    doc_id = "doc_a"
    artifacts = ArtifactManager(doc_id, base_dir=cfg.ingestion_dir)
    artifacts.document_md_path.write_text(
        '<visual id="Table_1" type="table">\nTable 1\n> desc\n</visual>',
        encoding="utf-8",
    )
    initial_summary = PipelineSummary(
        total_placeholders=1,
        total_manifest_visuals=1,
        stages_usage={"parse_text": StageUsage(calls=1, prompt_tokens=100, cached_tokens=40, candidates_tokens=20, reasoning_tokens=5, total_tokens=125)},
    )
    artifacts.save_json(artifacts.summary_json_path, initial_summary)
    artifacts.save_page_bboxes(
        1,
        PageLocalization(page=1, boxes=[VisualBox(id="Table_1", found=True, box_2d=[0, 0, 500, 500])]),
    )
    (artifacts.crops_dir / "Table_1__p1.png").write_bytes(b"png")

    monkeypatch.setattr(
        "ae.ingestion.service.decompose_visual",
        lambda *a, **kw: VisualDecomposition(mode="single", single_type="table"),
    )

    def _fake_parse(c, vis, crops, cap, config, context=""):
        c.record_usage("parse_table", prompt_tokens=50, candidates_tokens=30, total_tokens=80)
        return TableParseResult(status="success", table_body="| A |\n|---|\n| 1 |")

    monkeypatch.setattr("ae.ingestion.service.parse_single_visual", _fake_parse)
    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4")
    mock_page = MagicMock()
    mock_page.rect = fitz.Rect(0, 0, 600, 800)
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    monkeypatch.setattr("ae.ingestion.service.fitz.open", lambda *a, **kw: mock_doc)

    service = VisualParsingService(config=cfg, pdf_dir=tmp_path, ingestion_dir=cfg.ingestion_dir)
    service._client = client
    service.parse_visuals(doc_id, ["Table_1"])

    updated = PipelineSummary.model_validate(artifacts.load_json(artifacts.summary_json_path))
    assert updated.stages_usage["parse_text"].prompt_tokens == 100
    assert updated.stages_usage["parse_table"].prompt_tokens == 50


def test_service_honest_page_dpi_150(tmp_path: Path, monkeypatch):
    cfg = IngestionConfig(ingestion_dir=tmp_path / "ing")
    doc_id = "dpi_doc"
    artifacts = ArtifactManager(doc_id, base_dir=cfg.ingestion_dir)
    artifacts.document_md_path.write_text(
        '<visual id="Fig_1" type="molecular_structure">\nFig 1\n> desc\n</visual>',
        encoding="utf-8",
    )
    # Create low-res 768px stage 02 preview (should NOT prevent 150 DPI render for localize)
    (artifacts.pages_dir / "p1.png").write_bytes(b"lowres")

    seen_dpi: list[int] = []
    mock_pix = MagicMock()
    mock_pix.save.side_effect = lambda p: Path(p).write_bytes(b"highres_150")

    def _get_pixmap(**kwargs):
        if "dpi" in kwargs and "clip" not in kwargs:
            seen_dpi.append(int(kwargs["dpi"]))
        return mock_pix

    mock_page = MagicMock()
    mock_page.rect = fitz.Rect(0, 0, 600, 800)
    mock_page.get_pixmap.side_effect = _get_pixmap
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    monkeypatch.setattr("ae.ingestion.service.fitz.open", lambda *a, **kw: mock_doc)
    monkeypatch.setattr(
        "ae.ingestion.service.localize_page",
        lambda c, img, pno, man, config: PageLocalization(
            page=pno, boxes=[VisualBox(id="Fig_1", found=True, box_2d=[0, 0, 500, 500])]
        ),
    )
    monkeypatch.setattr(
        "ae.ingestion.service.decompose_visual",
        lambda *a, **kw: VisualDecomposition(mode="single", single_type="molecular_structure"),
    )
    monkeypatch.setattr(
        "ae.ingestion.service.parse_single_visual",
        lambda *a, **kw: MolecularParseResult(entities=[ChemicalEntity(label="1", smiles="CCO")]),
    )

    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4")
    service = VisualParsingService(config=cfg, pdf_dir=tmp_path, ingestion_dir=cfg.ingestion_dir)
    service._client = MagicMock()
    service.parse_visuals(doc_id, ["Fig_1"])

    assert 150 in seen_dpi
    assert (artifacts.visuals_geom_dir / "pages" / "p1_dpi150.png").exists()


def test_parse_batch_gemini_batch_api(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "batch_doc.pdf"
    doc = fitz.open()
    doc.new_page(width=400, height=600).insert_text(fitz.Point(50, 50), "Batch page 1 text")
    doc.save(str(pdf_path))
    doc.close()

    cfg = IngestionConfig(
        ingestion_dir=tmp_path / "ing",
        execution=ExecutionConfig(use_batch_api=True),
    )
    parser = GeminiParser(cfg) if False else None
    with monkeypatch.context() as m:
        m.setattr(GeminiParser, "_resolve_api_key", staticmethod(lambda: "dummy"))
        parser = GeminiParser(cfg)

    mock_client = MagicMock()
    mock_client.make_pdf_part.return_value = "pdf_part"
    mock_client.get_all_stage_usage.return_value = {}
    mock_client.get_scope_counters.return_value = {}

    def _fake_batch(specs, **kwargs):
        out = {}
        for s in specs:
            if s.stage_name == "parse_text":
                out[s.key] = BatchItemResult(
                    key=s.key,
                    text='Body text.\n\n<visual id="Table_1">Table 1. MIC</visual>',
                )
            elif s.stage_name == "classify":
                out[s.key] = BatchItemResult(
                    key=s.key,
                    parsed=Manifest(
                        visuals=[ManifestVisual(id="Table_1", type="table", pages=[1], description="MIC")]
                    ),
                )
        return out

    mock_client.run_batch_requests.side_effect = _fake_batch
    parser._client = mock_client

    outputs = parser.parse_batch([pdf_path])
    assert pdf_path.resolve() in outputs
    assert '<visual id="Table_1" type="table">' in outputs[pdf_path.resolve()]
    assert mock_client.run_batch_requests.call_count == 2


def test_ocsr_oracle_catches_rjc15_and_37_and_markush():
    assert formulas_compatible("C29H19FN4O", "C29H19FN4O") is True
    assert formulas_compatible("C29H19FN4O", "C30H21FN4O") is False

    # 1. Catch RJC cid=15 positional isomer (5-Cl-2-F vs 2-Cl-6-F)
    bad_15 = ChemicalEntity(
        label="15",
        smiles="Cn1c(-c2cc(Cl)ccc2F)nc2cc(-c3nc(-c4ccc(Cl)cc4)no3)ccc21",
    )
    rjc_ctx = (
        "Compound 15, which had an oxadiazole nucleus with a p-chlorophenyl group at C3, "
        "and a 2-chloro-6-fluorophenyl at C2 of the benzimidazole ring, with MIC 3.125 ug/mL."
    )
    findings_15 = verify_entities_with_oracle([bad_15], context=rjc_ctx)
    assert findings_15[0].passed is False
    assert bad_15.provenance_status == "mismatch_flagged"

    # Correct 2-Cl-6-F isomer passes
    good_15 = ChemicalEntity(
        label="15",
        smiles="Cn1c(-c2c(F)cccc2Cl)nc2cc(-c3nc(-c4ccc(Cl)cc4)no3)ccc21",
    )
    findings_15_ok = verify_entities_with_oracle([good_15], context=rjc_ctx)
    assert findings_15_ok[0].passed is True

    # 2. Catch j.ejmech.2019.03.026 compound 37 (pyridin-3-yl vs pyridin-4-yl) via py2opsin InChIKey + Formula
    bad_37 = ChemicalEntity(
        label="37",
        smiles="O=C(c1cccnc1)n1c(-c2cn(-c3ccccc3)nc2-c2ccc(F)cc2)nc2ccccc21",
    )
    ctx_37 = (
        "### 4.5.25. (2-(3-(4-fluorophenyl)-1-phenyl-1H-pyrazol-4-yl)-1H-benzo[d]imidazole-1-yl)(pyridin-4-yl)methanone (37)\n\n"
        "White powder, yield 76%. ESI-MS: 460.50 (C28H18FN5O, [M+H]+)."
    )
    mock_client = MagicMock()
    mock_client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[
            SmilesCorrectionItem(
                label="37",
                corrected_smiles="O=C(c1ccncc1)n1c(-c2cn(-c3ccccc3)nc2-c2ccc(F)cc2)nc2ccccc21",
            )
        ]
    )
    cfg = IngestionConfig()
    updated, _ = validate_and_correct_entities(
        [bad_37],
        crop_paths=[],
        client=mock_client,
        config=cfg,
        max_attempts=1,
        context=ctx_37,
    )
    assert updated[0].provenance_status == "verified_exact"
    assert mock_client.generate_structured.call_count == 1

    # 3. Verify Markush assembly via assemble_markush_single
    assembled, mk_finding = verify_markush_example(
        "O=C(c1ccncc1)n1c(-c2cn(-c3ccccc3)nc2-c2ccc([*:1])cc2)nc2ccccc21",
        {"1": "F[*:1]"},
        expected_iupac="(2-(3-(4-fluorophenyl)-1-phenyl-1H-pyrazol-4-yl)-1H-benzo[d]imidazole-1-yl)(pyridin-4-yl)methanone",
        expected_formula="C28H18FN5O",
    )
    assert assembled is not None
    assert mk_finding.passed is True
    assert mk_finding.provenance_status == "verified_exact"

    # 4. Zero false retries on already-valid structure without mismatch
    valid_only = ChemicalEntity(label="1", smiles="CN(C)c1ccc(C=CC=Nc2nc3ccccc3[nH]2)cc1")
    mock_client_zero = MagicMock()
    res_zero, _ = validate_and_correct_entities(
        [valid_only],
        crop_paths=[],
        client=mock_client_zero,
        config=cfg,
        max_attempts=1,
        context="",
    )
    assert res_zero[0].provenance_status == "unverified_valid"
    assert mock_client_zero.generate_structured.call_count == 0
