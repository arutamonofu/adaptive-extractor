"""Unit tests for local/template context extraction, HTML composite tables, and Markush attachment bindings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pymupdf as fitz

from ae.core.config.optimization_settings import (
    GeminiDefaultsConfig,
    GeminiParserConfig,
    IngestionConfig,
)
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.schemas import (
    AttachmentPoint,
    ChemicalEntity,
    ElementLocalizationResult,
    ElementManifestItem,
    MolecularParseResult,
    PageLocalization,
    VisualBox,
    VisualDecomposition,
)
from ae.ingestion.parsers.gemini.stages.context import (
    extract_element_template_context,
    extract_local_text_context,
)
from ae.ingestion.parsers.gemini.stages.rdkit_validation import validate_and_canonicalize
from ae.ingestion.parsers.gemini.stages.render_markdown import (
    extract_unparsed_visual_ids,
    render_composite_block,
    render_parse_body,
)
from ae.ingestion.service import VisualParsingService


def test_extract_local_text_context_collapses_neighbor_visuals_and_finds_cross_refs():
    huge_registry = "\n".join(f"| **97a_{i}** | `c1ccccc1` |" for i in range(60))
    filler = ("Unrelated background discussion paragraph. " * 15).strip()
    doc_md = (
        "<!-- document_id: test_paper -->\n\n"
        "### 1. Introduction\n\n"
        "Early studies on Table 11 showed promising MIC values for 3-pyridine derivatives.\n\n"
        f"{filler}\n\n"
        f"{filler}\n\n"
        f"{filler}\n\n"
        "### 2.3. Benzimidazoles\n\n"
        "Compound 98a, with a 3-pyridine moiety at C5, showed MIC = 0.25 µg/mL. "
        "Introduction of F at C6 on 98b kept potency.\n\n"
        '<visual id="Scheme_27" type="molecular_structure">\n'
        "Scheme 27. Synthesis of benzimidazole-pyrazoles 98a-98i.\n\n"
        f"{huge_registry}\n"
        "</visual>\n\n"
        "<!-- page_break -->\n\n"
        '<visual id="Table_11" type="table">\n'
        "Table 11. MIC values and SAR studies.\n"
        "> Table stub\n"
        "</visual>\n\n"
        "Next section discusses indoline derivatives 35."
    )

    ctx = extract_local_text_context(doc_md, "Table_11")

    assert "Compound 98a, with a 3-pyridine moiety at C5" in ctx
    assert "[Visual Scheme_27: Scheme 27. Synthesis of benzimidazole-pyrazoles 98a-98i.]" in ctx
    assert "97a_50" not in ctx
    assert "Next section discusses indoline derivatives 35." in ctx
    assert "Cross-referenced paragraphs:" in ctx
    assert "Early studies on Table 11 showed promising MIC values" in ctx


def test_extract_local_text_context_skips_unlabeled_cross_refs():
    doc_md = (
        "Some introductory text.\n\n"
        '<visual id="Unlabeled_p2_1" type="molecular_structure">\n'
        "> Unlabeled structure\n"
        "</visual>\n\n"
        "Following text."
    )
    ctx = extract_local_text_context(doc_md, "Unlabeled_p2_1")
    assert "Some introductory text." in ctx
    assert "Following text." in ctx
    assert "Cross-referenced paragraphs:" not in ctx


def test_extract_element_template_context_html_and_pipe_tables():
    html_table = (
        "<table>\n"
        "  <thead>\n"
        '    <tr><th rowspan="2">Compounds</th><th colspan="2">Substituents</th>'
        '<th colspan="2">Minimum Inhibitory Concentration µg/mL</th></tr>\n'
        "    <tr><th>C5</th><th>C6</th><th>Sa</th><th>Sa + HS</th></tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        '    <tr><td>98a</td><td><element id="mol_98a" /></td><td>H</td><td>0.25</td><td>4</td></tr>\n'
        '    <tr><td>98b</td><td><element id="mol_98b" /></td><td>F</td><td>0.25</td><td>2</td></tr>\n'
        "  </tbody>\n"
        "</table>"
    )
    ctx_b = extract_element_template_context(html_table, "mol_98b")
    assert "Minimum Inhibitory Concentration µg/mL" in ctx_b
    assert "<th>C5</th>" in ctx_b
    assert '<element id="mol_98b" />' in ctx_b
    assert "mol_98a" not in ctx_b

    pipe_table = (
        "| Compounds | C5 | C6 |\n"
        "| :--- | :--- | :--- |\n"
        '| 98a | <element id="mol_98a" /> | H |\n'
        '| 98b | <element id="mol_98b" /> | F |'
    )
    pipe_ctx = extract_element_template_context(pipe_table, "mol_98b")
    assert "| Compounds | C5 | C6 |" in pipe_ctx
    assert '<element id="mol_98b" />' in pipe_ctx
    assert "mol_98a" not in pipe_ctx


def test_attachment_points_schema_rdkit_and_rendering():
    schema = MolecularParseResult.model_json_schema()
    entity_props = schema["$defs"]["ChemicalEntity"]["properties"]
    assert "attachment_points" in entity_props
    # Ensure no dict / additionalProperties in generated Gemini schema
    assert "additionalProperties" not in str(schema)

    # RDKit canonicalization preserves [*:1] and [*:2]
    valid, canonical, err = validate_and_canonicalize("[*:1]c1ccc([*:2])cc1")
    assert valid is True
    assert err is None
    assert "[*:1]" in canonical
    assert "[*:2]" in canonical

    # Composite rendering includes Attachment Points column when present
    decomp = VisualDecomposition(
        mode="composite",
        composition_template='<element id="mol_98" />',
        elements=[
            ElementManifestItem(
                element_id="mol_98",
                type="molecular_structure",
                label="98a-98i",
                description="Core scaffold",
            )
        ],
    )
    mol_res = MolecularParseResult(
        entities=[
            ChemicalEntity(
                label="98a-98i",
                smiles="c1cc([*:1])cnn1",
                attachment_points=[
                    AttachmentPoint(placeholder="[*:1]", variable="R (C7)"),
                ],
            )
        ]
    )
    rendered_comp = render_composite_block(decomp, {"mol_98": mol_res}, visual_id="Scheme_27")
    assert "| Label | SMILES | Attachment Points |" in rendered_comp
    assert "| **98a-98i** | `c1cc([*:1])cnn1` | `[*:1]` = R (C7) |" in rendered_comp

    rendered_single = render_parse_body(mol_res)
    assert "| Label | SMILES | Attachment Points |" in rendered_single
    assert "`[*:1]` = R (C7)" in rendered_single


def test_extract_unparsed_visual_ids_handles_html_tables():
    doc = (
        '<visual id="Table_11" type="table">\n'
        "Table 11. MIC values.\n\n"
        '<table><tr><th>MIC µg/mL</th></tr><tr><td><element id="Table_11/mol_98a" label="98a" '
        'type="molecular_structure">[Unparsed: C5 substituent]</element></td></tr></table>\n'
        "</visual>\n\n"
        '<visual id="Table_12" type="table">\n'
        "Table 12. Complete HTML table.\n\n"
        "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>\n"
        "</visual>"
    )
    unparsed = extract_unparsed_visual_ids(doc)
    assert unparsed == ["Table_11/mol_98a"]


def test_service_passes_local_and_template_context(tmp_path: Path, monkeypatch):
    cfg = IngestionConfig(
        provider="gemini",
        ingestion_dir=tmp_path / "ingestion",
        gemini=GeminiParserConfig(
            defaults=GeminiDefaultsConfig(model="gemini-3.8-flash", temperature=0.0)
        ),
    )
    doc_id = "ctx_doc"
    artifacts = ArtifactManager(doc_id, base_dir=cfg.ingestion_dir)
    artifacts.document_md_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.document_md_path.write_text(
        "<!-- document_id: ctx_doc -->\n\n"
        "### 2.2. Indoline Derivatives\n\n"
        "Reaction of propargyl benzimidazole 33 with 2-azido-1-(indolin-1-yl)ethanone 34 gave 35.\n\n"
        '<visual id="Scheme_8" type="molecular_structure">\n'
        "Scheme 8. Synthesis of 35a-35g.\n"
        "> Synthesis scheme\n"
        "</visual>",
        encoding="utf-8",
    )
    artifacts.save_page_bboxes(
        1,
        PageLocalization(
            page=1, boxes=[VisualBox(id="Scheme_8", found=True, box_2d=[0, 0, 500, 500])]
        ),
    )
    crop_path = artifacts.crops_dir / "Scheme_8__p1.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake_png")

    captured: dict[str, str] = {}

    def _mock_decompose(
        client, visual_id, visual_crops, caption, visual_type, arts, config, description="", local_context=""
    ):
        captured["decompose_local_context"] = local_context
        return VisualDecomposition(
            mode="composite",
            composition_template='<element id="mol_35" />\n35a: R1 = H',
            elements=[
                ElementManifestItem(
                    element_id="mol_35",
                    type="molecular_structure",
                    label="35a",
                    description="Product scaffold 35a on right",
                )
            ],
        )

    def _mock_parse_single(client, elem_mv, crop_list, caption, config, context=""):
        captured["parse_element_context"] = context
        return MolecularParseResult(
            entities=[ChemicalEntity(label="35a", smiles="O=C(Cn1ccnn1)N1CCc2ccccc21")]
        )

    monkeypatch.setattr("ae.ingestion.service.decompose_visual", _mock_decompose)
    monkeypatch.setattr(
        "ae.ingestion.service.localize_elements",
        lambda *args, **kwargs: ElementLocalizationResult(boxes=[]),
    )
    monkeypatch.setattr("ae.ingestion.service.parse_single_visual", _mock_parse_single)

    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 dummy")
    mock_page = MagicMock()
    mock_page.rect = fitz.Rect(0, 0, 600, 800)
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    monkeypatch.setattr("ae.ingestion.service.fitz.open", lambda *args, **kwargs: mock_doc)

    service = VisualParsingService(config=cfg, pdf_dir=tmp_path, ingestion_dir=cfg.ingestion_dir)
    service._client = MagicMock()

    results = service.parse_visuals(doc_id, ["Scheme_8"])
    assert results["Scheme_8"].status == "success"
    assert "indolin-1-yl" in captured["decompose_local_context"]
    assert "Element Placement in Visual:" in captured["parse_element_context"]
    assert "35a: R1 = H" in captured["parse_element_context"]
    assert "Surrounding Article Context:" in captured["parse_element_context"]
    assert "indolin-1-yl" in captured["parse_element_context"]
