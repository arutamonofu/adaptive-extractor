"""Unit tests for VisualParsingService and visual parsing tool."""

from unittest.mock import MagicMock
from pathlib import Path
import pytest

from ae.core.config.optimization_settings import IngestionConfig, GeminiParserConfig, GeminiDefaultsConfig
from ae.ingestion.parsers.gemini.schemas import (
    ChartParseResult,
    ChemicalEntity,
    ElementLocalizationResult,
    ElementManifestItem,
    ElementSelectionResult,
    ManifestVisual,
    MolecularParseResult,
    TableParseResult,
    VisualBox,
    VisualDecomposition,
    VisualParseRecord,
)
from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.stages.parse_visuals import compose_parse_prompt
from ae.ingestion.parsers.gemini.stages.render_markdown import render_composite_block
from ae.ingestion.parsers.gemini.stages.segment_and_crop_visuals_elements import _compute_target_dpi
from ae.ingestion.service import VisualParsingService
from ae.ingestion.tool import make_visual_parsing_tool


@pytest.fixture
def mock_ingestion_config(tmp_path):
    return IngestionConfig(
        provider="gemini",
        ingestion_dir=tmp_path / "ingestion",
        gemini=GeminiParserConfig(
            defaults=GeminiDefaultsConfig(
                model="gemini-3.8-flash",
                temperature=0.0,
            ),
        ),
    )


def test_make_visual_parsing_tool_signature():
    service = MagicMock(spec=VisualParsingService)
    tool = make_visual_parsing_tool("doc_123", service=service)
    assert tool.name == "parse_visuals"
    assert "visuals" in tool.description.lower()


def test_tool_invocation_calls_service():
    service = MagicMock(spec=VisualParsingService)
    service.parse_visuals.return_value = {
        "Figure_1": VisualParseRecord(
            visual_id="Figure_1",
            status="success",
            mode="single",
            rendered_markdown="| Col A | Col B |\n| --- | --- |\n| 1 | 2 |",
        )
    }

    tool = make_visual_parsing_tool("doc_123", service=service)
    output = tool.invoke({"visual_ids": ["Figure_1"]})

    service.parse_visuals.assert_called_once_with("doc_123", ["Figure_1"])
    assert '<visual id="Figure_1">' in output
    assert "| Col A | Col B |" in output
    assert "</visual>" in output


def test_service_cached_parse(tmp_path, mock_ingestion_config):
    doc_id = "test_doc"
    artifacts = ArtifactManager(doc_id, base_dir=mock_ingestion_config.ingestion_dir)
    parsed_visuals_dir = artifacts.parsed_visuals_dir
    parsed_visuals_dir.mkdir(parents=True, exist_ok=True)

    # Put a cached parse record
    rec = VisualParseRecord(
        visual_id="Table_1",
        status="success",
        mode="single",
        rendered_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
    )
    (parsed_visuals_dir / "Table_1.json").write_text(rec.model_dump_json(), encoding="utf-8")

    # Put a markdown file with manifest stub
    doc_md = artifacts.document_md_path
    doc_md.parent.mkdir(parents=True, exist_ok=True)
    doc_md.write_text(
        "<!-- document_id: test_doc -->\n\n"
        "Some text before\n\n"
        '<visual id="Table_1" type="table">\n'
        "Table 1. Experimental data\n"
        "> Table 1 description\n"
        "</visual>\n\n"
        "Some text after",
        encoding="utf-8",
    )

    pdf_dummy = tmp_path / "test_doc.pdf"
    pdf_dummy.write_bytes(b"%PDF-1.4 dummy")

    service = VisualParsingService(
        config=mock_ingestion_config,
        pdf_dir=tmp_path,
        ingestion_dir=mock_ingestion_config.ingestion_dir,
    )

    results = service.parse_visuals(doc_id, ["Table_1"])
    assert "Table_1" in results
    assert results["Table_1"].status == "success"

    # Verify markdown in-place update
    updated_md = doc_md.read_text(encoding="utf-8")
    assert '<visual id="Table_1" type="table">' in updated_md
    assert "| A | B |" in updated_md
    assert "> Table 1 description" not in updated_md


def test_render_composite_block_molecular_assembly():
    decomp = VisualDecomposition(
        mode="composite",
        composition_template="<element id=\"mol_1\" /> + <element id=\"mol_2\" /> -> <element id=\"mol_3\" />",
        elements=[
            ElementManifestItem(element_id="mol_1", type="molecular_structure", label="1", description="Reactant 1"),
            ElementManifestItem(element_id="mol_2", type="molecular_structure", label="2", description="Reactant 2"),
            ElementManifestItem(element_id="mol_3", type="molecular_structure", label="3a-3b", description="Products"),
        ],
    )

    element_results = {
        "mol_1": MolecularParseResult(
            status="success",
            entities=[ChemicalEntity(label="1", smiles="CC", role="starting_material")],
        ),
        "mol_2": MolecularParseResult(
            status="success",
            entities=[ChemicalEntity(label="2", smiles="NN", role="starting_material")],
        ),
        "mol_3": MolecularParseResult(
            status="success",
            entities=[
                ChemicalEntity(label="3a", smiles="CCN", role="target_product"),
                ChemicalEntity(label="3b", smiles="CCNN", role="target_product"),
            ],
        ),
    }

    rendered = render_composite_block(decomp, element_results)

    # Narrative reaction equation should have clean bold labels
    assert "**1** + **2** -> **3a-3b**" in rendered

    # Unified deduplicated entities table rendered at the bottom
    assert "**Chemical Entities Registry:**" in rendered
    assert "| **1** | `CC` |" in rendered
    assert "| **2** | `NN` |" in rendered
    assert "| **3a** | `CCN` |" in rendered
    assert "| **3b** | `CCNN` |" in rendered


def test_compose_parse_prompt_includes_panel_label_focus():
    vis = ManifestVisual(
        id="mol_5a_5f",
        type="molecular_structure",
        description="Azide sulfonamide intermediates 5a-5f with shared R-group legend a-f",
        panel_label="5a-5f",
    )
    prompt, schema = compose_parse_prompt(vis, caption="Scheme 2", crops_count=2)
    assert schema == MolecularParseResult
    assert "Focus: Extract ONLY the chemical entity/series for label '5a-5f'" in prompt
    assert "Multiple cropped images are provided" in prompt


def test_render_composite_block_mixed_chart_and_molecules():
    decomp = VisualDecomposition(
        mode="composite",
        composition_template="Panel A:\n<element id=\"chart_a\" />\n\nPanel B: Lead <element id=\"mol_b\" />",
        elements=[
            ElementManifestItem(element_id="chart_a", type="chart", label="(a)", description="Dose-response plot"),
            ElementManifestItem(element_id="mol_b", type="molecular_structure", label="6a", description="Structure of 6a"),
        ],
    )
    element_results = {
        "chart_a": ChartParseResult(
            status="success",
            content="**MIC Comparison**\n\n| Compound | MIC |\n| :--- | :--- |\n| 6a | 64 |",
        ),
        "mol_b": MolecularParseResult(
            status="success",
            entities=[
                ChemicalEntity(label="6a", smiles="c1ccccc1"),
                ChemicalEntity(label="6a", smiles="c1ccccc1"),  # duplicate
            ],
        ),
    }

    rendered = render_composite_block(decomp, element_results)
    assert "**MIC Comparison**" in rendered
    assert "| Compound | MIC |" in rendered
    assert "Panel B: Lead **6a**" in rendered
    assert rendered.count("| **6a** | `c1ccccc1` |") == 1


def test_table_and_chart_schema_pruning():
    t_schema = TableParseResult.model_json_schema()
    assert list(t_schema["properties"].keys()) == ["content"]
    assert "status" not in t_schema["properties"]
    assert "format" not in t_schema["properties"]
    assert "warnings" not in t_schema["properties"]

    c_schema = ChartParseResult.model_json_schema()
    assert list(c_schema["properties"].keys()) == ["content"]
    assert "status" not in c_schema["properties"]
    assert "tables" not in c_schema["properties"]
    assert "warnings" not in c_schema["properties"]
    assert "$defs" not in c_schema

    m_schema = ManifestVisual.model_json_schema()
    assert "contains_types" not in m_schema["properties"]


def test_compute_target_dpi_adaptive_upscale(mock_ingestion_config):
    import pymupdf as fitz

    crop_cfg = mock_ingestion_config.gemini.crop
    # Small 100x60 pt atom rect -> long side = 100 pt -> (2000 * 72) / 100 = 1440 -> clamped to max_dpi=1200
    small_rect = fitz.Rect(0, 0, 100, 60)
    dpi = _compute_target_dpi(small_rect, crop_cfg, default_base_dpi=400)
    assert dpi == 1200


def test_molecular_parse_result_schema_pruning_and_backward_compatibility():
    schema = MolecularParseResult.model_json_schema()
    props = schema.get("properties", {})
    # Only entities should be exposed in root schema properties
    assert "entities" in props
    assert "reactions" not in props
    assert "scheme_type" not in props
    assert "summary" not in props
    assert "warnings" not in props
    assert "scaffold_smiles" not in props
    assert "status" not in props

    entity_props = schema.get("$defs", {}).get("ChemicalEntity", {}).get("properties", {})
    # Only label, smiles, and attachment_points should be exposed in ChemicalEntity
    assert "label" in entity_props
    assert "smiles" in entity_props
    assert "attachment_points" in entity_props
    assert "substituents" not in entity_props
    assert "notes" not in entity_props
    assert "name" not in entity_props
    assert "role" not in entity_props
    assert "smiles_valid" not in entity_props
    assert "rdkit_error" not in entity_props

    # Backward compatibility: can load legacy JSON with old fields via extra='ignore'
    legacy_json = {
        "status": "success",
        "scheme_type": "markush_series",
        "summary": "Some scheme",
        "scaffold_smiles": "c1ccccc1",
        "reactions": [{"step_id": "1", "reactants": ["A"], "products": ["B"]}],
        "warnings": ["warn"],
        "entities": [
            {
                "label": "1a",
                "name": "benzene",
                "smiles": "c1ccccc1",
                "role": "target_product",
                "substituents": [{"group": "R", "value": "H"}],
                "notes": "MIC = 5",
            }
        ],
    }
    obj = MolecularParseResult.model_validate(legacy_json)
    assert obj.status == "success"
    assert len(obj.entities) == 1
    assert obj.entities[0].label == "1a"
    assert obj.entities[0].smiles == "c1ccccc1"


def test_molecular_parse_result_invalid_smiles_property():
    res = MolecularParseResult(
        entities=[
            ChemicalEntity(label="1", smiles="c1ccccc1", smiles_valid=True),
            ChemicalEntity(label="2", smiles="INVALID@@", smiles_valid=False, rdkit_error="Parse error"),
        ]
    )
    assert hasattr(res, "invalid_smiles")
    invalids = res.invalid_smiles
    assert len(invalids) == 1
    assert invalids[0] == {"label": "2", "smiles": "INVALID@@", "error": "Parse error"}


def test_render_composite_block_clean_registry_table():
    decomp = VisualDecomposition(
        mode="composite",
        composition_template="Scheme: <element id='m1' />",
        elements=[
            ElementManifestItem(element_id="m1", type="molecular_structure", label="1a-1b", description="Compounds"),
        ],
    )
    element_results = {
        "m1": MolecularParseResult(
            entities=[
                ChemicalEntity(label="1a", smiles="CC"),
                ChemicalEntity(label="1b", smiles="CCO"),
            ]
        )
    }
    rendered = render_composite_block(decomp, element_results)
    assert "**Chemical Entities Registry:**" in rendered
    assert "| Label | SMILES |" in rendered
    assert "| :--- | :--- |" in rendered
    assert "| **1a** | `CC` |" in rendered
    assert "| **1b** | `CCO` |" in rendered


def test_unparsed_visuals_detection():
    from ae.ingestion.parsers.gemini.stages.render_markdown import (
        extract_unparsed_visual_ids,
        has_unparsed_visuals,
    )

    doc_mixed = (
        "# Document\n\n"
        '<visual id="Table_1" type="table">\n'
        "Table 1. Experimental yields\n"
        "> Summary of yields for 1a-1d\n"
        "</visual>\n\n"
        "Some text in between.\n\n"
        '<visual id="Figure_1" type="chart">\n'
        "| Compound | IC50 |\n| :--- | :--- |\n| 1a | 10 nM |\n"
        "</visual>\n\n"
        '<visual id="Scheme_1" type="molecular_structure">\n'
        "Scheme 1. Synthesis\n"
        "> Synthesis of compounds 1a-1d\n"
        "</visual>"
    )

    assert has_unparsed_visuals(doc_mixed) is True
    unparsed = extract_unparsed_visual_ids(doc_mixed)
    assert unparsed == ["Table_1", "Scheme_1"]
    assert "Figure_1" not in unparsed

    doc_all_parsed = (
        '<visual id="Table_1" type="table">\n'
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        "</visual>"
    )
    assert has_unparsed_visuals(doc_all_parsed) is False
    assert extract_unparsed_visual_ids(doc_all_parsed) == []


def test_tool_invocation_with_target_hint():
    service = MagicMock(spec=VisualParsingService)
    service.parse_visuals.return_value = {
        "Scheme_2": VisualParseRecord(
            visual_id="Scheme_2",
            status="partial",
            mode="composite",
            rendered_markdown="**4** + **5a-5f** --> **6a-6f**\n\n**Chemical Entities Registry:**\n| Label | SMILES |\n| :--- | :--- |\n| **6a** | `c1ccccc1` |",
            parsed_element_ids=["mol_6a_6f"],
            unparsed_element_ids=["mol_4", "mol_5a_5f"],
        )
    }

    tool = make_visual_parsing_tool("doc_test", service=service)
    output = tool.invoke({"visual_ids": ["Scheme_2"], "target_hint": "products 6a-6f"})

    service.parse_visuals.assert_called_once_with(
        "doc_test", ["Scheme_2"], target_hint="products 6a-6f"
    )
    assert '<visual id="Scheme_2">' in output
    assert "c1ccccc1" in output


def test_render_composite_block_with_unparsed_elements():
    decomp = VisualDecomposition(
        mode="composite",
        composition_template="<element id=\"mol_1\" /> + <element id=\"mol_2\" /> --> <element id=\"mol_3\" />",
        elements=[
            ElementManifestItem(element_id="mol_1", type="molecular_structure", label="1", description="Reactant 1"),
            ElementManifestItem(element_id="mol_2", type="molecular_structure", label="2", description="Reactant 2"),
            ElementManifestItem(element_id="mol_3", type="molecular_structure", label="3", description="Product 3"),
        ],
    )
    # Only mol_3 is parsed
    element_results = {
        "mol_3": MolecularParseResult(
            entities=[ChemicalEntity(label="3", smiles="CCC")]
        )
    }

    rendered = render_composite_block(decomp, element_results, visual_id="Scheme_9")
    assert '<element id="Scheme_9/mol_1" label="1" type="molecular_structure">[Unparsed: Reactant 1]</element>' in rendered
    assert '<element id="Scheme_9/mol_2" label="2" type="molecular_structure">[Unparsed: Reactant 2]</element>' in rendered
    assert "**3**" in rendered
    assert "| **3** | `CCC` |" in rendered


def test_unparsed_elements_detection_with_partial_visual():
    from ae.ingestion.parsers.gemini.stages.render_markdown import (
        extract_unparsed_visual_ids,
        has_unparsed_visuals,
    )

    doc_partial = (
        "# Results\n\n"
        '<visual id="Scheme_1" type="molecular_structure">\n'
        '<element id="Scheme_1/mol_1" label="1" type="molecular_structure">[Unparsed: Reactant 1]</element> --> **2**\n\n'
        "**Chemical Entities Registry:**\n"
        "| Label | SMILES |\n"
        "| :--- | :--- |\n"
        "| **2** | `CC` |\n"
        "</visual>"
    )

    assert has_unparsed_visuals(doc_partial) is True
    unparsed = extract_unparsed_visual_ids(doc_partial)
    assert unparsed == ["Scheme_1/mol_1"]

    # Also verify backward compatibility with legacy <atom ...> tags
    doc_legacy = (
        '<visual id="Scheme_2" type="molecular_structure">\n'
        '<atom id="Scheme_2/mol_4" label="4" type="molecular_structure">[Unparsed: Reactant 4]</atom>\n'
        '</visual>'
    )
    assert has_unparsed_visuals(doc_legacy) is True
    assert extract_unparsed_visual_ids(doc_legacy) == ["Scheme_2/mol_4"]


def test_select_elements_by_hint_heuristic_and_llm():
    from ae.ingestion.parsers.gemini.stages.segment_and_crop_visuals_elements import select_elements_by_hint
    from ae.ingestion.parsers.gemini.schemas import ElementSelectionResult

    elements = [
        ElementManifestItem(element_id="mol_4", type="molecular_structure", label="4", description="Reactant 4 alkyne"),
        ElementManifestItem(element_id="mol_5a_5f", type="molecular_structure", label="5a-5f", description="Azide intermediate 5a-5f"),
        ElementManifestItem(element_id="mol_6a_6f", type="molecular_structure", label="6a-6f", description="Hybrid products 6a-6f"),
    ]

    # Test 1: Heuristic match on label "6a"
    res1 = select_elements_by_hint(None, "Scheme_2", elements, "target products 6a-6f", None)
    assert res1 == ["mol_6a_6f"]

    # Test 2: Heuristic match on element_id "mol_4"
    res2 = select_elements_by_hint(None, "Scheme_2", elements, "only mol_4", None)
    assert res2 == ["mol_4"]

    # Test 3: LLM fallback when label/id is not verbatim in hint
    mock_client = MagicMock()
    mock_client.generate_structured.return_value = ElementSelectionResult(
        selected_element_ids=["mol_6a_6f"]
    )
    res3 = select_elements_by_hint(mock_client, "Scheme_2", elements, "antibacterial conjugates with MIC", None)
    assert res3 == ["mol_6a_6f"]
    mock_client.generate_structured.assert_called_once()


def test_strip_pdf_extension():
    from ae.ingestion.service import strip_pdf_extension

    assert strip_pdf_extension("acsomega.2c06142") == "acsomega.2c06142"
    assert strip_pdf_extension("acsomega.2c06142.pdf") == "acsomega.2c06142"
    assert strip_pdf_extension("acsomega.2c06142.PDF") == "acsomega.2c06142"
    assert strip_pdf_extension("data/raw/pdf/acsomega.2c06142.pdf") == "acsomega.2c06142"
    assert strip_pdf_extension("10.1021_acs.jmedchem.9b01150") == "10.1021_acs.jmedchem.9b01150"


def test_locate_pdf_with_dotted_doc_id(tmp_path):
    pdf_file = tmp_path / "acsomega.2c06142.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    service = VisualParsingService(pdf_dir=tmp_path)

    found_by_clean = service.locate_pdf("acsomega.2c06142")
    assert found_by_clean == pdf_file.resolve()

    found_by_full = service.locate_pdf("acsomega.2c06142.pdf")
    assert found_by_full == pdf_file.resolve()


def test_locate_pdf_case_insensitive(tmp_path):
    pdf_file = tmp_path / "RJC.2023.1638382.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    service = VisualParsingService(pdf_dir=tmp_path)

    found = service.locate_pdf("rjc.2023.1638382")
    assert found == pdf_file.resolve()


def test_resolve_doc_name_preserves_case_and_dots(tmp_path):
    ingestion_dir = tmp_path / "ingestion"
    (ingestion_dir / "acsomega.2c06142").mkdir(parents=True)
    (ingestion_dir / "RJC.2023.1638382").mkdir(parents=True)

    service = VisualParsingService(ingestion_dir=ingestion_dir)

    assert service._resolve_doc_name("acsomega.2c06142") == "acsomega.2c06142"
    assert service._resolve_doc_name("acsomega.2c06142.pdf") == "acsomega.2c06142"
    assert service._resolve_doc_name("rjc.2023.1638382") == "RJC.2023.1638382"
    assert service._resolve_doc_name("RJC.2023.1638382") == "RJC.2023.1638382"


def test_tool_formatting_includes_partial_notice_and_empty_handling():
    from ae.ingestion.tool import _format_visual_results

    assert _format_visual_results({}) == "No visual results returned."

    partial_out = _format_visual_results(
        {
            "Fig_2": VisualParseRecord(
                visual_id="Fig_2",
                status="partial",
                mode="composite",
                rendered_markdown="| item | val |\n|---|---|\n| A | 1 |",
                parsed_element_ids=["panel_a"],
                unparsed_element_ids=["panel_b"],
            )
        }
    )
    assert '<visual id="Fig_2">' in partial_out
    assert "[Partial parse for Fig_2: unparsed elements remain: panel_b." in partial_out


def test_service_single_visual_failure_preserves_stub_and_marks_failed(tmp_path, mock_ingestion_config, monkeypatch):
    doc_id = "doc_fail_single"
    artifacts = ArtifactManager(doc_id, base_dir=mock_ingestion_config.ingestion_dir)
    doc_md = artifacts.document_md_path
    doc_md.parent.mkdir(parents=True, exist_ok=True)
    original_md = (
        "<!-- document_id: doc_fail_single -->\n\n"
        '<visual id="Table_1" type="table">\n'
        "Table 1. Metrics\n"
        "> Important table description\n"
        "</visual>"
    )
    doc_md.write_text(original_md, encoding="utf-8")

    # Pre-seed localization & crop so no PDF/Gemini calls are needed
    from ae.ingestion.parsers.gemini.schemas import PageLocalization, VisualBox

    artifacts.save_page_bboxes(1, PageLocalization(page=1, boxes=[VisualBox(id="Table_1", found=True, box_2d=[0, 0, 500, 500])]))
    crop_path = artifacts.crops_dir / "Table_1__p1.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake_png")
    (artifacts.decompositions_dir).mkdir(parents=True, exist_ok=True)
    artifacts.save_json(
        artifacts.decompositions_dir / "Table_1.json",
        VisualDecomposition(mode="single", single_type="table"),
    )

    monkeypatch.setattr(
        "ae.ingestion.service.parse_single_visual",
        lambda *args, **kwargs: TableParseResult(status="failed", content=""),
    )
    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 dummy")

    service = VisualParsingService(
        config=mock_ingestion_config,
        pdf_dir=tmp_path,
        ingestion_dir=mock_ingestion_config.ingestion_dir,
    )
    service._client = MagicMock()
    # Mock fitz.open so it doesn't need a real PDF
    import pymupdf as fitz

    mock_page = MagicMock()
    mock_page.rect = fitz.Rect(0, 0, 600, 800)
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    monkeypatch.setattr("ae.ingestion.service.fitz.open", lambda *args, **kwargs: mock_doc)

    results = service.parse_visuals(doc_id, ["Table_1"])
    assert results["Table_1"].status == "failed"
    # Original stub and description on disk must NOT be overwritten with *[Parsing failed]*
    assert doc_md.read_text(encoding="utf-8") == original_md


def test_service_composite_failed_element_remains_unparsed(tmp_path, mock_ingestion_config, monkeypatch):
    doc_id = "doc_fail_elem"
    artifacts = ArtifactManager(doc_id, base_dir=mock_ingestion_config.ingestion_dir)
    doc_md = artifacts.document_md_path
    doc_md.parent.mkdir(parents=True, exist_ok=True)
    doc_md.write_text(
        "<!-- document_id: doc_fail_elem -->\n\n"
        '<visual id="Fig_1" type="chart">\nFig 1\n> Panels A and B\n</visual>',
        encoding="utf-8",
    )

    from ae.ingestion.parsers.gemini.schemas import PageLocalization

    artifacts.save_page_bboxes(1, PageLocalization(page=1, boxes=[VisualBox(id="Fig_1", found=True, box_2d=[0, 0, 500, 500])]))
    crop_path = artifacts.crops_dir / "Fig_1__p1.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake_png")
    artifacts.decompositions_dir.mkdir(parents=True, exist_ok=True)
    artifacts.save_json(
        artifacts.decompositions_dir / "Fig_1.json",
        VisualDecomposition(
            mode="composite",
            composition_template='<element id="panel_a" />\n<element id="panel_b" />',
            elements=[
                ElementManifestItem(element_id="panel_a", type="chart", label="A", description="Panel A"),
                ElementManifestItem(element_id="panel_b", type="chart", label="B", description="Panel B"),
            ],
        ),
    )
    artifacts.element_bboxes_dir.mkdir(parents=True, exist_ok=True)
    artifacts.save_json(
        artifacts.element_bboxes_dir / "Fig_1.json",
        ElementLocalizationResult(boxes=[]),
    )

    def _mock_parse(client, elem_mv, *args, **kwargs):
        if elem_mv.id == "panel_a":
            return ChartParseResult(status="success", content="| x | y |\n|---|---|\n| 1 | 2 |")
        return ChartParseResult(status="failed", content="")

    monkeypatch.setattr("ae.ingestion.service.parse_single_visual", _mock_parse)
    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 dummy")
    import pymupdf as fitz

    mock_page = MagicMock()
    mock_page.rect = fitz.Rect(0, 0, 600, 800)
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    monkeypatch.setattr("ae.ingestion.service.fitz.open", lambda *args, **kwargs: mock_doc)

    service = VisualParsingService(
        config=mock_ingestion_config,
        pdf_dir=tmp_path,
        ingestion_dir=mock_ingestion_config.ingestion_dir,
    )
    service._client = MagicMock()

    results = service.parse_visuals(doc_id, ["Fig_1"])
    rec = results["Fig_1"]
    assert rec.status == "partial"
    assert rec.parsed_element_ids == ["panel_a"]
    assert rec.unparsed_element_ids == ["panel_b"]
    assert '<element id="Fig_1/panel_b"' in rec.rendered_markdown




