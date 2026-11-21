"""Unit tests for the new HITL, GTEditor, and MarkdownEditor mechanisms in Reverse Engineering."""

import json
from pathlib import Path
import pandas as pd
import pytest

from ae.semantic_optimization.hitl import (
    GTEditor,
    InteractiveConflictResolver,
    MarkdownEditor,
    RulingsStore,
    _is_descriptive_rule_text,
)
from ae.semantic_optimization.models import (
    ExpertRuling,
    REConflict,
    ResolutionOption,
)


@pytest.fixture
def gt_test_files(tmp_path: Path):
    source_gt = tmp_path / "original_gt.csv"
    data = {
        "pdf": ["paper_1.pdf", "paper_1.pdf", "paper_2.pdf"],
        "formula": ["Fe3O4", "ZnO", "CuO"],
        "yield": [85.5, 90.0, 78.0],
    }
    pd.DataFrame(data).to_csv(source_gt, index=False)
    corrected_gt = tmp_path / "corrected_gt.csv"
    return source_gt, corrected_gt


@pytest.fixture
def ingestion_test_env(tmp_path: Path):
    ingest_dir = tmp_path / "ingestion"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    paper1_md = ingest_dir / "paper_1.md"
    paper1_content = (
        "# Title\n\n"
        "We synthesized compound Fe3O4 with high purity.\n\n"
        '<visual id="Table_1" type="table">\n'
        "Table 1. Optimization conditions\n"
        "> Summary of catalysts and yields\n"
        "</visual>\n"
    )
    paper1_md.write_text(paper1_content, encoding="utf-8")
    return ingest_dir


def test_gt_editor_lifecycle(gt_test_files):
    source_gt, corrected_gt = gt_test_files

    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt, resume=False)
    assert corrected_gt.exists()
    assert len(editor.df) == 3

    success = editor.update_cell(doc_id="paper_1", row_id_or_index="row_1", field_name="formula", new_value="Fe2O3")
    assert success is True
    df_disk = pd.read_csv(corrected_gt)
    assert df_disk.loc[0, "formula"] == "Fe2O3"
    df_orig = pd.read_csv(source_gt)
    assert df_orig.loc[0, "formula"] == "Fe3O4"

    editor.add_row(doc_id="paper_1", row_dict={"formula": "TiO2", "yield": 95.0})
    assert len(editor.df) == 4
    df_disk2 = pd.read_csv(corrected_gt)
    assert len(df_disk2) == 4
    assert df_disk2.iloc[-1]["formula"] == "TiO2"

    editor.drop_row(doc_id="paper_2", row_id_or_index="row_1")
    assert len(editor.df) == 3
    df_disk3 = pd.read_csv(corrected_gt)
    assert "paper_2.pdf" not in df_disk3["pdf"].values


def test_markdown_editor_patch_and_backup(ingestion_test_env):
    md_editor = MarkdownEditor(ingestion_dir=ingestion_test_env)

    # 1. Inspect unparsed visuals
    visuals = md_editor.get_unparsed_visuals_summary("paper_1")
    assert "Table_1" in visuals

    # 2. Patch markdown file
    paper1_file = ingestion_test_env / "paper_1.md"
    assert "Fe3O4" in paper1_file.read_text(encoding="utf-8")

    success = md_editor.patch_markdown("paper_1", old_value="Fe3O4", new_value="Fe2O3")
    assert success is True

    # Verify original backup created
    bak_file = ingestion_test_env / "paper_1.md.bak"
    assert bak_file.exists()
    assert "Fe3O4" in bak_file.read_text(encoding="utf-8")

    # Verify patched content
    assert "Fe2O3" in paper1_file.read_text(encoding="utf-8")


def test_conflict_resolver_interactive_patch_ingested_md(gt_test_files, ingestion_test_env, tmp_path: Path):
    source_gt, corrected_gt = gt_test_files
    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    md_editor = MarkdownEditor(ingestion_dir=ingestion_test_env)
    store = RulingsStore(storage_path=tmp_path / "rulings_md.json")

    conflict = REConflict(
        conflict_id="conf_vlm_typo",
        scope="column:formula",
        conflict_type="TEXT_VS_GT_MISMATCH",
        document_id="paper_1",
        row_id="row_1",
        raw_text_quote="Fe3O4",
        gt_value="Fe2O3",
        conflicting_value_or_rule="Fe3O4 in .md text vs Fe2O3 in GT",
        explanation="Text says Fe3O4 due to VLM error, GT has Fe2O3",
        options=[],
    )

    # Simulate user choosing PATCH_INGESTED_MD option
    # We provide input_fn: find option index for PATCH_INGESTED_MD
    selected_option_index = None

    def mock_input(prompt: str) -> str:
        # Resolver prints options then asks "Select option [1-N]: "
        # We simulate choosing option 2 (PATCH_INGESTED_MD)
        return "2"

    resolver = InteractiveConflictResolver(
        gt_editor=editor,
        rulings_store=store,
        md_editor=md_editor,
        interactive=True,
        input_fn=mock_input,
    )

    ruling, needs_recompute = resolver.resolve(conflict)
    assert needs_recompute is True
    assert ruling.action_type == "PATCH_INGESTED_MD"

    # GT must NOT be modified
    assert editor.df.loc[0, "formula"] == "Fe3O4"

    # .md must be patched with GT value (Fe2O3)
    md_text = (ingestion_test_env / "paper_1.md").read_text(encoding="utf-8")
    assert "Fe2O3" in md_text


def test_conflict_resolver_ignore_ingestion_defect(gt_test_files, ingestion_test_env, tmp_path: Path):
    source_gt, corrected_gt = gt_test_files
    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    md_editor = MarkdownEditor(ingestion_dir=ingestion_test_env)
    store = RulingsStore(storage_path=tmp_path / "rulings_ign.json")

    conflict = REConflict(
        conflict_id="conf_defect",
        scope="column:temperature",
        conflict_type="SUSPECTED_INGESTION_DEFECT",
        document_id="paper_1",
        row_id="row_1",
        raw_text_quote="RT",
        gt_value="25.0",
        conflicting_value_or_rule="OCR misread temperature",
        explanation="OCR defect in text",
        options=[],
    )

    # Option 4 is IGNORE_INGESTION_DEFECT (1=PATCH_GT_CELL, 2=PATCH_INGESTED_MD, 3=UNPACK_VISUAL_IN_MD, 4=IGNORE_INGESTION_DEFECT)
    resolver = InteractiveConflictResolver(
        gt_editor=editor,
        rulings_store=store,
        md_editor=md_editor,
        interactive=True,
        input_fn=lambda prompt: "4",
    )

    ruling, needs_recompute = resolver.resolve(conflict)
    assert needs_recompute is False
    assert ruling.action_type == "IGNORE_INGESTION_DEFECT"
    assert store.is_ignored("paper_1", "row_1", "temperature") is True


def test_resolve_batch_digest_and_baseline_conflict(gt_test_files, ingestion_test_env, tmp_path: Path):
    from ae.semantic_optimization.context import ArtifactManager

    source_gt, corrected_gt = gt_test_files
    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    md_editor = MarkdownEditor(ingestion_dir=ingestion_test_env)
    store = RulingsStore(storage_path=tmp_path / "rulings_batch.json")
    artifacts = ArtifactManager(tmp_path / "art", resume=True)

    # Pre-populate a positive column artifact so in-place patching can succeed without needing LLM rerun
    pos_col_path = artifacts.get_path("positive", "columns", "paper_1_row_1.json", iteration=None)
    pos_col_path.parent.mkdir(parents=True, exist_ok=True)
    pos_col_path.write_text(
        json.dumps(
            {
                "fields": [
                    {
                        "row_id": "row_1",
                        "field_name": "formula",
                        "raw_text": "Fe2O3",
                        "ground_truth": "Fe3O4",
                        "Suspected_Ingestion_Or_GT_Issue": "GT typo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    conflict_baseline = REConflict(
        conflict_id="conf_base_1",
        scope="rows",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="paper_1",
        row_id="row_1",
        gt_value="Excluded commercial control",
        conflicting_value_or_rule="Baseline prompt says include all reference controls",
        explanation="Baseline prompt contradicts GT row filtering",
        options=[],
    )
    conflict_gt_cell = REConflict(
        conflict_id="conf_gt_cell_1",
        scope="column:formula",
        conflict_type="SUSPECTED_GT_ERROR",
        document_id="paper_1",
        row_id="row_1",
        raw_text_quote="Fe2O3",
        gt_value="Fe3O4",
        conflicting_value_or_rule="Fe2O3",
        explanation="GT has Fe3O4 but text clearly states Fe2O3",
        options=[],
    )

    # For conflict_baseline (sorted 1st due to priority): option 5 is OVERRIDE_BASELINE
    # (options: 1=PATCH_INGESTED_MD, 2=UNPACK_VISUAL_IN_MD, 3=IGNORE_INGESTION_DEFECT, 4=PATCH_GT_DROP_ROW, 5=OVERRIDE_BASELINE, 6=KEEP_BASELINE, 7=ADD_CONDITION)
    # For conflict_gt_cell (sorted 2nd): option 1 is PATCH_GT_CELL
    answers = iter(["5", "1"])

    resolver = InteractiveConflictResolver(
        gt_editor=editor,
        rulings_store=store,
        md_editor=md_editor,
        interactive=True,
        input_fn=lambda _: next(answers),
        artifacts=artifacts,
    )

    batch_res = resolver.resolve_batch([conflict_gt_cell, conflict_baseline])
    assert len(batch_res.rulings) == 2
    assert batch_res.rulings[0].action_type == "OVERRIDE_BASELINE"
    assert batch_res.rulings[1].action_type == "PATCH_GT_CELL"
    # In-place artifact patch succeeded -> no full document LLM rerun needed!
    assert batch_res.needs_doc_recompute is False
    assert batch_res.needs_rule_recompute is True
    assert batch_res.recompute_scopes == {"rows", "column:formula"}

    # Verify positive artifact was patched in-place
    patched_data = json.loads(pos_col_path.read_text(encoding="utf-8"))
    assert patched_data["fields"][0]["ground_truth"] == "Fe2O3"
    assert editor.df.loc[0, "formula"] == "Fe2O3"


def test_gt_editor_invalid_indices(gt_test_files):
    source_gt, corrected_gt = gt_test_files
    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)

    # 1. Non-existent document
    assert editor.update_cell("nonexistent_doc", "row_1", "formula", "val") is False
    assert editor.drop_row("nonexistent_doc", "row_1") is False

    # 2. Out-of-bounds row_id (paper_1 has 2 rows)
    assert editor.update_cell("paper_1", "row_99", "formula", "val") is False
    assert editor.drop_row("paper_1", "row_99") is False

    # Ensure dataset length was not changed
    assert len(editor.df) == 3


def test_markdown_editor_single_occurrence_replace(ingestion_test_env):
    md_editor = MarkdownEditor(ingestion_dir=ingestion_test_env)
    paper1_file = ingestion_test_env / "paper_1.md"
    paper1_file.write_text("repeat_pattern and repeat_pattern", encoding="utf-8")

    success = md_editor.patch_markdown("paper_1", old_value="repeat_pattern", new_value="single_replaced")
    assert success is True
    patched_content = paper1_file.read_text(encoding="utf-8")
    assert patched_content == "single_replaced and repeat_pattern"


def test_rulings_store_format_rulings_for_prompt_filtering(tmp_path: Path):
    store = RulingsStore(storage_path=tmp_path / "rulings_filter.json")
    store.add_ruling(
        ExpertRuling(
            conflict_id="c1",
            scope="rows",
            conflict_type="BASELINE_POLICY_CONFLICT",
            action_type="OVERRIDE_BASELINE",
            decision_text="Keep commercial control rows",
            rationale="Standard industry practice",
        )
    )
    store.add_ruling(
        ExpertRuling(
            conflict_id="c2",
            scope="column:yield",
            conflict_type="CONDITIONAL_BRANCH_NEEDED",
            action_type="ADD_CONDITION",
            decision_text="Yield must be > 50%",
            rationale="Filter low yield",
        )
    )
    store.add_ruling(
        ExpertRuling(
            conflict_id="c3",
            scope="rows",
            conflict_type="SUSPECTED_GT_OMISSION",
            action_type="PATCH_GT_ADD_ROW",
            decision_text="Added row for compound 5",
            document_id="paper_1",
        )
    )
    store.add_ruling(
        ExpertRuling(
            conflict_id="c4",
            scope="rows",
            conflict_type="SUSPECTED_GT_ERROR",
            action_type="PATCH_GT_DROP_ROW",
            decision_text="Dropped row 2",
            document_id="paper_1",
            row_id="row_2",
        )
    )
    store.add_ruling(
        ExpertRuling(
            conflict_id="c5",
            scope="rows",
            conflict_type="SUSPECTED_INGESTION_DEFECT",
            action_type="PATCH_INGESTED_MD",
            decision_text="Fixed OCR typo in table 1",
            document_id="paper_1",
        )
    )

    # 1. only_policy_rules=True (used in prompt generation)
    policy_prompt = store.format_rulings_for_prompt("all", only_policy_rules=True)
    assert "Keep commercial control rows" in policy_prompt
    assert "Yield must be > 50%" in policy_prompt
    assert "PATCH_GT_ADD_ROW" not in policy_prompt
    assert "PATCH_GT_DROP_ROW" not in policy_prompt
    assert "PATCH_INGESTED_MD" not in policy_prompt

    # 2. only_policy_rules=False (displays all rulings)
    full_prompt = store.format_rulings_for_prompt("all", only_policy_rules=False)
    assert "PATCH_GT_ADD_ROW" in full_prompt
    assert "PATCH_GT_DROP_ROW" in full_prompt
    assert "PATCH_INGESTED_MD" in full_prompt


def test_conflict_resolver_options_for_omissions_and_rows(gt_test_files, ingestion_test_env, tmp_path: Path, capsys):
    source_gt, corrected_gt = gt_test_files
    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    md_editor = MarkdownEditor(ingestion_dir=ingestion_test_env)
    store = RulingsStore(storage_path=tmp_path / "rulings_options.json")

    # Conflict 1: SUSPECTED_GT_OMISSION with empty row_id
    conflict_omission = REConflict(
        conflict_id="c_omission",
        scope="rows",
        conflict_type="SUSPECTED_GT_OMISSION",
        document_id="paper_1",
        row_id="",
        conflicting_value_or_rule="Missing entity in Table 2",
        explanation="Candidate entity was extracted from text but not in GT",
        options=[],
    )

    resolver = InteractiveConflictResolver(
        gt_editor=editor,
        rulings_store=store,
        md_editor=md_editor,
        interactive=True,
        input_fn=lambda _: "1",
    )

    resolver.resolve(conflict_omission)
    out1 = capsys.readouterr().out
    assert "(PATCH_GT_ADD_ROW)" in out1
    assert "(PATCH_GT_DROP_ROW)" not in out1

    # Conflict 2: Error on concrete row_1
    conflict_row = REConflict(
        conflict_id="c_row",
        scope="rows",
        conflict_type="SUSPECTED_GT_ERROR",
        document_id="paper_1",
        row_id="row_1",
        explanation="Incorrect row",
        options=[],
    )
    resolver.resolve(conflict_row)
    out2 = capsys.readouterr().out
    assert "(PATCH_GT_DROP_ROW)" in out2


def test_gt_editor_multi_row_drop_without_index_shift(tmp_path: Path):
    source_gt = tmp_path / "gt_multi.csv"
    corrected_gt = tmp_path / "gt_multi_corrected.csv"
    pd.DataFrame(
        {
            "pdf": ["doc_a", "doc_a", "doc_a", "doc_a", "doc_b"],
            "compound": ["c1", "c2", "c3", "c4", "b1"],
            "val": [10, 20, 30, 40, 50],
        }
    ).to_csv(source_gt, index=False)

    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt, resume=False)

    # Drop row_2 (c2) and row_3 (c3) in the same batch before reindexing
    assert editor.drop_row("doc_a", "row_2") is True
    assert editor.drop_row("doc_a", "row_3") is True
    # Update original row_4 (c4) in the same batch — must still point to c4, not shift!
    assert editor.update_cell("doc_a", "row_4", "val", 44) is True
    # Re-dropping already-dropped row_2 before reindex must return False (not silently drop row_4)
    assert editor.drop_row("doc_a", "row_2") is False

    disk_df = pd.read_csv(corrected_gt)
    assert "_re_row_id" not in disk_df.columns
    doc_a_rows = disk_df[disk_df["pdf"] == "doc_a"]
    assert doc_a_rows["compound"].tolist() == ["c1", "c4"]
    assert doc_a_rows["val"].tolist() == [10, 44]

    # After reindexing, doc_a now has row_1 (c1) and row_2 (c4)
    editor.reindex_doc_rows("doc_a")
    assert editor.update_cell("doc_a", "row_2", "val", 99) is True
    disk_df2 = pd.read_csv(corrected_gt)
    assert disk_df2[disk_df2["pdf"] == "doc_a"]["val"].tolist() == [10, 99]


def test_rulings_store_semantic_dedup_across_changed_conflict_ids(gt_test_files, tmp_path: Path):
    """Conflicts already covered by rulings/ignored_examples must not re-prompt even if LLM emits a new conflict_id."""
    source_gt, corrected_gt = gt_test_files
    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    store = RulingsStore(storage_path=tmp_path / "rulings_semantic.json")

    store.add_ruling(
        ExpertRuling(
            conflict_id="conflict_baseline_smiles_transform",
            scope="column:smiles",
            conflict_type="BASELINE_POLICY_CONFLICT",
            action_type="KEEP_BASELINE",
            decision_text="Keep verbatim SMILES.",
            document_id="1570180811666140725185713",
            row_id="row_1",
            field_name="smiles",
        )
    )
    store.add_ruling(
        ExpertRuling(
            conflict_id="conflict_smiles_gt_error_41",
            scope="column:smiles",
            conflict_type="SUSPECTED_GT_ERROR",
            action_type="IGNORE_CASE",
            decision_text="Ignored acs.jafc.1c02545:row_41:smiles",
            document_id="acs.jafc.1c02545",
            row_id="row_41",
            field_name="smiles",
        )
    )
    store.mark_ignored("acs.jafc.1c02545", "row_41", "smiles")

    reemitted_baseline = REConflict(
        conflict_id="smiles_baseline_transform_conflict_001",
        scope="column:smiles",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="1570180811666140725185713",
        row_id="row_10",
        explanation="Duplicate baseline conflict with new LLM ID",
        options=[],
    )
    reemitted_ignored = REConflict(
        conflict_id="smiles_gt_error_acs_041",
        scope="column:smiles",
        conflict_type="SUSPECTED_GT_ERROR",
        document_id="acs.jafc.1c02545",
        row_id="row_41",
        explanation="Duplicate ignored row conflict with new LLM ID",
        options=[],
    )

    assert store.has_matching_ruling(reemitted_baseline) is True
    assert store.has_matching_ruling(reemitted_ignored) is True

    resolver = InteractiveConflictResolver(
        gt_editor=editor,
        rulings_store=store,
        interactive=True,
        input_fn=lambda _: (_ for _ in ()).throw(AssertionError("Must not prompt for already resolved conflicts")),
    )
    batch_res = resolver.resolve_batch([reemitted_baseline, reemitted_ignored])
    assert len(batch_res.rulings) == 2
    assert batch_res.needs_rule_recompute is False
    assert batch_res.needs_doc_recompute is False


def test_rulings_store_baseline_conflict_topic_isolation_across_documents(gt_test_files, tmp_path: Path):
    """A BASELINE_POLICY_CONFLICT ruling for Kekule formatting must NOT suppress a different Markush conflict on another doc."""
    source_gt, corrected_gt = gt_test_files
    GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    store = RulingsStore(storage_path=tmp_path / "rulings_topics.json")

    store.add_ruling(
        ExpertRuling(
            conflict_id="conflict_baseline_smiles_transform",
            scope="column:smiles",
            conflict_type="BASELINE_POLICY_CONFLICT",
            action_type="KEEP_BASELINE",
            decision_text="Preserve aromatic SMILES without Kekule canonicalization conversion.",
            document_id="doc_1",
            row_id="row_1",
            field_name="smiles",
        )
    )

    same_topic_other_doc = REConflict(
        conflict_id="new_id_kekule_doc2",
        scope="column:smiles",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc_2",
        row_id="row_5",
        conflicting_value_or_rule="GT converts aromatic SMILES to Kekule canonicalization",
        explanation="Kekule conversion vs aromatic SMILES",
        options=[],
    )
    different_topic_other_doc = REConflict(
        conflict_id="new_id_markush_doc2",
        scope="column:smiles",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc_2",
        row_id="row_8",
        conflicting_value_or_rule="Markush scaffold and substituent attachment assembly required",
        explanation="Substituent R-groups must be assembled onto core scaffold",
        options=[],
    )

    assert store.has_matching_ruling(same_topic_other_doc) is True
    assert store.has_matching_ruling(different_topic_other_doc) is False



def test_patch_markdown_preserves_document_id_header(tmp_path: Path):
    ing_dir = tmp_path / "ingestion"
    doc_dir = ing_dir / "ja.2017.70"
    doc_dir.mkdir(parents=True)
    md_path = doc_dir / "ja.2017.70.md"
    md_path.write_text("<!-- document_id: ja.2017.70 -->\n\nBody text here.", encoding="utf-8")

    editor = MarkdownEditor(ingestion_dir=ing_dir)
    # Short/header-matching string '1' must NOT corrupt <!-- document_id: ja.2017.70 -->
    assert editor.patch_markdown("ja.2017.70", "1", "PATCHED_SMILES") is True
    content = md_path.read_text(encoding="utf-8")
    assert content.startswith("<!-- document_id: ja.2017.70 -->")
    assert "<!-- Ingestion Patch: PATCHED_SMILES -->" in content


def test_patch_gt_cell_rejects_prose_instruction_without_user_override(tmp_path: Path):
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("pdf,target_value\npaper_1,50\n", encoding="utf-8")
    corr_path = tmp_path / "corr.csv"
    editor = GTEditor(gt_path, corr_path)
    store = RulingsStore(tmp_path / "rulings.json")

    conflict = REConflict(
        conflict_id="c_prose",
        scope="column:target_value",
        conflict_type="SUSPECTED_GT_ERROR",
        document_id="paper_1",
        row_id="row_1",
        gt_value="50",
        explanation="GT says 50 but text says 25",
        options=[
            ResolutionOption(
                action_type="PATCH_GT_CELL",
                label="Patch GT",
                proposed_rule_or_value="Update GT value from '50' to '25' to match source text",
            )
        ],
    )
    answers = iter(["1", "25"])
    resolver = InteractiveConflictResolver(
        gt_editor=editor,
        rulings_store=store,
        interactive=True,
        input_fn=lambda _: next(answers),
    )
    ruling, _ = resolver.resolve(conflict)
    assert editor.df.at[0, "target_value"] == "25"
    assert ruling.decision_text == "Corrected GT [target_value] to '25'"


def test_is_descriptive_rule_text():
    assert _is_descriptive_rule_text("0.202") is False
    assert _is_descriptive_rule_text("42") is False
    assert _is_descriptive_rule_text("µM") is False
    assert _is_descriptive_rule_text("short") is False
    assert _is_descriptive_rule_text(None) is False
    assert _is_descriptive_rule_text("Preserve verbatim decimal precision from source table") is True


def test_keep_baseline_rejects_numeric_literal_in_decision_text(gt_test_files, tmp_path: Path):
    source_gt, corrected_gt = gt_test_files
    editor = GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    store = RulingsStore(storage_path=tmp_path / "rulings_keep_baseline.json")

    conflict = REConflict(
        conflict_id="c_keep_lit",
        scope="column:yield",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="paper_1",
        row_id="row_1",
        conflicting_value_or_rule="0.202",
        explanation="Baseline requires reporting percentage yield rather than fractional ratio",
        options=[],
    )
    resolver = InteractiveConflictResolver(
        gt_editor=editor,
        rulings_store=store,
        interactive=True,
        input_fn=lambda _: "5",  # [5] KEEP_BASELINE
    )
    ruling, _ = resolver.resolve(conflict)
    assert ruling.action_type == "KEEP_BASELINE"
    assert "0.202" not in ruling.decision_text
    assert "percentage yield" in ruling.decision_text


def test_rulings_store_extended_dedup_generalize_and_ignore_case(gt_test_files, tmp_path: Path):
    source_gt, corrected_gt = gt_test_files
    GTEditor(source_gt_path=source_gt, corrected_gt_path=corrected_gt)
    store = RulingsStore(storage_path=tmp_path / "rulings_extended_dedup.json")

    # 1. GENERALIZE ruling on BASELINE_POLICY_CONFLICT should match cross-document
    store.add_ruling(
        ExpertRuling(
            conflict_id="c_gen_1",
            scope="column:yield",
            conflict_type="BASELINE_POLICY_CONFLICT",
            action_type="GENERALIZE",
            decision_text="Normalize concentration units to micromolar across all tables",
            document_id="doc_1",
            row_id="row_1",
            field_name="yield",
        )
    )
    other_doc_baseline = REConflict(
        conflict_id="c_gen_2",
        scope="column:yield",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc_2",
        row_id="row_3",
        conflicting_value_or_rule="Normalize concentration units to micromolar",
        explanation="Concentration units differ across tables",
        options=[],
    )
    assert store.has_matching_ruling(other_doc_baseline) is True

    # 2. Same (scope, doc) with changed conflict_type and row_id (row_22 -> gap_row_22)
    store.add_ruling(
        ExpertRuling(
            conflict_id="c_omission_row22",
            scope="rows",
            conflict_type="SUSPECTED_GT_OMISSION",
            action_type="ADD_CONDITION",
            decision_text="Exclude footnote control summary rows from extraction",
            document_id="doc_3",
            row_id="row_22",
        )
    )
    recomputed_gap = REConflict(
        conflict_id="c_baseline_gap_row22",
        scope="rows",
        conflict_type="BASELINE_POLICY_CONFLICT",
        document_id="doc_3",
        row_id="gap_row_22",
        conflicting_value_or_rule="Footnote control summary rows should be excluded",
        explanation="Exclude footnote control summary rows",
        options=[],
    )
    assert store.has_matching_ruling(recomputed_gap) is True

    # 3. Intra-document IGNORE_CASE on adjacent rows (row_1 vs row_2) with overlapping topic
    store.add_ruling(
        ExpertRuling(
            conflict_id="strain_code_spacing_001",
            scope="column:formula",
            conflict_type="SUSPECTED_INGESTION_DEFECT",
            action_type="IGNORE_CASE",
            decision_text="Extra space inside strain culture code identifier",
            document_id="doc_4",
            row_id="row_1",
            field_name="formula",
        )
    )
    same_doc_row2 = REConflict(
        conflict_id="strain_spacing_norm_003",
        scope="column:formula",
        conflict_type="SUSPECTED_INGESTION_DEFECT",
        document_id="doc_4",
        row_id="row_2",
        conflicting_value_or_rule="Extra space inside strain culture code identifier",
        explanation="Spacing defect in strain culture code",
        options=[],
    )
    other_doc_row2 = REConflict(
        conflict_id="strain_spacing_norm_004",
        scope="column:formula",
        conflict_type="SUSPECTED_INGESTION_DEFECT",
        document_id="doc_5",
        row_id="row_2",
        conflicting_value_or_rule="Extra space inside strain culture code identifier",
        explanation="Spacing defect in strain culture code",
        options=[],
    )
    assert store.has_matching_ruling(same_doc_row2) is True
    assert store.has_matching_ruling(other_doc_row2) is False
