"""Unit tests for domain-independent decoding sanity checks and ExtractionGraph retry behavior."""

from typing import Literal, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from ae.extraction.graph import ExtractionGraph
from ae.extraction.sanity import (
    SanityCheckConfig,
    StructuredOutputSanityError,
    validate_structured_output_sanity,
)


class SampleRecord(BaseModel):
    item_id: str = Field(description="Identifier")
    relation: Optional[str] = Field(default=None, description="Relation")
    unit: Optional[str] = Field(default=None, description="Unit")


class SampleOutput(BaseModel):
    records: list[SampleRecord] = Field(default_factory=list)


class ConstrainedRecord(BaseModel):
    item_id: str = Field(description="Identifier")
    relation: Literal["=", ">", "<"] = Field(default="=", description="Operator")
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Score")


class ConstrainedOutput(BaseModel):
    records: list[ConstrainedRecord] = Field(default_factory=list)


@pytest.mark.unit
class TestSanityValidation:
    def test_detects_ungrounded_cjk_and_bengali_scripts(self):
        doc = "Compound 3g showed MIC 25 ug/mL against E. coli."
        prompt = "Extract compound_id, relation, and unit."

        for bad_val in ("文明", "সং="):
            out = SampleOutput(
                records=[SampleRecord(item_id="3g", relation=bad_val, unit="ug/mL")]
            )
            with pytest.raises(StructuredOutputSanityError, match="ungrounded script"):
                validate_structured_output_sanity(
                    out,
                    document_text=doc,
                    system_instruction=prompt,
                    output_model=SampleOutput,
                )

    def test_allows_multilingual_scripts_when_present_in_document_or_prompt(self):
        # Russian document -> Cyrillic allowed
        ru_doc = "Соединение 3g показало активность 25 мкг/мл."
        ru_out = SampleOutput(
            records=[SampleRecord(item_id="3g", relation="равно", unit="мкг/мл")]
        )
        assert (
            validate_structured_output_sanity(
                ru_out, document_text=ru_doc, system_instruction="", output_model=SampleOutput
            )
            is ru_out
        )

        # Chinese document -> CJK allowed
        zh_doc = "化合物 3g 的最小抑菌浓度为 25 µg/mL。"
        zh_out = SampleOutput(
            records=[SampleRecord(item_id="化合物3g", relation="=", unit="µg/mL")]
        )
        assert (
            validate_structured_output_sanity(
                zh_out, document_text=zh_doc, system_instruction="", output_model=SampleOutput
            )
            is zh_out
        )

    def test_allows_scientific_greek_and_uniform_grounded_values(self):
        doc = "Compounds 1a, 1b, 1c, 1d all had MIC = 25 µg/mL."
        prompt = 'Extract relation (e.g., "=", ">", "<").'
        out = SampleOutput(
            records=[
                SampleRecord(item_id=f"1{c}", relation="=", unit="μg/mL")
                for c in ("a", "b", "c", "d")
            ]
        )
        assert (
            validate_structured_output_sanity(
                out, document_text=doc, system_instruction=prompt, output_model=SampleOutput
            )
            is out
        )

    def test_detects_control_and_replacement_characters(self):
        out = SampleOutput(
            records=[SampleRecord(item_id="1a", relation="=\ufffd", unit="mg/L")]
        )
        with pytest.raises(StructuredOutputSanityError, match="corrupted character"):
            validate_structured_output_sanity(
                out,
                document_text="Compound 1a 10 mg/L",
                system_instruction="Extract",
                output_model=SampleOutput,
            )

    def test_detects_sticky_ungrounded_symbolic_collapse_ignoring_markup(self):
        # Document has '>' only inside XML/markdown blockquote markup, and NO '>=' anywhere
        doc = (
            '<!-- document_id: doc1 -->\n'
            '<visual id="Table_1" type="table">\n'
            '> Table 1 summary\n'
            'Compound 1a: 12.5; 1b: 25; 1c: 50; 1d: 100\n'
            '</visual>'
        )
        prompt = 'Use "=" if no operator is stated.'
        out = SampleOutput(
            records=[
                SampleRecord(item_id=f"1{c}", relation=">=", unit="mg/L")
                for c in ("a", "b", "c", "d")
            ]
        )
        with pytest.raises(StructuredOutputSanityError, match="sticky ungrounded symbolic collapse"):
            validate_structured_output_sanity(
                out,
                document_text=doc,
                system_instruction=prompt,
                output_model=SampleOutput,
                config=SanityCheckConfig(min_records_for_collapse=4),
            )

    def test_detects_sticky_gt_collapse_ignoring_reaction_arrows_and_prompt_examples(self):
        doc = (
            "Synthesis scheme: A --> B (85% yield), B <-> C, C -> D, D <- E.\n"
            "Compounds 1a: 12.5 mg/L; 1b: 25 mg/L; 1c: 50 mg/L; 1d: 100 mg/L."
        )
        prompt = 'Extract relation operator (examples: "=", ">", "<").'
        out = SampleOutput(
            records=[
                SampleRecord(item_id=f"1{c}", relation=">", unit="mg/L")
                for c in ("a", "b", "c", "d")
            ]
        )
        with pytest.raises(StructuredOutputSanityError, match="sticky ungrounded symbolic collapse"):
            validate_structured_output_sanity(
                out,
                document_text=doc,
                system_instruction=prompt,
                output_model=SampleOutput,
                config=SanityCheckConfig(min_records_for_collapse=4),
            )


@pytest.mark.unit
class TestExtractionGraphSanityIntegration:
    def test_single_mode_retries_with_feedback_on_sanity_error(self, tmp_path):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        bad_output = SampleOutput(
            records=[SampleRecord(item_id="3g", relation="সং=", unit="ug/mL")]
        )
        good_output = SampleOutput(
            records=[SampleRecord(item_id="3g", relation="=", unit="ug/mL")]
        )
        mock_structured.invoke.side_effect = [bad_output, good_output]

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=SampleOutput,
            ingestion_dir=tmp_path,
            structured_output_batch_size=0,
            max_retries=2,
            retry_delay=0.0,
        )

        res = graph.run(
            document_text="<!-- document_id: doc_sanity -->\n\nCompound 3g: 25 ug/mL.",
            doc_id="doc_sanity",
            system_instruction="Extract records.",
        )

        assert res["extracted_data"].records[0].relation == "="
        assert mock_structured.invoke.call_count == 2
        retry_human_msg = mock_structured.invoke.call_args_list[1][0][0][-1].content
        assert "CORRECTIVE INSTRUCTION" in retry_human_msg

    def test_batched_mode_rejects_poisoned_batch_before_accumulating(self, tmp_path):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        poisoned_batch = SampleOutput(
            records=[
                SampleRecord(item_id="1a", relation="文明", unit="ug/mL"),
                SampleRecord(item_id="1b", relation="文明", unit="ug/mL"),
                SampleRecord(item_id="1c", relation="文明", unit="ug/mL"),
                SampleRecord(item_id="1d", relation="文明", unit="ug/mL"),
                SampleRecord(item_id="1e", relation="文明", unit="ug/mL"),
                SampleRecord(item_id="1f", relation="文明", unit="ug/mL"),
            ]
        )
        clean_batch_1 = SampleOutput(
            records=[
                SampleRecord(item_id="1a", relation="=", unit="ug/mL"),
                SampleRecord(item_id="1b", relation="=", unit="ug/mL"),
                SampleRecord(item_id="1c", relation="=", unit="ug/mL"),
                SampleRecord(item_id="1d", relation="=", unit="ug/mL"),
                SampleRecord(item_id="1e", relation="=", unit="ug/mL"),
            ]
        )
        clean_batch_2 = SampleOutput(
            records=[SampleRecord(item_id="1f", relation="=", unit="ug/mL")]
        )
        empty_batch_3 = SampleOutput(records=[])
        mock_structured.invoke.side_effect = [
            poisoned_batch,
            clean_batch_1,
            clean_batch_2,
            empty_batch_3,
        ]

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=SampleOutput,
            ingestion_dir=tmp_path,
            structured_output_batch_size=10,
            max_structured_output_batches=4,
            max_retries=2,
            retry_delay=0.0,
        )

        res = graph.run(
            document_text="<!-- document_id: doc_batch_sanity -->\n\nCompounds 1a-1f tested (MIC = 25 ug/mL).",
            doc_id="doc_batch_sanity",
            system_instruction='Extract records. Use "=" if no operator is stated.',
        )

        records = res["extracted_data"].records
        assert len(records) == 6
        assert all(r.relation == "=" for r in records)
        # Batch size was halved (10 -> 5) on sanity retry, and '文明' never leaked into batch 2 prompt
        batch2_prompt = mock_structured.invoke.call_args_list[2][0][0][-1].content
        assert "文明" not in batch2_prompt
        assert '"item_id":"1a"' in batch2_prompt

    def test_exhausted_sanity_retries_raise_error(self, tmp_path):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        bad_output = SampleOutput(
            records=[SampleRecord(item_id="3g", relation="文明", unit="ug/mL")]
        )
        mock_structured.invoke.return_value = bad_output

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=SampleOutput,
            ingestion_dir=tmp_path,
            structured_output_batch_size=0,
            max_retries=1,
            retry_delay=0.0,
        )

        with pytest.raises(StructuredOutputSanityError):
            graph.run(
                document_text="<!-- document_id: doc_fail -->\n\nCompound 3g.",
                doc_id="doc_fail",
            )
        assert mock_structured.invoke.call_count == 2

    def test_format_schema_summary_includes_constraints(self, tmp_path):
        graph = ExtractionGraph(
            llm=MagicMock(),
            output_model=ConstrainedOutput,
            ingestion_dir=tmp_path,
        )
        summary = graph._format_schema_summary()
        assert "allowed: ['=', '>', '<']" in summary
        assert "min=0.0, max=100.0" in summary


@pytest.mark.unit
class TestDeclarativeFieldValidatorsInGraph:
    def test_graph_retries_on_smiles_placeholder_and_succeeds(self, tmp_path):
        from ae.core.schema import FieldSpec, SchemaConfig, create_all_models

        cfg = SchemaConfig(
            name="chem_test",
            fields={
                "compound_id": FieldSpec(type=str, description="Compound ID"),
                "smiles": FieldSpec(type=str, description="Molecule SMILES", required=False, format="smiles"),
            },
            compare_fields=["compound_id"],
        )
        rec_model, out_model = create_all_models(cfg)

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        bad_out = out_model(records=[rec_model(compound_id="3g", smiles="c1ccc([*:1])cc1")])
        good_out = out_model(records=[rec_model(compound_id="3g", smiles="c1ccccc1")])
        mock_structured.invoke.side_effect = [bad_out, good_out]

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=out_model,
            ingestion_dir=tmp_path,
            structured_output_batch_size=0,
            max_retries=2,
            retry_delay=0.0,
        )
        assert "validators: [smiles]" in graph._format_schema_summary()

        res = graph.run(
            document_text="<!-- document_id: doc_chem -->\n\nCompound 3g is benzene.",
            doc_id="doc_chem",
            system_instruction="Extract compounds.",
        )
        assert res["extracted_data"].records[0].smiles == "c1ccccc1"
        assert mock_structured.invoke.call_count == 2
        retry_msg = mock_structured.invoke.call_args_list[1][0][0][-1].content
        assert "CORRECTIVE INSTRUCTION" in retry_msg
        assert "[*:1]" in retry_msg

    def test_graph_runs_field_validators_even_when_heuristic_sanity_disabled(self, tmp_path):
        from ae.core.schema import FieldSpec, SchemaConfig, create_all_models

        cfg = SchemaConfig(
            name="materials_test",
            fields={
                "material_id": FieldSpec(
                    type=str,
                    description="Material ID",
                    validators=[{"name": "not_regex", "patterns": [r"\bTBD\b"]}],
                ),
            },
            compare_fields=["material_id"],
        )
        rec_model, out_model = create_all_models(cfg)

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = out_model(records=[rec_model(material_id="TBD")])

        graph_enabled = ExtractionGraph(
            llm=mock_llm,
            output_model=out_model,
            ingestion_dir=tmp_path,
            structured_output_batch_size=0,
            max_retries=0,
            retry_delay=0.0,
            enable_sanity_checks=False,
            enable_field_validators=True,
        )
        with pytest.raises(StructuredOutputSanityError, match="not_regex"):
            graph_enabled.run(document_text="<!-- document_id: m1 -->\n\nMaterial A.", doc_id="m1")

        graph_disabled = ExtractionGraph(
            llm=mock_llm,
            output_model=out_model,
            ingestion_dir=tmp_path,
            structured_output_batch_size=0,
            max_retries=0,
            retry_delay=0.0,
            enable_sanity_checks=False,
            enable_field_validators=False,
        )
        res = graph_disabled.run(document_text="<!-- document_id: m1 -->\n\nMaterial A.", doc_id="m1")
        assert res["extracted_data"].records[0].material_id == "TBD"

    def test_nullify_and_warn_error_policies(self, tmp_path):
        from ae.core.schema import FieldSpec, SchemaConfig, create_all_models

        cfg = SchemaConfig(
            name="policies_test",
            fields={
                "code": FieldSpec(
                    type=str,
                    description="Code",
                    required=False,
                    default=None,
                    validators=[{"name": "not_regex", "patterns": [r"BAD"], "on_error": "nullify"}],
                ),
                "note": FieldSpec(
                    type=str,
                    description="Note",
                    required=False,
                    validators=[{"name": "not_regex", "patterns": [r"WARN_ME"], "on_error": "warn"}],
                ),
            },
            compare_fields=["code"],
        )
        rec_model, out_model = create_all_models(cfg)

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = out_model(
            records=[rec_model(code="BAD_CODE", note="WARN_ME_TEXT")]
        )

        graph = ExtractionGraph(
            llm=mock_llm,
            output_model=out_model,
            ingestion_dir=tmp_path,
            structured_output_batch_size=0,
            max_retries=0,
            retry_delay=0.0,
        )
        res = graph.run(document_text="<!-- document_id: p1 -->\n\nText.", doc_id="p1")
        rec = res["extracted_data"].records[0]
        assert rec.code is None
        assert rec.note == "WARN_ME_TEXT"


    def test_sanity_error_populates_failed_record_indices_and_partial_output(self):
        from ae.core.schema import FieldSpec, SchemaConfig, create_all_models
        from ae.extraction.sanity import (
            StructuredOutputSanityError,
            validate_structured_output_sanity,
        )

        cfg = SchemaConfig(
            name="chem_idx_test",
            fields={
                "compound_id": FieldSpec(type=str, description="Compound ID"),
                "smiles": FieldSpec(type=str, description="SMILES", required=False, format="smiles"),
            },
            compare_fields=["compound_id"],
        )
        rec_model, out_model = create_all_models(cfg)

        out_validator_fail = out_model(
            records=[
                rec_model(compound_id="1a", smiles="CCO"),
                rec_model(compound_id="1b", smiles="c1ccc([*:1])cc1"),
                rec_model(compound_id="1c", smiles="CCC"),
            ]
        )
        with pytest.raises(StructuredOutputSanityError) as exc_info:
            validate_structured_output_sanity(
                out_validator_fail,
                document_text="Compounds 1a, 1b, 1c.",
                output_model=out_model,
            )
        assert exc_info.value.failed_record_indices == [1]
        assert exc_info.value.partial_output is out_validator_fail

        out_ctrl_and_script = out_model(
            records=[
                rec_model(compound_id="1a", smiles="CCO"),
                rec_model(compound_id="1\x00b", smiles="CCO"),
                rec_model(compound_id="1c", smiles="CCO"),
                rec_model(compound_id="你好", smiles="CCO"),
            ]
        )
        with pytest.raises(StructuredOutputSanityError) as exc_info2:
            validate_structured_output_sanity(
                out_ctrl_and_script,
                document_text="Compounds 1a, 1b, 1c.",
                output_model=out_model,
            )
        assert exc_info2.value.failed_record_indices == [1, 3]
        assert exc_info2.value.partial_output is out_ctrl_and_script
