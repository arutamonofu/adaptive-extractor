"""Unit tests for declarative ExtractionToolRegistry, schema tools YAML wiring, and lazy RDKit Markush tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ae.core.schema import (
    FieldSpec,
    RegisteredToolProvider,
    SchemaConfig,
    extract_schema_tools,
    has_extraction_tool,
    load_schema_complete,
    load_schema_from_yaml,
    load_schema_with_models,
    save_schema_to_yaml,
)
from ae.core.schema.domain_tools.chemistry import (
    assemble_markush_batch,
    assemble_markush_single,
    build_markush_assembly_tool,
    is_markush_applicable,
)
from ae.core.schema.tools import (
    ExtractionToolRegistry,
    ExtractionToolSpec,
    normalize_tool_spec,
    serialize_tool_spec,
)


def custom_unit_tool_factory() -> RegisteredToolProvider:
    from langchain_core.tools import StructuredTool

    def _convert(value: float) -> str:
        return str(value * 1000.0)

    return RegisteredToolProvider(
        name="custom_unit_converter",
        build_tool=lambda **_: StructuredTool.from_function(
            func=_convert,
            name="custom_unit_converter",
            description="Convert mM to uM",
        ),
        is_applicable=lambda doc_text: "mM" in doc_text,
        policy_hint="Use custom_unit_converter for mM values.",
    )


class TestExtractionToolRegistryAndSchema:
    def test_tool_spec_normalization_and_serialization(self):
        s1 = normalize_tool_spec("assemble_markush_smiles")
        assert s1 == ExtractionToolSpec(name="assemble_markush_smiles")
        assert serialize_tool_spec(s1) == "assemble_markush_smiles"

        s2 = normalize_tool_spec({"name": "custom_tool", "params": {"precision": 2}, "enabled": False})
        assert s2.name == "custom_tool"
        assert s2.params == {"precision": 2}
        assert s2.enabled is False
        d2 = serialize_tool_spec(s2)
        assert d2 == {"name": "custom_tool", "enabled": False, "params": {"precision": 2}}

    def test_registry_supports_custom_dotted_callable_and_lazy_builtin(self):
        reg = ExtractionToolRegistry()
        assert reg.has("assemble_markush_smiles") is True
        assert has_extraction_tool("assemble_markush_smiles") is True

        spec = ExtractionToolSpec(
            name="custom_unit_converter",
            callable_path=f"{__name__}:custom_unit_tool_factory",
        )
        assert reg.validate_spec(spec) == []
        provider = reg.resolve(spec)
        assert provider.name == "custom_unit_converter"
        assert provider.is_applicable("IC50 = 2 mM") is True
        assert provider.is_applicable("IC50 = 2 ug/mL") is False
        tool = provider.build_tool()
        assert tool.invoke({"value": 1.5}) == "1500.0"

        bad_spec = ExtractionToolSpec(name="nonexistent_domain_tool")
        errs = reg.validate_spec(bad_spec)
        assert len(errs) == 1
        assert "nonexistent_domain_tool" in errs[0]

    def test_schema_yaml_round_trip_and_output_model_attachment(self, tmp_path: Path):
        yaml_path = tmp_path / "schema.yaml"
        yaml_path.write_text(
            "name: demo_task\n"
            "fields:\n"
            "  compound_id:\n"
            "    type: str\n"
            "    description: Identifier\n"
            "    required: false\n"
            "compare_fields:\n"
            "  - compound_id\n"
            "float_tolerance: 0.05\n"
            "tools:\n"
            "  - name: assemble_markush_smiles\n",
            encoding="utf-8",
        )

        cfg = load_schema_from_yaml(yaml_path)
        assert [t.name for t in cfg.tools] == ["assemble_markush_smiles"]
        assert cfg.validate() == []

        out_yaml = tmp_path / "saved_schema.yaml"
        save_schema_to_yaml(cfg, out_yaml)
        reloaded = load_schema_from_yaml(out_yaml)
        assert [t.name for t in reloaded.tools] == ["assemble_markush_smiles"]

        _, _, output_model = load_schema_with_models(yaml_path)
        extracted_tools = extract_schema_tools(output_model)
        assert [t.name for t in extracted_tools] == ["assemble_markush_smiles"]

        # Schema hash remains unaffected by tools metadata so existing agents stay compatible
        base_cfg = SchemaConfig(
            name="demo_task",
            fields={"compound_id": FieldSpec(type=str, description="Identifier", required=False)},
            compare_fields=["compound_id"],
            float_tolerance=0.05,
        )
        assert cfg.get_schema_hash() == base_cfg.get_schema_hash()
        assert cfg.get_validation_hash() != base_cfg.get_validation_hash()

    def test_load_real_project_schema_yaml_has_markush_tool(self):
        project_schema = Path("config/schema.yaml")
        if not project_schema.exists():
            pytest.skip("config/schema.yaml not present")
        bundle = load_schema_complete(project_schema, instruction_path=Path("config/baseline_instruction.txt"))
        tool_names = [t.name for t in extract_schema_tools(bundle.output_model)]
        assert "assemble_markush_smiles" in tool_names


class TestMarkushChemistryTool:
    def test_markush_applicability_predicate(self):
        assert is_markush_applicable("Plain text without attachment points") is False
        assert is_markush_applicable("Scaffold `Cc1nc(N[*:1])s1` with 5a `C[*:1]`") is True
        assert is_markush_applicable("Isotope dummy [2*]Cl") is True

    def test_assemble_real_acsomega_examples_and_ring_collisions(self):
        Chem = pytest.importorskip("rdkit.Chem")
        scaffold = "Cc1cc2nc(-c3ccc(-c4nnc(N[*:1])s4)cc3)[nH]c2cc1C"

        # Verify against real Kekule GT SMILES from acsomega.2c06142 (including ring numbering collisions 5d, 5e)
        cases = [
            ("5a", {"1": "C[*:1]"}, "N=1N=C(SC1NC)C=2C=CC(=CC2)C3=NC=4C=C(C(=CC4N3)C)C"),
            ("5d", {"1": "Clc1ccccc1[*:1]"}, "CC1=CC2=C(C=C1C)N=C(N2)C1=CC=C(C=C1)C1=NN=C(NC2=CC=CC=C2Cl)S1"),
            ("5e", {"1": "C1CCCCC1[*:1]"}, "CC1=CC2=C(C=C1C)N=C(N2)C1=CC=C(C=C1)C1=NN=C(NC2CCCCC2)S1"),
        ]
        for cid, subs, gt_smiles in cases:
            assembled, err = assemble_markush_single(scaffold, subs)
            assert err is None, f"Failed for {cid}: {err}"
            assert assembled is not None
            assert "[*" not in assembled
            assert Chem.MolToSmiles(Chem.MolFromSmiles(assembled), isomericSmiles=True) == Chem.MolToSmiles(
                Chem.MolFromSmiles(gt_smiles), isomericSmiles=True
            )

    def test_assemble_multiple_sites_symmetric_sites_and_hydrogen_capping(self):
        Chem = pytest.importorskip("rdkit.Chem")
        # Two distinct sites + H capping
        smi, err = assemble_markush_single("c1([*:1])ccc([*:2])cc1", {"1": "Cl[*:1]", "2": "H"})
        assert err is None
        assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == Chem.MolToSmiles(Chem.MolFromSmiles("Clc1ccccc1"))

        # Symmetric two occurrences of [*:1]
        sym_smi, sym_err = assemble_markush_single("c1([*:1])ccc([*:1])cc1", {"[*:1]": "C[*:1]"})
        assert sym_err is None
        assert Chem.MolToSmiles(Chem.MolFromSmiles(sym_smi)) == Chem.MolToSmiles(Chem.MolFromSmiles("Cc1ccc(C)cc1"))

    def test_batch_tool_isolates_errors_per_compound(self):
        pytest.importorskip("rdkit")
        tool = build_markush_assembly_tool()
        raw = tool.invoke(
            {
                "scaffold_smiles": "c1ccccc1[*:1]",
                "items": [
                    {"compound_id": "ok_1", "substituents": {"1": "C[*:1]"}},
                    {"compound_id": "bad_1", "substituents": {"1": "not_valid_smiles[[["}},
                ],
            }
        )
        payload = json.loads(raw)
        assert len(payload["assembled"]) == 1
        assert payload["assembled"][0]["compound_id"] == "ok_1"
        assert len(payload["errors"]) == 1
        assert payload["errors"][0]["compound_id"] == "bad_1"

        # Also accept JSON-stringified substituents/items emitted by Gemini tool calling
        raw_str_args = tool.invoke(
            {
                "scaffold_smiles": "c1ccccc1[*:1]",
                "items": json.dumps([{"compound_id": "5a", "substituents": '{"1": "C[*:1]"}'}]),
            }
        )
        payload_str = json.loads(raw_str_args)
        assert len(payload_str["assembled"]) == 1
        assert payload_str["assembled"][0]["compound_id"] == "5a"
        assert payload_str["errors"] == []


    def test_assemble_markush_preserves_tetrahedral_chirality(self):
        from rdkit import Chem
        from ae.core.schema.domain_tools.chemistry import assemble_markush_single

        scaf = "c1ccc(Cn2c(C3=NN(c4ccccc4)[C@H]([*:1])C3)nc3ccccc32)cc1"
        expected_24 = Chem.MolToSmiles(
            Chem.MolFromSmiles("c1ccc(Cn2c(C3=NN(c4ccccc4)[C@H](c4ccccc4)C3)nc3ccccc32)cc1"),
            isomericSmiles=True,
        )
        expected_28 = Chem.MolToSmiles(
            Chem.MolFromSmiles("c1ccc(Cn2c(C3=NN(c4ccccc4)[C@H](c4ccc(F)cc4)C3)nc3ccccc32)cc1"),
            isomericSmiles=True,
        )

        smi_24, err_24 = assemble_markush_single(scaf, {"1": "c1ccccc1[*:1]"})
        smi_28, err_28 = assemble_markush_single(scaf, {"1": "Fc1ccc([*:1])cc1"})
        assert err_24 is None and smi_24 == expected_24
        assert err_28 is None and smi_28 == expected_28

    def test_markush_policy_hint_enforces_iupac_crosscheck_and_no_dummy_calls(self):
        from ae.core.schema.domain_tools.chemistry import _MARKUSH_POLICY_HINT

        hint_lower = _MARKUSH_POLICY_HINT.lower()
        assert "iupac" in hint_lower
        assert "takes precedence" in hint_lower
        assert "compound_id" in hint_lower and '"test"' in hint_lower
