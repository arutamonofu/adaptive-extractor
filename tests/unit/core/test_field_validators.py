"""Unit tests for declarative field validators, registry, and schema integration."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from ae.core.schema import (
    FieldSpec,
    FieldValidationResult,
    FieldValidatorSpec,
    SchemaConfig,
    create_all_models,
    create_row_converter,
    extract_field_validators,
    load_schema_from_yaml,
    register_field_validator,
    save_schema_to_yaml,
    unregister_field_validator,
)
from ae.core.schema.dynamic_models import extract_schema_field_constraints
from ae.core.schema.validators import validate_field_value


def custom_even_length_validator(value: str) -> tuple[bool, str | None]:
    """Sample external validator callable for dotted-path testing."""
    if len(value) % 2 != 0:
        return False, "length must be even"
    return True, None


@pytest.mark.unit
class TestCoreDeclarativeValidators:
    def test_not_regex_and_forbidden_substrings(self):
        not_regex_spec = FieldValidatorSpec(
            name="not_regex",
            params={"patterns": [r"\[\*:\d+\]", r"\bTBD\b"]},
        )
        assert validate_field_value(not_regex_spec, "Fe3O4", field_name="formula").valid is True
        res_bad = validate_field_value(not_regex_spec, "c1ccc([*:1])cc1", field_name="formula")
        assert res_bad.valid is False
        assert "[*:1]" in (res_bad.error or "")

        res_tbd = validate_field_value(not_regex_spec, "Value is TBD", field_name="formula")
        assert res_tbd.valid is False

        sub_spec = FieldValidatorSpec(
            name="forbidden_substrings",
            params={"substrings": ["[Unparsed", "TODO"], "case_sensitive": False},
        )
        assert validate_field_value(sub_spec, "valid text", field_name="note").valid is True
        assert validate_field_value(sub_spec, "contains todo item", field_name="note").valid is False

    def test_regex_non_empty_and_length(self):
        regex_spec = FieldValidatorSpec(
            name="regex",
            params={"pattern": r"^[A-Z]{2}-\d+$", "fullmatch": True},
        )
        assert validate_field_value(regex_spec, "AB-123", field_name="code").valid is True
        assert validate_field_value(regex_spec, "ab-123", field_name="code").valid is False

        non_empty_spec = FieldValidatorSpec(name="non_empty")
        assert validate_field_value(non_empty_spec, "   ", field_name="code", required=False).valid is False
        assert validate_field_value(non_empty_spec, None, field_name="code", required=False).valid is True

        len_spec = FieldValidatorSpec(name="length", params={"min_length": 2, "max_length": 5})
        assert validate_field_value(len_spec, "abc", field_name="code").valid is True
        assert validate_field_value(len_spec, "a", field_name="code").valid is False
        assert validate_field_value(len_spec, "abcdef", field_name="code").valid is False

    def test_custom_registration_and_dotted_callable(self):
        register_field_validator(
            "starts_with_cif",
            lambda v: FieldValidationResult(valid=str(v).startswith("data_"), value=v, error="must start with data_"),
        )
        try:
            spec = FieldValidatorSpec(name="starts_with_cif")
            assert validate_field_value(spec, "data_123", field_name="cif").valid is True
            res = validate_field_value(spec, "bad_123", field_name="cif")
            assert res.valid is False
            assert "data_" in (res.error or "")
        finally:
            unregister_field_validator("starts_with_cif")

        dotted_spec = FieldValidatorSpec(
            name=f"{__name__}:custom_even_length_validator",
            callable_path=f"{__name__}:custom_even_length_validator",
        )
        assert validate_field_value(dotted_spec, "abcd", field_name="tag").valid is True
        res_odd = validate_field_value(dotted_spec, "abc", field_name="tag")
        assert res_odd.valid is False
        assert "even" in (res_odd.error or "")


@pytest.mark.unit
class TestSmilesDomainValidator:
    def test_smiles_validation_catches_invalid_and_dummy_atoms(self):
        spec = FieldValidatorSpec(name="smiles")

        assert validate_field_value(spec, "CCO", field_name="smiles", required=False).valid is True
        assert validate_field_value(spec, "c1ccccc1", field_name="smiles", required=False).valid is True
        assert validate_field_value(spec, None, field_name="smiles", required=False).valid is True
        assert validate_field_value(spec, "", field_name="smiles", required=False).valid is True
        # Missing-value sentinel allowed for optional fields by default
        assert validate_field_value(spec, "NaN", field_name="smiles", required=False).valid is True
        # Required field must reject "NaN" and empty strings
        assert validate_field_value(spec, "NaN", field_name="smiles", required=True).valid is False
        assert validate_field_value(spec, "", field_name="smiles", required=True).valid is False

        # Invalid syntax / valence
        res_bad = validate_field_value(spec, "C(C)(C)(C)(C)C", field_name="smiles", required=False)
        assert res_bad.valid is False
        assert "invalid SMILES" in (res_bad.error or "")

        # Unresolved Markush dummy atoms
        for placeholder in ("c1ccc([*:1])cc1", "c1ccc(*)cc1", "c1ccc([1*])cc1"):
            res_ph = validate_field_value(spec, placeholder, field_name="smiles", required=False)
            assert res_ph.valid is False
            assert "placeholder" in (res_ph.error or "").lower()

    def test_smiles_validator_options_dummy_fragments_canonicalize(self):
        allow_dummy = FieldValidatorSpec(name="smiles", params={"allow_dummy_atoms": True})
        assert validate_field_value(allow_dummy, "c1ccc([*:1])cc1", field_name="smiles").valid is True

        no_frags = FieldValidatorSpec(name="smiles", params={"allow_fragments": False})
        assert validate_field_value(no_frags, "CC(=O)[O-].[Na+]", field_name="smiles").valid is False

        canon_spec = FieldValidatorSpec(name="smiles", params={"canonicalize": True})
        res = validate_field_value(canon_spec, "OCC", field_name="smiles")
        assert res.valid is True
        assert res.value == "CCO"

    def test_chemistry_module_is_lazy_loaded(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ae.core.schema.domain_validators.chemistry", raising=False)
        cfg = SchemaConfig(
            name="materials_task",
            fields={
                "composition": FieldSpec(
                    type=str,
                    description="Composition",
                    validators=[{"name": "not_regex", "patterns": [r"\bTBD\b"]}],
                )
            },
            compare_fields=["composition"],
        )
        assert cfg.validate() == []
        _, out_model = create_all_models(cfg)
        assert "ae.core.schema.domain_validators.chemistry" not in sys.modules
        assert extract_field_validators(out_model)["composition"][0].name == "not_regex"


@pytest.mark.unit
class TestSchemaConfigAndModelsIntegration:
    def test_schema_hash_backward_compatible_and_sensitive_to_validators(self):
        base = SchemaConfig(
            name="chem",
            fields={"smiles": FieldSpec(type=str, description="SMILES", required=False)},
            compare_fields=["smiles"],
        )
        with_format = SchemaConfig(
            name="chem",
            fields={"smiles": FieldSpec(type=str, description="SMILES", required=False, format="smiles")},
            compare_fields=["smiles"],
        )
        with_refined = SchemaConfig(
            name="chem",
            fields={
                "smiles": FieldSpec(
                    type=str,
                    description="SMILES",
                    required=False,
                    choices=["CCO", "CCN"],
                    default="CCO",
                    pattern=r"^[A-Za-z0-9]+$",
                )
            },
            compare_fields=["smiles"],
        )
        # Runtime validators and refined constraints do not break agent compatibility hash
        assert base.get_schema_hash() == with_format.get_schema_hash()
        assert base.get_schema_hash() == with_refined.get_schema_hash()

        # Validation hash tracks runtime validators and constraints separately
        assert base.get_validation_hash() != with_format.get_validation_hash()
        assert base.get_validation_hash() != with_refined.get_validation_hash()

        # Schema without format/validators keeps identical canonical hash
        rebuilt = SchemaConfig(
            name="chem",
            fields={"smiles": FieldSpec(type=str, description="SMILES", required=False, validators=[])},
            compare_fields=["smiles"],
        )
        assert base.get_schema_hash() == rebuilt.get_schema_hash()
        assert base.get_validation_hash() == rebuilt.get_validation_hash()

    def test_yaml_roundtrip_and_validation_checks(self, tmp_path):
        yaml_content = """
name: chem_task
compare_fields:
  - compound_id
float_tolerance: 0.05
fields:
  compound_id:
    type: str
    description: Compound ID
    validators:
      - non_empty
  smiles:
    type: str
    description: Molecule SMILES
    required: false
    format: smiles
    validators:
      - name: not_regex
        patterns:
          - "\\\\[\\\\*:\\\\d+\\\\]"
        on_error: retry
        message: "Unresolved R-group placeholder"
"""
        yaml_path = tmp_path / "schema.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        loaded = load_schema_from_yaml(yaml_path)
        assert loaded.validate() == []
        assert loaded.fields["smiles"].format == "smiles"
        eff = loaded.fields["smiles"].get_effective_validators()
        assert [v.name for v in eff] == ["smiles", "not_regex"]

        saved_path = tmp_path / "saved.yaml"
        save_schema_to_yaml(loaded, saved_path)
        reloaded = load_schema_from_yaml(saved_path)
        assert reloaded.fields["smiles"].format == "smiles"
        assert [v.name for v in reloaded.fields["smiles"].get_effective_validators()] == ["smiles", "not_regex"]

    def test_schema_validate_catches_bad_validator_configs(self):
        bad_fmt = SchemaConfig(
            name="bad_fmt",
            fields={"x": FieldSpec(type=str, description="X", format="nonexistent_format")},
            compare_fields=["x"],
        )
        assert any("unknown format validator" in e for e in bad_fmt.validate())

        bad_nullify = SchemaConfig(
            name="bad_nullify",
            fields={
                "x": FieldSpec(
                    type=str,
                    description="X",
                    required=True,
                    validators=[{"name": "non_empty", "on_error": "nullify"}],
                )
            },
            compare_fields=["x"],
        )
        assert any("on_error='nullify'" in e for e in bad_nullify.validate())

    def test_json_schema_clean_for_gemini_and_row_converter_unblocked(self):
        cfg = SchemaConfig(
            name="chem_gemini",
            fields={
                "compound_id": FieldSpec(type=str, description="ID"),
                "smiles": FieldSpec(type=str, description="SMILES", required=False, format="smiles"),
            },
            compare_fields=["compound_id"],
        )
        rec_model, out_model = create_all_models(cfg)

        # JSON schema must remain clean for Gemini types.Schema
        schema_json = out_model.model_json_schema()
        from google.genai import _transformers, types

        processed = dict(schema_json)
        _transformers.process_schema(processed, None)
        assert types.Schema.model_validate(processed) is not None

        constraints = extract_schema_field_constraints(out_model)
        assert constraints["smiles"]["validators"] == ["smiles"]

        # Ground Truth row_converter must not drop rows due to extraction-time validators
        converter = create_row_converter(cfg, rec_model)
        row = pd.Series({"compound_id": "1a", "smiles": "LEGACY_OR_DIRTY_SMILES"})
        rec = converter(row)
        assert rec is not None
        assert rec.smiles == "LEGACY_OR_DIRTY_SMILES"


    def test_validate_smiles_canonicalize_matches_benzimidazole_tautomers_and_preserves_chirality(self):
        from ae.core.schema.domain_validators.chemistry import validate_smiles

        pred_8a = "O=[N+]([O-])c1ccc2[nH]c(-c3ccc(Nc4nc(-c5ccccc5)nc5ccccc45)cc3)nc2c1"
        gt_8a = "O=N(=O)C1=CC=C2N=C(NC2=C1)C=3C=CC(=CC3)NC=4N=C(N=C5C=CC=CC54)C=6C=CC=CC6"
        res_p = validate_smiles(pred_8a, canonicalize=True)
        res_g = validate_smiles(gt_8a, canonicalize=True)
        assert res_p.valid and res_g.valid
        assert res_p.value == res_g.value

        chiral_a = "c1ccc(Cn2c(C3=NN(c4ccccc4)[C@H](c4ccccc4)C3)nc3ccccc32)cc1"
        chiral_b = "c1ccc(Cn2c(C3=NN(c4ccccc4)[C@@H](c4ccccc4)C3)nc3ccccc32)cc1"
        assert validate_smiles(chiral_a, canonicalize=True).value != validate_smiles(chiral_b, canonicalize=True).value

    def test_smiles_isomeric_flag_and_compare_contract(self):
        from ae.core.schema.domain_validators.chemistry import validate_smiles
        from ae.core.schema.validators import (
            compare_field_values_via_validator,
            supports_value_comparison,
        )

        # isomeric=False strips E/Z and R/S during canonicalization
        ez_e = "C/C=C/C"
        ez_z = "C/C=C\\C"
        ez_flat = "CC=CC"
        assert validate_smiles(ez_e, canonicalize=True, isomeric=False).value == ez_flat
        assert validate_smiles(ez_z, canonicalize=True, isomeric=False).value == ez_flat

        spec_default = FieldValidatorSpec(name="smiles")
        assert supports_value_comparison(spec_default) is True

        # 2D match when one side has unassigned E/Z or R/S -> (True, None)
        assert compare_field_values_via_validator(spec_default, ez_flat, ez_e) == (True, None)
        assert compare_field_values_via_validator(spec_default, "CC(O)CC", "C[C@H](O)CC") == (True, None)

        # Explicit stereo conflict -> (False, "stereo_diff")
        assert compare_field_values_via_validator(spec_default, ez_e, ez_z) == (False, "stereo_diff")
        assert compare_field_values_via_validator(spec_default, "C[C@H](O)CC", "C[C@@H](O)CC") == (
            False,
            "stereo_diff",
        )

        # ignore_unassigned_stereo=False rejects missing stereo on one side
        spec_strict = FieldValidatorSpec(name="smiles", params={"ignore_unassigned_stereo": False})
        assert compare_field_values_via_validator(spec_strict, ez_flat, ez_e) == (False, "stereo_diff")

        # isomeric=False treats (E) vs (Z) as equivalent
        spec_flat = FieldValidatorSpec(name="smiles", params={"isomeric": False})
        assert compare_field_values_via_validator(spec_flat, ez_e, ez_z) == (True, None)

        # Invalid SMILES -> (False, "invalid_smiles")
        assert compare_field_values_via_validator(spec_default, "NOT_A_SMILES", ez_e) == (
            False,
            "invalid_smiles",
        )

    def test_compare_field_values_canonicalize_fallback(self):
        from ae.core.schema.validators import (
            compare_field_values_via_validator,
            supports_value_comparison,
        )

        def upper_canon(val: str, canonicalize: bool = False, **_: object) -> FieldValidationResult:
            if not isinstance(val, str) or not val.strip():
                return FieldValidationResult(valid=False, value=val, error="empty")
            return FieldValidationResult(valid=True, value=val.strip().upper() if canonicalize else val)

        register_field_validator("upper_canon_test", upper_canon)
        try:
            spec = FieldValidatorSpec(name="upper_canon_test")
            assert supports_value_comparison(spec) is True
            assert compare_field_values_via_validator(spec, "abc", " ABC ") == (True, None)
            assert compare_field_values_via_validator(spec, "abc", "xyz") == (False, "canonical_mismatch")
            assert compare_field_values_via_validator(spec, 123, "abc") == (False, "invalid_value")
        finally:
            unregister_field_validator("upper_canon_test")


@pytest.mark.unit
class TestUnitNumericQuantityAndComputeValidators:
    def test_unit_validator_pint_equivalence_and_rejection(self):
        pytest.importorskip("pint")
        from ae.core.schema.validators import (
            compare_field_values_via_validator,
            has_field_validator,
            supports_value_comparison,
        )

        assert has_field_validator("unit") is True
        spec = FieldValidatorSpec(name="unit")
        assert supports_value_comparison(spec) is True

        for u in ("µg/mL", "µg mL-1", "ug mL^-1", "µg·mL⁻¹", "µM", "uM", "mg/L"):
            assert validate_field_value(spec, u, field_name="target_units").valid is True

        assert validate_field_value(spec, "not_a_unit_xyz", field_name="target_units").valid is False
        assert validate_field_value(spec, "123", field_name="target_units").valid is False

        assert compare_field_values_via_validator(spec, "µg mL-1", "µg/mL") == (True, None)
        assert compare_field_values_via_validator(spec, "ug/mL", "µg/mL") == (True, None)
        assert compare_field_values_via_validator(spec, "uM", "µM") == (True, None)
        assert compare_field_values_via_validator(spec, "mg/L", "µg/mL") == (True, None)
        # Incompatible dimensionality -> deterministic False
        assert compare_field_values_via_validator(spec, "µM", "µg/mL") == (
            False,
            "incompatible_units",
        )
        # Compatible dimensionality with scale diff -> None (defer to coupled judge)
        assert compare_field_values_via_validator(spec, "mM", "µM") == (None, None)

        dim_spec = FieldValidatorSpec(name="unit", params={"expected_unit": "µg/mL"})
        assert validate_field_value(dim_spec, "mg/L", field_name="target_units").valid is True
        assert validate_field_value(dim_spec, "µM", field_name="target_units").valid is False

    def test_numeric_quantity_decimal_comma_std_interval_and_inequalities(self):
        from ae.core.schema.validators import (
            compare_field_values_via_validator,
            has_field_validator,
            supports_value_comparison,
        )

        assert has_field_validator("numeric_quantity") is True
        spec = FieldValidatorSpec(name="numeric_quantity")
        assert supports_value_comparison(spec) is True

        for val in ("19,1", "12.5 ± 0.4", "10–25", "> 128", "< 0.5", "≥ 64"):
            assert validate_field_value(spec, val, field_name="target_value").valid is True

        assert validate_field_value(spec, "10–5", field_name="target_value").valid is False
        assert validate_field_value(spec, "N/A", field_name="target_value", required=True).valid is False

        assert compare_field_values_via_validator(spec, "19,1", "19.1") == (True, None)
        assert compare_field_values_via_validator(spec, "12,50 ± 0,40", "12.5 ± 0.4") == (True, None)
        assert compare_field_values_via_validator(spec, "10–25", "10-25") == (True, None)
        # Inequality must NOT equal bare number
        assert compare_field_values_via_validator(spec, "> 128", "128") == (
            False,
            "relation_mismatch",
        )
        assert compare_field_values_via_validator(spec, "> 128", ">128") == (True, None)

    def test_compute_validators_simpleeval_invariants_and_security(self, tmp_path):
        pytest.importorskip("simpleeval")
        from ae.core.schema.quantity_validators import validate_record_compute_validators
        from ae.core.schema.validators import has_field_validator

        assert has_field_validator("compute_validators") is True

        # Cross-column invariant: pMIC = -log10(MIC_M)
        specs = [
            {
                "name": "compute_validators",
                "expression": "abs(pMIC - (-log10(MIC * 1e-6))) < 0.05",
                "message": "pMIC inconsistent with MIC (µM)",
            },
            {
                "name": "compute_validators",
                "expression": "MIC > 0 and MIC <= 10000",
            },
        ]
        good_rec = {"MIC": "10", "pMIC": "5.0"}
        bad_rec = {"MIC": "10", "pMIC": "3.0"}
        assert all(r.valid for r in validate_record_compute_validators(good_rec, specs))
        bad_res = validate_record_compute_validators(bad_rec, specs)
        assert bad_res[0].valid is False
        assert "pMIC inconsistent" in (bad_res[0].error or "")
        assert bad_res[1].valid is True

        # Security: unsafe expressions must fail validation and never execute
        for payload in (
            "__import__('os').system('echo pwned')",
            "().__class__.__bases__[0].__subclasses__()",
            "open('/etc/passwd').read()",
            "10 ** 100000",
        ):
            bad_spec = FieldValidatorSpec(
                name="compute_validators",
                params={"expression": payload},
            )
            res = validate_field_value(bad_spec, "10", field_name="target_value")
            assert res.valid is False

        # YAML roundtrip of compute_validators on SchemaConfig
        yaml_content = """
name: chem_compute
compare_fields:
  - target_value
float_tolerance: 0.05
fields:
  target_value:
    type: str
    description: MIC value
    validators:
      - numeric_quantity
  pMIC:
    type: str
    description: pMIC value
    required: false
compute_validators:
  - name: compute_validators
    expression: "abs(pMIC - (-log10(target_value * 1e-6))) < 0.05"
"""
        ypath = tmp_path / "compute_schema.yaml"
        ypath.write_text(yaml_content, encoding="utf-8")
        cfg = load_schema_from_yaml(ypath)
        assert cfg.validate() == []
        assert len(cfg.compute_validators) == 1

    def test_tautomer_enumerator_toggle(self):
        pytest.importorskip("rdkit")
        from ae.core.schema.validators import compare_field_values_via_validator

        # 2-hydroxypyridine vs 2-pyridone tautomers
        enol = "Oc1ccccn1"
        keto = "O=c1cccc[nH]1"
        spec_taut_on = FieldValidatorSpec(
            name="smiles", params={"canonicalize_tautomers": True}
        )
        spec_taut_off = FieldValidatorSpec(
            name="smiles", params={"canonicalize_tautomers": False}
        )
        assert compare_field_values_via_validator(spec_taut_on, enol, keto) == (True, None)
        assert compare_field_values_via_validator(spec_taut_off, enol, keto) == (
            False,
            "topology_mismatch",
        )

    def test_thousands_separator_european_decimal_and_escalation(self):
        from ae.core.schema.quantity_validators import (
            compare_numeric_quantity_values,
            compare_unit_values,
            parse_numeric_quantity,
            validate_record_compute_validators,
        )

        # Thousands separator vs decimal comma vs mixed European/US
        # Single comma group like "1,000" or "1,250" defaults to European decimal unless thousands_sep=True
        assert parse_numeric_quantity("1,250")[0].value == 1.25
        assert parse_numeric_quantity("1,000")[0].value == 1.0
        assert parse_numeric_quantity("1,250", thousands_sep=True)[0].value == 1250.0
        assert parse_numeric_quantity("1,000", thousands_sep=True)[0].value == 1000.0
        assert parse_numeric_quantity("1,000,000")[0].value == 1000000.0
        assert parse_numeric_quantity("19,1")[0].value == 19.1
        assert parse_numeric_quantity("0,001")[0].value == 0.001
        assert parse_numeric_quantity("1.500,25")[0].value == 1500.25
        assert parse_numeric_quantity("1,500.25")[0].value == 1500.25
        assert parse_numeric_quantity("1,250.5")[0].value == 1250.5
        assert compare_numeric_quantity_values("1,250", "1.25") == (True, None)
        assert compare_numeric_quantity_values("1,250", "1250") == (False, "numeric_mismatch")
        assert compare_numeric_quantity_values("1,000", "1.0") == (True, None)
        assert compare_numeric_quantity_values("1,000", "1000", thousands_sep=True) == (True, None)
        assert compare_numeric_quantity_values("1.500,25", "1500.25") == (True, None)

        # Unparseable/non-standard expressions escalate as (None, None)
        assert compare_numeric_quantity_values("1.5 × 10⁻³", "0.0015") == (None, None)
        assert compare_numeric_quantity_values("50%", "0.5") == (None, None)
        assert compare_numeric_quantity_values("10 (SD 2)", "10 ± 2") == (None, None)

        # Pint offset temperature units & unparseable domain units
        pytest.importorskip("pint")
        assert compare_unit_values("°C", "degC") == (True, None)
        assert compare_unit_values("°C", "K") == (None, None)
        assert compare_unit_values("°C", "K", strict_scale=True) == (False, "unit_scale_mismatch")
        assert compare_unit_values("IU/mL", "U/mL") == (None, None)
        assert compare_unit_values("CFU/mL", "cells/mL") == (None, None)

        # Compute validators preserve columns named value/val/x and safely handle None/inequalities
        pytest.importorskip("simpleeval")
        col_specs = [
            {"name": "compute_validators", "expression": "value > 0 and val == 2 and x == 5"},
        ]
        assert validate_record_compute_validators({"value": "10", "val": "2", "x": "5"}, col_specs)[0].valid is True
        assert validate_record_compute_validators({"value": "-1", "val": "2", "x": "5"}, col_specs)[0].valid is False

        ineq_specs = [
            {"name": "compute_validators", "expression": "MIC > 0 and MIC <= 10000"},
        ]
        for rec in ({"MIC": None}, {"MIC": ""}, {"MIC": ">4"}, {"MIC": "0.5 ± 0.1"}):
            assert validate_record_compute_validators(rec, ineq_specs)[0].valid is True

    def test_pint_whitelist_mcg_and_multislash(self):
        pytest.importorskip("pint")
        from ae.core.schema.quantity_validators import compare_unit_values

        spec = FieldValidatorSpec(name="unit")
        # Non-unit codes like C18, F1, K2 must be rejected
        for code in ("C18", "F1", "K2", "N2", "A2"):
            assert validate_field_value(spec, code, field_name="unit").valid is False

        # Known dimensional units with implicit positive exponents and mcg/multi-slash work
        for u in ("cm2", "m2", "m/s2", "W/m2", "N/mm2", "mcg/mL", "mg/kg/day", "mcg/kg/day", "mg/m2/day"):
            assert validate_field_value(spec, u, field_name="unit").valid is True

        assert compare_unit_values("mcg/mL", "µg/mL") == (True, None)
        assert compare_unit_values("mg/kg/day", "mg/(kg*day)") == (True, None)

    def test_numeric_quantity_zero_abs_tol_and_leading_dot_ranges(self):
        from ae.core.schema.quantity_validators import (
            compare_numeric_quantity_values,
            parse_numeric_quantity,
        )

        # abs_tol=0.0 prevents false matches for small magnitudes near 0 or 1e-13
        assert compare_numeric_quantity_values("1e-13", "5e-13") == (False, "numeric_mismatch")
        assert compare_numeric_quantity_values("1e-14", "0") == (False, "numeric_mismatch")
        assert compare_numeric_quantity_values("0", "0.0") == (True, None)

        # Leading-dot decimals in ranges
        parsed_range, err = parse_numeric_quantity(".5-1.0")
        assert err is None and parsed_range is not None
        assert parsed_range.kind == "range"
        assert parsed_range.low == 0.5 and parsed_range.high == 1.0

        parsed_bracket, err_b = parse_numeric_quantity("[.25, .75]")
        assert err_b is None and parsed_bracket is not None
        assert parsed_bracket.low == 0.25 and parsed_bracket.high == 0.75

        spec = FieldValidatorSpec(name="numeric_quantity")
        assert validate_field_value(spec, ".5-1.0", field_name="target_value").valid is True

    def test_compute_when_ast_validation_garbage_rejection_and_sanity_wiring(self):
        pytest.importorskip("simpleeval")
        from ae.core.schema.quantity_validators import validate_record_compute_validators
        from ae.extraction.sanity import (
            StructuredOutputSanityError,
            validate_output_field_validators,
            validate_structured_output_sanity,
        )

        # 1. validate_spec checks AST of 'when'
        bad_when_cfg = SchemaConfig(
            name="bad_when",
            fields={"mic": FieldSpec(type=str, description="MIC")},
            compare_fields=["mic"],
            compute_validators=[
                {
                    "name": "compute_validators",
                    "expression": "mic > 0",
                    "when": "__import__('os').system('echo x')",
                }
            ],
        )
        assert any("'when'" in err and "forbidden" in err for err in bad_when_cfg.validate())

        empty_when_cfg = SchemaConfig(
            name="empty_when",
            fields={"mic": FieldSpec(type=str, description="MIC")},
            compare_fields=["mic"],
            compute_validators=[
                {"name": "compute_validators", "expression": "mic > 0", "when": "   "}
            ],
        )
        assert any("'when'" in err for err in empty_when_cfg.validate())

        # 2. Unparseable garbage strings fail instead of silently passing via TypeError catch
        specs = [{"name": "compute_validators", "expression": "MIC > 0 and pMIC > 0"}]
        assert validate_record_compute_validators({"MIC": "garbage", "pMIC": "5"}, specs)[0].valid is False
        assert validate_record_compute_validators({"MIC": None, "pMIC": "bad"}, specs)[0].valid is False

        cond_specs = [
            {"name": "compute_validators", "expression": "unit == 'mg/mL' and MIC > 0"}
        ]
        assert validate_record_compute_validators({"unit": "mg/mL", "MIC": None}, cond_specs)[0].valid is True
        assert validate_record_compute_validators({"unit": "mg/mL", "MIC": "bad"}, cond_specs)[0].valid is False

        # 3. Schema-level compute_validators execute during sanity validation
        cfg = SchemaConfig(
            name="wired_compute",
            fields={
                "mic": FieldSpec(type=str, description="MIC"),
                "pmic": FieldSpec(type=str, description="pMIC"),
            },
            compare_fields=["mic"],
            compute_validators=[
                {
                    "name": "compute_validators",
                    "expression": "abs(pmic - (-log10(mic * 1e-6))) < 0.05",
                    "when": "mic > 0 and pmic > 0",
                    "message": "pMIC inconsistent with MIC",
                }
            ],
        )
        assert cfg.validate() == []
        rec_model, out_model = create_all_models(cfg)

        good_out = out_model(records=[rec_model(mic="10", pmic="5.0")])
        assert validate_output_field_validators(good_out, output_model=out_model) is good_out
        assert (
            validate_structured_output_sanity(
                good_out,
                document_text="MIC 10, pMIC 5.0",
                output_model=out_model,
            )
            is good_out
        )

        bad_out = out_model(
            records=[
                rec_model(mic="10", pmic="5.0"),
                rec_model(mic="10", pmic="3.0"),
            ]
        )
        with pytest.raises(StructuredOutputSanityError) as exc_info:
            validate_structured_output_sanity(
                bad_out,
                document_text="MIC 10, pMIC 3.0",
                output_model=out_model,
            )
        assert exc_info.value.failed_record_indices == [1]
        assert "pMIC inconsistent with MIC" in str(exc_info.value)
