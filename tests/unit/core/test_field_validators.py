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
