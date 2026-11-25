"""Tests for RDKit SMILES validation and self-correction loop in Gemini Ingestion v2."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ae.core.config.optimization_settings import (
    GeminiDefaultsConfig,
    GeminiParserConfig,
    IngestionConfig,
)
from ae.ingestion.parsers.gemini.schemas import ChemicalEntity
from ae.ingestion.parsers.gemini.stages import ocsr_oracle
from ae.ingestion.parsers.gemini.stages.ocsr_oracle import (
    OracleFinding,
    same_connectivity,
    substructure_violations,
    verify_entities_with_oracle,
)
from ae.ingestion.parsers.gemini.stages.rdkit_validation import (
    SmilesCorrectionItem,
    SmilesCorrectionResponse,
    validate_and_canonicalize,
    validate_and_correct_entities,
)

RJC_CTX = (
    "Compound 15, which had an oxadiazole nucleus with a p-chlorophenyl group at C3, "
    "and a 2-chloro-6-fluorophenyl at C2 of the benzimidazole ring, with MIC 3.125 ug/mL."
)
BAD_15 = "Cn1c(-c2cc(Cl)ccc2F)nc2cc(-c3nc(-c4ccc(Cl)cc4)no3)ccc21"
GOOD_15 = "Cn1c(-c2c(F)cccc2Cl)nc2cc(-c3nc(-c4ccc(Cl)cc4)no3)ccc21"
PARA_15 = "Cn1c(-c2ccc(Cl)cc2F)nc2cc(-c3nc(-c4ccc(Cl)cc4)no3)ccc21"

NAME_37 = (
    "(2-(3-(4-fluorophenyl)-1-phenyl-1H-pyrazol-4-yl)-1H-benzo[d]imidazole-1-yl)"
    "(pyridin-4-yl)methanone"
)
SMI_37 = "O=C(c1ccncc1)n1c(-c2cn(-c3ccccc3)nc2-c2ccc(F)cc2)nc2ccccc21"
CTX_37 = (
    f"### 4.5.25. {NAME_37} (37)\n\n"
    "White powder, yield 76%. ESI-MS: 460.50 (C28H18FN5O, [M+H]+)."
)
BAD_37 = "O=C(c1cccnc1)n1c(-c2cn(-c3ccccc3)nc2-c2ccc(F)cc2)nc2ccccc21"


def _default_config() -> IngestionConfig:
    return IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )


@pytest.fixture
def primed_opsin(monkeypatch):
    """Deterministic py2opsin results without java; mirrors real output shapes."""
    cache = {
        ("2-chloro-6-fluorophenyl", True): "*c1c(F)cccc1Cl",
        ("4-fluorophenyl", True): "*c1ccc(F)cc1",
        ("pyridin-4-yl", True): "*c1ccncc1",
        ("p-chlorophenyl", True): "*c1ccc(Cl)cc1",
        (NAME_37, False): SMI_37,
    }
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", cache)
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: "/fake/java")
    return cache


@pytest.fixture
def no_opsin(monkeypatch):
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", {})
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: None)


def test_validate_smiles_valid():
    valid, canonical, err = validate_and_canonicalize("CCO")
    assert valid is True
    assert canonical == "CCO"
    assert err is None

    # Check non-canonical representation gets canonicalized
    valid, canonical, err = validate_and_canonicalize("OCC")
    assert valid is True
    assert canonical == "CCO"
    assert err is None


def test_validate_smiles_invalid():
    valid, canonical, err = validate_and_canonicalize("NOT_A_SMILES")
    assert valid is False
    assert err is not None


def test_validate_smiles_dummy_atoms_markush():
    # Support for RDKit dummy atoms with atom map numbers for unexpanded Markush scaffolds
    valid, canonical, err = validate_and_canonicalize("c1ccc([*:1])cc1")
    assert valid is True
    assert canonical == "c1ccc([*:1])cc1"
    assert err is None


def test_extract_molecules_rdkit_self_correction():
    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )
    client = MagicMock()

    # Self-correction mock provides valid SMILES
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[
            SmilesCorrectionItem(label="1", corrected_smiles="CCO"),
        ],
    )

    entities = [
        ChemicalEntity(
            label="1",
            name="Berberrubine",
            smiles="INVALID_SMILES_123",
            role="starting_material",
        ),
        ChemicalEntity(
            label="2",
            name="AceticAcid",
            smiles="CC(=O)O",
            role="intermediate",
        ),
    ]

    updated, replacements = validate_and_correct_entities(
        entities=entities,
        crop_paths=[],
        client=client,
        config=config,
    )

    ent1 = next(e for e in updated if e.label == "1")
    assert ent1.smiles == "CCO"
    assert ent1.smiles_valid is True
    assert ent1.rdkit_error is None
    assert "INVALID_SMILES_123" in replacements
    assert client.generate_structured.call_count == 1


def test_extract_molecules_rdkit_correction_exhausted():
    config = IngestionConfig(
        gemini=GeminiParserConfig(defaults=GeminiDefaultsConfig(model="gemini-3.8-flash"))
    )
    client = MagicMock()

    # Correction keeps returning invalid SMILES
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[
            SmilesCorrectionItem(label="X", corrected_smiles="STILL_INVALID"),
        ],
    )

    entities = [
        ChemicalEntity(
            label="X",
            smiles="BAD_SMILES_00",
            role="target_product",
        ),
    ]

    updated, _ = validate_and_correct_entities(
        entities=entities,
        crop_paths=[],
        client=client,
        config=config,
    )

    ent = updated[0]
    # Retries exhausted: entity kept, marked invalid
    assert ent.smiles_valid is False
    assert ent.rdkit_error is not None
    # Called MoleculeValidationConfig.max_retries times (default 1)
    assert client.generate_structured.call_count == 1


def test_oracle_finding_carries_structured_expectations(primed_opsin):
    ent = ChemicalEntity(label="37", smiles=BAD_37, role="target_product")
    findings = verify_entities_with_oracle([ent], context=CTX_37)
    f = findings[0]
    assert f.provenance_status == "mismatch_flagged"
    assert f.expected_smiles == SMI_37
    assert f.expected_iupac == NAME_37
    assert f.connectivity_conflict is True
    assert f.expected_inchi_key is not None

    bad_ent = ChemicalEntity(label="15", smiles="NOT_A_SMILES", smiles_valid=False)
    f_bad = verify_entities_with_oracle([bad_ent], context=RJC_CTX)[0]
    assert f_bad.provenance_status == "mismatch_flagged"
    assert f_bad.expected_substructure_smiles == ["*c1ccc(Cl)cc1", "*c1c(F)cccc1Cl"]
    assert f_bad.expected_substructure_names == ["p-chlorophenyl", "2-chloro-6-fluorophenyl"]


def test_same_connectivity_tautomer_and_isomer():
    from rdkit import Chem

    a = Chem.MolFromSmiles("C[C@H](O)c1ccccc1")
    b = Chem.MolFromSmiles("CC(O)c1ccccc1")
    assert same_connectivity(a, b) is True

    x = Chem.MolFromSmiles("O=C(c1cccnc1)c1ccccc1")
    y = Chem.MolFromSmiles("O=C(c1ccncc1)c1ccccc1")
    assert same_connectivity(x, y) is False


def test_substructure_violations_empty_and_markush():
    from rdkit import Chem

    plain = OracleFinding(label="1")
    assert substructure_violations(Chem.MolFromSmiles("CCO"), plain) == []

    f = OracleFinding(label="1", expected_substructure_names=["aryl"],
                      expected_substructure_smiles=["*c1ccccc1"])
    assert substructure_violations(Chem.MolFromSmiles("[*:1]c1ccc(Cl)cc1"), f) == []


def test_correction_prompt_injects_hints_and_gate_rejects_wrong_isomer(primed_opsin):
    client = MagicMock()
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="15", corrected_smiles=PARA_15)],
    )
    entities = [
        ChemicalEntity(label="15", smiles=BAD_15),
        ChemicalEntity(label="2", smiles="CC(=O)O"),
    ]
    updated, replacements = validate_and_correct_entities(
        entities=entities, crop_paths=[], client=client, config=_default_config(),
        max_attempts=1, context=RJC_CTX,
    )

    prompt = client.generate_structured.call_args[0][0][-1]
    assert "Prose-derived constraints" in prompt
    assert "*c1c(F)cccc1Cl" in prompt
    assert "Label '2'" not in prompt

    ent15 = updated[0]
    _, canon_bad, _ = validate_and_canonicalize(BAD_15)
    assert ent15.smiles == canon_bad
    assert ent15.provenance_status == "mismatch_flagged"
    assert "rejected" in (ent15.oracle_notes or "")
    _, canon_para, _ = validate_and_canonicalize(PARA_15)
    assert canon_para not in replacements.values()


def test_gate_accepts_correct_isomer(primed_opsin):
    client = MagicMock()
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="15", corrected_smiles=GOOD_15)],
    )
    entities = [ChemicalEntity(label="15", smiles=BAD_15)]
    updated, replacements = validate_and_correct_entities(
        entities=entities, crop_paths=[], client=client, config=_default_config(),
        max_attempts=1, context=RJC_CTX,
    )
    ent15 = updated[0]
    _, canon_good, _ = validate_and_canonicalize(GOOD_15)
    assert ent15.smiles == canon_good
    assert ent15.provenance_status == "unverified_valid"
    assert "rejected" not in (ent15.oracle_notes or "")
    assert canon_good in replacements.values()


def test_prose_override_when_corrections_exhausted(primed_opsin):
    client = MagicMock()
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="37", corrected_smiles="CCO")],
    )
    entities = [ChemicalEntity(label="37", smiles=BAD_37)]
    updated, _ = validate_and_correct_entities(
        entities=entities, crop_paths=[], client=client, config=_default_config(),
        max_attempts=1, context=CTX_37,
    )
    ent37 = updated[0]
    _, canon_37, _ = validate_and_canonicalize(SMI_37)
    assert ent37.smiles == canon_37
    assert ent37.provenance_status == "verified_exact"
    assert "overridden from prose IUPAC heading" in (ent37.oracle_notes or "")

    # Deterministic override works without any model call
    entities2 = [ChemicalEntity(label="37", smiles=BAD_37)]
    updated2, _ = validate_and_correct_entities(
        entities=entities2, crop_paths=[], client=None, config=_default_config(),
        max_attempts=0, context=CTX_37,
    )
    assert updated2[0].smiles == canon_37


def test_override_blocked_when_formula_disagrees_with_name(primed_opsin):
    ctx = (
        f"### 4.5.25. {NAME_37} (37)\n\n"
        "Compound 37 HRMS calcd for C20H20FN5O."
    )
    entities = [ChemicalEntity(label="37", smiles=BAD_37)]
    updated, _ = validate_and_correct_entities(
        entities=entities, crop_paths=[], client=None, config=_default_config(),
        max_attempts=0, context=ctx,
    )
    ent37 = updated[0]
    _, canon_bad, _ = validate_and_canonicalize(BAD_37)
    assert ent37.smiles == canon_bad
    assert ent37.provenance_status == "mismatch_flagged"


def test_opsin_absent_parity(no_opsin):
    client = MagicMock()
    entities = [ChemicalEntity(label="15", smiles=BAD_15)]
    updated, _ = validate_and_correct_entities(
        entities=entities, crop_paths=[], client=client, config=_default_config(),
        max_attempts=1, context=RJC_CTX,
    )
    assert updated[0].provenance_status == "unverified_valid"
    assert client.generate_structured.call_count == 0

    client2 = MagicMock()
    entities2 = [ChemicalEntity(label="37", smiles=BAD_37)]
    updated2, _ = validate_and_correct_entities(
        entities=entities2, crop_paths=[], client=client2, config=_default_config(),
        max_attempts=1, context=CTX_37,
    )
    # Without OPSIN, isomer is not flagged; matching ESI-MS formula in CTX_37 still verifies formula-level
    assert updated2[0].provenance_status == "verified_exact"
    assert client2.generate_structured.call_count == 0


def test_markush_untouched(primed_opsin):
    client = MagicMock()
    mk = "[*:1]c1ccc(C2=NC(=N2)C)cc1"
    entities = [ChemicalEntity(label="m", smiles=mk)]
    updated, _ = validate_and_correct_entities(
        entities=entities, crop_paths=[], client=client, config=_default_config(),
        max_attempts=1, context=RJC_CTX,
    )
    assert updated[0].smiles == validate_and_canonicalize(mk)[1]
    assert updated[0].provenance_status == "unverified_valid"
    assert client.generate_structured.call_count == 0
