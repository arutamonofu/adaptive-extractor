"""Tests for RDKit SMILES validation and self-correction loop in Gemini Ingestion v2."""

from __future__ import annotations

from unittest.mock import MagicMock

from ae.core.config.optimization_settings import (
    GeminiDefaultsConfig,
    GeminiParserConfig,
    IngestionConfig,
)
from ae.ingestion.parsers.gemini.schemas import ChemicalEntity
from ae.ingestion.parsers.gemini.stages.rdkit_validation import (
    SmilesCorrectionItem,
    SmilesCorrectionResponse,
    validate_and_canonicalize,
    validate_and_correct_entities,
)


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
    # Called max_rdkit_retries times (2)
    assert client.generate_structured.call_count == 2
