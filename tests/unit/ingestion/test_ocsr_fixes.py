"""Regression tests for Contour 2 (EXP-01B) OCSR oracle, RDKit validation, and parse_visuals fixes."""

from __future__ import annotations

import concurrent.futures
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ae.core.config.optimization_settings import (
    IngestionConfig,
    IngestionStagesConfig,
    MoleculeExtractStageConfig,
    MoleculeValidationConfig,
    ParseVisualsStagesConfig,
    StageModelConfig,
)
from ae.ingestion.parsers.gemini.schemas import (
    ChemicalEntity,
    ManifestVisual,
    MolecularParseResult,
)
from ae.ingestion.parsers.gemini.stages import ocsr_oracle
from ae.ingestion.parsers.gemini.stages.ocsr_oracle import (
    _clean_latex_chemical_text,
    _resolve_iupac_via_opsin,
    extract_context_cues_for_labels,
    formulas_compatible,
    substructure_violations,
    verify_entities_with_oracle,
)
from ae.ingestion.parsers.gemini.stages.parse_visuals import parse_single_visual
from ae.ingestion.parsers.gemini.stages.rdkit_validation import (
    SmilesCorrectionItem,
    SmilesCorrectionResponse,
    validate_and_canonicalize,
    validate_and_correct_entities,
)

NAME_37 = (
    "(2-(3-(4-fluorophenyl)-1-phenyl-1H-pyrazol-4-yl)-1H-benzo[d]imidazole-1-yl)"
    "(pyridin-4-yl)methanone"
)
SMI_37 = "O=C(c1ccncc1)n1c(-c2cn(-c3ccccc3)nc2-c2ccc(F)cc2)nc2ccccc21"
BAD_37 = "O=C(c1cccnc1)n1c(-c2cn(-c3ccccc3)nc2-c2ccc(F)cc2)nc2ccccc21"


@pytest.fixture
def primed_opsin(monkeypatch):
    cache = {
        ("2-chloro-6-fluorophenyl", True): "*c1c(F)cccc1Cl",
        ("4-benzylphenyl", True): "*c1ccc(Cc2ccccc2)cc1",
        ("4-chlorobenzyl", True): "*Cc1ccc(Cl)cc1",
        ("4-chlorobenzoyl", True): "*C(=O)c1ccc(Cl)cc1",
        (NAME_37, False): SMI_37,
    }
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", cache)
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: "/fake/java")
    return cache


def test_clean_latex_chemical_text_preserves_newlines_for_headings_and_paragraphs():
    raw = (
        "Preceding text:\n"
        f"### 4.5.25. **{NAME_37}** (37)\n\n"
        "White powder,   yield 76%.   ESI-MS: 460.50 (C_{28}H_{18}FN_5O, [M+H]^+)."
    )
    cleaned = _clean_latex_chemical_text(raw)
    assert "\n\n" in cleaned
    assert cleaned.startswith("Preceding text:\n### 4.5.25.")

    cues = extract_context_cues_for_labels(raw, ["37"])
    assert cues["37"]["iupac_names"] == [NAME_37]
    assert cues["37"]["formulas"] == ["C28H18FN5O"]


def test_extract_cues_keeps_benzyl_and_benzoyl_substituents():
    ctx = (
        "Compound 10 carries a 4-chlorobenzyl group and a 4-chlorobenzoyl moiety.\n\n"
        "Compound 11 carries a 2-chloro-6-fluorophenyl ring."
    )
    cues = extract_context_cues_for_labels(ctx, ["10", "11"])
    assert "4-chlorobenzyl" in cues["10"]["substituents"]
    assert "4-chlorobenzoyl" in cues["10"]["substituents"]
    assert "2-chloro-6-fluorophenyl" in cues["11"]["substituents"]


def test_resolve_iupac_via_opsin_thread_safe_and_never_caches_failures(monkeypatch):
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", {})
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: "/fake/java")

    seen_tmp_paths: list[str] = []
    call_counter = {"count": 0}

    def fake_py2opsin(names, *, allow_radicals=False, wildcard_radicals=False, tmp_fpath=""):
        seen_tmp_paths.append(tmp_fpath)
        assert Path(tmp_fpath).exists()
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return False  # Simulate subprocess failure sentinel
        return [f"C_{idx}" for idx, _ in enumerate(names)]

    fake_mod = types.SimpleNamespace(py2opsin=fake_py2opsin)
    monkeypatch.setitem(sys.modules, "py2opsin", fake_mod)

    # First call fails -> returns None and MUST NOT cache None
    first = _resolve_iupac_via_opsin(["ethane"])
    assert first == {"ethane": None}
    assert ("ethane", False) not in ocsr_oracle._OPSIN_CACHE

    # Second call succeeds -> resolves and caches
    second = _resolve_iupac_via_opsin(["ethane"])
    assert second == {"ethane": "C_0"}
    assert ocsr_oracle._OPSIN_CACHE[("ethane", False)] == "C_0"

    # Concurrent calls must use distinct temporary files
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_resolve_iupac_via_opsin, [f"mol_{i}"]) for i in range(12)]
        results = [f.result() for f in futs]

    assert all(next(iter(r.values())) == "C_0" for r in results)
    assert len(set(seen_tmp_paths)) == len(seen_tmp_paths)
    assert all(not Path(p).exists() for p in seen_tmp_paths)


def test_radical_pattern_unified_for_stereo_radicals(monkeypatch):
    # Chiral wildcard radical SMILES should match achiral target consistently in both functions
    monkeypatch.setattr(
        ocsr_oracle,
        "_OPSIN_CACHE",
        {("2-chloro-6-fluorophenyl", True): "*[C@@H](Cl)c1c(F)cccc1Cl"},
    )
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: "/fake/java")

    from rdkit import Chem

    ent = ChemicalEntity(label="1", smiles="CC(Cl)c1c(F)cccc1Cl")
    findings = verify_entities_with_oracle(
        [ent],
        context="Compound 1 has a 2-chloro-6-fluorophenyl group.",
    )
    assert findings[0].passed is True
    assert substructure_violations(Chem.MolFromSmiles(ent.smiles), findings[0]) == []


def test_formulas_compatible_adducts_and_oracle_integration(primed_opsin):
    # Without adduct suffix, Na/K/NH4 mismatch fails
    assert formulas_compatible("C28H18FN5O", "C28H18FN5ONa") is False
    assert formulas_compatible("C28H18FN5O", "C28H18FN5OK") is False
    assert formulas_compatible("C28H18FN5O", "C28H22FN6O") is False

    # With adduct suffix in context or formula, matches cleanly
    assert formulas_compatible("C28H18FN5O", "C28H18FN5ONa", context="HRMS [M+Na]+") is True
    assert formulas_compatible("C28H18FN5O", "C28H18FN5OK", context="ESI-MS ([M+K]+)") is True
    assert formulas_compatible("C28H18FN5O", "C28H22FN6O", context="HRMS [M+NH4]+") is True
    assert formulas_compatible("C28H18FN5O", "C28H18FN5ONa [M+Na]+") is True
    assert formulas_compatible("C29H18FN5O", "C28H18FN5ONa", context="HRMS [M+Na]+") is False

    # End-to-end oracle + deterministic override with [M+Na]+
    ctx_na = (
        "Preceding text:\n"
        f"### 4.5.25. {NAME_37} (37)\n\n"
        "HRMS (ESI): calcd for C28H18FN5ONa [M+Na]+ 482.1393."
    )
    ent_ok = ChemicalEntity(label="37", smiles=SMI_37)
    findings_ok = verify_entities_with_oracle([ent_ok], context=ctx_na)
    assert findings_ok[0].passed is True
    assert findings_ok[0].provenance_status == "verified_exact"


def test_invalid_correction_does_not_overwrite_valid_smiles(primed_opsin):
    client = MagicMock()
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="37", corrected_smiles="BROKEN(((")],
    )
    ent = ChemicalEntity(label="37", smiles=BAD_37)
    _, canon_bad, _ = validate_and_canonicalize(BAD_37)

    ctx_no_override = (
        "Compound 37 has a 2-chloro-6-fluorophenyl substituent."
    )
    updated, replacements = validate_and_correct_entities(
        entities=[ent],
        crop_paths=[],
        client=client,
        config=IngestionConfig(),
        max_attempts=1,
        context=ctx_no_override,
    )

    assert updated[0].smiles_valid is True
    assert updated[0].smiles == canon_bad
    assert updated[0].rdkit_error is None
    assert "BROKEN(((" not in replacements.values()
    assert "rejected" in (updated[0].oracle_notes or "")


def test_enable_ocsr_oracle_false_skips_oracle_in_retry_loop(monkeypatch):
    called = {"count": 0}

    def fail_if_called(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("verify_entities_with_oracle must not be called when disabled")

    monkeypatch.setattr(
        "ae.ingestion.parsers.gemini.stages.rdkit_validation.verify_entities_with_oracle",
        fail_if_called,
    )

    cfg = IngestionConfig(
        stages=IngestionStagesConfig(
            parse_visuals=ParseVisualsStagesConfig(
                molecule=MoleculeExtractStageConfig(
                    validation=MoleculeValidationConfig(
                        validate_rdkit=True,
                        enable_ocsr_oracle=False,
                        max_retries=1,
                    )
                )
            )
        )
    )
    client = MagicMock()
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="1", corrected_smiles="CCO")],
    )
    updated, _ = validate_and_correct_entities(
        entities=[ChemicalEntity(label="1", smiles="BAD_SMILES")],
        crop_paths=[],
        client=client,
        config=cfg,
        context="Compound 1 HRMS calcd for C10H10O.",
    )
    assert called["count"] == 0
    assert updated[0].smiles == "CCO"
    assert updated[0].smiles_valid is True


def test_deterministic_override_for_rdkit_invalid_entity(primed_opsin):
    ctx = (
        "Preceding text:\n"
        f"### 4.5.25. {NAME_37} (37)\n\n"
        "HRMS (ESI): calcd for C28H18FN5ONa [M+Na]+ 482.1393."
    )
    ent = ChemicalEntity(label="37", smiles="UNPARSABLE((((")
    updated, replacements = validate_and_correct_entities(
        entities=[ent],
        crop_paths=[],
        client=None,
        config=IngestionConfig(),
        max_attempts=0,
        context=ctx,
    )
    _, canon_37, _ = validate_and_canonicalize(SMI_37)
    assert updated[0].smiles_valid is True
    assert updated[0].smiles == canon_37
    assert updated[0].rdkit_error is None
    assert updated[0].provenance_status == "verified_exact"
    assert replacements["UNPARSABLE(((("] == canon_37

    # Formula disagreement still blocks override even for RDKit-invalid SMILES
    bad_ctx = f"### 4.5.25. {NAME_37} (37)\n\nCompound 37 HRMS calcd for C10H10O."
    ent_bad = ChemicalEntity(label="37", smiles="UNPARSABLE((((")
    updated_bad, _ = validate_and_correct_entities(
        entities=[ent_bad],
        crop_paths=[],
        client=None,
        config=IngestionConfig(),
        max_attempts=0,
        context=bad_ctx,
    )
    assert updated_bad[0].smiles_valid is False
    assert updated_bad[0].smiles == "UNPARSABLE(((("


def test_stage_cfg_forwarded_max_retries_and_noop_short_circuits(primed_opsin):
    cfg = IngestionConfig(
        stages=IngestionStagesConfig(
            parse_visuals=ParseVisualsStagesConfig(
                molecule=MoleculeExtractStageConfig(
                    model=StageModelConfig(
                        model="gemini-3.8-pro",
                        temperature=0.2,
                        max_output_tokens=4096,
                        media_resolution="medium",
                        thinking_level="high",
                        thinking_budget=2048,
                    ),
                    validation=MoleculeValidationConfig(max_retries=2),
                )
            )
        )
    )
    client = MagicMock()
    # Return exact same SMILES (no-op) on attempt 1 -> must NOT invoke attempt 2
    client.generate_structured.return_value = SmilesCorrectionResponse(
        corrections=[SmilesCorrectionItem(label="37", corrected_smiles=BAD_37)],
    )
    # Without IUPAC override, use substituent mismatch to stay mismatch_flagged without override
    ctx = "Compound 37 has a 2-chloro-6-fluorophenyl group."
    ent = ChemicalEntity(label="37", smiles=BAD_37)
    _, canon_bad, _ = validate_and_canonicalize(BAD_37)

    updated, replacements = validate_and_correct_entities(
        entities=[ent],
        crop_paths=[],
        client=client,
        config=cfg,
        context=ctx,
    )

    assert client.generate_structured.call_count == 1
    kwargs = client.generate_structured.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-pro"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_output_tokens"] == 4096
    assert kwargs["media_resolution"] == "medium"
    assert kwargs["thinking_level"] == "high"
    assert kwargs["thinking_budget"] == 2048
    assert updated[0].smiles == canon_bad
    assert canon_bad not in replacements


def test_parse_single_visual_uses_rdkit_correction_stage_name(tmp_path: Path):
    cfg = IngestionConfig()
    vis = ManifestVisual(
        id="Scheme_1",
        type="molecular_structure",
        pages=[1],
        description="Compound 1",
    )
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"PNG")

    mock_client = MagicMock()
    mock_client.generate_structured.side_effect = [
        MolecularParseResult(entities=[ChemicalEntity(label="1", smiles="INVALID_@@@")]),
        SmilesCorrectionResponse(corrections=[SmilesCorrectionItem(label="1", corrected_smiles="CCO")]),
    ]

    res = parse_single_visual(mock_client, vis, [crop], "Scheme 1", cfg)
    assert isinstance(res, MolecularParseResult)
    assert res.entities[0].smiles == "CCO"
    assert mock_client.generate_structured.call_count == 2
    assert mock_client.generate_structured.call_args_list[0].kwargs["stage_name"] == "parse_molecular_structure"
    assert mock_client.generate_structured.call_args_list[1].kwargs["stage_name"] == "rdkit_correction"


def test_opsin_unavailable_logs_single_warning_and_uses_semaphore(monkeypatch, caplog):
    import logging
    import threading

    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", {})
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_UNAVAILABLE_WARNED", False)
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: None)

    assert isinstance(ocsr_oracle._OPSIN_SEMAPHORE, threading.Semaphore)

    with caplog.at_level(logging.WARNING):
        res1 = _resolve_iupac_via_opsin(["benzene"])
        res2 = _resolve_iupac_via_opsin(["toluene"])

    assert res1 == {"benzene": None}
    assert res2 == {"toluene": None}
    opsin_warnings = [r for r in caplog.records if "py2opsin IUPAC oracle unavailable" in r.message]
    assert len(opsin_warnings) == 1
