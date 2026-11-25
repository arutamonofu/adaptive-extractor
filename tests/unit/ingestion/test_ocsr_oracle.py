"""Focused unit tests for deterministic OCSR oracle (EXP-01B)."""

from __future__ import annotations

import sys
import types
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from ae.ingestion.parsers.gemini.schemas import ChemicalEntity
from ae.ingestion.parsers.gemini.stages import ocsr_oracle
from ae.ingestion.parsers.gemini.stages.ocsr_oracle import (
    OracleFinding,
    extract_context_cues_for_labels,
    formulas_compatible,
    parse_formula_counts,
    same_connectivity,
    substructure_violations,
    verify_entities_with_oracle,
)


def test_bare_formula_re_removed():
    assert not hasattr(ocsr_oracle, "_BARE_FORMULA_RE")


def test_len_one_fallback_restricts_unrelated_adjacent_page_paragraphs():
    # Paragraphs from earlier/later pages, cross-refs, callouts, or mentioning other compounds
    # must NOT be blindly bound to labels[0].
    ctx_unrelated = (
        "Preceding text:\n"
        "White solid, yield 81%. HRMS calcd for C15H12N2O.\n\n"
        "Compound 36 was synthesized by route A. HRMS calcd for C18H16N2O.\n\n"
        "Following text:\n"
        "Compound 38 showed weak activity. HRMS calcd for C22H20O3.\n\n"
        "Later characterization: HRMS calcd for C25H22O4.\n\n"
        "Cross-referenced paragraphs:\n"
        "Reference analog HRMS calcd for C19H18O2.\n\n"
        "In-text callouts:\n"
        "Callout note HRMS calcd for C21H20O3."
    )
    cues = extract_context_cues_for_labels(ctx_unrelated, ["37"])
    assert cues["37"]["formulas"] == []

    # Directly adjacent paragraph without other compound IDs IS allowed for single-entity visual
    ctx_adjacent = (
        "Preceding text:\n"
        "Unrelated earlier page paragraph. HRMS calcd for C15H12N2O.\n\n"
        "White powder, yield 76%. ESI-MS: 460.50 (C28H18FN5O, [M+H]+).\n\n"
        "Following text:\n"
        "Further biological assays were performed."
    )
    cues_adj = extract_context_cues_for_labels(ctx_adjacent, ["37"])
    assert cues_adj["37"]["formulas"] == ["C28H18FN5O"]

    # Adjacent paragraph after an experimental heading for another compound must NOT bind
    ctx_other_heading = (
        "Preceding text:\n"
        "### 4.5.24. Some other derivative (36)\n\n"
        "White powder, yield 70%. HRMS calcd for C20H20N2O."
    )
    cues_other = extract_context_cues_for_labels(ctx_other_heading, ["37"])
    assert cues_other["37"]["formulas"] == []


def test_sent_labels_require_anchors_for_short_numeric_labels():
    # Sentence for Compound 1 contains bare '2' ("2 equiv", "at C2,");
    # bare \b2\b must not match label "2" and spoil unique sentence anchoring.
    ctx = (
        "Compound 1 was treated with 2 equiv reagent at C2, affording HRMS calcd for C10H12N2O. "
        "Compound 2 gave HRMS calcd for C12H14N2O."
    )
    cues = extract_context_cues_for_labels(ctx, ["1", "2"])
    assert cues["1"]["formulas"] == ["C10H12N2O"]
    assert cues["2"]["formulas"] == ["C12H14N2O"]

    # Anchored forms (X) and X, still match short numeric labels
    ctx_anchored = "(1) gave HRMS calcd for C10H12N2O. 2, gave HRMS calcd for C12H14N2O."
    cues_anchored = extract_context_cues_for_labels(ctx_anchored, ["1", "2"])
    assert cues_anchored["1"]["formulas"] == ["C10H12N2O"]
    assert cues_anchored["2"]["formulas"] == ["C12H14N2O"]


def test_candidate_formulas_compatibility_across_multiple_formulas(monkeypatch):
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", OrderedDict())
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: None)

    # First formula mention is a different/salt formula; second matches ethanol (C2H6O)
    ctx = (
        "Compound 1 gave HRMS calcd for C10H10O. "
        "Compound 1 also gave HRMS calcd for C2H6O."
    )
    ent = ChemicalEntity(label="1", smiles="CCO")
    findings = verify_entities_with_oracle([ent], context=ctx)
    assert findings[0].passed is True
    assert findings[0].provenance_status == "verified_exact"
    assert findings[0].expected_formulas == ["C10H10O", "C2H6O"]
    assert findings[0].expected_formula == "C2H6O"

    from rdkit import Chem

    f_multi = OracleFinding(
        label="1",
        expected_formula="C10H10O",
        expected_formulas=["C10H10O", "C2H6O"],
    )
    assert substructure_violations(Chem.MolFromSmiles("CCO"), f_multi) == []
    assert len(substructure_violations(Chem.MolFromSmiles("CCC"), f_multi)) == 1


def test_same_connectivity_inchikey_tautomers_and_smarts_fallback():
    from rdkit import Chem

    # 2-hydroxypyridine vs 2-pyridone: tautomers share InChIKey[:14]
    enol = Chem.MolFromSmiles("Oc1ccccn1")
    keto = Chem.MolFromSmiles("O=c1cccc[nH]1")
    assert same_connectivity(enol, keto) is True

    # Regioisomers differ in InChIKey[:14]
    iso_a = Chem.MolFromSmiles("Oc1cccnc1")
    assert same_connectivity(enol, iso_a) is False

    # Dummy-atom radicals lack InChIKey -> fall back to SMARTS
    rad_a = Chem.MolFromSmiles("*c1ccc(Cl)cc1")
    rad_b = Chem.MolFromSmiles("*c1ccc(Cl)cc1")
    rad_c = Chem.MolFromSmiles("*c1cccc(Cl)c1")
    assert same_connectivity(rad_a, rad_b) is True
    assert same_connectivity(rad_a, rad_c) is False


def test_substituent_locant_regex_di_tri_substituted_rings():
    ctx = (
        "Compound 1 contains a 3,4-dichlorophenyl group. "
        "Compound 2 bears a 2,6-difluorophenyl ring. "
        "Compound 3 has a 3,4,5-trimethoxyphenyl substituent."
    )
    cues = extract_context_cues_for_labels(ctx, ["1", "2", "3"])
    assert "3,4-dichlorophenyl" in cues["1"]["substituents"]
    assert "2,6-difluorophenyl" in cues["2"]["substituents"]
    assert "3,4,5-trimethoxyphenyl" in cues["3"]["substituents"]


def test_adduct_suffix_regex_and_formulas_compatible_extended_adducts(monkeypatch):
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", OrderedDict())
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: None)

    assert formulas_compatible("C28H18FN5O", "C28H18ClFN5O [M+Cl]-") is True
    assert formulas_compatible("C28H18FN5O", "C28H18ClFN5O", context="HRMS ([M+Cl]-)") is True
    assert formulas_compatible("C28H18FN5O", "C29H19FN5O3 [M+HCOO]-") is True
    assert formulas_compatible("C28H18FN5O", "C28H20FN5O [M+2H]2+") is True
    assert formulas_compatible("C28H18FN5O", "C28H17FN5 [M+H-H2O]+") is True

    # Wrong formulas still fail even with adduct context
    assert formulas_compatible("C27H18FN5O", "C28H18ClFN5O [M+Cl]-") is False
    assert formulas_compatible("C28H18FN5O", "C28H18ClFN5O") is False

    # End-to-end cue extraction + oracle verification with [M+2H]2+ and [M+Cl]-
    ctx = (
        "Compound 1: ESI-MS calcd for C2H8O [M+2H]2+ 48.0.\n\n"
        "Compound 2: HRMS calcd for C2H6ClO [M+Cl]- 81.0."
    )
    e1 = ChemicalEntity(label="1", smiles="CCO")
    e2 = ChemicalEntity(label="2", smiles="CCO")
    findings = verify_entities_with_oracle([e1, e2], context=ctx)
    assert findings[0].passed is True
    assert findings[0].provenance_status == "verified_exact"
    assert findings[1].passed is True
    assert findings[1].provenance_status == "verified_exact"


def test_opsin_cache_lru_bounded(monkeypatch):
    cache: OrderedDict[tuple[str, bool], str | None] = OrderedDict()
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", cache)
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE_MAX_SIZE", 3)
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: "/fake/java")

    fake_py2opsin = MagicMock(side_effect=lambda names, **_kw: [f"SMI_{n}" for n in names])
    monkeypatch.setitem(sys.modules, "py2opsin", types.SimpleNamespace(py2opsin=fake_py2opsin))

    ocsr_oracle._resolve_iupac_via_opsin(["a", "b", "c"])
    assert list(cache.keys()) == [("a", False), ("b", False), ("c", False)]

    # Touch "a" so "b" becomes least recently used
    ocsr_oracle._resolve_iupac_via_opsin(["a"])
    ocsr_oracle._resolve_iupac_via_opsin(["d"])

    assert len(cache) == 3
    assert ("b", False) not in cache
    assert ("a", False) in cache
    assert ("c", False) in cache
    assert ("d", False) in cache


def test_formulas_compatible_empty_pred_and_extended_formula_parsing(monkeypatch):
    monkeypatch.setattr(ocsr_oracle, "_OPSIN_CACHE", OrderedDict())
    monkeypatch.setattr(ocsr_oracle.shutil, "which", lambda _x: None)

    assert formulas_compatible("", "C10H10O") is False
    assert formulas_compatible("INVALID", "C10H10O") is False
    assert formulas_compatible("O2", "C10H10O") is False
    assert formulas_compatible("C10H10O", "") is True

    assert parse_formula_counts("H2O") == {"H": 2, "O": 1}
    assert parse_formula_counts("CHCl3") == {"C": 1, "H": 1, "Cl": 3}
    assert parse_formula_counts("C6HF5") == {"C": 6, "H": 1, "F": 5}
    assert parse_formula_counts("C6Cl6") == {"C": 6, "Cl": 6}
    assert parse_formula_counts(r"\mathrm{C}_{6}\mathrm{HCl}_{3}") == {"C": 6, "H": 1, "Cl": 3}
    assert parse_formula_counts(r"\rm C_6Cl_6") == {"C": 6, "Cl": 6}

    ent = ChemicalEntity(label="1", smiles="O=O")
    findings = verify_entities_with_oracle([ent], context="Compound 1 HRMS calcd for C10H12N2O.")
    assert findings[0].passed is False
    assert findings[0].provenance_status == "mismatch_flagged"


def test_formula_candidate_re_word_boundary_and_halogenated_formulas():
    ctx_false = (
        "Compound 1 exists in several forms: CH2Cl2 was used as solvent. "
        "Compound 2 systems gave C6H6 as byproduct. Grams of C2H6O were recovered."
    )
    cues_false = extract_context_cues_for_labels(ctx_false, ["1", "2"])
    assert cues_false["1"]["formulas"] == []
    assert cues_false["2"]["formulas"] == []

    ctx_true = (
        "Compound 1 HRMS calcd for CHCl3. "
        "Compound 2 HRMS calcd for C6HF5. "
        "Compound 3 Anal. Calcd. for C6Cl6."
    )
    cues_true = extract_context_cues_for_labels(ctx_true, ["1", "2", "3"])
    assert cues_true["1"]["formulas"] == ["CHCl3"]
    assert cues_true["2"]["formulas"] == ["C6HF5"]
    assert cues_true["3"]["formulas"] == ["C6Cl6"]


def test_explicit_compound_id_re_ignores_nmr_and_stereo_descriptors():
    ctx_nmr = (
        "Preceding text:\n"
        "White solid; 1H NMR (2H, s) and isomer (2R), (3S), (1E), (2Z); HRMS calcd for C28H18FN5O."
    )
    cues = extract_context_cues_for_labels(ctx_nmr, ["37"])
    assert cues["37"]["formulas"] == ["C28H18FN5O"]

    # Real lowercase compound ID in parens still blocks foreign inheritance
    ctx_other = (
        "Preceding text:\n"
        "Derivative (2a) showed HRMS calcd for C28H18FN5O."
    )
    cues_other = extract_context_cues_for_labels(ctx_other, ["37"])
    assert cues_other["37"]["formulas"] == []


def test_extract_cues_wrapped_surrounding_context_and_alphanumeric_paragraph():
    wrapped_ctx = (
        "Element Placement in Visual:\n"
        '<element id="mol_37" />\n\n'
        "Surrounding Article Context:\n"
        "Preceding text:\n"
        "Unrelated earlier page paragraph. HRMS calcd for C15H12N2O.\n\n"
        "White powder, yield 76%. ESI-MS: 460.50 (C28H18FN5O, [M+H]+).\n\n"
        "Following text:\n"
        "Further biological assays were performed."
    )
    cues = extract_context_cues_for_labels(wrapped_ctx, ["37"])
    assert cues["37"]["formulas"] == ["C28H18FN5O"]

    ctx_alpha = (
        "2a was obtained as a white solid. HRMS calcd for C15H14N2O.\n\n"
        "2b was obtained as an oil. HRMS calcd for C16H16N2O."
    )
    cues_alpha = extract_context_cues_for_labels(ctx_alpha, ["2a", "2b"])
    assert cues_alpha["2a"]["formulas"] == ["C15H14N2O"]
    assert cues_alpha["2b"]["formulas"] == ["C16H16N2O"]
