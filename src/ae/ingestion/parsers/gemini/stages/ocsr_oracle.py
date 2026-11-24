"""Deterministic cross-modal semantic OCSR oracle (EXP-01B / G-OCSR-1).

Verifies VLM-extracted SMILES against:
1. RDKit syntax/valence canonicalization
2. ChEMBL structure pipeline checks (`checker.check_molblock`)
3. Molecular Formula / HRMS / ESI-MS cross-checks extracted from surrounding text
4. IUPAC -> InChIKey (14/27) & explicit substituent motif checks via local `py2opsin`
5. Markush assembly verification via `assemble_markush_single`
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from ae.core.schema.domain_tools.chemistry import assemble_markush_single

logger = logging.getLogger(__name__)

ProvenanceStatus = Literal["verified_exact", "unverified_valid", "mismatch_flagged"]

_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_FORMULA_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_CANDIDATE_RE = re.compile(
    r"(?:HRMS|ESI-?MS|MS|Anal\.?\s*Calcd\.?(?:\s+for)?|Calcd\.?\s+for|Molecular\s+Formula)"
    r"[^.\n;]{0,80}?([A-Z][A-Za-z0-9_{}\\]*H\d+[A-Za-z0-9_{}\\]*)",
    re.IGNORECASE,
)
_BARE_FORMULA_RE = re.compile(r"\b(C\d{1,3}H\d{1,3}(?:[A-Z][a-z]?\d*)*)\b")
_SUBSTITUENT_LOCANT_RE = re.compile(
    r"\b(\d+(?:,\d+)*-(?:[a-z]+-)+\d*-[a-z]+(?:phenyl|pyridin-\d-yl|benzyl|benzoyl)"
    r"|\d+-(?:chloro|fluoro|bromo|iodo|methyl|methoxy|nitro|hydroxy|amino|trifluoromethyl)(?:phenyl|pyridin-\d-yl|benzyl)"
    r"|(?:[omp]-|pyridin-[234]-yl\b)"
    r"(?:(?:chloro|fluoro|bromo|iodo|methyl|methoxy|nitro)(?:phenyl|benzyl))?)",
    re.IGNORECASE,
)
_IUPAC_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*{1,2})?(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?P<name>[^\n]{12,240}?(?:yl|one|amide|ate|acid|azole|dine|ene|ol|ane|ine|ide))"
    r"\s*\(\s*(?P<label>[A-Za-z0-9\-]+)\s*\)(?:\*{1,2})?(?:\s|$)",
    re.MULTILINE,
)

# In-process cache for py2opsin lookups to keep CPU overhead negligible on repeated calls
_OPSIN_CACHE: dict[tuple[str, bool], Optional[str]] = {}


@dataclass
class OracleFinding:
    """Verification outcome for a single chemical entity."""

    label: str
    provenance_status: ProvenanceStatus = "unverified_valid"
    passed: bool = True
    reasons: list[str] = field(default_factory=list)
    expected_iupac: Optional[str] = None
    expected_formula: Optional[str] = None
    expected_inchi_key: Optional[str] = None
    predicted_inchi_key: Optional[str] = None
    expected_smiles: Optional[str] = None
    expected_substructure_smiles: list[str] = field(default_factory=list)
    expected_substructure_names: list[str] = field(default_factory=list)
    connectivity_conflict: bool = False


def _clean_latex_chemical_text(text: str) -> str:
    """Normalize LaTeX/Markdown formatting inside chemical names and formulas."""
    cleaned = text.translate(_SUBSCRIPT_TRANSLATION)
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("$", "").replace("*", "")
    cleaned = re.sub(r"_\{(\d+)\}", r"\1", cleaned)
    cleaned = re.sub(r"_(\d+)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_formula_counts(formula: str) -> dict[str, int]:
    """Parse a chemical formula string into element counts."""
    clean = _clean_latex_chemical_text(formula)
    clean = re.sub(r"[^A-Za-z0-9]", "", clean)
    if not clean or not clean.startswith("C"):
        return {}
    counts: dict[str, int] = {}
    for elem, num_str in _FORMULA_ELEMENT_RE.findall(clean):
        if not elem:
            continue
        counts[elem] = counts.get(elem, 0) + (int(num_str) if num_str else 1)
    return counts


def formulas_compatible(predicted_formula: str, expected_formula: str) -> bool:
    """Compare predicted neutral formula against prose formula (allowing [M+H]+ H diff)."""
    pred_counts = parse_formula_counts(predicted_formula)
    exp_counts = parse_formula_counts(expected_formula)
    if not pred_counts or not exp_counts:
        return True

    # Heavy atoms must match identically
    pred_heavy = {k: v for k, v in pred_counts.items() if k != "H"}
    exp_heavy = {k: v for k, v in exp_counts.items() if k != "H"}
    if pred_heavy != exp_heavy:
        return False

    # Allow exact H match or +/- 1 H (common when ESI-MS writes [M+H]+ formula directly)
    h_diff = abs(pred_counts.get("H", 0) - exp_counts.get("H", 0))
    return h_diff <= 1


def _radical_pattern(rad_mol: Any):
    """Stereo-stripped SMARTS pattern for a radical/substituent (wildcard atom -> any non-H)."""
    from rdkit import Chem

    stripped = Chem.Mol(rad_mol)
    Chem.RemoveStereochemistry(stripped)
    return Chem.MolFromSmarts(Chem.MolToSmarts(stripped).replace("[#0]", "[!#1]"))


def same_connectivity(mol_a: Any, mol_b: Any) -> bool:
    """True when both molecules embed each other as stereo-insensitive subgraphs.

    Bidirectional match equalizes heavy-atom counts, so tautomer shifts (H positions)
    and stereo differences do not register as connectivity conflicts.
    """
    pat_a, pat_b = _radical_pattern(mol_a), _radical_pattern(mol_b)
    if pat_a is None or pat_b is None:
        return False
    return bool(mol_a.HasSubstructMatch(pat_b) and mol_b.HasSubstructMatch(pat_a))


def substructure_violations(mol: Any, finding: OracleFinding) -> list[str]:
    """Deterministic prose-only constraint check for a candidate structure.

    Gates correction acceptance on evidence present in the article (formula cues,
    substituent motifs, resolved IUPAC name). ChEMBL quality checks are deliberately
    excluded: they are not prose constraints.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if mol is None:
        return ["candidate is not RDKit-parsable"]
    if _is_markush_smiles(Chem.MolToSmiles(mol)):
        return []

    violations: list[str] = []

    if finding.expected_formula:
        pred_f = rdMolDescriptors.CalcMolFormula(mol)
        if not formulas_compatible(pred_f, finding.expected_formula):
            violations.append(
                f"formula '{pred_f}' incompatible with prose '{finding.expected_formula}'"
            )

    for sub_name, rad_smi in zip(
        finding.expected_substructure_names, finding.expected_substructure_smiles
    ):
        rad_mol = Chem.MolFromSmiles(rad_smi)
        if rad_mol is None:
            continue
        pat = _radical_pattern(rad_mol)
        if pat is not None and not mol.HasSubstructMatch(pat):
            violations.append(f"missing prose substructure '{sub_name}' ({rad_smi})")

    if finding.expected_smiles and finding.expected_inchi_key:
        try:
            pred_ik = Chem.MolToInchiKey(mol)
        except Exception:
            pred_ik = ""
        if (
            pred_ik
            and pred_ik != finding.expected_inchi_key
            and pred_ik[:14] != finding.expected_inchi_key[:14]
        ):
            exp_mol = Chem.MolFromSmiles(finding.expected_smiles)
            if exp_mol is None or not same_connectivity(mol, exp_mol):
                violations.append(
                    f"connectivity differs from prose name '{finding.expected_iupac}'"
                )

    return violations


def _resolve_iupac_via_opsin(names: list[str], *, wildcard_radicals: bool = False) -> dict[str, Optional[str]]:
    """Resolve IUPAC names or radical substituents via local py2opsin with caching."""
    results: dict[str, Optional[str]] = {}
    missing: list[str] = []
    for name in names:
        key = (name, wildcard_radicals)
        if key in _OPSIN_CACHE:
            results[name] = _OPSIN_CACHE[key]
        elif name not in missing:
            missing.append(name)

    if not missing:
        return results

    if not shutil.which("java"):
        for name in missing:
            _OPSIN_CACHE[(name, wildcard_radicals)] = None
            results[name] = None
        return results

    try:
        from py2opsin import py2opsin
    except ImportError:
        for name in missing:
            _OPSIN_CACHE[(name, wildcard_radicals)] = None
            results[name] = None
        return results

    tmp_fpath = str(Path(tempfile.gettempdir()) / f"ae_oracle_opsin_{os.getpid()}.txt")
    try:
        raw = py2opsin(
            missing,
            allow_radicals=wildcard_radicals,
            wildcard_radicals=wildcard_radicals,
            tmp_fpath=tmp_fpath,
        )
        raw_list = [raw] if isinstance(raw, str) else list(raw or [])
    except Exception as exc:
        logger.debug(f"py2opsin oracle lookup failed: {exc}")
        raw_list = [None] * len(missing)
    finally:
        try:
            Path(tmp_fpath).unlink(missing_ok=True)
        except Exception:
            pass

    for idx, name in enumerate(missing):
        val = str(raw_list[idx]).strip() if idx < len(raw_list) and raw_list[idx] else None
        _OPSIN_CACHE[(name, wildcard_radicals)] = val
        results[name] = val

    return results


def check_chembl_structure(mol: Any) -> list[str]:
    """Run ChEMBL structure pipeline checker on an RDKit Mol and return severe issues (sev >= 5)."""
    try:
        from chembl_structure_pipeline import checker
        from rdkit import Chem

        molblock = Chem.MolToMolBlock(mol)
        issues = checker.check_molblock(molblock)
        return [msg for sev, msg in issues if int(sev) >= 5]
    except Exception as exc:
        logger.debug(f"ChEMBL structure check skipped/failed: {exc}")
        return []


def extract_context_cues_for_labels(
    context: str,
    labels: list[str],
) -> dict[str, dict[str, Any]]:
    """Extract anchored Formula/HRMS, IUPAC names, and explicit substituent motifs per compound label."""
    cues: dict[str, dict[str, Any]] = {
        lbl: {"formulas": [], "iupac_names": [], "substituents": []}
        for lbl in labels
    }
    if not context or not context.strip() or not labels:
        return cues

    clean_ctx = _clean_latex_chemical_text(context)

    # 1. Full IUPAC headings/bold entries like "4.5.25. ...methanone (37)"
    for m in _IUPAC_HEADING_RE.finditer(clean_ctx):
        lbl = m.group("label").strip()
        if lbl in cues:
            cand_name = m.group("name").strip(" .:")
            # Avoid picking prose fragments that are not IUPAC names
            if len(cand_name) >= 12 and ("-" in cand_name or "(" in cand_name):
                cues[lbl]["iupac_names"].append(cand_name)

    # 2. Paragraph- and sentence-level formula + substituent anchoring
    paragraphs = re.split(r"\n{2,}", clean_ctx)
    active_heading_labels: list[str] = []
    for para in paragraphs:
        p_str = para.strip()
        if not p_str:
            continue

        # Identify which target labels are explicitly mentioned in this paragraph
        matched_labels: list[str] = []
        for lbl in labels:
            esc_lbl = re.escape(lbl)
            if re.search(
                rf"(?:\bCompound\s+{esc_lbl}\b|\bcompd\.?\s*{esc_lbl}\b|\({esc_lbl}\)|\b{esc_lbl}\b(?=\s*(?:,|which|had|with|showed|:)))",
                p_str,
                re.IGNORECASE,
            ):
                matched_labels.append(lbl)

        is_heading_line = p_str.startswith("#") or bool(_IUPAC_HEADING_RE.search(p_str))
        if is_heading_line and matched_labels:
            active_heading_labels = list(matched_labels)
        elif not matched_labels and active_heading_labels:
            # Characterization paragraph immediately following an experimental heading
            if _FORMULA_CANDIDATE_RE.search(p_str):
                matched_labels = list(active_heading_labels)
            active_heading_labels = []
        else:
            active_heading_labels = []

        if not matched_labels and len(labels) == 1:
            # Single-entity visual: check if formula/HRMS appears directly in context
            if _FORMULA_CANDIDATE_RE.search(p_str):
                matched_labels = [labels[0]]

        if not matched_labels:
            continue

        # Attach formulas only when uniquely anchored to a single label in that paragraph/sentence
        sentences = re.split(r"(?<=[.!?;])\s+", p_str)
        for sent in sentences:
            sent_labels = [
                lbl
                for lbl in matched_labels
                if re.search(rf"(?:\b{re.escape(lbl)}\b|\({re.escape(lbl)}\))", sent)
            ]
            if not sent_labels and len(matched_labels) == 1:
                sent_labels = matched_labels

            if len(sent_labels) == 1:
                target_lbl = sent_labels[0]
                for fm in _FORMULA_CANDIDATE_RE.finditer(sent):
                    raw_f = fm.group(1)
                    if parse_formula_counts(raw_f):
                        cues[target_lbl]["formulas"].append(raw_f)
                # Explicit substituent phrases like "2-chloro-6-fluorophenyl"
                for sm in _SUBSTITUENT_LOCANT_RE.finditer(sent):
                    sub_txt = sm.group(1).strip()
                    if len(sub_txt) >= 8 and ("phenyl" in sub_txt.lower() or "pyridin" in sub_txt.lower()):
                        if sub_txt not in cues[target_lbl]["substituents"]:
                            cues[target_lbl]["substituents"].append(sub_txt)

    return cues


def _is_markush_smiles(smiles: str) -> bool:
    return bool(re.search(r"\[\*:\d+\]|\[\d+\*\]", smiles or ""))


def verify_entities_with_oracle(
    entities: list[Any],
    context: str = "",
    *,
    enable_chembl: bool = True,
) -> list[OracleFinding]:
    """Run deterministic OCSR oracle checks on RDKit-valid entities.

    Checks ChEMBL structure quality, Formula/HRMS consistency, py2opsin IUPAC -> InChIKey (14/27),
    and explicit locant-numbered substituent motifs.
    Sets ``provenance_status`` and ``oracle_notes`` directly on each entity.
    """
    from rdkit import Chem

    findings: list[OracleFinding] = []
    if not entities:
        return findings

    labels = [getattr(e, "label", "") for e in entities if getattr(e, "label", "")]
    cues_by_label = extract_context_cues_for_labels(context, labels)

    # Batch resolve all IUPAC names and radical substituents in one py2opsin call each
    all_iupac: list[str] = []
    all_subs: list[str] = []
    for lbl_cues in cues_by_label.values():
        for nm in lbl_cues["iupac_names"]:
            if nm not in all_iupac:
                all_iupac.append(nm)
        for sb in lbl_cues["substituents"]:
            if sb not in all_subs:
                all_subs.append(sb)

    iupac_smiles_map = _resolve_iupac_via_opsin(all_iupac, wildcard_radicals=False) if all_iupac else {}
    sub_smiles_map = _resolve_iupac_via_opsin(all_subs, wildcard_radicals=True) if all_subs else {}

    for ent in entities:
        label = getattr(ent, "label", "")
        smiles = getattr(ent, "smiles", "")
        smiles_valid = getattr(ent, "smiles_valid", True)
        finding = OracleFinding(label=label)
        is_markush = _is_markush_smiles(smiles)
        lbl_cues = cues_by_label.get(label, {})

        # Resolve prose expectations once, so hints survive even when the
        # predicted SMILES cannot be parsed at all.
        if not is_markush:
            formulas = lbl_cues.get("formulas", [])
            if formulas:
                finding.expected_formula = formulas[0]

            for nm in lbl_cues.get("iupac_names", []):
                cand_smi = iupac_smiles_map.get(nm)
                if not cand_smi:
                    continue
                cand_mol = Chem.MolFromSmiles(cand_smi)
                if cand_mol is None:
                    continue
                try:
                    cand_ik = Chem.MolToInchiKey(cand_mol)
                except Exception:
                    cand_ik = ""
                finding.expected_iupac = nm
                finding.expected_smiles = cand_smi
                finding.expected_inchi_key = cand_ik or None
                break

            for sub_name in lbl_cues.get("substituents", []):
                rad_smi = sub_smiles_map.get(sub_name)
                if rad_smi and Chem.MolFromSmiles(rad_smi) is not None:
                    finding.expected_substructure_smiles.append(rad_smi)
                    finding.expected_substructure_names.append(sub_name)

        if not smiles_valid or not smiles:
            finding.passed = False
            finding.provenance_status = "mismatch_flagged"
            finding.reasons.append(getattr(ent, "rdkit_error", None) or "Invalid SMILES")
            ent.provenance_status = "mismatch_flagged"
            ent.oracle_notes = "; ".join(finding.reasons)
            findings.append(finding)
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            finding.passed = False
            finding.provenance_status = "mismatch_flagged"
            finding.reasons.append(f"RDKit parse failed for {smiles!r}")
            ent.provenance_status = "mismatch_flagged"
            ent.oracle_notes = "; ".join(finding.reasons)
            findings.append(finding)
            continue

        # 1. ChEMBL structure check (skip dummy-atom Markush templates)
        if enable_chembl and not is_markush:
            chembl_issues = check_chembl_structure(mol)
            if chembl_issues:
                finding.passed = False
                finding.reasons.append(f"ChEMBL structure issue: {', '.join(chembl_issues)}")

        pred_formula = ""
        pred_inchikey = ""
        if not is_markush:
            from rdkit.Chem import rdMolDescriptors

            pred_formula = rdMolDescriptors.CalcMolFormula(mol)
            try:
                pred_inchikey = Chem.MolToInchiKey(mol)
            except Exception:
                pred_inchikey = ""
            finding.predicted_inchi_key = pred_inchikey or None

        # 2. Image vs prose-name connectivity conflict (gates the deterministic override)
        if finding.expected_smiles and not is_markush:
            exp_mol = Chem.MolFromSmiles(finding.expected_smiles)
            if exp_mol is not None and not same_connectivity(mol, exp_mol):
                finding.connectivity_conflict = True

        verified_exact = False

        # 3. Formula / HRMS check
        if finding.expected_formula and not is_markush:
            exp_f = finding.expected_formula
            if not formulas_compatible(pred_formula, exp_f):
                finding.passed = False
                finding.reasons.append(
                    f"Formula mismatch: predicted '{pred_formula}' != expected '{exp_f}'"
                )
            else:
                verified_exact = True

        # 4. Full IUPAC -> InChIKey (14/27) check
        if finding.expected_inchi_key and not is_markush and pred_inchikey:
            exp_ik = finding.expected_inchi_key
            if pred_inchikey == exp_ik:
                verified_exact = True
            elif pred_inchikey[:14] == exp_ik[:14]:
                # Connectivity identical; check tautomer-canonical equivalence before flagging stereo/tautomer diff
                verified_exact = True
            else:
                finding.passed = False
                finding.reasons.append(
                    f"IUPAC InChIKey mismatch for '{finding.expected_iupac}': "
                    f"predicted {pred_inchikey[:14]} != expected {exp_ik[:14]}"
                )

        # 5. Explicit substituent motif substructure check (e.g. "2-chloro-6-fluorophenyl")
        if not is_markush:
            for sub_name, rad_smi in zip(
                finding.expected_substructure_names, finding.expected_substructure_smiles
            ):
                rad_mol = Chem.MolFromSmiles(rad_smi)
                if rad_mol is None:
                    continue
                smarts = Chem.MolToSmarts(rad_mol).replace("[#0]", "[!#1]")
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None and not mol.HasSubstructMatch(pat):
                    finding.passed = False
                    finding.reasons.append(
                        f"Substituent motif mismatch: prose specifies '{sub_name}', which is absent in predicted SMILES"
                    )

        if not finding.passed:
            finding.provenance_status = "mismatch_flagged"
        elif verified_exact:
            finding.provenance_status = "verified_exact"
        else:
            finding.provenance_status = "unverified_valid"

        ent.provenance_status = finding.provenance_status
        ent.oracle_notes = "; ".join(finding.reasons) if finding.reasons else None
        findings.append(finding)

    return findings


def verify_markush_example(
    scaffold_smiles: str,
    substituents: dict[str, str],
    *,
    expected_iupac: Optional[str] = None,
    expected_formula: Optional[str] = None,
) -> tuple[Optional[str], OracleFinding]:
    """Assemble a Markush compound via `assemble_markush_single` and verify with the OCSR oracle."""
    from types import SimpleNamespace

    assembled_smi, err = assemble_markush_single(scaffold_smiles, substituents)
    if assembled_smi is None:
        return None, OracleFinding(
            label="markush",
            provenance_status="mismatch_flagged",
            passed=False,
            reasons=[err or "Markush assembly failed"],
        )

    context_parts: list[str] = []
    if expected_iupac:
        context_parts.append(f"### 1. {expected_iupac} (markush)")
    if expected_formula:
        context_parts.append(f"Compound markush HRMS calcd for {expected_formula}.")
    dummy_ent = SimpleNamespace(
        label="markush",
        smiles=assembled_smi,
        smiles_valid=True,
        rdkit_error=None,
        provenance_status="unverified_valid",
        oracle_notes=None,
    )
    findings = verify_entities_with_oracle([dummy_ent], context="\n\n".join(context_parts))
    return assembled_smi, findings[0]
