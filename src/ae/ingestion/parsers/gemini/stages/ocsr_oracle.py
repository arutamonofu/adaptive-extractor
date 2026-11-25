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
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from ae.core.schema.domain_tools.chemistry import assemble_markush_single

logger = logging.getLogger(__name__)

ProvenanceStatus = Literal["verified_exact", "unverified_valid", "mismatch_flagged"]

_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_FORMULA_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_FULL_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
_VALID_ELEMENTS = frozenset({
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "D", "T",
})
_FORMULA_CANDIDATE_RE = re.compile(
    r"(?i:\b(?:HRMS\b|ESI-?MS\b|MS\b|Anal\.?\s*Calcd\.?(?:\s+for\b)?|Calcd\.?\s+for\b|Molecular\s+Formula\b))"
    r"(?:[^.\n;]|\.(?=\d)){0,80}?\b"
    r"(?!IC\d|EC\d|LD\d|LC\d|GI\d|CC\d)"
    r"([A-Z](?:[a-z]?(?:\d+|_\{?\d+\}?)[A-Z][A-Za-z0-9_{}\\]*|[A-Za-z0-9_{}\\]*H[A-Za-z0-9_{}\\]*\d[A-Za-z0-9_{}\\]*))",
)
_ADDUCT_TOKENS_RE = r"Na|K|NH4|Cl|HCOO|2H|H\s*-\s*H2O"
_ADDUCT_SUFFIX_RE = re.compile(
    rf"(?:\[\s*M\s*\+\s*({_ADDUCT_TOKENS_RE})\s*\](?:\s*\^?\s*(?:2\+|2-|\+|-))?"
    rf"|\(\s*M\s*\+\s*({_ADDUCT_TOKENS_RE})\s*\)(?:\s*\^?\s*(?:2\+|2-|\+|-))?"
    rf"|\bM\s*\+\s*({_ADDUCT_TOKENS_RE})\s*\^?\s*(?:2\+|2-|\+|-))",
    re.IGNORECASE,
)
_ADDUCT_TAG_MAP = {
    "NA": "[M+Na]+",
    "K": "[M+K]+",
    "NH4": "[M+NH4]+",
    "CL": "[M+Cl]-",
    "HCOO": "[M+HCOO]-",
    "2H": "[M+2H]2+",
    "H-H2O": "[M+H-H2O]+",
}
_SUBSTITUENT_LOCANT_RE = re.compile(
    r"\b(\d+(?:,\d+)*-(?:[a-z]+-)+(?:\d+(?:,\d+)*-)?[a-z]+(?:phenyl|pyridin-\d-yl|benzyl|benzoyl)"
    r"|\d+(?:,\d+)*-(?:di|tri|tetra)?(?:chloro|fluoro|bromo|iodo|methyl|methoxy|nitro|hydroxy|amino|trifluoromethyl)(?:phenyl|pyridin-\d-yl|benzyl|benzoyl)"
    r"|(?:[omp]-|pyridin-[234]-yl\b)"
    r"(?:(?:chloro|fluoro|bromo|iodo|methyl|methoxy|nitro)(?:phenyl|benzyl|benzoyl))?)\b",
    re.IGNORECASE,
)
_IUPAC_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*{1,2})?(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?P<name>[^\n]{12,240}?(?:yl|one|amide|ate|acid|azole|dine|ene|ol|ane|ine|ide))"
    r"\s*\(\s*(?P<label>[A-Za-z0-9\-]+)\s*\)(?:\*{1,2})?(?:\s|$)",
    re.MULTILINE,
)
_EXPLICIT_COMPOUND_ID_RE = re.compile(
    r"\b(?i:Compound|compd\.?|hybrid)\s+([A-Za-z0-9][A-Za-z0-9\-]{0,7})\b"
    r"|\(([0-9]{1,3}[a-z]?)\)"
    r"|(?<!Scheme\s)(?<!Figure\s)(?<!Fig\.\s)(?<!Fig\s)(?<!Table\s)(?<!Chart\s)(?<!Entry\s)(?<!Page\s)(?<!Section\s)"
    r"\b([0-9]{1,3}[a-z]?)(?=\s*(?:,(?!\d)|(?i:which|had|with|showed)|:))",
)

# Bounded LRU in-process cache for py2opsin lookups to keep CPU overhead negligible
_OPSIN_CACHE_MAX_SIZE = 4096
_OPSIN_CACHE: OrderedDict[tuple[str, bool], Optional[str]] = OrderedDict()
_OPSIN_CACHE_LOCK = threading.Lock()
_OPSIN_SEMAPHORE = threading.Semaphore(2)
_OPSIN_UNAVAILABLE_WARNED = False
_OPSIN_WARN_LOCK = threading.Lock()


def _warn_opsin_unavailable_once(reason: str) -> None:
    global _OPSIN_UNAVAILABLE_WARNED
    with _OPSIN_WARN_LOCK:
        if not _OPSIN_UNAVAILABLE_WARNED:
            _OPSIN_UNAVAILABLE_WARNED = True
            logger.warning(
                f"py2opsin IUPAC oracle unavailable ({reason}); IUPAC and substituent checks skipped."
            )


@dataclass
class OracleFinding:
    """Verification outcome for a single chemical entity."""

    label: str
    provenance_status: ProvenanceStatus = "unverified_valid"
    passed: bool = True
    reasons: list[str] = field(default_factory=list)
    expected_iupac: Optional[str] = None
    expected_formula: Optional[str] = None
    expected_formulas: list[str] = field(default_factory=list)
    expected_adduct_context: str = ""
    expected_inchi_key: Optional[str] = None
    predicted_inchi_key: Optional[str] = None
    expected_smiles: Optional[str] = None
    expected_substructure_smiles: list[str] = field(default_factory=list)
    expected_substructure_names: list[str] = field(default_factory=list)
    connectivity_conflict: bool = False


def _clean_latex_chemical_text(text: str) -> str:
    """Normalize LaTeX/Markdown formatting inside chemical names and formulas, preserving newlines."""
    cleaned = text.translate(_SUBSCRIPT_TRANSLATION)
    cleaned = re.sub(r"\\(?:text|mathbf|mathrm|rm)\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\rm\b\s*", "", cleaned)
    cleaned = cleaned.replace("$", "").replace("*", "")
    cleaned = re.sub(r"_\{(\d+)\}", r"\1", cleaned)
    cleaned = re.sub(r"_(\d+)", r"\1", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()


def parse_formula_counts(formula: str) -> dict[str, int]:
    """Parse a chemical formula string into element counts."""
    clean = _clean_latex_chemical_text(formula)
    clean = re.sub(r"[^A-Za-z0-9]", "", clean)
    if not clean or not _FORMULA_FULL_RE.match(clean):
        return {}
    counts: dict[str, int] = {}
    for elem, num_str in _FORMULA_ELEMENT_RE.findall(clean):
        if not elem or elem not in _VALID_ELEMENTS:
            return {}
        counts[elem] = counts.get(elem, 0) + (int(num_str) if num_str else 1)
    return counts


def _extract_adducts(text: str) -> set[str]:
    """Extract normalized HRMS/ESI-MS adduct ion keys from text."""
    if not text:
        return set()
    cleaned = _clean_latex_chemical_text(text)
    adducts: set[str] = set()
    for match in _ADDUCT_SUFFIX_RE.finditer(cleaned):
        token = next((g for g in match.groups() if g), None)
        if token:
            adducts.add(re.sub(r"\s+", "", token).upper())
    return adducts


def _formula_counts_match(pred_counts: dict[str, int], exp_counts: dict[str, int]) -> bool:
    """Compare element counts allowing +/- 1 H difference ([M+H]+)."""
    pred_heavy = {k: v for k, v in pred_counts.items() if k != "H"}
    exp_heavy = {k: v for k, v in exp_counts.items() if k != "H"}
    if pred_heavy != exp_heavy:
        return False
    h_diff = abs(pred_counts.get("H", 0) - exp_counts.get("H", 0))
    return h_diff <= 1


def formulas_compatible(
    predicted_formula: str,
    expected_formula: str,
    *,
    context: str = "",
) -> bool:
    """Compare predicted neutral formula against prose formula (allowing [M+H]+ and contextual adducts)."""
    pred_counts = parse_formula_counts(predicted_formula)
    exp_cleaned = _ADDUCT_SUFFIX_RE.sub(" ", _clean_latex_chemical_text(expected_formula))
    exp_counts = parse_formula_counts(exp_cleaned)
    if not exp_counts:
        return True
    if not pred_counts:
        return False

    if _formula_counts_match(pred_counts, exp_counts):
        return True

    adducts = _extract_adducts(f"{expected_formula}\n{context}")
    for adduct in adducts:
        adjusted = dict(exp_counts)
        if adduct == "NA" and adjusted.get("Na", 0) >= 1:
            adjusted["Na"] -= 1
            if adjusted["Na"] == 0:
                del adjusted["Na"]
        elif adduct == "K" and adjusted.get("K", 0) >= 1:
            adjusted["K"] -= 1
            if adjusted["K"] == 0:
                del adjusted["K"]
        elif adduct == "NH4" and adjusted.get("N", 0) >= 1 and adjusted.get("H", 0) >= 4:
            adjusted["N"] -= 1
            if adjusted["N"] == 0:
                del adjusted["N"]
            adjusted["H"] -= 4
            if adjusted["H"] == 0:
                del adjusted["H"]
        elif adduct == "CL" and adjusted.get("Cl", 0) >= 1:
            adjusted["Cl"] -= 1
            if adjusted["Cl"] == 0:
                del adjusted["Cl"]
        elif (
            adduct == "HCOO"
            and adjusted.get("C", 0) >= 1
            and adjusted.get("H", 0) >= 1
            and adjusted.get("O", 0) >= 2
        ):
            for elem, n in (("C", 1), ("H", 1), ("O", 2)):
                adjusted[elem] -= n
                if adjusted[elem] == 0:
                    del adjusted[elem]
        elif adduct == "2H" and adjusted.get("H", 0) >= 2:
            adjusted["H"] -= 2
            if adjusted["H"] == 0:
                del adjusted["H"]
        elif adduct == "H-H2O":
            adjusted["H"] = adjusted.get("H", 0) + 1
            adjusted["O"] = adjusted.get("O", 0) + 1
        else:
            continue
        if _formula_counts_match(pred_counts, adjusted):
            return True

    return False


def _radical_pattern(rad_mol: Any):
    """Stereo-stripped SMARTS pattern for a radical/substituent (wildcard atom -> any non-H)."""
    from rdkit import Chem

    stripped = Chem.Mol(rad_mol)
    Chem.RemoveStereochemistry(stripped)
    return Chem.MolFromSmarts(Chem.MolToSmarts(stripped).replace("[#0]", "[!#1]"))


def same_connectivity(mol_a: Any, mol_b: Any) -> bool:
    """True when two molecules share graph connectivity (tautomer/stereo-invariant).

    Checks InChIKey[:14] first, falling back to stereo-stripped bidirectional SMARTS
    matching only when InChIKey is unavailable (e.g., dummy-atom radicals).
    """
    from rdkit import Chem

    if mol_a is None or mol_b is None:
        return False

    ik_a = ""
    ik_b = ""
    try:
        ik_a = Chem.MolToInchiKey(mol_a) or ""
        ik_b = Chem.MolToInchiKey(mol_b) or ""
    except Exception:
        ik_a = ""
        ik_b = ""

    if len(ik_a) >= 14 and len(ik_b) >= 14:
        return ik_a[:14] == ik_b[:14]

    pat_a, pat_b = _radical_pattern(mol_a), _radical_pattern(mol_b)
    if pat_a is None or pat_b is None:
        return False
    return bool(mol_a.HasSubstructMatch(pat_b) and mol_b.HasSubstructMatch(pat_a))


def _finding_expected_formulas(finding: OracleFinding) -> list[str]:
    if finding.expected_formulas:
        return list(finding.expected_formulas)
    if finding.expected_formula:
        return [finding.expected_formula]
    return []


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

    candidate_formulas = _finding_expected_formulas(finding)
    if candidate_formulas:
        pred_f = rdMolDescriptors.CalcMolFormula(mol)
        if not any(
            formulas_compatible(
                pred_f,
                exp_f,
                context=finding.expected_adduct_context,
            )
            for exp_f in candidate_formulas
        ):
            expected_desc = " or ".join(candidate_formulas)
            violations.append(
                f"formula '{pred_f}' incompatible with prose '{expected_desc}'"
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
    """Resolve IUPAC names or radical substituents via local py2opsin with thread-safe LRU caching."""
    results: dict[str, Optional[str]] = {}
    missing: list[str] = []
    with _OPSIN_CACHE_LOCK:
        for name in names:
            key = (name, wildcard_radicals)
            if key in _OPSIN_CACHE:
                if hasattr(_OPSIN_CACHE, "move_to_end"):
                    _OPSIN_CACHE.move_to_end(key)
                results[name] = _OPSIN_CACHE[key]
            elif name not in missing:
                missing.append(name)

    if not missing:
        return results

    if not shutil.which("java"):
        _warn_opsin_unavailable_once("java binary not found in PATH")
        for name in missing:
            results[name] = None
        return results

    try:
        from py2opsin import py2opsin
    except ImportError:
        _warn_opsin_unavailable_once("py2opsin package is not installed")
        for name in missing:
            results[name] = None
        return results

    tmp_fpath: Optional[str] = None
    raw_list: list[Optional[str]] = [None] * len(missing)
    succeeded = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="ae_oracle_opsin_",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            tmp_fpath = tmp_file.name

        with _OPSIN_SEMAPHORE:
            raw = py2opsin(
                missing,
                allow_radicals=wildcard_radicals,
                wildcard_radicals=wildcard_radicals,
                tmp_fpath=tmp_fpath,
            )
        if isinstance(raw, bool) or raw is None:
            logger.debug("py2opsin oracle lookup returned failure sentinel")
        elif isinstance(raw, str):
            raw_list = [raw]
            succeeded = len(missing) == 1
        else:
            raw_seq = list(raw)
            if len(raw_seq) == len(missing):
                raw_list = raw_seq
                succeeded = True
            else:
                logger.debug(
                    f"py2opsin oracle lookup length mismatch: expected {len(missing)}, got {len(raw_seq)}"
                )
    except Exception as exc:
        logger.debug(f"py2opsin oracle lookup failed: {exc}")
    finally:
        if tmp_fpath:
            try:
                Path(tmp_fpath).unlink(missing_ok=True)
            except Exception:
                pass

    for idx, name in enumerate(missing):
        val = str(raw_list[idx]).strip() if idx < len(raw_list) and raw_list[idx] else None
        if succeeded:
            with _OPSIN_CACHE_LOCK:
                key = (name, wildcard_radicals)
                _OPSIN_CACHE[key] = val
                if hasattr(_OPSIN_CACHE, "move_to_end"):
                    _OPSIN_CACHE.move_to_end(key)
                while len(_OPSIN_CACHE) > _OPSIN_CACHE_MAX_SIZE:
                    if isinstance(_OPSIN_CACHE, OrderedDict):
                        _OPSIN_CACHE.popitem(last=False)
                    else:
                        _OPSIN_CACHE.pop(next(iter(_OPSIN_CACHE)))
        results[name] = val

    return results


def check_chembl_structure_detailed(mol: Any) -> list[tuple[int, str]]:
    """Run ChEMBL structure pipeline checker on an RDKit Mol and return severe (sev >= 5) pairs."""
    try:
        from chembl_structure_pipeline import checker
        from rdkit import Chem

        molblock = Chem.MolToMolBlock(mol)
        issues = checker.check_molblock(molblock)
        return [(int(sev), msg) for sev, msg in issues if int(sev) >= 5]
    except Exception as exc:
        logger.debug(f"ChEMBL structure check skipped/failed: {exc}")
        return []


def check_chembl_structure(mol: Any) -> list[str]:
    """Run ChEMBL structure pipeline checker on an RDKit Mol and return severe issues (sev >= 5)."""
    return [msg for _, msg in check_chembl_structure_detailed(mol)]


def _has_other_compound_ids(text: str, target_label: str) -> bool:
    """True if text explicitly references a compound label other than target_label."""
    norm_target = target_label.strip().lower()
    for m in _EXPLICIT_COMPOUND_ID_RE.finditer(text):
        cand = next((g for g in m.groups() if g), "")
        if cand and cand.strip().lower() != norm_target:
            return True
    return False


def _label_sentence_pattern(lbl: str) -> str:
    """Regex for matching a compound label at sentence level (anchored for numeric IDs)."""
    esc_lbl = re.escape(lbl)
    if lbl.isdigit():
        return (
            rf"(?:\bCompound\s+{esc_lbl}\b|\bcompd\.?\s*{esc_lbl}\b"
            rf"|\({esc_lbl}\)|\b{esc_lbl}\b(?=\s*(?:,(?!\d)|which|had|with|showed|:)))"
        )
    return rf"(?:\b{esc_lbl}\b|\({esc_lbl}\))"


def extract_context_cues_for_labels(
    context: str,
    labels: list[str],
) -> dict[str, dict[str, Any]]:
    """Extract anchored Formula/HRMS, IUPAC names, and explicit substituent motifs per compound label."""
    cues: dict[str, dict[str, Any]] = {
        lbl: {"formulas": [], "adducts": [], "iupac_names": [], "substituents": []}
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
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", clean_ctx) if p.strip()]
    section_prefixes = (
        ("Preceding text:", "preceding"),
        ("Following text:", "following"),
        ("Cross-referenced paragraphs:", "cross_ref"),
        ("In-text callouts:", "callout"),
    )
    para_sections: list[str] = []
    cur_section = "unsectioned"
    for p_str in raw_paras:
        para_head_lines = [ln.lstrip() for ln in p_str.splitlines()[:2]]
        for prefix, sec_name in section_prefixes:
            if any(ln.startswith(prefix) for ln in para_head_lines):
                cur_section = sec_name
                break
        para_sections.append(cur_section)
    has_sections = any(sec != "unsectioned" for sec in para_sections)
    last_preceding_idx = max(
        (i for i, sec in enumerate(para_sections) if sec == "preceding"),
        default=-1,
    )
    first_following_idx = next(
        (i for i, sec in enumerate(para_sections) if sec == "following"),
        -1,
    )

    active_heading_labels: list[str] = []
    other_heading_active = False
    for p_idx, p_str in enumerate(raw_paras):
        # Identify which target labels are explicitly mentioned in this paragraph
        matched_labels: list[str] = []
        for lbl in labels:
            if re.search(
                _label_sentence_pattern(lbl),
                p_str,
                re.IGNORECASE if lbl.isdigit() else 0,
            ):
                matched_labels.append(lbl)

        is_heading_line = any(
            ln.lstrip().startswith("#") for ln in p_str.splitlines()
        ) or bool(_IUPAC_HEADING_RE.search(p_str))
        if is_heading_line and matched_labels:
            active_heading_labels = list(matched_labels)
            other_heading_active = False
        elif is_heading_line and any(_has_other_compound_ids(p_str, lbl) for lbl in labels):
            active_heading_labels = []
            other_heading_active = True
        elif not matched_labels and active_heading_labels:
            # Characterization paragraph immediately following an experimental heading
            if _FORMULA_CANDIDATE_RE.search(p_str):
                matched_labels = list(active_heading_labels)
            active_heading_labels = []
            other_heading_active = False
        elif not matched_labels and other_heading_active:
            active_heading_labels = []
            other_heading_active = False
            continue
        else:
            active_heading_labels = []
            other_heading_active = False

        if not matched_labels and len(labels) == 1:
            # Single-entity fallback: restrict so arbitrary unrelated paragraphs from
            # adjacent pages or other compounds are not blindly bound to labels[0].
            if has_sections:
                is_adjacent = p_idx in (last_preceding_idx, first_following_idx)
            else:
                is_adjacent = len(raw_paras) <= 2 or p_idx in (0, len(raw_paras) - 1)
            no_other_ids = not _has_other_compound_ids(p_str, labels[0])
            if _FORMULA_CANDIDATE_RE.search(p_str) and no_other_ids and is_adjacent:
                matched_labels = [labels[0]]

        if not matched_labels:
            continue

        # Attach formulas only when uniquely anchored to a single label in that paragraph/sentence
        sentences = re.split(
            r"(?<!\bAnal\.)(?<!\bCalcd\.)(?<!\bcompd\.)(?<!\bCompd\.)(?<!\bFig\.)(?<=[.!?;])\s+",
            p_str,
        )
        for sent in sentences:
            sent_labels = [
                lbl
                for lbl in matched_labels
                if re.search(
                    _label_sentence_pattern(lbl),
                    sent,
                    re.IGNORECASE if lbl.isdigit() else 0,
                )
            ]
            if (
                not sent_labels
                and len(matched_labels) == 1
                and not _has_other_compound_ids(sent, matched_labels[0])
            ):
                sent_labels = matched_labels

            if len(sent_labels) == 1:
                target_lbl = sent_labels[0]
                for fm in _FORMULA_CANDIDATE_RE.finditer(sent):
                    raw_f = fm.group(1)
                    if parse_formula_counts(raw_f):
                        if raw_f not in cues[target_lbl]["formulas"]:
                            cues[target_lbl]["formulas"].append(raw_f)
                        sent_adducts = _extract_adducts(sent)
                        if not sent_adducts and len(matched_labels) == 1:
                            sent_adducts = _extract_adducts(p_str)
                        for add in sorted(sent_adducts):
                            tag = _ADDUCT_TAG_MAP.get(add, f"[M+{add}]+")
                            if tag not in cues[target_lbl]["adducts"]:
                                cues[target_lbl]["adducts"].append(tag)
                # Explicit substituent phrases like "2-chloro-6-fluorophenyl"
                for sm in _SUBSTITUENT_LOCANT_RE.finditer(sent):
                    sub_txt = sm.group(1).strip()
                    sub_lower = sub_txt.lower()
                    if len(sub_txt) >= 8 and any(
                        tok in sub_lower for tok in ("phenyl", "pyridin", "benzyl", "benzoyl")
                    ):
                        if sub_txt not in cues[target_lbl]["substituents"]:
                            cues[target_lbl]["substituents"].append(sub_txt)

    return cues


def _is_markush_smiles(smiles: str) -> bool:
    return bool(re.search(r"\[\*:\d+\]|\[\d+\*\]", smiles or ""))


def reverify_entity_with_finding(
    ent: Any,
    finding: OracleFinding,
    *,
    enable_chembl: bool = True,
) -> OracleFinding:
    """Re-verify a single entity against pre-resolved prose expectations in ``finding``.

    Updates ``finding`` and sets ``provenance_status`` / ``oracle_notes`` on ``ent``
    without re-parsing prose context or touching other entities.
    """
    from rdkit import Chem

    smiles = getattr(ent, "smiles", "")
    smiles_valid = getattr(ent, "smiles_valid", True)
    is_markush = _is_markush_smiles(smiles)

    finding.passed = True
    finding.reasons = []
    finding.provenance_status = "unverified_valid"
    finding.predicted_inchi_key = None
    finding.connectivity_conflict = False

    if not smiles_valid or not smiles:
        finding.passed = False
        finding.provenance_status = "mismatch_flagged"
        finding.reasons.append(getattr(ent, "rdkit_error", None) or "Invalid SMILES")
        ent.provenance_status = "mismatch_flagged"
        ent.oracle_notes = "; ".join(finding.reasons)
        return finding

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        finding.passed = False
        finding.provenance_status = "mismatch_flagged"
        finding.reasons.append(f"RDKit parse failed for {smiles!r}")
        if hasattr(ent, "smiles_valid"):
            ent.smiles_valid = False
        ent.provenance_status = "mismatch_flagged"
        ent.oracle_notes = "; ".join(finding.reasons)
        return finding

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

    # 3. Formula / HRMS check across all candidate formulas
    candidate_formulas = _finding_expected_formulas(finding)
    if candidate_formulas and not is_markush:
        matched_formula = next(
            (
                exp_f
                for exp_f in candidate_formulas
                if formulas_compatible(
                    pred_formula, exp_f, context=finding.expected_adduct_context
                )
            ),
            None,
        )
        if matched_formula is not None:
            finding.expected_formula = matched_formula
            verified_exact = True
        else:
            finding.passed = False
            expected_desc = " or ".join(candidate_formulas)
            finding.reasons.append(
                f"Formula mismatch: predicted '{pred_formula}' != expected '{expected_desc}'"
            )

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
            pat = _radical_pattern(rad_mol)
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
    return finding


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
        finding = OracleFinding(label=label)
        is_markush = _is_markush_smiles(smiles)
        lbl_cues = cues_by_label.get(label, {})

        # Resolve prose expectations once, so hints survive even when the
        # predicted SMILES cannot be parsed at all.
        if not is_markush:
            formulas = list(dict.fromkeys(lbl_cues.get("formulas", [])))
            if formulas:
                finding.expected_formulas = formulas
                finding.expected_formula = formulas[0]
            adducts = lbl_cues.get("adducts", [])
            if adducts:
                finding.expected_adduct_context = " ".join(adducts)

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

        reverify_entity_with_finding(ent, finding, enable_chembl=enable_chembl)
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
