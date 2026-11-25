"""Chemistry-domain extraction tools (lazy-loaded only when declared in schema)."""

from __future__ import annotations

import ast
from collections import OrderedDict
import json
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from ae.core.schema.tools import RegisteredToolProvider

_MARKUSH_DUMMY_RE = re.compile(r"\[\*:\d+\]|\[\d+\*\]")
_KEY_INT_RE = re.compile(r"(\d+)")
_HYDROGEN_TOKENS = frozenset({"h", "[h]", "*-h", "h-*"})

_MARKUSH_POLICY_HINT = (
    "TOOL `assemble_markush_smiles`:\n"
    "- The document contains unexpanded Markush scaffolds or R-group fragments with attachment points (`[*:1]`, `[*:2]`, ...).\n"
    "- NEVER mentally concatenate or manually rewrite SMILES strings with `[*:n]` placeholders.\n"
    "- Gather the complete core `scaffold_smiles` and each compound's substituent SMILES map (e.g. `{\"1\": \"C[*:1]\"}`), and call `assemble_markush_smiles` once per series in batch.\n"
    "- CROSS-CHECK VLM BLOCKS AGAINST PROSE/IUPAC: Always cross-check substituent positions, positional isomers, and scaffolds parsed in `<visual>` tables/schemes against explicit IUPAC names and compound descriptions in the document text/experimental section. If a `<visual>` block contradicts unambiguous chemical nomenclature in the prose, the text nomenclature takes precedence.\n"
    "- NO DUMMY CALLS: Never invoke `assemble_markush_smiles` with placeholder/test `compound_id` values (such as `\"test\"`); only call it with real `compound_id`s from the document."
)


def _parse_attachment_map_num(raw_key: Any) -> Optional[int]:
    """Extract numeric attachment index from keys like '1', '[*:1]', '1*', or 'R1'."""
    if isinstance(raw_key, int):
        return raw_key
    m = _KEY_INT_RE.search(str(raw_key or "").strip())
    return int(m.group(1)) if m else None


def _is_hydrogen_substituent(sub_smi: str, map_num: int) -> bool:
    """Return True if substituent represents explicit hydrogen capping."""
    cleaned = sub_smi.strip().lower().replace(" ", "")
    if cleaned in _HYDROGEN_TOKENS:
        return True
    if cleaned in (f"[h][*:{map_num}]", f"[*:{map_num}][h]", f"[h][{map_num}*]", f"[{map_num}*][h]"):
        return True
    return False


def _find_dummy_atoms(
    mol: Any,
    map_num: int,
    *,
    allow_single_unmapped_fallback: bool = True,
) -> List[Any]:
    """Locate dummy atoms matching atom-map or isotope number, with optional unmapped fallback."""
    matched: List[Any] = []
    all_dummies: List[Any] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            all_dummies.append(atom)
            if atom.GetAtomMapNum() == map_num or atom.GetIsotope() == map_num:
                matched.append(atom)
    if matched:
        return matched
    if (
        allow_single_unmapped_fallback
        and len(all_dummies) == 1
        and all_dummies[0].GetAtomMapNum() == 0
        and all_dummies[0].GetIsotope() == 0
    ):
        return all_dummies
    return []


def assemble_markush_single(
    scaffold_smiles: str,
    substituents: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    """Deterministically substitute R-group SMILES fragments into a Markush scaffold via RDKit.

    Args:
        scaffold_smiles: Core scaffold SMILES containing attachment points such as ``[*:1]``.
        substituents: Mapping from attachment index (e.g. ``"1"`` or ``"[*:1]"``) to substituent
            fragment SMILES (e.g. ``"C[*:1]"`` or ``"H"``).

    Returns:
        Tuple of ``(canonical_smiles, error_message)``.
    """
    if not scaffold_smiles or not isinstance(scaffold_smiles, str) or not scaffold_smiles.strip():
        return None, "scaffold_smiles must be a non-empty string"

    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        return None, (
            f"RDKit is required for Markush assembly: {exc}"
        )

    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")

    scaf_mol = Chem.MolFromSmiles(scaffold_smiles.strip())
    if scaf_mol is None:
        return None, f"Invalid scaffold_smiles: {scaffold_smiles!r}"

    rw = Chem.RWMol(scaf_mol)
    allow_scaf_unmapped = len(substituents or {}) == 1

    try:
        for raw_key, raw_sub in (substituents or {}).items():
            map_num = _parse_attachment_map_num(raw_key)
            if map_num is None:
                return None, f"Could not parse attachment index from key {raw_key!r}"
            if not isinstance(raw_sub, str) or not raw_sub.strip():
                return None, f"Empty substituent SMILES for attachment point {map_num}"

            sub_str = raw_sub.strip()

            while True:
                scaf_dummies = _find_dummy_atoms(
                    rw, map_num, allow_single_unmapped_fallback=allow_scaf_unmapped
                )
                if not scaf_dummies:
                    break
                scaf_dummy = scaf_dummies[0]
                neighbors = list(scaf_dummy.GetNeighbors())
                if len(neighbors) != 1:
                    return None, (
                        f"Attachment point [*:{map_num}] in scaffold must have degree 1 "
                        f"(got {len(neighbors)})"
                    )
                scaf_nbr = neighbors[0]
                scaf_bond = rw.GetBondBetweenAtoms(scaf_dummy.GetIdx(), scaf_nbr.GetIdx()).GetBondType()

                if _is_hydrogen_substituent(sub_str, map_num):
                    rw.RemoveAtom(scaf_dummy.GetIdx())
                    continue

                sub_mol = Chem.MolFromSmiles(sub_str)
                if sub_mol is None:
                    return None, f"Invalid substituent SMILES {sub_str!r} for [*:{map_num}]"

                sub_dummies = _find_dummy_atoms(
                    sub_mol, map_num, allow_single_unmapped_fallback=True
                )
                if len(sub_dummies) != 1:
                    return None, (
                        f"Substituent SMILES {sub_str!r} must contain exactly 1 attachment "
                        f"dummy atom for [*:{map_num}] (found {len(sub_dummies)})"
                    )
                sub_dummy = sub_dummies[0]
                sub_nbrs = list(sub_dummy.GetNeighbors())
                if len(sub_nbrs) != 1:
                    return None, (
                        f"Substituent attachment dummy in {sub_str!r} must have degree 1"
                    )
                sub_nbr_idx = sub_nbrs[0].GetIdx()
                sub_bond = sub_mol.GetBondBetweenAtoms(sub_dummy.GetIdx(), sub_nbr_idx).GetBondType()
                bond_type = sub_bond if sub_bond != Chem.BondType.SINGLE else scaf_bond

                if bond_type == Chem.BondType.SINGLE and hasattr(Chem, "molzip"):
                    # Preserve tetrahedral chirality natively via RDKit molzip
                    scaf_dummy_idx = scaf_dummy.GetIdx()
                    saved_labels: list[tuple[int, int, int]] = []
                    for atom in rw.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            a_idx = atom.GetIdx()
                            saved_labels.append((a_idx, atom.GetAtomMapNum(), atom.GetIsotope()))
                            atom.SetAtomMapNum(0)
                            atom.SetIsotope(0)
                    rw.GetAtomWithIdx(scaf_dummy_idx).SetAtomMapNum(9999)

                    sub_copy = Chem.RWMol(sub_mol)
                    for atom in sub_copy.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            atom.SetAtomMapNum(0)
                            atom.SetIsotope(0)
                    sub_copy.GetAtomWithIdx(sub_dummy.GetIdx()).SetAtomMapNum(9999)

                    zipped = Chem.molzip(rw.GetMol(), sub_copy.GetMol())
                    # Restore remaining dummy labels (indices >= scaf_dummy_idx shifted by -1)
                    rw_zipped = Chem.RWMol(zipped)
                    for orig_idx, amap, iso in saved_labels:
                        if orig_idx == scaf_dummy_idx:
                            continue
                        new_idx = orig_idx - 1 if orig_idx > scaf_dummy_idx else orig_idx
                        if 0 <= new_idx < rw_zipped.GetNumAtoms():
                            a = rw_zipped.GetAtomWithIdx(new_idx)
                            if a.GetAtomicNum() == 0:
                                a.SetAtomMapNum(amap)
                                a.SetIsotope(iso)
                    rw = rw_zipped
                else:
                    offset = rw.GetNumAtoms()
                    combo = Chem.RWMol(Chem.CombineMols(rw.GetMol(), sub_mol))
                    combo.AddBond(scaf_nbr.GetIdx(), offset + sub_nbr_idx, bond_type)
                    for rm_idx in sorted((scaf_dummy.GetIdx(), offset + sub_dummy.GetIdx()), reverse=True):
                        combo.RemoveAtom(rm_idx)
                    rw = combo

        mol = rw.GetMol()
        unresolved = [
            f"[*:{a.GetAtomMapNum() or a.GetIsotope() or '?'}]"
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 0
        ]
        if unresolved:
            return None, f"Unresolved Markush attachment point(s) remaining: {sorted(set(unresolved))}"

        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True), None
    except Exception as exc:
        return None, f"Markush assembly failed: {exc}"


def assemble_markush_batch(
    scaffold_smiles: str,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble a batch of Markush series items from a shared scaffold and R-group maps."""
    assembled: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for item in items or []:
        cid = str(item.get("compound_id", "")).strip()
        subs = item.get("substituents") or {}
        if not isinstance(subs, dict):
            errors.append({"compound_id": cid or "?", "error": "substituents must be a dict"})
            continue
        smi, err = assemble_markush_single(scaffold_smiles, subs)
        if smi is not None:
            assembled.append({"compound_id": cid, "smiles": smi})
        else:
            errors.append({"compound_id": cid or "?", "error": err or "assembly error"})

    return {
        "scaffold_smiles": scaffold_smiles,
        "assembled": assembled,
        "errors": errors,
    }


def is_markush_applicable(doc_text: str) -> bool:
    """Return True if ``doc_text`` contains unexpanded RDKit Markush attachment dummies."""
    if not doc_text:
        return False
    return bool(_MARKUSH_DUMMY_RE.search(doc_text))


def _parse_json_like_arg(val: Any) -> Any:
    """Coerce JSON-stringified tool arguments emitted by some LLM function-calling backends."""
    if not isinstance(val, str):
        return val
    cleaned = val.strip()
    if not cleaned:
        return val
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return val


class MarkushItemInput(BaseModel):
    """Single compound R-group assignment for batch Markush assembly."""

    compound_id: str = Field(description="Compound identifier matching the target compound row (e.g. '5a')")
    substituents: Dict[str, str] = Field(
        description=(
            "Mapping from attachment point index (e.g. '1' or '[*:1]') to substituent SMILES "
            "fragment (e.g. {'1': 'C[*:1]'} or {'1': 'H'})"
        )
    )

    @field_validator("substituents", mode="before")
    @classmethod
    def _coerce_substituents_dict(cls, v: Any) -> Any:
        parsed = _parse_json_like_arg(v)
        if isinstance(parsed, dict):
            return {str(k): str(val) for k, val in parsed.items()}
        return parsed


class AssembleMarkushBatchInput(BaseModel):
    """Arguments schema for assemble_markush_smiles tool."""

    scaffold_smiles: str = Field(
        description="Core scaffold SMILES with RDKit dummy attachment atoms (e.g. '[*:1]', '[*:2]')"
    )
    items: List[MarkushItemInput] = Field(
        description="List of compound IDs and their substituent fragment maps to assemble in batch"
    )

    @field_validator("items", mode="before")
    @classmethod
    def _coerce_items_list(cls, v: Any) -> Any:
        return _parse_json_like_arg(v)


def build_markush_assembly_tool(**_: Any) -> Any:
    """Build the LangChain StructuredTool for deterministic RDKit Markush assembly."""
    from langchain_core.tools import StructuredTool

    def _run_assemble(
        scaffold_smiles: str,
        items: List[Any],
    ) -> str:
        norm_items: List[Dict[str, Any]] = []
        for it in items:
            if isinstance(it, BaseModel):
                norm_items.append(it.model_dump())
            elif isinstance(it, dict):
                norm_items.append(it)
        res = assemble_markush_batch(scaffold_smiles=scaffold_smiles, items=norm_items)
        return json.dumps(res, ensure_ascii=False, indent=2)

    return StructuredTool.from_function(
        func=_run_assemble,
        name="assemble_markush_smiles",
        description=(
            "Deterministically substitute R-group SMILES fragments into a Markush core scaffold "
            "with `[*:1]`, `[*:2]` attachment points via RDKit and return canonical valid SMILES "
            "for a batch of compounds."
        ),
        args_schema=AssembleMarkushBatchInput,
    )


_CHEMISTRY_STRUCTURED_HINT = (
    "CHEMICAL STRUCTURE CROSS-CHECK (VLM vs PROSE/IUPAC):\n"
    "- Always cross-check substituent positions, positional isomers (e.g. 2-chloro-6-fluorophenyl vs "
    "5-chloro-2-fluorophenyl, 6-methylpyridin-3-yl vs 2-methylpyridin-4-yl), and scaffolds parsed in "
    "`<visual>` tables/registries against explicit IUPAC names and compound descriptions in the document "
    "text/experimental section and any computed tool results (if present).\n"
    "- If a `<visual>` SMILES contradicts unambiguous chemical nomenclature in the prose or authoritative "
    "computed SMILES in `--- COMPUTED TOOL RESULTS (AUTHORITATIVE) ---` (when provided), the prose IUPAC "
    "nomenclature / computed SMILES takes precedence."
)


def get_markush_assembly_tool_provider() -> RegisteredToolProvider:
    """Return the RegisteredToolProvider descriptor for Markush SMILES assembly."""
    return RegisteredToolProvider(
        name="assemble_markush_smiles",
        build_tool=build_markush_assembly_tool,
        is_applicable=is_markush_applicable,
        policy_hint=_MARKUSH_POLICY_HINT,
        structured_output_hint=_CHEMISTRY_STRUCTURED_HINT,
    )


_IUPAC_POLICY_HINT = (
    "TOOL `resolve_iupac_to_smiles`:\n"
    "- Use this tool when a compound is described by an explicit IUPAC chemical name in prose or the "
    "experimental section without a ready SMILES, OR when prose specifies explicit substituent locants "
    "(e.g., `2-chloro-6-fluorophenyl`, `3-chloro-2-fluorophenyl`, `6-methylpyridin-3-yl`) so you can "
    "verify/override `<visual>` OCSR SMILES against positional isomer errors.\n"
    "- Batch all needed IUPAC names into a single `resolve_iupac_to_smiles` call.\n"
    "- Calling `resolve_iupac_to_smiles` does NOT replace `assemble_markush_smiles`: if the document "
    "also contains target R-group series with `[*:n]` Markush scaffolds, still call `assemble_markush_smiles`."
)


class ResolveIupacBatchInput(BaseModel):
    """Arguments schema for resolve_iupac_to_smiles tool."""

    iupac_names: List[str] = Field(
        description="List of IUPAC chemical names to convert into canonical SMILES strings via OPSIN"
    )

    @field_validator("iupac_names", mode="before")
    @classmethod
    def _coerce_names_list(cls, v: Any) -> Any:
        parsed = _parse_json_like_arg(v)
        if isinstance(parsed, str):
            return [parsed]
        return parsed


_OPSIN_LOCK = threading.Lock()
_IUPAC_CACHE: OrderedDict[str, str] = OrderedDict()
_IUPAC_CACHE_MAX_SIZE = 4096
_IUPAC_DASH_TRANS = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
_IUPAC_WS_RE = re.compile(r"\s+")


def _normalize_iupac_cache_key(name: str) -> str:
    """Normalize Unicode dashes and whitespace in IUPAC cache keys."""
    return _IUPAC_WS_RE.sub(" ", name.translate(_IUPAC_DASH_TRANS)).strip()


_COMMON_LAB_SOLVENTS_RE = re.compile(
    r"(?<![A-Za-z0-9-])(?:"
    r"N\s*,\s*N\s*[-−–—]\s*dimethylformamide|"
    r"1\s*,\s*2\s*[-−–—]\s*dichloroethane|"
    r"2\s*[-−–—]\s*mercaptoethanol|"
    r"dimethyl\s+sulfoxide|"
    r"tetrahydrofuran"
    r")(?![A-Za-z0-9-])",
    re.IGNORECASE,
)

_IUPAC_LOCANT_PREFIX_RE = re.compile(
    r"(?:^|[\s(\[;,])(?:\d+(?:,\d+)*|[NOS](?:,[NOS])*)-"
    r"(?:chloro|fluoro|bromo|iodo|methyl|ethyl|propyl|butyl|pentyl|hexyl|phenyl|benzyl|"
    r"naphthyl|methoxy|ethoxy|hydroxy|amino|amido|nitro|cyano|carboxy|acetyl|formyl|oxo|"
    r"thio|mercapto|trifluoromethyl|trifluoromethoxy|pyridin|pyrimidin|pyrazin|pyridazin|"
    r"piperidin|piperazin|morpholin|pyrrolidin|thiophen|furan|pyrrol|imidazol|benzimidazol|"
    r"thiazol|oxazol|triazol|indol|quinolin|isoquinolin|purin|bis|tris|di|tri|tetra|penta)",
    re.IGNORECASE,
)
_IUPAC_LOCANT_SUFFIX_RE = re.compile(
    r"\b[a-z]{3,}-\d+-(?:yl|ylidene|ol|one|al|oic|ate|amide|amine|imine|ene|yne|thiol)\b",
    re.IGNORECASE,
)
_IUPAC_RING_FUSION_RE = re.compile(
    r"(?:\b\d+H-(?:benz|imid|pyr|ind|thi|fur|ox|triaz|quin)[a-z]+|"
    r"\b[a-z]+\[\d*,\d*-?[a-z]+\][a-z]+|\b\d+-\(\d+-)",
    re.IGNORECASE,
)


def is_iupac_applicable(doc_text: str) -> bool:
    """Return True if ``doc_text`` contains IUPAC-style locants or systematic chemical suffixes."""
    if not doc_text or not doc_text.strip():
        return False
    scrubbed = _COMMON_LAB_SOLVENTS_RE.sub(" ", doc_text)
    return bool(
        _IUPAC_LOCANT_PREFIX_RE.search(scrubbed)
        or _IUPAC_LOCANT_SUFFIX_RE.search(scrubbed)
        or _IUPAC_RING_FUSION_RE.search(scrubbed)
    )


def resolve_iupac_batch(iupac_names: List[str]) -> Dict[str, Any]:
    """Convert a batch of IUPAC chemical names to canonical SMILES using py2opsin."""
    cleaned_names = [str(n).strip() for n in (iupac_names or []) if str(n).strip()]
    if not cleaned_names:
        return {"resolved": [], "errors": [{"iupac_name": "", "error": "No IUPAC names provided."}]}

    name_to_key = {name: _normalize_iupac_cache_key(name) for name in cleaned_names}

    with _OPSIN_LOCK:
        cached_map: Dict[str, str] = {}
        for key in name_to_key.values():
            if key in _IUPAC_CACHE and key not in cached_map:
                _IUPAC_CACHE.move_to_end(key)
                cached_map[key] = _IUPAC_CACHE[key]

    uncached_names: List[str] = []
    seen_uncached: set[str] = set()
    for name in cleaned_names:
        key = name_to_key[name]
        if key not in cached_map and key not in seen_uncached:
            seen_uncached.add(key)
            uncached_names.append(key)

    fresh_resolved: Dict[str, str] = {}
    fresh_errors: Dict[str, str] = {}

    if uncached_names:
        if not shutil.which("java"):
            for name in uncached_names:
                fresh_errors[name] = (
                    "Java runtime ('java') not found in PATH; py2opsin requires Java."
                )
        else:
            py2opsin_fn: Any = None
            try:
                from py2opsin import py2opsin as py2opsin_fn
            except ImportError as exc:
                for name in uncached_names:
                    fresh_errors[name] = f"py2opsin is required for IUPAC resolution: {exc}"

            if py2opsin_fn is not None:
                raw_results: Any = None
                io_error_msg: Optional[str] = None
                with _OPSIN_LOCK:
                    tmp_fpath: Optional[str] = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w+",
                            suffix=".txt",
                            prefix="ae_py2opsin_",
                            delete=False,
                            encoding="utf-8",
                        ) as tmp_file:
                            tmp_fpath = tmp_file.name
                        try:
                            raw_results = py2opsin_fn(uncached_names, tmp_fpath=tmp_fpath)
                        except TypeError:
                            raw_results = py2opsin_fn(uncached_names)
                    except Exception as exc:
                        io_error_msg = f"OPSIN conversion failed: {exc}"
                    finally:
                        if tmp_fpath is not None:
                            try:
                                Path(tmp_fpath).unlink(missing_ok=True)
                            except Exception:
                                pass

                if io_error_msg is not None:
                    for name in uncached_names:
                        fresh_errors[name] = io_error_msg
                elif isinstance(raw_results, bool) or not isinstance(
                    raw_results, (str, list, tuple)
                ):
                    for name in uncached_names:
                        fresh_errors[name] = "OPSIN conversion failed: subprocess/I/O error"
                else:
                    if isinstance(raw_results, str):
                        raw_list: List[Optional[str]] = [raw_results]
                    else:
                        raw_list = list(raw_results)

                    try:
                        from rdkit import Chem, rdBase

                        rdBase.DisableLog("rdApp.error")
                        rdBase.DisableLog("rdApp.warning")
                    except ImportError:
                        Chem = None  # type: ignore[assignment]

                    for idx, name in enumerate(uncached_names):
                        raw_smi = raw_list[idx] if idx < len(raw_list) else None
                        smi = str(raw_smi).strip() if raw_smi else ""
                        if not smi:
                            fresh_errors[name] = "OPSIN could not parse IUPAC name"
                            continue
                        if Chem is not None:
                            mol = Chem.MolFromSmiles(smi)
                            if mol is None:
                                fresh_errors[name] = f"Invalid SMILES from OPSIN: {smi!r}"
                                continue
                            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                        fresh_resolved[name] = smi

                    if fresh_resolved:
                        with _OPSIN_LOCK:
                            for key, smi in fresh_resolved.items():
                                if key in _IUPAC_CACHE:
                                    _IUPAC_CACHE.move_to_end(key)
                                _IUPAC_CACHE[key] = smi
                                while len(_IUPAC_CACHE) > _IUPAC_CACHE_MAX_SIZE:
                                    _IUPAC_CACHE.popitem(last=False)

    resolved: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    for name in cleaned_names:
        key = name_to_key[name]
        if key in cached_map:
            resolved.append({"iupac_name": name, "smiles": cached_map[key]})
        elif key in fresh_resolved:
            resolved.append({"iupac_name": name, "smiles": fresh_resolved[key]})
        else:
            errors.append(
                {
                    "iupac_name": name,
                    "error": fresh_errors.get(key, "OPSIN could not parse IUPAC name"),
                }
            )

    return {"resolved": resolved, "errors": errors}


def build_iupac_to_smiles_tool(**_: Any) -> Any:
    """Build the LangChain StructuredTool for IUPAC-to-SMILES resolution via py2opsin."""
    from langchain_core.tools import StructuredTool

    def _run_resolve(iupac_names: List[str]) -> str:
        res = resolve_iupac_batch(iupac_names=iupac_names)
        return json.dumps(res, ensure_ascii=False, indent=2)

    return StructuredTool.from_function(
        func=_run_resolve,
        name="resolve_iupac_to_smiles",
        description=(
            "Convert a batch of IUPAC chemical names into canonical SMILES strings using local OPSIN "
            "(py2opsin) + RDKit. Useful when SMILES is missing or when cross-checking prose IUPAC names."
        ),
        args_schema=ResolveIupacBatchInput,
    )


def get_iupac_to_smiles_tool_provider() -> RegisteredToolProvider:
    """Return the RegisteredToolProvider descriptor for IUPAC-to-SMILES resolution."""
    return RegisteredToolProvider(
        name="resolve_iupac_to_smiles",
        build_tool=build_iupac_to_smiles_tool,
        is_applicable=is_iupac_applicable,
        policy_hint=_IUPAC_POLICY_HINT,
        auto_remind=False,
        structured_output_hint=_CHEMISTRY_STRUCTURED_HINT,
    )
