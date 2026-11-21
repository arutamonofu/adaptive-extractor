"""Chemistry-domain extraction tools (lazy-loaded only when declared in schema)."""

from __future__ import annotations

import ast
import json
import re
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


def get_markush_assembly_tool_provider() -> RegisteredToolProvider:
    """Return the RegisteredToolProvider descriptor for Markush SMILES assembly."""
    return RegisteredToolProvider(
        name="assemble_markush_smiles",
        build_tool=build_markush_assembly_tool,
        is_applicable=is_markush_applicable,
        policy_hint=_MARKUSH_POLICY_HINT,
    )
