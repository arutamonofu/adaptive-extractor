"""Chemistry-domain field validators (lazy-loaded only when declared in schema)."""

from __future__ import annotations

import functools
from typing import Any, Optional, Tuple

from ae.core.schema.validators import FieldValidationResult

_DEFAULT_NULL_SENTINELS = frozenset({"nan", "n/a", "null", "none"})
_STEREO_TOKENS = ("@", "/", "\\")


def _import_rdkit() -> Tuple[Any, Any]:
    """Lazily import RDKit Chem and rdBase with consistent error message."""
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for the 'smiles' field validator. "
            "Install optional chemistry dependencies "
            "(e.g. `pip install adaptive-extractor[chemistry]`)."
        ) from exc

    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")
    return Chem, rdBase


def _parse_smiles_mol(
    value: Any,
    *,
    allow_dummy_atoms: bool = False,
    allow_fragments: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    """Parse and validate a SMILES string into an RDKit Mol, returning (mol, error)."""
    if not isinstance(value, str):
        return None, f"expected SMILES string, got {type(value).__name__}"

    clean_smiles = value.strip()
    if not clean_smiles:
        return None, "empty SMILES string"

    Chem, _ = _import_rdkit()
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        return None, f"invalid SMILES syntax or valence: {clean_smiles!r}"

    if not allow_dummy_atoms:
        dummy_labels = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom_map = atom.GetAtomMapNum()
                isotope = atom.GetIsotope()
                if atom_map:
                    dummy_labels.append(f"[*:{atom_map}]")
                elif isotope:
                    dummy_labels.append(f"[{isotope}*]")
                else:
                    dummy_labels.append("*")
        if dummy_labels:
            unique_labels = ", ".join(dict.fromkeys(dummy_labels))
            return None, (
                f"unresolved Markush/dummy atom placeholder(s) ({unique_labels}) "
                f"in SMILES {clean_smiles!r}"
            )

    if not allow_fragments and len(Chem.GetMolFrags(mol)) > 1:
        return None, f"disconnected fragments are not allowed in SMILES {clean_smiles!r}"

    return mol, None


def _canonicalize_mol_to_smiles(
    mol: Any,
    *,
    canonicalize_tautomers: bool = True,
    isomeric: bool = True,
) -> str:
    """Canonicalize an RDKit Mol (optionally tautomer-normalized) to SMILES."""
    Chem, _ = _import_rdkit()
    canon_mol = mol
    if canonicalize_tautomers:
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize

            te = rdMolStandardize.TautomerEnumerator()
            te.SetRemoveSp3Stereo(not isomeric)
            te.SetRemoveBondStereo(not isomeric)
            canon_mol = te.Canonicalize(mol)
        except Exception:
            canon_mol = mol
    return str(Chem.MolToSmiles(canon_mol, isomericSmiles=isomeric))


@functools.lru_cache(maxsize=8192)
def _canonicalize_smiles_cached(
    clean_smiles: str,
    allow_dummy_atoms: bool,
    allow_fragments: bool,
    canonicalize_tautomers: bool,
    isomeric: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Parse and canonicalize a SMILES string with LRU caching."""
    mol, error = _parse_smiles_mol(
        clean_smiles,
        allow_dummy_atoms=allow_dummy_atoms,
        allow_fragments=allow_fragments,
    )
    if mol is None:
        return None, error
    return (
        _canonicalize_mol_to_smiles(
            mol,
            canonicalize_tautomers=canonicalize_tautomers,
            isomeric=isomeric,
        ),
        None,
    )


def validate_smiles(
    value: Any,
    *,
    required: bool = True,
    allow_dummy_atoms: bool = False,
    allow_fragments: bool = True,
    canonicalize: bool = False,
    canonicalize_tautomers: bool = True,
    isomeric: bool = True,
    ignore_unassigned_stereo: bool = True,
    allow_null_sentinels: bool = True,
    **_: Any,
) -> FieldValidationResult:
    """Validate a SMILES string via RDKit syntax/valence and placeholder checks.

    Args:
        value: Candidate SMILES value.
        required: Whether the schema field is required.
        allow_dummy_atoms: Whether Markush dummy/attachment atoms (``*``, ``[*:1]``)
            are permitted (default ``False`` for final extraction).
        allow_fragments: Whether disconnected multi-component SMILES (``.``) are permitted.
        canonicalize: Whether to normalize valid SMILES to RDKit canonical form.
        canonicalize_tautomers: Whether to normalize tautomers during canonicalization.
        isomeric: Whether to preserve stereochemistry in canonical SMILES.
        ignore_unassigned_stereo: Whether missing stereochemistry on one side is
            tolerated during value comparison.
        allow_null_sentinels: Whether common missing-value tokens (e.g. ``"NaN"``)
            are skipped when ``required`` is ``False``.

    Returns:
        FieldValidationResult indicating validity and optional canonicalized value.
    """
    _ = ignore_unassigned_stereo
    if not isinstance(value, str):
        return FieldValidationResult(
            valid=False,
            value=value,
            error=f"expected SMILES string, got {type(value).__name__}",
        )

    clean_smiles = value.strip()
    if not clean_smiles:
        return FieldValidationResult(valid=False, value=value, error="empty SMILES string")

    if not required and allow_null_sentinels and clean_smiles.lower() in _DEFAULT_NULL_SENTINELS:
        return FieldValidationResult(valid=True, value=value)

    canon, error = _canonicalize_smiles_cached(
        clean_smiles,
        allow_dummy_atoms,
        allow_fragments,
        canonicalize_tautomers if canonicalize else False,
        isomeric,
    )
    if canon is None:
        return FieldValidationResult(valid=False, value=value, error=error)

    final_val = canon if canonicalize else value
    return FieldValidationResult(valid=True, value=final_val)


def compare_smiles_values(
    val_a: Any,
    val_b: Any,
    *,
    isomeric: bool = True,
    ignore_unassigned_stereo: bool = True,
    canonicalize_tautomers: bool = True,
    allow_dummy_atoms: bool = False,
    allow_fragments: bool = True,
    **_: Any,
) -> Tuple[Optional[bool], Optional[str]]:
    """Compare two SMILES values with configurable stereochemistry handling."""
    if not isinstance(val_a, str) or not isinstance(val_b, str):
        return False, "invalid_smiles"
    clean_a = val_a.strip()
    clean_b = val_b.strip()
    if not clean_a or not clean_b:
        return False, "invalid_smiles"

    if not isomeric:
        flat_a, err_a = _canonicalize_smiles_cached(
            clean_a, allow_dummy_atoms, allow_fragments, canonicalize_tautomers, False
        )
        flat_b, err_b = _canonicalize_smiles_cached(
            clean_b, allow_dummy_atoms, allow_fragments, canonicalize_tautomers, False
        )
        if flat_a is None or flat_b is None or err_a is not None or err_b is not None:
            return False, "invalid_smiles"
        if flat_a == flat_b:
            return True, None
        return False, "topology_mismatch"

    iso_a, err_a = _canonicalize_smiles_cached(
        clean_a, allow_dummy_atoms, allow_fragments, canonicalize_tautomers, True
    )
    iso_b, err_b = _canonicalize_smiles_cached(
        clean_b, allow_dummy_atoms, allow_fragments, canonicalize_tautomers, True
    )
    if iso_a is None or iso_b is None or err_a is not None or err_b is not None:
        return False, "invalid_smiles"
    if iso_a == iso_b:
        return True, None

    flat_a, _ = _canonicalize_smiles_cached(
        clean_a, allow_dummy_atoms, allow_fragments, canonicalize_tautomers, False
    )
    flat_b, _ = _canonicalize_smiles_cached(
        clean_b, allow_dummy_atoms, allow_fragments, canonicalize_tautomers, False
    )
    if flat_a != flat_b:
        return False, "topology_mismatch"

    has_stereo_a = any(tok in iso_a for tok in _STEREO_TOKENS)
    has_stereo_b = any(tok in iso_b for tok in _STEREO_TOKENS)
    if ignore_unassigned_stereo and (not has_stereo_a or not has_stereo_b):
        return True, None

    return False, "stereo_diff"


setattr(validate_smiles, "compare_values", compare_smiles_values)
