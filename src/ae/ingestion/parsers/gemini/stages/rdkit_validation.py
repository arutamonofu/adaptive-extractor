"""RDKit validation and LLM self-correction for extracted SMILES strings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.schemas import ChemicalEntity
from ae.ingestion.parsers.gemini.stages.ocsr_oracle import (
    OracleFinding,
    _is_markush_smiles,
    formulas_compatible,
    substructure_violations,
    verify_entities_with_oracle,
)

logger = logging.getLogger(__name__)


class SmilesCorrectionItem(BaseModel):
    label: str
    corrected_smiles: str


class SmilesCorrectionResponse(BaseModel):
    corrections: list[SmilesCorrectionItem] = Field(default_factory=list)


def _read_rdkit_correction_prompt() -> str:
    prompt_file = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "04_parse_visuals"
        / "rdkit_correction.txt"
    )
    return prompt_file.read_text(encoding="utf-8")


def _format_prose_hints(finding: Optional[OracleFinding]) -> str:
    """Authoritative prose-derived constraints for a flagged entity (empty if none)."""
    if finding is None:
        return ""
    parts: list[str] = []
    for name, rad_smi in zip(
        finding.expected_substructure_names, finding.expected_substructure_smiles
    ):
        parts.append(f'must contain substructure "{rad_smi}" (prose name: {name})')
    if finding.expected_formula:
        parts.append(f"molecular formula must be compatible with {finding.expected_formula}")
    if finding.expected_smiles:
        suffix = f" (InChIKey {finding.expected_inchi_key})" if finding.expected_inchi_key else ""
        parts.append(
            f'the full prose name "{finding.expected_iupac}" resolves to '
            f"{finding.expected_smiles}{suffix}"
        )
    if not parts:
        return ""
    return (
        "\n  Prose-derived constraints (authoritative for locants and connectivity): "
        + "; ".join(parts)
        + "."
    )


def validate_and_canonicalize(smiles: str) -> tuple[bool, str, Optional[str]]:
    """Validate SMILES string using RDKit and return canonicalized representation.

    Returns:
        (is_valid, canonical_smiles_or_original, error_message_if_any)
    """
    if not smiles or not smiles.strip():
        return False, smiles, "Empty SMILES string"

    clean_smiles = smiles.strip()
    try:
        from rdkit import Chem, rdBase

        # Suppress RDKit C++ stderr spam
        rdBase.DisableLog("rdApp.error")
        rdBase.DisableLog("rdApp.warning")

        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return False, clean_smiles, f"RDKit Chem.MolFromSmiles failed to parse: {clean_smiles}"

        canonical = Chem.MolToSmiles(mol)
        return True, canonical, None
    except Exception as exc:
        return False, clean_smiles, str(exc)


def _candidate_violations(canonical_smiles: str, finding: Optional[OracleFinding]) -> list[str]:
    if finding is None:
        return []
    from rdkit import Chem

    return substructure_violations(Chem.MolFromSmiles(canonical_smiles), finding)


def _reappend_notes(entities: list[ChemicalEntity], notes_by_idx: dict[int, list[str]]) -> None:
    """Re-attach diagnostic notes after a full oracle re-verify (which rewrites oracle_notes)."""
    for idx, notes in notes_by_idx.items():
        ent = entities[idx]
        suffix = "; ".join(notes)
        ent.oracle_notes = f"{ent.oracle_notes}; {suffix}" if ent.oracle_notes else suffix


def _formula_of_agrees_with(smiles: str, expected_formula: str) -> bool:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return formulas_compatible(rdMolDescriptors.CalcMolFormula(mol), expected_formula)


def validate_and_correct_entities(
    entities: list[ChemicalEntity],
    crop_paths: list[Any],
    client: Optional[GeminiClient],
    config: Any,
    stage_model: Optional[str] = None,
    max_attempts: Optional[int] = None,
    stage_name: str = "rdkit_correction",
    context: str = "",
) -> tuple[list[ChemicalEntity], dict[str, str]]:
    """Validate entities via RDKit with iterative self-correction retries.

    Returns:
        (updated_entities, smiles_replacements_map)
    """
    mol_cfg = getattr(config.gemini, "molecules", None) if config and hasattr(config, "gemini") else None
    should_validate = getattr(mol_cfg, "validate_rdkit", True) if mol_cfg else True
    enable_oracle = getattr(mol_cfg, "enable_ocsr_oracle", True) if mol_cfg else True
    if max_attempts is not None:
        max_retries = max_attempts
    else:
        max_retries = getattr(mol_cfg, "max_rdkit_retries", 2) if mol_cfg else 2

    replacements: dict[str, str] = {}
    if not should_validate or not entities:
        return entities, replacements

    # Initial RDKit validation pass
    for ent in entities:
        orig = ent.smiles
        valid, can, err = validate_and_canonicalize(orig)
        if valid:
            ent.smiles = can
            ent.smiles_valid = True
            ent.rdkit_error = None
            if orig != can:
                replacements[orig] = can
        else:
            ent.smiles_valid = False
            ent.rdkit_error = err

    # Initial deterministic OCSR oracle check (ChEMBL, Formula/HRMS, py2opsin IUPAC/substituents)
    findings: Optional[list[OracleFinding]] = None
    if enable_oracle:
        findings = verify_entities_with_oracle(entities, context=context)
        if len(findings) != len(entities):
            findings = None

    # If client is provided and there are invalid SMILES or oracle mismatches, enter retry loop
    if client is not None and max_retries > 0:
        for attempt in range(1, max_retries + 1):
            invalid: list[tuple[int, ChemicalEntity, str]] = []
            for idx, e in enumerate(entities):
                if not e.smiles_valid:
                    invalid.append((idx, e, e.rdkit_error or "Invalid SMILES"))
                elif enable_oracle and getattr(e, "provenance_status", "unverified_valid") == "mismatch_flagged":
                    invalid.append(
                        (idx, e, getattr(e, "oracle_notes", None) or "Semantic oracle mismatch")
                    )
            if not invalid:
                break

            logger.info(
                f"Validation/Oracle found {len(invalid)} flagged SMILES. "
                f"Attempting Gemini self-correction ({attempt}/{max_retries})…",
            )

            items_lines = []
            for idx, ent, error_note in invalid:
                finding = findings[idx] if findings else None
                items_lines.append(
                    f"- Label '{ent.label}': SMILES '{ent.smiles}' "
                    f"(Validation Error: {error_note}){_format_prose_hints(finding)}"
                )
            template = _read_rdkit_correction_prompt()
            ctx_text = context.strip() if context and context.strip() else "(none)"
            prompt = template.replace("{invalid_items_list}", "\n".join(items_lines)).replace(
                "{context}", ctx_text
            )

            try:
                contents = list(crop_paths) + [prompt]
                corr_res: SmilesCorrectionResponse = client.generate_structured(
                    contents,
                    response_schema=SmilesCorrectionResponse,
                    stage_name=stage_name,
                    model=stage_model,
                    thinking_level="medium",
                )

                corr_map = {c.label: c.corrected_smiles for c in corr_res.corrections}
                rejection_notes: dict[int, list[str]] = {}

                for idx, ent, _ in invalid:
                    if ent.label not in corr_map:
                        continue
                    old_smiles = ent.smiles
                    corrected_smiles = corr_map[ent.label]
                    valid, can, err = validate_and_canonicalize(corrected_smiles)
                    if not valid:
                        ent.smiles = corrected_smiles
                        ent.smiles_valid = False
                        ent.rdkit_error = err
                        if old_smiles != corrected_smiles:
                            replacements[old_smiles] = corrected_smiles
                        continue

                    finding = findings[idx] if findings else None
                    violations = _candidate_violations(can, finding)
                    if violations:
                        rejection_notes.setdefault(idx, []).append(
                            f"candidate '{can}' rejected: {'; '.join(violations)}"
                        )
                        logger.info(
                            f"Correction for '{ent.label}' rejected by prose gate: {violations}"
                        )
                        continue
                    ent.smiles = can
                    ent.smiles_valid = True
                    ent.rdkit_error = None
                    replacements[old_smiles] = can

                findings = verify_entities_with_oracle(entities, context=context)
                if len(findings) != len(entities):
                    findings = None
                _reappend_notes(entities, rejection_notes)
            except Exception as exc:
                logger.warning(f"RDKit/Oracle self-correction call failed: {exc}")
                break

    # Deterministic last-resort override: the prose IUPAC heading resolves unambiguously,
    # the drawing disagrees on connectivity, and (when present) the prose formula agrees.
    if enable_oracle and findings and len(findings) == len(entities):
        override_notes: dict[int, list[str]] = {}
        for idx, (ent, f) in enumerate(zip(entities, findings)):
            if (
                f.provenance_status != "mismatch_flagged"
                or not f.expected_smiles
                or not f.connectivity_conflict
                or _is_markush_smiles(ent.smiles)
            ):
                continue
            ok, can, _ = validate_and_canonicalize(f.expected_smiles)
            if not ok:
                continue
            if f.expected_formula and not _formula_of_agrees_with(can, f.expected_formula):
                continue
            replacements[ent.smiles] = can
            ent.smiles = can
            ent.smiles_valid = True
            ent.rdkit_error = None
            override_notes[idx] = ["overridden from prose IUPAC heading (stereo may be lost)"]
        if override_notes:
            verify_entities_with_oracle(entities, context=context)
            _reappend_notes(entities, override_notes)

    return entities, replacements
