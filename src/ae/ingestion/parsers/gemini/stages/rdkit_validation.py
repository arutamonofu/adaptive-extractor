"""RDKit validation and LLM self-correction for extracted SMILES strings."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from pydantic import BaseModel, Field

from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.schemas import AttachmentPoint, ChemicalEntity
from ae.ingestion.parsers.gemini.stages.ocsr_oracle import (
    OracleFinding,
    _finding_expected_formulas,
    _is_markush_smiles,
    check_chembl_structure,
    check_chembl_structure_detailed,
    formulas_compatible,
    reverify_entity_with_finding,
    substructure_violations,
    verify_entities_with_oracle,
)

logger = logging.getLogger(__name__)


class SmilesCorrectionItem(BaseModel):
    label: str
    corrected_smiles: str
    trust_drawing: bool = Field(
        default=False,
        description=(
            "Set true when surrounding prose/hints refer to a different compound and "
            "the drawing structure should be kept instead of prose overrides."
        ),
    )
    prose_conflict_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation when trust_drawing is true due to conflicting prose context.",
    )
    attachment_points: Optional[list[AttachmentPoint]] = Field(
        default=None,
        description=(
            "Updated dummy-atom attachment mappings when corrected_smiles contains "
            "dummy atoms [*:1], [*:2], etc."
        ),
    )


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
    candidate_formulas = _finding_expected_formulas(finding)
    if candidate_formulas:
        parts.append(
            f"molecular formula must be compatible with {' or '.join(candidate_formulas)}"
        )
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


def _has_prose_flags(finding: Optional[OracleFinding]) -> bool:
    """True when an OracleFinding carries prose-derived constraints or expectations."""
    if finding is None:
        return False
    return bool(
        _finding_expected_formulas(finding)
        or finding.expected_smiles
        or finding.expected_inchi_key
        or finding.expected_substructure_smiles
    )


def _resolve_molecule_stage_cfg(config: Any) -> Any:
    """Safely fetch StageModelConfig for parse_visuals/molecule if available."""
    if config is None:
        return None
    gemini_cfg = getattr(config, "gemini", None)
    if gemini_cfg is not None and callable(getattr(gemini_cfg, "get_stage_config", None)):
        try:
            return gemini_cfg.get_stage_config("parse_visuals", sub_stage="molecule")
        except Exception:
            return None
    if callable(getattr(config, "get_stage_config", None)):
        try:
            return config.get_stage_config("parse_visuals", sub_stage="molecule")
        except Exception:
            return None
    return None


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
    elif mol_cfg is not None:
        max_retries = getattr(
            mol_cfg,
            "max_retries",
            getattr(mol_cfg, "max_rdkit_retries", 1),
        )
    else:
        max_retries = 2

    stage_cfg = _resolve_molecule_stage_cfg(config)
    cfg_model = getattr(stage_cfg, "model", None) if stage_cfg is not None else None
    resolved_model = stage_model or (cfg_model if isinstance(cfg_model, str) and cfg_model else None)
    cfg_temp = getattr(stage_cfg, "temperature", None) if stage_cfg is not None else None
    resolved_temp = (
        float(cfg_temp)
        if isinstance(cfg_temp, (int, float)) and not isinstance(cfg_temp, bool)
        else None
    )
    cfg_tokens = getattr(stage_cfg, "max_output_tokens", None) if stage_cfg is not None else None
    resolved_tokens = (
        cfg_tokens
        if isinstance(cfg_tokens, int) and not isinstance(cfg_tokens, bool)
        else None
    )
    cfg_res = getattr(stage_cfg, "media_resolution", None) if stage_cfg is not None else None
    resolved_res = cfg_res if isinstance(cfg_res, str) and cfg_res else None
    cfg_tlevel = getattr(stage_cfg, "thinking_level", None) if stage_cfg is not None else None
    resolved_tlevel = (
        cfg_tlevel.strip().lower()
        if isinstance(cfg_tlevel, str)
        and cfg_tlevel.strip().lower() in ("minimal", "low", "medium", "high")
        else "medium"
    )
    cfg_tbudget = getattr(stage_cfg, "thinking_budget", None) if stage_cfg is not None else None
    resolved_tbudget = (
        cfg_tbudget
        if isinstance(cfg_tbudget, int) and not isinstance(cfg_tbudget, bool)
        else None
    )

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
    trust_drawing_reasons: dict[int, Optional[str]] = {}
    if client is not None and max_retries > 0:
        noop_indices: set[int] = set()
        for attempt in range(1, max_retries + 1):
            invalid: list[tuple[int, ChemicalEntity, str]] = []
            for idx, e in enumerate(entities):
                if idx in noop_indices:
                    continue
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

            is_pure_syntax_only = all(not ent.smiles_valid for _, ent, _ in invalid) and not any(
                _has_prose_flags(findings[idx] if findings and idx < len(findings) else None)
                or any(
                    r.startswith("ChEMBL structure issue:")
                    for r in (findings[idx].reasons if findings and idx < len(findings) else [])
                )
                for idx, _, _ in invalid
            )
            attempt_tlevel = (
                "low"
                if is_pure_syntax_only
                and resolved_tbudget is None
                and resolved_tlevel in ("medium", "high")
                else resolved_tlevel
            )

            try:
                contents = list(crop_paths) + [prompt]
                gen_kwargs: dict[str, Any] = {
                    "response_schema": SmilesCorrectionResponse,
                    "stage_name": stage_name,
                    "model": resolved_model,
                    "thinking_level": attempt_tlevel,
                }
                if resolved_temp is not None:
                    gen_kwargs["temperature"] = resolved_temp
                if resolved_tokens is not None:
                    gen_kwargs["max_output_tokens"] = resolved_tokens
                if resolved_res is not None:
                    gen_kwargs["media_resolution"] = resolved_res
                if resolved_tbudget is not None:
                    gen_kwargs["thinking_budget"] = resolved_tbudget

                corr_res: SmilesCorrectionResponse = client.generate_structured(
                    contents,
                    **gen_kwargs,
                )

                corr_map = {
                    c.label.strip().lower(): c
                    for c in corr_res.corrections
                    if c.label
                }
                rejection_notes: dict[int, list[str]] = {}
                changed_indices: set[int] = set()
                flagged_improvements: set[int] = set()
                markush_to_concrete_indices: set[int] = set()

                for idx, ent, _ in invalid:
                    norm_label = ent.label.strip().lower()
                    if norm_label not in corr_map:
                        continue
                    corr_item = corr_map[norm_label]
                    old_smiles = ent.smiles
                    corrected_smiles = corr_item.corrected_smiles
                    if corr_item.trust_drawing:
                        reason = (corr_item.prose_conflict_reason or "").strip() or None
                        trust_drawing_reasons[idx] = reason

                    valid, can, err = validate_and_canonicalize(corrected_smiles)
                    if not valid:
                        if ent.smiles_valid:
                            rejection_notes.setdefault(idx, []).append(
                                f"invalid candidate '{corrected_smiles}' rejected ({err}); kept previous valid SMILES"
                            )
                            logger.info(
                                f"Invalid correction for '{ent.label}' rejected to preserve valid SMILES: {err}"
                            )
                            continue
                        ent.smiles = corrected_smiles
                        ent.smiles_valid = False
                        ent.rdkit_error = err
                        if corr_item.attachment_points is not None:
                            ent.attachment_points = list(corr_item.attachment_points)
                        if old_smiles != corrected_smiles:
                            replacements[old_smiles] = corrected_smiles
                            changed_indices.add(idx)
                        continue

                    if ent.smiles_valid and can == old_smiles:
                        if corr_item.attachment_points is not None:
                            ent.attachment_points = list(corr_item.attachment_points)
                        if corr_item.trust_drawing:
                            reason = trust_drawing_reasons.get(idx)
                            rejection_notes.setdefault(idx, []).append(
                                f"VLM trusted drawing over prose ({reason})"
                                if reason
                                else "VLM trusted drawing over prose"
                            )
                        noop_indices.add(idx)
                        continue

                    finding = findings[idx] if findings else None
                    if ent.smiles_valid and finding is not None:
                        is_chembl_only = bool(
                            finding.reasons
                            and all(r.startswith("ChEMBL structure issue:") for r in finding.reasons)
                        )
                        if is_chembl_only:
                            from rdkit import Chem

                            old_mol = Chem.MolFromSmiles(old_smiles)
                            new_mol = Chem.MolFromSmiles(can)
                            if old_mol is not None and new_mol is not None:
                                old_issues = check_chembl_structure_detailed(old_mol)
                                new_issues = check_chembl_structure_detailed(new_mol)
                                old_max = max((sev for sev, _ in old_issues), default=0)
                                new_max = max((sev for sev, _ in new_issues), default=0)
                                if len(new_issues) > len(old_issues) or new_max > old_max:
                                    rejection_notes.setdefault(idx, []).append(
                                        f"candidate '{can}' rejected: ChEMBL issues worsened "
                                        f"(count {len(old_issues)}->{len(new_issues)}, max severity {old_max}->{new_max})"
                                    )
                                    logger.info(
                                        f"Correction for '{ent.label}' rejected by ChEMBL regression gate: "
                                        f"count {len(old_issues)}->{len(new_issues)}, max_sev {old_max}->{new_max}"
                                    )
                                    continue

                    violations = _candidate_violations(can, finding)
                    if violations:
                        if not ent.smiles_valid:
                            # Accept RDKit-valid candidate as an improvement over unparseable SMILES,
                            # while keeping mismatch_flagged + diagnostic notes.
                            ent.smiles = can
                            ent.smiles_valid = True
                            ent.rdkit_error = None
                            ent.provenance_status = "mismatch_flagged"
                            if corr_item.attachment_points is not None:
                                ent.attachment_points = list(corr_item.attachment_points)
                            if _is_markush_smiles(old_smiles) and not _is_markush_smiles(can):
                                markush_to_concrete_indices.add(idx)
                            replacements[old_smiles] = can
                            changed_indices.add(idx)
                            flagged_improvements.add(idx)
                            rejection_notes.setdefault(idx, []).append(
                                f"accepted RDKit-valid candidate '{can}' over invalid SMILES despite prose violations: {'; '.join(violations)}"
                            )
                            logger.info(
                                f"Accepted RDKit-valid improvement for '{ent.label}' despite prose violations: {violations}"
                            )
                            continue
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
                    if corr_item.attachment_points is not None:
                        ent.attachment_points = list(corr_item.attachment_points)
                    if _is_markush_smiles(old_smiles) and not _is_markush_smiles(can):
                        markush_to_concrete_indices.add(idx)
                    replacements[old_smiles] = can
                    changed_indices.add(idx)

                if enable_oracle and changed_indices:
                    if findings and len(findings) == len(entities):
                        if markush_to_concrete_indices:
                            probe_entities = [
                                entities[i]
                                if i in markush_to_concrete_indices
                                else SimpleNamespace(
                                    label=entities[i].label,
                                    smiles=entities[i].smiles,
                                    smiles_valid=entities[i].smiles_valid,
                                    rdkit_error=getattr(entities[i], "rdkit_error", None),
                                )
                                for i in range(len(entities))
                            ]
                            fresh_findings = verify_entities_with_oracle(probe_entities, context=context)
                            if len(fresh_findings) == len(entities):
                                for idx in sorted(markush_to_concrete_indices):
                                    findings[idx] = fresh_findings[idx]
                        for idx in sorted(changed_indices - markush_to_concrete_indices):
                            findings[idx] = reverify_entity_with_finding(entities[idx], findings[idx])
                    else:
                        findings = verify_entities_with_oracle(entities, context=context)
                        if len(findings) != len(entities):
                            findings = None
                for idx in flagged_improvements:
                    entities[idx].provenance_status = "mismatch_flagged"
                    if findings and idx < len(findings):
                        findings[idx].provenance_status = "mismatch_flagged"
                        findings[idx].passed = False
                _reappend_notes(entities, rejection_notes)
            except Exception as exc:
                logger.warning(f"RDKit/Oracle self-correction call failed: {exc}")
                break

    # Deterministic last-resort override: the prose IUPAC heading resolves unambiguously,
    # the drawing disagrees on connectivity (or failed RDKit parsing), and prose constraints agree.
    if enable_oracle and findings and len(findings) == len(entities):
        override_notes: dict[int, list[str]] = {}
        skipped_override_notes: dict[int, list[str]] = {}
        for idx, (ent, f) in enumerate(zip(entities, findings)):
            is_rdkit_invalid = not ent.smiles_valid
            if (
                f.provenance_status != "mismatch_flagged"
                or not f.expected_smiles
                or (not f.connectivity_conflict and not is_rdkit_invalid)
                or _is_markush_smiles(ent.smiles)
            ):
                continue
            if idx in trust_drawing_reasons:
                reason = trust_drawing_reasons[idx]
                skipped_override_notes[idx] = [
                    f"deterministic override skipped (VLM trusted drawing: {reason})"
                    if reason
                    else "deterministic override skipped (VLM trusted drawing)"
                ]
                continue
            ok, can, _ = validate_and_canonicalize(f.expected_smiles)
            if not ok:
                continue
            if _candidate_violations(can, f):
                continue
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors

            can_mol = Chem.MolFromSmiles(can)
            if can_mol is None or check_chembl_structure(can_mol):
                continue
            candidate_formulas = _finding_expected_formulas(f)
            if candidate_formulas:
                can_formula = rdMolDescriptors.CalcMolFormula(can_mol)
                if not any(
                    formulas_compatible(can_formula, exp_f, context=f.expected_adduct_context)
                    for exp_f in candidate_formulas
                ):
                    continue
            replacements[ent.smiles] = can
            ent.smiles = can
            ent.smiles_valid = True
            ent.rdkit_error = None
            override_notes[idx] = ["overridden from prose IUPAC heading (stereo may be lost)"]
        if override_notes:
            for idx in sorted(override_notes):
                findings[idx] = reverify_entity_with_finding(entities[idx], findings[idx])
            _reappend_notes(entities, override_notes)
        if skipped_override_notes:
            _reappend_notes(entities, skipped_override_notes)

    return entities, replacements
