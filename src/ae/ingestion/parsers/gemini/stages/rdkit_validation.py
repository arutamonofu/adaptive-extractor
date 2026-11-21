"""RDKit validation and LLM self-correction for extracted SMILES strings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.schemas import ChemicalEntity

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
    if max_attempts is not None:
        max_retries = max_attempts
    else:
        max_retries = getattr(mol_cfg, "max_rdkit_retries", 2) if mol_cfg else 2

    replacements: dict[str, str] = {}
    if not should_validate or not entities:
        return entities, replacements

    # Initial validation pass
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

    # If client is provided and there are invalid SMILES, enter self-correction loop
    if client is not None and max_retries > 0:
        for attempt in range(1, max_retries + 1):
            invalid_items = [
                {"label": e.label, "smiles": e.smiles, "rdkit_error": e.rdkit_error}
                for e in entities
                if not e.smiles_valid
            ]
            if not invalid_items:
                break

            logger.info(
                f"RDKit validation found {len(invalid_items)} invalid SMILES. "
                f"Attempting Gemini self-correction ({attempt}/{max_retries})…",
            )

            items_lines = []
            for inv in invalid_items:
                items_lines.append(
                    f"- Label '{inv['label']}': SMILES '{inv['smiles']}' "
                    f"(Validation Error: {inv['rdkit_error']})"
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

                for ent in entities:
                    if not ent.smiles_valid and ent.label in corr_map:
                        corrected_smiles = corr_map[ent.label]
                        old_smiles = ent.smiles
                        valid, can, err = validate_and_canonicalize(corrected_smiles)
                        if valid:
                            ent.smiles = can
                            ent.smiles_valid = True
                            ent.rdkit_error = None
                            replacements[old_smiles] = can
                        else:
                            ent.smiles = corrected_smiles
                            ent.smiles_valid = False
                            ent.rdkit_error = err
                            if old_smiles != corrected_smiles:
                                replacements[old_smiles] = corrected_smiles
            except Exception as exc:
                logger.warning(f"RDKit self-correction call failed: {exc}")
                break

    return entities, replacements
