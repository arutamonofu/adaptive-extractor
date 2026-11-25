"""Stage 04: Parse Visuals (04_parse_visuals).

Parallel visual parsing with type-specific prompts and RDKit validation.
Extracts structured tables, charts, and chemical molecular structures.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Any, Union

from ae.ingestion.parsers.gemini.artifacts import ArtifactManager
from ae.ingestion.parsers.gemini.client import GeminiClient
from ae.ingestion.parsers.gemini.schemas import (
    ChartParseResult,
    Manifest,
    ManifestVisual,
    MolecularParseResult,
    TableParseResult,
)
from ae.ingestion.parsers.gemini.stages.context import extract_local_text_context
from ae.ingestion.parsers.gemini.stages.rdkit_validation import (
    validate_and_correct_entities,
)
from ae.ingestion.parsers.gemini.stages.silent_error_guard import (
    detect_visual_parse_silent_error,
)

logger = logging.getLogger(__name__)

VisualParseResult = Union[
    MolecularParseResult,
    ChartParseResult,
    TableParseResult,
]


def _read_prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / "04_parse_visuals" / name
    return path.read_text(encoding="utf-8")


def compose_parse_prompt(
    visual: ManifestVisual,
    caption: str,
    crops_count: int,
    context: str = "",
) -> tuple[str, type[VisualParseResult]]:
    """Compose the prompt and select the response schema based on visual type."""
    desc = visual.description
    panel = f"Panel {getattr(visual, 'panel_label', '')}" if getattr(visual, 'panel_label', None) else ""
    full_desc = f"{panel}: {desc}".strip(": ")
    ctx_text = context.strip() if context and context.strip() else "(none)"

    target_note = ""
    panel_lbl = getattr(visual, "panel_label", None)
    if panel_lbl:
        target_note = (
            f"\n\nFocus: Extract ONLY the chemical entity/series for label '{panel_lbl}' "
            f"as described in '{desc}'. Do not extract neighboring structures from the wider scheme "
            f"or bare substituent fragments from a shared R-group legend as standalone entities."
        )

    if visual.type == "table":
        base_prompt = _read_prompt("table.txt")
        prompt = (
            base_prompt.replace("{caption}", caption)
            .replace("{description}", full_desc)
            .replace("{context}", ctx_text)
        )

        if crops_count > 1:
            prompt += (
                "\n\nNote: The attached images are sequential page fragments of the SAME "
                "multi-page table. Stitch all rows together into one unified table output. "
                "Do not duplicate column headers."
            )
        return prompt, TableParseResult

    elif visual.type == "chart":
        base_prompt = _read_prompt("chart.txt")
        prompt = (
            base_prompt.replace("{caption}", caption)
            .replace("{description}", full_desc)
            .replace("{context}", ctx_text)
        )
        return prompt, ChartParseResult

    elif visual.type == "molecular_structure":
        base_prompt = _read_prompt("molecular_structure.txt")
        prompt = (
            base_prompt.replace("{caption}", caption)
            .replace("{description}", full_desc)
            .replace("{context}", ctx_text)
        )
        if target_note:
            prompt += target_note
        if crops_count > 1:
            prompt += (
                "\n\nNote: Multiple cropped images are provided for this element "
                "(e.g., core scaffold + separated R-group legend). Use all images together to assemble full structures."
            )
        return prompt, MolecularParseResult

    raise ValueError(f"Unknown visual type: {visual.type}")


def parse_single_visual(
    client: GeminiClient,
    visual: ManifestVisual,
    crop_paths: list[Path],
    caption: str,
    config: Any,
    context: str = "",
) -> VisualParseResult:
    """Execute Gemini parsing for a single visual."""
    prompt, schema_class = compose_parse_prompt(
        visual, caption, len(crop_paths), context=context
    )
    stage_cfg = config.gemini.get_stage_config("parse_visuals", sub_stage=visual.type)

    # Adaptive thinking_level: single-page tables use 'low', multi-page tables keep configured level
    adaptive_tbl = getattr(getattr(config, "execution", None), "adaptive_table_thinking", None)
    if adaptive_tbl is None:
        adaptive_tbl = getattr(getattr(config, "gemini", None), "adaptive_table_thinking", True)
    thinking_level = stage_cfg.thinking_level
    if (
        adaptive_tbl
        and visual.type == "table"
        and len(crop_paths) == 1
        and not getattr(visual, "panel_label", None)
        and str(thinking_level or "").lower() == "medium"
    ):
        thinking_level = "low"

    silent_guard = getattr(getattr(config, "execution", None), "silent_error_guard", None)
    if silent_guard is None:
        silent_guard = getattr(getattr(config, "gemini", None), "silent_error_guard", True)

    contents = list(crop_paths) + [prompt]
    try:
        res = client.generate_structured(
            contents,
            response_schema=schema_class,
            stage_name=f"parse_{visual.type}",
            model=stage_cfg.model,
            temperature=stage_cfg.temperature,
            max_output_tokens=stage_cfg.max_output_tokens,
            media_resolution=stage_cfg.media_resolution,
            thinking_level=thinking_level,
            thinking_budget=stage_cfg.thinking_budget,
        )
    except Exception as exc:
        logger.error(f"Parsing failed for visual '{visual.id}': {exc}")
        if visual.type == "table":
            return TableParseResult(status="failed", content="")
        elif visual.type == "chart":
            return ChartParseResult(status="failed")
        else:
            return MolecularParseResult(status="failed")

    if silent_guard:
        err_reason = detect_visual_parse_silent_error(res, visual.type)
        if err_reason:
            logger.warning(
                f"Silent-Error Guard triggered on visual '{visual.id}' ({err_reason}); "
                f"retrying at temperature=0.5..."
            )
            if hasattr(client, "record_event"):
                client.record_event("silent_error_retries")
            try:
                retry_res = client.generate_structured(
                    contents,
                    response_schema=schema_class,
                    stage_name=f"parse_{visual.type}",
                    model=stage_cfg.model,
                    temperature=0.5,
                    max_output_tokens=stage_cfg.max_output_tokens,
                    media_resolution=stage_cfg.media_resolution,
                    thinking_level=thinking_level,
                    thinking_budget=stage_cfg.thinking_budget,
                )
                if detect_visual_parse_silent_error(retry_res, visual.type) is None:
                    res = retry_res
            except Exception as retry_exc:
                logger.warning(f"Silent-Error Guard retry failed for '{visual.id}': {retry_exc}")

    # Step 4: RDKit validation & semantic OCSR oracle self-correction for chemical structures
    if isinstance(res, MolecularParseResult) and res.entities:
        logger.info(f"Validating {len(res.entities)} chemical entities via RDKit for '{visual.id}'…")
        mol_cfg = getattr(config.gemini, "molecules", None) if hasattr(config, "gemini") else None
        max_attempts = getattr(mol_cfg, "max_retries", 1) if mol_cfg is not None else 1
        corrected_entities, _ = validate_and_correct_entities(
            entities=res.entities,
            crop_paths=crop_paths,
            client=client,
            config=config,
            stage_model=stage_cfg.model,
            max_attempts=max_attempts,
            stage_name="rdkit_correction",
            context=context,
        )
        res.entities = corrected_entities

    return res


def run_parse_visuals(
    client: GeminiClient,
    manifest: Manifest,
    crops: dict[str, list[Path]],
    artifacts: ArtifactManager,
    config: Any,
) -> dict[str, VisualParseResult]:
    """Execute Stage 04: parse data from cropped visuals in parallel."""
    results: dict[str, VisualParseResult] = {}
    pending_visuals: list[ManifestVisual] = []

    # Check cached parse results
    for vis in manifest.visuals:
        cached_file = artifacts.parsed_visuals_dir / f"{vis.id}.json"
        if cached_file.exists():
            try:
                raw_data = artifacts.load_json(cached_file)
                _, schema_cls = compose_parse_prompt(vis, "", 1)
                results[vis.id] = schema_cls.model_validate(raw_data)
                continue
            except Exception as e:
                logger.warning(f"Could not load cached parsed result for '{vis.id}': {e}")
        pending_visuals.append(vis)

    if not pending_visuals:
        logger.info("Stage 04 [parse_visuals]: All visuals already parsed or none present.")
        return results

    # Build caption lookup from Stage 1 placeholders and optional text context
    caption_map: dict[str, str] = {}
    if artifacts.placeholders_json_path.exists():
        raw_ph = artifacts.load_json(artifacts.placeholders_json_path)
        caption_map = {item["id"]: item.get("caption", "") for item in raw_ph}

    source_md = ""
    if artifacts.text_md_path.exists():
        source_md = artifacts.load_text(artifacts.text_md_path)

    max_workers = getattr(config.gemini, "max_concurrent_requests", 4)
    logger.info(
        f"Stage 04 [parse_visuals]: Parsing {len(pending_visuals)} visual(s) in parallel (workers={max_workers})…",
    )

    failed_visuals: dict[str, Exception] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {}
        for vis in pending_visuals:
            crop_list = crops.get(vis.id, [])
            if not crop_list:
                logger.warning(f"No crop images found for visual '{vis.id}'. Marking as failed.")
                results[vis.id] = (
                    TableParseResult(status="failed", content="")
                    if vis.type == "table"
                    else ChartParseResult(status="failed")
                    if vis.type == "chart"
                    else MolecularParseResult(status="failed")
                )
                continue

            cap = caption_map.get(vis.id, "")
            vis_ctx = (
                extract_local_text_context(
                    source_md, vis.id, caption_map=caption_map, description=vis.description
                )
                if source_md
                else ""
            )
            future = executor.submit(
                parse_single_visual,
                client,
                vis,
                crop_list,
                cap,
                config,
                vis_ctx,
            )
            future_to_id[future] = vis.id

        for future in concurrent.futures.as_completed(future_to_id):
            v_id = future_to_id[future]
            try:
                res = future.result()
                results[v_id] = res
                artifacts.save_json(artifacts.parsed_visuals_dir / f"{v_id}.json", res)
            except Exception as exc:
                logger.error(f"Parsing failed for visual '{v_id}': {exc}")
                failed_visuals[v_id] = exc

    # Retry failed visuals sequentially once
    if failed_visuals:
        logger.warning(f"Retrying parsing on {len(failed_visuals)} failed visual(s) sequentially…")
        vis_map = {v.id: v for v in pending_visuals}
        for v_id in list(failed_visuals.keys()):
            vis = vis_map.get(v_id)
            if not vis:
                continue
            crop_list = crops.get(v_id, [])
            if not crop_list:
                continue
            cap = caption_map.get(v_id, "")
            vis_ctx = (
                extract_local_text_context(
                    source_md, vis.id, caption_map=caption_map, description=vis.description
                )
                if source_md
                else ""
            )
            time.sleep(2.0)
            try:
                res = parse_single_visual(client, vis, crop_list, cap, config, context=vis_ctx)
                results[v_id] = res
                artifacts.save_json(artifacts.parsed_visuals_dir / f"{v_id}.json", res)
                del failed_visuals[v_id]
            except Exception as retry_exc:
                logger.error(f"Retry failed for visual '{v_id}': {retry_exc}")
                results[v_id] = (
                    TableParseResult(status="failed", content="")
                    if vis.type == "table"
                    else ChartParseResult(status="failed")
                    if vis.type == "chart"
                    else MolecularParseResult(status="failed")
                )
                artifacts.save_json(artifacts.parsed_visuals_dir / f"{v_id}.json", results[v_id])

    logger.info(f"Stage 04 [parse_visuals]: Completed parsing for {len(results)} visual(s).")
    return results


# Alias for backward compatibility
run_parse = run_parse_visuals
