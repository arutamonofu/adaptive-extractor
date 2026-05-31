from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy
from PIL import Image

from ..config import model_options
from ..model_client import generate_parsed_json, get_model_client
from ..paths import PipelinePaths
from ..stage import load_targets, write_validation
from ..utils.image import clip_bbox_px, draw_bbox_overlay, norm_to_px
from ..utils.json import read_json, write_json


def validate_bbox_norm(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except (TypeError, ValueError):
        return False
    return 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def _parse_bbox_norm(value: Any) -> list[float] | None:
    if not validate_bbox_norm(value):
        return None
    return [float(v) for v in value]


def _px_to_norm(bbox_px: list[int], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = bbox_px
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _apply_margin(
    bbox_norm: list[float],
    width: int,
    height: int,
    margin_ratio: float,
    margin_px: int | None,
) -> tuple[list[float], list[int], list[int], bool]:
    bbox_px = norm_to_px(bbox_norm, width, height)
    mx = margin_px if margin_px is not None else round(width * margin_ratio)
    my = margin_px if margin_px is not None else round(height * margin_ratio)
    expanded = [bbox_px[0] - mx, bbox_px[1] - my, bbox_px[2] + mx, bbox_px[3] + my]
    clipped, was_clipped = clip_bbox_px(expanded, width, height)
    return _px_to_norm(clipped, width, height), bbox_px, clipped, was_clipped


def execute(
    cfg: dict[str, Any],
    paths: PipelinePaths,
    targets_override: list[dict[str, Any]] | None = None,
    lm: dspy.LM | None = None,
) -> list[dict[str, Any]]:
    opts = model_options(cfg, "bbox")
    client = get_model_client(opts["provider"], lm=lm)
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    template = (prompts_dir / "prompt_bbox.txt").read_text(encoding="utf-8")
    targets = (
        targets_override
        if targets_override is not None
        else load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
    )
    results: list[dict[str, Any]] = []
    service_dir = paths.service_dir("locate_bboxes")
    for target in targets:
        target_result: dict[str, Any] = {
            "target": target,
            "parts": [],
            "report_parts": [],
            "raw": [],
        }
        for page_image in target.get("page_images", []):
            page = int(page_image["page"])
            image_path = paths.root / page_image["image_path"]
            prompt = template.replace("{{TARGET_JSON}}", json.dumps(target, ensure_ascii=False, indent=2))
            raw_path = service_dir / f"{target['target_id']}.page_{page:04d}.raw_response.txt"
            raw, parsed, parse_warnings = generate_parsed_json(
                client,
                raw_response_path=raw_path,
                prompt=prompt,
                model=opts["model"],
                files=[image_path],
                temperature=opts["temperature"],
                max_output_tokens=opts["max_output_tokens"],
                thinking_level=opts["thinking_level"],
                thinking_budget=opts["thinking_budget"],
            )
            target_result["raw"].append({"page": page, "text": raw})
            if not isinstance(parsed, dict):
                parsed = {"bbox_norm": None, "warnings": ["bbox_response_not_object"]}
            if isinstance(parsed.get("parts"), list) and parsed["parts"]:
                part = next((p for p in parsed["parts"] if int(p.get("page", page)) == page), parsed["parts"][0])
            else:
                part = parsed
            bbox_norm = _parse_bbox_norm(part.get("bbox_norm"))
            warnings: list[Any] = part.get("warnings") if isinstance(part.get("warnings"), list) else []
            warnings.extend(parse_warnings)
            if bbox_norm is None:
                target_result["report_parts"].append(
                    {"page": page, "status": "failed", "warnings": warnings + ["bbox_missing_or_null"]}
                )
                continue
            with Image.open(image_path) as img:
                margin_bbox_norm, raw_bbox_px, margin_bbox_px, was_clipped = _apply_margin(
                    bbox_norm,
                    img.width,
                    img.height,
                    cfg["runtime"]["margin_ratio"],
                    cfg["runtime"]["bbox_margin_px"],
                )
            if was_clipped:
                warnings.append("bbox_margin_clipped_to_page")
            target_result["parts"].append(
                {
                    "page": page,
                    "bbox": {"norm": bbox_norm, "px": raw_bbox_px},
                    "bbox_with_margin": {"norm": margin_bbox_norm, "px": margin_bbox_px},
                }
            )
            target_result["report_parts"].append(
                {
                    "page": page,
                    "status": "ok" if not warnings else "warn",
                    "margin": {
                        "ratio": cfg["runtime"]["margin_ratio"],
                        "px": cfg["runtime"]["bbox_margin_px"],
                        "clipped_to_page": was_clipped,
                    },
                    "warnings": warnings,
                }
            )
            overlay_name = f"{target['target_id']}.page_{page:04d}.png"
            draw_bbox_overlay(image_path, raw_bbox_px, margin_bbox_px, paths.overlays / overlay_name)
        results.append(target_result)
    return results


def normalize(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in data:
        target = item["target"]
        normalized.append(
            {
                "target_id": target["target_id"],
                "parts": item["parts"],
            }
        )
    return normalized


def validate(results: list[dict[str, Any]], service_dir: Path) -> None:
    errors: list[str] = []
    for result in results:
        if not result.get("target_id"):
            errors.append("bbox:missing_target_id")
        if not isinstance(result.get("parts"), list):
            errors.append(f"{result.get('target_id')}:parts_not_list")
            continue
        if not result["parts"]:
            errors.append(f"{result.get('target_id')}:no_bbox_parts")
        for part in result["parts"]:
            bbox = part.get("bbox", {})
            bbox_with_margin = part.get("bbox_with_margin", {})
            if not validate_bbox_norm(bbox.get("norm")):
                errors.append(f"{result.get('target_id')}:invalid_bbox")
            if not validate_bbox_norm(bbox_with_margin.get("norm")):
                errors.append(f"{result.get('target_id')}:invalid_bbox_with_margin")
    write_validation(service_dir, "ok" if not errors else "fail", errors)


def run(cfg: dict[str, Any], paths: PipelinePaths, lm: dspy.LM | None = None) -> list[Path]:
    service_dir = paths.service_dir("locate_bboxes")
    service_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
    expected = [paths.bboxes / f"{target['target_id']}.json" for target in targets]

    cached_results: dict[str, dict[str, Any]] = {}
    targets_to_execute: list[dict[str, Any]] = []
    if cfg["run"]["force"]:
        targets_to_execute = targets
    else:
        for target, out_path in zip(targets, expected, strict=True):
            if out_path.exists():
                cached_results[target["target_id"]] = read_json(out_path)
            else:
                targets_to_execute.append(target)

    if not targets_to_execute:
        results = [cached_results[target["target_id"]] for target in targets]
        validate(results, service_dir)
        return expected

    executed = execute(cfg, paths, targets_to_execute, lm=lm)
    normalized = normalize(executed)
    results_by_id = {**cached_results, **{result["target_id"]: result for result in normalized}}
    out_paths: list[Path] = []
    for raw_item, result in zip(executed, normalized, strict=True):
        tid = result["target_id"]
        out_path = paths.bboxes / f"{tid}.json"
        write_json(out_path, result)
        write_json(
            service_dir / f"{tid}.report.json",
            {
                "target_id": tid,
                "status": "ok" if result["parts"] else "failed",
                "parts": raw_item["report_parts"],
                "raw_response_paths": [
                    f"service/locate_bboxes/{tid}.page_{entry['page']:04d}.raw_response.txt"
                    for entry in raw_item["raw"]
                ],
                "overlay_paths": [f"assets/overlays/{tid}.page_{entry['page']:04d}.png" for entry in raw_item["raw"]],
                "warnings": [warning for part in raw_item["report_parts"] for warning in part.get("warnings", [])],
            },
        )
        for entry in raw_item["raw"]:
            (service_dir / f"{tid}.page_{entry['page']:04d}.raw_response.txt").write_text(
                entry["text"], encoding="utf-8"
            )
        out_paths.append(out_path)
    results = [results_by_id[target["target_id"]] for target in targets]
    validate(results, service_dir)
    return expected
