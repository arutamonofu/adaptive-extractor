from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..paths import PipelinePaths
from ..stage import load_targets, write_validation
from ..utils.image import clip_bbox_px, draw_bbox_overlay, norm_to_px
from ..utils.json import read_json, write_json

CROP_SCHEMA_VERSION = "aee.figure_crop.v2"


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized.setdefault("schema_version", CROP_SCHEMA_VERSION)
    normalized.setdefault("target_id", "")
    normalized.setdefault("crop_path", "")
    normalized.setdefault("pages", [])
    normalized.setdefault("parts", [])
    normalized.setdefault("crop_width_px", 0)
    normalized.setdefault("crop_height_px", 0)
    normalized.setdefault("crop_area_ratio", 0)
    normalized.setdefault("warnings", [])
    normalized.setdefault("status", "fail" if normalized.get("failed") else "ok")
    normalized.pop("failed", None)
    return normalized


def _crop_parts(
    parts: list[dict[str, Any]],
    paths: PipelinePaths,
    tid: str,
) -> dict[str, Any]:
    part_crop_paths: list[Path] = []
    part_meta: list[dict[str, Any]] = []
    warnings_all: list[str] = []
    failed = False
    for idx, part in enumerate(parts, start=1):
        page = int(part["page"])
        bbox_norm = part.get("bbox_with_margin", {}).get("norm") or part.get("bbox", {}).get("norm") or []
        warnings: list[str] = [str(warning) for warning in part.get("warnings", [])]
        image_path = paths.page_assets / f"page_{page:04d}.png"
        if len(bbox_norm) != 4:
            warnings.append("invalid_or_missing_bbox")
            warnings_all.extend(warnings)
            failed = True
            continue
        with Image.open(image_path).convert("RGB") as img:
            bbox_px = norm_to_px([float(v) for v in bbox_norm], img.width, img.height)
            clipped, was_clipped = clip_bbox_px(bbox_px, img.width, img.height)
            if was_clipped:
                warnings.append("bbox_clipped_to_page")
            x0, y0, x1, y1 = clipped
            if x1 <= x0 or y1 <= y0:
                warnings.append("empty_crop")
                warnings_all.extend(warnings)
                failed = True
                continue
            area_ratio = ((x1 - x0) * (y1 - y0)) / (img.width * img.height)
            if area_ratio < 0.01:
                warnings.append("crop_area_ratio_lt_0_01")
            if area_ratio > 0.95:
                warnings.append("crop_area_ratio_gt_0_95")
            if x0 == 0 or y0 == 0 or x1 == img.width or y1 == img.height:
                warnings.append("crop_touches_page_boundary")
            part_path = paths.crop_assets / f"{tid}.part_{idx:04d}.png"
            img.crop((x0, y0, x1, y1)).save(str(part_path))
        part_crop_paths.append(part_path)
        meta = {
            "page": page,
            "bbox": part.get("bbox"),
            "bbox_with_margin": part.get("bbox_with_margin"),
            "bbox_px": clipped,
            "crop_path": f"assets/crops/{part_path.name}",
            "crop_width_px": x1 - x0,
            "crop_height_px": y1 - y0,
            "crop_area_ratio": area_ratio,
            "warnings": warnings,
        }
        part_meta.append(meta)
        warnings_all.extend(warnings)

    crop_path = paths.crop_assets / f"{tid}.png"
    if len(part_crop_paths) > 1:
        width = max(part["crop_width_px"] for part in part_meta)
        gap_px = 40
        height = sum(part["crop_height_px"] for part in part_meta) + gap_px * (len(part_crop_paths) - 1)
        composite = Image.new("RGB", (width, height), "white")
        y = 0
        for path in part_crop_paths:
            with Image.open(path).convert("RGB") as part_img:
                composite.paste(part_img, (0, y))
                y += part_img.height + gap_px
        composite.save(str(crop_path))
    elif part_crop_paths:
        part_crop_paths[0].replace(crop_path)
        width, height = part_meta[0]["crop_width_px"], part_meta[0]["crop_height_px"]
        part_meta[0]["crop_path"] = f"assets/crops/{crop_path.name}"
    else:
        width, height = 0, 0
        failed = True

    return {
        "crop_path": f"assets/crops/{crop_path.name}",
        "parts": part_meta,
        "crop_width_px": width,
        "crop_height_px": height,
        "crop_area_ratio": sum(part.get("crop_area_ratio", 0) for part in part_meta),
        "warnings": warnings_all,
        "failed": failed,
    }


def _write_overlay_for_part(paths: PipelinePaths, target_id: str, part: dict[str, Any]) -> None:
    page = int(part.get("page", 0))
    image_path = paths.page_assets / f"page_{page:04d}.png"
    raw_bbox = part.get("bbox", {}).get("px")
    margin_bbox = part.get("bbox_px")
    if not image_path.exists():
        return
    if not (isinstance(raw_bbox, list) and isinstance(margin_bbox, list) and len(raw_bbox) == 4 and len(margin_bbox) == 4):
        return
    overlay_name = f"{target_id}.page_{page:04d}.png"
    draw_bbox_overlay(
        image_path,
        [int(v) for v in raw_bbox],
        [int(v) for v in margin_bbox],
        paths.overlays / overlay_name,
    )


def _write_overlays(paths: PipelinePaths, result: dict[str, Any]) -> None:
    parts = result.get("parts", []) if isinstance(result.get("parts", []), list) else []
    for part in parts:
        if isinstance(part, dict):
            _write_overlay_for_part(paths, result.get("target_id", ""), part)


def execute(
    cfg: dict[str, Any],
    paths: PipelinePaths,
    targets_override: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    targets = (
        targets_override
        if targets_override is not None
        else load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
    )
    for target in targets:
        tid = target["target_id"]
        bbox_path = paths.bboxes / f"{tid}.json"
        if not bbox_path.exists():
            results.append({"target_id": tid, "parts": [], "warnings": ["missing_bbox"], "failed": True})
            continue
        bbox_data = read_json(bbox_path)
        whole_crop = _crop_parts(bbox_data.get("parts", []), paths, tid)
        if whole_crop["failed"]:
            results.append(
                {
                    "schema_version": CROP_SCHEMA_VERSION,
                    "target_id": tid,
                    "crop_path": whole_crop["crop_path"],
                    "pages": target.get("pages", []),
                    "parts": whole_crop["parts"],
                    "crop_width_px": whole_crop["crop_width_px"],
                    "crop_height_px": whole_crop["crop_height_px"],
                    "crop_area_ratio": whole_crop["crop_area_ratio"],
                    "status": "fail",
                    "warnings": whole_crop["warnings"],
                }
            )
            continue

        warnings_all = list(whole_crop["warnings"])

        results.append(
            {
                "schema_version": CROP_SCHEMA_VERSION,
                "target_id": tid,
                "crop_path": whole_crop["crop_path"],
                "pages": target.get("pages", []),
                "parts": whole_crop["parts"],
                "crop_width_px": whole_crop["crop_width_px"],
                "crop_height_px": whole_crop["crop_height_px"],
                "crop_area_ratio": whole_crop["crop_area_ratio"],
                "status": "warn" if warnings_all else "ok",
                "warnings": warnings_all,
            }
        )
    return results


def normalize(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_result(item) for item in data]


def validate(results: list[dict[str, Any]], service_dir: Path, paths: PipelinePaths) -> None:
    errors: list[str] = []
    for result in results:
        if result.get("status") == "fail":
            errors.append(f"{result.get('target_id')}:crop_failed")
        crop_path = paths.root / result.get("crop_path", "")
        if not crop_path.exists():
            errors.append(f"{result.get('target_id')}:missing_crop:{crop_path}")
    write_validation(service_dir, "ok" if not errors else "fail", errors)


def run(cfg: dict[str, Any], paths: PipelinePaths) -> list[Path]:
    service_dir = paths.service_dir("crop_figures")
    service_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
    expected = [paths.crops / f"{target['target_id']}.json" for target in targets]

    cached_results: dict[str, dict[str, Any]] = {}
    targets_to_execute: list[dict[str, Any]] = []
    if cfg["run"]["force"]:
        targets_to_execute = targets
    else:
        for target, out_path in zip(targets, expected, strict=True):
            if out_path.exists():
                cached = read_json(out_path)
                if cached.get("schema_version") == CROP_SCHEMA_VERSION:
                    cached_results[target["target_id"]] = cached
                    continue
            targets_to_execute.append(target)

    if not targets_to_execute:
        results = normalize([cached_results[target["target_id"]] for target in targets])
        for result in results:
            _write_overlays(paths, result)
        validate(results, service_dir, paths)
        return expected

    executed_results = normalize(execute(cfg, paths, targets_to_execute))
    results_by_id = {**cached_results, **{result["target_id"]: result for result in executed_results}}
    out_paths: list[Path] = []
    for result in executed_results:
        _write_overlays(paths, result)
        out_path = paths.crops / f"{result['target_id']}.json"
        write_json(out_path, result)
        write_json(
            service_dir / f"{result['target_id']}.report.json",
            {
                "target_id": result.get("target_id", ""),
                "status": result.get("status", "fail"),
                "warnings": result.get("warnings", []),
                "parts": result.get("parts", []),
            },
        )
        out_paths.append(out_path)
    results = normalize([results_by_id[target["target_id"]] for target in targets])
    for result in cached_results.values():
        _write_overlays(paths, result)
    validate(results, service_dir, paths)
    return expected
