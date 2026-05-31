from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..paths import PipelinePaths
from ..stage import load_targets, write_validation
from ..utils.json import read_json, write_json


def _short(text: str, limit: int = 180) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _warnings(*items: Any) -> str:
    values: list[str] = []
    for item in items:
        if isinstance(item, list):
            values.extend(str(v) for v in item)
        elif item:
            values.append(str(item))
    return "; ".join(values)


def _skip_reason(row: dict[str, Any]) -> str:
    if row.get("target_created"):
        return ""
    if row.get("routing_decision") != "skip":
        return "not_selected_by_markdown_anchor_gate"
    if not row.get("num"):
        return "no_numeric_chart"
    if row.get("rel") == "unlikely":
        return "low_schema_relevance"
    return "skipped"


def execute(cfg: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    manifest = read_json(paths.manifest)
    current_targets = {
        target["target_id"]: target
        for target in load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
        if isinstance(target, dict) and target.get("target_id")
    }
    rows: list[dict[str, Any]] = []
    status_counts = {"success": 0, "partial": 0, "failed": 0}
    warnings_count = 0
    for idx, visual in enumerate(manifest, start=1):
        tid = f"target_{idx:04d}"
        crop_status = ""
        extraction_status = ""
        crop_path = ""
        overlay_path = ""
        result_path = ""
        extra_warnings: list[str] = []
        is_current_target = tid in current_targets
        if is_current_target and (paths.bboxes / f"{tid}.json").exists():
            overlay_path = (
                f"assets/overlays/{tid}.page_{int(visual.get('pages', [0])[0]):04d}.png"
                if visual.get("pages") else ""
            )
        bbox_report_path = paths.service_dir("locate_bboxes") / f"{tid}.report.json"
        if is_current_target and bbox_report_path.exists():
            bbox_report = read_json(bbox_report_path)
            extra_warnings.extend(bbox_report.get("warnings", []))
        crop_meta_path = paths.crops / f"{tid}.json"
        if is_current_target and crop_meta_path.exists():
            crop_meta = read_json(crop_meta_path)
            crop_status = crop_meta.get("status", "")
            crop_path = crop_meta.get("crop_path", "")
            extra_warnings.extend(crop_meta.get("warnings", []))
        result_json_path = paths.extraction / f"{tid}.json"
        if is_current_target and result_json_path.exists():
            result = read_json(result_json_path)
            extraction_status = result.get("status", "")
            result_path = f"work/extraction/{tid}.json"
            extra_warnings.extend(result.get("warnings", []))
            if extraction_status in status_counts:
                status_counts[extraction_status] += 1
            elif extraction_status in {"crop_failed", "no_extractable_chart"}:
                status_counts["failed"] += 1
        warning_text = _warnings(extra_warnings)
        if warning_text:
            warnings_count += len([w for w in warning_text.split("; ") if w])
        rows.append(
            {
                "target_id": tid,
                "figure": visual.get("figure"),
                "pages": ",".join(str(p) for p in visual.get("pages", [])),
                "caption_short": _short(visual.get("caption", "")),
                "num": visual.get("num"),
                "rel": visual.get("rel"),
                "routing_decision": visual.get("routing_decision"),
                "target_created": is_current_target,
                "crop_status": crop_status,
                "extraction_status": extraction_status,
                "crop_path": crop_path,
                "overlay_path": overlay_path,
                "result_path": result_path,
                "warnings": warning_text,
            }
        )
    total = len(manifest)
    summary = {
        "total_visuals": total,
        "numeric_chart_visuals": sum(1 for item in manifest if item.get("num")),
        "routed_extract": len(current_targets),
        "routed_skip": total - len(current_targets),
        "successful_extractions": status_counts["success"],
        "partial_extractions": status_counts["partial"],
        "failed_extractions": status_counts["failed"],
        "warnings_count": warnings_count,
    }
    return {"rows": rows, "summary": summary}


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    return data


def validate(report: dict[str, Any], service_dir: Path) -> None:
    errors: list[str] = []
    if not isinstance(report.get("rows"), list):
        errors.append("report:rows_not_list")
    if not isinstance(report.get("summary"), dict):
        errors.append("report:summary_not_object")
    write_validation(service_dir, "ok" if not errors else "fail", errors)


def run(cfg: dict[str, Any], paths: PipelinePaths) -> Path:
    service_dir = paths.service_dir("report")
    service_dir.mkdir(parents=True, exist_ok=True)
    report = normalize(execute(cfg, paths))
    csv_path = service_dir / "visual_review.csv"
    rows = report["rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    write_json(service_dir / "pipeline_summary.json", report["summary"])
    routed = [row for row in rows if row["target_created"]]
    skipped = [row for row in rows if not row["target_created"]]
    lines = [
        "# Visual Extraction Report",
        "",
        "## Summary",
        f"- Total visual objects: {report['summary']['total_visuals']}",
        f"- Numeric chart objects: {report['summary']['numeric_chart_visuals']}",
        f"- Routed for extraction: {report['summary']['routed_extract']}",
        f"- Skipped: {report['summary']['routed_skip']}",
        f"- Successful extractions: {report['summary']['successful_extractions']}",
        f"- Partial extractions: {report['summary']['partial_extractions']}",
        f"- Failed extractions: {report['summary']['failed_extractions']}",
        "",
        "## Routed Targets",
    ]
    lines.extend(
        (
            f"- {row['target_id']} {row['figure']}: {row['extraction_status'] or 'pending'}"
        )
        for row in routed
    )
    lines.extend(["", "## Skipped Visuals"])
    lines.extend(f"- {row['target_id']} {row['figure']}: {_skip_reason(row)}" for row in skipped)
    lines.extend(["", "## Warnings"])
    warning_rows = [row for row in rows if row["warnings"]]
    lines.extend(f"- {row['target_id']} {row['figure']}: {row['warnings']}" for row in warning_rows)
    if not warning_rows:
        lines.append("- None")
    lines.extend(["", "## Artifacts", "- `visual_review.csv`", "- `pipeline_summary.json`"])
    (service_dir / "visual_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate(report, service_dir)
    return service_dir / "visual_report.md"
