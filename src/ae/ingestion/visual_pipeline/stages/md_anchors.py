from __future__ import annotations

from pathlib import Path
from typing import Any

from ..anchor_utils import SCHEMA_VERSION, parse_md_anchors
from ..paths import PipelinePaths
from ..stage import write_validation
from ..utils.json import read_json, write_json


def _cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("md_anchors", {}) or {}


def _path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def markdown_path(cfg: dict[str, Any], paths: PipelinePaths) -> Path | None:
    stage_cfg = _cfg(cfg)
    table_cfg = cfg.get("table_insertion", {}) or {}
    configured = _path(stage_cfg.get("markdown")) or _path(table_cfg.get("markdown"))
    if configured is not None:
        return configured
    if paths.input_markdown.exists():
        return paths.input_markdown
    return None


def execute(cfg: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    path = markdown_path(cfg, paths)
    if path is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_markdown": None,
            "anchors": [],
            "index": {},
            "duplicates": [],
            "invalid_blocks": [],
            "warnings": ["markdown_not_configured"],
        }
    report = parse_md_anchors(path.read_text(encoding="utf-8"))
    report["source_markdown"] = str(path)
    report["warnings"] = []
    return report


def validate(report: dict[str, Any], service_dir: Path) -> None:
    errors: list[str] = []
    warnings = list(report.get("warnings", []))
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("anchor_index:invalid_schema_version")
    if not isinstance(report.get("anchors"), list):
        errors.append("anchor_index:anchors_not_list")
    if not isinstance(report.get("index"), dict):
        errors.append("anchor_index:index_not_object")
    if report.get("duplicates"):
        warnings.append("duplicate_anchors")
    if report.get("invalid_blocks"):
        warnings.append("invalid_anchor_blocks")
    write_validation(service_dir, "ok" if not errors else "fail", errors, warnings)


def run(cfg: dict[str, Any], paths: PipelinePaths) -> Path:
    service_dir = paths.service_dir("md_anchors")
    service_dir.mkdir(parents=True, exist_ok=True)
    if paths.md_anchor_index.exists() and not cfg["run"]["force"]:
        report = read_json(paths.md_anchor_index)
        validate(report, service_dir)
        return paths.md_anchor_index
    report = execute(cfg, paths)
    write_json(paths.md_anchor_index, report)
    write_json(service_dir / "md_anchor_report.json", report)
    validate(report, service_dir)
    return paths.md_anchor_index
