from __future__ import annotations

from pathlib import Path
from typing import Any

from ..anchor_utils import canonical_visual_ref, parse_md_anchors
from ..paths import PipelinePaths
from ..stage import write_validation
from ..utils.json import read_json, write_json

SCHEMA_VERSION = "aee.visual_table_insertion.v1"
SUCCESS_STATUSES = {"success", "partial"}


def _cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("table_insertion", {}) or {}


def _path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _display_label(result: dict[str, Any], fallback: str) -> str:
    return str(result.get("figure") or fallback)


def _table_title(table: dict[str, Any], table_count: int) -> str:
    panel = table.get("panel")
    if table_count <= 1 and not panel:
        return ""
    if panel:
        return f"#### Panel {panel}"
    return "#### Whole figure"


def _clean_cell(value: Any) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text.replace("|", r"\|")


def _column_title(column: Any) -> str:
    if isinstance(column, dict):
        name = _clean_cell(column.get("name", ""))
        unit = _clean_cell(column.get("unit", ""))
        return f"{name} ({unit})" if unit else name
    return _clean_cell(column)


def _render_table(table: dict[str, Any], result: dict[str, Any], warnings: list[str]) -> str:
    target_id = result.get("target_id", "result")
    columns = table.get("columns", [])
    rows = table.get("rows", [])
    if not isinstance(columns, list) or not columns:
        warnings.append(f"{target_id}:table_missing_columns")
        return ""
    if not isinstance(rows, list):
        warnings.append(f"{target_id}:table_rows_not_list")
        rows = []
    headers = [_column_title(column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row_idx, row in enumerate(rows, start=1):
        values = row if isinstance(row, list) else [row]
        if len(values) != len(headers):
            warnings.append(f"{target_id}:row_{row_idx}_column_count_mismatch")
        padded = (values + [""] * len(headers))[: len(headers)]
        lines.append("| " + " | ".join(_clean_cell(value) for value in padded) + " |")
    return "\n".join(lines)


def _render_result(result: dict[str, Any], label: str, warnings: list[str]) -> str:
    tables = result.get("tables", [])
    if not isinstance(tables, list) or not tables:
        return ""
    display = _display_label(result, label)
    title = "table" if len(tables) == 1 else "tables"
    lines = [f"### Extracted {title} from {display}", ""]
    rendered_tables: list[str] = []
    for table in tables:
        if not isinstance(table, dict):
            warnings.append(f"{result.get('target_id', 'result')}:table_not_object")
            continue
        table_lines: list[str] = []
        table_title = _table_title(table, len(tables))
        if table_title:
            table_lines.extend([table_title, ""])
        rendered = _render_table(table, result, warnings)
        if rendered:
            table_lines.append(rendered)
        if table_lines:
            rendered_tables.append("\n".join(table_lines))
    if not rendered_tables:
        return ""
    lines.append("\n\n".join(rendered_tables))
    result_warnings = result.get("warnings", [])
    if isinstance(result_warnings, list) and result_warnings:
        lines.extend(["", "", "_Extraction warnings:_"])
        lines.extend(f"- {_clean_cell(warning)}" for warning in result_warnings)
    return "\n".join(lines).rstrip()


def _load_extractions(extraction_dir: Path) -> list[dict[str, Any]]:
    if not extraction_dir.exists():
        return []
    return [read_json(path) for path in sorted(extraction_dir.glob("target_*.json"))]


def _load_targets(targets_dir: Path) -> dict[str, dict[str, Any]]:
    if not targets_dir.exists():
        return {}
    targets: dict[str, dict[str, Any]] = {}
    for path in sorted(targets_dir.glob("target_*.json")):
        target = read_json(path)
        if isinstance(target, dict) and target.get("target_id"):
            targets[str(target["target_id"])] = target
    return targets


def _index_results(
    results: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    warnings: list[str]
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if not isinstance(result, dict):
            warnings.append("extraction_result_not_object")
            continue
        labels = [result.get("visual_ref"), result.get("figure")]
        target = targets.get(str(result.get("target_id", "")))
        if target:
            labels.append(target.get("visual_ref"))
            labels.append(target.get("figure"))
        normalized = {canonical_visual_ref(label) for label in labels}
        normalized.discard("")
        if not normalized:
            warnings.append(f"{result.get('target_id', 'result')}:visual_ref_not_canonicalized")
            continue
        for label in normalized:
            index.setdefault(label, []).append(result)
    return index


def _block_content(body: str) -> str:
    return f"\n{body}\n" if body else ""


def _replace_anchor_blocks(
    markdown: str,
    anchors: list[dict[str, Any]],
    index: dict[str, list[dict[str, Any]]],
    keep_empty_anchor_notes: bool,
    warnings: list[str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], int]:
    inserted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    tables_inserted = 0
    output = markdown
    for anchor in sorted(anchors, key=lambda item: int(item["content_start_offset"]), reverse=True):
        label = anchor["visual_ref"]
        results = index.get(label, [])
        if len(results) > 1:
            warnings.append(f"multiple_extraction_results_for_anchor:{label}")
        result = results[0] if results else None
        if result is None:
            warnings.append(f"missing_extraction_result:{label}")
            skipped.append({"normalized_label": label, "reason": "missing_extraction_result"})
            body = "<!-- No extracted visual tables: missing extraction result -->" if keep_empty_anchor_notes else ""
        elif result.get("status") not in SUCCESS_STATUSES:
            warnings.append(f"{result.get('target_id', label)}:extraction_status:{result.get('status')}")
            skipped.append({"normalized_label": label, "target_id": result.get("target_id"), "reason": result.get("status")})
            body = f"<!-- No extracted visual tables: {result.get('status')} -->" if keep_empty_anchor_notes else ""
        elif not result.get("tables"):
            skipped.append({"normalized_label": label, "target_id": result.get("target_id"), "reason": "no_tables"})
            body = "<!-- No extracted visual tables: no tables -->" if keep_empty_anchor_notes else ""
        else:
            body = _render_result(result, str(anchor.get("raw_label") or label), warnings)
            if body:
                tables_inserted += len(result.get("tables", []))
                inserted.append(
                    {
                        "normalized_label": label,
                        "raw_label": anchor.get("raw_label"),
                        "target_id": result.get("target_id"),
                        "tables": len(result.get("tables", [])),
                    }
                )
            else:
                skipped.append({"normalized_label": label, "target_id": result.get("target_id"), "reason": "empty_render"})
        start = int(anchor["content_start_offset"])
        end = int(anchor["content_end_offset"])
        output = output[:start] + _block_content(body) + output[end:]
    return output, inserted, skipped, tables_inserted


def execute(cfg: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    stage_cfg = _cfg(cfg)
    markdown_path = _path(stage_cfg.get("markdown"))
    if markdown_path is None and paths.input_markdown.exists():
        markdown_path = paths.input_markdown
    if markdown_path is None:
        raise ValueError("table_insertion.markdown is required")
    extraction_dir = _path(stage_cfg.get("extraction_dir")) or paths.extraction
    targets_dir = _path(stage_cfg.get("targets_dir")) or paths.targets
    keep_empty = bool(stage_cfg.get("keep_empty_anchor_notes", False))

    warnings: list[str] = []
    markdown = markdown_path.read_text(encoding="utf-8")
    anchor_report = parse_md_anchors(markdown)
    results = _load_extractions(extraction_dir)
    targets = _load_targets(targets_dir)
    index = _index_results(results, targets, warnings)
    output_markdown, inserted, skipped, tables_inserted = _replace_anchor_blocks(
        markdown,
        list(anchor_report["index"].values()),
        index,
        keep_empty,
        warnings,
    )
    anchor_labels = set(anchor_report["index"])
    for label, matches in sorted(index.items()):
        if label not in anchor_labels:
            warnings.append(f"missing_anchor:{label}")
            for result in matches:
                skipped.append({"normalized_label": label, "target_id": result.get("target_id"), "reason": "missing_anchor"})

    return {
        "schema_version": SCHEMA_VERSION,
        "source_markdown": str(markdown_path),
        "output_markdown": output_markdown,
        "anchors_found": anchor_report["anchors"],
        "duplicate_anchors": anchor_report["duplicates"],
        "invalid_anchors": anchor_report["invalid_blocks"],
        "inserted_tables": inserted,
        "inserted": inserted,
        "skipped": skipped,
        "failed_insertions": [
            item for item in skipped if isinstance(item, dict) and item.get("reason") != "missing_anchor"
        ],
        "warnings": warnings,
        "extraction_results_found": len(results),
        "tables_inserted": tables_inserted,
    }


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    anchors = data.get("anchors_found", [])
    data.get("inserted", [])
    skipped = data.get("skipped", [])
    warnings = data.get("warnings", [])
    summary = {
        "anchors_found": len(anchors) if isinstance(anchors, list) else 0,
        "extraction_results_found": int(data.get("extraction_results_found", 0) or 0),
        "tables_inserted": int(data.get("tables_inserted", 0) or 0),
        "figures_without_tables": [
            item for item in skipped
            if isinstance(item, dict) and item.get("reason") in {
                "no_tables", "failed", "crop_failed", "no_extractable_chart"
            }
        ]
        if isinstance(skipped, list)
        else [],
        "missing_anchors": [
            item for item in skipped if isinstance(item, dict) and item.get("reason") == "missing_anchor"
        ]
        if isinstance(skipped, list)
        else [],
        "missing_extraction_results": [
            item for item in skipped if isinstance(item, dict) and item.get("reason") == "missing_extraction_result"
        ]
        if isinstance(skipped, list)
        else [],
        "warnings": warnings if isinstance(warnings, list) else [],
    }
    normalized = dict(data)
    normalized["summary"] = summary
    return normalized


def validate(report: dict[str, Any], service_dir: Path) -> None:
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("report:invalid_schema_version")
    if not isinstance(report.get("output_markdown"), str):
        errors.append("report:output_markdown_not_string")
    if not isinstance(report.get("anchors_found"), list):
        errors.append("report:anchors_found_not_list")
    if not isinstance(report.get("summary"), dict):
        errors.append("report:summary_not_object")
    write_validation(service_dir, "ok" if not errors else "fail", errors, report.get("warnings", []))


def _write_report_md(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Table Insertion Report",
        "",
        "## Summary",
        f"- Anchors found: {summary['anchors_found']}",
        f"- Extraction results found: {summary['extraction_results_found']}",
        f"- Tables inserted: {summary['tables_inserted']}",
        f"- Figures without tables: {len(summary['figures_without_tables'])}",
        f"- Missing anchors: {len(summary['missing_anchors'])}",
        f"- Missing extraction results: {len(summary['missing_extraction_results'])}",
        f"- Warnings: {len(summary['warnings'])}",
        "",
        "## Inserted",
    ]
    inserted = report.get("inserted", [])
    lines.extend(
        f"- {item.get('target_id')} {item.get('normalized_label')}: {item.get('tables')} table(s)"
        for item in inserted
    )
    if not inserted:
        lines.append("- None")
    lines.extend(["", "## Skipped"])
    skipped = report.get("skipped", [])
    lines.extend(
        f"- {item.get('target_id', '')} {item.get('normalized_label')}: {item.get('reason')}"
        for item in skipped
    )
    if not skipped:
        lines.append("- None")
    lines.extend(["", "## Warnings"])
    warnings = report.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict[str, Any], paths: PipelinePaths) -> list[Path]:
    stage_cfg = _cfg(cfg)
    service_dir = _path(stage_cfg.get("report_dir")) or paths.service_dir("table_insertion")
    out_path = _path(stage_cfg.get("out")) or service_dir / "article.with_visual_tables.md"
    service_dir.mkdir(parents=True, exist_ok=True)
    report = normalize(execute(cfg, paths))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report["output_markdown"], encoding="utf-8")
    write_json(service_dir / "table_insertion_summary.json", report["summary"])
    write_json(service_dir / "table_insertion_report.json", {k: v for k, v in report.items() if k != "output_markdown"})
    _write_report_md(report, service_dir / "table_insertion_report.md")
    validate(report, service_dir)
    return [out_path, service_dir / "table_insertion_report.md", service_dir / "table_insertion_summary.json"]
