from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy
import yaml

from ..config import build_schema, model_options
from ..model_client import generate_parsed_json, get_model_client
from ..paths import PipelinePaths
from ..stage import load_targets, write_validation
from ..utils.json import read_json, write_json


def panel_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    while text.startswith("(") and text.endswith(")") and len(text) > 2:
        text = text[1:-1].strip()
    return text


def panel_is_relevant(panel: dict[str, Any]) -> bool:
    return panel.get("num") is True and str(panel.get("rel", "")).strip().lower() != "unlikely"


def relevant_panel_keys(target: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for panel in target.get("panels", []):
        if not isinstance(panel, dict):
            continue
        key = panel_key(panel.get("panel"))
        if key and panel_is_relevant(panel) and key not in keys:
            keys.append(key)
    return keys


def _allowed_panel_keys(target: dict[str, Any]) -> list[str] | None:
    allowed = relevant_panel_keys(target)
    return allowed or None


def _filter_tables_by_panel_metadata(parsed: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    allowed_panels = _allowed_panel_keys(target)
    if allowed_panels is None:
        return parsed
    allowed_panel_set = set(allowed_panels)

    tables = parsed.get("tables")
    if not isinstance(tables, list):
        parsed["tables"] = []
        warnings_val = parsed.get("warnings")
        warnings: list[str] = [str(w) for w in warnings_val] if isinstance(warnings_val, list) else []
        warnings.append("tables_not_list")
        warnings.extend(f"missing_extractable_panel:{panel}" for panel in allowed_panels)
        parsed["warnings"] = warnings
        if parsed.get("status") in {"success", "partial"}:
            parsed["status"] = "no_extractable_chart"
        return parsed

    kept: list[Any] = []
    warnings_val = parsed.get("warnings")
    warnings = [str(w) for w in warnings_val] if isinstance(warnings_val, list) else []
    covered_panels: set[str] = set()
    for table in tables:
        if not isinstance(table, dict):
            warnings.append("dropped_table_not_object")
            continue
        panel = table.get("panel")
        key = panel_key(panel)
        # If there is exactly one allowed panel, and the parsed table doesn't specify a valid panel,
        # auto-assign it to the allowed panel to prevent discarding relevant data.
        if (panel is None or key == "") and len(allowed_panels) == 1:
            warnings.append(f"auto_assigned_table_to_single_panel:{allowed_panels[0]}")
            table["panel"] = allowed_panels[0]
            panel = allowed_panels[0]
            key = panel_key(panel)

        if panel is None or key not in allowed_panel_set:
            warnings.append(f"dropped_table_disallowed_panel:{panel}")
            continue
        covered_panels.add(key)
        kept.append(table)

    missing_panels = [panel for panel in allowed_panels if panel not in covered_panels]
    warnings.extend(f"missing_extractable_panel:{panel}" for panel in missing_panels)

    parsed["tables"] = kept
    parsed["warnings"] = warnings
    if not kept and parsed.get("status") in {"success", "partial"}:
        parsed["status"] = "no_extractable_chart"
    elif kept and missing_panels and parsed.get("status") == "success":
        parsed["status"] = "partial"
    return parsed


def _crop_failed_result(target: dict[str, Any], crop: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "aee.chart_table_extraction.v1",
        "target_id": target["target_id"],
        "figure": target.get("figure"),
        "status": "crop_failed",
        "source": {
            "crop_path": crop.get("crop_path"),
            "caption": target.get("caption", ""),
            "rel": target.get("rel", "unlikely"),
        },
        "tables": [],
        "warnings": crop.get("warnings", []),
        "model_notes": "Extraction skipped because crop status was fail.",
    }


def execute(
    cfg: dict[str, Any],
    paths: PipelinePaths,
    targets_override: list[dict[str, Any]] | None = None,
    lm: dspy.LM | None = None,
) -> list[dict[str, Any]]:
    opts = model_options(cfg, "chart_extraction")
    task_config = yaml.safe_load(paths.input_task_config.read_text(encoding="utf-8")) or {}
    schema = build_schema(task_config)
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    template = (prompts_dir / "prompt_extract.txt").read_text(encoding="utf-8")
    client = get_model_client(opts["provider"], lm=lm)
    targets = (
        targets_override
        if targets_override is not None
        else load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
    )
    outputs: list[dict[str, Any]] = []
    for target in targets:
        tid = target["target_id"]
        crop_path = paths.crops / f"{tid}.json"
        if not crop_path.exists():
            outputs.append({"target": target, "result": _crop_failed_result(target, {"warnings": ["missing_crop"]})})
            continue
        crop = read_json(crop_path)
        image_path = paths.root / crop.get("crop_path", "")
        if crop.get("status") == "fail" or not image_path.exists():
            outputs.append({"target": target, "result": _crop_failed_result(target, crop)})
            continue
        prompt = (
            template.replace("{{TARGET_JSON}}", json.dumps(target, ensure_ascii=False, indent=2))
            .replace("{{SCHEMA}}", json.dumps(schema, ensure_ascii=False, indent=2))
            .replace("{{CROP_METADATA_JSON}}", json.dumps(crop, ensure_ascii=False, indent=2))
        )
        try:
            raw, parsed, parse_warnings = generate_parsed_json(
                client,
                raw_response_path=paths.service_dir("extract_chart_tables") / f"{tid}.raw_response.txt",
                prompt=prompt,
                model=opts["model"],
                files=[image_path],
                temperature=opts["temperature"],
                max_output_tokens=opts["max_output_tokens"],
                thinking_level=opts["thinking_level"],
                thinking_budget=opts["thinking_budget"],
            )
            if not isinstance(parsed, dict):
                raise ValueError("chart extraction response is not a JSON object")
            parsed.setdefault("schema_version", "aee.chart_table_extraction.v1")
            parsed.setdefault("target_id", tid)
            parsed.setdefault("figure", target.get("figure"))
            parsed.setdefault("source", {})
            if isinstance(parsed["source"], dict):
                parsed["source"].setdefault("crop_path", crop.get("crop_path"))
            parsed.setdefault("tables", [])
            parsed.setdefault("warnings", [])
            if isinstance(parsed["warnings"], list):
                parsed["warnings"].extend(parse_warnings)
            parsed = _filter_tables_by_panel_metadata(parsed, target)
            outputs.append({"target": target, "result": parsed, "raw_response": raw})
        except Exception as exc:
            outputs.append(
                {
                    "target": target,
                    "result": {
                        "schema_version": "aee.chart_table_extraction.v1",
                        "target_id": tid,
                        "figure": target.get("figure"),
                        "status": "failed",
                        "error": {"type": exc.__class__.__name__, "message": str(exc)},
                        "tables": [],
                        "warnings": [],
                    },
                }
            )
    return outputs


def normalize(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item["result"] for item in data]


def validate(results: list[dict[str, Any]], service_dir: Path) -> None:
    errors: list[str] = []
    for result in results:
        for key in ("schema_version", "target_id", "figure", "status", "tables", "warnings"):
            if key not in result:
                errors.append(f"{result.get('target_id', 'result')}:missing_key:{key}")
        if not isinstance(result.get("tables"), list):
            errors.append(f"{result.get('target_id')}:tables_not_list")
        if not isinstance(result.get("warnings"), list):
            errors.append(f"{result.get('target_id')}:warnings_not_list")
    write_validation(service_dir, "ok" if not errors else "fail", errors)


def run(cfg: dict[str, Any], paths: PipelinePaths, lm: dspy.LM | None = None) -> list[Path]:
    service_dir = paths.service_dir("extract_chart_tables")
    service_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
    expected = [paths.extraction / f"{target['target_id']}.json" for target in targets]

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
    executed_results = normalize(executed)
    results_by_id = {**cached_results, **{result["target_id"]: result for result in executed_results}}
    out_paths: list[Path] = []
    for item, result in zip(executed, executed_results, strict=True):
        tid = result["target_id"]
        out_path = paths.extraction / f"{tid}.json"
        write_json(out_path, result)
        if item.get("raw_response"):
            (service_dir / f"{tid}.raw_response.txt").write_text(item["raw_response"], encoding="utf-8")
        write_json(
            service_dir / f"{tid}.report.json",
            {
                "target_id": tid,
                "status": result.get("status"),
                "warnings": result.get("warnings", []),
                "raw_response_path": f"service/extract_chart_tables/{tid}.raw_response.txt"
                if item.get("raw_response")
                else None,
            },
        )
        out_paths.append(out_path)
    results = [results_by_id[target["target_id"]] for target in targets]
    validate(results, service_dir)
    return expected
