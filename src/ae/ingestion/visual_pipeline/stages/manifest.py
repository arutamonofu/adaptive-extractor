from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import dspy
import yaml

from ..config import build_schema, model_options
from ..model_client import generate_parsed_json, get_model_client
from ..paths import PipelinePaths
from ..stage import write_validation
from ..utils.json import read_json, write_json

RELEVANCE_RANK = {"unlikely": 0, "possible": 1, "likely": 2, "direct": 3}
VALID_RELEVANCE = set(RELEVANCE_RANK)


def normalize_relevance(value: object) -> str:
    value_s = str(value or "unlikely").strip().lower()
    return value_s if value_s in VALID_RELEVANCE else "unlikely"


def route_visual(num: bool, rel: str) -> tuple[str, str]:
    if not num:
        return "skip", "no_numeric_chart"
    if normalize_relevance(rel) == "unlikely":
        return "skip", "low_schema_relevance"
    return "extract", "numeric_chart_not_unlikely_for_schema"


def stronger_relevance(left: str, right: str) -> str:
    left_n = normalize_relevance(left)
    right_n = normalize_relevance(right)
    return left_n if RELEVANCE_RANK[left_n] >= RELEVANCE_RANK[right_n] else right_n


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().lower() == "true"


def _pages(value: Any) -> list[int]:
    pages: set[int] = set()
    for item in _as_list(value):
        if str(item).isdigit():
            pages.add(int(item))
    return sorted(pages)


def _figure(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _visual_figure(visual: dict[str, Any], fallback: str) -> str:
    for field in ("figure", "figure_key", "label", "visual_id", "norm", "normalized_label"):
        figure = _figure(visual.get(field))
        if figure:
            return figure
    return fallback


def _panel_route(panel: dict[str, Any]) -> dict[str, Any]:
    num = _bool(panel.get("num"))
    rel = normalize_relevance(panel.get("rel"))
    return {
        "panel": str(panel.get("panel") or panel.get("label") or ""),
        "num": num,
        "rel": rel,
    }


def _aggregate_visual(raw_visual: dict[str, Any], target_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warnings: list[str] = []
    panels = [_panel_route(dict(panel)) for panel in _as_list(raw_visual.get("panels")) if isinstance(panel, dict)]

    fig_num = _bool(raw_visual.get("num"))
    fig_rel = normalize_relevance(raw_visual.get("rel"))

    if panels:
        panel_num = any(panel.get("num") for panel in panels)
        panel_rel = "unlikely"
        for panel in panels:
            panel_rel = stronger_relevance(panel_rel, panel.get("rel", "unlikely"))
        if panel_num and not fig_num:
            warnings.append("figure_num_promoted_from_panels")
            fig_num = fig_num or panel_num
        if RELEVANCE_RANK[panel_rel] > RELEVANCE_RANK[fig_rel]:
            fig_rel = panel_rel
        elif RELEVANCE_RANK[fig_rel] > RELEVANCE_RANK[panel_rel] and panel_num:
            warnings.append("figure_rel_stronger_than_panel_rel")

    decision, _ = route_visual(fig_num, fig_rel)
    figure = _visual_figure(raw_visual, target_id)
    pages = _pages(raw_visual.get("pages"))
    audit_warnings = warnings + _as_list(raw_visual.get("_audit_warnings")) + _as_list(raw_visual.get("warnings"))
    audit = [
        {"level": "warning", "target_id": target_id, "figure": figure, "type": str(warning)}
        for warning in audit_warnings
        if warning
    ]

    return (
        {
            "figure": figure,
            "pages": pages,
            "caption": raw_visual.get("caption") or "",
            "num": fig_num,
            "rel": fig_rel,
            "routing_decision": decision,
            "panels": panels,
        },
        audit,
    )


def postprocess_manifest(raw_manifest: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    if not isinstance(raw_manifest, list):
        return [], [{"level": "error", "type": "invalid_manifest", "message": "raw manifest is not a list"}]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, visual in enumerate(raw_manifest, start=1):
        if not isinstance(visual, dict):
            audit.append({"level": "error", "type": "invalid_visual", "index": idx, "message": "visual is not an object"})
            continue
        groups[_visual_figure(visual, f"raw_visual_{idx:04d}")].append(visual)

    canonical_visuals: list[dict[str, Any]] = []
    for group_visuals in groups.values():
        base = dict(group_visuals[0])
        if len(group_visuals) > 1:
            all_pages: set[int] = set()
            captions: list[str] = []
            panels: list[dict[str, Any]] = []
            warnings = _as_list(base.get("warnings"))
            grouped_num = False
            grouped_rel = "unlikely"
            for item in group_visuals:
                all_pages.update(_pages(item.get("pages")))
                grouped_num = grouped_num or _bool(item.get("num"))
                grouped_rel = stronger_relevance(grouped_rel, normalize_relevance(item.get("rel")))
                if item.get("caption"):
                    captions.append(str(item["caption"]))
                panels.extend(panel for panel in _as_list(item.get("panels")) if isinstance(panel, dict))
            base["pages"] = sorted(all_pages)
            base["caption"] = max(captions, key=len) if captions else base.get("caption", "")
            base["num"] = grouped_num
            base["rel"] = grouped_rel
            base["panels"] = panels or base.get("panels", [])
            base["_audit_warnings"] = warnings + ["grouped_entries_with_same_figure"]
        target_id = f"target_{len(canonical_visuals) + 1:04d}"
        canonical_visual, audit_items = _aggregate_visual(base, target_id)
        canonical_visuals.append(canonical_visual)
        audit.extend(audit_items)

    return canonical_visuals, audit


def execute(cfg: dict[str, Any], paths: PipelinePaths, lm: dspy.LM | None = None) -> dict[str, Any]:
    opts = model_options(cfg, "manifest")
    task_config = yaml.safe_load(paths.input_task_config.read_text(encoding="utf-8")) or {}
    schema = build_schema(task_config)
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    template = (prompts_dir / "prompt_manifest.txt").read_text(encoding="utf-8")
    prompt = template.replace("{{SCHEMA}}", json.dumps(schema, ensure_ascii=False, indent=2)).replace(
        "{{PDF_NAME}}", paths.input_pdf.name
    )
    client = get_model_client(opts["provider"], lm=lm)
    raw, parsed, parse_warnings = generate_parsed_json(
        client,
        raw_response_path=paths.service_dir("manifest") / "raw_response.txt",
        prompt=prompt,
        model=opts["model"],
        files=[paths.input_pdf],
        temperature=opts["temperature"],
        max_output_tokens=opts["max_output_tokens"],
        thinking_level=opts["thinking_level"],
        thinking_budget=opts["thinking_budget"],
    )
    return {"raw_response": raw, "raw_manifest": parsed, "parse_warnings": parse_warnings}


def normalize(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return postprocess_manifest(data["raw_manifest"])


def validate(manifest: list[dict[str, Any]], service_dir: Path) -> None:
    errors: list[str] = []
    for idx, item in enumerate(manifest, start=1):
        if not isinstance(item, dict):
            errors.append(f"manifest[{idx}]:not_object")
            continue
        for key in ("figure", "pages", "caption", "num", "rel", "routing_decision", "panels"):
            if key not in item:
                errors.append(f"manifest[{idx}]:missing_key:{key}")
        if item.get("routing_decision") not in {"extract", "skip"}:
            errors.append(f"manifest[{idx}]:invalid_routing_decision")
    write_validation(service_dir, "ok" if not errors else "fail", errors)


def run(cfg: dict[str, Any], paths: PipelinePaths, lm: dspy.LM | None = None) -> Path:
    service_dir = paths.service_dir("manifest")
    service_dir.mkdir(parents=True, exist_ok=True)
    if paths.manifest.exists() and not cfg["run"]["force"]:
        validate(read_json(paths.manifest), service_dir)
        return paths.manifest
    executed = execute(cfg, paths, lm=lm)
    manifest, audit = normalize(executed)
    write_json(paths.manifest, manifest)
    write_json(service_dir / "raw_manifest.json", executed["raw_manifest"])
    write_json(service_dir / "audit.json", audit)
    if executed.get("parse_warnings"):
        write_json(service_dir / "parse_warnings.json", executed["parse_warnings"])
    (service_dir / "raw_response.txt").write_text(executed["raw_response"], encoding="utf-8")
    validate(manifest, service_dir)
    return paths.manifest
