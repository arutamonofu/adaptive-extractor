from __future__ import annotations

from pathlib import Path
from typing import Any

from ..anchor_utils import canonical_visual_ref
from ..paths import PipelinePaths
from ..stage import write_validation
from ..utils.json import read_json, write_json


def _load_anchor_index(paths: PipelinePaths) -> dict[str, Any]:
    if not paths.md_anchor_index.exists():
        return {"index": {}, "duplicates": [], "anchors": [], "invalid_blocks": []}
    data = read_json(paths.md_anchor_index)
    return data if isinstance(data, dict) else {"index": {}, "duplicates": [], "anchors": [], "invalid_blocks": []}


def execute(cfg: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    manifest = read_json(paths.manifest)
    anchor_report = _load_anchor_index(paths)
    anchor_index = anchor_report.get("index", {}) if isinstance(anchor_report.get("index"), dict) else {}
    duplicate_refs = {
        item.get("visual_ref")
        for item in anchor_report.get("duplicates", [])
        if isinstance(item, dict) and item.get("visual_ref")
    }
    targets: list[dict[str, Any]] = []
    skipped_no_anchor: list[dict[str, Any]] = []
    skipped_duplicate_anchor: list[dict[str, Any]] = []
    manifest_refs: set[str] = set()
    for idx, visual in enumerate(manifest, start=1):
        visual_ref = canonical_visual_ref(visual.get("figure"))
        if visual_ref:
            manifest_refs.add(visual_ref)
        if visual.get("routing_decision") != "extract":
            continue
        if visual_ref in duplicate_refs:
            skipped_duplicate_anchor.append({"figure": visual.get("figure"), "visual_ref": visual_ref})
            continue
        anchor = anchor_index.get(visual_ref)
        if not anchor:
            skipped_no_anchor.append({"figure": visual.get("figure"), "visual_ref": visual_ref})
            continue
        target_id = f"target_{idx:04d}"
        targets.append(
            {
                "target_id": target_id,
                "figure": visual.get("figure"),
                "visual_ref": visual_ref,
                "routing_decision": visual.get("routing_decision"),
                "md_anchor": {
                    "raw_label": anchor.get("raw_label"),
                    "visual_ref": anchor.get("visual_ref"),
                },
                "pages": visual.get("pages", []),
                "caption": visual.get("caption", ""),
                "num": visual.get("num", False),
                "rel": visual.get("rel", "unlikely"),
                "panels": visual.get("panels", []),
                "page_images": [
                    {"page": int(page), "image_path": f"assets/pages/page_{int(page):04d}.png"}
                    for page in visual.get("pages", [])
                ],
            }
        )
    anchor_refs = set(anchor_index)
    return {
        "targets": targets,
        "report": {
            "schema_version": "aee.md_anchor_gate_report.v1",
            "anchors_found": list(anchor_report.get("anchors", [])),
            "duplicate_anchors": list(anchor_report.get("duplicates", [])),
            "invalid_anchors": list(anchor_report.get("invalid_blocks", [])),
            "targets_created": [
                {
                    "target_id": target["target_id"],
                    "figure": target.get("figure"),
                    "visual_ref": target.get("visual_ref"),
                    "anchor_raw_label": target.get("md_anchor", {}).get("raw_label"),
                }
                for target in targets
            ],
            "visuals_skipped_no_anchor": skipped_no_anchor,
            "visuals_skipped_duplicate_anchor": skipped_duplicate_anchor,
            "anchors_without_matching_visual": [
                anchor_index[visual_ref] for visual_ref in sorted(anchor_refs - manifest_refs)
            ],
        },
    }


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    return data


def validate(targets: list[dict[str, Any]], service_dir: Path, paths: PipelinePaths) -> None:
    errors: list[str] = []
    for target in targets:
        for key in ("target_id", "figure", "pages", "caption", "num", "rel", "panels", "page_images"):
            if key not in target:
                errors.append(f"{target.get('target_id', 'target')}:missing_key:{key}")
        for page_image in target.get("page_images", []):
            for key in ("page", "image_path"):
                if key not in page_image:
                    errors.append(f"{target.get('target_id')}:page_image_missing_key:{key}")
    write_validation(service_dir, "ok" if not errors else "fail", errors)


def run(cfg: dict[str, Any], paths: PipelinePaths) -> list[Path]:
    service_dir = paths.service_dir("create_targets")
    service_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(paths.targets.glob("target_*.json"))
    if existing and not cfg["run"]["force"]:
        targets = [read_json(path) for path in existing]
        validate(targets, service_dir, paths)
        return existing
    for old in paths.targets.glob("target_*.json"):
        old.unlink()
    data = normalize(execute(cfg, paths))
    targets = data["targets"]
    out_paths: list[Path] = []
    for target in targets:
        out_path = paths.targets / f"{target['target_id']}.json"
        write_json(out_path, target)
        out_paths.append(out_path)
    write_json(service_dir / "anchor_gate_report.json", data["report"])
    validate(targets, service_dir, paths)
    return out_paths
