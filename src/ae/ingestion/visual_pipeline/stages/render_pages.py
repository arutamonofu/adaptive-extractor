from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..paths import PipelinePaths
from ..stage import load_targets, write_validation
from ..utils.json import read_json, write_json


def execute(cfg: dict[str, Any], paths: PipelinePaths) -> dict[str, Any]:
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required. Install with: pip install PyMuPDF") from exc
    targets = load_targets(paths.targets, cfg["run"].get("target_id"), cfg["run"].get("max_targets"))
    if targets:
        pages = sorted({int(page) for target in targets for page in target.get("pages", [])})
    elif paths.md_anchor_index.exists():
        pages = []
    else:
        manifest = read_json(paths.manifest)
        pages = sorted(
            {
                int(page)
                for visual in manifest
                if visual.get("routing_decision") == "extract"
                for page in visual.get("pages", [])
            }
        )
    doc = fitz.open(paths.input_pdf)
    dpi = cfg["runtime"]["dpi"]
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    rendered: list[dict[str, Any]] = []
    for page_num in pages:
        if page_num < 1 or page_num > len(doc):
            continue
        out_path = paths.page_assets / f"page_{page_num:04d}.png"
        if not out_path.exists() or cfg["run"]["force"]:
            pix = doc[page_num - 1].get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(out_path))
            width_px, height_px = pix.width, pix.height
        else:
            with Image.open(out_path) as img:
                width_px, height_px = img.size
        rendered.append(
            {
                "page": page_num,
                "image_path": f"assets/pages/page_{page_num:04d}.png",
                "width_px": width_px,
                "height_px": height_px,
            }
        )
    doc.close()
    return {"dpi": dpi, "pages": rendered}


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    return {"dpi": int(data["dpi"]), "pages": data.get("pages", [])}


def validate(pages: dict[str, Any], service_dir: Path, paths: PipelinePaths) -> None:
    errors: list[str] = []
    if not isinstance(pages.get("pages"), list):
        errors.append("pages:not_list")
    for item in pages.get("pages", []):
        if not isinstance(item, dict):
            errors.append("page:not_object")
            continue
        for key in ("page", "image_path", "width_px", "height_px"):
            if key not in item:
                errors.append(f"page:missing_key:{key}")
        if item.get("image_path"):
            image_path = paths.root / item["image_path"]
            if not image_path.exists():
                errors.append(f"page:missing_image:{image_path}")
    write_validation(service_dir, "ok" if not errors else "fail", errors)


def run(cfg: dict[str, Any], paths: PipelinePaths) -> Path:
    service_dir = paths.service_dir("render_pages")
    service_dir.mkdir(parents=True, exist_ok=True)
    if paths.pages.exists() and not cfg["run"]["force"]:
        validate(read_json(paths.pages), service_dir, paths)
        return paths.pages
    pages = normalize(execute(cfg, paths))
    write_json(paths.pages, pages)
    validate(pages, service_dir, paths)
    return paths.pages
