from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def norm_to_px(bbox_norm: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox_norm
    return [
        round(x0 * width),
        round(y0 * height),
        round(x1 * width),
        round(y1 * height),
    ]


def clip_bbox_px(bbox: list[int], width: int, height: int) -> tuple[list[int], bool]:
    x0, y0, x1, y1 = bbox
    clipped = [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]
    return clipped, clipped != bbox


def draw_overlay(page_path: Path, bbox_px: list[int], out_path: Path) -> None:
    image = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for offset in range(4):
        box = [bbox_px[0] - offset, bbox_px[1] - offset, bbox_px[2] + offset, bbox_px[3] + offset]
        draw.rectangle(box, outline=(255, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def draw_bbox_overlay(page_path: Path, raw_bbox_px: list[int], margin_bbox_px: list[int], out_path: Path) -> None:
    image = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for offset in range(4):
        margin_box = [
            margin_bbox_px[0] - offset,
            margin_bbox_px[1] - offset,
            margin_bbox_px[2] + offset,
            margin_bbox_px[3] + offset,
        ]
        raw_box = [
            raw_bbox_px[0] - offset,
            raw_bbox_px[1] - offset,
            raw_bbox_px[2] + offset,
            raw_bbox_px[3] + offset,
        ]
        draw.rectangle(margin_box, outline=(255, 0, 0))
        draw.rectangle(raw_box, outline=(0, 160, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
