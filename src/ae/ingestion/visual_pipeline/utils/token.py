from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

_GEMINI_3_IMAGE_TOKENS = {
    "unspecified": 1120,
    "low": 280,
    "medium": 560,
    "high": 1120,
    "ultra_high": 2240,
}


@dataclass(frozen=True)
class TokenCountResult:
    image_path: Path
    tokens: int
    mode: str
    model: str | None = None
    media_resolution: str | None = None
    width: int | None = None
    height: int | None = None
    detail: str | None = None


def _guess_mime_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _normalize_media_resolution(media_resolution: str | None) -> str:
    value = (media_resolution or "unspecified").strip().lower()
    aliases = {
        "default": "unspecified",
        "unspecified": "unspecified",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "ultra_high": "ultra_high",
        "ultrahigh": "ultra_high",
        "ultra-high": "ultra_high",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported media resolution: {media_resolution}")
    return aliases[value]


def estimate_image_tokens(
    image_path: Path,
    *,
    model: str | None = None,
    media_resolution: str | None = None,
) -> TokenCountResult:
    with Image.open(image_path) as image:
        width, height = image.size

    model_name = (model or "").strip().lower()
    resolution = _normalize_media_resolution(media_resolution)

    if model_name.startswith("gemini-3"):
        tokens = _GEMINI_3_IMAGE_TOKENS[resolution]
        detail = f"gemini-3:{resolution}"
    else:
        max_side = max(width, height)
        if max_side <= 384:
            tokens = 258
            detail = "legacy-fixed-258"
        else:
            tiles_x = -(-width // 768)
            tiles_y = -(-height // 768)
            tokens = tiles_x * tiles_y * 258
            detail = f"legacy-tiled:{tiles_x}x{tiles_y}"

    return TokenCountResult(
        image_path=image_path,
        tokens=tokens,
        mode="estimate",
        model=model,
        media_resolution=resolution,
        width=width,
        height=height,
        detail=detail,
    )


def count_image_tokens_api(
    image_path: Path,
    *,
    model: str,
    api_key: str | None = None,
    prompt: str | None = None,
) -> TokenCountResult:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    contents: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    if prompt:
        parts.append({"text": prompt})

    parts.append(
        {
            "inline_data": {
                "mime_type": _guess_mime_type(image_path),
                "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            }
        }
    )
    contents.append({"role": "user", "parts": parts})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens?key={key}"
    response = requests.post(url, json={"contents": contents}, timeout=180)
    response.raise_for_status()
    payload = response.json()
    tokens = int(payload["totalTokens"])
    return TokenCountResult(
        image_path=image_path,
        tokens=tokens,
        mode="api",
        model=model,
        detail="countTokens",
    )
