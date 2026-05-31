from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils.json import read_json, write_json


class ValidationError(RuntimeError):
    pass


def target_paths(targets_dir: Path, target_id: str | None = None, max_targets: int | None = None) -> list[Path]:
    paths = sorted(targets_dir.glob("target_*.json"))
    if target_id:
        paths = [path for path in paths if path.stem == target_id]
    if max_targets is not None:
        paths = paths[:max_targets]
    return paths


def load_targets(targets_dir: Path, target_id: str | None = None, max_targets: int | None = None) -> list[dict[str, Any]]:
    return [read_json(path) for path in target_paths(targets_dir, target_id, max_targets)]


def write_validation(service_dir: Path, status: str, errors: list[str], warnings: list[str] | None = None) -> None:
    write_json(
        service_dir / "validation.json",
        {
            "status": status,
            "errors": errors,
            "warnings": warnings or [],
        },
    )


def ok_status(warnings: list[str] | None = None) -> str:
    return "warn" if warnings else "ok"
