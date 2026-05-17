"""Helpers for running repository scripts without installing the package."""

from pathlib import Path
import sys


def add_src_to_path() -> None:
    """Make the src-layout package importable for direct script execution."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
