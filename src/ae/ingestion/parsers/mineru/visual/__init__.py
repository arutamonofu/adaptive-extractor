"""Visual pipeline integration stages for chart extraction and table insertion."""

from __future__ import annotations

from .stages.extract_chart_tables import extract_single_chart
from .stages.insert_visual_tables import replace_image_tags

__all__ = [
    "extract_single_chart",
    "replace_image_tags",
]
