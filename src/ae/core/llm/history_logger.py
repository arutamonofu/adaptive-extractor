"""LLM history logging utility."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _clean_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Remove duplicate fields from history entry.

    Args:
        entry: Raw history entry from LLM

    Returns:
        Cleaned entry with duplicates removed
    """
    cleaned: Dict[str, Any] = {
        "messages": entry.get("messages", []),
        "outputs": entry.get("outputs", []),
        "model": entry.get("model", ""),
    }

    # Keep only non-message kwargs
    kwargs = entry.get("kwargs", {})
    extra = {k: v for k, v in kwargs.items() if k != "messages"}
    if extra:
        cleaned["kwargs"] = extra

    return cleaned


def _clean_for_json(obj: Any) -> Any:
    """Recursively remove non-JSON-serializable objects from data structure.

    Args:
        obj: Object to clean (dict, list, or primitive)

    Returns:
        Cleaned object with only JSON-serializable values
    """
    if isinstance(obj, dict):
        return {
            k: _clean_for_json(v)
            for k, v in obj.items()
            if k != 'lm'  # Skip 'lm' field
        }
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert any other type to string (e.g., OllamaLM instances)
        return str(obj)


def save_history(lm: Any, output_path: Path) -> int:
    """Save LLM history to JSON file.

    Uses atomic write (temp file + rename) to prevent corruption
    if the process is interrupted (e.g., KeyboardInterrupt).
    Removes duplicate fields and non-JSON-serializable objects from history entries.

    Args:
        lm: LLM instance with .history attribute
        output_path: Path to output file

    Returns:
        Number of entries saved
    """
    if not hasattr(lm, "history") or not lm.history:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean history: remove duplicates and non-JSON-serializable fields
    clean_history = [_clean_for_json(_clean_entry(entry)) for entry in lm.history]

    # Write to temp file first, then rename atomically
    temp_path = output_path.with_suffix('.json.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(clean_history, f, ensure_ascii=False, indent=2)
        temp_path.replace(output_path)  # Atomic on POSIX
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise

    logger.info(f"Saved {len(clean_history)} LLM calls to {output_path}")
    return len(clean_history)


def save_optimization_history(
    student_lm: Any,
    teacher_lm: Optional[Any],
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> Dict[str, int]:
    """Save student and teacher LLM histories.

    Args:
        student_lm: Student LLM instance
        teacher_lm: Teacher LLM instance (optional)
        output_dir: Directory for output files
        timestamp: Optional timestamp string (default: current time)

    Returns:
        Dict with counts: {"student": N, "teacher": M}
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    counts: Dict[str, int] = {}

    student_count = save_history(
        student_lm,
        output_dir / f"student_lm_{timestamp}.json"
    )
    counts["student"] = student_count

    if teacher_lm is not None:
        teacher_count = save_history(
            teacher_lm,
            output_dir / f"teacher_lm_{timestamp}.json"
        )
        counts["teacher"] = teacher_count

    return counts


def save_extraction_history(
    lm: Any,
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> int:
    """Save extraction LLM history.

    Args:
        lm: LLM instance from extraction
        output_dir: Directory for output files
        timestamp: Optional timestamp string (default: current time)

    Returns:
        Number of entries saved
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"extraction_lm_{timestamp}.json"
    return save_history(lm, output_path)
