# src/ae/core/config/logging.py
"""Logging configuration for Adaptive Extractor."""

import logging
import sys
from typing import Any

_NOISY_LIBRARIES = [
    "RapidOCR",
    "rapidocr_onnxruntime",
    "pdfminer",
    "urllib3",
    "httpx",
    "httpcore",
    "filelock",
    "mlflow",
    "alembic",
]


def setup_logging(config: Any) -> logging.Logger:
    """Configure application logging.

    - Writes to stderr (separating logs from script output).
    - Suppresses chatter from third-party libraries.

    Args:
        config: Settings object containing project.log_level. Required.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        ValueError: If config is not provided.
    """
    if config is None:
        raise ValueError("Configuration object is required for setup_logging()")

    app_level = config.project.log_level.upper()

    logging.basicConfig(
        level=app_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True
    )

    silence_level = max(logging.WARNING, logging.getLogger().getEffectiveLevel())

    for lib in _NOISY_LIBRARIES:
        logging.getLogger(lib).setLevel(silence_level)

    return logging.getLogger("ae")
