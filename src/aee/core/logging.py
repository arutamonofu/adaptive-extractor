# src/aee/core/logging.py

import logging
import sys
from aee.core.config import settings

_NOISY_LIBRARIES = [
    "RapidOCR",
    "rapidocr_onnxruntime",
    "docling",
    "docling.pipeline",
    "pdfminer",
    "urllib3",
    "httpx",
    "httpcore",
    "filelock",
]

def setup_logging() -> logging.Logger:
    """
    Configures application logging.
    - Writes to stderr (separating logs from script output).
    - Suppresses chatter from third-party libraries.
    """
    app_level = settings.log_level.upper()

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

    return logging.getLogger("aee")