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
]

def setup_logging() -> logging.Logger:
    """
    Configures the global logging state for the application.

    Sets the root logger level based on configuration, defines a standard format,
    and silences/synchronizes specific third-party libraries to match the application level.

    Returns:
        The main application logger ('aee').
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    for lib in _NOISY_LIBRARIES:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(log_level)
        
        # Remove library-specific handlers to prevent duplicate/unformatted output
        if lib_logger.hasHandlers():
            lib_logger.handlers.clear()
        
        lib_logger.propagate = True

    return logging.getLogger("aee")