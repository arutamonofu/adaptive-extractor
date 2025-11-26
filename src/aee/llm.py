# src/aee/llm.py

import time
import logging
import dspy
from threading import Lock
from typing import Any
from aee.core.config import settings

logger = logging.getLogger(__name__)
_API_LOCK = Lock()

def _apply_throttling(lm: dspy.LM, delay: float) -> dspy.LM:
    """
    Wraps the LM call method to enforce thread-safe rate limiting.

    Args:
        lm: The DSPy Language Model instance.
        delay: Delay in seconds between requests.

    Returns:
        The modified LM instance.
    """
    original_call = lm.__call__

    def _throttled(*args, **kwargs):
        with _API_LOCK:
            time.sleep(delay)
            return original_call(*args, **kwargs)

    lm.__call__ = _throttled
    return lm

def create_lm(model: str, api_key: str, **kwargs: Any) -> dspy.LM:
    """
    Creates and configures a DSPy Language Model instance.

    Args:
        model: Model identifier (e.g., 'gemini/gemini-2.5-flash').
        api_key: API provider key.
        **kwargs: Overrides for default settings.

    Returns:
        Configured dspy.LM instance.

    Raises:
        ValueError: If api_key is missing.
    """
    if not api_key:
        raise ValueError(f"API key missing for model: {model}")

    logger.info(f"Connecting to LLM: {model}")

    params = {
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        **kwargs
    }

    lm = dspy.LM(model=model, api_key=api_key, **params)

    if settings.rate_limit_delay > 0:
        lm = _apply_throttling(lm, settings.rate_limit_delay)

    return lm

def setup_student() -> dspy.LM:
    """
    Configures and registers the global student model in DSPy settings.

    Returns:
        The configured student LM.
    """
    lm = create_lm(settings.student_model, settings.student_api_key)
    dspy.settings.configure(lm=lm)
    return lm

def setup_teacher() -> dspy.LM:
    """
    Configures the teacher model, falling back to the student model if undefined.

    Returns:
        The configured teacher LM.
    """
    if not settings.teacher_model or not settings.teacher_api_key:
        logger.warning("Teacher config missing. Using Student as Teacher.")
        return setup_student()

    return create_lm(settings.teacher_model, settings.teacher_api_key)