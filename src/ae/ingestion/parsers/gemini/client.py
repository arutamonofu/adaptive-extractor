"""Gemini API client wrapping the google-genai SDK for Ingestion v2."""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional, TypeVar

from PIL import Image as PILImage
from pydantic import BaseModel

from ae.core.exceptions import (
    GeminiEmptyResponseError,
    GeminiMaxTokensError,
    GeminiRecitationError,
    GeminiSafetyError,
)
from ae.ingestion.parsers.gemini.schemas import StageUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MEDIA_RESOLUTION_MAP = {
    "high": "MEDIA_RESOLUTION_HIGH",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "low": "MEDIA_RESOLUTION_LOW",
    "unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
}


class GeminiClient:
    """Wrapper around google-genai SDK with retry logic, rate limiting, and token tracking."""

    def __init__(self, config: Any, api_key: str) -> None:
        """Initialize client.

        Args:
            config: GeminiParserConfig or compatible configuration.
            api_key: Gemini API key string.
        """
        from google import genai

        self._config = config
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

        max_concurrent = getattr(config, "max_concurrent_requests", 4)
        if not isinstance(max_concurrent, int):
            max_concurrent = 4
        self._semaphore = threading.Semaphore(max_concurrent)

        max_retries = getattr(config, "max_retries", 5)
        self._max_retries: int = max_retries if isinstance(max_retries, int) else 5
        retry_delay = getattr(config, "retry_delay", 2.0)
        self._retry_delay: float = (
            float(retry_delay) if isinstance(retry_delay, (int, float)) else 2.0
        )
        timeout = getattr(config, "timeout", 300)
        self._timeout: int = timeout if isinstance(timeout, int) else 300

        # Default parameters from config.defaults
        defaults = getattr(config, "defaults", None)
        self._default_model = getattr(defaults, "model", "gemini-3-flash-preview")
        self._default_temperature = getattr(defaults, "temperature", 0.0)
        self._default_max_output_tokens = getattr(defaults, "max_output_tokens", 65536)
        self._default_media_resolution = getattr(defaults, "media_resolution", "high")
        self._default_thinking_level = getattr(defaults, "thinking_level", "low")
        self._default_thinking_budget = getattr(defaults, "thinking_budget", None)

        # Thread-safe usage statistics tracking
        self._usage_lock = threading.Lock()
        self._stage_usage: dict[str, StageUsage] = {}

    # ------------------------------------------------------------------
    # Usage metrics
    # ------------------------------------------------------------------

    def record_usage(
        self,
        stage: str,
        prompt_tokens: int = 0,
        candidates_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        with self._usage_lock:
            if stage not in self._stage_usage:
                self._stage_usage[stage] = StageUsage()
            usage = self._stage_usage[stage]
            usage.calls += 1
            usage.prompt_tokens += prompt_tokens
            usage.candidates_tokens += candidates_tokens
            usage.total_tokens += total_tokens

    def get_all_stage_usage(self) -> dict[str, StageUsage]:
        with self._usage_lock:
            return {k: v.model_copy() for k, v in self._stage_usage.items()}

    # ------------------------------------------------------------------
    # Files API
    # ------------------------------------------------------------------

    def upload_file(self, file_path: Path) -> Any:
        """Upload a file via the Files API and return the file handle."""
        logger.info(f"Uploading {file_path.name} to Gemini Files API…")
        file_handle = self._client.files.upload(file=str(file_path))
        logger.info(f"Uploaded: {file_handle.name}")
        return file_handle

    def delete_file(self, file_handle: Any) -> None:
        """Delete an uploaded file handle."""
        if file_handle is None:
            return
        try:
            name = getattr(file_handle, "name", str(file_handle))
            self._client.files.delete(name=name)
            logger.debug(f"Deleted uploaded file: {name}")
        except Exception as exc:
            logger.warning(f"Failed to delete uploaded file: {exc}")

    # ------------------------------------------------------------------
    # Content normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_contents(contents: list[Any]) -> list[Any]:
        """Convert Path items to opened PIL Images."""
        normalized = []
        for item in contents:
            if isinstance(item, Path):
                normalized.append(PILImage.open(item))
            else:
                normalized.append(item)
        return normalized

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_config(
        self,
        *,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        media_resolution: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        response_schema: Optional[type[T]] = None,
    ) -> Any:
        temp = temperature if temperature is not None else self._default_temperature
        tokens = max_output_tokens or self._default_max_output_tokens
        res_str = media_resolution or self._default_media_resolution
        norm_res = _MEDIA_RESOLUTION_MAP.get(res_str.lower(), res_str)

        kwargs: dict[str, Any] = {
            "temperature": temp,
            "max_output_tokens": tokens,
            "media_resolution": norm_res,
            "automatic_function_calling": self._genai.types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        }

        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction

        if thinking_budget is not None:
            t_budget: Optional[int] = thinking_budget
            t_level: Optional[str] = None
        elif thinking_level is not None:
            t_budget = None
            t_level = thinking_level
        else:
            t_budget = self._default_thinking_budget
            t_level = self._default_thinking_level

        if t_budget is not None:
            kwargs["thinking_config"] = self._genai.types.ThinkingConfig(
                thinking_budget=t_budget,
            )
        elif t_level:
            lvl = t_level.lower()
            if lvl == "none":
                kwargs["thinking_config"] = self._genai.types.ThinkingConfig(thinking_budget=0)
            else:
                kwargs["thinking_config"] = self._genai.types.ThinkingConfig(thinking_level=lvl)

        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema

        return self._genai.types.GenerateContentConfig(**kwargs)

    # ------------------------------------------------------------------
    # Generation helpers & Validation
    # ------------------------------------------------------------------

    def _validate_response(
        self,
        response: Any,
        stage_name: str,
        allow_empty: bool = False,
    ) -> None:
        """Validate Gemini API response for safety blocks, recitation, and missing content."""
        if response is None:
            raise GeminiEmptyResponseError(
                stage=stage_name,
                details=f"Gemini API returned None response in stage '{stage_name}'.",
            )

        # Check prompt-level feedback/blocks
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason:
                block_reason_str = getattr(block_reason, "name", str(block_reason))
                raise GeminiSafetyError(
                    stage=stage_name,
                    finish_reason=block_reason_str,
                    block_reason=block_reason_str,
                )

        # Check candidate availability
        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise GeminiEmptyResponseError(
                stage=stage_name,
                details=f"Gemini API returned response without candidates in stage '{stage_name}'.",
            )

        candidate = candidates[0]
        finish_reason_raw = getattr(candidate, "finish_reason", None)
        finish_reason_str = (
            getattr(finish_reason_raw, "name", str(finish_reason_raw))
            if finish_reason_raw is not None
            else "UNKNOWN"
        )
        if "." in finish_reason_str:
            finish_reason_str = finish_reason_str.split(".")[-1]
        finish_reason_str = finish_reason_str.upper()

        if finish_reason_str == "RECITATION":
            raise GeminiRecitationError(stage=stage_name)

        if finish_reason_str in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"):
            safety_ratings = getattr(candidate, "safety_ratings", None)
            raise GeminiSafetyError(
                stage=stage_name,
                finish_reason=finish_reason_str,
                safety_ratings=safety_ratings,
            )

        parts = getattr(getattr(candidate, "content", None), "parts", None)
        if finish_reason_str == "MAX_TOKENS":
            if not parts and not allow_empty:
                raise GeminiMaxTokensError(
                    stage=stage_name,
                    details=(
                        f"Gemini reached MAX_TOKENS without generating "
                        f"any content in stage '{stage_name}'."
                    ),
                )
            logger.warning(
                f"Stage '{stage_name}': Generation reached MAX_TOKENS limit; "
                f"output may be truncated.",
            )

        if not parts and not allow_empty:
            raise GeminiEmptyResponseError(
                stage=stage_name,
                details=(
                    f"Gemini returned candidate with empty content parts "
                    f"(finish_reason={finish_reason_str}) in stage '{stage_name}'."
                ),
            )

    def generate_text(
        self,
        contents: list[Any],
        *,
        stage_name: str = "text",
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        media_resolution: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        allow_empty: bool = False,
    ) -> str:
        """Generate plain-text response with retry logic and rate limiting."""
        model_name = model or self._default_model
        norm_contents = self._normalize_contents(contents)
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            media_resolution=media_resolution,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
        )

        with self._semaphore:
            response = self._call_with_retry(model_name, norm_contents, config)

        self._record_response_usage(stage_name, response)
        self._validate_response(response, stage_name=stage_name, allow_empty=allow_empty)

        text = response.text
        if text is None or (not allow_empty and not text.strip()):
            raise GeminiEmptyResponseError(
                stage=stage_name,
                details=f"Gemini returned empty or whitespace-only text in stage '{stage_name}'.",
            )
        return text

    def generate_structured(
        self,
        contents: list[Any],
        response_schema: type[T],
        *,
        stage_name: str = "structured",
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        media_resolution: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ) -> T:
        """Generate structured Pydantic response with retry logic."""
        model_name = model or self._default_model
        norm_contents = self._normalize_contents(contents)
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            media_resolution=media_resolution,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            response_schema=response_schema,
        )

        with self._semaphore:
            response = self._call_with_retry(model_name, norm_contents, config)

        self._record_response_usage(stage_name, response)
        self._validate_response(response, stage_name=stage_name)

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed  # type: ignore[return-value]

        # Fallback parsing if .parsed is not populated
        text = response.text
        if not text or not text.strip():
            raise GeminiEmptyResponseError(
                stage=stage_name,
                details=(
                    f"Gemini returned empty text for schema {response_schema.__name__} "
                    f"in stage '{stage_name}'."
                ),
            )
        return response_schema.model_validate_json(text)

    # ------------------------------------------------------------------
    # Retry loop & Token tracking
    # ------------------------------------------------------------------

    def _call_with_retry(self, model: str, contents: list, config: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"Gemini API call failed (attempt {attempt}/{self._max_retries}): {exc}",
                )
                if attempt < self._max_retries:
                    base_delay = self._retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0.5, 2.0)
                    delay = min(60.0, base_delay + jitter)
                    exc_str = str(exc)
                    if any(
                        c in exc_str for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
                    ):
                        delay = max(delay, min(60.0, 4.0 * attempt + jitter))
                    logger.info(f"Retrying Gemini API call in {delay:.1f}s...")
                    time.sleep(delay)

        raise RuntimeError(
            f"Gemini API call failed after {self._max_retries} attempts: {last_exc}",
        ) from last_exc

    def _record_response_usage(self, stage: str, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (
                prompt_tokens + candidates_tokens
            )
            self.record_usage(stage, prompt_tokens, candidates_tokens, total_tokens)
        else:
            self.record_usage(stage, 0, 0, 0)
