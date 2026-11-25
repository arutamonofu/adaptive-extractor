"""Gemini API client wrapping the google-genai SDK for Ingestion v2."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, TypeVar, Union

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

_CURRENT_DOC_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ae_gemini_doc_scope", default="__global__"
)

INLINE_PDF_MAX_BYTES = 18 * 1024 * 1024  # 18 MB safe inline threshold below 20 MB API limit
INLINE_BATCH_MAX_BYTES = 18 * 1024 * 1024


@dataclass
class BatchRequestSpec:
    """Single request specification for Gemini Batch API execution."""

    key: str
    contents: list[Any]
    stage_name: str
    model: Optional[str] = None
    system_instruction: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    media_resolution: Optional[str] = None
    thinking_level: Optional[str] = None
    thinking_budget: Optional[int] = None
    response_schema: Optional[type[BaseModel]] = None
    doc_scope: Optional[str] = None
    uploaded_handles: list[Any] = field(default_factory=list, repr=False, compare=False)


@dataclass
class BatchItemResult:
    """Result of a single item in a Gemini Batch API job."""

    key: str
    text: Optional[str] = None
    parsed: Optional[Any] = None
    error: Optional[str] = None

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
        max_retries = getattr(config, "max_retries", 5)
        self._max_retries: int = max_retries if isinstance(max_retries, int) else 5
        retry_delay = getattr(config, "retry_delay", 2.0)
        self._retry_delay: float = (
            float(retry_delay) if isinstance(retry_delay, (int, float)) else 2.0
        )
        timeout = getattr(config, "timeout", 300)
        self._timeout: int = timeout if isinstance(timeout, int) else 300

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self._timeout > 0:
            try:
                sig = inspect.signature(genai.Client)
                accepts_http_options = "http_options" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if accepts_http_options:
                    timeout_ms = int(self._timeout * 1000)
                    http_options_cls = getattr(getattr(genai, "types", None), "HttpOptions", None)
                    if http_options_cls is not None:
                        client_kwargs["http_options"] = http_options_cls(timeout=timeout_ms)
                    else:
                        client_kwargs["http_options"] = {"timeout": timeout_ms}
            except (TypeError, ValueError):
                pass

        try:
            self._client = genai.Client(**client_kwargs)
        except TypeError:
            client_kwargs.pop("http_options", None)
            self._client = genai.Client(**client_kwargs)

        max_concurrent = getattr(config, "max_concurrent_requests", 4)
        if not isinstance(max_concurrent, int):
            max_concurrent = 4
        self._semaphore = threading.Semaphore(max_concurrent)

        # Default parameters from config.defaults
        defaults = getattr(config, "defaults", None)
        self._default_model = getattr(defaults, "model", "gemini-3-flash-preview")
        self._default_temperature = getattr(defaults, "temperature", 0.0)
        self._default_max_output_tokens = getattr(defaults, "max_output_tokens", 65536)
        self._default_media_resolution = getattr(defaults, "media_resolution", "high")
        self._default_thinking_level = getattr(defaults, "thinking_level", "low")
        self._default_thinking_budget = getattr(defaults, "thinking_budget", None)

        # Thread-safe usage statistics tracking scoped by doc_id
        self._usage_lock = threading.Lock()
        self._stage_usage_by_scope: dict[str, dict[str, StageUsage]] = {}
        self._counters_by_scope: dict[str, dict[str, int]] = {}

    @property
    def _stage_usage(self) -> dict[str, StageUsage]:
        """Backward-compatible view of stage usage for the active scope."""
        scope = _CURRENT_DOC_SCOPE.get()
        return self._stage_usage_by_scope.setdefault(scope, {})

    @contextlib.contextmanager
    def doc_scope(self, doc_id: Optional[str]) -> Iterator[None]:
        """Bind token and counter tracking to a specific document ID."""
        scope_key = doc_id.strip() if isinstance(doc_id, str) and doc_id.strip() else "__global__"
        token = _CURRENT_DOC_SCOPE.set(scope_key)
        try:
            yield
        finally:
            _CURRENT_DOC_SCOPE.reset(token)

    def reset_stage_usage(self, doc_id: Optional[str] = None) -> None:
        """Clear accumulated usage statistics for a scope."""
        scope = doc_id or _CURRENT_DOC_SCOPE.get()
        with self._usage_lock:
            self._stage_usage_by_scope.pop(scope, None)
            self._counters_by_scope.pop(scope, None)

    def record_event(self, metric_name: str, count: int = 1, doc_id: Optional[str] = None) -> None:
        """Increment a per-document operational counter (e.g., t4_skipped_calls)."""
        scope = doc_id or _CURRENT_DOC_SCOPE.get()
        with self._usage_lock:
            counters = self._counters_by_scope.setdefault(scope, {})
            counters[metric_name] = counters.get(metric_name, 0) + count

    def get_scope_counters(self, doc_id: Optional[str] = None) -> dict[str, int]:
        """Return operational counters for the active or specified document scope."""
        scope = doc_id or _CURRENT_DOC_SCOPE.get()
        with self._usage_lock:
            return dict(self._counters_by_scope.get(scope, {}))

    # ------------------------------------------------------------------
    # Usage metrics
    # ------------------------------------------------------------------

    def record_usage(
        self,
        stage: str,
        prompt_tokens: int = 0,
        candidates_tokens: int = 0,
        total_tokens: int = 0,
        *,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        doc_id: Optional[str] = None,
    ) -> None:
        scope = doc_id or _CURRENT_DOC_SCOPE.get()
        with self._usage_lock:
            scope_map = self._stage_usage_by_scope.setdefault(scope, {})
            if stage not in scope_map:
                scope_map[stage] = StageUsage()
            usage = scope_map[stage]
            usage.calls += 1
            usage.prompt_tokens += prompt_tokens
            usage.cached_tokens += cached_tokens
            usage.candidates_tokens += candidates_tokens
            usage.reasoning_tokens += reasoning_tokens
            usage.total_tokens += total_tokens

    def get_all_stage_usage(self, doc_id: Optional[str] = None) -> dict[str, StageUsage]:
        scope = doc_id or _CURRENT_DOC_SCOPE.get()
        with self._usage_lock:
            scope_map = self._stage_usage_by_scope.get(scope, {})
            return {k: v.model_copy() for k, v in scope_map.items()}

    # ------------------------------------------------------------------
    # Files API & Inline Parts
    # ------------------------------------------------------------------

    def make_pdf_part(self, pdf_source: Union[Path, bytes]) -> Any:
        """Create an inline Gemini Part from PDF bytes or a PDF file path."""
        data = pdf_source.read_bytes() if isinstance(pdf_source, Path) else pdf_source
        return self._genai.types.Part.from_bytes(data=data, mime_type="application/pdf")

    def upload_file(self, file_path: Path) -> Any:
        """Upload a file via the Files API and return the file handle."""
        logger.info(f"Uploading {file_path.name} to Gemini Files API…")
        file_handle = self._client.files.upload(file=str(file_path))
        self.record_event("files_api_uploads")
        logger.info(f"Uploaded: {file_handle.name}")
        return file_handle

    def delete_file(self, file_handle: Any) -> None:
        """Delete an uploaded file handle."""
        if file_handle is None:
            return
        try:
            name = getattr(file_handle, "name", str(file_handle))
            self._client.files.delete(name=name)
            self.record_event("files_api_deletes")
            logger.debug(f"Deleted uploaded file: {name}")
        except Exception as exc:
            logger.warning(f"Failed to delete uploaded file: {exc}")

    # ------------------------------------------------------------------
    # Content normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_contents(contents: list[Any]) -> list[Any]:
        """Convert Path items to in-memory PIL Images and close file handles."""
        normalized = []
        for item in contents:
            if isinstance(item, Path):
                with PILImage.open(item) as img:
                    img.load()
                    copied = img.copy()
                    copied.format = img.format
                normalized.append(copied)
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
            detail = (
                f"Gemini reached MAX_TOKENS without generating any content in stage '{stage_name}'."
                if not parts
                else f"Gemini generation truncated due to MAX_TOKENS limit in stage '{stage_name}'."
            )
            raise GeminiMaxTokensError(stage=stage_name, details=detail)

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

    def _record_response_usage(
        self, stage: str, response: Any, *, doc_id: Optional[str] = None
    ) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
            candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
            reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (
                prompt_tokens + candidates_tokens + reasoning_tokens
            )
            self.record_usage(
                stage,
                prompt_tokens,
                candidates_tokens,
                total_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                doc_id=doc_id,
            )
        else:
            self.record_usage(stage, 0, 0, 0, doc_id=doc_id)

    # ------------------------------------------------------------------
    # Gemini Batch API support
    # ------------------------------------------------------------------

    def _to_batch_part(self, item: Any) -> Any:
        """Normalize a content item into a Gemini types.Part or string for Batch API."""
        types = self._genai.types
        if isinstance(item, Path):
            suffix = item.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            if suffix == ".pdf":
                mime = "application/pdf"
            return types.Part.from_bytes(data=item.read_bytes(), mime_type=mime)
        return item

    def run_batch_requests(
        self,
        specs: list[BatchRequestSpec],
        *,
        display_name: str = "ae-ingest-batch",
        poll_interval: float = 5.0,
        timeout_s: float = 1800.0,
    ) -> dict[str, BatchItemResult]:
        """Execute heterogeneous requests via Gemini Batch API (50% discount tier).

        Automatically chunks requests so each inline Batch API submission stays below
        ``INLINE_BATCH_MAX_BYTES``. Falls back per-item error reporting if any item fails.
        """
        results: dict[str, BatchItemResult] = {}
        if not specs:
            return results

        types = self._genai.types

        # Partition specs into payload-bounded inline chunks
        chunks: list[list[BatchRequestSpec]] = []
        current_chunk: list[BatchRequestSpec] = []
        current_bytes = 0

        for spec in specs:
            est_bytes = 0
            for item in spec.contents:
                if isinstance(item, Path) and item.exists():
                    est_bytes += item.stat().st_size
                elif isinstance(item, str):
                    est_bytes += len(item.encode("utf-8"))
                elif hasattr(item, "inline_data") and getattr(item.inline_data, "data", None):
                    est_bytes += len(item.inline_data.data)
                else:
                    est_bytes += 4096
            # Base64 expansion (~1.34x) + JSON overhead
            est_wire = int(est_bytes * 1.4) + 2048
            if current_chunk and current_bytes + est_wire > INLINE_BATCH_MAX_BYTES:
                chunks.append(current_chunk)
                current_chunk = []
                current_bytes = 0
            current_chunk.append(spec)
            current_bytes += est_wire
        if current_chunk:
            chunks.append(current_chunk)

        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED",
        }

        for chunk_idx, chunk_specs in enumerate(chunks, 1):
            model_name = chunk_specs[0].model or self._default_model
            inlined_requests = []
            for spec in chunk_specs:
                norm_contents = [self._to_batch_part(c) for c in spec.contents]
                cfg = self._build_config(
                    system_instruction=spec.system_instruction,
                    temperature=spec.temperature,
                    max_output_tokens=spec.max_output_tokens,
                    media_resolution=spec.media_resolution,
                    thinking_level=spec.thinking_level,
                    thinking_budget=spec.thinking_budget,
                    response_schema=spec.response_schema,
                )
                inlined_requests.append(
                    types.InlinedRequest(
                        model=spec.model or model_name,
                        contents=norm_contents,
                        metadata={"key": spec.key},
                        config=cfg,
                    )
                )

            logger.info(
                f"Submitting Gemini Batch job '{display_name}' chunk {chunk_idx}/{len(chunks)} "
                f"({len(chunk_specs)} requests, model={model_name})..."
            )
            job = self._client.batches.create(
                model=model_name,
                src=inlined_requests,
                config=types.CreateBatchJobConfig(display_name=f"{display_name}-{chunk_idx}"),
            )

            deadline = time.monotonic() + timeout_s
            while True:
                state_str = str(getattr(job, "state", ""))
                if "." in state_str:
                    state_str = state_str.split(".")[-1]
                state_str = state_str.upper()
                if state_str in terminal_states:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Batch job {getattr(job, 'name', display_name)} timed out after {timeout_s}s"
                    )
                time.sleep(poll_interval)
                job = self._client.batches.get(name=job.name)

            dest = getattr(job, "dest", None)
            inlined_responses = getattr(dest, "inlined_responses", None) or []

            # Also support JSONL file destination if SDK returns file_name instead of inlined_responses
            if not inlined_responses and getattr(dest, "file_name", None):
                raw_bytes = self._client.files.download(file=dest.file_name)
                lines = raw_bytes.decode("utf-8").splitlines()
                for idx, spec in enumerate(chunk_specs):
                    if idx >= len(lines):
                        results[spec.key] = BatchItemResult(key=spec.key, error="Missing batch output line")
                        continue
                    row = json.loads(lines[idx])
                    resp_dict = row.get("response")
                    if not resp_dict:
                        results[spec.key] = BatchItemResult(
                            key=spec.key, error=str(row.get("error") or "Empty batch response")
                        )
                        continue
                    resp_obj = types.GenerateContentResponse.model_validate(resp_dict)
                    self._record_response_usage(spec.stage_name, resp_obj, doc_id=spec.doc_scope)
                    try:
                        self._validate_response(resp_obj, stage_name=spec.stage_name)
                        text = resp_obj.text or ""
                        parsed = (
                            spec.response_schema.model_validate_json(text)
                            if spec.response_schema is not None and text.strip()
                            else None
                        )
                        results[spec.key] = BatchItemResult(key=spec.key, text=text, parsed=parsed)
                    except Exception as exc:
                        results[spec.key] = BatchItemResult(key=spec.key, error=str(exc))
                continue

            for idx, spec in enumerate(chunk_specs):
                if idx >= len(inlined_responses):
                    results[spec.key] = BatchItemResult(
                        key=spec.key, error=f"Missing inlined batch response (state={state_str})"
                    )
                    continue
                ir = inlined_responses[idx]
                if getattr(ir, "error", None):
                    results[spec.key] = BatchItemResult(key=spec.key, error=str(ir.error))
                    continue
                resp = getattr(ir, "response", None)
                self._record_response_usage(spec.stage_name, resp, doc_id=spec.doc_scope)
                try:
                    self._validate_response(resp, stage_name=spec.stage_name)
                    text = getattr(resp, "text", None) or ""
                    parsed = getattr(resp, "parsed", None)
                    if parsed is None and spec.response_schema is not None and text.strip():
                        parsed = spec.response_schema.model_validate_json(text)
                    results[spec.key] = BatchItemResult(key=spec.key, text=text, parsed=parsed)
                except Exception as exc:
                    results[spec.key] = BatchItemResult(key=spec.key, error=str(exc))

        return results
