import logging
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import dspy

logger = logging.getLogger(__name__)


class ModelClient:
    def generate_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        files: list[Path] | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        raise NotImplementedError


class ModelGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.finish_reason = finish_reason


class StubModelClient(ModelClient):
    def generate_json(self, **_: Any) -> str:
        raise RuntimeError(
            "No model provider configured. Provide a valid core dspy.LM client."
        )


class CoreLMModelClient(ModelClient):
    def __init__(self, lm: dspy.LM) -> None:
        self.lm = lm

    def generate_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        files: list[Path] | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        provider_name = getattr(self.lm, "provider", "").lower()

        messages: list[dict[str, Any]]
        # Format messages according to provider compatibility
        if "ollama" in provider_name:
            images = []
            for path in files or []:
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                images.append(data)
            messages = [{"role": "user", "content": prompt}]
            if images:
                messages[0]["images"] = images
        else:
            # Standard OpenAI/OpenRouter multimodal format
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for path in files or []:
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"}
                })
            messages = [{"role": "user", "content": content_parts}]

        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens

        # OpenRouter / API provider reasoning support
        if thinking_level:
            kwargs["reasoning"] = {
                "enabled": True,
                "thinking_budget": thinking_budget
            }

        try:
            response_list = self.lm(prompt=messages, **kwargs)
            if not response_list or not response_list[0]:
                raise ModelGenerationError("Empty response from model")
            return response_list[0]
        except Exception as exc:
            if isinstance(exc, ModelGenerationError):
                raise
            raise ModelGenerationError(f"Core LM call failed: {exc}") from exc


def get_model_client(provider: str | None = None, lm: dspy.LM | None = None) -> ModelClient:
    if lm is not None:
        if provider:
            logger.warning(
                f"[get_model_client] Ignoring configured provider '{provider}' "
                "because an explicit dspy.LM instance was provided."
            )
        return CoreLMModelClient(lm)

    # Fallback to configured settings in DSPy
    active_lm = getattr(dspy.settings, "lm", None)
    if active_lm is not None:
        if provider:
            logger.warning(
                f"[get_model_client] Ignoring configured provider '{provider}' "
                "because a global dspy.settings.lm client is active."
            )
        return CoreLMModelClient(active_lm)

    if provider:
        logger.warning(
            f"[get_model_client] Configured provider '{provider}' was requested, "
            "but no active dspy.LM is set. Falling back to StubModelClient."
        )
    return StubModelClient()


def _loads_with_missing_comma_repair(text: str) -> Any:
    original_error: json.JSONDecodeError | None = None
    candidate = text
    for _ in range(8):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            if original_error is None:
                original_error = exc
            if exc.msg != "Expecting ',' delimiter":
                raise original_error
            if exc.pos <= 0 or exc.pos >= len(candidate):
                raise original_error
            candidate = f"{candidate[: exc.pos]},{candidate[exc.pos :]}"
    if original_error is not None:
        raise original_error
    return json.loads(text)


def parse_model_json(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return _loads_with_missing_comma_repair(stripped)
        except json.JSONDecodeError:
            pass

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S | re.I)
    if fence:
        fenced = fence.group(1)
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            return _loads_with_missing_comma_repair(fenced)

    start = min([i for i in (stripped.find("{"), stripped.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise json.JSONDecodeError("No JSON object or array found", stripped, 0)
    end_obj = stripped.rfind("}")
    end_arr = stripped.rfind("]")
    end = max(end_obj, end_arr)
    if end < start:
        raise json.JSONDecodeError("No complete JSON value found", stripped, start)
    extracted = stripped[start : end + 1]
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return _loads_with_missing_comma_repair(extracted)


class ModelJsonParseError(RuntimeError):
    def __init__(self, message: str, *, raw_response_path: Path | None = None) -> None:
        super().__init__(message)
        self.raw_response_path = raw_response_path


def _parse_retries(default: int = 2) -> int:
    value = os.environ.get("VISUAL_JSON_PARSE_RETRIES")
    if value is None or value == "":
        return default
    return max(0, int(value))


def _attempt_path(path: Path, attempt: int) -> Path:
    return path.with_name(f"{path.stem}.attempt_{attempt:02d}{path.suffix}")


def _write_raw_attempt(path: Path | None, attempt: int, raw: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    _attempt_path(path, attempt).write_text(raw, encoding="utf-8")


def _increase_max_output_tokens(generate_kwargs: dict[str, Any]) -> None:
    value = generate_kwargs.get("max_output_tokens")
    if isinstance(value, int) and value > 0:
        generate_kwargs["max_output_tokens"] = value * 2


def generate_parsed_json(
    client: ModelClient,
    *,
    raw_response_path: Path | None = None,
    parse_retries: int | None = None,
    **generate_kwargs: Any,
) -> tuple[str, Any, list[str]]:
    total_attempts = 1 + (parse_retries if parse_retries is not None else _parse_retries())
    last_error: json.JSONDecodeError | None = None
    warnings: list[str] = []

    kwargs = dict(generate_kwargs)
    for attempt in range(1, total_attempts + 1):
        try:
            raw = client.generate_json(**kwargs)
        except ModelGenerationError as exc:
            raw = exc.raw_text
            _write_raw_attempt(raw_response_path, attempt, raw)
            warnings.append(f"model_json_generation_retry:{attempt}:finish_reason={exc.finish_reason}")
            _increase_max_output_tokens(kwargs)
            if attempt < total_attempts:
                continue
            raise ModelJsonParseError(
                f"Model did not finish after {total_attempts} attempt(s): finish_reason={exc.finish_reason}",
                raw_response_path=raw_response_path,
            ) from exc
        _write_raw_attempt(raw_response_path, attempt, raw)
        try:
            parsed = parse_model_json(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            warnings.append(f"model_json_parse_retry:{attempt}:{exc.msg}")
            _increase_max_output_tokens(kwargs)
            if attempt < total_attempts:
                continue
            raise ModelJsonParseError(
                f"Model returned invalid JSON after {total_attempts} attempt(s): {exc.msg}",
                raw_response_path=raw_response_path,
            ) from exc
        if attempt > 1:
            warnings.append(f"model_json_recovered_after_retry:{attempt}")
        return raw, parsed, warnings

    raise ModelJsonParseError(
        f"Model returned invalid JSON after {total_attempts} attempt(s): {last_error or 'unknown parse error'}",
        raw_response_path=raw_response_path,
    )
