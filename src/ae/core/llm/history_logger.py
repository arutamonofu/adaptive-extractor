"""LLM history logging utility, streaming recorder, and LangChain callback handler."""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
import traceback as tb_module
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STREAM_BUFFER_CHARS = 2_000_000
_DEFAULT_MAX_ENTRY_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_FILE_BYTES = 250 * 1024 * 1024
_DEFAULT_MAX_IN_MEMORY_ENTRIES = 200
_MAX_JSON_DEPTH = 12

_PRESERVED_ENTRY_FIELDS = (
    "timestamp",
    "source",
    "doc_id",
    "node",
    "stage",
    "attempt",
    "batch_idx",
    "status",
    "provider",
    "tool_calls",
    "finish_reason",
    "usage",
    "error",
    "partial_output",
    "llm_run_id",
)

_LOG_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar("ae_llm_log_context", default=None)


def get_extraction_log_context() -> Dict[str, Any]:
    """Return a copy of the current extraction LLM logging context."""
    ctx = _LOG_CONTEXT.get()
    return dict(ctx) if ctx else {}


def update_extraction_log_context(**kwargs: Any) -> None:
    """Update fields in the current extraction LLM logging context in-place."""
    ctx = _LOG_CONTEXT.get()
    if ctx is None:
        return
    for key, value in kwargs.items():
        if value is None:
            ctx.pop(key, None)
        else:
            ctx[key] = value


@contextmanager
def extraction_log_context(**kwargs: Any) -> Iterator[Dict[str, Any]]:
    """Context manager to bind extraction logging metadata (doc_id, node, stage, etc.)."""
    parent = _LOG_CONTEXT.get()
    merged = dict(parent) if parent else {}
    for key, value in kwargs.items():
        if value is not None:
            merged[key] = value
    token = _LOG_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _LOG_CONTEXT.reset(token)


def _sanitize_text(text: str) -> str:
    """Ensure text contains no invalid UTF-16 surrogates."""
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8")


def _extract_thinking_from_content(content: Any) -> Optional[str]:
    """Extract thinking/reasoning blocks from structured AIMessage content lists."""
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
            val = block.get("thinking") or block.get("reasoning") or block.get("text")
            if isinstance(val, str) and val.strip():
                parts.append(val)
    return "\n".join(parts) if parts else None


def _serialize_base_message(msg: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain BaseMessage into a uniform chat-message dictionary."""
    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    else:
        role = getattr(msg, "type", "unknown")

    result: Dict[str, Any] = {
        "role": role,
        "content": _clean_for_json(msg.content),
    }

    if isinstance(msg, AIMessage):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = _clean_for_json(tool_calls)
        additional = getattr(msg, "additional_kwargs", None) or {}
        if not tool_calls and additional.get("tool_calls"):
            result["tool_calls"] = _clean_for_json(additional["tool_calls"])
        reasoning = additional.get("reasoning_content") or additional.get("reasoning")
        if reasoning:
            result["reasoning"] = _clean_for_json(reasoning)

    if isinstance(msg, ToolMessage):
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            result["tool_call_id"] = str(tool_call_id)
        name = getattr(msg, "name", None)
        if name:
            result["name"] = str(name)

    return result


def _clean_for_json(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert objects into JSON-serializable primitives with depth safety."""
    if _depth > _MAX_JSON_DEPTH:
        return _sanitize_text(str(obj))

    if isinstance(obj, BaseMessage):
        return _serialize_base_message(obj)
    if isinstance(obj, BaseModel):
        try:
            return _clean_for_json(obj.model_dump(mode="json"), _depth + 1)
        except Exception:
            return _sanitize_text(str(obj))
    if isinstance(obj, BaseException):
        return {
            "type": type(obj).__name__,
            "message": _sanitize_text(str(obj)),
        }
    if isinstance(obj, dict):
        return {
            _sanitize_text(str(k)): _clean_for_json(v, _depth + 1)
            for k, v in obj.items()
            if k != "lm"
        }
    if isinstance(obj, (list, tuple, set)):
        return [_clean_for_json(item, _depth + 1) for item in obj]
    if isinstance(obj, str):
        return _sanitize_text(obj)
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return _sanitize_text(str(obj))


def _clean_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a history entry while preserving standard and diagnostic fields."""
    cleaned: Dict[str, Any] = {
        "messages": entry.get("messages", []),
        "outputs": entry.get("outputs", []),
        "model": entry.get("model", ""),
    }

    if "latency_s" in entry and entry["latency_s"] is not None:
        cleaned["latency_s"] = entry["latency_s"]

    for field in _PRESERVED_ENTRY_FIELDS:
        if field in entry and entry[field] is not None:
            val = entry[field]
            if isinstance(val, (list, dict, str)) and not val and field not in ("error",):
                continue
            cleaned[field] = val

    kwargs = entry.get("kwargs", {})
    if isinstance(kwargs, dict):
        extra = {k: v for k, v in kwargs.items() if k != "messages"}
        if extra:
            cleaned["kwargs"] = extra

    return cleaned


def format_error_payload(exc: BaseException, include_traceback: bool = True) -> Dict[str, Any]:
    """Format an exception into a structured, JSON-safe dictionary."""
    payload: Dict[str, Any] = {
        "type": type(exc).__name__,
        "message": _sanitize_text(str(exc)),
    }
    if include_traceback:
        tb_str = "".join(tb_module.format_exception(type(exc), exc, exc.__traceback__))
        if tb_str:
            payload["traceback"] = _sanitize_text(tb_str[-8000:])
    return payload


def _truncate_text(text: str, max_chars: int, reason: str) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n... [TRUNCATED: {reason}, original_chars={len(text)}]"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker


def _enforce_entry_size(entry: Dict[str, Any], max_entry_bytes: int) -> str:
    """Serialize entry to compact JSON string, truncating oversized fields if needed."""
    serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    encoded_len = len(serialized.encode("utf-8"))
    if encoded_len <= max_entry_bytes:
        return serialized

    shrunk = copy.deepcopy(entry)
    # Iteratively trim largest string payloads (messages, outputs, partial_output)
    target_chars = max(1024, max_entry_bytes // 4)
    for key in ("outputs", "partial_output"):
        val = shrunk.get(key)
        if isinstance(val, str) and len(val) > target_chars:
            shrunk[key] = _truncate_text(
                val, target_chars, f"exceeded max_entry_bytes={max_entry_bytes}"
            )
        elif isinstance(val, list):
            shrunk[key] = [
                _truncate_text(
                    item,
                    target_chars,
                    f"exceeded max_entry_bytes={max_entry_bytes}",
                )
                if isinstance(item, str)
                else item
                for item in val
            ]

    messages = shrunk.get("messages")
    if isinstance(messages, list) and messages:
        per_msg_budget = max(512, target_chars // len(messages))
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                msg["content"] = _truncate_text(
                    msg["content"],
                    per_msg_budget,
                    f"exceeded max_entry_bytes={max_entry_bytes}",
                )

    serialized = json.dumps(shrunk, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) <= max_entry_bytes:
        return serialized

    # Hard fallback summary if still oversized
    minimal = {
        "timestamp": shrunk.get("timestamp"),
        "doc_id": shrunk.get("doc_id"),
        "node": shrunk.get("node"),
        "stage": shrunk.get("stage"),
        "status": shrunk.get("status"),
        "model": shrunk.get("model"),
        "error": shrunk.get("error"),
        "truncated_entry": True,
        "original_bytes": encoded_len,
    }
    return json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))


class LLMHistoryRecorder:
    """Thread-safe, memory- and disk-safe LLM interaction recorder.

    Supports immediate append-only JSONL streaming to disk (O(1) RAM) as well
    as bounded in-memory buffering when no stream path is configured yet.
    """

    def __init__(
        self,
        max_entry_bytes: int = _DEFAULT_MAX_ENTRY_BYTES,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        max_in_memory_entries: int = _DEFAULT_MAX_IN_MEMORY_ENTRIES,
    ) -> None:
        self.max_entry_bytes = max(1024, int(max_entry_bytes))
        self.max_file_bytes = max(1024, int(max_file_bytes))
        self.max_in_memory_entries = max(1, int(max_in_memory_entries))

        self._lock = threading.Lock()
        self._buffer: deque[Dict[str, Any]] = deque(maxlen=self.max_in_memory_entries)
        self._stream_path: Optional[Path] = None
        self._stream_file: Optional[Any] = None
        self._bytes_written: int = 0
        self._recorded_count: int = 0
        self._file_limit_warned: bool = False
        self._seen_run_keys: deque[tuple[str, str, str]] = deque(maxlen=4096)
        self._seen_run_keys_set: set[tuple[str, str, str]] = set()

    @property
    def recorded_count(self) -> int:
        with self._lock:
            return self._recorded_count

    @property
    def stream_path(self) -> Optional[Path]:
        with self._lock:
            return self._stream_path

    @property
    def is_streaming(self) -> bool:
        with self._lock:
            return self._stream_path is not None

    def get_entries(self) -> List[Dict[str, Any]]:
        """Return buffered entries (only populated when not streaming directly to disk)."""
        with self._lock:
            return list(self._buffer)

    def start_stream(self, output_path: Path) -> Path:
        """Configure an append-only JSONL stream at output_path and flush any buffered entries.

        The file is opened lazily on the first write so sessions with zero recorded
        entries do not leave empty 0-byte files on disk.
        """
        with self._lock:
            if self._stream_path == output_path and self._stream_file is not None:
                return output_path

            self._close_stream_unlocked()
            self._stream_path = output_path
            self._stream_file = None
            self._bytes_written = output_path.stat().st_size if output_path.exists() else 0
            self._file_limit_warned = False

            while self._buffer:
                buffered_entry = self._buffer.popleft()
                self._write_line_unlocked(buffered_entry)

            return output_path

    def record(self, entry: Dict[str, Any]) -> None:
        """Normalize and record a single LLM interaction or error entry safely."""
        try:
            cleaned = _clean_for_json(_clean_entry(entry))
            if "timestamp" not in cleaned:
                cleaned["timestamp"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            logger.warning(f"Failed to serialize LLM history entry: {exc}")
            return

        with self._lock:
            run_id = cleaned.get("llm_run_id")
            status = cleaned.get("status")
            stage = cleaned.get("stage")
            if run_id:
                dedup_key = (str(run_id), str(status or ""), str(stage or ""))
                if dedup_key in self._seen_run_keys_set:
                    return
                if len(self._seen_run_keys) == self._seen_run_keys.maxlen:
                    evicted = self._seen_run_keys.popleft()
                    self._seen_run_keys_set.discard(evicted)
                self._seen_run_keys.append(dedup_key)
                self._seen_run_keys_set.add(dedup_key)

            if self._stream_path is not None:
                if self._write_line_unlocked(cleaned):
                    self._recorded_count += 1
            else:
                self._buffer.append(cleaned)
                self._recorded_count += 1

    def _ensure_stream_open_unlocked(self) -> bool:
        if self._stream_file is not None:
            return True
        if self._stream_path is None:
            return False
        try:
            self._stream_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream_file = open(self._stream_path, "a", encoding="utf-8")
            return True
        except Exception as exc:
            logger.warning(f"Failed opening LLM history stream {self._stream_path}: {exc}")
            return False

    def _write_line_unlocked(self, cleaned_entry: Dict[str, Any]) -> bool:
        if not self._ensure_stream_open_unlocked() or self._stream_file is None:
            return False
        try:
            line = _enforce_entry_size(cleaned_entry, self.max_entry_bytes) + "\n"
            line_bytes = len(line.encode("utf-8"))
            if self._bytes_written + line_bytes > self.max_file_bytes:
                if not self._file_limit_warned:
                    logger.warning(
                        f"LLM history file {self._stream_path} reached max_file_bytes="
                        f"{self.max_file_bytes}; skipping further disk writes."
                    )
                    self._file_limit_warned = True
                return False

            self._stream_file.write(line)
            self._stream_file.flush()
            self._bytes_written += line_bytes
            return True
        except Exception as exc:
            logger.warning(f"Failed writing LLM history entry to {self._stream_path}: {exc}")
            return False

    def flush(self) -> None:
        with self._lock:
            if self._stream_file is not None:
                try:
                    self._stream_file.flush()
                except Exception as exc:
                    logger.warning(f"Failed flushing LLM history stream: {exc}")

    def close(self) -> None:
        with self._lock:
            self._close_stream_unlocked()

    def _close_stream_unlocked(self) -> None:
        if self._stream_file is not None:
            try:
                self._stream_file.flush()
                self._stream_file.close()
            except Exception as exc:
                logger.warning(f"Failed closing LLM history stream: {exc}")
            finally:
                self._stream_file = None

    def save_json(
        self,
        output_path: Path,
        extra_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> int:
        """Atomically save buffered (and optional extra) entries to a JSON array file."""
        with self._lock:
            entries = list(self._buffer)
            if extra_entries:
                for item in extra_entries:
                    entries.append(_clean_for_json(_clean_entry(item)))

        if not entries:
            return 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            temp_path.replace(output_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        logger.info(f"Saved {len(entries)} LLM calls to {output_path}")
        return len(entries)


class ExtractionHistoryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for chat calls, tool calls, stream cutoffs, and errors."""

    def __init__(
        self,
        recorder: LLMHistoryRecorder,
        max_stream_buffer_chars: int = _DEFAULT_MAX_STREAM_BUFFER_CHARS,
        default_source: str = "extraction_graph",
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.max_stream_buffer_chars = max(1024, int(max_stream_buffer_chars))
        self.default_source = default_source

        self._lock = threading.Lock()
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._seen_run_ids: deque[str] = deque(maxlen=512)
        self._seen_run_id_set: set[str] = set()

    def _mark_seen(self, run_key: str) -> bool:
        """Return True if run_key was not yet seen, and register it in bounded history."""
        if run_key in self._seen_run_id_set:
            return False
        if len(self._seen_run_ids) == self._seen_run_ids.maxlen:
            evicted = self._seen_run_ids.popleft()
            self._seen_run_id_set.discard(evicted)
        self._seen_run_ids.append(run_key)
        self._seen_run_id_set.add(run_key)
        return True

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        ctx = get_extraction_log_context()
        update_extraction_log_context(callback_invoked=True, last_run_id=run_key)

        invocation_params = kwargs.get("invocation_params") or {}
        meta = metadata or {}
        model_name = (
            invocation_params.get("model")
            or invocation_params.get("model_name")
            or meta.get("ls_model_name")
            or (serialized or {}).get("kwargs", {}).get("model")
            or (serialized or {}).get("kwargs", {}).get("model_name")
            or ""
        )
        provider_name = meta.get("ls_provider") or ""
        node_name = ctx.get("node") or meta.get("langgraph_node")

        flat_messages: List[Dict[str, Any]] = []
        for batch in messages or []:
            for msg in batch:
                flat_messages.append(_clean_for_json(msg))

        safe_kwargs = {
            k: v
            for k, v in invocation_params.items()
            if k not in ("messages", "api_key", "google_api_key")
        }

        with self._lock:
            if not self._mark_seen(f"start:{run_key}"):
                return
            self._active_runs[run_key] = {
                "start_time": time.perf_counter(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "messages": flat_messages,
                "model": model_name,
                "provider": provider_name,
                "doc_id": ctx.get("doc_id"),
                "node": node_name,
                "stage": ctx.get("stage") or node_name,
                "attempt": ctx.get("attempt"),
                "batch_idx": ctx.get("batch_idx"),
                "source": ctx.get("source") or self.default_source,
                "kwargs": safe_kwargs,
                "stream_chunks": [],
                "stream_chars": 0,
            }

    def on_llm_new_token(
        self,
        token: Union[str, list[Union[str, dict[str, Any]]]],
        *,
        chunk: Any = None,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if not token:
            return
        if isinstance(token, list):
            token_text = "".join(
                item if isinstance(item, str) else str(item.get("text", ""))
                for item in token
            )
        else:
            token_text = token
        if not token_text:
            return
        run_key = str(run_id)
        with self._lock:
            run_info = self._active_runs.get(run_key)
            if run_info is None:
                return
            current_chars = run_info["stream_chars"]
            if current_chars >= self.max_stream_buffer_chars:
                return
            remaining = self.max_stream_buffer_chars - current_chars
            piece = token_text[:remaining]
            run_info["stream_chunks"].append(piece)
            run_info["stream_chars"] += len(piece)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        with self._lock:
            if not self._mark_seen(f"end:{run_key}"):
                return
            run_info = self._active_runs.pop(run_key, None)

        if run_info is None:
            return

        latency_s = round(time.perf_counter() - run_info["start_time"], 4)
        outputs: List[Any] = []
        tool_calls: List[Any] = []
        finish_reason: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None
        model_name = run_info["model"]

        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                gen_info = getattr(gen, "generation_info", None) or {}
                if not finish_reason:
                    finish_reason = gen_info.get("finish_reason")
                if msg is not None:
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        text_parts = [
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        thinking = _extract_thinking_from_content(content)
                        if text_parts:
                            outputs.append("".join(text_parts))
                        elif content:
                            outputs.append(_clean_for_json(content))
                        if thinking:
                            run_info["kwargs"]["thinking"] = thinking
                    else:
                        outputs.append(str(content) if content is not None else "")

                    msg_tc = getattr(msg, "tool_calls", None)
                    if msg_tc:
                        tool_calls.extend(_clean_for_json(msg_tc))
                    else:
                        add_tc = (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls")
                        if add_tc:
                            tool_calls.extend(_clean_for_json(add_tc))

                    resp_meta = getattr(msg, "response_metadata", None) or {}
                    if not finish_reason:
                        finish_reason = (
                            resp_meta.get("finish_reason")
                            or resp_meta.get("finish_reasons")
                        )
                    if not model_name:
                        model_name = (
                            resp_meta.get("model_name")
                            or resp_meta.get("model")
                            or model_name
                        )

                    msg_usage = getattr(msg, "usage_metadata", None)
                    if msg_usage:
                        usage = _clean_for_json(msg_usage)
                else:
                    text = getattr(gen, "text", "")
                    outputs.append(text)

        if not usage and getattr(response, "llm_output", None):
            llm_out = response.llm_output or {}
            token_usage = llm_out.get("token_usage") or llm_out.get("usage_metadata")
            if token_usage:
                usage = _clean_for_json(token_usage)
            if not model_name:
                model_name = llm_out.get("model_name") or model_name

        entry: Dict[str, Any] = {
            "timestamp": run_info["timestamp"],
            "source": run_info["source"],
            "doc_id": run_info["doc_id"],
            "node": run_info["node"],
            "stage": run_info["stage"],
            "attempt": run_info["attempt"],
            "batch_idx": run_info["batch_idx"],
            "status": "success",
            "model": model_name,
            "provider": run_info["provider"],
            "latency_s": latency_s,
            "messages": run_info["messages"],
            "outputs": outputs,
            "tool_calls": tool_calls or None,
            "finish_reason": finish_reason,
            "usage": usage,
            "llm_run_id": run_key,
            "kwargs": run_info["kwargs"],
        }
        self.recorder.record(entry)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        update_extraction_log_context(llm_error_recorded_run_id=run_key)
        with self._lock:
            if not self._mark_seen(f"err:{run_key}"):
                return
            run_info = self._active_runs.pop(run_key, None)

        if run_info is None:
            ctx = get_extraction_log_context()
            self.recorder.record(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": ctx.get("source") or self.default_source,
                    "doc_id": ctx.get("doc_id"),
                    "node": ctx.get("node"),
                    "stage": ctx.get("stage"),
                    "attempt": ctx.get("attempt"),
                    "batch_idx": ctx.get("batch_idx"),
                    "status": "error",
                    "model": "",
                    "messages": [],
                    "outputs": [],
                    "error": format_error_payload(error),
                    "llm_run_id": run_key,
                }
            )
            return

        latency_s = round(time.perf_counter() - run_info["start_time"], 4)
        partial_output = "".join(run_info["stream_chunks"]) if run_info["stream_chunks"] else None

        entry: Dict[str, Any] = {
            "timestamp": run_info["timestamp"],
            "source": run_info["source"],
            "doc_id": run_info["doc_id"],
            "node": run_info["node"],
            "stage": run_info["stage"],
            "attempt": run_info["attempt"],
            "batch_idx": run_info["batch_idx"],
            "status": "error",
            "model": run_info["model"],
            "provider": run_info["provider"],
            "latency_s": latency_s,
            "messages": run_info["messages"],
            "outputs": [],
            "partial_output": partial_output,
            "error": format_error_payload(error),
            "llm_run_id": run_key,
            "kwargs": run_info["kwargs"],
        }
        self.recorder.record(entry)


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

    clean_history = [_clean_for_json(_clean_entry(entry)) for entry in lm.history]

    temp_path = output_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(clean_history, f, ensure_ascii=False, indent=2)
        temp_path.replace(output_path)
    except Exception:
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
    """Save student and teacher LLM histories."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    counts: Dict[str, int] = {}

    student_count = save_history(
        student_lm,
        output_dir / f"student_{timestamp}.json",
    )
    counts["student"] = student_count

    if teacher_lm is not None:
        teacher_count = save_history(
            teacher_lm,
            output_dir / f"teacher_{timestamp}.json",
        )
        counts["teacher"] = teacher_count

    return counts


def save_extraction_history(
    lm: Any,
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> int:
    """Save extraction LLM history."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"extraction_{timestamp}.json"
    return save_history(lm, output_path)


def save_semantic_optimization_history(
    lm: Any,
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> int:
    """Save semantic prompt optimization LLM history."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"semantic_{timestamp}.json"
    return save_history(lm, output_path)


def save_ingestion_history(
    visual_lm: Optional[Any],
    teacher_lm: Optional[Any],
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> Dict[str, int]:
    """Save ingestion visual extractor and teacher LLM histories."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    counts: Dict[str, int] = {}

    if visual_lm is not None:
        visual_count = save_history(
            visual_lm,
            output_dir / f"visual_{timestamp}.json",
        )
        counts["visual"] = visual_count

    if teacher_lm is not None:
        teacher_count = save_history(
            teacher_lm,
            output_dir / f"teacher_{timestamp}.json",
        )
        counts["teacher"] = teacher_count

    return counts
