# src/ae/llm/provider.py
"""LLM provider implementations for Adaptive Extractor.

This module provides LLM provider implementations that bypass litellm to avoid
JSON serialization issues during MIPROv2 optimization.

Architecture:
    BaseLMProvider (abstract)
    ├── BaseHTTPProvider (abstract)
    │   ├── OllamaLM (Ollama API)
    │   └── OpenRouterLM (OpenRouter/OpenAI-compatible API)
    └── TransformersLM (HuggingFace Transformers local inference)

Usage:
    from ae.core.llm.provider import create_lm

    lm = create_lm(config, enable_circuit_breaker=True)
    response = lm("Your prompt here")
"""

from collections import deque
import copy
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dspy
import requests

from ae.core.config.settings import CircuitBreakerConfig, LLMInstanceConfig
from ae.core.llm.circuit_breaker import CircuitBreaker, CircuitBreakerError
from ae.core.llm.history_logger import format_error_payload, get_extraction_log_context

logger = logging.getLogger(__name__)

# Global rate limiters shared across all LM instances of the same (base_url, model).
# Key: (provider base_url, model), Value: RateLimiter instance.
_PROVIDER_RATE_LIMITERS: Dict[Tuple[str, str], "RateLimiter"] = {}
_PROVIDER_RATE_LIMITERS_LOCK = threading.Lock()


from typing import Protocol, runtime_checkable

@runtime_checkable
class LMProvider(Protocol):
    """PEP-544 Protocol representing a Language Model provider."""
    model: str
    temperature: float
    max_retries: int
    top_p: float
    history: List[Dict[str, Any]]

    def clear_history(self) -> None:
        ...

    def deepcopy(self) -> "LMProvider":
        ...

    def reset_copy(self) -> "LMProvider":
        ...

    def copy(self, **kwargs) -> "LMProvider":
        ...

    def __call__(
        self,
        prompt: Optional[Union[str, List[Dict[str, str]]]] = None,
        **kwargs: Any
    ) -> List[str]:
        ...


class DSPyLMAdapter(dspy.LM):
    """Adapter that wraps an LMProvider to inherit from dspy.LM for DSPy integration."""

    def __init__(self, provider: LMProvider):
        model = getattr(provider, "model", "unknown")
        super().__init__(model=model)
        self.provider = provider
        self.model = model
        self.temperature = getattr(provider, "temperature", 0.0)
        self.max_retries = getattr(provider, "max_retries", 3)
        self.top_p = getattr(provider, "top_p", 1.0)
        
        if hasattr(provider, "kwargs"):
            self.kwargs = provider.kwargs
        else:
            self.kwargs = {}

    @property
    def history(self) -> List[Dict[str, Any]]:
        return getattr(self.provider, "history", [])

    @history.setter
    def history(self, val: List[Dict[str, Any]]) -> None:
        try:
            self.provider.history = val
        except AttributeError:
            pass

    def clear_history(self) -> None:
        if hasattr(self.provider, "clear_history"):
            self.provider.clear_history()

    def reset_circuit_breaker(self) -> None:
        if hasattr(self.provider, "reset_circuit_breaker"):
            self.provider.reset_circuit_breaker()

    def get_circuit_breaker_stats(self) -> Optional[dict]:
        if hasattr(self.provider, "get_circuit_breaker_stats"):
            return self.provider.get_circuit_breaker_stats()
        return None

    def deepcopy(self) -> "DSPyLMAdapter":
        if hasattr(self.provider, "deepcopy"):
            return DSPyLMAdapter(self.provider.deepcopy())
        return DSPyLMAdapter(self.provider)

    def reset_copy(self) -> "DSPyLMAdapter":
        if hasattr(self.provider, "reset_copy"):
            return DSPyLMAdapter(self.provider.reset_copy())
        return DSPyLMAdapter(self.provider)

    def copy(self, **kwargs) -> "DSPyLMAdapter":
        if hasattr(self.provider, "copy"):
            return DSPyLMAdapter(self.provider.copy(**kwargs))
        return DSPyLMAdapter(self.provider)

    def __call__(self, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, **kwargs) -> List[str]:
        if hasattr(self.provider, "temperature"):
            self.provider.temperature = self.temperature
        if hasattr(self.provider, "max_retries"):
            self.provider.max_retries = self.max_retries
        if hasattr(self.provider, "top_p"):
            self.provider.top_p = self.top_p
        return self.provider(prompt=prompt, **kwargs)

    def forward(self, prompt=None, **kwargs):
        if hasattr(self.provider, "temperature"):
            self.provider.temperature = self.temperature
        if hasattr(self.provider, "max_retries"):
            self.provider.max_retries = self.max_retries
        if hasattr(self.provider, "top_p"):
            self.provider.top_p = self.top_p
        return self.provider(prompt=prompt, **kwargs)


class BaseLMProvider(ABC):
    """Abstract base for all LLM providers (HTTP and non-HTTP).

    Contains shared logic used by both HTTP-based providers
    (OllamaLM, OpenRouterLM) and local inference (TransformersLM).
    """

    MAX_HISTORY = 200  # Keep only last N interactions to save RAM

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize the base provider.

        Args:
            config: Configuration for the LLM instance.
            circuit_breaker: Optional circuit breaker for failure protection.
        """
        # Store config for deepcopy
        self._config = config
        self._shared_cost = {"cumulative_cost": 0.0}

        # Common LLM parameters
        self.model = config.model
        self.temperature = config.temperature
        self.max_retries = config.max_retries
        self.top_p = config.top_p

        # Circuit breaker
        self._circuit_breaker = circuit_breaker

        # History tracking
        self._history: deque[Dict[str, Any]] = deque(maxlen=self.MAX_HISTORY)
        self._history_hook: Optional[Callable[[Dict[str, Any]], None]] = None

        self.kwargs: Dict[str, Any] = {}

    @property
    def cumulative_cost(self) -> float:
        """Get the cumulative cost of requests made by this provider."""
        return self._shared_cost["cumulative_cost"]

    @cumulative_cost.setter
    def cumulative_cost(self, val: float) -> None:
        """Set the cumulative cost."""
        self._shared_cost["cumulative_cost"] = val

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get the interaction history as a list."""
        return list(self._history)

    @history.setter
    def history(self, val: Any) -> None:
        """Set the interaction history (converts to deque)."""
        self._history = deque(val or [], maxlen=self.MAX_HISTORY)

    def set_history_hook(self, hook: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Attach or clear a callback invoked on each recorded interaction."""
        self._history_hook = hook

    def _update_history(
        self,
        messages: List[Dict[str, Any]],
        response: str,
        kwargs: Dict[str, Any],
        latency_s: Optional[float] = None,
        status: str = "success",
        error: Optional[BaseException] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the history with the latest interaction and notify hook if set."""
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "messages"}
        ctx = get_extraction_log_context()

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
            "outputs": [response] if status == "success" else [],
            "model": self.model,
            "status": status,
            "kwargs": kwargs_clean,
        }
        provider_name = getattr(self, "provider", None)
        if provider_name:
            entry["provider"] = provider_name
        if latency_s is not None:
            entry["latency_s"] = latency_s
        if usage is None and hasattr(self, "_thread_local"):
            usage = getattr(self._thread_local, "last_usage", None)
        if isinstance(usage, dict) and usage:
            entry["usage"] = usage
        if error is not None:
            entry["error"] = format_error_payload(error)
        if ctx.get("doc_id"):
            entry["doc_id"] = ctx["doc_id"]
        if ctx.get("source"):
            entry["source"] = ctx["source"]
        else:
            entry["source"] = "lm_provider"
        if ctx.get("stage"):
            entry["stage"] = ctx["stage"]
        if ctx.get("node"):
            entry["node"] = ctx["node"]

        self._history.append(entry)
        if self._history_hook is not None:
            try:
                self._history_hook(entry)
            except Exception as hook_exc:
                logger.warning(f"History hook failed: {hook_exc}")

    def clear_history(self) -> None:
        """Clear the interaction history."""
        self._history.clear()

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
            logger.info(f"Reset circuit breaker for {self.model}")

    def get_circuit_breaker_stats(self) -> Optional[dict]:
        """Get circuit breaker statistics."""
        if self._circuit_breaker:
            return self._circuit_breaker.get_stats()
        return None

    def deepcopy(self):
        """Create a deep copy of this LM instance."""
        cb_copy = copy.deepcopy(self._circuit_breaker) if self._circuit_breaker else None
        new_instance = self.__class__(self._config, circuit_breaker=cb_copy)
        new_instance._history = copy.deepcopy(self._history)
        new_instance._shared_cost = self._shared_cost
        return new_instance

    def reset_copy(self):
        """Create a copy with same config but empty history."""
        cb_copy = copy.deepcopy(self._circuit_breaker) if self._circuit_breaker else None
        copy_instance = self.__class__(self._config, circuit_breaker=cb_copy)
        copy_instance._shared_cost = self._shared_cost
        return copy_instance

    def copy(self, **kwargs):
        """Create a copy sharing history with the original (for MIPROv2)."""
        cb_copy = copy.deepcopy(self._circuit_breaker) if self._circuit_breaker else None
        new_instance = self.__class__(self._config, circuit_breaker=cb_copy)
        new_instance._history = self._history  # Share history
        new_instance._history_hook = self._history_hook
        new_instance._shared_cost = self._shared_cost

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value

        if hasattr(new_instance, "_warned_zero_temp_rollout"):
            new_instance._warned_zero_temp_rollout = False

        return new_instance

    def __call__(self, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, **kwargs) -> List[str]:
        """Call the LLM. Fully overrides dspy.LM.__call__ via Python MRO.

        In DSPy 3.x, dspy.LM.__call__ is decorated with @with_callbacks and calls
        self.forward() + self._process_lm_response(). Since our providers return
        List[str] (not a LiteLLM ModelResponse), we must fully replace __call__
        rather than overriding forward() alone, which would break _process_lm_response.

        Python MRO ensures our __call__ is found first when DSPy resolves the method
        on the instance, so dspy.context() / dspy.settings.lm calls go through here.
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def forward(self, prompt=None, **kwargs):  # type: ignore[override]
        """Stub: not used. Our __call__ fully replaces dspy.LM.__call__ via MRO."""
        # Called only if dspy internals somehow invoke forward() directly.
        return self(prompt=prompt, **kwargs)


class BaseHTTPProvider(BaseLMProvider, ABC):
    """Abstract base class for HTTP-based LLM providers."""

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        super().__init__(config, circuit_breaker=circuit_breaker)

        self.timeout = config.timeout
        self.provider: str = ""
        self.base_url: str = ""
        self._reasoning_details: Optional[List[Dict[str, Any]]] = None

        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

        # Initialize rate limiter if configured
        self._rate_limiter: Optional[RateLimiter] = None
        if config.rate_limit_delay is not None and config.rate_limit_delay > 0:
            if config.provider == "ollama" and config.ollama:
                b_url = config.ollama.ollama_base_url or "http://localhost:11434"
                provider_url = b_url.rstrip("/") + "/api/chat"
            elif config.provider == "api" and config.api:
                provider_url = (config.api.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
            else:
                provider_url = f"{config.provider}"
            self._rate_limiter = _get_provider_rate_limiter(
                provider_url, config.rate_limit_delay, model=config.model
            )

        # Thread-local storage for latency tracking
        import threading
        self._thread_local = threading.local()

    def __call__(self, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, **kwargs) -> List[str]:
        """Execute HTTP request to the LLM provider.

        Fully overrides dspy.LM.__call__ via Python MRO so that DSPy 3.x internal
        calls (e.g. inside dspy.context()) are routed here instead of going to
        dspy.LM.forward() + LiteLLM.
        """
        messages = kwargs.pop("messages", None)
        if prompt is None:
            prompt = messages
        if prompt is None:
            return [""]

        messages = self._normalize_prompt(prompt)
        kwargs_copy = {k: v for k, v in kwargs.items() if k != "messages"}
        payload = self._prepare_payload(messages, **kwargs_copy)
        call_start = time.perf_counter()
        if hasattr(self, "_thread_local"):
            self._thread_local.last_latency_s = None
            self._thread_local.last_usage = None
        try:
            text_response = self._execute_request(payload)
        except Exception as exc:
            latency_s = getattr(self._thread_local, "last_latency_s", None)
            if latency_s is None:
                latency_s = round(time.perf_counter() - call_start, 4)
            self._update_history(
                messages,
                "",
                kwargs,
                latency_s=latency_s,
                status="error",
                error=exc,
                usage=getattr(self._thread_local, "last_usage", None),
            )
            raise
        latency_s = getattr(self._thread_local, "last_latency_s", None)
        self._update_history(
            messages,
            text_response,
            kwargs,
            latency_s=latency_s,
            usage=getattr(self._thread_local, "last_usage", None),
        )
        return [text_response]

    def _normalize_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        if self._reasoning_details is not None:
            reasoning_details = self._reasoning_details
            enhanced_messages: List[Dict[str, Any]] = []
            for i, msg in enumerate(prompt):
                enhanced_msg: Dict[str, Any] = msg.copy()
                if msg.get("role") == "assistant" and reasoning_details and i == len(prompt) - 1:
                    enhanced_msg["reasoning_details"] = reasoning_details
                enhanced_messages.append(enhanced_msg)
            self._reasoning_details = None
            return enhanced_messages

        return prompt

    @abstractmethod
    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        pass

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        attempt = 0
        last_exception: Optional[Exception] = None
        if hasattr(self, "_thread_local"):
            self._thread_local.last_latency_s = None
            self._thread_local.last_usage = None

        while attempt < self.max_retries:
            if self._rate_limiter:
                self._rate_limiter.wait()

            logger.info(f"[{self.provider}] Sending request to {self.model} (Attempt {attempt + 1}/{self.max_retries})...")
            start_time = time.perf_counter()
            try:
                if self._circuit_breaker:
                    res = self._circuit_breaker.call(self._make_request, payload)
                else:
                    res = self._make_request(payload)

                duration = time.perf_counter() - start_time
                logger.info(f"Request completed in {duration:.2f}s")
                if hasattr(self, "_thread_local"):
                    self._thread_local.last_latency_s = duration
                if self._rate_limiter:
                    self._rate_limiter.update_last_call_time()
                return res
            except CircuitBreakerError:
                timeout = self._circuit_breaker.reset_timeout if self._circuit_breaker else "unknown"
                logger.warning(
                    f"Circuit breaker OPEN for {self.model}. "
                    f"Retry after {timeout}s."
                )
                raise
            except Exception as e:
                last_exception = e
                attempt += 1
                logger.warning(f"{self.provider} error (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    import random
                    sleep_time = (2 ** attempt) + random.random()
                    time.sleep(sleep_time)

        if last_exception:
            logger.error(f"{self.provider} failed after {self.max_retries} retries: {last_exception}")
            raise last_exception
        else:
            raise RuntimeError(f"{self.provider} request failed without exception")

    @abstractmethod
    def _make_request(self, payload: Dict[str, Any]) -> str:
        pass


class OllamaLM(BaseHTTPProvider):
    """LLM provider for Ollama with circuit breaker protection."""

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        if circuit_breaker is None:
            raise ValueError("circuit_breaker is required")

        super().__init__(config, circuit_breaker=circuit_breaker)

        oc = config.ollama
        if oc is None:
            raise ValueError(
                "Ollama configuration (ollama section) is required when using "
                "the Ollama provider."
            )
        self.num_ctx = oc.num_ctx
        self.num_predict = oc.num_predict
        self.stream = oc.stream
        self.repeat_penalty = oc.repeat_penalty
        self.repeat_last_n = oc.repeat_last_n
        self.provider = "Ollama"

        if not oc.ollama_base_url:
            raise ValueError(
                "OLLAMA_BASE_URL environment variable must be set in .env file. "
                "Set OLLAMA_STUDENT_BASE_URL or OLLAMA_TEACHER_BASE_URL as appropriate."
            )
        self.base_url = oc.ollama_base_url.rstrip("/") + "/api/chat"

    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "repeat_last_n": self.repeat_last_n,
            },
            "stream": self.stream
        }

    def _make_request(self, payload: Dict[str, Any]) -> str:
        try:
            with requests.post(
                self.base_url,
                json=payload,
                stream=self.stream,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()

                full_content = []
                if self.stream:
                    logger.info(f"[LLM] Streaming response from {self.model}...")

                for line in response.iter_lines():
                    if line:
                        try:
                            body = json.loads(line)
                            if "message" in body and "content" in body["message"]:
                                content_chunk = body["message"]["content"]
                                full_content.append(content_chunk)
                                if self.stream:
                                    print(content_chunk, end='', flush=True)
                            if body.get("done", False):
                                if self.stream:
                                    print()
                                break
                        except json.JSONDecodeError:
                            logger.warning("Failed to decode JSON response line")
                            continue

                return "".join(full_content)
        except requests.Timeout:
            logger.error(f"Request to Ollama timed out after {self.timeout} seconds")
            raise
        except requests.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise
        except requests.HTTPError as e:
            logger.error(f"Ollama API returned HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Ollama request: {e}")
            raise


_CACHE_SPLIT_DELIMITERS = (
    "[[ ## document_text ## ]]\n",
    "[[ ## target_gt_row ## ]]\n",
    "[[ ## target_gt_rows ## ]]\n",
    "[[ ## generalized_trigger ## ]]\n",
    "[[ ## mismatched_rows_context ## ]]\n",
    "[[ ## missed_gt_rows ## ]]\n",
    "[[ ## false_positive_rows ## ]]\n",
)

_RE_BATCH_DELIMITERS = (
    "[[ ## target_gt_row ## ]]\n",
    "[[ ## target_gt_rows ## ]]\n",
    "[[ ## generalized_trigger ## ]]\n",
    "[[ ## mismatched_rows_context ## ]]\n",
    "[[ ## missed_gt_rows ## ]]\n",
    "[[ ## false_positive_rows ## ]]\n",
)


def apply_prompt_caching(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply native prompt caching headers/structure to messages.
    
    For a single DSPy string prompt, splits at the dynamic suffix delimiter
    (e.g. '[[ ## document_text ## ]]\n', '[[ ## target_gt_row ## ]]\n',
    '[[ ## target_gt_rows ## ]]\n', or '[[ ## generalized_trigger ## ]]\n')
    to separate the static signature/context prefix from the dynamic suffix.
    Adds cache_control blocks for Anthropic/OpenRouter.
    """
    if not messages:
        return messages

    # Case 1: Single message containing the DSPy document or RE target-row delimiter
    if (
        len(messages) == 1
        and messages[0].get("role") in ("system", "user")
        and isinstance(messages[0].get("content"), str)
    ):
        content = messages[0]["content"]
        for delimiter in _CACHE_SPLIT_DELIMITERS:
            if delimiter in content:
                # Split at the last occurrence to ensure all static few-shot demos
                # and shared document context are included in static_prefix and cached.
                parts = content.rsplit(delimiter, 1)
                static_prefix = parts[0] + delimiter
                dynamic_suffix = parts[1]
                logger.info("Detected DSPy prompt with document delimiter. Splitting for prompt caching.")
                return [
                    {
                        "role": messages[0]["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": static_prefix,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": dynamic_suffix,
                    },
                ]

    # Case 2: Standard message list. Cache the first message (e.g. system prompt)
    # and split any user message containing an RE batch/target delimiter.
    new_messages = []
    for i, msg in enumerate(messages):
        content_val = msg.get("content", "")
        matched_delimiter = (
            next((d for d in _RE_BATCH_DELIMITERS if d in content_val), None)
            if msg.get("role") == "user" and isinstance(content_val, str)
            else None
        )
        if i == 0:
            if isinstance(content_val, str):
                new_msg = {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": content_val,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
                new_messages.append(new_msg)
            elif isinstance(content_val, list):
                new_content = []
                for block in content_val:
                    if isinstance(block, dict) and block.get("type") == "text":
                        new_block = block.copy()
                        new_block["cache_control"] = {"type": "ephemeral"}
                        new_content.append(new_block)
                    else:
                        new_content.append(block)
                new_msg = msg.copy()
                new_msg["content"] = new_content
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        elif matched_delimiter is not None:
            parts = content_val.rsplit(matched_delimiter, 1)
            static_prefix = parts[0] + matched_delimiter
            dynamic_suffix = parts[1]
            new_messages.append(
                {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": static_prefix,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": dynamic_suffix,
                        },
                    ],
                }
            )
        else:
            new_messages.append(msg)

    return new_messages


def model_supports_sampling_params(model_id: Optional[str]) -> bool:
    """Return False for reasoning-only model slugs that reject temperature/top_p on OpenRouter."""
    if not model_id or not isinstance(model_id, str):
        return True
    slug = model_id.strip().lower()
    if slug.startswith("openai/"):
        tail = slug.split("/", 1)[1]
        if tail.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
            return False
    if slug.startswith("anthropic/"):
        tail = slug.split("/", 1)[1]
        if any(tok in tail for tok in ("-5", "fable-5")):
            return False
    return True


MODEL_PRICING = {
    "anthropic/claude-3-5-sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "anthropic/claude-3-sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "anthropic/claude-sonnet-4.6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "anthropic/claude-sonnet-5.5": {"input": 2.0, "output": 10.0, "cache_read": 0.1, "cache_write": 2.5},
    "anthropic/claude-haiku-5.5": {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125},
    "google/gemini-2.5-pro": {"input": 1.25, "output": 5.0, "cache_read": 0.125},
    "google/gemini-2.5-flash": {"input": 0.075, "output": 0.3, "cache_read": 0.0075},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cache_read": 0.075},
    "google/gemini-3.8-flash": {"input": 0.825, "output": 4.125, "cache_read": 0.0825},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75, "cache_read": 0.075},
    "google/gemini-3.7-flash": {"input": 0.75, "output": 3.75, "cache_read": 0.075},
    "gemini-4-hydrogen": {"input": 1.5, "output": 7.5, "cache_read": 0.075},
    "google/gemini-4-hydrogen": {"input": 1.5, "output": 7.5, "cache_read": 0.075},
    "gemini-4-neon": {"input": 3.0, "output": 15.0, "cache_read": 0.15},
    "google/gemini-4-neon": {"input": 3.0, "output": 15.0, "cache_read": 0.15},
    "google/gemma-4-31b-it": {"input": 0.2, "output": 0.4, "cache_read": 0.2},
    "meta-llama/llama-3-70b-instruct": {"input": 0.59, "output": 0.79, "cache_read": 0.059},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28, "cache_read": 0.014},
    "deepseek/deepseek-v4.1-flash": {"input": 0.0498, "output": 0.1992, "cache_read": 0.000996},
    "deepseek/deepseek-v4-pro": {"input": 0.9, "output": 1.5, "cache_read": 0.2},
    "openai/gpt-6-luna": {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125},
    "openai/gpt-6.1-terra": {"input": 1.0, "output": 5.0, "cache_read": 0.05, "cache_write": 1.25},
    "openai/gpt-6.1-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.1, "cache_write": 2.5},
    "qwen/qwen3.8-flash": {"input": 0.15, "output": 0.47, "cache_read": 0.016},
    "qwen/qwen3-vl-235b-a22b-instruct": {"input": 0.21, "output": 1.9, "cache_read": 0.1},
    "qwen/qwen3.5-397b-a17b": {"input": 0.45, "output": 3.0, "cache_read": 0.22},
    "z-ai/glm-5.3-flash": {"input": 0.15, "output": 0.5, "cache_read": 0.03},
    "inclusionai/ling-3.0-flash-vl": {"input": 0.06, "output": 0.18, "cache_read": 0.012},
}


class OpenRouterLM(BaseHTTPProvider):
    """LLM provider for OpenRouter with direct HTTP calls."""

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        super().__init__(config, circuit_breaker=circuit_breaker)

        api_cfg = config.api
        if api_cfg is None:
            raise ValueError(
                "API configuration (api section) is required when using "
                "the OpenRouter provider."
            )
        self.max_tokens = api_cfg.max_tokens
        self.provider = "OpenRouter"
        self.reasoning = api_cfg.reasoning
        self.openrouter_cache = getattr(api_cfg, "openrouter_cache", None)
        self.prompt_caching = getattr(api_cfg, "prompt_caching", None)
        self.prompt_cache_options = getattr(api_cfg, "prompt_cache_options", None)
        import uuid
        self.session_id = getattr(api_cfg, "session_id", None) or f"ae-{uuid.uuid4().hex[:12]}"
        self.provider_preferences = getattr(api_cfg, "provider", None)
        self.response_format = getattr(api_cfg, "response_format", None)

        if api_cfg.api_key is None:
            raise ValueError(
                "API key must be set for OpenRouter. "
                "Set OPENROUTER_API_KEY in .env file."
            )
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        self.api_key = api_cfg.api_key.get_secret_value()
        self.base_url = (api_cfg.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"

    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Pop caching parameters so they are not sent to standard OpenRouter API
        prompt_caching = kwargs.pop("prompt_caching", self.prompt_caching)
        prompt_cache_options = kwargs.pop("prompt_cache_options", self.prompt_cache_options)
        openrouter_cache = kwargs.pop("openrouter_cache", self.openrouter_cache)
        session_id = kwargs.pop("session_id", self.session_id)
        provider = kwargs.pop("provider", self.provider_preferences)

        if prompt_caching:
            messages = apply_prompt_caching(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if model_supports_sampling_params(self.model):
            payload["temperature"] = temperature
            payload["top_p"] = kwargs.get("top_p", self.top_p)

        if prompt_cache_options is not None:
            payload["prompt_cache_options"] = prompt_cache_options

        reasoning = kwargs.get("reasoning", self.reasoning)
        if reasoning is not None:
            payload["reasoning"] = reasoning

        response_format = kwargs.pop("response_format", self.response_format)
        if response_format is not None:
            payload["response_format"] = response_format

        if session_id is not None:
            payload["session_id"] = session_id

        if provider is not None:
            from pydantic import BaseModel
            if isinstance(provider, BaseModel):
                payload["provider"] = provider.model_dump(by_alias=True, exclude_none=True)
            elif isinstance(provider, dict):
                mapped_provider = {}
                for k, v in provider.items():
                    if k == "priority_order" or k == "order":
                        mapped_provider["order"] = v
                    elif k == "require_parameter_support" or k == "require_parameters":
                        mapped_provider["require_parameters"] = v
                    else:
                        mapped_provider[k] = v
                payload["provider"] = mapped_provider

        payload = {k: v for k, v in payload.items() if v is not None}
        
        if openrouter_cache is not None:
            payload["_openrouter_cache"] = openrouter_cache

        return payload

    def _make_request(self, payload: Dict[str, Any]) -> str:
        openrouter_cache = payload.pop("_openrouter_cache", None)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/adaptive-extractor/adaptive-extractor",
            "X-Title": "Adaptive Extractor",
        }
        if openrouter_cache:
            headers["X-OpenRouter-Cache"] = "true"
            logger.info("OpenRouter response cache enabled for this request")

        try:
            with requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                try:
                    data = response.json()
                except json.JSONDecodeError as je:
                    logger.error(
                        f"Failed to parse JSON response from OpenRouter. "
                        f"Status code: {response.status_code}. "
                        f"Content preview (first 1000 chars): {response.text[:1000]!r}"
                    )
                    raise je

                if "choices" in data and len(data["choices"]) > 0:
                    provider_name = data.get("provider")
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                        total_tokens = int(
                            usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
                        )
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
                        cache_write_tokens = int(prompt_details.get("cache_write_tokens", 0) or 0)
                        completion_details = usage.get("completion_tokens_details") or {}
                        reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)

                        log_msg = (
                            f"OpenRouter usage for {self.model} (provider={provider_name}): "
                            f"prompt_tokens={prompt_tokens} (cached={cached_tokens}, write={cache_write_tokens}), "
                            f"completion_tokens={completion_tokens}"
                        )
                        logger.info(log_msg)

                        # Cost calculation ($)
                        input_price = getattr(self._config, "input_price_per_1m", None)
                        output_price = getattr(self._config, "output_price_per_1m", None)
                        cache_read_price = getattr(self._config, "cache_read_price_per_1m", None)

                        if input_price is None or output_price is None:
                            pricing = MODEL_PRICING.get(self.model, {"input": 1.0, "output": 1.0, "cache_read": 0.5})
                            input_price = pricing["input"]
                            output_price = pricing["output"]
                            cache_read_price = pricing.get("cache_read", input_price * 0.1)

                        non_cached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
                        cost = (
                            (non_cached_prompt_tokens * input_price) +
                            (cached_tokens * (cache_read_price or (input_price * 0.1))) +
                            (completion_tokens * output_price)
                        ) / 1_000_000.0

                        self.cumulative_cost += cost
                        logger.info(f"Request cost: ${cost:.6f} | Cumulative: ${self.cumulative_cost:.6f}")

                        usage_payload: Dict[str, Any] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "cached_tokens": cached_tokens,
                            "cache_write_tokens": cache_write_tokens,
                            "reasoning_tokens": reasoning_tokens,
                            "cost": round(cost, 8),
                        }
                        if usage.get("cost") is not None:
                            usage_payload["provider_cost"] = usage.get("cost")
                        if provider_name:
                            usage_payload["provider"] = provider_name
                        if hasattr(self, "_thread_local"):
                            self._thread_local.last_usage = usage_payload

                    message = data["choices"][0]["message"]
                    content = message.get("content", "")
                    self._reasoning_details = message.get("reasoning_details")
                    return content
                else:
                    logger.error(f"Unexpected OpenRouter response: {data}")
                    raise ValueError("Empty or invalid response from OpenRouter")

        except requests.Timeout:
            logger.error(f"Request to OpenRouter timed out after {self.timeout} seconds")
            raise
        except requests.ConnectionError as e:
            logger.error(f"Failed to connect to OpenRouter at {self.base_url}: {e}")
            raise
        except requests.HTTPError as e:
            logger.error(f"OpenRouter API returned HTTP error: {e}")
            try:
                error_data = e.response.json()
                logger.error(f"Error details: {error_data}")
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OpenRouter request: {e}")
            raise


class GeminiLM(BaseHTTPProvider):
    """LLM provider for Google Gemini with circuit breaker, rate limiting, and thinking support."""

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        super().__init__(config, circuit_breaker=circuit_breaker)

        gemini_cfg = config.gemini
        api_key_str: Optional[str] = None
        if gemini_cfg and gemini_cfg.api_key:
            api_key_str = gemini_cfg.api_key.get_secret_value()
        if not api_key_str:
            import os
            api_key_str = os.getenv("GEMINI_API_KEY")

        if not api_key_str:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY in environment or config."
            )

        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=api_key_str)
        self.provider = "Gemini"
        self.max_output_tokens = (
            getattr(gemini_cfg, "max_output_tokens", 65536) if gemini_cfg else 65536
        )
        self.thinking_level = (
            getattr(gemini_cfg, "thinking_level", None) if gemini_cfg else None
        )
        self.thinking_budget = (
            getattr(gemini_cfg, "thinking_budget", None) if gemini_cfg else None
        )

    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = str(content)
            elif role == "assistant":
                contents.append(
                    self._genai.types.Content(
                        role="model",
                        parts=[self._genai.types.Part.from_text(text=str(content))],
                    )
                )
            else:
                contents.append(
                    self._genai.types.Content(
                        role="user",
                        parts=[self._genai.types.Part.from_text(text=str(content))],
                    )
                )

        if not contents and system_instruction:
            contents.append(
                self._genai.types.Content(
                    role="user",
                    parts=[self._genai.types.Part.from_text(text=system_instruction)],
                )
            )
            system_instruction = None

        thinking_config = None
        thinking_level = kwargs.get("thinking_level", self.thinking_level)
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)

        if thinking_level:
            thinking_config = self._genai.types.ThinkingConfig(
                thinking_level=thinking_level.lower()
            )
        elif thinking_budget is not None:
            thinking_config = self._genai.types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)

        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config

        gen_config = self._genai.types.GenerateContentConfig(**config_kwargs)

        return {
            "contents": contents,
            "config": gen_config,
        }

    def _make_request(self, payload: Dict[str, Any]) -> str:
        contents = payload["contents"]
        config = payload["config"]

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        if response is None:
            raise ValueError(f"Gemini API returned None response for model '{self.model}'")

        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            reasoning_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
            cached_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
            total_tokens = int(
                getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
            )
            in_price = (getattr(self._config, "input_price_per_1m", None) if self._config else None) or 0.0
            out_price = (getattr(self._config, "output_price_per_1m", None) if self._config else None) or 0.0
            cache_read_price = (
                getattr(self._config, "cache_read_price_per_1m", None) if self._config else None
            )
            if cache_read_price is None:
                cache_read_price = in_price * 0.1
            non_cached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
            cost = (
                (non_cached_prompt_tokens * in_price)
                + (cached_tokens * cache_read_price)
                + (completion_tokens * out_price)
            ) / 1_000_000.0
            if self._config:
                self.cumulative_cost += cost
                logger.info(
                    f"Gemini usage for {self.model}: {prompt_tokens} input + {completion_tokens} output tokens "
                    f"(${cost:.6f}, cumulative: ${self.cumulative_cost:.6f})"
                )
            if hasattr(self, "_thread_local"):
                self._thread_local.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "cost": round(cost, 8),
                }

        text = getattr(response, "text", None)
        if text is None:
            candidates = getattr(response, "candidates", None)
            if candidates:
                parts = getattr(getattr(candidates[0], "content", None), "parts", None)
                if parts:
                    text = "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
        return text or ""


class TeacherWrapper(dspy.Module):
    """Wrapper to use LLM providers as teacher for MIPROv2 bootstrapping."""

    def __init__(self, signature_class: Type[dspy.Signature], teacher_lm: dspy.LM):
        super().__init__()
        self.signature_class = signature_class
        self.teacher_lm = teacher_lm
        self.prog = dspy.ChainOfThought(signature_class, lm=teacher_lm)

    def forward(self, document_text: str) -> dspy.Prediction:
        return self.prog(document_text=document_text)

    def predictors(self) -> List[dspy.Predict]:
        return [self.prog.predict]

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


class RateLimiter:
    """Thread-safe rate limiter for LLM instances."""

    def __init__(self, delay: float):
        if delay < 0:
            raise ValueError("Delay cannot be negative")
        self.delay = delay
        self.lock = Lock()
        self.last_call_time: Optional[float] = None

    def __deepcopy__(self, memo) -> 'RateLimiter':
        # Share the exact same RateLimiter instance on deepcopy so that copied LM instances
        # continue to share the same rate limits for the endpoint.
        return self

    def __copy__(self) -> 'RateLimiter':
        return self

    def wait(self) -> None:
        """Wait if the rate limit delay has not elapsed since the last call."""
        with self.lock:
            if self.last_call_time is not None and self.delay > 0:
                elapsed = time.monotonic() - self.last_call_time
                if elapsed < self.delay:
                    sleep_time = self.delay - elapsed
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s before request")
                    time.sleep(sleep_time)

    def update_last_call_time(self) -> None:
        """Update the timestamp of the last successful call."""
        with self.lock:
            self.last_call_time = time.monotonic()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait()
            try:
                result = func(*args, **kwargs)
                self.update_last_call_time()
                return result
            except Exception:
                raise
        return wrapper


def _get_provider_rate_limiter(
    base_url: str, delay: float, model: Optional[str] = None
) -> RateLimiter:
    """Get or create a global rate limiter for a (base_url, model) pair.

    Args:
        base_url: The provider's base URL.
        delay: Rate limit delay in seconds.
        model: Model identifier so different models on the same gateway do not share limits.

    Returns:
        Shared RateLimiter instance for this (base_url, model).
    """
    key = (base_url, model or "")
    with _PROVIDER_RATE_LIMITERS_LOCK:
        if key not in _PROVIDER_RATE_LIMITERS:
            _PROVIDER_RATE_LIMITERS[key] = RateLimiter(delay)
            logger.debug(f"Created global rate limiter for {key} (delay={delay}s)")
        return _PROVIDER_RATE_LIMITERS[key]


class LLMProviderRegistry:
    """Registry for LLM providers (Open-Closed Principle compliant)."""

    def __init__(self) -> None:
        self._providers: Dict[str, Type[LMProvider]] = {}

    def register(self, name: str, provider_class: Type[LMProvider]) -> None:
        """Register an LLM provider class."""
        self._providers[name.lower()] = provider_class
        logger.debug(f"Registered LLM provider '{name}' with class '{provider_class.__name__}'")

    def create_lm(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> LMProvider:
        """Create a language model instance using the registered provider."""
        provider_name = config.provider.lower()
        if provider_name not in self._providers:
            raise ValueError(
                f"Unknown provider: {config.provider}. Registered providers: {sorted(list(self._providers.keys()))}"
            )
        provider_class = self._providers[provider_name]
        return provider_class(config, circuit_breaker=circuit_breaker)


# Global registry singleton
LLM_PROVIDER_REGISTRY = LLMProviderRegistry()

# Register default providers
LLM_PROVIDER_REGISTRY.register("ollama", OllamaLM)
LLM_PROVIDER_REGISTRY.register("api", OpenRouterLM)
LLM_PROVIDER_REGISTRY.register("gemini", GeminiLM)


def create_lm(
    config: LLMInstanceConfig,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> LMProvider:
    """Create a language model instance.

    Args:
        config: Configuration for the LLM instance.
        circuit_breaker_config: Circuit breaker configuration.
        enable_circuit_breaker: Whether to enable circuit breaker.
        enable_cache: Override config's enable_cache setting (optional).

    Returns:
        LMProvider: Language model instance.

    Raises:
        ValueError: If configuration is invalid.
    """
    if not config.model:
        raise ValueError("Model name cannot be empty")

    logger.info(f"Initializing LLM: {config.model} (provider: {config.provider})")

    # NOTE: dspy.configure_cache() must NOT be called here.
    # Each call creates a new FanoutCache(shards=16) on ~/.dspy_cache/, and calling
    # it twice (once for student, once for teacher) causes SQLite lock contention
    # across all 16 shards, resulting in a 5-6 minute freeze during MIPROv2 Step 2.
    # Cache must be configured exactly once, before any LM is created (e.g. in
    # setup_language_models or at CLI startup).
    use_cache = enable_cache if enable_cache is not None else config.enable_cache
    logger.debug(f"LM cache setting: {'enabled' if use_cache else 'disabled'} (managed externally)")

    circuit_breaker = None
    if enable_circuit_breaker:
        if circuit_breaker_config is None:
            raise ValueError("circuit_breaker_config is required when enable_circuit_breaker is True")
        failure_threshold = circuit_breaker_config.failure_threshold
        reset_timeout = circuit_breaker_config.reset_timeout

        provider_name = config.provider
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            half_open_max_calls=circuit_breaker_config.half_open_max_calls,
            name=f"{provider_name}-{config.model}",
        )
        logger.info(
            f"Circuit breaker enabled for {config.model} "
            f"(threshold={failure_threshold}, "
            f"timeout={reset_timeout}s)"
        )

    lm = LLM_PROVIDER_REGISTRY.create_lm(config, circuit_breaker=circuit_breaker)

    # Note: rate limiter is now initialized internally inside BaseHTTPProvider.__init__
    # and applied during _execute_request, which correctly integrates with DSPy's
    # class-level __call__ dispatch mechanism.
    return lm


def setup_student(
    llm_config: LLMInstanceConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> LMProvider:
    """Set up the student language model and configure DSPy globally."""
    if llm_config is None:
        raise ValueError("llm_config is required for setup_student")

    lm = create_lm(
        llm_config,
        circuit_breaker_config=circuit_breaker_config,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    dspy.settings.configure(lm=DSPyLMAdapter(lm))
    logger.info(f"Student LLM configured: {llm_config.model}")
    return lm


def setup_teacher(
    llm_config: LLMInstanceConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> LMProvider:
    """Set up the teacher language model."""
    if llm_config is None:
        raise ValueError("llm_config is required for setup_teacher")

    lm = create_lm(
        llm_config,
        circuit_breaker_config=circuit_breaker_config,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    logger.info(f"Teacher LLM configured: {llm_config.model}")
    return lm


def setup_visual_extractor(
    llm_config: LLMInstanceConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> LMProvider:
    """Set up the visual extractor language model."""
    if llm_config is None:
        raise ValueError("llm_config is required for setup_visual_extractor")

    lm = create_lm(
        llm_config,
        circuit_breaker_config=circuit_breaker_config,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    logger.info(f"VisualExtractor LLM configured: {llm_config.model}")
    return lm
