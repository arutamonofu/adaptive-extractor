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
from functools import wraps
from threading import Lock
from typing import Any, Dict, List, Optional, Type, Union

import dspy
import requests

from ae.core.config.settings import CircuitBreakerConfig, LLMInstanceConfig
from ae.core.llm.circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)

# Global rate limiters shared across all LM instances of the same provider.
# Key: provider base_url, Value: RateLimiter instance.
_PROVIDER_RATE_LIMITERS: Dict[str, "RateLimiter"] = {}
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

    def _update_history(self, messages: List[Dict[str, Any]], response: str, kwargs: Dict[str, Any], latency_s: Optional[float] = None) -> None:
        """Update the history with the latest interaction."""
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "messages"}

        entry = {
            "messages": messages,
            "outputs": [response],
            "model": self.model,
            "kwargs": kwargs_clean
        }
        if latency_s is not None:
            entry["latency_s"] = latency_s
        self._history.append(entry)

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
                provider_url = f"{config.provider}-{config.model}"
            self._rate_limiter = _get_provider_rate_limiter(provider_url, config.rate_limit_delay)

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
        text_response = self._execute_request(payload)
        latency_s = getattr(self._thread_local, "last_latency_s", None)
        self._update_history(messages, text_response, kwargs, latency_s=latency_s)
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


def apply_prompt_caching(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply native prompt caching headers/structure to messages.
    
    For a single DSPy string prompt, splits at the '[[ ## document_text ## ]]\n'
    delimiter to separate the static signature and instructions from the dynamic 
    document content. Adds cache_control blocks for Anthropic.
    """
    if not messages:
        return messages

    # Case 1: Single message containing the DSPy document delimiter
    if (
        len(messages) == 1 
        and messages[0].get("role") in ("system", "user") 
        and isinstance(messages[0].get("content"), str)
    ):
        content = messages[0]["content"]
        delimiter = "[[ ## document_text ## ]]\n"
        if delimiter in content:
            # Split at the last occurrence to ensure all static few-shot demos
            # are included in the static_prefix and successfully cached.
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
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": dynamic_suffix
                }
            ]

    # Case 2: Standard message list. Cache the first message (e.g. system prompt)
    new_messages = []
    for i, msg in enumerate(messages):
        if i == 0:
            content_val = msg.get("content", "")
            if isinstance(content_val, str):
                new_msg = {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": content_val,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
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
        else:
            new_messages.append(msg)

    return new_messages


MODEL_PRICING = {
    "anthropic/claude-3-5-sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "anthropic/claude-3-sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "google/gemini-2.5-pro": {"input": 1.25, "output": 5.0, "cache_read": 0.125},
    "google/gemini-2.5-flash": {"input": 0.075, "output": 0.3, "cache_read": 0.0075},
    "meta-llama/llama-3-70b-instruct": {"input": 0.59, "output": 0.79, "cache_read": 0.059},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28, "cache_read": 0.014},
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
        import uuid
        self.session_id = getattr(api_cfg, "session_id", None) or f"ae-{uuid.uuid4().hex[:12]}"
        self.provider_preferences = getattr(api_cfg, "provider", None)

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
        openrouter_cache = kwargs.pop("openrouter_cache", self.openrouter_cache)
        session_id = kwargs.pop("session_id", self.session_id)
        provider = kwargs.pop("provider", self.provider_preferences)

        if prompt_caching:
            messages = apply_prompt_caching(messages)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", self.top_p),
        }

        reasoning = kwargs.get("reasoning", self.reasoning)
        if reasoning is not None:
            payload["reasoning"] = reasoning

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
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        prompt_details = usage.get("prompt_tokens_details", {})
                        cached_tokens = prompt_details.get("cached_tokens", 0)
                        cache_write_tokens = prompt_details.get("cache_write_tokens", 0)
                        
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


def _get_provider_rate_limiter(base_url: str, delay: float) -> RateLimiter:
    """Get or create a global rate limiter for a provider base_url.

    Args:
        base_url: The provider's base URL (used as unique key).
        delay: Rate limit delay in seconds.

    Returns:
        Shared RateLimiter instance for this provider.
    """
    with _PROVIDER_RATE_LIMITERS_LOCK:
        if base_url not in _PROVIDER_RATE_LIMITERS:
            _PROVIDER_RATE_LIMITERS[base_url] = RateLimiter(delay)
            logger.debug(f"Created global rate limiter for {base_url} (delay={delay}s)")
        return _PROVIDER_RATE_LIMITERS[base_url]


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


def setup_ingestor(
    llm_config: LLMInstanceConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> LMProvider:
    """Set up the ingestor language model."""
    if llm_config is None:
        raise ValueError("llm_config is required for setup_ingestor")

    lm = create_lm(
        llm_config,
        circuit_breaker_config=circuit_breaker_config,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    logger.info(f"Ingestor LLM configured: {llm_config.model}")
    return lm
