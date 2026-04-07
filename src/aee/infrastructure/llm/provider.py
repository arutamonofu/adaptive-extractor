# src/aee/llm/provider.py
"""LLM provider implementations for AutoEvoExtractor.

This module provides LLM provider implementations that bypass litellm to avoid
JSON serialization issues during MIPROv2 optimization.

Architecture:
    BaseLMProvider (abstract)
    ├── BaseHTTPProvider (abstract)
    │   ├── OllamaLM (Ollama API)
    │   └── OpenRouterLM (OpenRouter/OpenAI-compatible API)
    └── TransformersLM (HuggingFace Transformers local inference)

Usage:
    from aee.infrastructure.llm.provider import create_lm

    lm = create_lm(config, enable_circuit_breaker=True)
    response = lm("Your prompt here")
"""

import time
import copy
import logging
import json
import threading
import requests
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, List, Union, Optional, Dict, Tuple, Type
from functools import wraps

import dspy
from aee.infrastructure.config.settings import LLMInstanceConfig, Settings, CircuitBreakerConfig, TransformersConfig
from aee.infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class BaseLMProvider(dspy.LM, ABC):
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
        super().__init__(config.model)

        # Store config for deepcopy
        self._config = config

        # Common LLM parameters
        self.model = config.model
        self.temperature = config.temperature
        self.max_retries = config.max_retries
        self.top_p = config.top_p

        # Circuit breaker
        self._circuit_breaker = circuit_breaker

        # History tracking
        self.history: List[Dict[str, Any]] = []

    def _update_history(self, messages: List[Dict[str, Any]], response: str, kwargs: Dict[str, Any]) -> None:
        """Update the history with the latest interaction."""
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "messages"}

        self.history.append({
            "messages": messages,
            "outputs": [response],
            "model": self.model,
            "kwargs": kwargs_clean
        })

        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    def clear_history(self) -> None:
        """Clear the interaction history."""
        self.history.clear()

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
        new_instance.history = copy.deepcopy(self.history)
        return new_instance

    def reset_copy(self):
        """Create a copy with same config but empty history."""
        cb_copy = copy.deepcopy(self._circuit_breaker) if self._circuit_breaker else None
        copy_instance = self.__class__(self._config, circuit_breaker=cb_copy)
        copy_instance.history = []
        return copy_instance

    def copy(self, **kwargs):
        """Create a copy sharing history with the original (for MIPROv2)."""
        new_instance = copy.deepcopy(self)
        new_instance.history = self.history  # Share history

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

    @abstractmethod
    def __call__(self, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, **kwargs) -> List[str]:
        """Call the LLM with a prompt."""
        ...


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

        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

    def __call__(self, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, **kwargs) -> List[str]:
        if prompt is None:
            prompt = kwargs.get("messages")
        if prompt is None:
            return [""]

        messages = self._normalize_prompt(prompt)
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("messages", None)
        payload = self._prepare_payload(messages, **kwargs_copy)
        text_response = self._execute_request(payload)
        self._update_history(messages, text_response, kwargs)
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

        while attempt < self.max_retries:
            try:
                if self._circuit_breaker:
                    return self._circuit_breaker.call(self._make_request, payload)
                else:
                    return self._make_request(payload)
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
                    sleep_time = (2 ** attempt) + (0.1 * attempt)
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


class OpenRouterLM(BaseHTTPProvider):
    """LLM provider for OpenRouter with direct HTTP calls."""

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        super().__init__(config, circuit_breaker=circuit_breaker)

        noc = config.non_ollama
        self.max_tokens = noc.max_tokens
        self.provider = "OpenRouter"
        self.reasoning = noc.reasoning

        if noc.api_key is None:
            raise ValueError(
                "API key must be set for OpenRouter. "
                "Set OPENROUTER_API_KEY in .env file."
            )
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        self.api_key = noc.api_key.get_secret_value()
        self.base_url = (noc.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"

    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

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

        payload = {k: v for k, v in payload.items() if v is not None}
        return payload

    def _make_request(self, payload: Dict[str, Any]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/autoevoextractor/autoevoextractor",
            "X-Title": "AutoEvoExtractor",
        }

        try:
            with requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
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


class TransformersLM(BaseLMProvider):
    """HuggingFace Transformers local inference provider.

    Uses class-level shared model cache to avoid loading models multiple times
    (important for MIPROv2 deepcopy behavior).
    """

    _model_cache: Dict[str, Tuple[Any, Any]] = {}
    _model_loading: Dict[str, threading.Lock] = {}

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        super().__init__(config, circuit_breaker=circuit_breaker)

        self.provider = "Transformers"
        self.transformers_config = config.transformers
        self.max_new_tokens = self.transformers_config.max_new_tokens

        # Load or get model (self.model becomes the model object, self.model_name keeps the string)
        self.model_name: str = config.model
        self.model, self.tokenizer = self._load_or_get_model(
            config.model, self.transformers_config
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache. Useful for testing or freeing VRAM."""
        cls._model_cache.clear()
        cls._model_loading.clear()

    @classmethod
    def _load_or_get_model(
        cls,
        model_name: str,
        transformers_config: TransformersConfig,
    ) -> Tuple[Any, Any]:
        """Load model from cache or HuggingFace with thread-safe loading."""
        if model_name in cls._model_cache:
            logger.debug(f"Model {model_name} found in cache, reusing")
            return cls._model_cache[model_name]

        if model_name not in cls._model_loading:
            cls._model_loading[model_name] = threading.Lock()

        with cls._model_loading[model_name]:
            if model_name in cls._model_cache:
                logger.debug(f"Model {model_name} found in cache (after lock), reusing")
                return cls._model_cache[model_name]

            model, tokenizer = cls._load_model(model_name, transformers_config)
            cls._model_cache[model_name] = (model, tokenizer)
            return cls._model_cache[model_name]

    @classmethod
    def _load_model(
        cls,
        model_name: str,
        config: TransformersConfig,
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer from HuggingFace."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model {model_name} via Transformers provider...")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

        load_kwargs = {
            "device_map": config.device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.trust_remote_code,
            "attn_implementation": config.attn_implementation,
        }

        if config.load_in_4bit or config.load_in_8bit:
            load_kwargs["load_in_4bit"] = config.load_in_4bit
            load_kwargs["load_in_8bit"] = config.load_in_8bit
            del load_kwargs["torch_dtype"]

        logger.info("Loading model weights... (this may take 1-5 minutes)")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

        model.eval()

        if hasattr(model, "device"):
            device = str(model.device)
        else:
            device = str(next(model.parameters()).device)

        logger.info(
            f"Model {model_name} loaded successfully "
            f"(device: {device}, dtype: {config.torch_dtype})"
        )

        return model, tokenizer

    def __call__(
        self,
        prompt: Optional[Union[str, List[Dict[str, str]]]] = None,
        **kwargs,
    ) -> List[str]:
        """Call the LLM with a prompt."""
        if prompt is None:
            prompt = kwargs.get("messages")
        if prompt is None:
            return [""]

        messages = self._normalize_prompt(prompt)

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.model.device)  # type: ignore[attr-defined]

        output_ids = self._generate(input_ids, **kwargs)

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        self._update_history(messages, response, kwargs)
        return [response]

    def _generate(self, input_ids: Any, **kwargs) -> Any:
        """Generate with circuit breaker protection."""
        import torch

        def _do_generate():
            with torch.no_grad():
                return self.model.generate(
                    input_ids,
                    max_new_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    top_p=kwargs.get("top_p", self.top_p),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        if self._circuit_breaker:
            return self._circuit_breaker.call(_do_generate)
        return _do_generate()

    def _normalize_prompt(
        self, prompt: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        """Normalize prompt to messages format."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt


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
        return RateLimiter(delay=self.delay)

    def __copy__(self) -> 'RateLimiter':
        return self.__deepcopy__({})

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.last_call_time is not None and self.delay > 0:
                    elapsed = time.monotonic() - self.last_call_time
                    if elapsed < self.delay:
                        time.sleep(self.delay - elapsed)

                result = func(*args, **kwargs)

            self.last_call_time = time.monotonic()
            return result
        return wrapper


def _apply_rate_limit(lm: dspy.LM, delay: float) -> dspy.LM:
    """Apply a thread-safe rate limit specific to this LM instance."""
    rate_limiter = RateLimiter(delay)
    original_call = lm.__call__
    lm.__call__ = rate_limiter(original_call)
    return lm


def create_lm(
    config: LLMInstanceConfig,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> dspy.LM:
    """Create a language model instance.

    Args:
        config: Configuration for the LLM instance.
        circuit_breaker_config: Circuit breaker configuration.
        enable_circuit_breaker: Whether to enable circuit breaker.
        enable_cache: Override config's enable_cache setting (optional).

    Returns:
        dspy.LM: Language model instance.

    Raises:
        ValueError: If configuration is invalid.
    """
    if not config.model:
        raise ValueError("Model name cannot be empty")

    logger.info(f"Initializing LLM: {config.model} (provider: {config.provider})")

    use_cache = enable_cache if enable_cache is not None else config.enable_cache

    if use_cache:
        dspy.configure_cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
        )
        logger.debug("DSPy cache enabled (disk + memory)")
    else:
        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )
        logger.info("DSPy cache disabled for fresh predictions")

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

    if config.provider == "ollama":
        lm = OllamaLM(config, circuit_breaker=circuit_breaker)
    elif config.provider == "api":
        lm = OpenRouterLM(config, circuit_breaker=circuit_breaker)
    elif config.provider == "transformers":
        lm = TransformersLM(config, circuit_breaker=circuit_breaker)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

    if config.rate_limit_delay > 0:
        lm = _apply_rate_limit(lm, config.rate_limit_delay)

    return lm


def setup_student(
    config: Settings,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> dspy.LM:
    """Set up the student language model and configure DSPy globally."""
    if config is None:
        raise ValueError("config is required for setup_student")

    lm = create_lm(
        config.llm.student,
        circuit_breaker_config=config.circuit_breaker,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    dspy.settings.configure(lm=lm)
    logger.info(f"Student LLM configured: {config.llm.student.model}")
    return lm


def setup_teacher(
    config: Settings,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> dspy.LM:
    """Set up the teacher language model."""
    if config is None:
        raise ValueError("config is required for setup_teacher")

    lm = create_lm(
        config.llm.teacher,
        circuit_breaker_config=config.circuit_breaker,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    logger.info(f"Teacher LLM configured: {config.llm.teacher.model}")
    return lm
