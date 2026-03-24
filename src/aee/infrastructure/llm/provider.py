# src/aee/llm/provider.py
"""LLM provider implementations for AutoEvoExtractor.

This module provides LLM provider implementations that bypass litellm to avoid
JSON serialization issues during MIPROv2 optimization.

Architecture:
    BaseHTTPProvider (abstract)
    ├── OllamaLM (Ollama API)
    └── OpenRouterLM (OpenRouter/OpenAI-compatible API)

Usage:
    from aee.infrastructure.llm.provider import create_lm

    lm = create_lm(config, enable_circuit_breaker=True)
    response = lm("Your prompt here")
"""

import time
import logging
import json
import requests
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, List, Union, Optional, Dict, Type
from functools import wraps

import dspy
from aee.infrastructure.config.settings import LLMInstanceConfig, Settings, CircuitBreakerConfig
from aee.infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)


class BaseHTTPProvider(dspy.LM, ABC):
    """Abstract base class for HTTP-based LLM providers.

    Provides common functionality for LLM providers that use direct HTTP calls
    instead of litellm, including:
    - Request/response handling
    - Retry logic with exponential backoff
    - Circuit breaker integration
    - History tracking
    - Copy/deepcopy support for MIPROv2 optimization

    Subclasses must implement:
    - _prepare_payload(): Create provider-specific request payload
    - _make_request(): Execute HTTP request to provider API
    """

    MAX_HISTORY = 200  # Keep only last N interactions to save RAM

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize the base HTTP provider.

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
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.top_p = config.top_p

        # Provider-specific (set by subclasses)
        self.provider: str = ""
        self.base_url: str = ""

        # Circuit breaker
        self._circuit_breaker = circuit_breaker

        # History tracking
        self.history: List[Dict[str, Any]] = []

        # Reasoning details for OpenRouter reasoning models (initialized here for all providers)
        self._reasoning_details: Optional[List[Dict[str, Any]]] = None

        # Validate common configuration
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

    def __call__(self, prompt: Optional[Union[str, List[Dict[str, str]]]] = None, **kwargs) -> List[str]:
        """Call the LLM with a prompt.

        Args:
            prompt: Prompt string or list of messages.
            **kwargs: Additional arguments.

        Returns:
            List of response strings.
        """
        if prompt is None:
            prompt = kwargs.get("messages")

        if prompt is None:
            return [""]

        # Normalize prompt to messages format
        messages = self._normalize_prompt(prompt)

        # Prepare request payload
        # Remove 'messages' from kwargs to avoid passing it twice to _prepare_payload
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("messages", None)
        payload = self._prepare_payload(messages, **kwargs_copy)

        # Execute request with retry logic
        text_response = self._execute_request(payload)

        # Store in history
        self._update_history(messages, text_response, kwargs)

        return [text_response]

    def _normalize_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Normalize prompt to messages format.

        Args:
            prompt: Prompt string or list of messages.

        Returns:
            List of message dictionaries.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        # For reasoning models, preserve reasoning_details from previous responses
        # This allows the model to continue reasoning from where it left off
        if self._reasoning_details is not None:
            # Add reasoning_details to assistant messages in the conversation
            reasoning_details = self._reasoning_details  # Local variable for mypy
            enhanced_messages: List[Dict[str, Any]] = []
            for i, msg in enumerate(prompt):
                enhanced_msg: Dict[str, Any] = msg.copy()
                # Attach reasoning_details only to the LAST assistant message
                # to avoid contaminating few-shot examples with reasoning from other requests
                if msg.get("role") == "assistant" and reasoning_details and i == len(prompt) - 1:
                    enhanced_msg["reasoning_details"] = reasoning_details
                enhanced_messages.append(enhanced_msg)
            # Clear reasoning_details after using them
            self._reasoning_details = None
            return enhanced_messages

        return prompt

    @abstractmethod
    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Prepare the request payload for the provider API.

        Args:
            messages: List of message dictionaries.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with request payload.
        """
        pass

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        """Execute the request with retry logic and circuit breaker protection.

        Args:
            payload: Request payload.

        Returns:
            Response text.

        Raises:
            CircuitBreakerError: If circuit breaker is open.
        """
        attempt = 0
        last_exception: Optional[Exception] = None

        while attempt < self.max_retries:
            try:
                # Use circuit breaker if available
                if self._circuit_breaker:
                    return self._circuit_breaker.call(
                        self._make_request, payload
                    )
                else:
                    return self._make_request(payload)
            except CircuitBreakerError:
                # Circuit breaker is open, don't retry
                logger.warning(
                    f"Circuit breaker OPEN for {self.model}. "
                    f"Retry after {self._circuit_breaker.reset_timeout}s."
                )
                raise
            except Exception as e:
                last_exception = e
                attempt += 1
                logger.warning(f"{self.provider} error (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    sleep_time = (2 ** attempt) + (0.1 * attempt)
                    time.sleep(sleep_time)

        if last_exception:
            logger.error(f"{self.provider} failed after {self.max_retries} retries: {last_exception}")
            raise last_exception
        else:
            # This should never happen, but just in case
            raise RuntimeError(f"{self.provider} request failed without exception")

    @abstractmethod
    def _make_request(self, payload: Dict[str, Any]) -> str:
        """Make a single HTTP request to the provider API.

        Args:
            payload: Request payload.

        Returns:
            Response text.

        Raises:
            requests.RequestException: If the request fails.
        """
        pass

    def _update_history(self, messages: List[Dict[str, Any]], response: str, kwargs: Dict[str, Any]) -> None:
        """Update the history with the latest interaction.

        Args:
            messages: List of message dictionaries.
            response: Response text.
            kwargs: Additional arguments.
        """
        # Remove 'messages' from kwargs to avoid duplication
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "messages"}

        self.history.append({
            "messages": messages,
            "outputs": [response],
            "model": self.model,
            "kwargs": kwargs_clean
        })

        # Trim history to MAX_HISTORY
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
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats, or None if not enabled.
        """
        if self._circuit_breaker:
            return self._circuit_breaker.get_stats()
        return None

    def deepcopy(self):
        """Create a deep copy of this LM instance.

        Returns:
            A new instance with the same configuration and history.
        """
        import copy
        cb_copy = copy.deepcopy(self._circuit_breaker) if self._circuit_breaker else None
        new_instance = self.__class__(self._config, circuit_breaker=cb_copy)
        # Copy history to the new instance
        new_instance.history = copy.deepcopy(self.history)
        return new_instance

    def reset_copy(self):
        """Create a copy of this LM instance with reset state.

        Returns:
            A new instance with the same configuration and empty history.
        """
        import copy
        cb_copy = copy.deepcopy(self._circuit_breaker) if self._circuit_breaker else None
        copy_instance = self.__class__(self._config, circuit_breaker=cb_copy)
        copy_instance.history = []
        return copy_instance

    def copy(self, **kwargs):
        """Create a copy of this LM instance sharing history with the original.

        Overrides dspy.LM.copy() to preserve history across MIPROv2 instruction
        generation rollouts. MIPROv2 creates copies with unique rollout_id for
        each instruction candidate; sharing history ensures all LLM calls are logged.

        Args:
            **kwargs: Parameters to update in the copy (e.g., rollout_id, temperature).

        Returns:
            A new instance with shared history reference.
        """
        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = self.history  # Share history with original

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


class OllamaLM(BaseHTTPProvider):
    """LLM provider for Ollama with circuit breaker protection.

    Uses direct HTTP calls to Ollama API, bypassing litellm to avoid
    JSON serialization issues during MIPROv2 optimization.
    """

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize the Ollama LLM provider.

        Args:
            config: Configuration for the LLM instance.
            circuit_breaker: Optional circuit breaker for failure protection.
        """
        # Circuit breaker is required for Ollama
        if circuit_breaker is None:
            raise ValueError("circuit_breaker is required")

        super().__init__(config, circuit_breaker=circuit_breaker)

        # Ollama-specific configuration
        oc = config.ollama
        self.num_ctx = oc.num_ctx
        self.num_predict = oc.num_predict
        self.stream = oc.stream
        self.repeat_penalty = oc.repeat_penalty
        self.repeat_last_n = oc.repeat_last_n
        self.provider = "Ollama"

        # Validate Ollama configuration
        if not oc.ollama_base_url:
            raise ValueError(
                "OLLAMA_BASE_URL environment variable must be set in .env file. "
                "Set OLLAMA_STUDENT_BASE_URL or OLLAMA_TEACHER_BASE_URL as appropriate."
            )
        self.base_url = oc.ollama_base_url.rstrip("/") + "/api/chat"

    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Prepare the request payload for Ollama API.

        Args:
            messages: List of message dictionaries.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with Ollama-specific request payload.
        """
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
        """Make a single request to the Ollama API.

        Args:
            payload: Request payload.

        Returns:
            Response text.

        Raises:
            requests.RequestException: If the request fails.
        """
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
                                    print()  # New line at the end of response
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
    """LLM provider for OpenRouter with direct HTTP calls.

    Uses direct HTTP calls to OpenRouter API, bypassing litellm to avoid
    JSON serialization issues during MIPROv2 optimization.

    Supports any OpenRouter model with OpenAI-compatible API format.
    """

    def __init__(
        self,
        config: LLMInstanceConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize the OpenRouter LLM provider.

        Args:
            config: Configuration for the LLM instance.
            circuit_breaker: Optional circuit breaker for failure protection.
        """
        super().__init__(config, circuit_breaker=circuit_breaker)

        # OpenRouter-specific configuration
        noc = config.non_ollama
        self.max_tokens = noc.max_tokens
        self.provider = "OpenRouter"
        self.reasoning = noc.reasoning

        # Validate configuration
        if noc.api_key is None:
            raise ValueError(
                "API key must be set for OpenRouter. "
                "Set OPENROUTER_API_KEY in .env file."
            )
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        self.api_key = noc.api_key.get_secret_value()

        # Build OpenRouter API URL
        self.base_url = (noc.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"

    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Prepare the request payload for OpenRouter API.

        Args:
            messages: List of message dictionaries.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with OpenAI-compatible request payload.
        """
        # Override temperature and max_tokens if provided in kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", self.top_p),
        }

        # Add reasoning configuration if provided (for OpenRouter reasoning models)
        reasoning = kwargs.get("reasoning", self.reasoning)
        if reasoning is not None:
            payload["reasoning"] = reasoning

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return payload

    def _make_request(self, payload: Dict[str, Any]) -> str:
        """Make a single request to the OpenRouter API.

        Args:
            payload: Request payload.

        Returns:
            Response text.

        Raises:
            requests.RequestException: If the request fails.
        """
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

                # Extract content from response
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0]["message"]
                    content = message.get("content", "")

                    # Store reasoning_details for subsequent requests (OpenRouter reasoning models)
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
            # Try to extract error details from response
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
    """Wrapper to use LLM providers as teacher for MIPROv2 bootstrapping.

    DSPy teleprompters expect teacher to be a dspy.Module with predictors().
    This wrapper allows using raw LLM (OllamaLM, OpenRouterLM) as teacher.

    Note: Uses ChainOfThought to match the structure of UniversalExtractor (student).
    """

    def __init__(self, signature_class: Type[dspy.Signature], teacher_lm: dspy.LM):
        """Initialize the teacher wrapper.

        Args:
            signature_class: DSPy signature class defining the task.
            teacher_lm: Teacher language model (e.g., OllamaLM, OpenRouterLM).
        """
        super().__init__()
        self.signature_class = signature_class
        self.teacher_lm = teacher_lm
        # Use ChainOfThought to match UniversalExtractor structure (student)
        self.prog = dspy.ChainOfThought(signature_class, lm=teacher_lm)

    def forward(self, document_text: str) -> dspy.Prediction:
        """Execute the extraction pipeline.

        Args:
            document_text: The full content of the document.

        Returns:
            dspy.Prediction with extracted data.
        """
        return self.prog(document_text=document_text)

    def predictors(self) -> List[dspy.Predict]:
        """Return list of predictors for teleprompter bootstrapping.

        Returns:
            List containing the single predictor.
        """
        return [self.prog.predict]

    def __deepcopy__(self, memo):
        """Return self to share the same teacher_lm instance.

        MIPROv2 creates copies of the teacher module during optimization.
        By returning self, we ensure all copies use the same teacher_lm,
        so all LLM call history is collected in one place.

        Args:
            memo: Deepcopy memo dictionary.

        Returns:
            self (same instance)
        """
        memo[id(self)] = self
        return self


class RateLimiter:
    """Thread-safe rate limiter for LLM instances."""

    def __init__(self, delay: float):
        """Initialize rate limiter.

        Args:
            delay: Delay in seconds between calls.
        """
        if delay < 0:
            raise ValueError("Delay cannot be negative")
        self.delay = delay
        self.lock = Lock()
        self.last_call_time: Optional[float] = None

    def __deepcopy__(self, memo) -> 'RateLimiter':
        """Create a deep copy of the rate limiter.

        Args:
            memo: Deepcopy memo dictionary.

        Returns:
            New RateLimiter instance with the same delay but fresh state.
        """
        # Create new instance with same delay but fresh lock
        return RateLimiter(delay=self.delay)

    def __copy__(self) -> 'RateLimiter':
        """Create a shallow copy of the rate limiter.

        Returns:
            New RateLimiter instance with the same delay but fresh state.
        """
        return self.__deepcopy__({})

    def __call__(self, func):
        """Apply rate limiting to a function.

        Args:
            func: Function to wrap.

        Returns:
            Wrapped function with rate limiting.
        """
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
    """Apply a thread-safe rate limit specific to this LM instance.

    Args:
        lm: Language model instance.
        delay: Delay in seconds.

    Returns:
        dspy.LM: Rate-limited language model.
    """
    rate_limiter = RateLimiter(delay)
    original_call = lm.__call__
    lm.__call__ = rate_limiter(original_call)
    return lm


def create_lm(
    config: LLMInstanceConfig,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,  # Override config if provided
) -> dspy.LM:
    """Create a language model instance.

    Args:
        config: Configuration for the LLM instance.
        circuit_breaker_config: Circuit breaker configuration. If None, uses defaults.
        enable_circuit_breaker: Whether to enable circuit breaker protection.
        enable_cache: Override config's enable_cache setting (optional).

    Returns:
        dspy.LM: Language model instance (OllamaLM or OpenRouterLM).

    Raises:
        ValueError: If configuration is invalid.
    """
    if not config.model:
        raise ValueError("Model name cannot be empty")

    logger.info(f"Initializing LLM: {config.model} (Ollama: {config.use_ollama})")

    # Determine cache setting: override takes precedence, then config
    use_cache = enable_cache if enable_cache is not None else config.enable_cache

    # Configure DSPy global cache settings
    # This affects all DSPy LLM calls, including custom providers
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

    # Create circuit breaker if enabled
    circuit_breaker = None
    if enable_circuit_breaker:
        if circuit_breaker_config is None:
            raise ValueError("circuit_breaker_config is required when enable_circuit_breaker is True")
        failure_threshold = circuit_breaker_config.failure_threshold
        reset_timeout = circuit_breaker_config.reset_timeout

        # Create provider-specific circuit breaker name
        provider_name = "ollama" if config.use_ollama else "openrouter"
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

    if config.use_ollama:
        lm = OllamaLM(config, circuit_breaker=circuit_breaker)
    else:
        lm = OpenRouterLM(config, circuit_breaker=circuit_breaker)

    # Apply rate limiting if configured
    if config.rate_limit_delay > 0:
        lm = _apply_rate_limit(lm, config.rate_limit_delay)

    return lm


def setup_student(
    config: Settings,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> dspy.LM:
    """Set up the student language model and configure DSPy globally.

    This function creates the student LM and configures DSPy to use it
    via dspy.settings.configure(lm=lm). Call this function once at
    application startup to set up the global DSPy configuration.

    Args:
        config: Application settings. Required.
        enable_circuit_breaker: Whether to enable circuit breaker.
        enable_cache: Override config's enable_cache setting (optional).

    Returns:
        dspy.LM: Student language model.

    Raises:
        ValueError: If config is None.
    """
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
    """Set up the teacher language model.

    Note: Unlike setup_student(), this function does NOT configure DSPy
    globally. The teacher LM is used explicitly in optimization workflows.

    Args:
        config: Application settings. Required.
        enable_circuit_breaker: Whether to enable circuit breaker.
        enable_cache: Override config's enable_cache setting (optional).

    Returns:
        dspy.LM: Teacher language model.

    Raises:
        ValueError: If config is None.
    """
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
