# src/aee/llm/provider.py
"""LLM provider implementations for AutoEvoExtractor."""

import time
import logging
import json
import requests
from threading import Lock
from typing import Any, List, Union, Optional, Dict
from functools import wraps
from time import monotonic

import dspy
from aee.infrastructure.config import settings
from aee.infrastructure.config.settings import LLMInstanceConfig, Settings, CircuitBreakerConfig
from aee.infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)

class OllamaLM(dspy.LM):
    """Custom LLM provider for Ollama with circuit breaker protection."""

    MAX_HISTORY = 100  # Keep only last N interactions to save RAM

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
        super().__init__(config.model)

        # Store config for deepcopy
        self._config = config

        self.model = config.model
        self.temperature = config.temperature
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.top_p = config.top_p

        oc = config.ollama
        self.base_url = oc.ollama_base_url.rstrip("/") + "/api/chat"
        self.num_ctx = oc.num_ctx
        self.num_predict = oc.num_predict
        self.stream = oc.stream
        self.repeat_penalty = oc.repeat_penalty
        self.repeat_last_n = oc.repeat_last_n
        self.provider = "ollama"
        self.history: List[Dict[str, Any]] = []

        # Initialize circuit breaker
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5,
            reset_timeout=60.0,
            name=f"ollama-{self.model}",
        )

        # Validate configuration
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if not self.base_url:
            raise ValueError("Base URL cannot be empty")

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
        payload = self._prepare_payload(messages)

        # Execute request with retry logic
        text_response = self._execute_request(payload)

        # Store in history
        self._update_history(messages, text_response, kwargs)

        return [text_response]

    def _normalize_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Normalize prompt to messages format.
        
        Args:
            prompt: Prompt string or list of messages.
            
        Returns:
            List of message dictionaries.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def _prepare_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare the request payload.
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            Dictionary with request payload.
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
                # Use circuit breaker for the actual request
                return self._circuit_breaker.call(
                    self._make_request, payload
                )
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
                logger.warning(f"Ollama error (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    sleep_time = (2 ** attempt) + (0.1 * attempt)
                    time.sleep(sleep_time)
        
        if last_exception:
            logger.error(f"Ollama failed after {self.max_retries} retries: {last_exception}")
            raise last_exception
        else:
            # This should never happen, but just in case
            raise RuntimeError("Ollama request failed without exception")

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

    def _update_history(self, messages: List[Dict[str, str]], response: str, kwargs: Dict[str, Any]) -> None:
        """Update the history with the latest interaction.
        
        Args:
            messages: List of message dictionaries.
            response: Response text.
            kwargs: Additional arguments.
        """
        self.history.append({
            "prompt": messages,
            "messages": messages,
            "outputs": [response],
            "model": self.model,
            "kwargs": kwargs
        })

        # Trim history to MAX_HISTORY
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    def clear_history(self) -> None:
        """Clear the interaction history."""
        self.history.clear()

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._circuit_breaker.reset()
        logger.info(f"Reset circuit breaker for {self.model}")

    def get_circuit_breaker_stats(self) -> dict:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats.
        """
        return self._circuit_breaker.get_stats()

    def deepcopy(self) -> 'OllamaLM':
        """Create a deep copy of this LM instance.

        Returns:
            A new OllamaLM instance with the same configuration.
        """
        import copy
        # Create new instance with same config but fresh circuit breaker state
        return OllamaLM(self._config, circuit_breaker=copy.deepcopy(self._circuit_breaker))

    def reset_copy(self) -> 'OllamaLM':
        """Create a copy of this LM instance with reset state.

        Returns:
            A new OllamaLM instance with the same configuration and empty history.
        """
        import copy
        copy_instance = OllamaLM(self._config, circuit_breaker=copy.deepcopy(self._circuit_breaker))
        copy_instance.history = []
        return copy_instance


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
        dspy.LM: Language model instance.

    Raises:
        ValueError: If configuration is invalid.
    """
    if not config.model:
        raise ValueError("Model name cannot be empty")

    logger.info(f"Initializing LLM: {config.model} (Ollama: {config.use_ollama})")

    # Determine cache setting: override takes precedence, then config
    use_cache = enable_cache if enable_cache is not None else config.enable_cache

    # Configure DSPy global cache settings
    # This affects all DSPy LLM calls, including OllamaLM
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
    if enable_circuit_breaker and config.use_ollama:
        # Get circuit breaker settings from config or use fallback defaults
        if circuit_breaker_config:
            failure_threshold = circuit_breaker_config.failure_threshold
            reset_timeout = circuit_breaker_config.reset_timeout
        else:
            # Fallback defaults if no config provided
            failure_threshold = 5
            reset_timeout = 60.0

        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            name=f"ollama-{config.model}",
        )
        logger.info(
            f"Circuit breaker enabled for {config.model} "
            f"(threshold={failure_threshold}, "
            f"timeout={reset_timeout}s)"
        )

    if config.use_ollama:
        lm = OllamaLM(config, circuit_breaker=circuit_breaker)
    else:
        # Validate non-Ollama configuration
        if config.non_ollama.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

        api_key = config.non_ollama.api_key.get_secret_value() if config.non_ollama.api_key else None
        lm = dspy.LM(
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.non_ollama.max_tokens,
            cache=use_cache
        )

    # Apply rate limiting if configured
    if config.rate_limit_delay > 0:
        lm = _apply_rate_limit(lm, config.rate_limit_delay)

    return lm


def setup_student(
    config: Optional[Settings] = None,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> dspy.LM:
    """Set up the student language model.

    Args:
        config: Configuration for the LLM instance.
        enable_circuit_breaker: Whether to enable circuit breaker.
        enable_cache: Override config's enable_cache setting (optional).

    Returns:
        dspy.LM: Student language model.
    """
    current_settings = config or settings

    lm = create_lm(
        current_settings.llm.student,
        circuit_breaker_config=current_settings.circuit_breaker,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    dspy.settings.configure(lm=lm)
    logger.info(f"Student LLM configured: {current_settings.llm.student.model}")
    return lm


def setup_teacher(
    config: Optional[Settings] = None,
    enable_circuit_breaker: bool = True,
    enable_cache: Optional[bool] = None,
) -> dspy.LM:
    """Set up the teacher language model.

    Args:
        config: Configuration for the LLM instance.
        enable_circuit_breaker: Whether to enable circuit breaker.
        enable_cache: Override config's enable_cache setting (optional).

    Returns:
        dspy.LM: Teacher language model.
    """
    current_settings = config or settings

    lm = create_lm(
        current_settings.llm.teacher,
        circuit_breaker_config=current_settings.circuit_breaker,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_cache=enable_cache,
    )
    logger.info(f"Teacher LLM configured: {current_settings.llm.teacher.model}")
    return lm