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
from aee.config import settings
from aee.config.settings import LLMInstanceConfig, Settings

logger = logging.getLogger(__name__)

class OllamaLM(dspy.LM):
    """Custom LLM provider for Ollama."""
    
    MAX_HISTORY = 100  # Keep only last N interactions to save RAM

    def __init__(self, config: LLMInstanceConfig):
        """Initialize the Ollama LLM provider.
        
        Args:
            config: Configuration for the LLM instance.
        """
        super().__init__(config.model)

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
        """Execute the request with retry logic.
        
        Args:
            payload: Request payload.
            
        Returns:
            Response text.
        """
        attempt = 0
        last_exception: Optional[Exception] = None
        
        while attempt < self.max_retries:
            try:
                return self._make_request(payload)
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


def create_lm(config: LLMInstanceConfig) -> dspy.LM:
    """Create a language model instance.
    
    Args:
        config: Configuration for the LLM instance.
        
    Returns:
        dspy.LM: Language model instance.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    if not config.model:
        raise ValueError("Model name cannot be empty")
    
    logger.info(f"Initializing LLM: {config.model} (Ollama: {config.use_ollama})")

    if config.use_ollama:
        lm = OllamaLM(config)
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
            cache=True
        )

    # Apply rate limiting if configured
    if config.rate_limit_delay > 0:
        lm = _apply_rate_limit(lm, config.rate_limit_delay)

    return lm


def setup_student(config: Optional[Settings] = None) -> dspy.LM:
    """Set up the student language model.
    
    Args:
        config: Configuration for the LLM instance.
        
    Returns:
        dspy.LM: Student language model.
    """
    current_settings = config or settings
    
    lm = create_lm(current_settings.llm.student)
    dspy.settings.configure(lm=lm)
    logger.info(f"Student LLM configured: {current_settings.llm.student.model}")
    return lm


def setup_teacher(config: Optional[Settings] = None) -> dspy.LM:
    """Set up the teacher language model.
    
    Args:
        config: Configuration for the LLM instance.
        
    Returns:
        dspy.LM: Teacher language model.
    """
    current_settings = config or settings
    
    lm = create_lm(current_settings.llm.teacher)
    logger.info(f"Teacher LLM configured: {current_settings.llm.teacher.model}")
    return lm