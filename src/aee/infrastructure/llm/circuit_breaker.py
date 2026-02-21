"""Circuit Breaker implementation for LLM calls.

This module provides a circuit breaker pattern implementation to prevent
cascade failures when calling LLM APIs.
"""

import logging
import time
from enum import Enum
from threading import Lock
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit tripped, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for protecting LLM API calls.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through. Failure count increases on errors.
    - OPEN: Circuit tripped after reaching failure threshold. All requests fail immediately.
    - HALF_OPEN: After reset timeout, allow one test request. Success resets circuit.

    Example:
        ```python
        breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=60,
            half_open_max_calls=1
        )

        @breaker
        def call_llm(prompt: str) -> str:
            return llm.generate(prompt)

        try:
            result = call_llm("Hello")
        except CircuitBreakerError:
            logger.warning("Circuit breaker is open, LLM unavailable")
        ```
    """

    def __init__(
        self,
        failure_threshold: int,
        reset_timeout: float,
        half_open_max_calls: int,
        name: str = "default",
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            reset_timeout: Seconds to wait before transitioning to half-open.
            half_open_max_calls: Max test calls allowed in half-open state.
            name: Name for logging purposes.
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout must be positive")
        if half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be positive")

        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition from OPEN to HALF_OPEN."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info(
                    f"CircuitBreaker '{self.name}': transitioning from OPEN to HALF_OPEN "
                    f"(timeout {self.reset_timeout}s elapsed)"
                )
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorate a function with circuit breaker protection.

        Args:
            func: Function to protect.

        Returns:
            Wrapped function with circuit breaker logic.
        """
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result of the function call.

        Raises:
            CircuitBreakerError: If circuit is open.
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"CircuitBreaker '{self.name}' is OPEN. "
                    f"Retry after {self.reset_timeout}s."
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"CircuitBreaker '{self.name}' HALF_OPEN call limit reached"
                    )
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    f"CircuitBreaker '{self.name}': HALF_OPEN test succeeded, "
                    "transitioning to CLOSED"
                )
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"CircuitBreaker '{self.name}': HALF_OPEN test failed, "
                    "transitioning to OPEN"
                )
                self._state = CircuitState.OPEN

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        f"CircuitBreaker '{self.name}': failure threshold "
                        f"({self.failure_threshold}) reached, transitioning to OPEN"
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"CircuitBreaker '{self.name}': manual reset")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def get_stats(self) -> dict:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats.
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "last_failure_time": self._last_failure_time,
            }

    def __deepcopy__(self, memo) -> 'CircuitBreaker':
        """Create a deep copy of the circuit breaker.
        
        Args:
            memo: Deepcopy memo dictionary.
            
        Returns:
            New CircuitBreaker instance with the same configuration but fresh state.
        """
        # Create new instance with same config but fresh state (no lock sharing)
        return CircuitBreaker(
            failure_threshold=self.failure_threshold,
            reset_timeout=self.reset_timeout,
            half_open_max_calls=self.half_open_max_calls,
            name=self.name,
        )

    def __copy__(self) -> 'CircuitBreaker':
        """Create a shallow copy of the circuit breaker.
        
        Returns:
            New CircuitBreaker instance with the same configuration but fresh state.
        """
        return self.__deepcopy__({})
