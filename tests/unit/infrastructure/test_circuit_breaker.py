"""Unit tests for CircuitBreaker.

Tests cover:
- Circuit breaker state transitions
- Failure threshold handling
- Reset timeout behavior
- Half-open state testing
- Decorator usage
- Statistics and reset
"""

import time

import pytest

from aee.infrastructure.llm import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)


@pytest.mark.unit
class TestCircuitBreakerInit:
    """Tests for CircuitBreaker initialization."""

    def test_valid_initialization(self):
        """Test circuit breaker initializes correctly."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=30.0,
            half_open_max_calls=1,
            name="test",
        )

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_threshold == 5
        assert breaker.reset_timeout == 30.0
        assert breaker.half_open_max_calls == 1
        assert breaker.name == "test"

    def test_zero_failure_threshold_raises(self):
        """Test that zero failure threshold raises error."""
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreaker(failure_threshold=0, reset_timeout=30.0, half_open_max_calls=1)

    def test_negative_failure_threshold_raises(self):
        """Test that negative failure threshold raises error."""
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreaker(failure_threshold=-1, reset_timeout=30.0, half_open_max_calls=1)

    def test_zero_reset_timeout_raises(self):
        """Test that zero reset timeout raises error."""
        with pytest.raises(ValueError, match="reset_timeout must be positive"):
            CircuitBreaker(failure_threshold=5, reset_timeout=0, half_open_max_calls=1)

    def test_zero_half_open_calls_raises(self):
        """Test that zero half-open calls raises error."""
        with pytest.raises(ValueError, match="half_open_max_calls must be positive"):
            CircuitBreaker(failure_threshold=5, reset_timeout=30.0, half_open_max_calls=0)


@pytest.mark.unit
class TestCircuitBreakerClosedState:
    """Tests for CLOSED state behavior."""

    def test_successful_call_resets_failure_count(self):
        """Test that successful call resets failure count."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0, half_open_max_calls=1)

        # Simulate some failures
        breaker._failure_count = 2

        # Successful call
        breaker._on_success()

        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_failure_below_threshold_keeps_closed(self):
        """Test that failures below threshold keep circuit closed."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0, half_open_max_calls=1)

        # Two failures (below threshold of 3)
        breaker._on_failure()
        breaker._on_failure()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 2

    def test_failure_at_threshold_opens_circuit(self):
        """Test that reaching failure threshold opens circuit."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0, half_open_max_calls=1)

        # Three failures (at threshold)
        breaker._on_failure()
        breaker._on_failure()
        breaker._on_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker._failure_count == 3


@pytest.mark.unit
class TestCircuitBreakerOpenState:
    """Tests for OPEN state behavior."""

    def test_call_when_open_raises_error(self):
        """Test that calling when circuit is open raises error."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=30.0, half_open_max_calls=1)

        # Trip the circuit
        breaker._on_failure()

        assert breaker.state == CircuitState.OPEN

        # Try to call a function
        def dummy_func():
            return "result"

        with pytest.raises(CircuitBreakerError, match="is OPEN"):
            breaker.call(dummy_func)

    def test_open_state_transitions_to_half_open_after_timeout(self):
        """Test that OPEN transitions to HALF_OPEN after timeout."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.1,  # 100ms for fast test
            half_open_max_calls=1,
        )

        # Trip the circuit
        breaker._on_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Check state transition
        assert breaker.state == CircuitState.HALF_OPEN


@pytest.mark.unit
class TestCircuitBreakerHalfOpenState:
    """Tests for HALF_OPEN state behavior."""

    def test_half_open_allows_limited_calls(self):
        """Test that HALF_OPEN allows limited test calls."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.01,
            half_open_max_calls=2,
        )

        # Trip and transition to half-open
        breaker._on_failure()
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

        # First call should succeed
        def success_func():
            return "ok"

        result = breaker.call(success_func)
        assert result == "ok"

        # Circuit should be closed after success
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in HALF_OPEN reopens circuit."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.01,
            half_open_max_calls=2,
        )

        # Trip and transition to half-open
        breaker._on_failure()
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failing call
        def fail_func():
            raise Exception("Test error")

        with pytest.raises(Exception):
            breaker.call(fail_func)

        # Circuit should be open again
        assert breaker.state == CircuitState.OPEN

    def test_half_open_exceeds_max_calls_raises(self):
        """Test that exceeding half-open call limit raises error."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=0.01,
            half_open_max_calls=1,
        )

        # Trip and transition to half-open
        breaker._on_failure()
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

        # First call uses the one allowed call
        breaker._half_open_calls = 1

        # Second call should fail
        with pytest.raises(CircuitBreakerError, match="HALF_OPEN call limit reached"):
            breaker.call(lambda: "test")


@pytest.mark.unit
class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator usage."""

    def test_decorator_protects_function(self):
        """Test that decorator protects function with circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=30.0, half_open_max_calls=1)

        @breaker
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_decorator_handles_exceptions(self):
        """Test that decorator handles exceptions correctly."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=30.0, half_open_max_calls=1)

        @breaker
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

        # Circuit should be open now
        assert breaker.state == CircuitState.OPEN


@pytest.mark.unit
class TestCircuitBreakerReset:
    """Tests for manual reset functionality."""

    def test_manual_reset_closes_circuit(self):
        """Test that manual reset closes circuit."""
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=30.0, half_open_max_calls=1)

        # Trip the circuit
        breaker._on_failure()
        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._success_count == 0

    def test_manual_reset_clears_stats(self):
        """Test that manual reset clears statistics."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0, half_open_max_calls=1)

        # Add some stats
        breaker._failure_count = 2
        breaker._success_count = 5
        breaker._on_failure()  # Sets last_failure_time

        # Reset
        breaker.reset()

        assert breaker._failure_count == 0
        assert breaker._success_count == 0
        assert breaker._last_failure_time is None


@pytest.mark.unit
class TestCircuitBreakerStats:
    """Tests for statistics functionality."""

    def test_get_stats_returns_data(self):
        """Test that get_stats returns correct data."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=30.0,
            half_open_max_calls=2,
            name="test_stats",
        )

        # Add some activity
        breaker._failure_count = 2
        breaker._success_count = 3

        stats = breaker.get_stats()

        assert stats["name"] == "test_stats"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2
        assert stats["success_count"] == 3
        assert stats["failure_threshold"] == 5
        assert stats["reset_timeout"] == 30.0


@pytest.mark.unit
class TestCircuitBreakerCopy:
    """Tests for copy functionality."""

    def test_deepcopy_creates_fresh_instance(self):
        """Test that deepcopy creates new instance with fresh state."""
        breaker1 = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=30.0,
            half_open_max_calls=1,
            name="original",
        )

        # Modify state
        breaker1._failure_count = 2

        import copy
        breaker2 = copy.deepcopy(breaker1)

        # Config should be same
        assert breaker2.failure_threshold == 3
        assert breaker2.reset_timeout == 30.0
        assert breaker2.name == "original"

        # State should be fresh
        assert breaker2._failure_count == 0
        assert breaker2.state == CircuitState.CLOSED

    def test_copy_creates_fresh_instance(self):
        """Test that copy creates new instance with fresh state."""
        breaker1 = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=30.0,
            half_open_max_calls=1,
            name="original",
        )

        breaker1._failure_count = 2

        import copy
        breaker2 = copy.copy(breaker1)

        # Config should be same
        assert breaker2.failure_threshold == 3
        assert breaker2.name == "original"

        # State should be fresh
        assert breaker2._failure_count == 0


@pytest.mark.unit
class TestCircuitBreakerIntegration:
    """Integration tests for complete circuit breaker workflows."""

    def test_full_circuit_trip_and_recovery(self):
        """Test complete workflow: closed -> open -> half-open -> closed."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            reset_timeout=0.05,
            half_open_max_calls=1,
        )

        # Start closed
        assert breaker.state == CircuitState.CLOSED

        # Trip circuit with failures
        def fail():
            raise Exception("fail")

        for _ in range(2):
            try:
                breaker.call(fail)
            except Exception:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.06)
        assert breaker.state == CircuitState.HALF_OPEN

        # Successful call closes circuit
        def succeed():
            return "ok"

        result = breaker.call(succeed)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    def test_multiple_functions_same_breaker(self):
        """Test that one breaker can protect multiple functions."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0, half_open_max_calls=1)

        @breaker
        def func1():
            return 1

        @breaker
        def func2():
            return 2

        assert func1() == 1
        assert func2() == 2

        # Failures from any function count toward threshold
        def fail():
            raise Exception("fail")

        for _ in range(3):
            try:
                breaker.call(fail)
            except Exception:
                pass

        # Both functions should now be blocked
        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            func1()

        with pytest.raises(CircuitBreakerError):
            func2()
