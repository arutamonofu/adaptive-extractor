# mypy: ignore-errors
import json
from unittest.mock import patch

import pytest
import responses

from ae.core.config import ApiConfig, LLMInstanceConfig, OllamaConfig
from ae.core.llm import CircuitBreaker, CircuitBreakerError, OllamaLM, OpenRouterLM
from ae.core.llm.history_logger import save_history, save_optimization_history


@pytest.fixture
def ollama_config():
    return LLMInstanceConfig(
        provider="ollama",
        model="test-model",
        timeout=60,
        max_retries=1,
        temperature=0.0,
        rate_limit_delay=0.0,
        top_p=0.1,
        enable_cache=False,
        ollama=OllamaConfig(
            num_ctx=1024,
            num_predict=256,
            repeat_penalty=1.0,
            repeat_last_n=64,
            stream=False,
            ollama_base_url="http://localhost:11434",
        ),
        api=ApiConfig(max_tokens=256),
    )


@pytest.fixture
def openrouter_config():
    return LLMInstanceConfig(
        provider="api",
        model="test-model",
        timeout=60,
        max_retries=1,
        temperature=0.5,
        rate_limit_delay=0.0,
        top_p=0.9,
        enable_cache=False,
        api=ApiConfig(max_tokens=256, api_key="sk-test-key", base_url="https://openrouter.ai/api/v1"),
    )


@pytest.fixture
def circuit_breaker():
    return CircuitBreaker(failure_threshold=3, reset_timeout=30.0, half_open_max_calls=1, name="test")


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

@pytest.mark.unit
class TestCircuitBreaker:
    """Tests for CircuitBreaker states and transitions."""

    def test_circuit_breaker_transitions(self):
        """Test transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05, half_open_max_calls=1)
        assert cb.state.name == "CLOSED"

        # Record failures to trip the circuit
        cb._on_failure()
        assert cb.state.name == "CLOSED"
        cb._on_failure()
        assert cb.state.name == "OPEN"

        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "should not run")

        # Wait for timeout to transition to HALF_OPEN
        import time
        time.sleep(0.06)

        # Call in HALF_OPEN should succeed and close the circuit
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state.name == "CLOSED"

    def test_decorator(self):
        """Test circuit breaker decorator usage."""
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=10.0, half_open_max_calls=1)

        @cb
        def failing_function():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            failing_function()

        assert cb.state.name == "OPEN"
        with pytest.raises(CircuitBreakerError):
            failing_function()


# =============================================================================
# History Logger Tests
# =============================================================================

@pytest.mark.unit
class TestHistoryLogger:
    """Tests for LLM history logging."""

    def test_save_history(self, tmp_path):
        """Test saving LLM history to files."""
        class MockLM:
            def __init__(self):
                self.history = [{"inputs": "hello", "outputs": "world"}]

        history_file = tmp_path / "history.json"
        save_history(MockLM(), history_file)

        assert history_file.exists()
        data = json.loads(history_file.read_text())
        assert data[0]["outputs"] == "world"

    def test_save_optimization_history(self, tmp_path):
        """Test saving both student and teacher histories."""
        class MockLM:
            def __init__(self, history):
                self.history = history

        student = MockLM([{"inputs": "student"}])
        teacher = MockLM([{"inputs": "teacher"}])

        out_dir = tmp_path / "logs"
        counts = save_optimization_history(student, teacher, out_dir)

        assert counts["student"] == 1
        assert counts["teacher"] == 1
        assert len(list(out_dir.glob("student_lm_*.json"))) == 1
        assert len(list(out_dir.glob("teacher_lm_*.json"))) == 1


# =============================================================================
# LLM Providers Tests
# =============================================================================

@pytest.mark.unit
class TestLLMProviders:
    """Tests for HTTP-based and Transformers provider implementations."""

    @responses.activate
    def test_ollama_successful_request(self, ollama_config, circuit_breaker):
        """Test successful Ollama API request."""
        responses.post(
            "http://localhost:11434/api/chat",
            json={"message": {"content": "Test response"}, "done": True},
            status=200,
        )
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        result = lm("Test prompt")
        assert result == ["Test response"]

    @responses.activate
    @patch("time.sleep")
    def test_ollama_retry_on_failure(self, mock_sleep, ollama_config, circuit_breaker):
        """Test retry logic on transient failures without real delay."""
        ollama_config.max_retries = 3

        # First failure, then success
        responses.post("http://localhost:11434/api/chat", status=503)
        responses.post(
            "http://localhost:11434/api/chat",
            json={"message": {"content": "Success after retry"}, "done": True},
            status=200,
        )

        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        result = lm("Test prompt")
        assert result == ["Success after retry"]
        assert mock_sleep.call_count == 1

    @responses.activate
    def test_openrouter_headers(self, openrouter_config, circuit_breaker):
        """Test custom headers are set for OpenRouter."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "Test response"}}]},
            status=200,
        )
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm("Test prompt")

        request_headers = responses.calls[0].request.headers
        assert request_headers["Authorization"] == "Bearer sk-test-key"

