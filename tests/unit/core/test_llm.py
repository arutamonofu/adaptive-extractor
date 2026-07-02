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

    @responses.activate
    def test_openrouter_response_caching_headers(self, openrouter_config, circuit_breaker):
        """Test X-OpenRouter-Cache header is set when openrouter_cache=True."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "Test response"}}]},
            status=200,
        )
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm("Test prompt", openrouter_cache=True)

        request_headers = responses.calls[0].request.headers
        assert request_headers["X-OpenRouter-Cache"] == "true"

    def test_apply_prompt_caching_dspy_delimiter(self):
        """Test splitting and formatting DSPy string prompts for caching."""
        from ae.core.llm.provider import apply_prompt_caching

        dspy_prompt = (
            "System instructions here...\n"
            "[[ ## document_text ## ]]\n"
            "Dynamic document content here"
        )
        messages = [{"role": "system", "content": dspy_prompt}]
        
        cached_messages = apply_prompt_caching(messages)
        
        assert len(cached_messages) == 2
        assert cached_messages[0]["role"] == "system"
        assert isinstance(cached_messages[0]["content"], list)
        assert cached_messages[0]["content"][0]["type"] == "text"
        assert cached_messages[0]["content"][0]["text"] == "System instructions here...\n[[ ## document_text ## ]]\n"
        assert cached_messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        
        assert cached_messages[1]["role"] == "user"
        assert cached_messages[1]["content"] == "Dynamic document content here"

    def test_apply_prompt_caching_dspy_delimiter_few_shot(self):
        """Test splitting at the last delimiter when few-shot demos are present."""
        from ae.core.llm.provider import apply_prompt_caching

        few_shot_prompt = (
            "Instructions here...\n"
            "[[ ## document_text ## ]]\n"
            "Demo Document 1\n"
            "[[ ## extracted_data ## ]]\n"
            "Demo Output 1\n"
            "[[ ## document_text ## ]]\n"
            "Target validation document content"
        )
        messages = [{"role": "user", "content": few_shot_prompt}]
        cached_messages = apply_prompt_caching(messages)

        assert len(cached_messages) == 2
        
        # Verify static_prefix contains instructions and the few-shot demo
        static_part = cached_messages[0]["content"][0]
        assert static_part["type"] == "text"
        assert static_part["cache_control"] == {"type": "ephemeral"}
        assert "Demo Document 1" in static_part["text"]
        assert "Target validation document" not in static_part["text"]
        assert static_part["text"].endswith("[[ ## document_text ## ]]\n")

        # Verify dynamic_suffix contains only the final document content
        assert cached_messages[1]["role"] == "user"
        assert cached_messages[1]["content"] == "Target validation document content"

    def test_apply_prompt_caching_standard_messages(self):
        """Test standard messages format converts the first message content to block with cache_control."""
        from ae.core.llm.provider import apply_prompt_caching

        messages = [
            {"role": "system", "content": "Static instructions"},
            {"role": "user", "content": "Dynamic question"}
        ]
        
        cached_messages = apply_prompt_caching(messages)
        
        assert len(cached_messages) == 2
        assert cached_messages[0]["role"] == "system"
        assert isinstance(cached_messages[0]["content"], list)
        assert cached_messages[0]["content"][0]["type"] == "text"
        assert cached_messages[0]["content"][0]["text"] == "Static instructions"
        assert cached_messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        
        assert cached_messages[1]["role"] == "user"
        assert cached_messages[1]["content"] == "Dynamic question"

    @responses.activate
    def test_openrouter_prompt_caching_call(self, openrouter_config, circuit_breaker):
        """Test that prompt_caching parameter is passed and processed in request payload."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "Test response"}}]},
            status=200,
        )
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm("Test prompt", prompt_caching=True)

        req_body = json.loads(responses.calls[0].request.body)
        messages = req_body["messages"]
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert messages[0]["content"][0]["text"] == "Test prompt"

    @responses.activate
    def test_openrouter_session_id(self, openrouter_config, circuit_breaker):
        """Test that session_id is auto-generated or passed correctly in payload."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "Test response"}}]},
            status=200,
        )
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        assert lm.session_id.startswith("ae-")
        
        lm("Test prompt", session_id="custom-session-id")
        req_body = json.loads(responses.calls[0].request.body)
        assert req_body["session_id"] == "custom-session-id"

    @responses.activate
    def test_openrouter_usage_logging(self, openrouter_config, circuit_breaker, caplog):
        """Test that usage metrics and caching statistics are logged if present in the response."""
        import logging
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Test response"}}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {
                        "cached_tokens": 100,
                        "cache_write_tokens": 20
                    }
                }
            },
            status=200,
        )
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        with caplog.at_level(logging.INFO):
            lm("Test prompt")
            
        assert any(
            "OpenRouter usage" in record.message and "cached=100" in record.message
            for record in caplog.records
        )

    @responses.activate
    def test_openrouter_provider_preferences(self, openrouter_config, circuit_breaker, caplog):
        """Test that provider preferences are passed correctly in request payload with alias mapping."""
        import logging
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "Test response"}}]},
            status=200,
        )
        # 1. Test when configured in api config as OpenRouterServiceProviderPreferences
        from ae.core.config.settings import OpenRouterServiceProviderPreferences
        
        with caplog.at_level(logging.WARNING):
            provider_cfg = OpenRouterServiceProviderPreferences(
                priority_order=["Parasail/FP8", "chutes/fp8", "deepinfra-unknown"],
                require_parameter_support=True
            )
        
        # Verify lowercase normalisation warning was logged (for Parasail/FP8)
        assert any("is not lowercase" in record.message and "Parasail/FP8" in record.message for record in caplog.records)
        # Verify unrecognized provider slug warning was logged (only for deepinfra-unknown)
        assert any("deepinfra-unknown" in record.message and "is not recognized" in record.message for record in caplog.records)
        # Verify no unrecognized warning was logged for parasail/fp8 or chutes/fp8
        assert not any("parasail/fp8" in record.message and "is not recognized" in record.message for record in caplog.records)
        assert not any("chutes/fp8" in record.message and "is not recognized" in record.message for record in caplog.records)
        
        # Verify values are cleaned and normalised to lowercase
        assert provider_cfg.priority_order == ["parasail/fp8", "chutes/fp8", "deepinfra-unknown"]

        openrouter_config.api.provider = provider_cfg
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm("Test prompt")
        req_body = json.loads(responses.calls[0].request.body)
        assert req_body["provider"] == {
            "order": ["parasail/fp8", "chutes/fp8", "deepinfra-unknown"],
            "require_parameters": True
        }

        # 2. Test when passed dynamically as kwargs override (dictionary)
        lm("Test prompt", provider={"priority_order": ["together"], "require_parameter_support": False})
        req_body_2 = json.loads(responses.calls[1].request.body)
        assert req_body_2["provider"] == {
            "order": ["together"],
            "require_parameters": False
        }

    @responses.activate
    def test_openrouter_cost_calculation(self, openrouter_config, circuit_breaker):
        """Test that prompt/completion/cached token costs are correctly computed and accumulated."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Costed response"}}],
                "usage": {
                    "prompt_tokens": 100000,
                    "completion_tokens": 50000,
                    "prompt_tokens_details": {
                        "cached_tokens": 80000,
                        "cache_write_tokens": 20000
                    }
                }
            },
            status=200,
        )
        # Configure model prices in configuration (prices per 1M tokens)
        openrouter_config.input_price_per_1m = 10.0
        openrouter_config.output_price_per_1m = 20.0
        openrouter_config.cache_read_price_per_1m = 1.0

        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        assert lm.cumulative_cost == 0.0

        lm("Test prompt")
        # Expected calculation:
        # non-cached prompt tokens = 100000 - 80000 = 20000
        # non-cached cost = 20000 * 10 / 1_000_000 = 0.20
        # cached prompt cost = 80000 * 1 / 1_000_000 = 0.08
        # completion cost = 50000 * 20 / 1_000_000 = 1.00
        # total expected cost = 0.20 + 0.08 + 1.00 = 1.28
        assert abs(lm.cumulative_cost - 1.28) < 1e-6

    @responses.activate
    def test_provider_latency_logging(self, openrouter_config, circuit_breaker, caplog):
        """Test that request latency is logged during BaseHTTPProvider._execute_request."""
        import logging
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "Timed response"}}]},
            status=200,
        )
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        with caplog.at_level(logging.INFO):
            lm("Test prompt")
            
        assert any(
            "Request completed in" in record.message
            for record in caplog.records
        )

    @responses.activate
    def test_provider_latency_recorded_in_history(self, openrouter_config, circuit_breaker):
        """Test that request latency is recorded in provider's history."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "Timed response"}}]},
            status=200,
        )
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm("Test prompt")
        
        assert len(lm.history) == 1
        assert "latency_s" in lm.history[0]
        assert isinstance(lm.history[0]["latency_s"], float)
        assert lm.history[0]["latency_s"] >= 0.0





