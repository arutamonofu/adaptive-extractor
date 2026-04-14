"""Tests for LLM provider implementations.

Tests for BaseHTTPProvider, OllamaLM, OpenRouterLM, and create_lm factory function.
"""

import os
from unittest.mock import patch

import pytest
import responses

from aee.infrastructure.config import ApiConfig, LLMInstanceConfig, OllamaConfig
from aee.infrastructure.llm import BaseHTTPProvider, CircuitBreaker, OllamaLM, OpenRouterLM, create_lm

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def ollama_config():
    """Create a valid Ollama configuration for testing."""
    return LLMInstanceConfig(
        provider="ollama",
        model="mistral-small3.1-24b-128k:latest",
        timeout=600,
        max_retries=2,
        temperature=0.5,
        rate_limit_delay=0.0,
        top_p=0.9,
        enable_cache=True,
        ollama=OllamaConfig(
            num_ctx=8192,
            num_predict=2048,
            repeat_penalty=1.1,
            repeat_last_n=512,
            stream=False,
            ollama_base_url="http://localhost:11434",
        ),
    )


@pytest.fixture
def openrouter_config():
    """Create a valid OpenRouter configuration for testing."""
    return LLMInstanceConfig(
        provider="api",
        model="openai/gpt-4o-mini",
        timeout=600,
        max_retries=2,
        temperature=0.5,
        rate_limit_delay=0.0,
        top_p=0.9,
        enable_cache=True,
        api=ApiConfig(
            api_key="sk-test-key",
            max_tokens=4096,
            base_url="https://openrouter.ai/api/v1",
        ),
    )


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker for testing."""
    return CircuitBreaker(
        failure_threshold=5,
        reset_timeout=30.0,
        half_open_max_calls=1,
        name="test-provider",
    )


# =============================================================================
# Test BaseHTTPProvider (Abstract Class)
# =============================================================================


class TestBaseHTTPProvider:
    """Tests for BaseHTTPProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseHTTPProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseHTTPProvider.__new__(BaseHTTPProvider)

    def test_abstract_methods_defined(self):
        """Test that abstract methods are properly defined."""
        # Check that abstract methods exist
        assert "_prepare_payload" in BaseHTTPProvider.__abstractmethods__
        assert "_make_request" in BaseHTTPProvider.__abstractmethods__


# =============================================================================
# Test OllamaLM
# =============================================================================


class TestOllamaLM:
    """Tests for OllamaLM class."""

    def test_init(self, ollama_config, circuit_breaker):
        """Test OllamaLM initialization."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        assert lm.model == "mistral-small3.1-24b-128k:latest"
        assert lm.provider == "Ollama"
        assert lm.temperature == 0.5
        assert lm.timeout == 600
        assert lm.max_retries == 2
        assert lm.base_url == "http://localhost:11434/api/chat"
        assert lm.history == []

    def test_init_requires_circuit_breaker(self, ollama_config):
        """Test that OllamaLM requires circuit breaker."""
        with pytest.raises(ValueError, match="circuit_breaker is required"):
            OllamaLM(ollama_config, circuit_breaker=None)

    def test_normalize_prompt_string(self, ollama_config, circuit_breaker):
        """Test prompt normalization from string."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        messages = lm._normalize_prompt("Test prompt")

        assert messages == [{"role": "user", "content": "Test prompt"}]

    def test_normalize_prompt_messages(self, ollama_config, circuit_breaker):
        """Test prompt normalization from messages list."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        input_messages = [{"role": "user", "content": "Test"}]
        messages = lm._normalize_prompt(input_messages)

        assert messages == input_messages

    def test_prepare_payload(self, ollama_config, circuit_breaker):
        """Test Ollama payload preparation."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        messages = [{"role": "user", "content": "Test"}]
        payload = lm._prepare_payload(messages)

        assert payload["model"] == "mistral-small3.1-24b-128k:latest"
        assert payload["messages"] == messages
        assert "options" in payload
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_ctx"] == 8192
        assert payload["stream"] is False

    def test_history_tracking(self, ollama_config, circuit_breaker):
        """Test that history is properly tracked."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        # Mock the _execute_request method to avoid actual HTTP calls
        with patch.object(lm, "_execute_request", return_value="Test response"):
            lm("Test prompt")

        assert len(lm.history) == 1
        assert lm.history[0]["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert lm.history[0]["outputs"] == ["Test response"]

    def test_history_trimming(self, ollama_config, circuit_breaker):
        """Test that history is trimmed to MAX_HISTORY."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        # Add more than MAX_HISTORY entries
        for i in range(lm.MAX_HISTORY + 50):
            lm.history.append({"prompt": f"prompt_{i}"})

        # Trigger history update (which trims)
        with patch.object(lm, "_execute_request", return_value="Test"):
            lm("Test")

        assert len(lm.history) == lm.MAX_HISTORY

    def test_clear_history(self, ollama_config, circuit_breaker):
        """Test history clearing."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        lm.history.append({"prompt": "test"})
        lm.clear_history()

        assert lm.history == []

    def test_copy_shares_history(self, ollama_config, circuit_breaker):
        """Test that copy() shares history with original."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        with patch.object(lm, "_execute_request", return_value="Response"):
            lm("Test prompt")

        lm_copy = lm.copy(temperature=0.8)

        # History should be shared
        assert lm_copy.history is lm.history
        assert lm_copy.temperature == 0.8  # Updated parameter

    def test_deepcopy_independent(self, ollama_config, circuit_breaker):
        """Test that deepcopy() creates independent copy."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        # Add history directly (without calling __call__ to avoid mocking issues)
        lm.history.append({"prompt": "test", "outputs": ["Response"]})

        lm_copy = lm.deepcopy()

        # History should be independent (deep copied)
        assert lm_copy.history is not lm.history
        # Note: history is copied as part of deepcopy, so both should have same content
        assert len(lm_copy.history) == len(lm.history)

        # Modify original history
        lm.history.append({"prompt": "new"})
        assert len(lm_copy.history) != len(lm.history)

    def test_reset_copy_clears_history(self, ollama_config, circuit_breaker):
        """Test that reset_copy() clears history."""
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        with patch.object(lm, "_execute_request", return_value="Response"):
            lm("Test prompt")

        lm_copy = lm.reset_copy()

        assert lm_copy.history == []
        assert len(lm.history) > 0

    def test_call_with_messages_in_kwargs(self, ollama_config, circuit_breaker):
        """Test that calling with messages in kwargs doesn't cause 'multiple values' error.

        This is a regression test for the issue where DSPy bootstrap passes messages
        via kwargs, which conflicted with the positional messages parameter.
        """
        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        # Mock _execute_request to avoid actual HTTP calls
        with patch.object(lm, "_execute_request", return_value="Test response") as mock_execute:
            # This should not raise TypeError about 'multiple values for argument messages'
            result = lm(messages=[{"role": "user", "content": "Test"}])

            assert result == ["Test response"]
            assert mock_execute.called
            # Verify the payload was prepared correctly
            call_args = mock_execute.call_args
            payload = call_args[0][0]
            assert payload["messages"] == [{"role": "user", "content": "Test"}]


# =============================================================================
# Test OpenRouterLM
# =============================================================================


class TestOpenRouterLM:
    """Tests for OpenRouterLM class."""

    def test_init(self, openrouter_config, circuit_breaker):
        """Test OpenRouterLM initialization."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        assert lm.model == "openai/gpt-4o-mini"
        assert lm.provider == "OpenRouter"
        assert lm.temperature == 0.5
        assert lm.timeout == 600
        assert lm.max_retries == 2
        assert lm.base_url == "https://openrouter.ai/api/v1/chat/completions"
        assert lm.history == []

    def test_init_requires_api_key(self):
        """Test that OpenRouterLM requires API key."""
        # API key validation happens at config level for non-Ollama
        # This test verifies that config validation works correctly
        with pytest.raises(ValueError, match="API key must be set"):
            LLMInstanceConfig(
                provider="api",
                model="openai/gpt-4o-mini",
                timeout=600,
                max_retries=2,
                temperature=0.5,
                rate_limit_delay=0.0,
                top_p=0.9,
                repeat_penalty=1.1,
                repeat_last_n=512,
                enable_cache=True,
                ollama=OllamaConfig(
                    num_ctx=8192,
                    num_predict=2048,
                    repeat_penalty=1.1,
                    repeat_last_n=512,
                    stream=False,
                ),
                api=ApiConfig(
                    max_tokens=4096,
                ),
            )

    def test_normalize_prompt_string(self, openrouter_config, circuit_breaker):
        """Test prompt normalization from string."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        messages = lm._normalize_prompt("Test prompt")

        assert messages == [{"role": "user", "content": "Test prompt"}]

    def test_prepare_payload(self, openrouter_config, circuit_breaker):
        """Test OpenRouter payload preparation."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        messages = [{"role": "user", "content": "Test"}]
        payload = lm._prepare_payload(messages)

        assert payload["model"] == "openai/gpt-4o-mini"
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 4096
        # OpenRouter uses OpenAI-compatible format (no 'options' field)
        assert "options" not in payload

    def test_prepare_payload_with_kwargs_override(self, openrouter_config, circuit_breaker):
        """Test that kwargs can override default parameters."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        messages = [{"role": "user", "content": "Test"}]
        payload = lm._prepare_payload(messages, temperature=0.9, max_tokens=2048)

        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 2048

    def test_call_with_messages_in_kwargs(self, openrouter_config, circuit_breaker):
        """Test that calling with messages in kwargs doesn't cause 'multiple values' error.

        This is a regression test for the issue where DSPy bootstrap passes messages
        via kwargs, which conflicted with the positional messages parameter.
        """
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        # Mock _execute_request to avoid actual HTTP calls
        with patch.object(lm, "_execute_request", return_value="Test response") as mock_execute:
            # This should not raise TypeError about 'multiple values for argument messages'
            result = lm(messages=[{"role": "user", "content": "Test"}])

            assert result == ["Test response"]
            assert mock_execute.called
            # Verify the payload was prepared correctly
            call_args = mock_execute.call_args
            payload = call_args[0][0]
            assert payload["messages"] == [{"role": "user", "content": "Test"}]

    def test_history_tracking(self, openrouter_config, circuit_breaker):
        """Test that history is properly tracked."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        with patch.object(lm, "_execute_request", return_value="Test response"):
            lm("Test prompt")

        assert len(lm.history) == 1
        assert lm.history[0]["outputs"] == ["Test response"]

    def test_copy_shares_history(self, openrouter_config, circuit_breaker):
        """Test that copy() shares history with original."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        with patch.object(lm, "_execute_request", return_value="Response"):
            lm("Test prompt")

        lm_copy = lm.copy(temperature=0.8)

        assert lm_copy.history is lm.history
        assert lm_copy.temperature == 0.8

    def test_deepcopy_independent(self, openrouter_config, circuit_breaker):
        """Test that deepcopy() creates independent copy."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        with patch.object(lm, "_execute_request", return_value="Response"):
            lm("Test prompt")

        lm_copy = lm.deepcopy()

        assert lm_copy.history is not lm.history

    def test_init_with_reasoning(self, circuit_breaker):
        """Test OpenRouterLM initialization with reasoning configuration."""
        config = LLMInstanceConfig(
            provider="api",
            model="openai/gpt-oss-120b:free",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=0.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-test-key",
                max_tokens=4096,
                base_url="https://openrouter.ai/api/v1",
                reasoning={"enabled": True},
            ),
        )
        lm = OpenRouterLM(config, circuit_breaker=circuit_breaker)

        assert lm.reasoning == {"enabled": True}
        assert lm._reasoning_details is None

    def test_prepare_payload_with_reasoning(self, circuit_breaker):
        """Test that reasoning is added to payload when configured."""
        config = LLMInstanceConfig(
            provider="api",
            model="openai/gpt-oss-120b:free",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=0.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-test-key",
                max_tokens=4096,
                base_url="https://openrouter.ai/api/v1",
                reasoning={"enabled": True},
            ),
        )
        lm = OpenRouterLM(config, circuit_breaker=circuit_breaker)
        messages = [{"role": "user", "content": "Test"}]
        payload = lm._prepare_payload(messages)

        assert payload["reasoning"] == {"enabled": True}

    def test_prepare_payload_reasoning_kwargs_override(self, openrouter_config, circuit_breaker):
        """Test that reasoning can be overridden via kwargs."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        messages = [{"role": "user", "content": "Test"}]
        payload = lm._prepare_payload(messages, reasoning={"enabled": False})

        assert payload["reasoning"] == {"enabled": False}

    def test_normalize_prompt_with_reasoning_details(self, circuit_breaker):
        """Test that reasoning_details are preserved in messages."""
        config = LLMInstanceConfig(
            provider="api",
            model="openai/gpt-oss-120b:free",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=0.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-test-key",
                max_tokens=4096,
                base_url="https://openrouter.ai/api/v1",
                reasoning={"enabled": True},
            ),
        )
        lm = OpenRouterLM(config, circuit_breaker=circuit_breaker)

        # Simulate reasoning_details from previous response
        lm._reasoning_details = [{"type": "text", "text": "Let me think..."}]

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        enhanced_messages = lm._normalize_prompt(messages)

        # reasoning_details should be added to assistant message
        assert len(enhanced_messages) == 2
        assert enhanced_messages[1]["role"] == "assistant"
        assert enhanced_messages[1]["reasoning_details"] == [{"type": "text", "text": "Let me think..."}]
        # reasoning_details should be cleared after use
        assert lm._reasoning_details is None

    def test_normalize_prompt_string_ignores_reasoning_details(self, circuit_breaker):
        """Test that string prompts don't get reasoning_details."""
        config = LLMInstanceConfig(
            provider="api",
            model="openai/gpt-oss-120b:free",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=0.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-test-key",
                max_tokens=4096,
                base_url="https://openrouter.ai/api/v1",
                reasoning={"enabled": True},
            ),
        )
        lm = OpenRouterLM(config, circuit_breaker=circuit_breaker)
        lm._reasoning_details = [{"type": "text", "text": "Let me think..."}]

        messages = lm._normalize_prompt("Simple prompt")

        # String prompts should not be affected by reasoning_details
        assert messages == [{"role": "user", "content": "Simple prompt"}]
        # reasoning_details should still be set (not cleared for string prompts)
        assert lm._reasoning_details == [{"type": "text", "text": "Let me think..."}]


# =============================================================================
# Test HTTP Calls with responses library
# =============================================================================


class TestOllamaLMHTTP:
    """Tests for OllamaLM HTTP request handling."""

    @responses.activate
    def test_successful_request(self, ollama_config, circuit_breaker):
        """Test successful Ollama API request."""
        # Mock the Ollama API response
        responses.post(
            "http://localhost:11434/api/chat",
            json={
                "message": {"content": "Test response"},
                "done": True,
            },
            status=200,
        )

        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        result = lm("Test prompt")

        assert result == ["Test response"]
        assert len(responses.calls) == 1

    @responses.activate
    def test_retry_on_failure(self, ollama_config, circuit_breaker):
        """Test retry logic on transient failures."""
        # Update config to allow more retries
        ollama_config.max_retries = 3

        # First two calls fail, third succeeds
        responses.post(
            "http://localhost:11434/api/chat",
            json={"error": "Service unavailable"},
            status=503,
        )
        responses.post(
            "http://localhost:11434/api/chat",
            json={"error": "Service unavailable"},
            status=503,
        )
        responses.post(
            "http://localhost:11434/api/chat",
            json={
                "message": {"content": "Success after retry"},
                "done": True,
            },
            status=200,
        )

        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        result = lm("Test prompt")

        assert result == ["Success after retry"]
        assert len(responses.calls) == 3

    @responses.activate
    def test_timeout_handling(self, ollama_config, circuit_breaker):
        """Test timeout exception handling."""
        import requests

        responses.post(
            "http://localhost:11434/api/chat",
            body=requests.exceptions.Timeout("Request timed out"),
        )

        lm = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)

        with pytest.raises(requests.exceptions.Timeout):
            lm("Test prompt")


class TestOpenRouterLMHTTP:
    """Tests for OpenRouterLM HTTP request handling."""

    @responses.activate
    def test_successful_request(self, openrouter_config, circuit_breaker):
        """Test successful OpenRouter API request."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "Test response"}}
                ]
            },
            status=200,
        )

        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        result = lm("Test prompt")

        assert result == ["Test response"]

    @responses.activate
    def test_authorization_header(self, openrouter_config, circuit_breaker):
        """Test that Authorization header is correctly set."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "Test response"}}
                ]
            },
            status=200,
        )

        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm("Test prompt")

        # Check the request headers
        assert len(responses.calls) == 1
        request_headers = responses.calls[0].request.headers
        assert request_headers["Authorization"] == "Bearer sk-test-key"

    @responses.activate
    def test_openrouter_headers(self, openrouter_config, circuit_breaker):
        """Test that OpenRouter-specific headers are set."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "Test response"}}
                ]
            },
            status=200,
        )

        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm("Test prompt")

        request_headers = responses.calls[0].request.headers
        assert request_headers["HTTP-Referer"] == "https://github.com/autoevoextractor/autoevoextractor"
        assert request_headers["X-Title"] == "AutoEvoExtractor"

    @responses.activate
    def test_error_response_parsing(self, openrouter_config, circuit_breaker):
        """Test error response parsing."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
            status=401,
        )

        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        with pytest.raises(Exception):  # HTTPError
            lm("Test prompt")

    @responses.activate
    def test_reasoning_details_from_response(self, circuit_breaker):
        """Test that reasoning_details are extracted from response."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Test response",
                            "reasoning_details": [
                                {"type": "text", "text": "Let me think step by step..."}
                            ]
                        }
                    }
                ]
            },
            status=200,
        )

        config = LLMInstanceConfig(
            provider="api",
            model="openai/gpt-oss-120b:free",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=0.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-test-key",
                max_tokens=4096,
                base_url="https://openrouter.ai/api/v1",
                reasoning={"enabled": True},
            ),
        )
        lm = OpenRouterLM(config, circuit_breaker=circuit_breaker)
        result = lm("Test prompt")

        assert result == ["Test response"]
        assert lm._reasoning_details == [{"type": "text", "text": "Let me think step by step..."}]


# =============================================================================
# Test create_lm Factory Function
# =============================================================================


class TestCreateLM:
    """Tests for create_lm factory function."""

    def test_create_ollama_lm(self, ollama_config):
        """Test create_lm creates OllamaLM for Ollama config."""
        from aee.infrastructure.config import CircuitBreakerConfig

        # OllamaLM requires circuit breaker, so we need to provide config
        cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            reset_timeout=30.0,
            half_open_max_calls=1,
        )
        lm = create_lm(ollama_config, circuit_breaker_config=cb_config, enable_circuit_breaker=True)

        assert isinstance(lm, OllamaLM)
        assert lm.provider == "Ollama"

    def test_create_openrouter_lm(self, openrouter_config):
        """Test create_lm creates OpenRouterLM for non-Ollama config."""
        lm = create_lm(openrouter_config, circuit_breaker_config=None, enable_circuit_breaker=False)

        assert isinstance(lm, OpenRouterLM)
        assert lm.provider == "OpenRouter"

    def test_create_lm_with_circuit_breaker(self, ollama_config):
        """Test create_lm creates circuit breaker when enabled."""
        from aee.infrastructure.config import CircuitBreakerConfig

        cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            reset_timeout=30.0,
            half_open_max_calls=1,
        )

        lm = create_lm(ollama_config, circuit_breaker_config=cb_config, enable_circuit_breaker=True)

        assert isinstance(lm, OllamaLM)
        assert lm._circuit_breaker is not None

    def test_create_lm_openrouter_with_circuit_breaker(self, openrouter_config):
        """Test create_lm creates circuit breaker for OpenRouter when enabled."""
        from aee.infrastructure.config import CircuitBreakerConfig

        cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            reset_timeout=30.0,
            half_open_max_calls=1,
        )

        lm = create_lm(openrouter_config, circuit_breaker_config=cb_config, enable_circuit_breaker=True)

        assert isinstance(lm, OpenRouterLM)
        assert lm._circuit_breaker is not None


# =============================================================================
# Test create_lm with OpenRouter (Original Tests - Updated)
# =============================================================================


class TestCreateLMOpenRouter:
    """Tests for create_lm function with OpenRouter configuration."""

    def test_create_lm_with_base_url(self, monkeypatch: pytest.MonkeyPatch):
        """Test create_lm passes api_base to OpenRouterLM when base_url is set."""
        # Clear all API keys and set only OPENROUTER_API_KEY
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        config = LLMInstanceConfig(
            provider="api",
            model="openrouter/qwen/qwen3.5-397b-a17b",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=1.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-or-test-key",
                max_tokens=8192,
                base_url="https://openrouter.ai/api/v1",
            ),
        )

        # Create LM (circuit breaker disabled for testing)
        lm = create_lm(config, enable_circuit_breaker=False)

        # Verify LM is created and is OpenRouterLM
        assert lm is not None
        assert isinstance(lm, OpenRouterLM)
        assert lm.model == "openrouter/qwen/qwen3.5-397b-a17b"

    def test_create_lm_without_base_url(self, monkeypatch: pytest.MonkeyPatch):
        """Test create_lm works without base_url (uses provider default)."""
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")

        config = LLMInstanceConfig(
            provider="api",
            model="openai/gpt-4o-mini",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=1.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-openai-test-key",
                max_tokens=4096,
            ),
        )

        lm = create_lm(config, enable_circuit_breaker=False)

        assert lm is not None
        assert isinstance(lm, OpenRouterLM)
        assert lm.model == "openai/gpt-4o-mini"

    def test_create_lm_openrouter_model_format(self, monkeypatch: pytest.MonkeyPatch):
        """Test create_lm with various OpenRouter model name formats."""
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        test_models = [
            "openrouter/qwen/qwen3.5-397b-a17b",
            "openrouter/meta-llama/llama-3.1-405b-instruct",
            "openrouter/anthropic/claude-3-5-sonnet",
        ]

        for model_name in test_models:
            config = LLMInstanceConfig(
                provider="api",
                model=model_name,
                timeout=600,
                max_retries=2,
                temperature=0.5,
                rate_limit_delay=1.0,
                top_p=0.9,
                repeat_penalty=1.1,
                repeat_last_n=512,
                enable_cache=True,
                ollama=OllamaConfig(
                    num_ctx=8192,
                    num_predict=2048,
                    repeat_penalty=1.1,
                    repeat_last_n=512,
                    stream=False,
                ),
                api=ApiConfig(
                    api_key="sk-or-test-key",
                    max_tokens=8192,
                    base_url="https://openrouter.ai/api/v1",
                ),
            )

            lm = create_lm(config, enable_circuit_breaker=False)
            assert lm is not None
            assert isinstance(lm, OpenRouterLM)
            assert lm.model == model_name

    def test_create_lm_validation_requires_api_key(self, monkeypatch: pytest.MonkeyPatch):
        """Test that config validation raises error when API key is not set."""
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)

        # Config validation happens at LLMInstanceConfig creation time
        with pytest.raises(ValueError, match="API key must be set"):
            LLMInstanceConfig(
                provider="api",
                model="openai/gpt-4o-mini",
                timeout=600,
                max_retries=2,
                temperature=0.5,
                rate_limit_delay=1.0,
                top_p=0.9,
                repeat_penalty=1.1,
                repeat_last_n=512,
                enable_cache=True,
                ollama=OllamaConfig(
                    num_ctx=8192,
                    num_predict=2048,
                    repeat_penalty=1.1,
                    repeat_last_n=512,
                    stream=False,
                ),
                api=ApiConfig(
                    max_tokens=4096,
                ),
            )

    def test_create_lm_validation_max_tokens(self, monkeypatch: pytest.MonkeyPatch):
        """Test that create_lm raises error when max_tokens is invalid."""
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
            if key in os.environ:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Create valid config first (Pydantic allows max_tokens=0)
        config = LLMInstanceConfig(
            provider="api",
            model="openai/gpt-4o-mini",
            timeout=600,
            max_retries=2,
            temperature=0.5,
            rate_limit_delay=1.0,
            top_p=0.9,
            repeat_penalty=1.1,
            repeat_last_n=512,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=8192,
                num_predict=2048,
                repeat_penalty=1.1,
                repeat_last_n=512,
                stream=False,
            ),
            api=ApiConfig(
                api_key="sk-or-test-key",
                max_tokens=0,  # Invalid: must be positive
            ),
        )

        # Validation happens at LM creation time in OpenRouterLM.__init__
        with pytest.raises(ValueError, match="Max tokens must be positive"):
            create_lm(config, enable_circuit_breaker=False)


# =============================================================================
# Test ProviderRateLimiter
# =============================================================================


class TestProviderRateLimiter:
    """Tests for global provider-level rate limiting."""

    def test_shared_limiter_same_base_url(self, openrouter_config, circuit_breaker):
        """Student and Teacher with same base_url share one rate limiter."""
        from aee.infrastructure.llm.provider import (
            _PROVIDER_RATE_LIMITERS,
            _apply_rate_limit,
        )

        # Clear any existing limiters for clean test
        _PROVIDER_RATE_LIMITERS.clear()

        lm1 = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm2 = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        base_url = "https://openrouter.ai/api/v1/chat/completions"
        delay = 0.01  # Small delay for fast test

        lm1 = _apply_rate_limit(lm1, base_url, delay)
        lm2 = _apply_rate_limit(lm2, base_url, delay)

        # Both should use the same RateLimiter instance
        limiter1 = _PROVIDER_RATE_LIMITERS[base_url]
        assert limiter1 is _PROVIDER_RATE_LIMITERS[base_url]

        # Cleanup
        _PROVIDER_RATE_LIMITERS.clear()

    def test_independent_limiters_different_providers(self, ollama_config, openrouter_config, circuit_breaker):
        """Ollama and OpenRouter have independent rate limiters."""
        from aee.infrastructure.llm.provider import (
            _PROVIDER_RATE_LIMITERS,
            _apply_rate_limit,
        )

        _PROVIDER_RATE_LIMITERS.clear()

        lm_ollama = OllamaLM(ollama_config, circuit_breaker=circuit_breaker)
        lm_openrouter = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)

        ollama_url = "http://localhost:11434/api/chat"
        openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        delay = 0.01

        lm_ollama = _apply_rate_limit(lm_ollama, ollama_url, delay)
        lm_openrouter = _apply_rate_limit(lm_openrouter, openrouter_url, delay)

        # Should have two different limiter instances
        assert _PROVIDER_RATE_LIMITERS[ollama_url] is not _PROVIDER_RATE_LIMITERS[openrouter_url]

        # Cleanup
        _PROVIDER_RATE_LIMITERS.clear()
