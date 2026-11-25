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
        assert len(list(out_dir.glob("student_*.json"))) == 1
        assert len(list(out_dir.glob("teacher_*.json"))) == 1

    def test_save_ingestion_history(self, tmp_path):
        """Test saving visual extractor and teacher histories."""
        from ae.core.llm.history_logger import save_ingestion_history

        class MockLM:
            def __init__(self, history):
                self.history = history

        visual = MockLM([{"inputs": "visual"}])
        teacher = MockLM([{"inputs": "teacher"}])

        out_dir = tmp_path / "logs"
        counts = save_ingestion_history(visual, teacher, out_dir)

        assert counts["visual"] == 1
        assert counts["teacher"] == 1
        assert len(list(out_dir.glob("visual_*.json"))) == 1
        assert len(list(out_dir.glob("teacher_*.json"))) == 1




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

    def test_apply_prompt_caching_re_delimiters(self):
        """Test splitting multi-message RE user prompts on target_gt_rows and generalized_trigger."""
        from ae.core.llm.provider import apply_prompt_caching

        for delimiter in ("[[ ## target_gt_rows ## ]]\n", "[[ ## generalized_trigger ## ]]\n"):
            messages = [
                {"role": "system", "content": "RE instructions"},
                {"role": "user", "content": f"Shared raw_text and gt_table\n{delimiter}Dynamic rows"},
            ]
            cached = apply_prompt_caching(messages)
            assert len(cached) == 2
            assert cached[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
            assert isinstance(cached[1]["content"], list)
            assert len(cached[1]["content"]) == 2
            assert cached[1]["content"][0]["text"] == f"Shared raw_text and gt_table\n{delimiter}"
            assert cached[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
            assert cached[1]["content"][1]["text"] == "Dynamic rows"

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

    @responses.activate
    def test_provider_error_recorded_in_history_and_hook(self, openrouter_config, circuit_breaker):
        """Test that failed HTTP requests record an error entry in history and invoke _history_hook."""
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"error": "bad gateway"},
            status=502,
        )
        hook_entries = []
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        lm.set_history_hook(hook_entries.append)

        with pytest.raises(Exception):
            lm("Failing prompt")

        assert len(lm.history) == 1
        assert lm.history[0]["status"] == "error"
        assert lm.history[0]["outputs"] == []
        assert "error" in lm.history[0]
        assert len(hook_entries) == 1
        assert hook_entries[0]["status"] == "error"

    def test_apply_prompt_caching_re_target_gt_row_delimiter(self):
        """Test splitting multi-message and single-message RE batch prompts on [[ ## target_gt_row ## ]]."""
        from ae.core.llm.provider import apply_prompt_caching

        # Multi-message DSPy chat format (system + large user message with target_gt_row suffix)
        messages = [
            {"role": "system", "content": "RE System Instructions"},
            {
                "role": "user",
                "content": (
                    "[[ ## raw_text ## ]]\nLong document content...\n"
                    "[[ ## gt_table ## ]]\nrow_1,val\n"
                    "[[ ## target_gt_row ## ]]\nrow_6,batch_val\n"
                ),
            },
        ]
        cached = apply_prompt_caching(messages)
        assert len(cached) == 2
        assert cached[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert isinstance(cached[1]["content"], list)
        assert len(cached[1]["content"]) == 2
        assert cached[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert cached[1]["content"][0]["text"].endswith("[[ ## target_gt_row ## ]]\n")
        assert "cache_control" not in cached[1]["content"][1]
        assert cached[1]["content"][1]["text"] == "row_6,batch_val\n"

        # Single-message fallback format
        single = [
            {
                "role": "user",
                "content": "Prefix\n[[ ## target_gt_row ## ]]\nrow_1,a\n",
            }
        ]
        cached_single = apply_prompt_caching(single)
        assert len(cached_single) == 2
        assert cached_single[0]["content"][0]["text"] == "Prefix\n[[ ## target_gt_row ## ]]\n"
        assert cached_single[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert cached_single[1]["content"] == "row_1,a\n"

    @pytest.mark.parametrize(
        "delimiter",
        [
            "[[ ## mismatched_rows_context ## ]]\n",
            "[[ ## missed_gt_rows ## ]]\n",
            "[[ ## false_positive_rows ## ]]\n",
        ],
    )
    def test_apply_prompt_caching_re_residual_delimiters(self, delimiter: str):
        """Residual RE batch delimiters split static document/GT prefix from dynamic batch suffix."""
        from ae.core.llm.provider import apply_prompt_caching

        messages = [
            {"role": "system", "content": "Residual Analysis System"},
            {
                "role": "user",
                "content": (
                    "[[ ## document_text ## ]]\nFull document...\n"
                    "[[ ## gt_table ## ]]\ncol1,col2\n"
                    f"{delimiter}batch_item_1\n"
                ),
            },
        ]
        cached = apply_prompt_caching(messages)
        assert len(cached) == 2
        assert cached[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert cached[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert cached[1]["content"][0]["text"].endswith(delimiter)
        assert "cache_control" not in cached[1]["content"][1]
        assert cached[1]["content"][1]["text"] == "batch_item_1\n"

        single = [{"role": "user", "content": f"StaticDoc\n{delimiter}batch_item_2\n"}]
        cached_single = apply_prompt_caching(single)
        assert len(cached_single) == 2
        assert cached_single[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert cached_single[0]["content"][0]["text"] == f"StaticDoc\n{delimiter}"
        assert cached_single[1]["content"] == "batch_item_2\n"

    @patch("ae.core.llm.provider.requests.post")
    def test_openrouter_usage_recorded_in_history(self, mock_post, openrouter_config, circuit_breaker):
        """OpenRouterLM records unified usage dict into lm.history[-1]['usage']."""
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "provider": "Google",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 50},
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
        }
        mock_post.return_value = mock_response

        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        result = lm("hello")
        assert result == ["ok"]
        usage = lm.history[-1]["usage"]
        assert usage["prompt_tokens"] == 1000
        assert usage["completion_tokens"] == 200
        assert usage["total_tokens"] == 1200
        assert usage["cached_tokens"] == 800
        assert usage["cache_write_tokens"] == 50
        assert usage["reasoning_tokens"] == 30
        assert usage["provider"] == "Google"
        assert isinstance(usage["cost"], float) and usage["cost"] > 0

        # Subsequent error must not leak previous usage
        mock_post.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            lm("hello again")
        assert lm.history[-1]["status"] == "error"
        assert "usage" not in lm.history[-1]

    def test_gemini_usage_recorded_in_history(self, circuit_breaker):
        """GeminiLM records unified usage dict into lm.history[-1]['usage']."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from pydantic import SecretStr
        from ae.core.config.llm_settings import LLMInstanceConfig
        from ae.core.llm.provider import GeminiLM

        cfg = LLMInstanceConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key=SecretStr("test-key"),
            max_retries=1,
            temperature=0.0,
            top_p=1.0,
            enable_cache=False,
            input_price_per_1m=0.1,
            output_price_per_1m=0.4,
            cache_read_price_per_1m=0.01,
        )
        fake_usage = SimpleNamespace(
            prompt_token_count=500,
            candidates_token_count=100,
            thoughts_token_count=25,
            cached_content_token_count=200,
            total_token_count=625,
        )
        fake_response = SimpleNamespace(text="gemini-ok", usage_metadata=fake_usage)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        with patch("google.genai.Client", return_value=fake_client):
            lm = GeminiLM(cfg, circuit_breaker=circuit_breaker)
            result = lm("ping")

        assert result == ["gemini-ok"]
        usage = lm.history[-1]["usage"]
        assert usage["prompt_tokens"] == 500
        assert usage["completion_tokens"] == 100
        assert usage["reasoning_tokens"] == 25
        assert usage["cached_tokens"] == 200
        assert usage["total_tokens"] == 625
        assert isinstance(usage["cost"], float) and usage["cost"] > 0

    def test_openrouter_rate_limiter_keyed_by_base_url_and_model(self, circuit_breaker):
        """Different models on OpenRouter must not share or overwrite each other's RateLimiter."""
        from ae.core.config.llm_settings import ApiConfig, LLMInstanceConfig

        cfg_qwen = LLMInstanceConfig(
            provider="api",
            model="qwen/qwen3.5-397b-a17b",
            timeout=60,
            max_retries=1,
            temperature=0.0,
            rate_limit_delay=10.0,
            top_p=1.0,
            enable_cache=False,
            api=ApiConfig(max_tokens=256, api_key="sk-test", base_url="https://openrouter.ai/api/v1"),
        )
        cfg_gemma = LLMInstanceConfig(
            provider="api",
            model="google/gemma-4-31b-it",
            timeout=60,
            max_retries=1,
            temperature=0.0,
            rate_limit_delay=0.25,
            top_p=1.0,
            enable_cache=False,
            api=ApiConfig(max_tokens=256, api_key="sk-test", base_url="https://openrouter.ai/api/v1"),
        )
        cfg_gemma_2 = cfg_gemma.model_copy(deep=True)

        lm_qwen = OpenRouterLM(cfg_qwen, circuit_breaker=circuit_breaker)
        lm_gemma_1 = OpenRouterLM(cfg_gemma, circuit_breaker=circuit_breaker)
        lm_gemma_2 = OpenRouterLM(cfg_gemma_2, circuit_breaker=circuit_breaker)

        assert lm_qwen._rate_limiter is not lm_gemma_1._rate_limiter
        assert lm_qwen._rate_limiter.delay == 10.0
        assert lm_gemma_1._rate_limiter.delay == 0.25
        assert lm_gemma_1._rate_limiter is lm_gemma_2._rate_limiter

    def test_openrouter_response_format_in_payload(self, openrouter_config, circuit_breaker):
        """Optional response_format is forwarded in OpenRouterLM._prepare_payload."""
        lm = OpenRouterLM(openrouter_config, circuit_breaker=circuit_breaker)
        default_payload = lm._prepare_payload([{"role": "user", "content": "hi"}])
        assert "response_format" not in default_payload

        json_payload = lm._prepare_payload(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        assert json_payload["response_format"] == {"type": "json_object"}

    @pytest.mark.parametrize(
        "delimiter",
        [
            "[[ ## positive_rows_analyses ## ]]\n",
            "[[ ## positive_columns_analyses ## ]]\n",
            "[[ ## draft_prompt ## ]]\n",
            "[[ ## column_profiles ## ]]\n",
        ],
    )
    def test_apply_prompt_caching_generalization_and_schema_delimiters(self, delimiter: str):
        """Generalization and schema_refinement delimiters split static schema/baseline prefix from dynamic payloads."""
        from ae.core.llm.provider import apply_prompt_caching

        messages = [
            {"role": "system", "content": "Generalization System"},
            {
                "role": "user",
                "content": (
                    "[[ ## extraction_schema ## ]]\nSchema YAML\n"
                    "[[ ## baseline_prompt ## ]]\nBase Prompt\n"
                    f"{delimiter}Dynamic JSON payload\n"
                ),
            },
        ]
        cached = apply_prompt_caching(messages)
        assert len(cached) == 2
        assert cached[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert cached[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert cached[1]["content"][0]["text"].endswith(delimiter)
        assert "cache_control" not in cached[1]["content"][1]
        assert cached[1]["content"][1]["text"] == "Dynamic JSON payload\n"

    def test_gemini_lm_uses_model_pricing_fallback(self, circuit_breaker):
        """GeminiLM falls back to MODEL_PRICING when explicit per-1M prices are unset."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from pydantic import SecretStr
        from ae.core.config.llm_settings import LLMInstanceConfig
        from ae.core.llm.provider import GeminiLM, MODEL_PRICING

        cfg = LLMInstanceConfig(
            provider="gemini",
            model="gemini-3.8-flash",
            api_key=SecretStr("test-key"),
            max_retries=1,
            temperature=0.0,
            top_p=1.0,
            enable_cache=False,
        )
        fake_usage = SimpleNamespace(
            prompt_token_count=1_000_000,
            candidates_token_count=1_000_000,
            thoughts_token_count=0,
            cached_content_token_count=0,
            total_token_count=2_000_000,
        )
        fake_response = SimpleNamespace(text="ok", usage_metadata=fake_usage)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        with patch("google.genai.Client", return_value=fake_client):
            lm = GeminiLM(cfg, circuit_breaker=circuit_breaker)
            lm("ping")

        expected_cost = MODEL_PRICING["gemini-3.8-flash"]["input"] + MODEL_PRICING["gemini-3.8-flash"]["output"]
        assert lm.cumulative_cost == pytest.approx(expected_cost)
        assert lm.history[-1]["usage"]["cost"] == pytest.approx(expected_cost)

    def test_gemini_lm_includes_thoughts_tokens_in_cost(self, circuit_breaker):
        """GeminiLM includes thoughts_token_count in billable output cost."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from pydantic import SecretStr
        from ae.core.config.llm_settings import LLMInstanceConfig
        from ae.core.llm.provider import GeminiLM, MODEL_PRICING

        cfg = LLMInstanceConfig(
            provider="gemini",
            model="gemini-3.8-flash",
            api_key=SecretStr("test-key"),
            max_retries=1,
            temperature=0.0,
            top_p=1.0,
            enable_cache=False,
        )
        fake_usage = SimpleNamespace(
            prompt_token_count=1_000_000,
            candidates_token_count=500_000,
            thoughts_token_count=500_000,
            cached_content_token_count=0,
            total_token_count=2_000_000,
        )
        fake_response = SimpleNamespace(text="ok", usage_metadata=fake_usage)
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        with patch("google.genai.Client", return_value=fake_client):
            lm = GeminiLM(cfg, circuit_breaker=circuit_breaker)
            lm("ping")

        expected_cost = MODEL_PRICING["gemini-3.8-flash"]["input"] + MODEL_PRICING["gemini-3.8-flash"]["output"]
        assert lm.cumulative_cost == pytest.approx(expected_cost)
        assert lm.history[-1]["usage"]["cost"] == pytest.approx(expected_cost)

    def test_apply_prompt_caching_caches_long_human_document_message(self):
        """Long HumanMessage document payloads in messages[1] receive ephemeral cache_control."""
        from ae.core.llm.provider import apply_prompt_caching

        long_doc = "Paragraph of scientific article. " * 100
        messages = [
            {"role": "system", "content": "Extract structured records."},
            {"role": "user", "content": long_doc},
        ]
        cached = apply_prompt_caching(messages)
        assert len(cached) == 2
        assert cached[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert isinstance(cached[1]["content"], list)
        assert cached[1]["content"][0]["type"] == "text"
        assert cached[1]["content"][0]["text"] == long_doc
        assert cached[1]["content"][0]["cache_control"] == {"type": "ephemeral"}






