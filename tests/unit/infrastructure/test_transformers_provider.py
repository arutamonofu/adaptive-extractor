"""Unit tests for TransformersLM provider implementation."""

import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from aee.infrastructure.config.settings import (
    LLMInstanceConfig,
    OllamaConfig,
    NonOllamaConfig,
    TransformersConfig,
)
from aee.infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitBreakerError
from aee.infrastructure.llm.provider import TransformersLM, BaseLMProvider, create_lm


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def transformers_config():
    """Create a valid Transformers configuration for testing."""
    return LLMInstanceConfig(
        provider="transformers",
        model="test-model",
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
        non_ollama=NonOllamaConfig(
            max_tokens=4096,
        ),
        transformers=TransformersConfig(
            device_map="auto",
            torch_dtype="float16",
            load_in_4bit=False,
            load_in_8bit=False,
            trust_remote_code=False,
            max_new_tokens=4096,
            do_sample=True,
            attn_implementation="sdpa",
        ),
    )


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker for testing."""
    return CircuitBreaker(
        failure_threshold=5,
        reset_timeout=30.0,
        half_open_max_calls=1,
        name="transformers-test",
    )


@pytest.fixture
def mock_model_and_tokenizer():
    """Mock model and tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.decode.return_value = "Test response"
    mock_tokenizer.pad_token_id = 50256

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.parameters.return_value = [MagicMock(device="cpu")]

    return mock_model, mock_tokenizer


# =============================================================================
# Test TransformersConfig
# =============================================================================


class TestTransformersConfig:
    """Test TransformersConfig model."""

    def test_defaults(self):
        """Test TransformersConfig has correct default values."""
        config = TransformersConfig()

        assert config.device_map == "auto"
        assert config.torch_dtype == "float16"
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False
        assert config.trust_remote_code is False
        assert config.max_new_tokens == 4096
        assert config.do_sample is True
        assert config.attn_implementation == "sdpa"
        assert config.repetition_penalty == 1.2
        assert config.no_repeat_ngram_size == 0


# =============================================================================
# Test LLMInstanceConfig with provider
# =============================================================================


class TestLLMInstanceConfig:
    """Test LLMInstanceConfig with provider enum."""

    def test_provider_validation_ollama(self):
        """Test provider='ollama' requires URL."""
        config = LLMInstanceConfig(
            provider="ollama",
            model="test-model",
            timeout=60,
            max_retries=3,
            temperature=0.5,
            rate_limit_delay=0.0,
            top_p=0.9,
            repeat_penalty=1.0,
            repeat_last_n=64,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=4096,
                num_predict=1024,
                repeat_penalty=1.0,
                repeat_last_n=64,
                stream=False,
                ollama_base_url="http://localhost:11434",
            ),
            non_ollama=NonOllamaConfig(max_tokens=4096),
        )
        assert config.provider == "ollama"

    def test_provider_validation_api_requires_key(self):
        """Test provider='api' requires API key."""
        with pytest.raises(ValueError, match="API key must be set"):
            LLMInstanceConfig(
                provider="api",
                model="test-model",
                timeout=60,
                max_retries=3,
                temperature=0.5,
                rate_limit_delay=0.0,
                top_p=0.9,
                repeat_penalty=1.0,
                repeat_last_n=64,
                enable_cache=True,
                ollama=OllamaConfig(
                    num_ctx=4096,
                    num_predict=1024,
                    repeat_penalty=1.0,
                    repeat_last_n=64,
                    stream=False,
                ),
                non_ollama=NonOllamaConfig(
                    max_tokens=4096,
                ),
            )

    def test_provider_transformers_no_api_key(self):
        """Test provider='transformers' doesn't require API key."""
        config = LLMInstanceConfig(
            provider="transformers",
            model="test-model",
            timeout=60,
            max_retries=3,
            temperature=0.5,
            rate_limit_delay=0.0,
            top_p=0.9,
            repeat_penalty=1.0,
            repeat_last_n=64,
            enable_cache=True,
            ollama=OllamaConfig(
                num_ctx=4096,
                num_predict=1024,
                repeat_penalty=1.0,
                repeat_last_n=64,
                stream=False,
            ),
            non_ollama=NonOllamaConfig(max_tokens=4096),
        )
        assert config.provider == "transformers"


# =============================================================================
# Test TransformersLM
# =============================================================================


class TestTransformersLM:
    """Test TransformersLM provider."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_init_loads_model(self, mock_tokenizer_cls, mock_model_cls, transformers_config, circuit_breaker):
        """Test initialization loads model."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        # Clear cache before test
        TransformersLM.clear_cache()

        lm = TransformersLM(transformers_config, circuit_breaker=circuit_breaker)

        assert lm.model is not None
        assert lm.tokenizer is not None
        mock_model_cls.from_pretrained.assert_called_once()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_shared_cache(self, mock_tokenizer_cls, mock_model_cls, transformers_config):
        """Test two instances share the same model (class-level cache)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        lm1 = TransformersLM(transformers_config)
        lm2 = TransformersLM(transformers_config)

        # Both should use the same cached model
        assert lm1.model is lm2.model
        assert lm1.tokenizer is lm2.tokenizer
        # Model should be loaded only once
        assert mock_model_cls.from_pretrained.call_count == 1

    def test_clear_cache(self, transformers_config):
        """Test clear_cache empties the class cache."""
        TransformersLM.clear_cache()

        # Manually add something to cache to test clearing
        TransformersLM._model_cache["test"] = ("model", "tokenizer")
        TransformersLM._model_loading["test"] = threading.Lock()

        assert len(TransformersLM._model_cache) == 1
        assert len(TransformersLM._model_loading) == 1

        TransformersLM.clear_cache()

        assert len(TransformersLM._model_cache) == 0
        assert len(TransformersLM._model_loading) == 0

    def test_normalize_prompt_string(self, transformers_config, mock_model_and_tokenizer):
        """Test string prompt is converted to messages format."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        with patch.object(TransformersLM, '_load_or_get_model', return_value=(mock_model, mock_tokenizer)):
            lm = TransformersLM(transformers_config)
            messages = lm._normalize_prompt("Hello world")

        assert messages == [{"role": "user", "content": "Hello world"}]

    def test_normalize_prompt_list(self, transformers_config, mock_model_and_tokenizer):
        """Test list of messages is passed through."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        input_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        with patch.object(TransformersLM, '_load_or_get_model', return_value=(mock_model, mock_tokenizer)):
            lm = TransformersLM(transformers_config)
            messages = lm._normalize_prompt(input_messages)

        assert messages == input_messages

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_call(self, mock_tokenizer_cls, mock_model_cls, transformers_config):
        """Test __call__ generates response."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        lm = TransformersLM(transformers_config)
        result = lm("Test prompt")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "Generated text"
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_model.generate.assert_called_once()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_history_tracking(self, mock_tokenizer_cls, mock_model_cls, transformers_config):
        """Test calls are logged to history."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        lm = TransformersLM(transformers_config)
        lm("Test")

        assert len(lm.history) == 1
        assert lm.history[0]["outputs"] == ["Response"]

    def test_deepcopy(self, transformers_config, mock_model_and_tokenizer):
        """Test deepcopy creates new wrapper, same model."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        with patch.object(TransformersLM, '_load_or_get_model', return_value=(mock_model, mock_tokenizer)):
            lm = TransformersLM(transformers_config)
            lm_copy = lm.deepcopy()

        assert lm_copy is not lm
        assert lm_copy.model is lm.model  # Same model from cache
        assert lm_copy.history == lm.history

    def test_reset_copy(self, transformers_config, mock_model_and_tokenizer):
        """Test reset_copy clears history."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        with patch.object(TransformersLM, '_load_or_get_model', return_value=(mock_model, mock_tokenizer)):
            lm = TransformersLM(transformers_config)
            lm("Test")  # Add to history
            lm_copy = lm.reset_copy()

        assert lm_copy.history == []
        assert len(lm.history) == 1

    def test_copy_shares_history(self, transformers_config, mock_model_and_tokenizer):
        """Test copy() shares History list."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        with patch.object(TransformersLM, '_load_or_get_model', return_value=(mock_model, mock_tokenizer)):
            lm = TransformersLM(transformers_config)
            lm_copy = lm.copy()

            lm("Test")

        assert len(lm.history) == 1
        assert len(lm_copy.history) == 1
        assert lm.history is lm_copy.history  # Same list object

    def test_copy_uses_cached_model(self, transformers_config, mock_model_and_tokenizer):
        """Test copy() reuses the cached model instead of duplicating it in VRAM.

        This is critical for MIPROv2: DSPy calls lm.copy() during bootstrapping
        for each example. If copy() duplicated the PyTorch model, VRAM would be
        exhausted after just a few calls (~14 GB per copy for a 27B model at 4-bit).
        """
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        with patch.object(TransformersLM, '_load_or_get_model', return_value=(mock_model, mock_tokenizer)):
            lm = TransformersLM(transformers_config)
            lm_copy = lm.copy()

        assert lm_copy.model is lm.model  # Same model object from cache, not a duplicate
        assert lm_copy.tokenizer is lm.tokenizer

    def test_copy_shares_history_with_kwargs(self, transformers_config, mock_model_and_tokenizer):
        """Test copy() shares history and applies kwargs correctly."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        with patch.object(TransformersLM, '_load_or_get_model', return_value=(mock_model, mock_tokenizer)):
            lm = TransformersLM(transformers_config)
            lm_copy = lm.copy(temperature=1.0, rollout_id=42)

        # History is shared
        lm("Test")
        assert len(lm.history) == 1
        assert len(lm_copy.history) == 1
        assert lm.history is lm_copy.history

        # Kwargs are applied
        assert lm_copy.temperature == 1.0
        assert lm_copy.kwargs.get("rollout_id") == 42

        # Original is unchanged
        assert lm.temperature == 0.5
        assert "rollout_id" not in lm.kwargs

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_generate_timeout_is_passed(self, mock_tokenizer_cls, mock_model_cls, transformers_config):
        """Test that config timeout is passed to model.generate() as max_time."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        lm = TransformersLM(transformers_config)
        lm("Test")

        # Verify max_time was passed to generate()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_time"] == transformers_config.timeout

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_generate_cuda_oom_without_circuit_breaker(self, mock_tokenizer_cls, mock_model_cls, transformers_config, caplog):
        """Test CUDA OOM is logged and re-raised even without circuit breaker."""
        import logging

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        lm = TransformersLM(transformers_config)

        with pytest.raises(RuntimeError):
            with caplog.at_level(logging.ERROR):
                lm("Test")

        assert any("CUDA OOM" in record.message for record in caplog.records)

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_pad_token_id_fallback_logs_warning(
        self, mock_tokenizer_cls, mock_model_cls, transformers_config, caplog
    ):
        """Test that missing pad_token_id triggers a warning when falling back to eos_token_id."""
        import logging

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        with caplog.at_level(logging.WARNING):
            TransformersLM(transformers_config)

        assert any("pad_token_id" in record.message for record in caplog.records)
        assert mock_tokenizer.pad_token_id == 50256

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_no_pad_token_id_raises_error(self, mock_tokenizer_cls, mock_model_cls, transformers_config):
        """Test that missing both pad_token_id and eos_token_id raises ValueError."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        with pytest.raises(ValueError, match="neither pad_token_id nor eos_token_id"):
            TransformersLM(transformers_config)

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_circuit_breaker_protection(self, mock_tokenizer_cls, mock_model_cls, transformers_config, caplog):
        """Test circuit breaker trips on model errors (including CUDA OOM)."""
        import logging

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        cb = CircuitBreaker(
            failure_threshold=1,
            reset_timeout=30.0,
            half_open_max_calls=1,
            name="test-cb",
        )

        TransformersLM.clear_cache()

        lm = TransformersLM(transformers_config, circuit_breaker=cb)

        # First call should fail and trip circuit breaker
        with pytest.raises(RuntimeError):
            with caplog.at_level(logging.ERROR):
                lm("Test")

        # Verify OOM was logged
        assert any("CUDA OOM" in record.message for record in caplog.records)

        # Second call should fail immediately with CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            lm("Test")

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_repetition_penalty_passed_to_generate(
        self, mock_tokenizer_cls, mock_model_cls, transformers_config
    ):
        """Test that repetition_penalty > 1.0 is passed to model.generate()."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        lm = TransformersLM(transformers_config)
        lm("Test")

        # Verify repetition_penalty was passed to generate()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["repetition_penalty"] == 1.2

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_no_repeat_ngram_size_passed_to_generate(
        self, mock_tokenizer_cls, mock_model_cls, transformers_config
    ):
        """Test that no_repeat_ngram_size >= 2 is passed to model.generate()."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        # Set no_repeat_ngram_size to 3
        transformers_config.transformers.no_repeat_ngram_size = 3

        lm = TransformersLM(transformers_config)
        lm("Test")

        # Verify no_repeat_ngram_size was passed to generate()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["no_repeat_ngram_size"] == 3

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_repetition_control_defaults(
        self, mock_tokenizer_cls, mock_model_cls, transformers_config
    ):
        """Test that default repetition settings pass repetition_penalty=1.2 but skip no_repeat_ngram_size."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        # Ensure defaults (repetition_penalty=1.2, no_repeat_ngram_size=0)
        transformers_config.transformers.repetition_penalty = 1.2
        transformers_config.transformers.no_repeat_ngram_size = 0

        lm = TransformersLM(transformers_config)
        lm("Test")

        call_kwargs = mock_model.generate.call_args[1]
        # repetition_penalty=1.2 should be passed (default > 1.0)
        assert call_kwargs.get("repetition_penalty") == 1.2
        # no_repeat_ngram_size=0 should NOT be passed (< 2)
        assert "no_repeat_ngram_size" not in call_kwargs


# =============================================================================
# Test create_lm factory
# =============================================================================


class TestCreateLM:
    """Test create_lm factory function."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_create_lm_transformers(self, mock_tokenizer_cls, mock_model_cls, transformers_config):
        """Test create_lm returns TransformersLM for provider='transformers'."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = [MagicMock(device="cpu")]
        mock_model_cls.from_pretrained.return_value = mock_model

        TransformersLM.clear_cache()

        from aee.infrastructure.llm.provider import create_lm

        lm = create_lm(
            transformers_config,
            enable_circuit_breaker=False,
        )

        assert isinstance(lm, TransformersLM)
