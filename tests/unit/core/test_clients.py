"""Unit tests for stage-specific LLM clients (ExtractionClient and SemanticOptimizationClient)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from ae.core.config.optimization_settings import CircuitBreakerConfig, ExtractionConfig, SemanticOptimizationConfig
from ae.extraction.client import ExtractionClient
from ae.semantic_optimization.client import SemanticOptimizationClient


class TestExtractionClient:
    def test_initialization_from_config(self):
        ext_cfg = ExtractionConfig(
            execution={"enable_cache": True, "max_tool_iterations": 7, "save_llm_history": True},
            api_client={"timeout": 120, "max_retries": 3, "rate_limit_delay": 1.0},
            model={"provider": "gemini", "model": "gemini-3.8-flash", "temperature": 0.0},
        )
        cb_cfg = CircuitBreakerConfig(failure_threshold=5, reset_timeout=30.0, half_open_max_calls=1)

        client = ExtractionClient(config=ext_cfg, circuit_breaker_config=cb_cfg)
        assert client.model == "gemini-3.8-flash"
        assert client.provider == "gemini"
        assert client.llm_config.enable_cache is True
        assert client.llm_config.timeout == 120
        assert client.llm_config.max_retries == 3
        assert client.llm_config.rate_limit_delay == 1.0

    @patch("ae.extraction.client.create_chat_model")
    def test_get_chat_model_lazy_init(self, mock_create_chat_model):
        mock_chat = MagicMock()
        mock_create_chat_model.return_value = mock_chat

        client = ExtractionClient(
            config=ExtractionConfig(
                model={"provider": "gemini", "model": "gemini-3.8-flash"}
            )
        )
        assert client._chat_model is None
        chat1 = client.get_chat_model()
        assert chat1 is mock_chat
        mock_create_chat_model.assert_called_once()
        assert mock_create_chat_model.call_args.kwargs.get("max_retries") == 1

        # Cached on subsequent access
        chat2 = client.get_chat_model()
        assert chat2 is mock_chat
        assert mock_create_chat_model.call_count == 1

    @patch("ae.extraction.client.create_lm")
    def test_get_lm_provider_and_dspy_adapter(self, mock_create_lm):
        mock_lm = MagicMock()
        mock_lm.model = "gemini-3.8-flash"
        mock_lm.history = [{"call": 1}]
        mock_create_lm.return_value = mock_lm

        client = ExtractionClient(
            config=ExtractionConfig(
                model={"provider": "gemini", "model": "gemini-3.8-flash"}
            )
        )
        provider = client.get_lm_provider()
        assert provider is mock_lm

        adapter = client.get_dspy_adapter()
        assert adapter is not None
        assert adapter.provider is mock_lm

    def test_save_history(self, tmp_path: Path):
        client = ExtractionClient(
            config=ExtractionConfig(
                model={"provider": "gemini", "model": "gemini-3.8-flash"}
            )
        )
        mock_lm = MagicMock()
        mock_lm.history = [{"messages": [{"role": "user", "content": "hi"}], "outputs": ["hello"], "model": "gemini-3.8-flash"}]
        client._lm_provider = mock_lm

        saved_count = client.save_history(tmp_path, timestamp="test_ts")
        assert saved_count == 1
        saved_file = tmp_path / "extraction_test_ts.json"
        assert saved_file.exists()

    def test_save_llm_history_disabled_writes_nothing(self, tmp_path: Path):
        client = ExtractionClient(
            config=ExtractionConfig(
                execution={"save_llm_history": False},
                model={"provider": "gemini", "model": "gemini-3.8-flash"},
            )
        )
        assert client.history_recorder is None
        assert client.get_callback_handler() is None
        assert client.start_history_session(tmp_path, timestamp="disabled_ts") is None
        assert client.save_history(tmp_path, timestamp="disabled_ts") == 0
        assert not (tmp_path / "extraction_disabled_ts.jsonl").exists()
        assert not (tmp_path / "extraction_disabled_ts.json").exists()

    def test_streaming_session_unifies_chat_model_and_lm_provider(self, tmp_path: Path):
        import json
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class FakeChat(BaseChatModel):
            model_name: str = "gemini-3.8-flash"

            @property
            def _llm_type(self) -> str:
                return "fake"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="graph ok"))])

        client = ExtractionClient(
            config=ExtractionConfig(
                execution={"save_llm_history": True},
                model={"provider": "gemini", "model": "gemini-3.8-flash"},
            )
        )
        stream_path = client.start_history_session(tmp_path, timestamp="unified_ts")
        assert stream_path == tmp_path / "extraction_unified_ts.jsonl"

        with patch("ae.extraction.client.create_chat_model") as mock_create_chat:
            mock_create_chat.side_effect = lambda cfg, callbacks=None, **kwargs: FakeChat(
                callbacks=callbacks
            )
            chat = client.get_chat_model()
            chat.invoke([HumanMessage(content="doc text")])

        client._record_lm_provider_entry(
            {
                "source": "dspy_fallback",
                "doc_id": "doc_fallback",
                "messages": [{"role": "user", "content": "fallback prompt"}],
                "outputs": ["fallback ok"],
                "model": "gemini-3.8-flash",
                "status": "success",
            }
        )

        saved_count = client.save_history(tmp_path, timestamp="unified_ts")
        assert saved_count == 2
        lines = stream_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        entries = [json.loads(line) for line in lines]
        assert entries[0]["outputs"] == ["graph ok"]
        assert entries[1]["source"] == "dspy_fallback"


class TestSemanticOptimizationClient:
    def test_initialization_from_config(self):
        re_cfg = SemanticOptimizationConfig(
            execution={"use_split": "train", "val_split": "val", "f1_threshold": 0.90, "enable_cache": True},
            api_client={"timeout": 240, "max_retries": 4, "rate_limit_delay": 5.0},
            model={
                "provider": "api",
                "model": "qwen/qwen3.5-397b-a17b",
                "temperature": 0.8,
                "api": {"max_tokens": 8192, "api_key": SecretStr("test-key")},
            },
        )
        client = SemanticOptimizationClient(config=re_cfg)
        assert client.model == "qwen/qwen3.5-397b-a17b"
        assert client.provider == "api"
        assert client.llm_config.timeout == 240
        assert client.llm_config.temperature == 0.8

    @patch("ae.semantic_optimization.client.create_chat_model")
    def test_get_chat_model_lazy_init(self, mock_create_chat_model):
        mock_chat = MagicMock()
        mock_create_chat_model.return_value = mock_chat

        client = SemanticOptimizationClient(
            config=SemanticOptimizationConfig(
                model={"provider": "gemini", "model": "gemini-3.8-flash"}
            )
        )
        assert client._chat_model is None
        chat = client.get_chat_model()
        assert chat is mock_chat
        mock_create_chat_model.assert_called_once()

    def test_create_validation_extractor(self):
        ext_cfg = ExtractionConfig(
            model={"provider": "gemini", "model": "gemini-3.8-flash"}
        )
        re_cfg = SemanticOptimizationConfig(
            model={"provider": "gemini", "model": "gemini-3.8-flash"}
        )
        client = SemanticOptimizationClient(config=re_cfg, extraction_config=ext_cfg)

        import dspy
        class DummySignature(dspy.Signature):
            """Dummy."""
            doc: str = dspy.InputField()
            extracted_data: str = dspy.OutputField()

        extractor = client.create_validation_extractor(DummySignature)
        assert extractor is not None
        assert extractor._client is client.get_validation_client()
        assert extractor._client.model == "gemini-3.8-flash"

    def test_save_history(self, tmp_path: Path):
        client = SemanticOptimizationClient(
            config=SemanticOptimizationConfig(
                model={"provider": "gemini", "model": "test-re-model"}
            )
        )
        mock_lm = MagicMock()
        mock_lm.history = [{"messages": [{"role": "user", "content": "hi"}], "outputs": ["hello"], "model": "test-re-model"}]
        client._lm_provider = mock_lm

        saved_count = client.save_history(tmp_path, timestamp="test_ts")
        assert saved_count == 1
        saved_file = tmp_path / "re_test_ts.json"
        assert saved_file.exists()

    @patch("ae.semantic_optimization.client.create_lm")
    @patch("ae.semantic_optimization.client.create_chat_model")
    def test_unified_history_combines_chat_model_and_lm_provider(
        self, mock_create_chat_model, mock_create_lm, tmp_path: Path
    ):
        """SemanticOptimizationClient records both ChatModel callbacks and LMProvider entries into re_{ts}.json."""
        import json
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        mock_chat = MagicMock()
        mock_create_chat_model.return_value = mock_chat

        client = SemanticOptimizationClient(
            config=SemanticOptimizationConfig(
                model={"provider": "gemini", "model": "test-re-model"}
            )
        )
        chat = client.get_chat_model()
        assert chat is mock_chat
        _, kwargs = mock_create_chat_model.call_args
        assert "callbacks" in kwargs and len(kwargs["callbacks"]) == 1
        handler = kwargs["callbacks"][0]

        # Simulate a ChatModel call via callback handler
        handler.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id="run-1",
            invocation_params={"model": "test-re-model"},
        )
        handler.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content="READY"))]]),
            run_id="run-1",
        )

        # Simulate an LMProvider call via history hook
        mock_lm = MagicMock()
        mock_lm.history = []
        mock_create_lm.return_value = mock_lm
        client.get_lm_provider()
        hook = mock_lm.set_history_hook.call_args[0][0]
        hook({"messages": [{"role": "user", "content": "dspy"}], "outputs": ["ok"], "model": "test-re-model"})

        saved_count = client.save_history(tmp_path, timestamp="unified_ts")
        assert saved_count == 2
        saved_file = tmp_path / "re_unified_ts.json"
        assert saved_file.exists()
        data = json.loads(saved_file.read_text(encoding="utf-8"))
        assert len(data) == 2
