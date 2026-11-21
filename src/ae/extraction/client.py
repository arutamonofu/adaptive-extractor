"""Extraction client for handling LLM interactions during extraction."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import dspy
from langchain_core.language_models.chat_models import BaseChatModel

from ae.core.config.llm_settings import LLMInstanceConfig
from ae.core.config.optimization_settings import CircuitBreakerConfig, ExtractionConfig
from ae.core.llm.factory import create_chat_model
from ae.core.llm.history_logger import (
    ExtractionHistoryCallbackHandler,
    LLMHistoryRecorder,
)
from ae.core.llm.provider import DSPyLMAdapter, LMProvider, create_lm

logger = logging.getLogger(__name__)


class ExtractionClient:
    """Dedicated LLM client for batch and interactive data extraction."""

    def __init__(
        self,
        config: Union[ExtractionConfig, LLMInstanceConfig],
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_circuit_breaker: bool = True,
        save_llm_history: Optional[bool] = None,
    ) -> None:
        """Initialize ExtractionClient.

        Args:
            config: ExtractionConfig or pre-built LLMInstanceConfig.
            circuit_breaker_config: Optional circuit breaker configuration.
            enable_circuit_breaker: Whether to wrap LLM calls with circuit breaker.
            save_llm_history: Optional override controlling LLM history logging.
        """
        self.extraction_config: Optional[ExtractionConfig]
        if isinstance(config, ExtractionConfig):
            self.extraction_config = config
            self.llm_config = config.get_llm_config()
            default_save_history = bool(config.save_llm_history)
        else:
            self.extraction_config = None
            self.llm_config = config
            default_save_history = True

        self.save_llm_history: bool = (
            default_save_history if save_llm_history is None else bool(save_llm_history)
        )
        self.circuit_breaker_config = circuit_breaker_config
        self.enable_circuit_breaker = enable_circuit_breaker

        self._chat_model: Optional[BaseChatModel] = None
        self._lm_provider: Optional[LMProvider] = None
        self._dspy_adapter: Optional[DSPyLMAdapter] = None
        self._lm_provider_hooked: bool = False

        self._history_recorder: Optional[LLMHistoryRecorder] = (
            LLMHistoryRecorder() if self.save_llm_history else None
        )
        self._callback_handler: Optional[ExtractionHistoryCallbackHandler] = None

    @classmethod
    def from_settings(cls, settings: Any) -> "ExtractionClient":
        """Instantiate ExtractionClient directly from application Settings."""
        return cls(
            config=settings.extraction,
            circuit_breaker_config=settings.circuit_breaker,
        )

    @property
    def model(self) -> str:
        """Return target model identifier."""
        return self.llm_config.model

    @property
    def provider(self) -> str:
        """Return provider type (gemini, api, ollama)."""
        return self.llm_config.provider

    @property
    def history_recorder(self) -> Optional[LLMHistoryRecorder]:
        """Return the active LLMHistoryRecorder when save_llm_history is enabled."""
        return self._history_recorder

    def get_callback_handler(self) -> Optional[ExtractionHistoryCallbackHandler]:
        """Get or lazily initialize the LangChain history callback handler."""
        if not self.save_llm_history or self._history_recorder is None:
            return None
        if self._callback_handler is None:
            self._callback_handler = ExtractionHistoryCallbackHandler(self._history_recorder)
        return self._callback_handler

    def _record_lm_provider_entry(self, entry: Dict[str, Any]) -> None:
        if self._history_recorder is not None:
            self._history_recorder.record(entry)

    def _attach_lm_provider_hook(self, provider: Any) -> None:
        if not self.save_llm_history or self._history_recorder is None:
            return
        if hasattr(provider, "set_history_hook") and callable(provider.set_history_hook):
            provider.set_history_hook(self._record_lm_provider_entry)
            self._lm_provider_hooked = True

    def start_history_session(
        self,
        output_dir: Path,
        timestamp: Optional[str] = None,
    ) -> Optional[Path]:
        """Start incremental JSONL history streaming into output_dir if enabled."""
        if not self.save_llm_history or self._history_recorder is None:
            return None
        if self._history_recorder.is_streaming:
            return self._history_recorder.stream_path
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"extraction_{timestamp}.jsonl"
        stream_path = self._history_recorder.start_stream(output_path)
        if self._lm_provider is not None and not self._lm_provider_hooked:
            self._attach_lm_provider_hook(self._lm_provider)
        return stream_path

    def get_chat_model(self) -> BaseChatModel:
        """Get or lazily initialize LangChain BaseChatModel."""
        if self._chat_model is None:
            logger.info(
                f"Initializing Extraction ChatModel: {self.llm_config.model} "
                f"(provider={self.llm_config.provider})"
            )
            handler = self.get_callback_handler()
            if handler is not None:
                self._chat_model = create_chat_model(
                    self.llm_config, callbacks=[handler], max_retries=1
                )
            else:
                self._chat_model = create_chat_model(self.llm_config, max_retries=1)
        return self._chat_model

    def get_lm_provider(self) -> LMProvider:
        """Get or lazily initialize underlying LMProvider."""
        if self._lm_provider is None:
            logger.info(
                f"Initializing Extraction LMProvider: {self.llm_config.model} "
                f"(cache={self.llm_config.enable_cache})"
            )
            self._lm_provider = create_lm(
                self.llm_config,
                circuit_breaker_config=self.circuit_breaker_config,
                enable_circuit_breaker=self.enable_circuit_breaker,
                enable_cache=self.llm_config.enable_cache,
            )
            self._attach_lm_provider_hook(self._lm_provider)
        return self._lm_provider

    def get_dspy_adapter(self) -> DSPyLMAdapter:
        """Get or lazily initialize DSPy adapter for DSPy modules."""
        if self._dspy_adapter is None:
            self._dspy_adapter = DSPyLMAdapter(self.get_lm_provider())
        return self._dspy_adapter

    def configure_dspy(self) -> DSPyLMAdapter:
        """Configure DSPy global settings with this extraction LM."""
        adapter = self.get_dspy_adapter()
        dspy.settings.configure(lm=adapter)
        logger.info(f"DSPy configured with Extraction model: {self.model}")
        return adapter

    def save_history(
        self,
        output_dir: Path,
        timestamp: Optional[str] = None,
    ) -> int:
        """Flush or save extraction LLM interaction history to disk."""
        if not self.save_llm_history or self._history_recorder is None:
            return 0

        if self._history_recorder.is_streaming:
            if (
                self._lm_provider is not None
                and not self._lm_provider_hooked
                and getattr(self._lm_provider, "history", None)
            ):
                for entry in self._lm_provider.history:
                    self._history_recorder.record(entry)
                self._lm_provider_hooked = True
            self._history_recorder.close()
            count = self._history_recorder.recorded_count
            if count > 0:
                logger.info(
                    f"Saved {count} LLM calls to {self._history_recorder.stream_path}"
                )
            return count

        extra_entries = None
        if (
            self._lm_provider is not None
            and not self._lm_provider_hooked
            and getattr(self._lm_provider, "history", None)
        ):
            extra_entries = list(self._lm_provider.history)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"extraction_{timestamp}.json"
        return self._history_recorder.save_json(output_path, extra_entries=extra_entries)
