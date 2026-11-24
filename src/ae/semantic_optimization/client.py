"""Reverse engineering client for handling LLM interactions during semantic prompt engineering."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import dspy
from langchain_core.language_models.chat_models import BaseChatModel

from ae.core.config.llm_settings import LLMInstanceConfig
from ae.core.config.optimization_settings import CircuitBreakerConfig, ExtractionConfig, SemanticOptimizationConfig
from ae.core.llm.factory import create_chat_model
from ae.core.llm.history_logger import (
    ExtractionHistoryCallbackHandler,
    LLMHistoryRecorder,
)
from ae.core.llm.provider import DSPyLMAdapter, LMProvider, create_lm
from ae.extraction.client import ExtractionClient

logger = logging.getLogger(__name__)


class SemanticOptimizationClient:
    """Dedicated LLM client for Reverse Engineering semantic analysis and validation."""

    def __init__(
        self,
        config: Union[SemanticOptimizationConfig, LLMInstanceConfig],
        extraction_config: Optional[ExtractionConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_circuit_breaker: bool = True,
        save_llm_history: Optional[bool] = None,
    ) -> None:
        """Initialize SemanticOptimizationClient.

        Args:
            config: SemanticOptimizationConfig or pre-built LLMInstanceConfig.
            extraction_config: Optional extraction configuration used during validation evaluation.
            circuit_breaker_config: Optional circuit breaker configuration.
            enable_circuit_breaker: Whether to wrap LLM calls with circuit breaker.
            save_llm_history: Optional override controlling LLM history logging.
        """
        if isinstance(config, SemanticOptimizationConfig):
            self.re_config = config
            self.llm_config = config.get_llm_config()
            default_save_history = bool(config.save_llm_history)
        else:
            self.re_config = None
            self.llm_config = config
            default_save_history = True

        self.save_llm_history: bool = (
            default_save_history if save_llm_history is None else bool(save_llm_history)
        )
        self.extraction_config = extraction_config
        self.circuit_breaker_config = circuit_breaker_config
        self.enable_circuit_breaker = enable_circuit_breaker

        self._chat_model: Optional[BaseChatModel] = None
        self._lm_provider: Optional[LMProvider] = None
        self._dspy_adapter: Optional[DSPyLMAdapter] = None
        self._validation_client: Optional[ExtractionClient] = None
        self._lm_provider_hooked: bool = False
        self._history_recorder: Optional[LLMHistoryRecorder] = (
            LLMHistoryRecorder() if self.save_llm_history else None
        )
        self._callback_handler: Optional[ExtractionHistoryCallbackHandler] = None

    @classmethod
    def from_settings(cls, settings: Any) -> "SemanticOptimizationClient":
        """Instantiate SemanticOptimizationClient directly from application Settings."""
        return cls(
            config=settings.semantic_optimization,
            extraction_config=settings.extraction,
            circuit_breaker_config=settings.circuit_breaker,
        )

    @property
    def model(self) -> str:
        """Return target model identifier."""
        return self.llm_config.model

    @property
    def provider(self) -> str:
        """Return provider type."""
        return self.llm_config.provider

    @property
    def history(self) -> list:
        """Return combined RE history from recorder or underlying LM provider."""
        if self._history_recorder is not None:
            entries = self._history_recorder.get_entries()
            if entries:
                return entries
        if self._lm_provider is not None:
            return getattr(self._lm_provider, "history", [])
        return []

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

    def get_chat_model(self) -> BaseChatModel:
        """Get or lazily initialize LangChain BaseChatModel for manifest unpacking."""
        if self._chat_model is None:
            logger.info(
                f"Initializing SemanticOptimization ChatModel: {self.llm_config.model} "
                f"(provider={self.llm_config.provider})"
            )
            handler = self.get_callback_handler()
            if handler is not None:
                self._chat_model = create_chat_model(self.llm_config, callbacks=[handler])
            else:
                self._chat_model = create_chat_model(self.llm_config)
        return self._chat_model

    def get_lm_provider(self) -> LMProvider:
        """Get or lazily initialize underlying LMProvider for DSPy predictors."""
        if self._lm_provider is None:
            logger.info(
                f"Initializing SemanticOptimization LMProvider: {self.llm_config.model} "
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
        """Get or lazily initialize DSPy adapter for DSPy context."""
        if self._dspy_adapter is None:
            self._dspy_adapter = DSPyLMAdapter(self.get_lm_provider())
        return self._dspy_adapter

    def get_validation_client(self) -> ExtractionClient:
        """Get or initialize ExtractionClient for validation F1 evaluation in SemanticOptimizationGraph."""
        if self._validation_client is None:
            if self.extraction_config is not None:
                self._validation_client = ExtractionClient(
                    config=self.extraction_config,
                    circuit_breaker_config=self.circuit_breaker_config,
                )
            else:
                from ae.core.config.settings import Settings
                settings = Settings.load(load_env_file=True)
                self._validation_client = ExtractionClient.from_settings(settings)
        return self._validation_client

    def create_validation_extractor(
        self,
        signature_class: Any,
        ingestion_dir: Optional[Path] = None,
    ) -> Any:
        """Create a UniversalExtractor configured with the extraction validation client."""
        from ae.extraction.agent import UniversalExtractor

        val_client = self.get_validation_client()
        return UniversalExtractor(
            signature_class,
            client=val_client,
            ingestion_dir=ingestion_dir,
        )

    def save_history(
        self,
        output_dir: Path,
        timestamp: Optional[str] = None,
    ) -> int:
        """Save RE and validation/judge LLM interaction history to disk."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        total_saved = 0
        if self.save_llm_history and self._history_recorder is not None:
            extra_entries = None
            if (
                self._lm_provider is not None
                and not self._lm_provider_hooked
                and getattr(self._lm_provider, "history", None)
            ):
                extra_entries = list(self._lm_provider.history)
            output_path = output_dir / f"re_{timestamp}.json"
            total_saved += self._history_recorder.save_json(output_path, extra_entries=extra_entries)

        if self._validation_client is not None:
            try:
                total_saved += int(
                    self._validation_client.save_history(output_dir, timestamp=timestamp) or 0
                )
            except Exception as exc:
                logger.warning(f"Failed to save validation/judge LLM history: {exc}")
        return total_saved
