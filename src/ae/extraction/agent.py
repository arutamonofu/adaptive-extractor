"""Extraction agents and lifecycle management.

This module defines base agent interfaces, concrete extractor implementations,
and protocols for serialization.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Type, Union, runtime_checkable

import dspy

from ae.core.llm.circuit_breaker import CircuitBreakerError
from ae.core.llm.history_logger import extraction_log_context

logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"<!--\s*document_id:\s*([^\s>]+)\s*-->")


class BaseAgent(ABC):
    """Abstract base class for extraction agents.

    Allows the application layer to work with agents without depending on
    specific DSPy implementations.
    """

    @abstractmethod
    def forward(self, document_text: str) -> Any:
        """Execute the extraction pipeline.

        Args:
            document_text: The input text to process.

        Returns:
            Extraction result.
        """
        pass

    @abstractmethod
    def save(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Save the agent to a file."""
        pass

    @abstractmethod
    def load(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Load the agent from a file."""
        pass


class UniversalExtractorMeta(type(BaseAgent), type(dspy.Module)):  # type: ignore[misc]
    """Metaclass for UniversalExtractor to resolve metaclass conflict."""
    pass


class UniversalExtractor(BaseAgent, dspy.Module, metaclass=UniversalExtractorMeta):
    """Task-agnostic extraction agent.

    Wraps a specific task signature with Chain-of-Thought reasoning,
    executing a LangGraph ReAct ExtractionGraph with tool-calling capabilities.
    """

    def __init__(
        self,
        signature_class: Type[dspy.Signature],
        client: Optional[Any] = None,
        chat_model: Optional[Any] = None,
        ingestion_dir: Optional[Path] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the UniversalExtractor."""
        BaseAgent.__init__(self)
        dspy.Module.__init__(self)
        self.prog = dspy.ChainOfThought(signature_class)
        self.signature_class = signature_class
        self._client = client
        self._chat_model = chat_model
        self._ingestion_dir = Path(ingestion_dir) if ingestion_dir else None
        self._config_path = Path(config_path) if config_path else None
        self._graph: Optional[Any] = None

    def _resolve_config_path(self) -> Optional[Path]:
        """Return explicit config_path from self or attached client when available."""
        if isinstance(self._config_path, Path):
            return self._config_path
        raw_cfg = getattr(self._client, "config_path", None)
        if isinstance(raw_cfg, (str, Path)):
            return Path(raw_cfg)
        return None

    def _get_or_create_graph(self) -> Optional[Any]:
        """Lazy initialization of ExtractionGraph from application settings."""
        if self._graph is not None:
            return self._graph

        try:
            from ae.core.config.settings import Settings
            from ae.extraction.client import ExtractionClient
            from ae.extraction.graph import ExtractionGraph

            output_field = self.signature_class.output_fields.get("extracted_data")
            output_model = getattr(output_field, "annotation", None)
            if output_model is None or not (isinstance(output_model, type) and issubclass(output_model, object)):
                return None

            cfg_path = self._resolve_config_path()
            settings = (
                Settings.load(config_path=cfg_path, load_env_file=True)
                if cfg_path is not None
                else Settings.load(load_env_file=True)
            )
            if self._chat_model is not None:
                llm = self._chat_model
            elif self._client is not None:
                llm = self._client.get_chat_model()
            else:
                client = ExtractionClient.from_settings(settings)
                self._client = client
                llm = client.get_chat_model()

            ext_cfg = getattr(self._client, "extraction_config", None) or settings.extraction
            max_iterations = getattr(ext_cfg, "max_tool_iterations", 5)
            api_cfg = getattr(ext_cfg, "api_client", None)
            max_retries = getattr(api_cfg, "max_retries", 5) if api_cfg else 5
            retry_delay = getattr(api_cfg, "rate_limit_delay", 2.0) if api_cfg else 2.0
            batch_size = getattr(ext_cfg, "structured_output_batch_size", 0)
            max_batches = getattr(ext_cfg, "max_structured_output_batches", 10)
            enable_sanity_checks = getattr(ext_cfg, "enable_sanity_checks", True)
            agent_thinking_level = getattr(ext_cfg, "agent_thinking_level", "low")
            exec_cfg = getattr(ext_cfg, "execution", None)
            enable_record_dedup = bool(
                getattr(
                    ext_cfg,
                    "enable_record_dedup",
                    getattr(exec_cfg, "enable_record_dedup", True) if exec_cfg is not None else True,
                )
            )
            ingestion_dir = self._ingestion_dir or settings.paths.ingestion_dir

            history_recorder = getattr(self._client, "history_recorder", None)

            self._graph = ExtractionGraph(
                llm=llm,
                output_model=output_model,
                max_tool_iterations=max_iterations,
                ingestion_dir=ingestion_dir,
                max_retries=max_retries,
                retry_delay=retry_delay,
                structured_output_batch_size=batch_size,
                max_structured_output_batches=max_batches,
                history_recorder=history_recorder,
                enable_sanity_checks=enable_sanity_checks,
                agent_thinking_level=agent_thinking_level,
                enable_record_dedup=enable_record_dedup,
                config_path=cfg_path,
            )
            return self._graph
        except Exception as e:
            logger.warning(f"Could not initialize ExtractionGraph, falling back to DSPy prog: {e}")
            return None

    def _resolve_latest_document_text(
        self,
        document_text: str,
        graph: Optional[Any],
    ) -> str:
        """Resolve the most enriched document text from disk and/or graph state for fallback."""
        m = _DOC_ID_RE.search(document_text)
        if not m:
            return document_text

        doc_id = m.group(1).strip()
        candidates: list[str] = []

        # 1. Fresh read from disk via DocumentRepository
        try:
            from ae.core.storage.documents import DocumentRepository

            if graph is not None and getattr(graph, "document_repo", None) is not None:
                disk_text = graph.document_repo.get(doc_id, getattr(graph, "ingestion_dir", None))
            else:
                ingestion_dir = self._ingestion_dir
                if ingestion_dir is None:
                    from ae.core.config.settings import Settings

                    cfg_path = self._resolve_config_path()
                    ingestion_dir = (
                        Settings.load(config_path=cfg_path, load_env_file=False)
                        if cfg_path is not None
                        else Settings.load(load_env_file=False)
                    ).paths.ingestion_dir
                disk_text = DocumentRepository(ingestion_dir).get(doc_id)

            if disk_text:
                candidates.append(disk_text)
        except Exception as e:
            logger.debug(f"Could not refresh document '{doc_id}' from disk for fallback: {e}")

        # 2. In-memory enriched text recorded by ExtractionGraph
        if graph is not None and hasattr(graph, "get_last_document_text"):
            graph_text = graph.get_last_document_text(doc_id)
            if graph_text:
                candidates.append(graph_text)

        # 3. Original input text
        candidates.append(document_text)

        from ae.ingestion.parsers.gemini.stages.render_markdown import extract_unparsed_visual_ids

        # Pick candidate with fewest remaining unparsed visuals (ties prefer disk -> graph -> original)
        best_text = min(
            candidates,
            key=lambda txt: len(extract_unparsed_visual_ids(txt)),
        )
        if not best_text.startswith(f"<!-- document_id: {doc_id} -->"):
            best_text = f"<!-- document_id: {doc_id} -->\n\n{best_text}"

        remaining = len(extract_unparsed_visual_ids(best_text))
        if best_text != document_text:
            logger.info(
                f"Refreshed fallback document_text for '{doc_id}' "
                f"(len={len(best_text)}, remaining_unparsed_visuals={remaining})"
            )
        return best_text

    def _run_dspy_fallback(self, document_text: str) -> dspy.Prediction:
        """Execute DSPy ChainOfThought fallback, respecting client adapter if configured."""
        if self._client is not None and hasattr(self._client, "get_dspy_adapter"):
            try:
                adapter = self._client.get_dspy_adapter()
                if isinstance(adapter, dspy.LM):
                    with dspy.context(lm=adapter):
                        return self.prog(document_text=document_text)
            except Exception as e:
                logger.debug(f"Could not bind client DSPy adapter for fallback: {e}")
        return self.prog(document_text=document_text)

    def forward(self, document_text: str) -> dspy.Prediction:
        """Execute the extraction pipeline via ExtractionGraph or DSPy fallback."""
        graph = self._get_or_create_graph()
        if graph is not None:
            try:
                instruction = ""
                if (
                    hasattr(self.prog, "predict")
                    and hasattr(self.prog.predict, "signature")
                    and getattr(self.prog.predict.signature, "instructions", None)
                ):
                    instruction = self.prog.predict.signature.instructions or ""
                elif hasattr(self.prog, "extended_signature") and hasattr(self.prog.extended_signature, "instructions"):
                    instruction = self.prog.extended_signature.instructions or ""
                elif self.signature_class.__doc__:
                    instruction = self.signature_class.__doc__

                res = graph.run(
                    document_text=document_text,
                    system_instruction=instruction,
                )
                if res.get("extracted_data") is not None:
                    return dspy.Prediction(
                        extracted_data=res["extracted_data"],
                        rationale=res.get("rationale", ""),
                    )
            except CircuitBreakerError:
                raise
            except Exception as exc:
                logger.warning(f"ExtractionGraph execution failed: {exc}. Falling back to DSPy prog.")

        fallback_text = self._resolve_latest_document_text(document_text, graph)
        m = _DOC_ID_RE.search(fallback_text)
        fallback_doc_id = m.group(1).strip() if m else None
        with extraction_log_context(
            doc_id=fallback_doc_id,
            source="dspy_fallback",
            stage="dspy_fallback",
        ):
            return self._run_dspy_fallback(fallback_text)

    def save(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Save the agent to a file."""
        dspy.Module.save(self, path, *args, **kwargs)

    def load(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Load the agent from a file."""
        dspy.Module.load(self, path, *args, **kwargs)


@runtime_checkable
class SerializableAgent(Protocol):
    """Protocol for agents that can be serialized."""

    def dump_state(self) -> Dict[str, Any]:
        """Dump agent state to a dictionary."""
        ...


@runtime_checkable
class SaveableAgent(Protocol):
    """Protocol for agents that can be saved to a file."""

    def save(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Save agent to a file."""
        ...
