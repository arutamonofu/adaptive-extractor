"""ExtractionGraph: LangGraph ReAct extraction workflow with tool calling."""

from __future__ import annotations

import functools
import http.client
import inspect
import json
import logging
import operator
import random
import re
import ssl
import time
from pathlib import Path
from typing import Annotated, Any, Callable, NotRequired, Optional, Sequence, Type, TypedDict, get_args, get_origin
from unittest.mock import NonCallableMock

import httpx
import requests
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_core import PydanticUndefined

from ae.core.config.settings import Settings
from ae.core.llm.circuit_breaker import CircuitBreakerError
from ae.core.llm.provider import model_supports_sampling_params
from ae.core.llm.history_logger import (
    ExtractionHistoryCallbackHandler,
    LLMHistoryRecorder,
    extraction_log_context,
    format_error_payload,
    get_extraction_log_context,
    update_extraction_log_context,
)
from ae.core.schema.dynamic_models import (
    extract_schema_field_constraints,
    extract_schema_tools,
)
from ae.core.schema.tools import (
    DEFAULT_TOOL_REGISTRY,
    ExtractionToolSpec,
    RegisteredToolProvider,
    normalize_tool_spec,
)
from ae.core.storage.documents import DocumentRepository
from ae.extraction.sanity import (
    SanityCheckConfig,
    StructuredOutputSanityError,
    validate_output_field_validators,
    validate_structured_output_sanity,
)
from ae.ingestion.parsers.gemini.stages.render_markdown import (
    extract_unparsed_visual_ids,
    has_unparsed_visuals,
    replace_manifest_with_enrichment,
)
from ae.ingestion.service import VisualParsingService
from ae.ingestion.tool import make_visual_parsing_tool

logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"<!--\s*document_id:\s*([^\s>]+)\s*-->")
_VISUAL_BLOCK_EXTRACT_RE = re.compile(
    r'<visual\s+id=["\'](?P<id>[^"\']+)["\'][^>]*>.*?</visual>',
    re.DOTALL,
)
_UNPARSED_ELEMENT_TAG_RE = re.compile(
    r"<(?:element|atom)\s+id=[\"'](?P<id>[^\"']+)[\"'][^>]*>\[Unparsed",
    re.IGNORECASE,
)
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_RETRYABLE_MSG_MARKERS = (
    "connection reset",
    "readerror",
    "remoteprotocolerror",
    "timeout",
    "timed out",
    "unavailable",
    "resource_exhausted",
    "deadline_exceeded",
    "rate limit",
    "connection aborted",
    "broken pipe",
    "eof occurred",
    "429",
    "500",
    "502",
    "503",
    "504",
)


def _is_parse_or_truncation_error(exc: Exception) -> bool:
    """Return True if exception looks like malformed/truncated structured output."""
    if isinstance(exc, (OutputParserException, ValidationError, json.JSONDecodeError)):
        return True
    msg = str(exc).lower()
    return any(tok in msg for tok in ("unterminated", "eof while parsing", "truncat", "jsondecodeerror"))


def _is_retryable_error(exc: Exception) -> bool:
    """Determine whether an LLM/network error is transient and safe to retry."""
    if isinstance(exc, (CircuitBreakerError, KeyboardInterrupt, SystemExit)):
        return False

    if _is_parse_or_truncation_error(exc):
        return True

    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status_code, int):
        if status_code in _RETRYABLE_STATUS_CODES:
            return True
        if 400 <= status_code < 500:
            return False

    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            OSError,
            httpx.HTTPError,
            requests.RequestException,
            http.client.HTTPException,
            ssl.SSLError,
        ),
    ):
        return True

    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_MSG_MARKERS)


def _supports_record_batching(output_model: Type[BaseModel]) -> bool:
    """Return True if output_model is a collection schema with a `records` list field."""
    if not output_model or not hasattr(output_model, "model_fields"):
        return False
    field_info = output_model.model_fields.get("records")
    if field_info is None:
        return False
    annotation = field_info.annotation
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            annotation = args[0]
            origin = get_origin(annotation)
    return origin in (list, Sequence, tuple) or (
        isinstance(annotation, type) and issubclass(annotation, list)
    )


def _extract_record_model(output_model: Type[BaseModel]) -> Optional[Type[BaseModel]]:
    """Extract the record BaseModel class from an output_model with a `records: list[...]` field."""
    if not output_model or not hasattr(output_model, "model_fields"):
        return None
    field_info = output_model.model_fields.get("records")
    if field_info is None:
        return None
    annotation = field_info.annotation
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            annotation = args[0]
            origin = get_origin(annotation)
    if origin in (list, Sequence, tuple):
        args = get_args(annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    return None


def _tool_output_has_error(output: str) -> bool:
    """Return True if a domain tool output string represents an error or non-empty errors list."""
    stripped = output.strip()
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    return bool(parsed.get("error") or parsed.get("errors"))


_DASH_TRANS = str.maketrans({"−": "-", "–": "-", "—": "-", "\u03bc": "\u00b5"})
_WS_RE = re.compile(r"\s+")
_FLOAT_STR_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_COLLECTION_ID_RE = re.compile(
    r"\b(ATCC|MTCC|NCTC|DSM|CMCC|CGMCC|NBRC|JCM|NCIM|CIP|LMG|CCM|KCTC|BCRC)\s*[-#]?\s*[A-Za-z0-9]+\b",
    re.IGNORECASE,
)
_PHENOTYPE_TAG_RE = re.compile(
    r"\b(MSSA|MRSA|VRE|VRSA|ESBL|MDR|XDR|PRSP|CRKP|CRPA|CRAB)\b",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=4096)
def _canonicalize_smiles_for_dedup(raw_smiles: str) -> str:
    """Canonicalize SMILES to InChIKey (via chembl_structure_pipeline + RDKit) for dedup."""
    cleaned = raw_smiles.strip()
    if not cleaned:
        return ""
    try:
        from rdkit import Chem, rdBase

        rdBase.DisableLog("rdApp.error")
        rdBase.DisableLog("rdApp.warning")
    except ImportError:
        return cleaned

    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return cleaned

    try:
        from chembl_structure_pipeline import standardizer

        mol = standardizer.standardize_mol(mol)
    except Exception:
        pass

    try:
        inchi_key = Chem.MolToInchiKey(mol)
        if inchi_key:
            return inchi_key
    except Exception:
        pass

    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return cleaned


def _normalize_scalar_for_dedup(key: str, value: Any) -> Any:
    """Normalize field values canonically for deduplication without fuzzy string matching."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if not isinstance(value, str):
        return value

    cleaned = _WS_RE.sub(" ", value.translate(_DASH_TRANS)).strip()
    if not cleaned:
        return ""

    key_l = key.lower()
    if "smiles" in key_l:
        return _canonicalize_smiles_for_dedup(cleaned)

    if "unit" in key_l:
        cleaned = re.sub(r"\bug\b", "\u00b5g", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\buM\b", "\u00b5M", cleaned)
        try:
            import pint

            ureg = pint.UnitRegistry()
            qty = ureg.Quantity(1.0, cleaned)
            return f"{qty.to_base_units().units}"
        except Exception:
            return cleaned

    if any(tok in key_l for tok in ("bacteria", "strain", "organism")):
        if _COLLECTION_ID_RE.search(cleaned):
            stripped = _PHENOTYPE_TAG_RE.sub("", cleaned)
            stripped = re.sub(r"[\(\)\[\]\|,;/]+", " ", stripped)
            return _WS_RE.sub(" ", stripped).strip().lower()
        return cleaned

    if _FLOAT_STR_RE.match(cleaned):
        try:
            num = float(cleaned)
            return str(int(num)) if num.is_integer() else f"{num:g}"
        except (ValueError, OverflowError):
            pass

    return cleaned


def _normalize_for_dedup(value: Any, *, field_key: str = "") -> Any:
    if isinstance(value, list):
        return [_normalize_for_dedup(v, field_key=field_key) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_for_dedup(v, field_key=str(k)) for k, v in value.items()}
    return _normalize_scalar_for_dedup(field_key, value)


def _record_dedup_key(record: Any) -> str:
    """Build a canonical deduplication key for an extracted record."""
    if isinstance(record, BaseModel):
        data = record.model_dump(mode="json")
    elif isinstance(record, dict):
        data = record
    else:
        return repr(record)
    return json.dumps(
        _normalize_for_dedup(data),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _make_finalize_evidence_tool() -> Any:
    """Build the terminal tool allowing the agent to exit tool-routing without text drafts."""
    from langchain_core.tools import StructuredTool

    class _FinalizeEvidenceInput(BaseModel):
        confirm: bool = Field(
            default=True,
            description="Set to true when all required visual and domain tools have been called.",
        )

    def _finalize(confirm: bool = True) -> str:
        return "Evidence gathering finalized."

    return StructuredTool.from_function(
        func=_finalize,
        name="finalize_evidence",
        description=(
            "Call this tool when all needed visual parsing and domain tools have completed "
            "and you are ready to proceed to structured extraction."
        ),
        args_schema=_FinalizeEvidenceInput,
    )


def _split_visual_id(visual_id: str) -> tuple[str, bool]:
    """Return (root_visual_id, is_subelement) for a visual or element ID."""
    cleaned = str(visual_id).strip()
    if "/" in cleaned:
        return cleaned.partition("/")[0].strip(), True
    if "#" in cleaned:
        return cleaned.partition("#")[0].strip(), True
    return cleaned, False


def _normalize_hint(target_hint: Optional[str]) -> Optional[str]:
    """Normalize target_hint string; return None when empty or absent."""
    if isinstance(target_hint, str):
        stripped = target_hint.strip()
        if stripped:
            return stripped.lower()
    return None


def _extract_visual_block(text: str, visual_id: str) -> Optional[str]:
    """Extract the <visual id='visual_id'>...</visual> block from text if present."""
    if not text or "<visual" not in text:
        return None
    for match in _VISUAL_BLOCK_EXTRACT_RE.finditer(text):
        if match.group("id").strip() == visual_id:
            return match.group(0)
    return None


def _classify_visual_outcome(
    vid: str,
    tool_output: str,
    updated_doc_text: str,
) -> tuple[str, bool]:
    """Classify the outcome of a parse_visuals call for `vid`.

    Returns:
        (outcome, root_fully_resolved) where outcome is 'completed', 'partial', or 'failed',
        and root_fully_resolved is True when a sub-element parse resolved the last remaining
        unparsed element of its root visual.
    """
    out = (tool_output or "").strip()
    root_id, is_sub = _split_visual_id(vid)

    if (
        not out
        or out.startswith("Parsing failed for")
        or out == "No visual_ids provided."
        or f"[{vid}: parsing failed]" in out
        or f"[{root_id}: parsing failed]" in out
    ):
        return "failed", False

    block_in_out = _extract_visual_block(out, root_id)
    block_in_doc = _extract_visual_block(updated_doc_text, root_id)
    ref_block = block_in_out or block_in_doc

    if ref_block is None or "*[Parsing failed]*" in ref_block:
        return "failed", False

    unparsed_in_block = {m.group("id").strip() for m in _UNPARSED_ELEMENT_TAG_RE.finditer(ref_block)}
    has_partial_note = f"[Partial parse for {root_id}" in out

    if is_sub:
        if vid in unparsed_in_block:
            return "failed", False
        root_resolved = not unparsed_in_block and not has_partial_note
        return "completed", root_resolved

    if has_partial_note or bool(unparsed_in_block):
        return "partial", False
    return "completed", True


def _merge_tool_visuals_into_document(
    doc_text: str,
    tool_messages: Sequence[BaseMessage],
) -> str:
    """Merge <visual> blocks from tool messages into doc_text if not already enriched."""
    updated = doc_text
    for msg in tool_messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or "<visual" not in content:
            continue
        for match in _VISUAL_BLOCK_EXTRACT_RE.finditer(content):
            vis_id = match.group("id").strip()
            block = match.group(0).strip()
            updated = replace_manifest_with_enrichment(updated, vis_id, block)
    return updated


class ExtractionState(TypedDict):
    """State for multi-turn extraction agent with tool calling."""

    doc_id: str
    document_text: str
    system_instruction: str
    messages: Annotated[list[BaseMessage], operator.add]
    tool_iterations: int
    attempted_visuals: list[str]
    completed_visuals: NotRequired[list[str]]
    partial_visuals: NotRequired[list[str]]
    attempted_hints: NotRequired[dict[str, list[str]]]
    visual_attempt_counts: NotRequired[dict[str, int]]
    tool_results: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    executed_tool_signatures: NotRequired[list[str]]
    non_visual_tools_called_since_doc_change: NotRequired[list[str]]
    last_turn_had_non_visual: NotRequired[bool]
    last_turn_had_tool_error: NotRequired[bool]
    domain_reminder_sent: NotRequired[bool]
    extracted_data: Optional[Any]
    rationale: Optional[str]


class ExtractionGraph:
    """LangGraph ReAct extraction workflow.

    Workflow:
      START -> prepare_input -> (conditional: agent_node vs structured_output_node)
                                      |
                                      +--> agent_node -> (conditional: tool_node vs structured_output_node)
                                                               |
                                            tool_node <--------+
                                                |
                                                +--> (conditional: agent_node vs structured_output_node)
      structured_output_node -> END
    """

    def __init__(
        self,
        llm: BaseChatModel,
        output_model: Type[BaseModel],
        service: Optional[VisualParsingService] = None,
        max_tool_iterations: int = 5,
        max_visual_attempts: int = 2,
        ingestion_dir: Optional[Path] = None,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        structured_output_batch_size: int = 0,
        max_structured_output_batches: int = 10,
        use_streaming: bool = True,
        history_recorder: Optional[LLMHistoryRecorder] = None,
        enable_sanity_checks: bool = True,
        enable_field_validators: bool = True,
        agent_thinking_level: Optional[str] = "low",
        tools: Optional[Sequence[Union[str, Dict[str, Any], ExtractionToolSpec]]] = None,
        enable_record_dedup: bool = True,
        agent_llm: Optional[BaseChatModel] = None,
        repair_llm: Optional[BaseChatModel] = None,
    ) -> None:
        self.llm = llm
        self.agent_llm = agent_llm
        self.repair_llm = repair_llm
        self.output_model = output_model
        self.service = service
        self.max_tool_iterations = max_tool_iterations
        self.max_visual_attempts = max(1, int(max_visual_attempts))
        self.ingestion_dir = ingestion_dir or Path("data/interim/ingestion")
        self.document_repo = DocumentRepository(self.ingestion_dir)
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.structured_output_batch_size = max(0, int(structured_output_batch_size))
        self.max_structured_output_batches = max(1, int(max_structured_output_batches))
        self.use_streaming = use_streaming
        self.enable_sanity_checks = bool(enable_sanity_checks)
        self.enable_field_validators = bool(enable_field_validators)
        self.agent_thinking_level = agent_thinking_level
        self.enable_record_dedup = bool(enable_record_dedup)
        self._sanity_config = SanityCheckConfig(
            enabled=self.enable_sanity_checks,
            check_field_validators=self.enable_field_validators,
        )
        self.history_recorder = history_recorder
        self._callback_handler: Optional[ExtractionHistoryCallbackHandler] = (
            ExtractionHistoryCallbackHandler(history_recorder)
            if history_recorder is not None
            else None
        )
        self._last_document_texts: dict[str, str] = {}
        self._last_disk_texts: dict[str, str] = {}

        if tools is not None:
            self._tool_specs: list[ExtractionToolSpec] = [normalize_tool_spec(t) for t in tools]
        else:
            self._tool_specs = extract_schema_tools(output_model)
        self._domain_tool_providers: list[RegisteredToolProvider] = []
        self._domain_tool_params: dict[str, dict[str, Any]] = {}
        for t_spec in self._tool_specs:
            if not t_spec.enabled:
                continue
            try:
                provider = DEFAULT_TOOL_REGISTRY.resolve(t_spec)
                self._domain_tool_providers.append(provider)
                self._domain_tool_params[provider.name] = dict(t_spec.params)
            except Exception as exc:
                logger.warning(f"Could not resolve schema extraction tool '{t_spec.name}': {exc}")

        # Compile LangGraph
        self.app = self._build_graph()

    def _active_domain_tools(self, doc_text: str) -> list[RegisteredToolProvider]:
        """Return schema-configured domain tool providers applicable to ``doc_text``."""
        active: list[RegisteredToolProvider] = []
        for provider in self._domain_tool_providers:
            try:
                if provider.is_applicable(doc_text):
                    active.append(provider)
            except Exception as exc:
                logger.debug(f"Tool applicability check failed for '{provider.name}': {exc}")
        return active

    def _build_domain_tool_instance(
        self,
        provider: RegisteredToolProvider,
        *,
        doc_id: str,
        doc_text: str,
    ) -> Any:
        """Instantiate a LangChain tool from ``provider``, passing only accepted context kwargs."""
        params = dict(self._domain_tool_params.get(provider.name, {}))
        candidate_kwargs: dict[str, Any] = {
            "doc_id": doc_id,
            "doc_text": doc_text,
            **params,
        }
        try:
            sig = inspect.signature(provider.build_tool)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if accepts_var_kw:
                return provider.build_tool(**candidate_kwargs)
            accepted = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
            return provider.build_tool(**accepted)
        except (ValueError, TypeError):
            return provider.build_tool()

    @staticmethod
    def _format_domain_tools_policy(providers: Sequence[RegisteredToolProvider]) -> str:
        """Format policy instructions for active schema-declared domain tools."""
        hints = [p.policy_hint.strip() for p in providers if (p.policy_hint or "").strip()]
        if not hints:
            return ""
        return "DOMAIN EXTRACTION TOOLS:\n" + "\n\n".join(hints)

    @staticmethod
    def _tool_call_signature(tool_name: str, tool_args: Any) -> str:
        """Build a canonical signature string for deduplicating identical non-visual tool calls."""
        return json.dumps(
            {"name": tool_name, "args": tool_args},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _format_tool_results_section(state: ExtractionState) -> str:
        """Format computed non-visual tool outputs as an authoritative section for structured extraction."""
        results = state.get("tool_results") or []
        if not results:
            return ""
        chunks: list[str] = ["--- COMPUTED TOOL RESULTS (AUTHORITATIVE) ---"]
        for entry in results:
            t_name = entry.get("tool", "tool")
            t_out = entry.get("output", "")
            chunks.append(f"[{t_name}]\n{t_out}")
        return "\n\n".join(chunks)

    def get_last_document_text(self, doc_id: str) -> Optional[str]:
        """Return the latest in-memory document text recorded for doc_id, if any."""
        return self._last_document_texts.get(doc_id)

    def _record_history_event(
        self,
        *,
        status: str,
        node: Optional[str] = None,
        stage: Optional[str] = None,
        doc_id: Optional[str] = None,
        messages: Optional[Sequence[Any]] = None,
        outputs: Optional[Sequence[Any]] = None,
        tool_calls: Optional[Sequence[Any]] = None,
        error: Optional[BaseException] = None,
        latency_s: Optional[float] = None,
        attempt: Optional[int] = None,
        batch_idx: Optional[int] = None,
        llm_run_id: Optional[str] = None,
    ) -> None:
        if self.history_recorder is None:
            return
        ctx = get_extraction_log_context()
        entry: dict[str, Any] = {
            "source": ctx.get("source") or "extraction_graph",
            "doc_id": doc_id or ctx.get("doc_id"),
            "node": node or ctx.get("node"),
            "stage": stage or ctx.get("stage"),
            "attempt": attempt if attempt is not None else ctx.get("attempt"),
            "batch_idx": batch_idx if batch_idx is not None else ctx.get("batch_idx"),
            "status": status,
            "model": getattr(self.llm, "model", None) or getattr(self.llm, "model_name", "") or "",
            "messages": list(messages) if messages is not None else [],
            "outputs": list(outputs) if outputs is not None else [],
        }
        if tool_calls:
            entry["tool_calls"] = list(tool_calls)
        if latency_s is not None:
            entry["latency_s"] = latency_s
        if error is not None:
            entry["error"] = format_error_payload(error)
        if llm_run_id:
            entry["llm_run_id"] = llm_run_id
        self.history_recorder.record(entry)

    def _build_graph(self) -> Any:
        workflow = StateGraph(ExtractionState)

        workflow.add_node("prepare_input", self._prepare_input)
        workflow.add_node("agent_node", self._agent_node)
        workflow.add_node("tool_node", self._tool_node)
        workflow.add_node("structured_output_node", self._structured_output_node)

        workflow.add_edge(START, "prepare_input")
        workflow.add_conditional_edges(
            "prepare_input",
            self._route_after_prepare,
            {
                "agent": "agent_node",
                "structured_output": "structured_output_node",
            },
        )
        workflow.add_conditional_edges(
            "agent_node",
            self._should_continue,
            {
                "tools": "tool_node",
                "structured_output": "structured_output_node",
            },
        )
        workflow.add_conditional_edges(
            "tool_node",
            self._route_after_tool,
            {
                "agent": "agent_node",
                "structured_output": "structured_output_node",
            },
        )
        workflow.add_edge("structured_output_node", END)

        return workflow.compile()

    def _resolve_service(self) -> VisualParsingService:
        if self.service is not None:
            return self.service
        try:
            settings = Settings.load(load_env_file=True)
            self.service = VisualParsingService(
                config=settings.ingestion,
                pdf_dir=settings.paths.pdf_dir,
                ingestion_dir=settings.paths.ingestion_dir,
            )
        except Exception:
            self.service = VisualParsingService()
        return self.service

    def _format_schema_summary(self) -> str:
        """Extract a readable summary of target fields, descriptions, and constraints from output_model."""
        constraints = extract_schema_field_constraints(self.output_model)
        if not constraints:
            return ""

        summary_lines = []
        for name, info in constraints.items():
            if info.get("required"):
                req_str = "required"
            elif "default" in info and info["default"] is not None:
                req_str = f"optional, default: {info['default']!r}"
            else:
                req_str = "optional"

            constraint_parts = []
            if info.get("enum"):
                allowed = ", ".join(repr(c) for c in info["enum"])
                constraint_parts.append(f"allowed: [{allowed}]")
            if info.get("pattern"):
                constraint_parts.append(f"pattern: {info['pattern']!r}")
            if "minimum" in info or "maximum" in info:
                bounds = []
                if "minimum" in info:
                    bounds.append(f"min={info['minimum']}")
                if "maximum" in info:
                    bounds.append(f"max={info['maximum']}")
                constraint_parts.append(", ".join(bounds))
            if info.get("validators"):
                constraint_parts.append(f"validators: [{', '.join(info['validators'])}]")

            suffix = f" [{' | '.join(constraint_parts)}]" if constraint_parts else ""
            desc = info.get("description")
            if desc:
                summary_lines.append(f"- `{name}` ({req_str}){suffix}: {desc}")
            else:
                summary_lines.append(f"- `{name}` ({req_str}){suffix}")

        if summary_lines:
            return "TARGET SCHEMA FIELDS:\n" + "\n".join(summary_lines)
        return ""

    def _prepare_input(self, state: ExtractionState) -> dict[str, Any]:
        """Extract doc_id, re-read fresh document from disk, and init message sequence."""
        raw_text = state.get("document_text", "")
        doc_id = state.get("doc_id", "")

        if not doc_id:
            m = _DOC_ID_RE.search(raw_text)
            if m:
                doc_id = m.group(1).strip()
            else:
                doc_id = "unknown_document"

        # Read latest document markdown from disk (resetting from Stage 02 manifest stubs if available
        # so inference/validation starts clean while reusing cached 04_parse_visuals/ for 0 VLM tokens)
        doc_file = self.document_repo._find_document_path(doc_id, self.ingestion_dir)

        if doc_file and doc_file.exists():
            try:
                manifest_md_file = doc_file.parent / "02_manifest_visuals" / "manifest_enriched_text.md"
                if manifest_md_file.is_file() and manifest_md_file.stat().st_size > 0:
                    raw_text = manifest_md_file.read_text(encoding="utf-8")
                    doc_file.write_text(raw_text, encoding="utf-8")
                    logger.debug(
                        f"Reset document '{doc_id}' from clean manifest stubs {manifest_md_file}"
                    )
                else:
                    raw_text = doc_file.read_text(encoding="utf-8")
                    logger.debug(f"Re-read fresh document text for '{doc_id}' from {doc_file}")
                self._last_disk_texts[doc_id] = raw_text
            except Exception as e:
                logger.warning(f"Could not read document file {doc_file}: {e}")

        # Ensure doc_id tag
        if not raw_text.startswith(f"<!-- document_id: {doc_id} -->"):
            raw_text = f"<!-- document_id: {doc_id} -->\n\n" + raw_text

        self._last_document_texts[doc_id] = raw_text
        update_extraction_log_context(doc_id=doc_id, node="prepare_input", stage="prepare_input")

        sys_inst = state.get("system_instruction", "")
        schema_summary = self._format_schema_summary()
        schema_part = f"\n\n{schema_summary}" if schema_summary else ""
        domain_tools_policy = self._format_domain_tools_policy(self._domain_tool_providers)
        domain_part = f"\n\n{domain_tools_policy}" if domain_tools_policy else ""

        sys_prompt = (
            f"{sys_inst}{schema_part}\n\n"
            "VISUAL PARSING POLICY:\n"
            f"TOOL ITERATION BUDGET: You have at most {self.max_tool_iterations} tool-calling iteration(s) "
            "for this document.\n"
            "1. Priority Cascade & Budget-Aware Batching:\n"
            "   - If required target data resides in unparsed manifest stubs (`<visual id=\"...\" ...>`), call `parse_visuals`.\n"
            "   - Complementary batching: Request MULTIPLE visual IDs in one call when they contain complementary parts of the target data (e.g., definitions/entities in one visual and measurements/attributes in another).\n"
            "   - Budget-aware batching: When there are many relevant unparsed visuals relative to your remaining iterations (e.g., dense multi-visual documents, or when <= 2 iterations remain), batch ALL clearly relevant unparsed visuals together in a single `parse_visuals` call so you are not cut off by the iteration limit.\n"
            "   - Selective staggering: Only defer lower-confidence competing candidates to a later iteration when you have ample remaining iterations (> 2 left). On your FINAL remaining iteration, batch ALL remaining needed or plausible visuals in one call.\n"
            "   - Scoped vs full parsing: You may provide `target_hint` (e.g. 'section A items', 'panel c metrics') for complex multi-part figures to selectively parse matching sub-elements. If a partial result is insufficient, you may later request specific `<element id=\"...\">[Unparsed: ...]</element>` stubs OR re-call `parse_visuals` on the root visual ID WITHOUT `target_hint` for a full parse.\n"
            "   - Failure recovery: If a visual parsing call fails, you may retry it once within your remaining budget.\n"
            "2. Manifest Plausibility Gate (No Blind Hunting & No Premature Null):\n"
            "   - Do NOT call `finalize_evidence` if target data is missing from text, but there are unparsed visuals whose manifest metadata (`> description` or caption) indicates relevant entity IDs, metrics, categories, or column names.\n"
            "   - Do NOT call `parse_visuals` blindly if missing fields are absent from text AND none of the remaining unparsed visual manifest stubs describe such data. In that case, missing fields should remain null — call `finalize_evidence`.\n"
            f"3. Output Constraint: This is a tool-routing step ONLY — do NOT mentally extract, transcribe, or draft table/record data, and do NOT write prose explanations. When no (more) tool calls are needed, call `finalize_evidence`.{domain_part}"
        ).strip()

        human_prompt = f"Document `{doc_id}`:\n\n{raw_text}"

        return {
            "doc_id": doc_id,
            "document_text": raw_text,
            "tool_iterations": 0,
            "attempted_visuals": list(state.get("attempted_visuals", [])),
            "completed_visuals": list(state.get("completed_visuals", [])),
            "partial_visuals": list(state.get("partial_visuals", [])),
            "attempted_hints": {
                k: list(v) for k, v in state.get("attempted_hints", {}).items()
            },
            "visual_attempt_counts": dict(state.get("visual_attempt_counts", {})),
            "executed_tool_signatures": list(state.get("executed_tool_signatures", [])),
            "non_visual_tools_called_since_doc_change": list(
                state.get("non_visual_tools_called_since_doc_change", [])
            ),
            "last_turn_had_non_visual": False,
            "last_turn_had_tool_error": False,
            "messages": [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=human_prompt),
            ],
        }

    def _route_after_prepare(self, state: ExtractionState) -> str:
        """Route after preparing input: enter ReAct tool loop when visuals or domain tools apply."""
        if self.max_tool_iterations <= 0:
            return "structured_output"
        doc_text = state.get("document_text", "")
        if has_unparsed_visuals(doc_text) or self._active_domain_tools(doc_text):
            return "agent"
        return "structured_output"

    def _invoke_with_retry(
        self,
        operation_name: str,
        fn: Callable[[], Any],
        on_retry: Optional[Callable[[int, Exception], None]] = None,
    ) -> Any:
        """Execute an LLM call with exponential backoff on transient/network errors."""
        total_attempts = max(1, self.max_retries + 1)
        last_exc: Optional[Exception] = None

        for attempt in range(1, total_attempts + 1):
            update_extraction_log_context(
                attempt=attempt,
                callback_invoked=False,
                last_run_id=None,
                llm_error_recorded_run_id=None,
            )
            call_start = time.perf_counter()
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                ctx = get_extraction_log_context()
                last_run_id = ctx.get("last_run_id")
                err_run_id = ctx.get("llm_error_recorded_run_id")
                callback_invoked = bool(ctx.get("callback_invoked"))
                if self.history_recorder is not None and (
                    err_run_id is None or err_run_id != last_run_id
                ):
                    is_post_validation = isinstance(exc, StructuredOutputSanityError) or callback_invoked
                    self._record_history_event(
                        status="error",
                        stage="post_validation_error" if is_post_validation else None,
                        error=exc,
                        latency_s=0.0 if is_post_validation else round(time.perf_counter() - call_start, 4),
                        attempt=attempt,
                        llm_run_id=last_run_id,
                    )
                if attempt >= total_attempts or not _is_retryable_error(exc):
                    raise

                if on_retry is not None:
                    on_retry(attempt, exc)

                if self.retry_delay > 0:
                    base_delay = self.retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0.25, 1.25)
                    delay = min(60.0, base_delay + jitter)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt}/{total_attempts}): {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt}/{total_attempts}): {exc}. Retrying..."
                    )

        assert last_exc is not None
        raise last_exc

    def _apply_retry_temperature(
        self,
        runnable: Any,
        is_retry: bool,
        base_llm: Optional[Any] = None,
    ) -> Any:
        """Raise temperature to 0.7 only on retry/repair for real chat models that support it."""
        target_llm = base_llm if base_llm is not None else self.llm
        model_id = getattr(target_llm, "model", None) or getattr(target_llm, "model_name", None)
        if (
            is_retry
            and isinstance(target_llm, BaseChatModel)
            and not isinstance(target_llm, NonCallableMock)
            and model_supports_sampling_params(model_id)
            and hasattr(runnable, "bind")
        ):
            try:
                return runnable.bind(temperature=0.7)
            except Exception:
                return runnable
        return runnable

    def _deduplicate_output(self, result: Any) -> Any:
        """Apply canonical record deduplication across all structured_output_node modes."""
        if not self.enable_record_dedup or result is None:
            return result
        records = getattr(result, "records", None)
        if not isinstance(records, list) or len(records) <= 1:
            return result

        deduped: list[Any] = []
        seen_keys: set[str] = set()
        for rec in records:
            key = _record_dedup_key(rec)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(rec)

        if len(deduped) == len(records):
            return result

        logger.info(
            f"Canonical post-deduplication removed {len(records) - len(deduped)} "
            f"duplicate record(s) ({len(records)} -> {len(deduped)})"
        )
        if isinstance(result, BaseModel):
            return result.model_copy(update={"records": deduped})
        return self.output_model(records=deduped)

    def _agent_node(self, state: ExtractionState) -> dict[str, Any]:
        """Invoke LLM with statically bound visual, domain, and terminal tools."""
        update_extraction_log_context(
            doc_id=state["doc_id"],
            node="agent_node",
            stage="agent_node",
            batch_idx=None,
        )
        doc_text = state.get("document_text", "")
        service = self._resolve_service()
        tools_to_bind: list[Any] = [make_visual_parsing_tool(state["doc_id"], service)]
        for provider in self._domain_tool_providers:
            tools_to_bind.append(
                self._build_domain_tool_instance(
                    provider,
                    doc_id=state["doc_id"],
                    doc_text=doc_text,
                )
            )
        tools_to_bind.append(_make_finalize_evidence_tool())

        target_agent_llm = self.agent_llm if self.agent_llm is not None else self.llm
        bind_kwargs: dict[str, Any] = {}
        is_real_chat_model = (
            isinstance(target_agent_llm, BaseChatModel)
            and not isinstance(target_agent_llm, NonCallableMock)
        )
        if (
            self.agent_thinking_level is not None
            and is_real_chat_model
            and hasattr(target_agent_llm, "thinking_level")
        ):
            bind_kwargs["thinking_level"] = self.agent_thinking_level
        if is_real_chat_model:
            try:
                sig = inspect.signature(target_agent_llm.bind_tools)
                if "tool_choice" in sig.parameters:
                    llm_type = str(getattr(target_agent_llm, "_llm_type", "") or "").lower()
                    bind_kwargs["tool_choice"] = "required" if "openai" in llm_type else "any"
            except (ValueError, TypeError):
                pass
        model_with_tools = target_agent_llm.bind_tools(tools_to_bind, **bind_kwargs)

        is_retry = False

        def _call_agent() -> Any:
            target = self._apply_retry_temperature(model_with_tools, is_retry, base_llm=target_agent_llm)
            return target.invoke(state["messages"])

        def _on_agent_retry(_attempt: int, _exc: Exception) -> None:
            nonlocal is_retry
            is_retry = True

        call_start = time.perf_counter()
        response = self._invoke_with_retry(
            f"agent_node[{state['doc_id']}]",
            _call_agent,
            on_retry=_on_agent_retry,
        )
        if self.history_recorder is not None and not get_extraction_log_context().get("callback_invoked"):
            self._record_history_event(
                status="success",
                node="agent_node",
                stage="agent_node",
                doc_id=state["doc_id"],
                messages=state["messages"],
                outputs=[getattr(response, "content", response)],
                tool_calls=getattr(response, "tool_calls", None),
                latency_s=round(time.perf_counter() - call_start, 4),
            )
        return {
            "messages": [response],
        }

    def _is_visual_call_eligible(
        self,
        vid: str,
        norm_hint: Optional[str],
        attempted_visuals: set[str],
        completed_visuals: set[str],
        partial_visuals: set[str],
        attempted_hints: dict[str, set[str]],
        attempt_counts: dict[str, int],
    ) -> bool:
        """Return True if requesting (vid, norm_hint) is allowed and not a redundant loop."""
        cleaned = str(vid).strip()
        if not cleaned:
            return False

        root_id, is_sub = _split_visual_id(cleaned)

        if is_sub:
            if cleaned in completed_visuals or cleaned in attempted_visuals:
                return False
            if root_id in completed_visuals:
                return False
            return attempt_counts.get(cleaned, 0) < self.max_visual_attempts

        if cleaned in completed_visuals:
            return False
        if cleaned in attempted_visuals and cleaned not in partial_visuals:
            return False
        if attempt_counts.get(cleaned, 0) >= self.max_visual_attempts:
            return False

        if cleaned in partial_visuals:
            if norm_hint is not None:
                if norm_hint in attempted_hints.get(cleaned, set()):
                    return False
                return attempt_counts.get(f"{cleaned}::hint::{norm_hint}", 0) < self.max_visual_attempts
            return attempt_counts.get(f"{cleaned}::__full__", 0) < self.max_visual_attempts

        if norm_hint is not None:
            if norm_hint in attempted_hints.get(cleaned, set()):
                return attempt_counts.get(f"{cleaned}::hint::{norm_hint}", 0) < self.max_visual_attempts
        return True

    def _eligible_unparsed_visuals(
        self,
        doc_text: str,
        attempted_visuals: set[str],
        completed_visuals: set[str],
        partial_visuals: set[str],
        attempted_hints: dict[str, set[str]],
        attempt_counts: dict[str, int],
    ) -> list[str]:
        """Return unparsed or partial visual/element IDs still eligible for tool calls."""
        eligible: list[str] = []
        seen: set[str] = set()
        for v in extract_unparsed_visual_ids(doc_text):
            if v not in seen and self._is_visual_call_eligible(
                v, None, attempted_visuals, completed_visuals, partial_visuals, attempted_hints, attempt_counts
            ):
                eligible.append(v)
                seen.add(v)
        for pv in partial_visuals:
            if pv not in seen and self._is_visual_call_eligible(
                pv, None, attempted_visuals, completed_visuals, partial_visuals, attempted_hints, attempt_counts
            ):
                eligible.append(pv)
                seen.add(pv)
        return eligible

    @staticmethod
    def _tracking_sets_from_state(
        state: ExtractionState,
    ) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, int], set[str]]:
        """Extract normalized tracking collections from ExtractionState."""
        attempted = set(state.get("attempted_visuals", []))
        completed = set(state.get("completed_visuals", []))
        partial = set(state.get("partial_visuals", []))
        hints = {k: set(v) for k, v in state.get("attempted_hints", {}).items()}
        counts = dict(state.get("visual_attempt_counts", {}))
        return attempted, completed, partial, hints, counts

    def _should_continue(self, state: ExtractionState) -> str:
        """Route to tool execution or finalize structured output."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)

        if not tool_calls:
            return "structured_output"

        if state.get("tool_iterations", 0) >= self.max_tool_iterations:
            return "structured_output"

        attempted, completed, partial, hints, counts = self._tracking_sets_from_state(state)
        executed_sigs = set(state.get("executed_tool_signatures", []))
        has_new_calls = False
        for tc in tool_calls:
            tc_name = tc.get("name")
            if tc_name == "finalize_evidence":
                continue
            args = tc.get("args", {}) or {}
            if tc_name == "parse_visuals":
                v_ids = args.get("visual_ids", []) or []
                norm_hint = _normalize_hint(args.get("target_hint"))
                if any(
                    self._is_visual_call_eligible(
                        v, norm_hint, attempted, completed, partial, hints, counts
                    )
                    for v in v_ids
                ):
                    has_new_calls = True
                    break
            else:
                sig = self._tool_call_signature(str(tc_name or ""), args)
                if sig not in executed_sigs:
                    has_new_calls = True
                    break

        if has_new_calls:
            return "tools"

        if any(tc.get("name") == "finalize_evidence" for tc in tool_calls):
            return "structured_output"

        if not state.get("domain_reminder_sent", False):
            doc_text = state.get("document_text", "")
            active_names = {
                p.name for p in self._active_domain_tools(doc_text) if getattr(p, "auto_remind", True)
            }
            called_names = set(state.get("non_visual_tools_called_since_doc_change", []))
            if active_names - called_names:
                return "tools"

        return "structured_output"

    def _tool_node(self, state: ExtractionState) -> dict[str, Any]:
        """Execute tool calls, update visual state tracking, and refresh document context."""
        update_extraction_log_context(
            doc_id=state["doc_id"],
            node="tool_node",
            stage="tool_node",
            batch_idx=None,
        )
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])

        doc_id = state["doc_id"]
        prev_doc_text = state.get("document_text", "")
        prev_active_providers = self._active_domain_tools(prev_doc_text)
        prev_active_names = {p.name for p in prev_active_providers}

        domain_tools_by_name: dict[str, Any] = {}
        for provider in self._domain_tool_providers:
            domain_tools_by_name[provider.name] = self._build_domain_tool_instance(
                provider, doc_id=doc_id, doc_text=prev_doc_text
            )

        tool_messages: list[BaseMessage] = []
        attempted = list(state.get("attempted_visuals", []))
        completed = list(state.get("completed_visuals", []))
        partial = list(state.get("partial_visuals", []))
        hints_map: dict[str, list[str]] = {
            k: list(v) for k, v in state.get("attempted_hints", {}).items()
        }
        attempt_counts: dict[str, int] = dict(state.get("visual_attempt_counts", {}))
        executed_sigs = list(state.get("executed_tool_signatures", []))
        executed_sigs_set = set(executed_sigs)
        non_visual_called = list(state.get("non_visual_tools_called_since_doc_change", []))
        new_tool_results: list[dict[str, Any]] = []
        ran_non_visual = False
        ran_non_visual_error = False

        executed_batches: list[tuple[list[str], Optional[str], str]] = []

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {}) or {}
            call_id = tc.get("id", "")

            if tool_name == "parse_visuals":
                service = self._resolve_service()
                tool = make_visual_parsing_tool(doc_id, service)
                raw_ids = [str(v).strip() for v in (tool_args.get("visual_ids", []) or []) if str(v).strip()]
                raw_hint = tool_args.get("target_hint")
                norm_hint = _normalize_hint(raw_hint)

                attempted_set = set(attempted)
                completed_set = set(completed)
                partial_set = set(partial)
                hints_set_map = {k: set(v) for k, v in hints_map.items()}

                eligible_ids: list[str] = []
                skipped_ids: list[str] = []
                for vid in raw_ids:
                    if vid not in eligible_ids and self._is_visual_call_eligible(
                        vid, norm_hint, attempted_set, completed_set, partial_set, hints_set_map, attempt_counts
                    ):
                        eligible_ids.append(vid)
                    elif vid not in eligible_ids:
                        skipped_ids.append(vid)

                if not eligible_ids:
                    if raw_ids:
                        res = (
                            f"Visuals {raw_ids} have already been fully parsed, attempted with the same hint, "
                            f"or reached the retry limit."
                        )
                    else:
                        res = "No visual_ids provided."
                else:
                    invoke_kwargs: dict[str, Any] = {"visual_ids": eligible_ids}
                    if norm_hint is not None and isinstance(raw_hint, str):
                        invoke_kwargs["target_hint"] = raw_hint.strip()
                    res = str(tool.invoke(invoke_kwargs))
                    executed_batches.append((eligible_ids, norm_hint, res))
                    if skipped_ids:
                        res = f"[Skipped already-completed/exhausted visuals: {skipped_ids}]\n\n{res}"
            elif tool_name == "finalize_evidence":
                res = "Evidence gathering finalized."
            elif tool_name in domain_tools_by_name:
                sig = self._tool_call_signature(tool_name, tool_args)
                if sig in executed_sigs_set:
                    res = f"Tool `{tool_name}` with identical arguments was already executed."
                else:
                    try:
                        res = str(domain_tools_by_name[tool_name].invoke(tool_args))
                        if _tool_output_has_error(res):
                            ran_non_visual_error = True
                    except Exception as exc:
                        logger.warning(f"Domain tool '{tool_name}' failed for doc '{doc_id}': {exc}")
                        res = json.dumps({"error": f"Tool execution failed: {exc}"})
                        ran_non_visual_error = True
                    executed_sigs.append(sig)
                    executed_sigs_set.add(sig)
                    if tool_name not in non_visual_called:
                        non_visual_called.append(tool_name)
                    new_tool_results.append(
                        {"tool": tool_name, "args": tool_args, "output": res}
                    )
                    ran_non_visual = True
            else:
                res = f"Tool {tool_name} not recognized."
                sig = self._tool_call_signature(str(tool_name), tool_args)
                if sig not in executed_sigs_set:
                    executed_sigs.append(sig)
                    executed_sigs_set.add(sig)

            tool_messages.append(ToolMessage(content=str(res), tool_call_id=call_id))

        # Re-read freshly enriched document from disk only if disk actually changed
        doc_file = self.document_repo._find_document_path(doc_id, self.ingestion_dir)
        enriched_content = ""
        if doc_file and doc_file.exists():
            try:
                enriched_content = doc_file.read_text(encoding="utf-8")
            except Exception:
                pass

        prev_disk = self._last_disk_texts.get(doc_id)
        if enriched_content and enriched_content != prev_disk:
            updated_doc_text = enriched_content
            self._last_disk_texts[doc_id] = enriched_content
        else:
            updated_doc_text = state["document_text"]
            if tool_messages:
                updated_doc_text = _merge_tool_visuals_into_document(
                    updated_doc_text, tool_messages
                )

        # Classify each executed visual outcome after merging so partial/failed visuals remain retryable
        for eligible_ids, norm_hint, batch_res in executed_batches:
            for vid in eligible_ids:
                root_id, is_sub = _split_visual_id(vid)
                outcome, root_resolved = _classify_visual_outcome(vid, batch_res, updated_doc_text)

                if outcome == "completed":
                    if vid not in completed:
                        completed.append(vid)
                    if vid not in attempted:
                        attempted.append(vid)
                    if root_resolved:
                        if root_id not in completed:
                            completed.append(root_id)
                        if root_id not in attempted:
                            attempted.append(root_id)
                        if root_id in partial:
                            partial.remove(root_id)
                elif outcome == "partial":
                    if root_id not in partial:
                        partial.append(root_id)
                    if norm_hint is not None:
                        hint_list = hints_map.setdefault(root_id, [])
                        if norm_hint not in hint_list:
                            hint_list.append(norm_hint)
                    else:
                        full_key = f"{root_id}::__full__"
                        attempt_counts[full_key] = attempt_counts.get(full_key, 0) + 1
                else:
                    # Failed attempt: increment retry counter; exhaust only at max_visual_attempts
                    if norm_hint is not None and not is_sub:
                        hint_list = hints_map.setdefault(root_id, [])
                        if norm_hint not in hint_list:
                            hint_list.append(norm_hint)
                        hint_key = f"{root_id}::hint::{norm_hint}"
                        attempt_counts[hint_key] = attempt_counts.get(hint_key, 0) + 1
                    else:
                        new_count = attempt_counts.get(vid, 0) + 1
                        attempt_counts[vid] = new_count
                        if new_count >= self.max_visual_attempts and vid not in attempted:
                            attempted.append(vid)
                        if not is_sub and norm_hint is None:
                            full_key = f"{root_id}::__full__"
                            attempt_counts[full_key] = attempt_counts.get(full_key, 0) + 1

        self._last_document_texts[doc_id] = updated_doc_text

        newly_active_hint = ""
        sent_reminder = bool(state.get("domain_reminder_sent", False))
        if updated_doc_text != prev_doc_text:
            non_visual_called = []
            sent_reminder = False
            curr_active = self._active_domain_tools(updated_doc_text)
            newly_active = [p for p in curr_active if p.name not in prev_active_names]
            if newly_active:
                policy_txt = self._format_domain_tools_policy(newly_active)
                if policy_txt:
                    newly_active_hint = f"\n\n{policy_txt}"
        elif not ran_non_visual and not executed_batches and not sent_reminder:
            curr_active = [
                p for p in self._active_domain_tools(updated_doc_text) if getattr(p, "auto_remind", True)
            ]
            pending_providers = [p for p in curr_active if p.name not in set(non_visual_called)]
            if pending_providers:
                pending_names = ", ".join(p.name for p in pending_providers)
                policy_txt = self._format_domain_tools_policy(pending_providers)
                newly_active_hint = (
                    f"\n\n[ACTION REQUIRED: Visual parsing is complete, but active domain tool(s) "
                    f"({pending_names}) have not been called yet. Call `{pending_names}` before finalizing.]"
                )
                if policy_txt:
                    newly_active_hint += f"\n{policy_txt}"
                sent_reminder = True

        new_iterations = state.get("tool_iterations", 0) + 1
        remaining = max(0, self.max_tool_iterations - new_iterations)
        eligible_remaining = self._eligible_unparsed_visuals(
            updated_doc_text,
            set(attempted),
            set(completed),
            set(partial),
            {k: set(v) for k, v in hints_map.items()},
            attempt_counts,
        )

        if tool_messages:
            if remaining == 1:
                budget_note = (
                    f"\n\n[Budget status: {new_iterations}/{self.max_tool_iterations} tool iteration(s) used, "
                    f"1 FINAL iteration remaining; {len(eligible_remaining)} eligible unparsed/partial visual(s). "
                    f"Batch all remaining needed visuals in your next call or reply READY.]"
                )
            else:
                budget_note = (
                    f"\n\n[Budget status: {new_iterations}/{self.max_tool_iterations} tool iteration(s) used, "
                    f"{remaining} remaining; {len(eligible_remaining)} eligible unparsed/partial visual(s).]"
                )
            last_tool_msg = tool_messages[-1]
            tool_messages[-1] = ToolMessage(
                content=f"{last_tool_msg.content}{newly_active_hint}{budget_note}",
                tool_call_id=last_tool_msg.tool_call_id,
            )

        out_state: dict[str, Any] = {
            "messages": tool_messages,
            "document_text": updated_doc_text,
            "tool_iterations": new_iterations,
            "attempted_visuals": attempted,
            "completed_visuals": completed,
            "partial_visuals": partial,
            "attempted_hints": hints_map,
            "visual_attempt_counts": attempt_counts,
            "executed_tool_signatures": executed_sigs,
            "non_visual_tools_called_since_doc_change": non_visual_called,
            "last_turn_had_non_visual": ran_non_visual,
            "last_turn_had_tool_error": ran_non_visual_error,
            "domain_reminder_sent": sent_reminder,
        }
        if new_tool_results:
            out_state["tool_results"] = new_tool_results
        return out_state

    def _route_after_tool(self, state: ExtractionState) -> str:
        """Route after tool execution: to agent_node or structured_output_node."""
        if state.get("tool_iterations", 0) >= self.max_tool_iterations:
            return "structured_output"
        doc_text = state.get("document_text", "")
        attempted, completed, partial, hints, counts = self._tracking_sets_from_state(state)
        eligible = self._eligible_unparsed_visuals(
            doc_text,
            attempted,
            completed,
            partial,
            hints,
            counts,
        )
        if eligible:
            return "agent"
        active_names = {
            p.name for p in self._active_domain_tools(doc_text) if getattr(p, "auto_remind", True)
        }
        called_names = set(state.get("non_visual_tools_called_since_doc_change", []))
        if active_names - called_names:
            return "agent"
        if state.get("last_turn_had_tool_error", False):
            return "agent"
        return "structured_output"

    def _build_structured_messages(
        self,
        state: ExtractionState,
        user_instruction: str,
    ) -> list[BaseMessage]:
        """Build a clean single-turn prompt from the enriched document_text and tool results."""
        doc_text = state.get("document_text", "")
        sys_inst = state.get("system_instruction", "").strip()
        schema_summary = self._format_schema_summary()
        structured_hints: list[str] = []
        seen_hints: set[str] = set()
        for provider in self._domain_tool_providers:
            hint = str(getattr(provider, "structured_output_hint", "") or "").strip()
            if hint and hint not in seen_hints:
                seen_hints.add(hint)
                structured_hints.append(hint)
        parts = [p for p in (sys_inst, schema_summary, *structured_hints) if p]
        full_sys_inst = "\n\n".join(parts)

        tool_results_section = self._format_tool_results_section(state)
        tool_results_part = f"\n\n{tool_results_section}" if tool_results_section else ""

        if doc_text and doc_text.strip():
            messages: list[BaseMessage] = []
            if full_sys_inst:
                messages.append(SystemMessage(content=full_sys_inst))
            doc_id = state.get("doc_id") or "unknown_document"
            messages.append(
                HumanMessage(
                    content=f"Document `{doc_id}`:\n\n{doc_text}{tool_results_part}"
                )
            )
            messages.append(HumanMessage(content=user_instruction))
            return messages

        # Fallback for unit tests providing only messages without document_text
        structured_messages: list[BaseMessage] = list(state.get("messages", []))
        if structured_messages and isinstance(structured_messages[0], SystemMessage):
            structured_messages[0] = SystemMessage(content=full_sys_inst)
        while (
            structured_messages
            and isinstance(structured_messages[-1], AIMessage)
            and not getattr(structured_messages[-1], "tool_calls", None)
        ):
            structured_messages.pop()
        if not structured_messages or not isinstance(structured_messages[-1], HumanMessage):
            structured_messages.append(HumanMessage(content=user_instruction))
        return structured_messages

    def _call_structured_model(
        self,
        structured_model: Any,
        messages: list[BaseMessage],
    ) -> Any:
        """Invoke structured_model using streaming keep-alive for real chat models."""
        can_stream = (
            self.use_streaming
            and isinstance(self.llm, BaseChatModel)
            and not isinstance(self.llm, NonCallableMock)
            and isinstance(structured_model, Runnable)
            and not isinstance(structured_model, NonCallableMock)
        )
        if can_stream:
            try:
                last_parsed: Any = None
                for chunk in structured_model.stream(messages):
                    if chunk is not None:
                        last_parsed = chunk
                if last_parsed is not None:
                    return last_parsed
            except NotImplementedError:
                pass
        return structured_model.invoke(messages)

    def _validate_structured_result(self, result: Any, state: ExtractionState) -> Any:
        """Run domain-independent sanity checks and declarative field validators on parsed output."""
        if not self.enable_sanity_checks:
            if self.enable_field_validators:
                return validate_output_field_validators(
                    result,
                    output_model=self.output_model,
                )
            return result
        grounding_doc_text = state.get("document_text", "")
        tool_results_section = self._format_tool_results_section(state)
        if tool_results_section:
            grounding_doc_text = f"{grounding_doc_text}\n\n{tool_results_section}"
        return validate_structured_output_sanity(
            result,
            document_text=grounding_doc_text,
            system_instruction=state.get("system_instruction", ""),
            output_model=self.output_model,
            config=self._sanity_config,
        )

    @staticmethod
    def _format_sanity_retry_feedback(exc: Exception) -> str:
        """Build a concise, domain-independent corrective prompt suffix after a sanity failure."""
        return (
            f"\n\n[CORRECTIVE INSTRUCTION: Your previous structured output failed a sanity check ({exc}). "
            "Re-extract carefully from the document, ensuring all field values strictly match the schema "
            "constraints and are grounded in the document text and language without extraneous or repeated "
            "hallucinated tokens.]"
        )

    def _attempt_local_record_repair(
        self,
        state: ExtractionState,
        exc: StructuredOutputSanityError,
    ) -> Optional[Any]:
        """Attempt targeted repair of a small number of failing records without full regeneration."""
        partial = exc.partial_output
        if partial is None:
            return None
        record_model = _extract_record_model(self.output_model)
        if record_model is None:
            return None
        records = list(getattr(partial, "records", None) or [])
        if not records:
            return None

        failed = [idx for idx in exc.failed_record_indices if 0 <= idx < len(records)]
        threshold = min(15, max(1, len(records) // 3))
        if not failed or len(records) <= len(failed) or len(failed) > threshold:
            return None

        failed_items: list[str] = []
        for idx in failed:
            rec = records[idx]
            if isinstance(rec, BaseModel):
                dumped = rec.model_dump(mode="json")
            elif isinstance(rec, dict):
                dumped = rec
            else:
                dumped = str(rec)
            failed_items.append(
                f"- index {idx}: {json.dumps(dumped, ensure_ascii=False, separators=(',', ':'))}"
            )
        reasons_text = "\n".join(f"- {r}" for r in exc.reasons) if exc.reasons else f"- {exc}"
        repair_instruction = (
            "The following extracted record(s) failed validation:\n"
            f"{chr(10).join(failed_items)}\n\n"
            "Validation error(s):\n"
            f"{reasons_text}\n\n"
            f"Repair ONLY these {len(failed)} record(s) at their exact 0-based index using the document "
            "and tool results, preserving all other fields accurately. Return them in `repairs`."
        )

        patched_item_model = create_model(
            f"{record_model.__name__}PatchItem",
            index=(int, Field(..., description="0-based index of the failed record to repair")),
            record=(record_model, Field(..., description="Repaired record matching the schema")),
        )
        patched_output_model = create_model(
            f"{self.output_model.__name__}PatchOutput",
            repairs=(list[patched_item_model], Field(..., description="List of repaired records")),  # type: ignore[valid-type]
        )

        repair_messages = self._build_structured_messages(state, repair_instruction)
        target_repair_llm = self.repair_llm if self.repair_llm is not None else self.llm
        repair_model = self._apply_retry_temperature(
            target_repair_llm.with_structured_output(patched_output_model),
            True,
            base_llm=target_repair_llm,
        )
        repair_res = self._call_structured_model(repair_model, repair_messages)

        patched_records = list(records)
        failed_set = set(failed)
        applied = 0
        for item in getattr(repair_res, "repairs", None) or []:
            idx = getattr(item, "index", None)
            rec = getattr(item, "record", None)
            if idx in failed_set and rec is not None:
                patched_records[idx] = rec
                applied += 1
        if applied == 0:
            return None

        if isinstance(partial, BaseModel):
            candidate = partial.model_copy(update={"records": patched_records})
        else:
            candidate = self.output_model(records=patched_records)
        return self._validate_structured_result(candidate, state)

    def _validate_or_repair_structured_result(
        self,
        raw_res: Any,
        state: ExtractionState,
        max_local_repairs: int = 1,
    ) -> Any:
        """Validate structured output, attempting local record repair before full retry."""
        try:
            return self._validate_structured_result(raw_res, state)
        except StructuredOutputSanityError as exc:
            current_exc: StructuredOutputSanityError = exc
            for _ in range(max_local_repairs):
                try:
                    repaired = self._attempt_local_record_repair(state, current_exc)
                    if repaired is not None:
                        return repaired
                    break
                except StructuredOutputSanityError as repair_exc:
                    current_exc = repair_exc
                except Exception as repair_err:
                    logger.warning(
                        f"Local record repair failed for doc '{state.get('doc_id', '')}': {repair_err}"
                    )
                    break
            raise exc

    def _extract_records_batched(
        self,
        state: ExtractionState,
        structured_model: Any,
    ) -> Any:
        """Extract records iteratively in bounded batches for dense documents."""
        doc_id = state.get("doc_id", "unknown_document")
        current_batch_size = self.structured_output_batch_size
        accumulated: list[Any] = []
        seen_keys: set[str] = set()

        for batch_idx in range(1, self.max_structured_output_batches + 1):
            batch_stage = f"structured_output_batch_{batch_idx}"
            update_extraction_log_context(
                doc_id=doc_id,
                node="structured_output_node",
                stage=batch_stage,
                batch_idx=batch_idx,
            )
            last_batch_messages: list[BaseMessage] = []
            retry_feedback: Optional[str] = None
            is_batch_retry = False

            def _run_batch() -> Any:
                nonlocal last_batch_messages
                if batch_idx == 1:
                    instruction = (
                        "Extract the structured records according to the schema based on the document "
                        "and extracted visual objects. "
                        f"If the document contains many records, return up to {current_batch_size} "
                        "complete records in document order (starting from the beginning) so the "
                        "response stays concise and complete. Do not invent or truncate records."
                    )
                else:
                    already_lines = []
                    for rec in accumulated:
                        if isinstance(rec, BaseModel):
                            dumped = rec.model_dump(mode="json", exclude_none=True)
                        elif isinstance(rec, dict):
                            dumped = {k: v for k, v in rec.items() if v is not None}
                        else:
                            dumped = rec
                        already_lines.append(
                            json.dumps(dumped, ensure_ascii=False, separators=(",", ":"))
                        )
                    already_block = "\n".join(already_lines)
                    instruction = (
                        f"The following {len(accumulated)} records have ALREADY been extracted "
                        f"from the document (do NOT repeat them):\n{already_block}\n\n"
                        f"Extract the NEXT up to {current_batch_size} remaining unique records from "
                        "the document in document order that are NOT listed above. "
                        "If all records in the document have already been extracted, return an empty "
                        "`records` list."
                    )
                if retry_feedback:
                    instruction = f"{instruction}{retry_feedback}"
                last_batch_messages = self._build_structured_messages(state, instruction)
                target_model = self._apply_retry_temperature(structured_model, is_batch_retry)
                raw_res = self._call_structured_model(target_model, last_batch_messages)
                return self._validate_or_repair_structured_result(raw_res, state)

            def _on_batch_retry(attempt: int, exc: Exception) -> None:
                nonlocal current_batch_size, retry_feedback, is_batch_retry
                is_batch_retry = True
                if isinstance(exc, StructuredOutputSanityError):
                    retry_feedback = self._format_sanity_retry_feedback(exc)
                if _is_parse_or_truncation_error(exc) and current_batch_size > 5:
                    new_size = max(5, current_batch_size // 2)
                    logger.warning(
                        f"Parse/truncation error on doc '{doc_id}' batch {batch_idx} "
                        f"(attempt {attempt}): reducing batch size {current_batch_size} -> {new_size}"
                    )
                    current_batch_size = new_size

            batch_start = time.perf_counter()
            try:
                batch_res = self._invoke_with_retry(
                    f"structured_output[{doc_id}][batch={batch_idx}]",
                    _run_batch,
                    on_retry=_on_batch_retry,
                )
            except CircuitBreakerError:
                raise
            except Exception as exc:
                if batch_idx > 1 and accumulated:
                    logger.warning(
                        f"Continuation batch {batch_idx} failed for doc '{doc_id}' ({exc}); "
                        f"preserving {len(accumulated)} already-extracted records."
                    )
                    break
                raise

            if self.history_recorder is not None and not get_extraction_log_context().get("callback_invoked"):
                self._record_history_event(
                    status="success",
                    node="structured_output_node",
                    stage=batch_stage,
                    doc_id=doc_id,
                    messages=last_batch_messages,
                    outputs=[batch_res],
                    batch_idx=batch_idx,
                    latency_s=round(time.perf_counter() - batch_start, 4),
                )

            batch_records = list(getattr(batch_res, "records", []) or [])
            new_records: list[Any] = []
            for rec in batch_records:
                key = _record_dedup_key(rec)
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_records.append(rec)

            accumulated.extend(new_records)

            if not batch_records:
                logger.info(
                    f"Batched extraction for doc '{doc_id}' completed in {batch_idx} batch(es) "
                    f"(empty batch, total={len(accumulated)} records)."
                )
                break

            if batch_idx > 1 and not new_records:
                logger.info(
                    f"Batched extraction for doc '{doc_id}' converged at batch {batch_idx} "
                    f"(0 new records, total={len(accumulated)})."
                )
                break

            if batch_idx == self.max_structured_output_batches:
                logger.warning(
                    f"Reached max_structured_output_batches={self.max_structured_output_batches} "
                    f"for doc '{doc_id}' (total={len(accumulated)} records)."
                )

        return self.output_model(records=accumulated)

    def _structured_output_node(self, state: ExtractionState) -> dict[str, Any]:
        """Produce final structured output conforming to output_model."""
        update_extraction_log_context(
            doc_id=state["doc_id"],
            node="structured_output_node",
            stage="structured_output",
            batch_idx=None,
        )
        logger.info(f"Producing structured output for doc '{state['doc_id']}'...")
        # Fast path: if last message already contains valid JSON, avoid 2nd LLM call
        if state["messages"]:
            last_content = getattr(state["messages"][-1], "content", "")
            if isinstance(last_content, str) and last_content.strip().startswith("{"):
                try:
                    parsed = self.output_model.model_validate_json(last_content.strip())
                    parsed = self._validate_structured_result(parsed, state)
                    parsed = self._deduplicate_output(parsed)
                    return {
                        "extracted_data": parsed,
                        "rationale": "",
                    }
                except Exception:
                    pass

        try:
            structured_model = self.llm.with_structured_output(self.output_model)

            if self.structured_output_batch_size > 0 and _supports_record_batching(
                self.output_model
            ):
                res = self._extract_records_batched(state, structured_model)
            else:
                base_instruction = (
                    "Extract the structured records according to the schema based on the document "
                    "and extracted visual objects."
                )
                retry_feedback: Optional[str] = None
                last_messages: list[BaseMessage] = []
                is_single_retry = False

                def _run_single() -> Any:
                    nonlocal last_messages
                    inst = f"{base_instruction}{retry_feedback}" if retry_feedback else base_instruction
                    last_messages = self._build_structured_messages(state, inst)
                    target_model = self._apply_retry_temperature(structured_model, is_single_retry)
                    raw_res = self._call_structured_model(target_model, last_messages)
                    return self._validate_or_repair_structured_result(raw_res, state)

                def _on_single_retry(_attempt: int, exc: Exception) -> None:
                    nonlocal retry_feedback, is_single_retry
                    is_single_retry = True
                    if isinstance(exc, StructuredOutputSanityError):
                        retry_feedback = self._format_sanity_retry_feedback(exc)

                call_start = time.perf_counter()
                res = self._invoke_with_retry(
                    f"structured_output[{state['doc_id']}]",
                    _run_single,
                    on_retry=_on_single_retry,
                )
                if self.history_recorder is not None and not get_extraction_log_context().get("callback_invoked"):
                    self._record_history_event(
                        status="success",
                        node="structured_output_node",
                        stage="structured_output",
                        doc_id=state["doc_id"],
                        messages=last_messages,
                        outputs=[res],
                        latency_s=round(time.perf_counter() - call_start, 4),
                    )

            res = self._deduplicate_output(res)
            return {
                "extracted_data": res,
                "rationale": "",
            }
        except Exception as exc:
            logger.error(f"Failed to produce structured output: {exc}")
            raise

    def run(
        self,
        document_text: str,
        doc_id: Optional[str] = None,
        system_instruction: str = "",
    ) -> ExtractionState:
        """Run extraction workflow end-to-end."""
        resolved_doc_id = doc_id or ""
        if not resolved_doc_id and document_text:
            m = _DOC_ID_RE.search(document_text)
            if m:
                resolved_doc_id = m.group(1).strip()

        initial_state: ExtractionState = {
            "doc_id": doc_id or "",
            "document_text": document_text,
            "system_instruction": system_instruction,
            "messages": [],
            "tool_iterations": 0,
            "attempted_visuals": [],
            "completed_visuals": [],
            "partial_visuals": [],
            "attempted_hints": {},
            "visual_attempt_counts": {},
            "tool_results": [],
            "executed_tool_signatures": [],
            "non_visual_tools_called_since_doc_change": [],
            "last_turn_had_non_visual": False,
            "last_turn_had_tool_error": False,
            "extracted_data": None,
            "rationale": None,
        }
        with extraction_log_context(
            doc_id=resolved_doc_id or None,
            source="extraction_graph",
        ):
            try:
                if self._callback_handler is not None:
                    final_state: ExtractionState = self.app.invoke(
                        initial_state,
                        config={"callbacks": [self._callback_handler]},
                    )
                else:
                    final_state = self.app.invoke(initial_state)
                return final_state
            except Exception as exc:
                if self.history_recorder is not None:
                    ctx = get_extraction_log_context()
                    self._record_history_event(
                        status="error",
                        node="graph",
                        stage="graph_failure",
                        doc_id=ctx.get("doc_id") or resolved_doc_id or None,
                        error=exc,
                        llm_run_id=ctx.get("last_run_id"),
                    )
                raise
