"""Domain-independent sanity checks for LLM structured extraction outputs."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from ae.core.schema.dynamic_models import (
    extract_field_validators,
    extract_schema_field_constraints,
)
from ae.core.schema.validators import validate_field_value

logger = logging.getLogger(__name__)

# Scripts universally permitted in scientific/technical extraction regardless of document language.
_ALWAYS_ALLOWED_SCRIPTS = frozenset({"LATIN", "GREEK"})

_XML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_XML_TAG_RE = re.compile(r"<[^>]+>")
_BLOCKQUOTE_RE = re.compile(r"(?m)^>\s*")
_REACTION_ARROW_RE = re.compile(r"<-->|-->|->|<-")


class StructuredOutputSanityError(OutputParserException):
    """Raised when structured output passes JSON syntax but fails decoding sanity checks."""

    def __init__(
        self,
        message: str,
        *,
        reasons: Optional[Sequence[str]] = None,
        field_names: Optional[Sequence[str]] = None,
        failed_record_indices: Optional[Sequence[int]] = None,
        partial_output: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.reasons = list(reasons or [])
        self.field_names = list(field_names or [])
        self.failed_record_indices: list[int] = sorted(
            set(int(i) for i in (failed_record_indices or []))
        )
        self.partial_output = partial_output


@dataclass(frozen=True)
class SanityCheckConfig:
    """Configuration for domain-independent structured output sanity validation."""

    enabled: bool = True
    check_control_chars: bool = True
    check_ungrounded_scripts: bool = True
    check_sticky_ungrounded_symbols: bool = True
    check_field_validators: bool = True
    min_records_for_collapse: int = 4
    collapse_ratio_threshold: float = 0.85


def _letter_script(ch: str) -> Optional[str]:
    """Return the Unicode script family for a letter character, or None for non-letters."""
    if not unicodedata.category(ch).startswith("L"):
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return None
    return name.split(" ", 1)[0]


def _extract_scripts(text: str) -> set[str]:
    """Extract all Unicode letter script families present in text."""
    scripts: set[str] = set()
    for ch in text:
        script = _letter_script(ch)
        if script is not None:
            scripts.add(script)
    return scripts


def _find_corrupted_char(text: str) -> Optional[str]:
    """Return a description of the first replacement, control, or surrogate char if present."""
    for ch in text:
        code = ord(ch)
        if ch == "\ufffd":
            return "U+FFFD replacement character"
        if (code < 32 and ch not in ("\t", "\n", "\r")) or code == 0x7F:
            return f"control character U+{code:04X}"
        cat = unicodedata.category(ch)
        if cat in ("Co", "Cs"):
            return f"invalid/private-use character U+{code:04X}"
    return None


def _strip_document_markup(text: str) -> str:
    """Strip XML/HTML tags, markdown blockquotes, and reaction arrows so markup does not falsely ground symbols."""
    cleaned = _XML_COMMENT_RE.sub(" ", text)
    cleaned = _XML_TAG_RE.sub(" ", cleaned)
    cleaned = _BLOCKQUOTE_RE.sub(" ", cleaned)
    cleaned = _REACTION_ARROW_RE.sub(" ", cleaned)
    return cleaned


def _schema_metadata_text(
    output_model: Optional[Type[BaseModel]],
    constraints: Dict[str, Dict[str, Any]],
) -> str:
    """Collect field names, descriptions, enum choices, and defaults as grounding text."""
    parts: List[str] = []
    if output_model is not None and getattr(output_model, "__name__", None):
        parts.append(output_model.__name__)
    for field_name, info in constraints.items():
        parts.append(field_name)
        desc = info.get("description")
        if isinstance(desc, str) and desc:
            parts.append(desc)
        default = info.get("default")
        if isinstance(default, str) and default:
            parts.append(default)
        for choice in info.get("enum") or []:
            if isinstance(choice, str) and choice:
                parts.append(choice)
    return "\n".join(parts)


def _iter_record_dicts(output: Any) -> List[Dict[str, Any]]:
    """Extract a list of record dictionaries from a BaseModel, dict, or list."""
    if output is None:
        return []
    if isinstance(output, BaseModel):
        if hasattr(output, "records") and isinstance(getattr(output, "records"), list):
            records = getattr(output, "records")
        else:
            return [output.model_dump(mode="python")]
    elif isinstance(output, dict):
        if isinstance(output.get("records"), list):
            records = output["records"]
        else:
            return [output]
    elif isinstance(output, list):
        records = output
    else:
        return []

    dicts: List[Dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, BaseModel):
            dicts.append(rec.model_dump(mode="python"))
        elif isinstance(rec, dict):
            dicts.append(rec)
    return dicts


def _iter_string_values(value: Any) -> Iterable[str]:
    """Yield all string leaves from nested dict/list/scalar values."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_string_values(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_string_values(item)


def _iter_record_objects(output: Any) -> List[Any]:
    """Extract mutable record objects (BaseModel or dict) from output."""
    if output is None:
        return []
    if isinstance(output, BaseModel):
        if hasattr(output, "records") and isinstance(getattr(output, "records"), list):
            return list(getattr(output, "records"))
        return [output]
    if isinstance(output, dict):
        if isinstance(output.get("records"), list):
            return list(output["records"])
        return [output]
    if isinstance(output, list):
        return list(output)
    return []


def _get_record_field(rec: Any, field_name: str) -> Any:
    if isinstance(rec, dict):
        return rec.get(field_name)
    return getattr(rec, field_name, None)


def _set_record_field(rec: Any, field_name: str, value: Any) -> None:
    if isinstance(rec, dict):
        rec[field_name] = value
    elif isinstance(rec, BaseModel):
        setattr(rec, field_name, value)


def validate_output_field_validators(
    output: Any,
    *,
    output_model: Optional[Type[BaseModel]] = None,
) -> Any:
    """Run declarative FieldSpec validators attached to ``output_model``.

    Completely domain-agnostic: delegates validation to the registered
    ``FieldValidatorRegistry`` handlers declared on each field.
    """
    if output_model is None:
        return output

    validators_by_field = extract_field_validators(output_model)
    if not validators_by_field:
        return output

    records = _iter_record_objects(output)
    if not records:
        return output

    constraints = extract_schema_field_constraints(output_model)
    reasons: List[str] = []
    flagged_fields: set[str] = set()
    failed_indices: set[int] = set()

    for rec_idx, rec in enumerate(records):
        for field_name, specs in validators_by_field.items():
            field_info = constraints.get(field_name, {})
            is_required = bool(field_info.get("required", True))
            default_val = field_info.get("default")
            current_val = _get_record_field(rec, field_name)

            for spec in specs:
                res = validate_field_value(
                    spec,
                    current_val,
                    field_name=field_name,
                    required=is_required,
                    record=rec,
                )
                if res.valid:
                    if res.value != current_val:
                        _set_record_field(rec, field_name, res.value)
                        current_val = res.value
                    continue

                err_detail = res.error or "validation failed"
                if spec.on_error == "nullify":
                    logger.warning(
                        f"Field '{field_name}' at record {rec_idx} failed validator '{spec.name}' "
                        f"({err_detail}); nullifying to {default_val!r}"
                    )
                    _set_record_field(rec, field_name, default_val)
                    break
                if spec.on_error == "warn":
                    logger.warning(
                        f"Field '{field_name}' at record {rec_idx} failed validator '{spec.name}' "
                        f"({err_detail}); keeping value {current_val!r}"
                    )
                    continue

                # Default policy: "retry" -> raise StructuredOutputSanityError
                flagged_fields.add(field_name)
                failed_indices.add(rec_idx)
                reasons.append(
                    f"field '{field_name}' at record {rec_idx} failed validator '{spec.name}': "
                    f"{err_detail} (value={current_val!r})"
                )
                break

    if reasons:
        unique_reasons = list(dict.fromkeys(reasons))
        summary = "; ".join(unique_reasons[:5])
        if len(unique_reasons) > 5:
            summary += f" (+{len(unique_reasons) - 5} more)"
        raise StructuredOutputSanityError(
            f"Structured output field validation failed: {summary}",
            reasons=unique_reasons,
            field_names=sorted(flagged_fields),
            failed_record_indices=sorted(failed_indices),
            partial_output=output,
        )

    return output


def validate_structured_output_sanity(
    output: Any,
    *,
    document_text: str = "",
    system_instruction: str = "",
    output_model: Optional[Type[BaseModel]] = None,
    config: Optional[SanityCheckConfig] = None,
) -> Any:
    """Validate structured output against domain-independent decoding sanity checks.

    Checks performed:
    1. Re-validates against `output_model` if output is a raw dict/list.
    2. Detects corrupted/replacement/control characters in any string field.
    3. Detects ungrounded non-Latin/non-Greek Unicode scripts absent from the
       document text, system instruction, and schema metadata.
    4. Detects sticky/collapsed purely symbolic values across multi-record arrays
       that are absent from markup-stripped document text.

    Raises:
        StructuredOutputSanityError: If any sanity check fails.
    """
    cfg = config or SanityCheckConfig()
    if not cfg.enabled:
        return output

    if output_model is not None and not isinstance(output, output_model):
        try:
            output = output_model.model_validate(output)
        except Exception as exc:
            raise StructuredOutputSanityError(
                f"Structured output failed schema validation: {exc}",
                reasons=[f"schema_validation: {exc}"],
            ) from exc

    constraints = extract_schema_field_constraints(output_model) if output_model is not None else {}
    records = _iter_record_dicts(output)
    if not records:
        return output

    schema_text = _schema_metadata_text(output_model, constraints)
    reference_corpus = f"{document_text}\n{system_instruction}\n{schema_text}"
    reference_scripts = _extract_scripts(reference_corpus) | _ALWAYS_ALLOWED_SCRIPTS
    clean_doc_text = _strip_document_markup(document_text)

    reasons: List[str] = []
    flagged_fields: set[str] = set()
    failed_indices: set[int] = set()

    # Track per-field string values for collapse detection
    field_values: Dict[str, List[str]] = {}

    for rec_idx, rec in enumerate(records):
        for field_name, raw_val in rec.items():
            for str_val in _iter_string_values(raw_val):
                stripped = str_val.strip()
                if not stripped:
                    continue

                if cfg.check_control_chars:
                    bad_char = _find_corrupted_char(stripped)
                    if bad_char is not None:
                        flagged_fields.add(field_name)
                        failed_indices.add(rec_idx)
                        reasons.append(
                            f"corrupted character ({bad_char}) in field '{field_name}' at record {rec_idx}"
                        )

                if cfg.check_ungrounded_scripts:
                    val_scripts = _extract_scripts(stripped)
                    foreign = sorted(val_scripts - reference_scripts)
                    if foreign:
                        flagged_fields.add(field_name)
                        failed_indices.add(rec_idx)
                        reasons.append(
                            f"ungrounded script(s) {foreign} in field '{field_name}' "
                            f"(value={stripped!r}) at record {rec_idx}"
                        )

                if isinstance(raw_val, str):
                    field_values.setdefault(field_name, []).append(stripped)

    if cfg.check_sticky_ungrounded_symbols and len(records) >= cfg.min_records_for_collapse:
        for field_name, vals in field_values.items():
            if len(vals) < cfg.min_records_for_collapse:
                continue
            # Skip fields already constrained by an explicit schema enum
            if constraints.get(field_name, {}).get("enum"):
                continue

            counts = Counter(vals)
            top_val, top_count = counts.most_common(1)[0]
            ratio = top_count / len(vals)
            if top_count >= cfg.min_records_for_collapse and ratio >= cfg.collapse_ratio_threshold:
                is_purely_symbolic = not any(ch.isalnum() for ch in top_val)
                if is_purely_symbolic:
                    grounded = top_val in clean_doc_text
                    if not grounded:
                        flagged_fields.add(field_name)
                        reasons.append(
                            f"sticky ungrounded symbolic collapse in field '{field_name}': "
                            f"{top_val!r} repeated {top_count}/{len(vals)} times without grounding in document"
                        )

    if reasons:
        # Deduplicate while preserving order and cap length for concise retry feedback
        unique_reasons = list(dict.fromkeys(reasons))
        summary = "; ".join(unique_reasons[:5])
        if len(unique_reasons) > 5:
            summary += f" (+{len(unique_reasons) - 5} more)"
        raise StructuredOutputSanityError(
            f"Structured output sanity check failed: {summary}",
            reasons=unique_reasons,
            field_names=sorted(flagged_fields),
            failed_record_indices=sorted(failed_indices),
            partial_output=output,
        )

    if cfg.check_field_validators and output_model is not None:
        validate_output_field_validators(output, output_model=output_model)

    return output
