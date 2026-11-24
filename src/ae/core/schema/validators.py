"""Declarative, domain-agnostic field validator registry and specifications.

Provides ``FieldValidatorSpec``, ``FieldValidationResult``, and a central
``FieldValidatorRegistry`` for attaching opt-in syntactic/semantic validation
rules to ``FieldSpec`` without coupling core extraction to any specific domain.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

OnErrorPolicy = Literal["retry", "nullify", "warn"]
_VALID_ON_ERROR_POLICIES = frozenset({"retry", "nullify", "warn"})

ValidatorCallable = Callable[..., Any]


@dataclass(frozen=True)
class FieldValidationResult:
    """Outcome of executing a single declarative field validator."""

    valid: bool
    value: Any = None
    error: Optional[str] = None


@dataclass(frozen=True)
class FieldValidatorSpec:
    """Normalized declarative validator specification attached to a field."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    on_error: OnErrorPolicy = "retry"
    message: Optional[str] = None
    callable_path: Optional[str] = None
    ignore_values: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Validator 'name' must be a non-empty string")
        if self.on_error not in _VALID_ON_ERROR_POLICIES:
            raise ValueError(
                f"Invalid on_error policy {self.on_error!r}; "
                f"expected one of {sorted(_VALID_ON_ERROR_POLICIES)}"
            )

    def __hash__(self) -> int:
        canonical_params = json.dumps(self.params, sort_keys=True, default=str)
        return hash(
            (
                self.name,
                canonical_params,
                self.on_error,
                self.message,
                self.callable_path,
                self.ignore_values,
            )
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-serializable dict for schema hashing."""
        data: Dict[str, Any] = {
            "name": self.name,
            "on_error": self.on_error,
        }
        if self.params:
            data["params"] = self.params
        if self.message is not None:
            data["message"] = self.message
        if self.callable_path is not None:
            data["callable"] = self.callable_path
        if self.ignore_values:
            data["ignore_values"] = list(self.ignore_values)
        return data


def normalize_validator_spec(
    raw: Union[str, Dict[str, Any], FieldValidatorSpec],
) -> FieldValidatorSpec:
    """Normalize string shorthand, dict config, or FieldValidatorSpec into FieldValidatorSpec."""
    if isinstance(raw, FieldValidatorSpec):
        return raw

    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            raise ValueError("Validator specification string cannot be empty")
        if ":" in cleaned:
            return FieldValidatorSpec(name=cleaned, callable_path=cleaned)
        return FieldValidatorSpec(name=cleaned)

    if isinstance(raw, dict):
        callable_path = raw.get("callable")
        raw_name = raw.get("name") or raw.get("type") or callable_path
        if not raw_name or not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Validator dict must specify a non-empty 'name' or 'callable'")
        name = raw_name.strip()
        if callable_path is None and ":" in name:
            callable_path = name

        on_error = raw.get("on_error", "retry")
        message = raw.get("message")
        raw_ignore = raw.get("ignore_values") or ()
        if isinstance(raw_ignore, str):
            ignore_values: Tuple[str, ...] = (raw_ignore,)
        else:
            ignore_values = tuple(str(v) for v in raw_ignore)

        reserved_keys = {
            "name",
            "type",
            "callable",
            "on_error",
            "message",
            "ignore_values",
            "params",
        }
        params: Dict[str, Any] = {}
        if isinstance(raw.get("params"), dict):
            params.update(raw["params"])
        for k, v in raw.items():
            if k not in reserved_keys:
                params[k] = v

        return FieldValidatorSpec(
            name=name,
            params=params,
            on_error=on_error,
            message=message,
            callable_path=str(callable_path).strip() if callable_path else None,
            ignore_values=ignore_values,
        )

    raise ValueError(f"Unsupported validator specification type: {type(raw).__name__}")


def serialize_validator_spec(
    raw: Union[str, Dict[str, Any], FieldValidatorSpec],
) -> Union[str, Dict[str, Any]]:
    """Serialize a validator specification concisely for YAML persistence."""
    spec = normalize_validator_spec(raw)
    is_plain_named = (
        not spec.params
        and spec.on_error == "retry"
        and spec.message is None
        and not spec.ignore_values
        and (spec.callable_path is None or spec.callable_path == spec.name)
    )
    if is_plain_named:
        return spec.name

    out: Dict[str, Any] = {"name": spec.name}
    if spec.callable_path and spec.callable_path != spec.name:
        out["callable"] = spec.callable_path
    if spec.on_error != "retry":
        out["on_error"] = spec.on_error
    if spec.message is not None:
        out["message"] = spec.message
    if spec.ignore_values:
        out["ignore_values"] = list(spec.ignore_values)
    out.update(spec.params)
    return out


def _coerce_patterns(params: Dict[str, Any]) -> List[str]:
    patterns = params.get("patterns")
    single = params.get("pattern")
    result: List[str] = []
    if single is not None:
        result.append(str(single))
    if isinstance(patterns, str):
        result.append(patterns)
    elif isinstance(patterns, Iterable):
        result.extend(str(p) for p in patterns)
    return result


def _coerce_substrings(params: Dict[str, Any]) -> List[str]:
    substrings = params.get("substrings")
    single = params.get("substring")
    result: List[str] = []
    if single is not None:
        result.append(str(single))
    if isinstance(substrings, str):
        result.append(substrings)
    elif isinstance(substrings, Iterable):
        result.extend(str(s) for s in substrings)
    return result


def _validate_not_regex(value: Any, **params: Any) -> FieldValidationResult:
    """Fail if any configured regex pattern matches the string value."""
    if not isinstance(value, str):
        return FieldValidationResult(valid=True, value=value)
    patterns = _coerce_patterns(params)
    flags = 0 if params.get("case_sensitive", True) else re.IGNORECASE
    for pat in patterns:
        match = re.search(pat, value, flags)
        if match:
            return FieldValidationResult(
                valid=False,
                value=value,
                error=f"matched forbidden pattern {pat!r} (matched {match.group(0)!r})",
            )
    return FieldValidationResult(valid=True, value=value)


def _validate_regex(value: Any, **params: Any) -> FieldValidationResult:
    """Fail if the string value does not match the configured regex pattern."""
    if not isinstance(value, str):
        return FieldValidationResult(valid=True, value=value)
    pattern = params.get("pattern")
    if not pattern:
        return FieldValidationResult(valid=False, value=value, error="Missing 'pattern' parameter")
    flags = 0 if params.get("case_sensitive", True) else re.IGNORECASE
    matcher = re.fullmatch if params.get("fullmatch", False) else re.search
    if not matcher(str(pattern), value, flags):
        return FieldValidationResult(
            valid=False,
            value=value,
            error=f"value {value!r} does not match required pattern {pattern!r}",
        )
    return FieldValidationResult(valid=True, value=value)


def _validate_forbidden_substrings(value: Any, **params: Any) -> FieldValidationResult:
    """Fail if any forbidden substring is present in the string value."""
    if not isinstance(value, str):
        return FieldValidationResult(valid=True, value=value)
    substrings = _coerce_substrings(params)
    case_sensitive = bool(params.get("case_sensitive", True))
    haystack = value if case_sensitive else value.lower()
    for sub in substrings:
        needle = sub if case_sensitive else sub.lower()
        if needle in haystack:
            return FieldValidationResult(
                valid=False,
                value=value,
                error=f"contains forbidden substring {sub!r}",
            )
    return FieldValidationResult(valid=True, value=value)


def _validate_non_empty(value: Any, **_: Any) -> FieldValidationResult:
    """Fail if string value is empty or whitespace-only."""
    if isinstance(value, str) and not value.strip():
        return FieldValidationResult(valid=False, value=value, error="value must not be empty")
    return FieldValidationResult(valid=True, value=value)


def _validate_length(value: Any, **params: Any) -> FieldValidationResult:
    """Validate string or collection length against min_length / max_length."""
    if not hasattr(value, "__len__"):
        return FieldValidationResult(valid=True, value=value)
    length = len(value)
    min_len = params.get("min_length")
    max_len = params.get("max_length")
    if min_len is not None and length < int(min_len):
        return FieldValidationResult(
            valid=False,
            value=value,
            error=f"length {length} is less than min_length={min_len}",
        )
    if max_len is not None and length > int(max_len):
        return FieldValidationResult(
            valid=False,
            value=value,
            error=f"length {length} exceeds max_length={max_len}",
        )
    return FieldValidationResult(valid=True, value=value)


def _resolve_dotted_callable(callable_path: str) -> ValidatorCallable:
    """Import a callable from 'module.path:attr' or 'module.path.attr'."""
    if ":" in callable_path:
        module_name, attr_name = callable_path.split(":", 1)
    elif "." in callable_path:
        module_name, attr_name = callable_path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid callable path {callable_path!r}; expected 'module:func'")
    module = importlib.import_module(module_name.strip())
    fn = getattr(module, attr_name.strip(), None)
    if fn is None or not callable(fn):
        raise ValueError(f"Callable {callable_path!r} not found or not callable")
    return fn


# Optional domain-specific validators loaded lazily only when referenced by a schema.
_LAZY_BUILTIN_PROVIDERS: Dict[str, str] = {
    "smiles": "ae.core.schema.domain_validators.chemistry:validate_smiles",
    "unit": "ae.core.schema.quantity_validators:validate_unit",
    "numeric_quantity": "ae.core.schema.quantity_validators:validate_numeric_quantity",
    "compute_validators": "ae.core.schema.quantity_validators:validate_compute_expression",
    "compute": "ae.core.schema.quantity_validators:validate_compute_expression",
}


class FieldValidatorRegistry:
    """Registry mapping validator names to callables with lazy domain provider support."""

    def __init__(self) -> None:
        self._validators: Dict[str, ValidatorCallable] = {}
        self._lazy_providers: Dict[str, str] = dict(_LAZY_BUILTIN_PROVIDERS)
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("not_regex", _validate_not_regex)
        self.register("forbidden_patterns", _validate_not_regex)
        self.register("regex", _validate_regex)
        self.register("forbidden_substrings", _validate_forbidden_substrings)
        self.register("not_contains", _validate_forbidden_substrings)
        self.register("non_empty", _validate_non_empty)
        self.register("length", _validate_length)

    def register(self, name: str, fn: ValidatorCallable) -> None:
        """Register a validator callable under ``name``."""
        key = name.strip()
        if not key:
            raise ValueError("Validator name must be non-empty")
        if not callable(fn):
            raise TypeError(f"Validator for {key!r} must be callable")
        self._validators[key] = fn

    def unregister(self, name: str) -> None:
        """Remove a registered validator (does not remove built-in lazy providers)."""
        self._validators.pop(name.strip(), None)

    def has(self, name: str) -> bool:
        """Return True if ``name`` is registered or available as a lazy provider."""
        key = name.strip()
        return key in self._validators or key in self._lazy_providers

    def resolve(self, spec: FieldValidatorSpec) -> ValidatorCallable:
        """Resolve ``spec`` to a callable validator function."""
        if spec.name in self._validators:
            return self._validators[spec.name]
        if spec.callable_path:
            return _resolve_dotted_callable(spec.callable_path)
        if spec.name in self._lazy_providers:
            fn = _resolve_dotted_callable(self._lazy_providers[spec.name])
            self._validators[spec.name] = fn
            return fn
        raise ValueError(f"Unknown field validator {spec.name!r}")

    def validate_spec(self, spec: FieldValidatorSpec) -> List[str]:
        """Validate that a FieldValidatorSpec references a known validator and valid params."""
        errors: List[str] = []
        if not self.has(spec.name) and not spec.callable_path:
            errors.append(f"unknown validator {spec.name!r}")
            return errors

        if spec.callable_path and not self.has(spec.name):
            try:
                _resolve_dotted_callable(spec.callable_path)
            except Exception as exc:
                errors.append(f"invalid callable {spec.callable_path!r}: {exc}")

        if spec.name in ("not_regex", "forbidden_patterns"):
            patterns = _coerce_patterns(spec.params)
            if not patterns:
                errors.append(f"validator {spec.name!r} requires 'pattern' or 'patterns'")
            for pat in patterns:
                try:
                    re.compile(pat)
                except re.error as exc:
                    errors.append(f"invalid regex {pat!r} in {spec.name!r}: {exc}")

        if spec.name == "regex":
            regex_pat = spec.params.get("pattern")
            if not regex_pat or not isinstance(regex_pat, str):
                errors.append("validator 'regex' requires a string 'pattern'")
            else:
                try:
                    re.compile(regex_pat)
                except re.error as exc:
                    errors.append(f"invalid regex {regex_pat!r} in 'regex': {exc}")

        if spec.name in ("forbidden_substrings", "not_contains"):
            if not _coerce_substrings(spec.params):
                errors.append(f"validator {spec.name!r} requires 'substring' or 'substrings'")

        if spec.name == "length":
            min_len = spec.params.get("min_length")
            max_len = spec.params.get("max_length")
            if min_len is None and max_len is None:
                errors.append("validator 'length' requires 'min_length' and/or 'max_length'")
            elif min_len is not None and max_len is not None and int(min_len) > int(max_len):
                errors.append("validator 'length': min_length cannot exceed max_length")

        if spec.name == "numeric_quantity":
            min_v = spec.params.get("min_value")
            max_v = spec.params.get("max_value")
            if min_v is not None and max_v is not None and float(min_v) > float(max_v):
                errors.append("validator 'numeric_quantity': min_value cannot exceed max_value")

        if spec.name in ("compute_validators", "compute"):
            expr = spec.params.get("expression") or spec.params.get("expr")
            if not expr or not isinstance(expr, str):
                errors.append(f"validator {spec.name!r} requires a string 'expression'")
            else:
                from ae.core.schema.quantity_validators import validate_compute_syntax

                syntax_err = validate_compute_syntax(expr)
                if syntax_err:
                    errors.append(f"validator {spec.name!r}: {syntax_err}")

        return errors


DEFAULT_VALIDATOR_REGISTRY = FieldValidatorRegistry()


def register_field_validator(name: str, fn: ValidatorCallable) -> None:
    """Register a custom field validator in the global registry."""
    DEFAULT_VALIDATOR_REGISTRY.register(name, fn)


def unregister_field_validator(name: str) -> None:
    """Unregister a field validator from the global registry."""
    DEFAULT_VALIDATOR_REGISTRY.unregister(name)


def has_field_validator(name: str) -> bool:
    """Return True if a field validator is registered or lazily available."""
    return DEFAULT_VALIDATOR_REGISTRY.has(name)


def validate_validator_config(spec: FieldValidatorSpec) -> List[str]:
    """Validate a FieldValidatorSpec against the global registry."""
    return DEFAULT_VALIDATOR_REGISTRY.validate_spec(spec)


def _normalize_callable_return(raw_result: Any, original_value: Any) -> FieldValidationResult:
    if isinstance(raw_result, FieldValidationResult):
        return raw_result
    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        ok, err = raw_result
        return FieldValidationResult(valid=bool(ok), value=original_value, error=err)
    if isinstance(raw_result, bool):
        return FieldValidationResult(
            valid=raw_result,
            value=original_value,
            error=None if raw_result else "validation failed",
        )
    raise TypeError(
        f"Validator returned unsupported result type {type(raw_result).__name__}; "
        "expected FieldValidationResult, bool, or (bool, error)"
    )


def validate_field_value(
    spec: FieldValidatorSpec,
    value: Any,
    *,
    field_name: str = "",
    required: bool = True,
    registry: Optional[FieldValidatorRegistry] = None,
    record: Optional[Any] = None,
) -> FieldValidationResult:
    """Execute ``spec`` on ``value`` with consistent optional/sentinel handling."""
    if value is None:
        return FieldValidationResult(valid=True, value=value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped and not required and spec.name != "non_empty":
            return FieldValidationResult(valid=True, value=value)
        if spec.ignore_values and stripped in spec.ignore_values:
            return FieldValidationResult(valid=True, value=value)

    reg = registry or DEFAULT_VALIDATOR_REGISTRY
    fn = reg.resolve(spec)
    kwargs = dict(spec.params)
    kwargs.setdefault("required", required)
    if field_name:
        kwargs.setdefault("field_name", field_name)
    if record is not None:
        kwargs.setdefault("record", record)

    try:
        raw_res = fn(value, **kwargs)
    except TypeError:
        # Support simple user callables with signature fn(value) without **kwargs
        raw_res = fn(value)

    result = _normalize_callable_return(raw_res, value)
    if not result.valid and spec.message:
        return FieldValidationResult(valid=False, value=result.value, error=spec.message)
    return result


def supports_value_comparison(
    spec: FieldValidatorSpec,
    registry: Optional[FieldValidatorRegistry] = None,
) -> bool:
    """Return True if the resolved validator provides ``compare_values`` or ``canonicalize``."""
    reg = registry or DEFAULT_VALIDATOR_REGISTRY
    try:
        fn = reg.resolve(spec)
    except Exception:
        return False
    if callable(getattr(fn, "compare_values", None)):
        return True
    try:
        return "canonicalize" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def compare_field_values_via_validator(
    spec: FieldValidatorSpec,
    val_a: Any,
    val_b: Any,
    *,
    registry: Optional[FieldValidatorRegistry] = None,
) -> Tuple[Optional[bool], Optional[str]]:
    """Delegate semantic equivalence of two values to a validator hook or canonicalizer."""
    reg = registry or DEFAULT_VALIDATOR_REGISTRY
    try:
        fn = reg.resolve(spec)
    except Exception:
        return None, None

    compare_fn = getattr(fn, "compare_values", None)
    if callable(compare_fn):
        try:
            raw = compare_fn(val_a, val_b, **spec.params)
        except TypeError:
            raw = compare_fn(val_a, val_b)
        if isinstance(raw, tuple) and len(raw) == 2:
            match_val, reason = raw
            return (None if match_val is None else bool(match_val)), reason
        if isinstance(raw, bool):
            return raw, (None if raw else "validator_mismatch")
        return None, None

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}

    if "canonicalize" not in params:
        return None, None

    if val_a is None or val_b is None:
        return None, None
    if isinstance(val_a, str) and not val_a.strip():
        return None, None
    if isinstance(val_b, str) and not val_b.strip():
        return None, None

    canon_spec = FieldValidatorSpec(
        name=spec.name,
        params={**spec.params, "canonicalize": True},
        on_error=spec.on_error,
        message=spec.message,
        callable_path=spec.callable_path,
        ignore_values=spec.ignore_values,
    )
    try:
        res_a = validate_field_value(canon_spec, val_a, required=True)
        res_b = validate_field_value(canon_spec, val_b, required=True)
    except Exception:
        return None, None

    if not (res_a.valid and res_a.value is not None) or not (res_b.valid and res_b.value is not None):
        return False, "invalid_value"

    if res_a.value == res_b.value:
        return True, None
    return False, "canonical_mismatch"
