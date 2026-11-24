"""Declarative, domain-agnostic extraction tool registry and specifications.

Provides ``ExtractionToolSpec``, ``RegisteredToolProvider``, and a central
``ExtractionToolRegistry`` for attaching opt-in domain tools (such as chemical
Markush assemblers, unit converters, or lookup calculators) via schema YAML
without coupling ``ExtractionGraph`` to any specific domain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .validators import _resolve_dotted_callable


@dataclass(frozen=True)
class ExtractionToolSpec:
    """Normalized declarative extraction tool specification attached to a schema."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    callable_path: Optional[str] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Extraction tool 'name' must be a non-empty string")

    def __hash__(self) -> int:
        canonical_params = json.dumps(self.params, sort_keys=True, default=str)
        return hash((self.name, canonical_params, self.callable_path, self.enabled))

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-serializable dict for schema validation hashing."""
        data: Dict[str, Any] = {
            "name": self.name,
            "enabled": self.enabled,
        }
        if self.params:
            data["params"] = self.params
        if self.callable_path is not None:
            data["callable"] = self.callable_path
        return data


@dataclass(frozen=True)
class RegisteredToolProvider:
    """Resolved runtime descriptor for an extraction tool plugin."""

    name: str
    build_tool: Callable[..., Any]
    is_applicable: Callable[[str], bool] = lambda _doc_text: True
    policy_hint: str = ""
    auto_remind: bool = True
    structured_output_hint: str = ""


def normalize_tool_spec(
    raw: Union[str, Dict[str, Any], ExtractionToolSpec],
) -> ExtractionToolSpec:
    """Normalize string shorthand, dict config, or ExtractionToolSpec."""
    if isinstance(raw, ExtractionToolSpec):
        return raw

    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            raise ValueError("Extraction tool specification string cannot be empty")
        if ":" in cleaned:
            return ExtractionToolSpec(name=cleaned, callable_path=cleaned)
        return ExtractionToolSpec(name=cleaned)

    if isinstance(raw, dict):
        callable_path = raw.get("callable")
        raw_name = raw.get("name") or raw.get("type") or callable_path
        if not raw_name or not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Extraction tool dict must specify a non-empty 'name' or 'callable'")
        name = raw_name.strip()
        if callable_path is None and ":" in name:
            callable_path = name

        enabled = bool(raw.get("enabled", True))
        reserved_keys = {"name", "type", "callable", "enabled", "params"}
        params: Dict[str, Any] = {}
        if isinstance(raw.get("params"), dict):
            params.update(raw["params"])
        for k, v in raw.items():
            if k not in reserved_keys:
                params[k] = v

        return ExtractionToolSpec(
            name=name,
            params=params,
            callable_path=str(callable_path).strip() if callable_path else None,
            enabled=enabled,
        )

    raise ValueError(f"Unsupported extraction tool specification type: {type(raw).__name__}")


def serialize_tool_spec(
    raw: Union[str, Dict[str, Any], ExtractionToolSpec],
) -> Union[str, Dict[str, Any]]:
    """Serialize an ExtractionToolSpec concisely for YAML persistence."""
    spec = normalize_tool_spec(raw)
    is_plain_named = (
        not spec.params
        and spec.enabled
        and (spec.callable_path is None or spec.callable_path == spec.name)
    )
    if is_plain_named:
        return spec.name

    out: Dict[str, Any] = {"name": spec.name}
    if spec.callable_path and spec.callable_path != spec.name:
        out["callable"] = spec.callable_path
    if not spec.enabled:
        out["enabled"] = False
    if spec.params:
        out["params"] = spec.params
    return out


# Optional domain-specific tool providers loaded lazily only when referenced by a schema.
_LAZY_BUILTIN_TOOL_PROVIDERS: Dict[str, str] = {
    "assemble_markush_smiles": (
        "ae.core.schema.domain_tools.chemistry:get_markush_assembly_tool_provider"
    ),
    "resolve_iupac_to_smiles": (
        "ae.core.schema.domain_tools.chemistry:get_iupac_to_smiles_tool_provider"
    ),
}


def _coerce_to_provider(name: str, resolved_obj: Any) -> RegisteredToolProvider:
    """Convert a factory result, callable, or plain BaseTool into RegisteredToolProvider."""
    if isinstance(resolved_obj, RegisteredToolProvider):
        return resolved_obj

    # Plain LangChain BaseTool or mock tool instance with `.name` & `.invoke`
    if not callable(resolved_obj) and hasattr(resolved_obj, "invoke"):
        tool_name = str(getattr(resolved_obj, "name", None) or name)
        hint = str(getattr(resolved_obj, "description", "") or "")
        return RegisteredToolProvider(
            name=tool_name,
            build_tool=lambda _t=resolved_obj, **_: _t,
            is_applicable=lambda _doc_text: True,
            policy_hint=hint,
        )

    if callable(resolved_obj):
        # Invoke zero-arg provider factories, or wrap BaseTool factories/objects
        try:
            probe = resolved_obj()
        except TypeError:
            probe = None

        if isinstance(probe, RegisteredToolProvider):
            return probe
        if probe is not None and hasattr(probe, "invoke"):
            return RegisteredToolProvider(
                name=str(getattr(probe, "name", None) or name),
                build_tool=resolved_obj,
                is_applicable=lambda _doc_text: True,
                policy_hint=str(getattr(probe, "description", "") or ""),
            )
        return RegisteredToolProvider(
            name=name,
            build_tool=resolved_obj,
            is_applicable=lambda _doc_text: True,
        )

    raise TypeError(
        f"Resolved extraction tool provider for {name!r} has unsupported type "
        f"{type(resolved_obj).__name__}"
    )


class ExtractionToolRegistry:
    """Registry mapping schema tool names to lazy or custom RegisteredToolProviders."""

    def __init__(self) -> None:
        self._providers: Dict[str, RegisteredToolProvider] = {}
        self._lazy_providers: Dict[str, str] = dict(_LAZY_BUILTIN_TOOL_PROVIDERS)

    def register(
        self,
        name: str,
        provider_or_callable: Union[RegisteredToolProvider, Callable[..., Any]],
        *,
        is_applicable: Optional[Callable[[str], bool]] = None,
        policy_hint: str = "",
    ) -> None:
        """Register an extraction tool provider or builder under ``name``."""
        key = name.strip()
        if not key:
            raise ValueError("Extraction tool name must be non-empty")
        if isinstance(provider_or_callable, RegisteredToolProvider):
            self._providers[key] = provider_or_callable
            return
        if callable(provider_or_callable):
            self._providers[key] = RegisteredToolProvider(
                name=key,
                build_tool=provider_or_callable,
                is_applicable=is_applicable or (lambda _doc_text: True),
                policy_hint=policy_hint,
            )
            return
        raise TypeError(f"Provider for {key!r} must be RegisteredToolProvider or callable")

    def unregister(self, name: str) -> None:
        """Remove a registered tool provider (does not remove built-in lazy entries)."""
        self._providers.pop(name.strip(), None)

    def has(self, name: str) -> bool:
        """Return True if ``name`` is registered or available as a lazy provider."""
        key = name.strip()
        return key in self._providers or key in self._lazy_providers

    def resolve(self, spec: Union[str, Dict[str, Any], ExtractionToolSpec]) -> RegisteredToolProvider:
        """Resolve ``spec`` into a RegisteredToolProvider."""
        norm = normalize_tool_spec(spec)
        if norm.name in self._providers:
            return self._providers[norm.name]
        if norm.callable_path:
            resolved_obj = _resolve_dotted_callable(norm.callable_path)
            return _coerce_to_provider(norm.name, resolved_obj)
        if norm.name in self._lazy_providers:
            resolved_obj = _resolve_dotted_callable(self._lazy_providers[norm.name])
            provider = _coerce_to_provider(norm.name, resolved_obj)
            self._providers[norm.name] = provider
            return provider
        raise ValueError(f"Unknown extraction tool {norm.name!r}")

    def validate_spec(self, spec: Union[str, Dict[str, Any], ExtractionToolSpec]) -> List[str]:
        """Validate that an ExtractionToolSpec references a known or importable tool."""
        errors: List[str] = []
        norm = normalize_tool_spec(spec)
        if not self.has(norm.name) and not norm.callable_path:
            errors.append(f"unknown extraction tool {norm.name!r}")
            return errors

        if norm.callable_path and not self.has(norm.name):
            try:
                resolved_obj = _resolve_dotted_callable(norm.callable_path)
                _coerce_to_provider(norm.name, resolved_obj)
            except Exception as exc:
                errors.append(f"invalid callable {norm.callable_path!r}: {exc}")
        return errors


DEFAULT_TOOL_REGISTRY = ExtractionToolRegistry()


def register_extraction_tool(
    name: str,
    provider_or_callable: Union[RegisteredToolProvider, Callable[..., Any]],
    *,
    is_applicable: Optional[Callable[[str], bool]] = None,
    policy_hint: str = "",
) -> None:
    """Register a custom extraction tool in the global registry."""
    DEFAULT_TOOL_REGISTRY.register(
        name,
        provider_or_callable,
        is_applicable=is_applicable,
        policy_hint=policy_hint,
    )


def unregister_extraction_tool(name: str) -> None:
    """Unregister an extraction tool from the global registry."""
    DEFAULT_TOOL_REGISTRY.unregister(name)


def has_extraction_tool(name: str) -> bool:
    """Return True if an extraction tool is registered or lazily available."""
    return DEFAULT_TOOL_REGISTRY.has(name)


def validate_tool_config(spec: Union[str, Dict[str, Any], ExtractionToolSpec]) -> List[str]:
    """Validate an ExtractionToolSpec against the global registry."""
    return DEFAULT_TOOL_REGISTRY.validate_spec(spec)


def resolve_extraction_tool(
    spec: Union[str, Dict[str, Any], ExtractionToolSpec],
) -> RegisteredToolProvider:
    """Resolve an ExtractionToolSpec using the global registry."""
    return DEFAULT_TOOL_REGISTRY.resolve(spec)
