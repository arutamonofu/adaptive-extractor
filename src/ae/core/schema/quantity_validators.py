"""Domain-agnostic unit, numeric quantity, and safe compute validators."""

from __future__ import annotations

import ast
import functools
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ae.core.schema.validators import (
    FieldValidationResult,
    FieldValidatorSpec,
    normalize_validator_spec,
)

_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "−": "-",
        "–": "-",
        "—": "-",
    }
)
_SUPERSCRIPT_GROUP_RE = re.compile(r"([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_UNIT_NEG_EXP_RE = re.compile(r"([A-Za-zµμ°%]+)-(\d+)")
_UNIT_POS_EXP_RE = re.compile(r"([A-Za-zµμ°%]+)(\d+)")

_REL_PREFIX_RE = re.compile(r"^\s*(>=|<=|≥|≤|>|<|=|~|≈)\s*(.+)$")
_STD_SPLIT_RE = re.compile(r"^(.+?)\s*(?:±|\+/-|\+-)\s*(.+)$")
_NUM_TOKEN = r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?"
_RANGE_SPLIT_RE = re.compile(
    rf"^\s*({_NUM_TOKEN})\s*(?:-|–|—|−|\.\.|to)\s*({_NUM_TOKEN})\s*$"
)
_BRACKET_RANGE_RE = re.compile(
    rf"^\s*\[\s*({_NUM_TOKEN})\s*[,;]\s*({_NUM_TOKEN})\s*\]\s*$"
)

_FORBIDDEN_AST_NODES = (
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.NamedExpr,
)

_SAFE_COMPUTE_MAX_EXPR_LEN = 512
_SAFE_COMPUTE_MAX_EXP = 1000.0


@functools.lru_cache(maxsize=1)
def _get_unit_registry() -> Any:
    """Lazily initialize and cache a Pint UnitRegistry."""
    try:
        import pint
    except ImportError as exc:
        raise RuntimeError(
            "Pint is required for the 'unit' field validator (`pip install pint`)."
        ) from exc
    return pint.UnitRegistry()


def normalize_unit_expression(raw: str) -> str:
    """Normalize scientific unit typography into a Pint-parseable expression."""
    text = raw.strip()
    text = text.replace("·", "*").replace("•", "*").replace("⋅", "*").replace("×", "*")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("μ", "µ")

    def _replace_sup(match: re.Match[str]) -> str:
        ascii_exp = match.group(1).translate(_SUPERSCRIPT_MAP)
        return f"**{ascii_exp}"

    text = _SUPERSCRIPT_GROUP_RE.sub(_replace_sup, text)
    text = text.replace("^", "**")
    text = _UNIT_NEG_EXP_RE.sub(r"\1**-\2", text)
    return text


@functools.lru_cache(maxsize=4096)
def _parse_pint_unit(raw: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse a unit string into a Pint Unit object, returning (unit, error)."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty unit string"
    norm = normalize_unit_expression(raw)
    ureg = _get_unit_registry()
    try:
        unit = ureg.Unit(norm)
        return unit, None
    except Exception as exc:
        # Fallback for expressions like "cm2" or "m2"
        alt = _UNIT_POS_EXP_RE.sub(r"\1**\2", norm)
        if alt != norm:
            try:
                unit = ureg.Unit(alt)
                return unit, None
            except Exception:
                pass
        return None, f"invalid unit {raw!r}: {exc}"


def validate_unit(
    value: Any,
    *,
    canonicalize: bool = False,
    expected_unit: Optional[str] = None,
    **_: Any,
) -> FieldValidationResult:
    """Validate a physical/chemical unit string via Pint."""
    if not isinstance(value, str):
        return FieldValidationResult(
            valid=False,
            value=value,
            error=f"expected unit string, got {type(value).__name__}",
        )
    unit, err = _parse_pint_unit(value)
    if unit is None:
        return FieldValidationResult(valid=False, value=value, error=err)

    if expected_unit:
        exp_unit, exp_err = _parse_pint_unit(str(expected_unit))
        if exp_unit is None:
            return FieldValidationResult(
                valid=False,
                value=value,
                error=f"invalid expected_unit {expected_unit!r}: {exp_err}",
            )
        if unit.dimensionality != exp_unit.dimensionality:
            return FieldValidationResult(
                valid=False,
                value=value,
                error=(
                    f"unit {value!r} dimensionality {unit.dimensionality} "
                    f"incompatible with {expected_unit!r} ({exp_unit.dimensionality})"
                ),
            )

    final_val = str(unit) if canonicalize else value
    return FieldValidationResult(valid=True, value=final_val)


def compare_unit_values(
    val_a: Any,
    val_b: Any,
    *,
    strict_scale: bool = False,
    **_: Any,
) -> Tuple[Optional[bool], Optional[str]]:
    """Compare two unit strings via Pint dimensionality and conversion factor."""
    if not isinstance(val_a, str) or not isinstance(val_b, str):
        return False, "invalid_unit"
    u_a, err_a = _parse_pint_unit(val_a)
    u_b, err_b = _parse_pint_unit(val_b)
    if u_a is None or u_b is None or err_a is not None or err_b is not None:
        return False, "invalid_unit"

    if u_a.dimensionality != u_b.dimensionality:
        return False, "incompatible_units"

    ureg = _get_unit_registry()
    try:
        factor = float((1.0 * ureg.Quantity(1.0, u_a)).to(u_b).magnitude)
    except Exception:
        return False, "incompatible_units"

    if math.isclose(factor, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return True, None
    if strict_scale:
        return False, "unit_scale_mismatch"
    # Compatible dimensionality with non-1 scale factor (e.g. mM vs uM) may still match
    # when evaluated jointly with the coupled numeric value by SemanticJudge.
    return None, None


setattr(validate_unit, "compare_values", compare_unit_values)


@dataclass(frozen=True)
class ParsedQuantity:
    """Normalized representation of a numeric quantity, ± std, interval, or inequality."""

    relation: str
    kind: str  # "point" | "range"
    value: Optional[float] = None
    std: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None

    def to_canonical_str(self) -> str:
        """Render parsed quantity in canonical ASCII format."""
        prefix = "" if self.relation == "=" else f"{self.relation} "
        if self.kind == "range" and self.low is not None and self.high is not None:
            return f"{prefix}{self.low:g}-{self.high:g}"
        if self.value is None:
            return ""
        if self.std is not None:
            return f"{prefix}{self.value:g} ± {self.std:g}"
        return f"{prefix}{self.value:g}"


def _parse_decimal_number(token: str) -> Optional[float]:
    """Parse numeric token supporting European decimal commas ('19,1' -> 19.1)."""
    s = token.strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        return None
    if "," in s:
        if "." not in s and s.count(",") == 1:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_numeric_quantity(raw: Any) -> Tuple[Optional[ParsedQuantity], Optional[str]]:
    """Parse a numeric quantity supporting decimal commas, ± std, ranges, and >/<."""
    if isinstance(raw, bool):
        return None, "boolean is not a numeric quantity"
    if isinstance(raw, (int, float)):
        val = float(raw)
        if not math.isfinite(val):
            return None, "non-finite number"
        return ParsedQuantity(relation="=", kind="point", value=val), None
    if not isinstance(raw, str):
        return None, f"expected numeric string or number, got {type(raw).__name__}"

    text = raw.strip()
    if not text:
        return None, "empty numeric quantity"

    relation = "="
    rel_match = _REL_PREFIX_RE.match(text)
    if rel_match:
        raw_rel, body = rel_match.group(1), rel_match.group(2).strip()
        relation = {"≥": ">=", "≤": "<=", "≈": "~"}.get(raw_rel, raw_rel)
    else:
        body = text

    if not body:
        return None, f"missing numeric value after operator {relation!r}"

    # 1. Check ± standard deviation notation
    std_match = _STD_SPLIT_RE.match(body)
    if std_match:
        mean_val = _parse_decimal_number(std_match.group(1))
        std_val = _parse_decimal_number(std_match.group(2))
        if mean_val is None or std_val is None or std_val < 0:
            return None, f"invalid ± quantity {raw!r}"
        return (
            ParsedQuantity(relation=relation, kind="point", value=mean_val, std=std_val),
            None,
        )

    # 2. Check interval / range notation
    range_match = _RANGE_SPLIT_RE.match(body) or _BRACKET_RANGE_RE.match(body)
    if range_match:
        low_val = _parse_decimal_number(range_match.group(1))
        high_val = _parse_decimal_number(range_match.group(2))
        if low_val is None or high_val is None or low_val > high_val:
            return None, f"invalid numeric interval {raw!r}"
        return (
            ParsedQuantity(relation=relation, kind="range", low=low_val, high=high_val),
            None,
        )

    # 3. Single scalar number
    num_val = _parse_decimal_number(body)
    if num_val is None:
        return None, f"invalid numeric quantity {raw!r}"
    return ParsedQuantity(relation=relation, kind="point", value=num_val), None


def validate_numeric_quantity(
    value: Any,
    *,
    canonicalize: bool = False,
    allow_inequality: bool = True,
    allow_range: bool = True,
    allow_std: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    **_: Any,
) -> FieldValidationResult:
    """Validate and optionally canonicalize a numeric quantity string."""
    parsed, err = parse_numeric_quantity(value)
    if parsed is None:
        return FieldValidationResult(valid=False, value=value, error=err)

    if not allow_inequality and parsed.relation != "=":
        return FieldValidationResult(
            valid=False, value=value, error=f"inequality operator {parsed.relation!r} not allowed"
        )
    if not allow_range and parsed.kind == "range":
        return FieldValidationResult(valid=False, value=value, error="intervals are not allowed")
    if not allow_std and parsed.std is not None:
        return FieldValidationResult(valid=False, value=value, error="± std notation not allowed")

    bounds_vals = (
        [parsed.low, parsed.high]
        if parsed.kind == "range" and parsed.low is not None and parsed.high is not None
        else ([parsed.value] if parsed.value is not None else [])
    )
    for num in bounds_vals:
        if min_value is not None and num < float(min_value):
            return FieldValidationResult(
                valid=False, value=value, error=f"value {num} is below min_value={min_value}"
            )
        if max_value is not None and num > float(max_value):
            return FieldValidationResult(
                valid=False, value=value, error=f"value {num} exceeds max_value={max_value}"
            )

    final_val = parsed.to_canonical_str() if canonicalize else value
    return FieldValidationResult(valid=True, value=final_val)


def compare_numeric_quantity_values(
    val_a: Any,
    val_b: Any,
    *,
    rel_tol: float = 1e-9,
    require_std_match: bool = True,
    **_: Any,
) -> Tuple[Optional[bool], Optional[str]]:
    """Compare two numeric quantities preserving inequality relations, ± std, and intervals."""
    qa, err_a = parse_numeric_quantity(val_a)
    qb, err_b = parse_numeric_quantity(val_b)
    if qa is None or qb is None or err_a is not None or err_b is not None:
        return False, "invalid_numeric_quantity"

    if qa.relation != qb.relation:
        return False, "relation_mismatch"
    if qa.kind != qb.kind:
        return False, "quantity_kind_mismatch"

    if qa.kind == "range":
        assert qa.low is not None and qa.high is not None
        assert qb.low is not None and qb.high is not None
        ok = math.isclose(qa.low, qb.low, rel_tol=rel_tol, abs_tol=1e-12) and math.isclose(
            qa.high, qb.high, rel_tol=rel_tol, abs_tol=1e-12
        )
        return (True, None) if ok else (False, "numeric_mismatch")

    assert qa.value is not None and qb.value is not None
    if not math.isclose(qa.value, qb.value, rel_tol=rel_tol, abs_tol=1e-12):
        return False, "numeric_mismatch"

    if require_std_match:
        if (qa.std is None) != (qb.std is None):
            return False, "std_mismatch"
        if qa.std is not None and qb.std is not None:
            if not math.isclose(qa.std, qb.std, rel_tol=rel_tol, abs_tol=1e-12):
                return False, "std_mismatch"

    return True, None


setattr(validate_numeric_quantity, "compare_values", compare_numeric_quantity_values)


def _safe_pow(base: Any, exp: Any) -> Any:
    if isinstance(exp, (int, float)) and abs(float(exp)) > _SAFE_COMPUTE_MAX_EXP:
        raise ValueError(f"exponent {exp} exceeds safe limit {_SAFE_COMPUTE_MAX_EXP}")
    return base**exp


def _coerce_variable_value(val: Any) -> Any:
    """Coerce plain numeric strings (including decimal comma) to float for compute expressions."""
    if isinstance(val, bool) or val is None:
        return val
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(float(val)) else val
    if isinstance(val, str):
        parsed, err = parse_numeric_quantity(val)
        if err is None and parsed is not None and parsed.kind == "point" and parsed.relation == "=":
            if parsed.std is None and parsed.value is not None:
                return parsed.value
        return val.strip()
    return val


def validate_compute_syntax(expression: str) -> Optional[str]:
    """Validate expression length and AST safety without evaluating it."""
    if not isinstance(expression, str) or not expression.strip():
        return "compute validator expression must be a non-empty string"
    if len(expression) > _SAFE_COMPUTE_MAX_EXPR_LEN:
        return f"compute expression exceeds max length ({_SAFE_COMPUTE_MAX_EXPR_LEN})"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"invalid compute expression syntax: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            return f"forbidden AST node {type(node).__name__} in compute expression"
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return f"forbidden identifier {node.id!r} in compute expression"
    return None


def evaluate_compute_expression(expression: str, variables: Dict[str, Any]) -> Any:
    """Safely evaluate an expression using simpleeval with restricted AST and math functions."""
    syntax_err = validate_compute_syntax(expression)
    if syntax_err:
        raise ValueError(syntax_err)

    try:
        import simpleeval
    except ImportError as exc:
        raise RuntimeError(
            "simpleeval is required for 'compute_validators' (`pip install simpleeval`)."
        ) from exc

    operators = simpleeval.DEFAULT_OPERATORS.copy()
    operators[ast.Pow] = _safe_pow
    functions = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "log10": math.log10,
        "log": math.log,
        "log2": math.log2,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "pow": _safe_pow,
        "float": float,
        "int": int,
        "bool": bool,
        "isnan": math.isnan,
        "isfinite": math.isfinite,
    }
    safe_names = {
        str(k): _coerce_variable_value(v)
        for k, v in variables.items()
        if isinstance(k, str) and not k.startswith("_")
    }
    evaluator = simpleeval.SimpleEval(operators=operators, functions=functions, names=safe_names)
    return evaluator.eval(expression)


def _record_to_dict(record: Any) -> Dict[str, Any]:
    if record is None:
        return {}
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "model_dump"):
        return dict(record.model_dump())
    if hasattr(record, "__dict__"):
        return dict(vars(record))
    return {}


def validate_compute_expression(
    value: Any,
    *,
    expression: Optional[str] = None,
    expr: Optional[str] = None,
    when: Optional[str] = None,
    record: Optional[Any] = None,
    **_: Any,
) -> FieldValidationResult:
    """Evaluate a field or cross-column invariant expression via simpleeval."""
    target_expr = expression or expr
    if not target_expr:
        return FieldValidationResult(
            valid=False,
            value=value,
            error="compute_validators requires an 'expression' parameter",
        )

    variables = _record_to_dict(record)
    variables["value"] = value
    variables["val"] = value
    variables["x"] = value

    try:
        if when:
            if not bool(evaluate_compute_expression(str(when), variables)):
                return FieldValidationResult(valid=True, value=value)
        outcome = evaluate_compute_expression(str(target_expr), variables)
    except Exception as exc:
        return FieldValidationResult(
            valid=False,
            value=value,
            error=f"compute validator error for {target_expr!r}: {exc}",
        )

    if not bool(outcome):
        return FieldValidationResult(
            valid=False,
            value=value,
            error=f"compute invariant failed: {target_expr}",
        )
    return FieldValidationResult(valid=True, value=value)


def validate_record_compute_validators(
    record: Any,
    specs: Sequence[Union[str, Dict[str, Any], FieldValidatorSpec]],
) -> List[FieldValidationResult]:
    """Run a list of cross-column compute validators on a single record."""
    results: List[FieldValidationResult] = []
    rec_dict = _record_to_dict(record)
    for raw_spec in specs:
        spec = normalize_validator_spec(raw_spec)
        res = validate_compute_expression(
            rec_dict,
            expression=spec.params.get("expression") or spec.params.get("expr"),
            when=spec.params.get("when"),
            record=rec_dict,
        )
        if not res.valid and spec.message:
            res = FieldValidationResult(valid=False, value=res.value, error=spec.message)
        results.append(res)
    return results
