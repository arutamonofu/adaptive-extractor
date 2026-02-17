"""Converters for nanozyme data.

This module provides functions to convert between different data formats,
particularly for loading ground truth data from CSV files.
"""

import logging
from typing import Any, List, Literal, Optional, Type

import pandas as pd

from .models import VALID_ACTIVITIES, NanozymeExperiment

logger = logging.getLogger(__name__)


def _get_value(
    row: pd.Series,
    key: str,
    type_cast: Type = str,
    alt_keys: Optional[List[str]] = None,
) -> Any:
    """Safely extract and convert a value from a pandas Series.

    Args:
        row: Pandas Series to extract from
        key: Primary key to look up
        type_cast: Type to cast the value to
        alt_keys: Alternative keys to try if primary key is not found

    Returns:
        Converted value or None if not found/invalid
    """
    # Try primary key first
    val = row.get(key)

    # Try alternative keys if primary not found or empty
    if (pd.isna(val) or val == "") and alt_keys:
        for alt in alt_keys:
            val = row.get(alt)
            if not (pd.isna(val) or val == ""):
                break

    # Check for explicit null values
    if pd.isna(val) or str(val).strip().lower() in ("nan", "", "none"):
        return None

    # Attempt type conversion
    try:
        return type_cast(val)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to cast '{val}' to {type_cast.__name__}: {e}")
        return None


def row_to_nanozyme(row: pd.Series) -> Optional[NanozymeExperiment]:
    """Convert a pandas Series row to a NanozymeExperiment object.

    This function is used to load ground truth data from CSV files.
    It handles various field name variations and performs validation.

    Args:
        row: Pandas Series containing experiment data

    Returns:
        NanozymeExperiment object or None if required fields are missing
    """
    # Extract required formula field
    formula = _get_value(row, "formula")
    if not formula:
        logger.debug("Missing required 'formula' field")
        return None

    # Process activity with validation
    raw_activity = str(_get_value(row, "activity", str) or "").lower()
    activity: Literal[
        "peroxidase",
        "oxidase",
        "catalase",
        "laccase",
        "superoxide_dismutase",
        "glucose oxidase",
        "other",
    ] = "other"

    for valid_activity in VALID_ACTIVITIES:
        if valid_activity in raw_activity:
            activity = valid_activity  # type: ignore
            break

    # Create experiment object with all available fields
    try:
        return NanozymeExperiment(
            formula=formula,
            surface=_get_value(row, "surface"),
            syngony=_get_value(row, "syngony"),
            length=_get_value(row, "length", float),
            width=_get_value(row, "width", float),
            depth=_get_value(row, "depth", float),
            activity=activity,
            reaction_type=_get_value(row, "reaction_type"),
            km_value=_get_value(row, "km_val", float, ["km_value"]),
            km_unit=_get_value(row, "km_unit"),
            vmax_value=_get_value(row, "vmax_value", float, ["vmax_val"]),
            vmax_unit=_get_value(row, "vmax_unit"),
            ph=_get_value(row, "ph", float),
            temperature=_get_value(row, "temp", float, ["temperature"]),
            c_min=_get_value(row, "c_min", float),
            c_max=_get_value(row, "c_max", float),
            c_const=_get_value(row, "c_cons", float, ["c_const"]),
            c_const_unit=_get_value(row, "c_cons_unit", str, ["c_const_unit"]),
            ccat_value=_get_value(row, "ccat_value", float),
            ccat_unit=_get_value(row, "ccat_unit"),
        )
    except Exception as e:
        logger.error(f"Failed to create NanozymeExperiment: {e}")
        return None
