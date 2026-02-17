"""Nanozyme experiment data models.

This module defines the Pydantic models for nanozyme experiments,
including the experiment structure and extraction output format.
"""

import logging
import re
from typing import Annotated, Any, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field

from aee.domain.entities import Experiment

logger = logging.getLogger(__name__)

# Constants for validation
VALID_ACTIVITIES = [
    "peroxidase",
    "oxidase",
    "catalase",
    "laccase",
    "superoxide_dismutase",
    "glucose oxidase",
]


def clean_number(v: Any) -> Optional[float]:
    """Convert various input types to a float, handling common formatting issues.

    Args:
        v: Input value to convert (None, numeric, or string)

    Returns:
        Converted float value or None if conversion fails
    """
    if v is None:
        return None

    if isinstance(v, (float, int)):
        return float(v)

    if isinstance(v, str):
        # Normalize different dash characters to standard minus
        s = v.replace("−", "-").replace("–", "-").replace("—", "-").strip()

        # Fix scientific notation formatting (e.g., "10 -5" -> "10-5")
        s = re.sub(r"(?i)10\s*[-]\s*(\d+)", r"10-\1", s)

        # Extract the first valid floating point number
        pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
        match = re.search(pattern, s)

        if match:
            try:
                return float(match.group())
            except (ValueError, TypeError):
                logger.debug(f"Failed to convert '{match.group()}' to float")
                return None

    return None


# Type alias for robust numeric fields
RobustFloat = Annotated[Optional[float], BeforeValidator(clean_number)]


class NanozymeExperiment(Experiment):
    """Represents a single nanozyme experiment with all relevant parameters.

    This model defines the structure for nanozyme experiments, including
    material properties, catalytic activity, and kinetic parameters.
    """

    formula: str = Field(..., description="Chemical formula (e.g., Fe3O4, Au, CuO).")
    surface: Optional[str] = Field(
        None,
        description="Surface chemistry (naked, PEG, PVP, citrate), Polymer (oleic acid, BSA), or Surfactant (l-ascorbic acid, sodium citrate, etc).",
    )
    syngony: Optional[str] = Field(
        None,
        description="Crystal system: cubic, hexagonal, tetragonal, monoclinic, orthorhombic, trigonal, amorphous, triclinic.",
    )
    length: RobustFloat = Field(None, description="Length or size/diameter in nm.")
    width: RobustFloat = Field(None, description="Width in nm (if applicable).")
    depth: RobustFloat = Field(None, description="Depth in nm (if applicable).")
    activity: Literal[
        "peroxidase",
        "oxidase",
        "catalase",
        "laccase",
        "superoxide_dismutase",
        "glucose oxidase",
        "other",
    ] = Field(..., description="Type of catalytic activity.")
    reaction_type: Optional[str] = Field(
        None,
        description="Format: 'Substrate + Co-substrate'. Tracks: TMB+H2O2 (TMB varies), H2O2+TMB (H2O2 varies).",
    )
    km_value: RobustFloat = Field(None, description="Michaelis constant Km (Mantissa).")
    km_unit: Optional[str] = Field(None, description="Unit for Km (e.g., mM).")
    vmax_value: RobustFloat = Field(
        None, description="Max reaction rate Vmax (Mantissa)."
    )
    vmax_unit: Optional[str] = Field(
        None, description="Unit for Vmax (e.g., M/s, mM/s)."
    )
    ph: RobustFloat = Field(None, description="pH level.")
    temperature: RobustFloat = Field(None, description="Temperature in °C.")
    c_min: RobustFloat = Field(
        None, description="Min concentration of variable substrate (mM)."
    )
    c_max: RobustFloat = Field(
        None, description="Max concentration of variable substrate (mM)."
    )
    c_const: RobustFloat = Field(
        None, description="Concentration of fixed co-substrate (mM)."
    )
    c_const_unit: Optional[str] = Field(None, description="Unit for c_const.")
    ccat_value: RobustFloat = Field(
        None, description="Concentration of the nanozyme/catalyst."
    )
    ccat_unit: Optional[str] = Field(
        None, description="Unit for catalyst conc (e.g., mcg/ml, mg/mL)."
    )


class NanozymeExtractionOutput(BaseModel):
    """Container for extracted nanozyme experiments.

    This model wraps a list of extracted nanozyme experiments and is used
    as the output type for the DSPy signature.
    """

    experiments: List[NanozymeExperiment] = Field(default_factory=list)
