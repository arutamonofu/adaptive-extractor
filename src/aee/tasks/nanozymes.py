# src/aee/tasks/nanozymes.py

from typing import List, Optional, Literal, Any, Type
import pandas as pd
import dspy
from pydantic import BaseModel, Field

# --- 1. Data Schema ---

class NanozymeExperiment(BaseModel):
    """
    Structured representation of a single nanozyme kinetic experiment.
    """
    # Material Properties
    formula: str = Field(..., description="Chemical formula (e.g., Fe3O4).")
    surface: Optional[str] = Field(None, description="Surface modification (e.g., PEG, naked).")
    syngony: Optional[str] = Field(None, description="Crystal structure.")
    
    # Dimensions
    length_nm: Optional[float] = Field(None, description="Length in nm.")
    width_nm: Optional[float] = Field(None, description="Width in nm.")
    depth_nm: Optional[float] = Field(None, description="Depth in nm.")

    # Activity & Reaction
    activity: Literal["peroxidase", "oxidase", "catalase", "laccase", "superoxide_dismutase", "glucose oxidase", "other"] = Field(
        ..., description="Type of catalytic activity."
    )
    reaction_type: Optional[str] = Field(None, description="Substrate pair (e.g., 'TMB + H2O2').")

    # Kinetic Parameters
    km_value: Optional[float] = Field(None, description="Michaelis constant (Km).")
    km_unit: Optional[str] = Field(None, description="Unit for Km.")
    vmax_value: Optional[float] = Field(None, description="Maximum velocity (Vmax).")
    vmax_unit: Optional[str] = Field(None, description="Unit for Vmax.")

    # Experimental Conditions
    ph: Optional[float] = Field(None, description="pH level.")
    temperature: Optional[float] = Field(None, description="Temperature in °C.")
    
    # Substrate Concentrations
    c_min: Optional[float] = Field(None, description="Min concentration of variable substrate.")
    c_max: Optional[float] = Field(None, description="Max concentration of variable substrate.")
    c_const: Optional[float] = Field(None, description="Concentration of fixed co-substrate.")

class NanozymeExtractionOutput(BaseModel):
    """Container for extracted experiments."""
    experiments: List[NanozymeExperiment] = Field(default_factory=list)


# --- 2. DSPy Signature ---

class NanozymeSignature(dspy.Signature):
    """
    Extract structured kinetic data from scientific text about nanozymes.
    
    CRITICAL INSTRUCTIONS:
    1.  **Reaction Logic**: Determine the reaction type based on variable vs constant substrates.
        - "H2O2 + TMB": H2O2 varies, TMB is constant.
        - "TMB + H2O2": TMB varies, H2O2 is constant.
    2.  **Scope**: Extract separate entries for distinct kinetic tracks. 
    3.  **Missing Data**: Use null for unspecified values. Do not hallucinate.
    """

    document_text: str = dspy.InputField(
        desc="Full Markdown text of the article, including tables."
    )
    extracted_data: NanozymeExtractionOutput = dspy.OutputField(
        desc="JSON list of valid nanozyme experiments."
    )


# --- 3. Utilities ---

def row_to_nanozyme(row: pd.Series) -> NanozymeExperiment:
    """
    Converts a Pandas CSV row into a NanozymeExperiment model.
    Handles NaN values and type casting.
    """
    def _safe_get(key: str, cast_type: Type = str) -> Any:
        val = row.get(key)
        if pd.isna(val) or str(val).lower() == "nan" or val == "":
            return None
        try:
            return cast_type(val)
        except (ValueError, TypeError):
            return None

    return NanozymeExperiment(
        formula=_safe_get("formula") or "Unknown",
        surface=_safe_get("surface"),
        syngony=_safe_get("syngony"),
        length_nm=_safe_get("length", float),
        width_nm=_safe_get("width", float),
        depth_nm=_safe_get("depth", float),
        activity=_safe_get("activity") or "other",
        reaction_type=_safe_get("reaction_type"),
        km_value=_safe_get("km_value", float),
        km_unit=_safe_get("km_unit"),
        vmax_value=_safe_get("vmax_value", float),
        vmax_unit=_safe_get("vmax_unit"),
        ph=_safe_get("ph", float),
        temperature=_safe_get("temperature", float),
        c_min=_safe_get("c_min", float),
        c_max=_safe_get("c_max", float),
        c_const=_safe_get("c_const", float),
    )


# --- 4. Export ---

task_config = {
    "name": "nanozymes",
    "description": "Extraction of kinetic parameters for nanozymes",
    "signature": NanozymeSignature,
    "output_model": NanozymeExtractionOutput,
    "row_converter": row_to_nanozyme,
    # Fields used for calculating F1/Precision/Recall matches
    "compare_fields": [
        "formula", "activity", "reaction_type",
        "km_value", "vmax_value", "ph", "temperature"
    ]
}