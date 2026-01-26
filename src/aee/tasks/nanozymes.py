# src/aee/tasks/nanozymes.py

import logging
import re
from typing import Annotated, Any, List, Literal, Optional, Type

import dspy
import pandas as pd
from pydantic import BaseModel, BeforeValidator, Field

from aee.core.config import settings

logger = logging.getLogger(__name__)

def clean_number(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (float, int)):
        return float(v)

    if isinstance(v, str):
        s = v.replace("−", "-").replace("–", "-").replace("—", "-").strip()
        s = re.sub(r'(?i)10\s*[-]\s*(\d+)', r'10-\1', s)
        pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
        match = re.search(pattern, s)
        if match:
            try:
                return float(match.group())
            except (ValueError, TypeError):
                return None
    return None

RobustFloat = Annotated[Optional[float], BeforeValidator(clean_number)]

class NanozymeExperiment(BaseModel):    
    formula: str = Field(..., description="Chemical formula (e.g., Fe3O4, Au, CuO).")
    surface: Optional[str] = Field(None, description="Surface chemistry (naked, PEG, PVP, citrate), Polymer (olelic acid, BSA), or Surfactant (l-ascorbic acid, sodium citrate, etc).")
    syngony: Optional[str] = Field(None, description="Crystal system: cubic, hexagonal, tetragonal, monoclinic, orthorhombic, trigonal, amorphous, triclinic.")
    length: RobustFloat = Field(None, description="Length or size/diameter in nm.")
    width: RobustFloat = Field(None, description="Width in nm (if applicable).")
    depth: RobustFloat = Field(None, description="Depth in nm (if applicable).")
    activity: Literal["peroxidase", "oxidase", "catalase", "laccase", "superoxide_dismutase", "glucose oxidase", "other"] = Field(..., description="Type of catalytic activity.")
    reaction_type: Optional[str] = Field(None, description="Format: 'Substrate + Co-substrate'. Tracks: TMB+H2O2 (TMB varies), H2O2+TMB (H2O2 varies).")
    km_value: RobustFloat = Field(None, description="Michaelis constant Km (Mantissa).")
    km_unit: Optional[str] = Field(None, description="Unit for Km (e.g., mM).")
    vmax_value: RobustFloat = Field(None, description="Max reaction rate Vmax (Mantissa).")
    vmax_unit: Optional[str] = Field(None, description="Unit for Vmax (e.g., M/s, mM/s).")
    ph: RobustFloat = Field(None, description="pH level.")
    temperature: RobustFloat = Field(None, description="Temperature in °C.")
    c_min: RobustFloat = Field(None, description="Min concentration of variable substrate (mM).")
    c_max: RobustFloat = Field(None, description="Max concentration of variable substrate (mM).")
    c_const: RobustFloat = Field(None, description="Concentration of fixed co-substrate (mM).")
    c_const_unit: Optional[str] = Field(None, description="Unit for c_const.")
    ccat_value: RobustFloat = Field(None, description="Concentration of the nanozyme/catalyst.")
    ccat_unit: Optional[str] = Field(None, description="Unit for catalyst conc (e.g., mcg/ml, mg/mL).")

class NanozymeExtractionOutput(BaseModel):
    experiments: List[NanozymeExperiment] = Field(default_factory=list)


class NanozymeSignature(dspy.Signature):
    """
    You are helpful assistant in chemistry, specializing in nanozymes. Your task is to analyze scientific articles and extract detailed information about various experiments with nanozymes. It is crucial for you to accurately and comprehensively describe each experiment separately, without referring to other experiments in the article.

    Usually, the articles contain several experiments with nanozymes with different parameters, such as:
    - Formula (e.g. Fe3O4)
    - Activity (usually peroxidase, oxidase, catalase or laccase)
    - Syngony (usually cubic, hexagonal, tetragonal, monoclinic, orthorhombic, trigonal, amorphous or triclinic)
    - Length, width and depth (or just size or diameter)
    - Surface chemistry (naked by default or poly(ethylene oxide), poly(N-Vinylpyrrolidone), Tetrakis(4-carboxyphenyl)porphine or other)
    - Polymer used in synthesis (none or poly(N-Vinylpyrrolidone), oleic acid, poly(ethylene oxide), BSA or other)
    - Surfactant (none or l-ascorbic acid, ethylene glycol, sodium citrate, cetrimonium bromide, citric acid, trisodium citrate, ascorbic acid or other)
    - Molar mass, Michaelis constant Km, molar maximum reaction rate Vmax
    - Reaction type (substrat + co-substrat) (TMB + H2O2, H2O2 + TMB, TMB, ABTS + H2O2, H2O2, OPD + H2O2, H2O2 + GSH or other)
    - Minimum concentration of the substrate when measuring catalytic activity C_min (mM)
    - Maximum concentration of the substrate when measuring catalytic activity C_max (mM)
    - Concentration of the co-substrate when measuring the catalytic activity (mM)
    - Concentration of nanoparticles in the measurement of catalytic (mg/mL)
    - pH at which the catalytic activity was measured and temperature at which the research was carried out (°C).

    You need to find all the experiments with different values mentioned in the article and extraction them as separate objects. It's imperative that each of these elements is addressed independently for every experiment, providing a complete and isolated description with accurate measurements in appropriate units. This approach will ensure a comprehensive and clear understanding of each experiment as an individual entity within the scientific literature on nanozymes.

    CRITICAL RULES FOR EXTRACTION:
    1. Keep the numerical values in the right units of measurement. It is critically important to extract all the numerical values as in the example, especially important are formula, activity, syngony, length, width, depth (size or diameter), Km, Vmax, reaction type.
    2. Usually such parameters as Michaelis constant Km (mM), Vmax, mM/s are obtained in two experiments for every type of nanoparticle. You must determine what type of reaction such parameters as Michaelis constant Km (mM), Vmax, mM/s belong to.
    
    LOGIC FOR REACTION TYPES (TRACKS):
    - Reaction type is H2O2+TMB when H2O2 is a substrate and TMB in co-cubstrate.
    - Reaction type is TMB +H2O2 when TMB is a substrate and H2O2 in co-cubstrate.
    
    For example in pair H2O2 and TMB:
    - In first case (you call this case as Reaction type TMB+H2O2): H2O2 plays role as a co-substrate with constant concentration(C(const), mM) and TMB plays role as a substrate with concentrations from Cmin,mM to Cmax, mM.
    - In second case (you call this case as Reaction type H2O2+TMB): TMB plays role as a co-substrate with constant concentration(C(const), mM) and H2O2 plays role as a substrate with concentrations from Cmin,mM to Cmax, mM.
    
    Please divide all the data into 2 tracks: where H2O2 was a substrate and its concentration varied and where H2O2 was a co-substrate and had a constant concentration. 
    Please show data only for those nanoparticles for which the kinetic assay was performed. All other parameters from the example are also important.
    Do not attempt to use knowledge you already have.
    """

    document_text: str = dspy.InputField(
        desc="Full text content of the scientific article."
    )
    extracted_data: NanozymeExtractionOutput = dspy.OutputField(
        desc="A list of structured experiments matching the schema."
    )

def row_to_nanozyme(row: pd.Series) -> Optional[NanozymeExperiment]:
    def _get(key: str, type_cast: Type = str, alt_keys: List[str] = None) -> Any:
        val = row.get(key)
        if (pd.isna(val) or val == "") and alt_keys:
            for alt in alt_keys:
                val = row.get(alt)
                if not (pd.isna(val) or val == ""):
                    break
        if pd.isna(val) or str(val).strip().lower() in ("nan", "", "none"):
            return None
        try: return type_cast(val)
        except: return None

    formula = _get("formula")
    if not formula: return None

    raw_activity = str(_get("activity")).lower()
    valid_activities = ["peroxidase", "oxidase", "catalase", "laccase", "superoxide_dismutase", "glucose oxidase"]
    activity = next((a for a in valid_activities if a in raw_activity), "other")

    return NanozymeExperiment(
        formula=formula,
        surface=_get("surface"),
        syngony=_get("syngony"),
        length=_get("length", float),
        width=_get("width", float),
        depth=_get("depth", float),
        activity=activity,
        reaction_type=_get("reaction_type"),
        km_value=_get("km_val", float, ["km_value"]),
        km_unit=_get("km_unit"),
        vmax_value=_get("vmax_value", float, ["vmax_val"]),
        vmax_unit=_get("vmax_unit"),
        ph=_get("ph", float),
        temperature=_get("temp", float, ["temperature"]),
        c_min=_get("c_min", float),
        c_max=_get("c_max", float),
        c_const=_get("c_cons", float, ["c_const"]),
        c_const_unit=_get("c_cons_unit", str, ["c_const_unit"]),
        ccat_value=_get("ccat_value", float),
        ccat_unit=_get("ccat_unit"),
    )

task_config = {
    "name": "nanozymes",
    "signature": NanozymeSignature,
    "output_model": NanozymeExtractionOutput,
    "row_converter": row_to_nanozyme,
    "compare_fields": settings.task.evaluation.compare_fields
}