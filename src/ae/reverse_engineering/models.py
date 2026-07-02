# src/ae/re/models.py
"""Pydantic data models for structured LLM outputs in the RE pipeline."""

from typing import Literal
from pydantic import BaseModel

# --- Phase 1: Positive Analysis ---

class RowAnalysis(BaseModel):
    row_id: str
    Row_Instantiation_Trigger_class: Literal["Single_Entity", "Composite_Vector"]
    Row_Instantiation_Trigger_description: str
    Entity_System_Role_class: Literal["Target", "Reference", "Background"]
    Entity_System_Role_description: str
    Entity_Filtration_Boundary_class: Literal[
        "Exhaustive_Set", "Extremum_Only", "Baseline_Only", "Taxonomic_Class"
    ]
    Entity_Filtration_Boundary_description: str

class PositiveRowsOutput(BaseModel):
    rows: list[RowAnalysis]


class FieldAnalysis(BaseModel):
    field_name: str
    raw_text: str
    ground_truth: str
    Semantic_Core: str
    Analytical_Method: str          # or "Not_Specified"
    System_Conditions: str          # or "Not_Specified"
    Hierarchy_Level_class: Literal["Macro_System", "Micro_Component", "External_Environment"]
    Hierarchy_Level_description: str
    Semantic_Binding_class: Literal["Syntax_Direct", "Context_Inherited", "Absent_In_Text"]
    Semantic_Binding_description: str
    Text_Precision_class: Literal["Exact_Value", "Fuzzy_Or_Range"]
    Text_Precision_description: str
    Value_Multiplicity_class: Literal["Single_Instance", "Multiple_Instances"]
    Value_Multiplicity_description: str
    Transformation_description: str

class PositiveColumnsOutput(BaseModel):
    fields: list[FieldAnalysis]


# --- Phase 2: Positive Consolidation ---

class ConsolidatedRowsOutput(BaseModel):
    Row_Instantiation_Trigger_generalized: str


class ConsolidatedColumnsOutput(BaseModel):
    Semantic_Core_generalized: str


# --- Phase 3: Negative Analysis ---

class GapRowAnalysis(BaseModel):
    gap_row_id: str
    raw_text: str
    gap_entity_description: str
    Row_Instantiation_Trigger_class: Literal["Single_Entity", "Composite_Vector"]
    Row_Instantiation_Trigger_description: str
    Entity_System_Role_class: Literal["Target", "Reference", "Background"]
    Entity_System_Role_description: str
    Entity_Filtration_Boundary_class: Literal[
        "Exhaustive_Set", "Extremum_Only", "Baseline_Only", "Taxonomic_Class"
    ]
    Entity_Filtration_Boundary_description: str

class NegativeRowsOutput(BaseModel):
    gap_rows: list[GapRowAnalysis]


class CandidateAnalysis(BaseModel):
    raw_text: str
    candidate_value: str
    ground_truth: str
    Semantic_Core: str
    Analytical_Method: str
    System_Conditions: str
    Hierarchy_Level_class: Literal["Macro_System", "Micro_Component", "External_Environment"]
    Hierarchy_Level_description: str
    Semantic_Binding_class: Literal["Syntax_Direct", "Context_Inherited", "Absent_In_Text"]
    Semantic_Binding_description: str
    Text_Precision_class: Literal["Exact_Value", "Fuzzy_Or_Range"]
    Text_Precision_description: str
    Value_Multiplicity_class: Literal["Single_Instance", "Multiple_Instances"]
    Value_Multiplicity_description: str

class NegativeFieldResult(BaseModel):
    field_name: str
    candidates: list[CandidateAnalysis]

class NegativeColumnsOutput(BaseModel):
    fields: list[NegativeFieldResult]


# --- Phase 4: Generalization ---

class Anomaly(BaseModel):
    source_reference: str
    anomaly_description: str

class RowInstructions(BaseModel):
    Row_Inclusion_Instructions: list[str]
    Row_Exclusion_Instructions: list[str]
    Validation_Rationale: str

class GeneralizedRowsOutput(BaseModel):
    Instructions: RowInstructions
    Anomalies: list[Anomaly]


class ColumnInstructions(BaseModel):
    Column_Inclusion_Instructions: list[str]
    Column_Exclusion_Instructions: list[str]
    Transformation_Instructions: list[str]

class SourceReference(BaseModel):
    document_id: str
    positive_reference_ids: list[str]
    negative_reference_ids: list[str]

class ColumnAnomaly(BaseModel):
    source_reference: SourceReference
    anomaly_description: str

class GeneralizedColumnsOutput(BaseModel):
    Instructions: ColumnInstructions
    Anomalies: list[ColumnAnomaly]



