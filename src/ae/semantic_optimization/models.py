# src/ae/reverse_engineering/models.py
"""Pydantic data models for structured LLM outputs in the RE pipeline."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

# --- Conflict & HITL Models ---

ConflictType = Literal[
    "CROSS_DOC_CONTRADICTION",
    "CONDITIONAL_BRANCH_NEEDED",
    "TEXT_VS_GT_MISMATCH",
    "SUSPECTED_GT_ERROR",
    "SUSPECTED_GT_OMISSION",
    "SUSPECTED_INGESTION_DEFECT",
    "BASELINE_POLICY_CONFLICT",
]

ActionType = Literal[
    "GENERALIZE",
    "ADD_CONDITION",
    "OVERRIDE_BASELINE",
    "KEEP_BASELINE",
    "PATCH_GT_CELL",
    "PATCH_GT_DROP_ROW",
    "PATCH_GT_ADD_ROW",
    "PATCH_INGESTED_MD",
    "UNPACK_VISUAL_IN_MD",
    "IGNORE_INGESTION_DEFECT",
    "IGNORE_CASE",
]


class ResolutionOption(BaseModel):
    action_type: ActionType
    label: str
    proposed_rule_or_value: str = ""


class REConflict(BaseModel):
    conflict_id: str
    scope: str  # "rows" or "column:<field_name>"
    conflict_type: ConflictType
    document_id: str
    row_id: str
    raw_text_quote: str = ""
    gt_value: str = ""
    conflicting_value_or_rule: str = ""
    explanation: str
    options: list[ResolutionOption] = Field(default_factory=list)


class ExpertRuling(BaseModel):
    conflict_id: str
    scope: str
    conflict_type: ConflictType
    action_type: ActionType
    decision_text: str
    document_id: str = ""
    row_id: str = ""
    field_name: str = ""


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
    rows: list[RowAnalysis] = Field(default_factory=list)


class FieldAnalysis(BaseModel):
    row_id: str = Field(default="row_1", description="Identifier of the target row (e.g., row_1)")
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
    Suspected_Ingestion_Or_GT_Issue: str = Field(
        default="None",
        description="Factual mismatch between text and GT due to potential OCR/VLM hallucination or GT typo"
    )


class PositiveColumnsOutput(BaseModel):
    fields: list[FieldAnalysis] = Field(default_factory=list)


# --- Phase 2: Positive Consolidation ---

class ConsolidatedRowsOutput(BaseModel):
    Row_Instantiation_Trigger_generalized: str
    Entity_Role_and_Filtration_Policy: str = ""
    Conflicts: list[REConflict] = Field(default_factory=list)


class ConsolidatedColumnsOutput(BaseModel):
    Semantic_Core_generalized: str
    Analytical_and_Binding_Policy: str = ""
    Conflicts: list[REConflict] = Field(default_factory=list)


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
    gap_rows: list[GapRowAnalysis] = Field(default_factory=list)


class CandidateAnalysis(BaseModel):
    row_id: str = Field(default="row_1", description="Identifier of the target row (e.g., row_1)")
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
    row_id: str = Field(default="row_1", description="Identifier of the target row (e.g., row_1)")
    field_name: str
    candidates: list[CandidateAnalysis] = Field(default_factory=list)


class NegativeColumnsOutput(BaseModel):
    fields: list[NegativeFieldResult] = Field(default_factory=list)


# --- Phase 4: Generalization / Unified Rule Synthesis ---

class Anomaly(BaseModel):
    source_reference: str
    anomaly_description: str


class RowInstructions(BaseModel):
    Row_Inclusion_Instructions: list[str] = Field(default_factory=list)
    Row_Exclusion_Instructions: list[str] = Field(default_factory=list)
    Validation_Rationale: str = ""


class GeneralizedRowsOutput(BaseModel):
    Row_Instantiation_Trigger_generalized: str = Field(
        default="",
        description="Consolidated row creation trigger synthesized across active documents",
    )
    Entity_Role_and_Filtration_Policy: str = Field(
        default="",
        description="Consolidated policy for target vs reference entities and filtration boundaries",
    )
    Instructions: RowInstructions
    Anomalies: list[Anomaly] = Field(default_factory=list)
    Conflicts: list[REConflict] = Field(default_factory=list)


class ColumnInstructions(BaseModel):
    Column_Inclusion_Instructions: list[str] = Field(default_factory=list)
    Column_Exclusion_Instructions: list[str] = Field(default_factory=list)
    Transformation_Instructions: list[str] = Field(default_factory=list)


class SourceReference(BaseModel):
    document_id: str
    positive_reference_ids: list[str]
    negative_reference_ids: list[str]


class ColumnAnomaly(BaseModel):
    source_reference: SourceReference
    anomaly_description: str


class GeneralizedColumnsOutput(BaseModel):
    Semantic_Core_generalized: str = Field(
        default="",
        description="Consolidated semantic core for the column across active documents",
    )
    Analytical_and_Binding_Policy: str = Field(
        default="",
        description="Consolidated policy for analytical methods, hierarchy level, and binding",
    )
    Instructions: ColumnInstructions
    Anomalies: list[ColumnAnomaly] = Field(default_factory=list)
    Conflicts: list[REConflict] = Field(default_factory=list)


class PromptHarmonizationOutput(BaseModel):
    harmonized_prompt: str = Field(
        description="Complete harmonized extraction prompt preserving all field sections, cross-column dependencies, and non-conflicting baseline instructions"
    )
    harmonization_notes: list[str] = Field(
        default_factory=list,
        description="List of cross-column alignments, deduplications, and baseline integrations applied",
    )


# --- Phase 5: Schema Constraint Inference ---

class FieldConstraintProposal(BaseModel):
    field_name: str
    description: str = Field(
        default="",
        description="Concise schema-level field description suitable for JSON Schema",
    )
    is_closed_enum: bool = Field(
        default=False,
        description="True ONLY if the field has a finite closed vocabulary of discrete categories/operators/labels",
    )
    choices: list[str] = Field(
        default_factory=list,
        description="Exhaustive valid string choices when is_closed_enum is True",
    )
    default_value: Optional[str] = Field(
        default=None,
        description="Explicit default value ONLY when the prompt/rules define a deterministic default for absent values",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Conservative regex pattern ONLY for strictly invariant syntactic formats; otherwise null",
    )
    min_value: Optional[float] = Field(
        default=None,
        description="Hard ontological minimum for numeric fields; otherwise null",
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Hard ontological maximum for numeric fields; otherwise null",
    )
    rationale: str = ""

    @field_validator("choices", mode="before")
    @classmethod
    def _coerce_choices(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            stripped = v.strip()
            return [stripped] if stripped else []
        return v

    @field_validator("description", "rationale", mode="before")
    @classmethod
    def _coerce_empty_str(cls, v: Any) -> Any:
        return "" if v is None else v

    @field_validator("is_closed_enum", mode="before")
    @classmethod
    def _coerce_is_closed_enum(cls, v: Any) -> Any:
        return False if v is None else v


class SchemaRefinementOutput(BaseModel):
    fields: list[FieldConstraintProposal] = Field(default_factory=list)


# --- Phase 6: Residual Error-Driven Analysis ---

class ResidualColumnsOutput(BaseModel):
    positive_fields: list[FieldAnalysis] = Field(default_factory=list)
    negative_fields: list[NegativeFieldResult] = Field(default_factory=list)


class ResidualDocReport(BaseModel):
    doc_id: str
    mode: Literal["residual", "zero_delta_skip", "alignment_fallback"]
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    pred_count: int = 0
    gt_count: int = 0
    tp_rows: int = 0
    fn_rows: int = 0
    fp_rows: int = 0
    mismatch_pairs: int = 0
    analyzed_fn_row_ids: list[str] = Field(default_factory=list)
    analyzed_mismatch_row_ids: list[str] = Field(default_factory=list)
    grouped_signatures: dict[str, int] = Field(default_factory=dict)
    affected_scopes: list[str] = Field(default_factory=list)
    fallback_reason: str = ""
    input_hash: str = ""
