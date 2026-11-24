"""Pydantic schemas for all stages of Ingestion v2 Gemini pipeline."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

# ---------------------------------------------------------------------------
# Stage 1: Transcription Placeholders
# ---------------------------------------------------------------------------


class PlaceholderItem(BaseModel):
    """Visual placeholder identified in the Markdown text."""

    id: str = Field(description="Unique placeholder ID, e.g. 'Figure_3', 'Unlabeled_p1_1'")
    caption: str = Field(
        default="",
        description="Original caption paragraph following the visual",
    )
    page: int = Field(default=1, description="1-based PDF page number where visual occurs")
    original_tag: str = Field(default="", description="Original raw placeholder tag")


# ---------------------------------------------------------------------------
# Stage 2: Manifest & Classification
# ---------------------------------------------------------------------------

VisualType = Literal["table", "chart", "molecular_structure", "mixed"]


class ManifestVisual(BaseModel):
    """A whole visual manifested during Ingestion (Stage 3)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Exact visual ID, e.g. 'Figure_3', 'Scheme_1', 'Table_2'")
    type: VisualType = Field(description="Primary visual type or 'mixed'")
    pages: list[int] = Field(
        default_factory=lambda: [1],
        description="1-based page numbers where visual occurs",
    )
    description: str = Field(
        default="",
        description="Informative content sketch (compound codes, columns, axes, reactions)",
    )
    panel_label: SkipJsonSchema[Optional[str]] = Field(
        default=None,
        exclude=True,
        description="Target element or panel label for focused parsing",
    )


class Manifest(BaseModel):
    """Complete document visual manifest."""

    visuals: list[ManifestVisual] = Field(default_factory=list)
    discarded_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy Decomposition Schemas (Steps T3 - T7)
# ---------------------------------------------------------------------------


class ElementManifestItem(BaseModel):
    """An elementary visual unit of a composite visual."""

    element_id: str = Field(
        validation_alias=AliasChoices("element_id", "atom_id"),
        description="Unique element ID, e.g. 'elem_1', 'mol_4a_4l', 'panel_a'",
    )
    type: Literal["table", "chart", "molecular_structure"] = Field(description="Element modality")
    page_index: int = Field(default=0, description="0-based index of visual crop page fragment")
    label: str = Field(default="", description="Label visible on image, e.g. '1', '3a-3j', '(a)'")
    description: str = Field(description="Guidance for what to tightly include in element bbox")
    box_2d: Optional[list[int]] = Field(
        default=None,
        description="Bounding box [ymin, xmin, ymax, xmax] on the visual crop in normalized 0..1000 coordinates",
    )
    extra_boxes_2d: list[list[int]] = Field(
        default_factory=list,
        description="Additional disjoint [ymin, xmin, ymax, xmax] boxes on the visual crop (e.g. separated R-group legend)",
    )

    @property
    def atom_id(self) -> str:
        return self.element_id


AtomManifestItem = ElementManifestItem


class VisualDecomposition(BaseModel):
    """Decomposition assessment of a cropped visual (Step T3)."""

    mode: Literal["single", "composite"] = Field(
        description=(
            "'single' ONLY if the crop contains strictly one elementary visual unit "
            "(1 molecular/Markush scaffold, 1 chart coordinate system, or 1 pure text table); "
            "'composite' whenever there are >=2 molecular scaffolds, any reaction scheme, "
            "embedded drawings in tables, multi-panel figures, or annotation callout boxes"
        ),
    )
    single_type: Optional[Literal["table", "chart", "molecular_structure"]] = None
    composition_template: Optional[str] = Field(
        default=None,
        description=(
            "Complete narrative/relational skeleton (reaction arrows, conditions, yields, "
            "bioactivity callouts, Markdown pipe table for flat grids, or HTML <table> with "
            "colspan/rowspan for multi-level headers/merged cells) with <element id='...' /> placeholders"
        ),
    )
    elements: list[ElementManifestItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("elements", "atoms"),
    )

    @property
    def atoms(self) -> list[ElementManifestItem]:
        return self.elements


class VisualBox(BaseModel):
    """Bounding box detection for a visual on a page or element on a visual crop."""

    id: str = Field(description="Visual ID matching manifest or element ID")
    box_2d: list[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="[ymin, xmin, ymax, xmax] in normalised 0..1000 coordinates",
    )
    extra_boxes_2d: list[list[int]] = Field(
        default_factory=list,
        description="Additional disjoint [ymin, xmin, ymax, xmax] boxes if the visual has a spatially separated R-group legend or panel",
    )
    found: bool = Field(default=True, description="Whether the visual was successfully localized")


class ElementLocalizationResult(BaseModel):
    """Bounding boxes of elements on a visual crop (Step T4)."""

    boxes: list[VisualBox] = Field(default_factory=list)


AtomLocalizationResult = ElementLocalizationResult


class ElementSelectionResult(BaseModel):
    """LLM response for selecting elements matching a target hint."""

    selected_element_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("selected_element_ids", "selected_atom_ids"),
        description="List of element_id strings strictly relevant to target_hint",
    )

    @property
    def selected_atom_ids(self) -> list[str]:
        return self.selected_element_ids


AtomSelectionResult = ElementSelectionResult


class VisualParseRecord(BaseModel):
    """Final assembled result for a manifested visual (Step T7)."""

    visual_id: str
    status: Literal["success", "partial", "failed"] = "success"
    mode: Literal["single", "composite"] = "single"
    rendered_markdown: str = ""
    parsed_element_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("parsed_element_ids", "parsed_atom_ids"),
    )
    unparsed_element_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("unparsed_element_ids", "unparsed_atom_ids"),
    )
    summary: str = ""
    error_message: Optional[str] = None
    invalid_smiles: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def parsed_atom_ids(self) -> list[str]:
        return self.parsed_element_ids

    @property
    def unparsed_atom_ids(self) -> list[str]:
        return self.unparsed_element_ids


# ---------------------------------------------------------------------------
# Stage 4: Localization & Bounding Boxes
# ---------------------------------------------------------------------------


class PageLocalization(BaseModel):
    """Localization result for a single PDF page image."""

    page: int = Field(description="1-based page number")
    boxes: list[VisualBox] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 6: Extraction Results
# ---------------------------------------------------------------------------


class AttachmentPoint(BaseModel):
    """Mapping between a SMILES dummy atom placeholder and its visual variable/position label."""

    model_config = ConfigDict(extra="ignore")

    placeholder: str = Field(
        description="Dummy atom token as written in smiles, e.g. '[*:1]', '[*:2]'"
    )
    variable: str = Field(
        description=(
            "Exact variable group or position label from the visual or table context "
            "(e.g., 'R', 'R1', 'Ar', 'R (C7)', 'C5 attachment')."
        )
    )


class ChemicalEntity(BaseModel):
    """A chemical structure extracted from a visual."""

    model_config = ConfigDict(extra="ignore")

    label: str = Field(description="Compound identifier on the visual, e.g. '1', '2a', '5f'")
    smiles: str = Field(description="Complete valid RDKit-parseable SMILES of the molecule")
    attachment_points: list[AttachmentPoint] = Field(
        default_factory=list,
        description=(
            "Required whenever smiles contains dummy atom(s) [*:1], [*:2], etc.: "
            "maps each dummy atom placeholder to its variable/position label on the visual."
        ),
    )
    smiles_valid: SkipJsonSchema[bool] = Field(
        default=True, description="RDKit validation outcome"
    )
    rdkit_error: SkipJsonSchema[Optional[str]] = Field(
        default=None, description="RDKit error message if invalid"
    )
    provenance_status: SkipJsonSchema[
        Literal["verified_exact", "unverified_valid", "mismatch_flagged"]
    ] = Field(
        default="unverified_valid",
        description="Cross-modal semantic OCSR oracle verification status",
    )
    oracle_notes: SkipJsonSchema[Optional[str]] = Field(
        default=None,
        description="Diagnostic notes from cross-modal OCSR oracle checks",
    )


class MolecularParseResult(BaseModel):
    """Result of molecular structure parsing."""

    model_config = ConfigDict(extra="ignore")

    status: SkipJsonSchema[Literal["success", "no_extractable_structure", "failed"]] = "success"
    entities: list[ChemicalEntity] = Field(
        default_factory=list,
        description="List of chemical entities (label and canonical SMILES)",
    )

    @property
    def invalid_smiles(self) -> list[dict[str, Any]]:
        """Expose invalid SMILES entries for pipeline diagnostics."""
        return [
            {"label": e.label, "smiles": e.smiles, "error": e.rdkit_error}
            for e in self.entities
            if not e.smiles_valid
        ]


class ChartParseResult(BaseModel):
    """Result of chart parsing."""

    model_config = ConfigDict(extra="ignore")

    status: SkipJsonSchema[Literal["success", "no_extractable_chart", "failed"]] = "success"
    chart_type: str = Field(
        default="",
        description="Plot or spectrum type (e.g. 'bar chart', 'dose-response curve', 'line plot')",
    )
    x_axis: str = Field(default="", description="X-axis variable name and unit")
    y_axis: str = Field(default="", description="Y-axis variable name and unit")
    series_names: list[str] = Field(
        default_factory=list,
        description="Legend series or compound identifiers plotted",
    )
    data_table: str = Field(
        default="",
        description="Digitized chart data formatted as Markdown pipe table(s)",
    )
    annotations: list[str] = Field(
        default_factory=list,
        description="Fitted metrics (IC50, Kd, R^2), significance markers, or panel notes",
    )
    content: SkipJsonSchema[str] = Field(
        default="",
        description="Digitized chart data formatted as Markdown table(s) and bulleted annotations",
    )

    @model_validator(mode="after")
    def _sync_chart_content(self) -> "ChartParseResult":
        if not self.content and (self.data_table or self.annotations):
            parts: list[str] = []
            if self.data_table.strip():
                parts.append(self.data_table.strip())
            if self.annotations:
                notes = "\n".join(
                    f"- {a.strip()}" if not a.strip().startswith(("-", "*")) else a.strip()
                    for a in self.annotations
                    if a and a.strip()
                )
                if notes:
                    parts.append(notes)
            self.content = "\n\n".join(parts).strip()
        elif self.content and not self.data_table:
            self.data_table = self.content.strip()
        return self


class TableParseResult(BaseModel):
    """Result of table parsing."""

    model_config = ConfigDict(extra="ignore")

    status: SkipJsonSchema[Literal["success", "no_extractable_table", "failed"]] = "success"
    table_format: Literal["markdown", "html"] = Field(
        default="markdown",
        description="'markdown' for regular pipe tables, 'html' when multi-level headers/rowspan/colspan are needed",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="Ordered column header names",
    )
    table_body: str = Field(
        default="",
        description="Complete table in Markdown pipe or HTML <table>...</table> format (without footnotes)",
    )
    footnotes: list[str] = Field(
        default_factory=list,
        description="All table footnotes, experimental conditions, and abbreviations below the table",
    )
    content: SkipJsonSchema[str] = Field(
        default="",
        description="Complete table in Markdown or HTML format, including all footnotes",
    )

    @model_validator(mode="after")
    def _sync_table_content(self) -> "TableParseResult":
        if not self.content and (self.table_body or self.footnotes):
            parts: list[str] = []
            if self.table_body.strip():
                parts.append(self.table_body.strip())
            if self.footnotes:
                fn_text = "\n".join(f.strip() for f in self.footnotes if f and f.strip())
                if fn_text:
                    parts.append(fn_text)
            self.content = "\n\n".join(parts).strip()
        elif self.content and not self.table_body:
            self.table_body = self.content.strip()
        return self


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class StageUsage(BaseModel):
    """API calls and token usage statistics for a stage."""

    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    candidates_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class PipelineSummary(BaseModel):
    """Ingestion summary metadata."""

    total_placeholders: int = 0
    total_manifest_visuals: int = 0
    total_parsed_visuals: int = 0
    counts_by_type: dict[str, int] = Field(default_factory=dict)
    counts_by_status: dict[str, int] = Field(default_factory=dict)
    discarded_placeholder_ids: list[str] = Field(default_factory=list)
    not_localized_visual_ids: list[str] = Field(default_factory=list)
    invalid_smiles: list[dict[str, Any]] = Field(default_factory=list)
    t4_skipped_calls: int = 0
    t4_fallback_calls: int = 0
    silent_error_retries: int = 0
    stages_usage: dict[str, StageUsage] = Field(default_factory=dict)



