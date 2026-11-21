"""DSPy Signatures for Error-Driven / Residual Semantic Prompt Optimization."""

import dspy

from ae.semantic_optimization.models import (
    NegativeRowsOutput,
    PositiveRowsOutput,
    ResidualColumnsOutput,
)


class ResidualPositiveRowAnalysis(dspy.Signature):
    """<mission>
Diagnose why the extractor running `current_prompt` missed the target `missed_gt_rows` present in `gt_table`, using `contrast_tp_rows` (rows the extractor already got right) and `raw_text` as contrastive evidence.
</mission>

<execution_constraints>
1. Evaluate `gt_table` (the FULL Ground Truth table for this document) as a whole alongside `raw_text` and `current_prompt` to diagnose why `missed_gt_rows` were skipped (e.g. missing trigger pattern, overly narrow inclusion scope, overlooked document section/table/caption, or uncaptured composite intersection) while `contrast_tp_rows` succeeded.
2. Generate `RowAnalysis` entries in `rows` strictly for every row present in `missed_gt_rows` (preserving each row's exact `row_id`). Do not emit entries for `contrast_tp_rows` or invent new rows.
3. Explicitly note in `Row_Instantiation_Trigger_description` or `Entity_Filtration_Boundary_description` what aspect of `missed_gt_rows` was unaddressed or excluded by `current_prompt`.
4. Do NOT infer `Extremum_Only` merely because `missed_gt_rows` is a subset of `gt_table`; base `Entity_Filtration_Boundary` strictly on comparing the FULL `gt_table` against `raw_text`.
5. Formulate responses strictly in domain-specific taxonomy native to the source text, standalone and self-sufficient for an external reader without access to `raw_text`.
</execution_constraints>

<evaluation_metrics>
<metric name="Row_Instantiation_Trigger">
<classes>
<class name="Single_Entity">The row is built around one main object. Everything else in the row just describes this single object.</class>
<class name="Composite_Vector">The row is built only when two or more specific properties or conditions intersect. If any change, a new row must be created.</class>
</classes>
</metric>

<metric name="Entity_System_Role">
<classes>
<class name="Target">The object is the main focus of the text.</class>
<class name="Reference">A known benchmark or control measured only to compare against the target.</class>
<class name="Background">A background or environment element setting conditions.</class>
</classes>
</metric>

<metric name="Entity_Filtration_Boundary">
<classes>
<class name="Exhaustive_Set">Every available version or state of the target class was included without performance filtering.</class>
<class name="Extremum_Only">Only extreme versions (best/worst/final/limit) were included.</class>
<class name="Baseline_Only">Only starting/reference/zero state was included.</class>
<class name="Taxonomic_Class">Individual objects were grouped into a single category-level row.</class>
</classes>
</metric>
</evaluation_metrics>

<output_contract>
Return strictly valid JSON matching `PositiveRowsOutput` (`{"rows": [...]}`).
</output_contract>"""

    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    current_prompt: str = dspy.InputField(desc="Current optimized extraction instruction prompt (.txt)")
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Full Ground Truth table for the document as CSV string")
    missed_gt_rows: str = dspy.InputField(desc="Missed Ground Truth rows (False Negatives) as CSV string")
    contrast_tp_rows: str = dspy.InputField(desc="Sample successfully extracted GT rows (True Positives) for contrast")

    analysis: PositiveRowsOutput = dspy.OutputField(desc="Residual positive row analysis for missed GT rows")


class ResidualNegativeRowAnalysis(dspy.Signature):
    """<mission>
Diagnose concrete False Positive rows erroneously extracted using `current_prompt` that are absent from `gt_table`, distinguishing homogeneous excluded distractor cohorts from genuine Ground Truth omissions.
</mission>

<execution_constraints>
1. Inspect the provided `false_positive_rows` (extracted by the current prompt but unmatched in `gt_table`) alongside `raw_text` and `gt_table`.
2. CRITICAL — COHORT DEDUPLICATION VS INDIVIDUAL GT OMISSIONS:
   - **Potential Ground Truth Omissions (emit individually):** If a false-positive row semantically satisfies the positive row-creation trigger, target system role, and inclusion scope of `gt_table` (looks like an accidental annotator omission in GT rather than an excluded distractor), emit a separate `gap_row` for each such item.
   - **Homogeneous Distractor Cohorts (deduplicate per cohort):** When multiple `false_positive_rows` share the exact same exclusion reason (e.g., reference/control entities, preparatory/intermediate items, non-target experimental conditions/species, or off-target tables), emit **exactly one representative `gap_row` per cohort**: quote one representative verbatim substring in `raw_text` and summarize the full cohort scope, members, and `cohort_count` in `gap_entity_description`.
3. The `raw_text` extraction must be an exact, character-for-character substring of the source document containing the representative entity and its local context.
4. Formulate responses strictly in domain-specific taxonomy native to the source text, standalone and self-sufficient.
5. If no false positives correspond to genuine source entities or omissions, return `{"gap_rows": []}`.
</execution_constraints>

<evaluation_metrics>
<metric name="Row_Instantiation_Trigger">
<classes>
<class name="Single_Entity">Row built around one main object.</class>
<class name="Composite_Vector">Row built on intersection of two or more properties/conditions.</class>
</classes>
</metric>
<metric name="Entity_System_Role">
<classes>
<class name="Target">Main focus of the text.</class>
<class name="Reference">Benchmark or control measured for comparison.</class>
<class name="Background">Background or environmental condition.</class>
</classes>
</metric>
<metric name="Entity_Filtration_Boundary">
<classes>
<class name="Exhaustive_Set">All states included.</class>
<class name="Extremum_Only">Only extreme states included.</class>
<class name="Baseline_Only">Only baseline/zero state included.</class>
<class name="Taxonomic_Class">Grouped by taxonomic/category class.</class>
</classes>
</metric>
</evaluation_metrics>

<output_contract>
Return strictly valid JSON matching `NegativeRowsOutput` (`{"gap_rows": [...]}`).
</output_contract>"""

    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    current_prompt: str = dspy.InputField(desc="Current optimized extraction instruction prompt (.txt)")
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Full Ground Truth table for the document as CSV string")
    false_positive_rows: str = dspy.InputField(desc="Grouped representative False Positive rows extracted by current_prompt with cohort counts (JSON)")

    analysis: NegativeRowsOutput = dspy.OutputField(desc="Residual negative row analysis for extracted false positives")


class ResidualColumnAnalysis(dspy.Signature):
    """<mission>
Diagnose field-level discrepancies on aligned rows where `current_prompt` produced `predicted_value` != `gt_value`, producing targeted positive transformation/ingestion diagnoses and negative competing-candidate entries ONLY for the mismatched fields listed in `mismatched_rows_context`.
</mission>

<execution_constraints>
1. Evaluate strictly the `(row_id, field_name)` pairs listed in `mismatched_rows_context`. Do NOT analyze fields that already matched.
2. Preserve exact `row_id` and `field_name` in both `positive_fields` and `negative_fields`.
3. For `positive_fields` (`FieldAnalysis`):
   - Emit a `FieldAnalysis` entry whenever `gt_value` is populated (non-null/non-empty). Skip `positive_fields` ONLY if `gt_value` is null/empty (`spurious_pred`).
   - `raw_text` must be an exact, character-for-character substring of the document containing the source value and local context.
   - Describe the exact transformation/binding rule needed so the extractor outputs `gt_value` instead of `predicted_value`. If `occurrence_count_in_doc > 1`, note that this is a systematic pattern across multiple rows.
   - CRITICAL (INGESTION/GT FIDELITY): Never invent a fake rule to justify a factual mismatch between `raw_text` and `gt_value`. If they differ factually, set `Transformation_description` to `"No valid transformation: factual mismatch between ingested text and GT"` and record the discrepancy in `Suspected_Ingestion_Or_GT_Issue` for HITL triage.
4. For `negative_fields` (`NegativeFieldResult`):
   - Whenever `predicted_value` is non-null/non-empty and represents an incorrect competing value, wrong alternative condition, or spurious extraction (especially when `gt_value` is null), emit a `NegativeFieldResult` for `(row_id, field_name)` with a `CandidateAnalysis` (`candidate_value = predicted_value`, `ground_truth = gt_value or "null"`, verbatim `raw_text` quote if present in text).
5. Formulate all descriptions in standalone, domain-specific terminology.
</execution_constraints>

<output_contract>
Return strictly valid JSON matching `ResidualColumnsOutput` (`{"positive_fields": [...], "negative_fields": [...]}`).
</output_contract>"""

    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    current_prompt: str = dspy.InputField(desc="Current optimized extraction instruction prompt (.txt)")
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Full Ground Truth table for the document as CSV string")
    mismatched_rows_context: str = dspy.InputField(
        desc="JSON batch of aligned rows with ONLY their mismatched fields: {row_id, row_context, mismatches: [{field_name, predicted_value, gt_value, mismatch_type, occurrence_count_in_doc}]}"
    )

    analysis: ResidualColumnsOutput = dspy.OutputField(desc="Targeted positive and negative column analysis for mismatched fields")
