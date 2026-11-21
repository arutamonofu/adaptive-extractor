"""DSPy Signatures for Row-related reverse engineering tasks."""

import dspy
from ae.semantic_optimization.models import (
    PositiveRowsOutput,
    ConsolidatedRowsOutput,
    NegativeRowsOutput,
    GeneralizedRowsOutput,
)


class PositiveRowAnalysis(dspy.Signature):
    """<mission>
Analyze the target Ground Truth rows in the context of the full Ground Truth table and source text to identify the document's row creation and filtering rules.
</mission>

<execution_constraints>
1. Evaluate `gt_table` (the FULL Ground Truth table for this document) as a whole alongside `raw_text` to determine the global row-creation pattern, entity system roles, and `Entity_Filtration_Boundary` (`Exhaustive_Set` vs `Extremum_Only` vs `Baseline_Only` vs `Taxonomic_Class`).
2. Do NOT infer `Extremum_Only` or artificial annotator filtering merely because `target_gt_rows` is a subset of `gt_table`: `target_gt_rows` is only a representative sample selected for per-row analysis. Base `Entity_Filtration_Boundary` strictly on comparing the FULL `gt_table` against `raw_text`.
3. Generate `RowAnalysis` entries in the output `rows` array strictly for every row present in `target_gt_rows` (preserving each row's exact `row_id`). Do not invent or look for missing rows at this stage.
4. The response must be formulated strictly using the domain-specific terminology, taxonomy, and technical concepts native to the source text.
5. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
</execution_constraints>

<evaluation_metrics>
<metric name="Row_Instantiation_Trigger">
<definition>What is the primary reason for creating a new row in the dataset?</definition>
<classes>
<class name="Single_Entity">The row is built around one main object. Everything else in the row just describes this single object.</class>
<class name="Composite_Vector">The row is built only when two or more specific properties or conditions intersect (e.g., Object A under Condition B). If any of these change, a new row must be created.</class>
</classes>
</metric>

<metric name="Entity_System_Role">
<definition>What is the role of the main object in this row?</definition>
<classes>
<class name="Target">The object is the main focus of the text.</class>
<class name="Reference">A known object (benchmark or control) measured only to compare it with the target.</class>
<class name="Background">A background or environment element that sets the conditions but is not the object itself.</class>
</classes>
</metric>

<metric name="Entity_Filtration_Boundary">
<definition>How did the annotator filter the available objects in the text to decide which ones get their own row? (Attention: Focus only on the object itself and the FULL `gt_table` vs `raw_text`. Do not analyze column-level transformations).</definition>
<classes>
<class name="Exhaustive_Set">Every available version or state of the object was included in the table. No filters were applied.</class>
<class name="Extremum_Only">Only the extreme versions of the object (best/worst result, final stage, or limit) were included. All intermediate options were ignored.</class>
<class name="Baseline_Only">Only the starting, reference, or "zero" state of the object was included. Modified or later states were ignored.</class>
<class name="Taxonomic_Class">Individual objects were grouped together into a single row that describes the entire category or class as a whole.</class>
</classes>
</metric>
</evaluation_metrics>

<input_data>
1. Raw text (.md).
2. Full Ground Truth table (.csv) (`gt_table`).
3. Target representative Ground Truth rows subset (.csv) (`target_gt_rows`).
4. Baseline prompt (.txt).
5. Schema (.yaml).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below.
<response_schema>
{
  "rows": [
    {
      "row_id": "Identifier for the row (e.g., row_1)",
      "Row_Instantiation_Trigger_class": "Single_Entity | Composite_Vector",
      "Row_Instantiation_Trigger_description": "Standalone, domain-specific justification of the chosen class.",
      "Entity_System_Role_class": "Target | Reference | Background",
      "Entity_System_Role_description": "Standalone, domain-specific justification of the chosen class.",
      "Entity_Filtration_Boundary_class": "Exhaustive_Set | Extremum_Only | Baseline_Only | Taxonomic_Class",
      "Entity_Filtration_Boundary_description": "Standalone, domain-specific justification of the chosen class."
    }
  ]
}
</response_schema>
</output_contract>"""
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Full Ground Truth table for the document as CSV string")
    target_gt_rows: str = dspy.InputField(desc="Target subset of representative Ground Truth rows as CSV string to analyze")

    analysis: PositiveRowsOutput = dspy.OutputField(desc="Row analysis result")


class RowConsolidation(dspy.Signature):
    """<mission>
Synthesize row trigger analyses from multiple documents into a consolidated row creation and filtering policy.
Detect contradictions and ambiguities across documents instead of blindly over-generalizing.
</mission>

<execution_constraints>
1. Synthesize a generalized row creation rule and filtration policy across all provided document analyses.
2. If previous expert rulings exist, treat them as immutable hard constraints.
3. CRITICAL: Do NOT blur or erase contradictions across documents. If Document A extracted preparatory/intermediate items or baseline controls while Document B strictly excluded them, or if row trigger boundaries contradict each other:
   - Identify the contradiction.
   - If a conditional rule can cleanly distinguish them based on document structure or domain factors, formulate an IF/ELSE condition.
   - If genuine ambiguity exists, or if you suspect an annotator error in Ground Truth, or an unextracted visual manifest stub in .md, flag it in the Conflicts array with actionable resolution options.
4. Formulate the response using domain-specific taxonomy.
5. Order `options` in each conflict in descending order of recommendation so the most strongly recommended resolution option is always placed first (`options[0]`).
6. Write all text strictly in English; never include CJK or other foreign-script characters.
</execution_constraints>

<input_data>
1. Array of complete row analyses from different documents (.json).
2. Baseline prompt (.txt).
3. Schema (.yaml).
4. Expert rulings from human supervisor (.txt).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below.
<response_schema>
{
  "Row_Instantiation_Trigger_generalized": "Standalone, domain-specific description of when a row is created.",
  "Entity_Role_and_Filtration_Policy": "Policy specifying target entity roles (Target vs Reference) and boundary filters (Exhaustive_Set vs Extremum_Only).",
  "Conflicts": [
    {
      "conflict_id": "Unique string id",
      "scope": "rows",
      "conflict_type": "TEXT_VS_GT_MISMATCH | SUSPECTED_INGESTION_DEFECT | CROSS_DOC_CONTRADICTION | CONDITIONAL_BRANCH_NEEDED | SUSPECTED_GT_ERROR | BASELINE_POLICY_CONFLICT",
      "document_id": "Target document id",
      "row_id": "row_id or all",
      "raw_text_quote": "Relevant text quote if applicable",
      "gt_value": "GT description or entity name",
      "conflicting_value_or_rule": "Conflicting rule or document behavior",
      "explanation": "Clear explanation of the conflict (noting potential GT error or missing visual/ingestion defect)",
      "options": [
        {
          "action_type": "GENERALIZE | ADD_CONDITION | OVERRIDE_BASELINE | PATCH_GT_DROP_ROW | IGNORE_INGESTION_DEFECT | IGNORE_CASE",
          "label": "Short label for choice menu",
          "proposed_rule_or_value": "Rule or value text"
        }
      ]
    }
  ]
}
</response_schema>
</output_contract>"""
    row_analyses: str = dspy.InputField(desc="JSON array containing full row analyses from different documents")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    expert_rulings: str = dspy.InputField(desc="Authoritative human expert rulings and constraints (.txt)")

    consolidation: ConsolidatedRowsOutput = dspy.OutputField(desc="Consolidated row trigger and policy result")


class NegativeRowAnalysis(dspy.Signature):
    """<mission>
Scan the source text to identify objects/measurements absent from the full Ground Truth table, distinguishing homogeneous excluded distractor cohorts from individual suspected Ground Truth omissions.
</mission>

<execution_constraints>
1. Compare `raw_text` against the FULL `gt_table` array to identify entities/measurements present in the source text that are absent from `gt_table`.
2. CRITICAL — COHORT DEDUPLICATION VS INDIVIDUAL GT OMISSIONS:
   - **Potential Ground Truth Omissions (emit individually):** If a missing entity or measurement semantically satisfies the positive row-creation trigger, target system role, and inclusion scope of `gt_table` (i.e., it looks like an accidental annotator omission or a label/identifier mismatch rather than an excluded category), emit a separate `gap_row` for each such item.
   - **Homogeneous Distractor Cohorts (deduplicate per cohort):** When multiple entities or table cells within the same table or section are excluded for the exact same structural or semantic reason (e.g., non-target experimental conditions/species/substrates in a table, benchmark/reference controls, preparatory/intermediate entries in a diagram or protocol, or an entire off-target assay table/stub), do NOT emit dozens of repetitive `gap_rows`. Emit **exactly one representative `gap_row` per cohort/exclusion reason**: provide an exact verbatim quote of one representative member in `raw_text`, and summarize the full cohort scope, members/categories, and approximate count in `gap_entity_description`.
3. Base your measurements strictly on what is explicitly written in the text. Do not guess why the annotator left them out.
4. The `raw_text` extraction must be an exact, character-for-character substring of the source document containing the representative entity and its local context.
5. The response must be formulated strictly using the domain-specific terminology, taxonomy, and technical concepts native to the source text.
6. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
7. If an excluded entity class, control group, or secondary experiment appears in an unparsed `<visual id="...">` manifest stub (caption and `> description`), cite the exact substring from that manifest stub as a single representative cohort `gap_row`.
</execution_constraints>

<evaluation_metrics>
<metric name="Row_Instantiation_Trigger">
<definition>What is the primary reason for creating a new row in the dataset?</definition>
<classes>
<class name="Single_Entity">The row is built around one main object. Everything else in the row just describes this single object.</class>
<class name="Composite_Vector">The row is built only when two or more specific properties or conditions intersect (e.g., Object A under Condition B). If any of these change, a new row must be created.</class>
</classes>
</metric>

<metric name="Entity_System_Role">
<definition>What is the role of the main object in this row?</definition>
<classes>
<class name="Target">The object is the main focus of the text.</class>
<class name="Reference">A known object (benchmark or control) measured only to compare it with the target.</class>
<class name="Background">A background or environment element that sets the conditions but is not the object itself.</class>
</classes>
</metric>

<metric name="Entity_Filtration_Boundary">
<definition>How did the annotator filter the available objects in the text to decide which ones get their own row? (Attention: Focus only on the object itself. Do not analyze its fields).</definition>
<classes>
<class name="Exhaustive_Set">Every available version or state of the object was included in the table. No filters were applied.</class>
<class name="Extremum_Only">Only the extreme versions of the object (best/worst result, final stage, or limit) were included. All intermediate options were ignored.</class>
<class name="Baseline_Only">Only the starting, reference, or "zero" state of the object was included. Modified or later states were ignored.</class>
<class name="Taxonomic_Class">Individual objects were grouped together into a single row that describes the entire category or class as a whole.</class>
</classes>
</metric>
</evaluation_metrics>

<input_data>
1. Raw text (.md)
2. Ground Truth table (.csv)
3. Baseline prompt (.txt)
4. Schema (.yaml)
5. Generalized row instantiation and filtration policy (.json).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below via Structured Outputs. If no candidates exist, return {"gap_rows": []}.
<response_schema>
{
  "gap_rows": [
    {
      "gap_row_id": "Identifier for the missing row (e.g., gap_row_1)",
      "raw_text": "Strict quote from the source text containing the entity and its local context.",
      "gap_entity_description": "Standalone, domain-specific description of the entity.",
      "Row_Instantiation_Trigger_class": "Single_Entity | Composite_Vector",
      "Row_Instantiation_Trigger_description": "Standalone, domain-specific justification of the chosen class.",
      "Entity_System_Role_class": "Target | Reference | Background",
      "Entity_System_Role_description": "Standalone, domain-specific justification of the chosen class.",
      "Entity_Filtration_Boundary_class": "Exhaustive_Set | Extremum_Only | Baseline_Only | Taxonomic_Class",
      "Entity_Filtration_Boundary_description": "Standalone, domain-specific justification of the chosen class."
    }
  ]
}
</response_schema>
</output_contract>"""
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Ground Truth table as CSV string")
    generalized_trigger: str = dspy.InputField(desc="Row trigger description from positive analysis of this document or consolidated policy")

    analysis: NegativeRowsOutput = dspy.OutputField(desc="Negative row analysis result")


class RowGeneralization(dspy.Signature):
    """<mission>
Synthesize consolidated row trigger policies and definitive row inclusion/exclusion instructions by comparing positive Ground Truth rows against negative gap candidates.
Refine and augment the user's baseline prompt while detecting Ground Truth omissions, cross-document contradictions, and conflicts against the baseline prompt.
</mission>

<execution_constraints>
1. Synthesize the consolidated `Row_Instantiation_Trigger_generalized`, `Entity_Role_and_Filtration_Policy`, and imperative `Row_Inclusion_Instructions` / `Row_Exclusion_Instructions`.
2. Honor all human expert rulings as immutable hard constraints.
3. REFINE & AUGMENT BASELINE PROMPT: Preserve non-conflicting record selection rules from the user's `baseline_prompt`. Do not discard valid constraints from `baseline_prompt` simply because a small training set did not showcase every edge case.
4. BASELINE CONFLICT DETECTION: If the empirical Ground Truth patterns contradict an explicit instruction in `baseline_prompt` (e.g., `baseline_prompt` says to include reference controls or preparatory/intermediate items while GT excludes them, or vice versa), flag it in `Conflicts` with `conflict_type: "BASELINE_POLICY_CONFLICT"` and provide options for `OVERRIDE_BASELINE` (follow GT) and `KEEP_BASELINE` (keep baseline prompt rule).
5. Distinguish genuine negative distractors (unrelated compounds, starting materials, baseline controls) from SUSPECTED GT OMISSIONS. If a negative gap row shares the exact same properties and target status as positive rows, do NOT invent an artificial exclusion rule that would hurt recall: flag it as a `SUSPECTED_GT_OMISSION` conflict.
6. Formulate general semantic principles first, placing specific entity names from the training texts in parentheses as examples `(e.g., generic reference control)` to avoid overfitting to specific molecules.
7. Write every instruction in the imperative mood (e.g., "Include ...", "Exclude ...").
8. Order `options` in each conflict in descending order of recommendation so the most strongly recommended resolution option is always placed first (`options[0]`).
9. Write all text strictly in English; never include CJK or other foreign-script characters.
</execution_constraints>

<synthesis_directives>
<directive name="Row_Inclusion_Instructions">
<definition>A set of instructions including entities. Each instruction must be written in the imperative mood.</definition>
</directive>
<directive name="Row_Exclusion_Instructions">
<definition>A set of instructions excluding entities. Each instruction must be written in the imperative mood.</definition>
</directive>
<directive name="Rationale">
<definition>Justification why these instructions make sense for extraction from the document.</definition>
</directive>
</synthesis_directives>

<input_data>
1. Positive row analysis from different documents (.json).
2. Negative row analysis from different documents (.json).
3. Baseline prompt (.txt).
4. Schema (.yaml).
5. Consolidated row policy (.json).
6. Expert rulings (.txt).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below.
<response_schema>
{
  "Row_Instantiation_Trigger_generalized": "Standalone, domain-specific description of when a row is created.",
  "Entity_Role_and_Filtration_Policy": "Policy specifying target entity roles (Target vs Reference) and boundary filters (Exhaustive_Set vs Extremum_Only).",
  "Instructions": {
    "Row_Inclusion_Instructions": [
      "Instruction 1",
      "Instruction 2"
    ],
    "Row_Exclusion_Instructions": [
      "Instruction 1",
      "Instruction 2"
    ],
    "Validation_Rationale": "Standalone justification."
  },
  "Conflicts": [
    {
      "conflict_id": "Unique string id",
      "scope": "rows",
      "conflict_type": "BASELINE_POLICY_CONFLICT | SUSPECTED_GT_OMISSION | TEXT_VS_GT_MISMATCH | SUSPECTED_INGESTION_DEFECT | CROSS_DOC_CONTRADICTION | CONDITIONAL_BRANCH_NEEDED",
      "document_id": "Document ID",
      "row_id": "gap_row_X or row_Y",
      "raw_text_quote": "Exact text quote",
      "gt_value": "Entity or row description",
      "conflicting_value_or_rule": "Reason for contradiction or conflicting baseline prompt rule",
      "explanation": "Clear explanation of why this is a conflict (against baseline prompt, across docs, or suspected GT omission)",
      "options": [
        {
          "action_type": "OVERRIDE_BASELINE | KEEP_BASELINE | PATCH_GT_ADD_ROW | ADD_CONDITION | GENERALIZE | IGNORE_INGESTION_DEFECT | IGNORE_CASE",
          "label": "Short label",
          "proposed_rule_or_value": "Proposed instruction or row data JSON"
        }
      ]
    }
  ]
}
</response_schema>
</output_contract>"""
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    positive_rows_analyses: str = dspy.InputField(desc="JSON array of positive row analyses from different documents")
    negative_rows_analyses: str = dspy.InputField(desc="JSON array of negative row analyses (gaps) from different documents")
    consolidated_policy: str = dspy.InputField(desc="Consolidated row instantiation and filtration policy (.json)")
    expert_rulings: str = dspy.InputField(desc="Authoritative human expert rulings and constraints (.txt)")

    generalization: GeneralizedRowsOutput = dspy.OutputField(desc="Generalized row instructions result")
