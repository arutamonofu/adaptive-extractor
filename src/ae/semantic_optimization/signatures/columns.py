"""DSPy Signatures for Column-related reverse engineering tasks."""

import dspy
from ae.semantic_optimization.models import (
    PositiveColumnsOutput,
    ConsolidatedColumnsOutput,
    NegativeColumnsOutput,
    GeneralizedColumnsOutput,
    PromptHarmonizationOutput,
)


class PositiveColumnAnalysis(dspy.Signature):
    """<mission>
Analyze each populated field within the target row(s). Compare the final table value with the source text to understand exactly what the data represents and how the annotator modified it.
</mission>

<execution_constraints>
1. Evaluate strictly the provided target GT row(s) (each identified by `row_id`, e.g., `row_1`, `row_2`).
2. Analyze only populated fields. Categorically skip null/empty/NaN values.
3. Include the corresponding `row_id` for every analyzed field entry.
4. The raw text extraction must be an exact, character-for-character substring of the source document containing the value and local context.
5. If a cell value is traceable in one form or another in several text fragments, prioritize local context, direct extraction, and the minimal number of transformation steps.
6. The response must be formulated strictly using the domain-specific terminology, taxonomy, and technical concepts native to the source text.
7. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
8. CRITICAL (INGESTION FIDELITY): The source document (.md) is produced by automated PDF/OCR/VLM parsing and may contain recognition hallucinations (e.g. corrupted structured/encoded identifiers, misread symbols or exponents in tables, or unextracted <visual> stubs). Under NO circumstances should you invent a fake mathematical, chemical, or logical rule in Transformation_description to justify a factual discrepancy between raw_text and ground_truth. If raw_text and ground_truth differ factually (beyond standard unit conversion, rounding, or lexical formatting), state "No valid transformation: factual mismatch between ingested text and GT" in Transformation_description and record the discrepancy in Suspected_Ingestion_Or_GT_Issue.
</execution_constraints>

<evaluation_metrics>
<metric name="Semantic_Core">
<definition>What is the narrow definition of the physical or logical entity itself?
Strictly excluding all references to analytical instruments, measurement methods, and environmental conditions.</definition>
</metric>

<metric name="Analytical_Method">
<definition>By what specific instrument, analytical method, or mathematical formula was this value obtained or calculated in the text? If not explicitly stated, return Not_Specified.</definition>
</metric>

<metric name="System_Conditions">
<definition>Under what specific conditions (e.g., physical, chemical, or systemic environmental) was this value gotten? If not explicitly stated, return Not_Specified.</definition>
</metric>

<metric name="Hierarchy_Level">
<definition>To which level of system hierarchy does the cell parameter presented in the text belong?</definition>
<classes>
<class name="Macro_System">The parameter describes the anchor object as a single, compositionally coherent entity. It is a fundamental property that extends to the entire object as a whole.</class>
<class name="Micro_Component">The parameter describes a part of the main object, without being extrapolated to the object as a whole.</class>
<class name="External_Environment">The parameter describes the outside space, background factors, or infrastructure around the object.</class>
</classes>
</metric>

<metric name="Semantic_Binding">
<definition>Which binding approach is used in the text to associate the attribute with the anchor node?</definition>
<classes>
<class name="Syntax_Direct">The text explicitly links this value directly to the object.</class>
<class name="Context_Inherited">The value is not directly linked to the object, but inherited from the macro-context or describes the general environment or class of objects it belongs to.</class>
<class name="Absent_In_Text">The value is not written in the document at all.</class>
</classes>
</metric>

<metric name="Text_Precision">
<definition>Did the original text fragment contain semantic ambiguity, approximation, or qualitative estimation?</definition>
<classes>
<class name="Exact_Value">An exact, specific, or unambiguous value.</class>
<class name="Fuzzy_Or_Range">An approximate, estimated, or range value.</class>
</classes>
</metric>

<metric name="Value_Multiplicity">
<definition>Does the source text offer an array of alternative values within a single method?</definition>
<classes>
<class name="Single_Instance">The original text contains exactly one physical value, or the strict ontological nature of the field (for example, an explicit request for minimum, maximum, or a specific state) acted as a semantic filter, leaving exactly one logically valid candidate in the proposed array.</class>
<class name="Multiple_Instances">The text offers an array or pool of multiple values, each of which satisfies the semantics of the field, taking into account its strict ontology.</class>
</classes>
</metric>

<metric name="Transformation">
<definition>What specific operations did the annotator perform to transform the raw text into the final value?</definition>
</metric>
</evaluation_metrics>

<input_data>
1. Raw text (.md).
2. Whole Ground Truth table (.csv).
3. Target Ground Truth row (.csv).
4. Baseline prompt (.txt).
5. Schema (.yaml).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below. Skip fields with null, empty, or NaN values in the GT row(s).
<response_schema>
{
  "fields": [
    {
      "row_id": "Identifier of the target row (e.g., row_1)",
      "field_name": "Name of the target field",
      "raw_text": "Exact, character-for-character substring of the source document containing the value and local context",
      "ground_truth": "Exact value from the GT cell",
      "Semantic_Core": "Standalone, domain-specific definition.",
      "Analytical_Method": "Standalone, domain-specific description, or 'Not_Specified'.",
      "System_Conditions": "Standalone, domain-specific description, or 'Not_Specified'.",
      "Hierarchy_Level_class": "Macro_System | Micro_Component | External_Environment",
      "Hierarchy_Level_description": "Standalone, domain-specific justification of the chosen class.",
      "Semantic_Binding_class": "Syntax_Direct | Context_Inherited | Absent_In_Text",
      "Semantic_Binding_description": "Standalone, domain-specific justification of the chosen class.",
      "Text_Precision_class": "Exact_Value | Fuzzy_Or_Range",
      "Text_Precision_description": "Standalone, domain-specific justification of the chosen class.",
      "Value_Multiplicity_class": "Single_Instance | Multiple_Instances",
      "Value_Multiplicity_description": "Standalone, domain-specific justification of the chosen class.",
      "Transformation_description": "Comprehensive detailed explanation of valid transformations, or factual mismatch disclaimer.",
      "Suspected_Ingestion_Or_GT_Issue": "None or description of mismatch (e.g. OCR error, hallucination, or GT typo)"
    }
  ]
}
</response_schema>
</output_contract>"""
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Whole Ground Truth table as CSV string")
    target_gt_row: str = dspy.InputField(desc="Target Ground Truth row(s) batch as CSV string")

    analysis: PositiveColumnsOutput = dspy.OutputField(desc="Column analysis result")


class ColumnConsolidation(dspy.Signature):
    """<mission>
Consolidate field-level analyses across multiple documents into a unified semantic core and extraction policy for a specific column.
Detect contradictory transformations, ground truth errors, or differing analytical methods instead of blindly over-generalizing.
</mission>

<execution_constraints>
1. Synthesize the generalized Semantic Core and the Analytical/Binding Policy across all provided field analyses.
2. If human expert rulings exist for this column, treat them as immutable hard constraints.
3. CRITICAL: Detect and flag contradictions:
   - If one document converts units while another extracts raw numbers, or if one inherits from a parent group/context while another requires explicit per-item mentions.
   - If an analysis shows Semantic_Binding_class = "Absent_In_Text" or Suspected_Ingestion_Or_GT_Issue != "None", register a TEXT_VS_GT_MISMATCH, SUSPECTED_INGESTION_DEFECT, or SUSPECTED_GT_ERROR conflict. Highlight that the cause may be either an annotator GT error OR an OCR/VLM parser hallucination in the .md file.
   - If the documents warrant a conditional rule (e.g., IF an attribute is explicitly inherited from a parent/group header, propagate it to child entries), formulate it or flag CONDITIONAL_BRANCH_NEEDED.
4. Formulate the response using domain-specific taxonomy.
5. Order `options` in each conflict in descending order of recommendation so the most strongly recommended resolution option is always placed first (`options[0]`).
6. Write all text strictly in English; never include CJK or other foreign-script characters.
</execution_constraints>

<input_data>
1. Array of field analyses for this field from different documents (.json).
2. Baseline prompt (.txt).
3. Schema (.yaml).
4. Field name.
5. Expert rulings (.txt).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below.
<response_schema>
{
  "Semantic_Core_generalized": "Standalone, domain-specific description of the semantic core.",
  "Analytical_and_Binding_Policy": "Policy on analytical methods, hierarchy level, binding expectations, and valid transformations.",
  "Conflicts": [
    {
      "conflict_id": "Unique string id",
      "scope": "column:<field_name>",
      "conflict_type": "TEXT_VS_GT_MISMATCH | SUSPECTED_INGESTION_DEFECT | SUSPECTED_GT_ERROR | CROSS_DOC_CONTRADICTION | CONDITIONAL_BRANCH_NEEDED | BASELINE_POLICY_CONFLICT",
      "document_id": "Document ID",
      "row_id": "row_id",
      "raw_text_quote": "Relevant text quote",
      "gt_value": "Ground truth cell value",
      "conflicting_value_or_rule": "Conflicting rule or document behavior",
      "explanation": "Clear explanation of why this is a conflict (noting whether it could be a GT error or an ingestion/OCR hallucination in .md)",
      "options": [
        {
          "action_type": "PATCH_GT_CELL | PATCH_INGESTED_MD | IGNORE_INGESTION_DEFECT | GENERALIZE | ADD_CONDITION | OVERRIDE_BASELINE | IGNORE_CASE",
          "label": "Short label",
          "proposed_rule_or_value": "Rule text or STRICTLY the exact literal replacement cell value for PATCH_GT_CELL (never an explanatory sentence)"
        }
      ]
    }
  ]
}
</response_schema>
</output_contract>"""
    field_analyses: str = dspy.InputField(desc="JSON array containing complete field analyses from different documents")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    field_name: str = dspy.InputField(desc="Name of the field being consolidated")
    expert_rulings: str = dspy.InputField(desc="Authoritative human expert rulings and constraints (.txt)")

    consolidation: ConsolidatedColumnsOutput = dspy.OutputField(desc="Consolidated column core and policy result")


class NegativeColumnAnalysis(dspy.Signature):
    """<mission>
Scan the source text to find and describe alternative or skipped values for the fields of the target row(s) that match the fields descriptions but were ignored in the Ground Truth table.
</mission>

<execution_constraints>
1. Evaluate strictly the provided target GT row(s) (each identified by `row_id`, e.g., `row_1`, `row_2`).
2. Any candidate must belong to the exact same main object that generated the corresponding target row. Do not extract values that belong to other objects or background systems.
3. Only extract genuine competing alternatives or rejected measurements of the same parameter. Do NOT extract the source text fragment that is already the direct pre-transformation source of the Ground Truth value (such as a differently formatted or uncanonicalized representation of the exact same entity/identifier, a number before decimal-separator or trailing-zero formatting like `0.030` vs `0,03`, or unstripped markdown/HTML tags). A valid negative candidate must be a genuinely different competing measurement, alternative assay/condition, or conflicting value in the document that was rejected in favor of the Ground Truth value.
4. If there are multiple valid alternatives for a field, extract all of them.
5. Include the corresponding `row_id` for every analyzed field and candidate entry.
6. The raw text extraction must be an exact, character-for-character substring of the source document containing the value and local context.
7. The response must be formulated strictly using the domain-specific terminology, taxonomy, and technical concepts native to the source text.
8. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
9. If an alternative assay, measurement method, or condition for the target entity is described in an unparsed `<visual id="...">` manifest stub (caption and `> description`), you may directly cite the exact substring from that manifest stub as a negative candidate.
</execution_constraints>

<evaluation_metrics>
<metric name="Semantic_Core">
<definition>What is the narrow definition of the physical or logical entity itself?
Strictly excluding all references to analytical instruments, measurement methods, and environmental conditions.</definition>
</metric>

<metric name="Analytical_Method">
<definition>By what specific instrument, analytical method, or mathematical formula was this value obtained or calculated in the text? If not explicitly stated, return Not_Specified.</definition>
</metric>

<metric name="System_Conditions">
<definition>Under what specific conditions (e.g., physical, chemical, or systemic environmental) was this value gotten? If not explicitly stated, return Not_Specified.</definition>
</metric>

<metric name="Hierarchy_Level">
<definition>To which level of system hierarchy does the cell parameter presented in the text belong?</definition>
<classes>
<class name="Macro_System">The parameter describes the anchor object as a single, compositionally coherent entity. It is a fundamental property that extends to the entire object as a whole.</class>
<class name="Micro_Component">The parameter describes a part of the main object, without being extrapolated to the object as a whole.</class>
<class name="External_Environment">The parameter describes the outside space, background factors, or infrastructure around the object.</class>
</classes>
</metric>

<metric name="Semantic_Binding">
<definition>Which binding approach is used in the text to associate the attribute with the anchor node?</definition>
<classes>
<class name="Syntax_Direct">The text explicitly links this value directly to the object.</class>
<class name="Context_Inherited">The value is not directly linked to the object, but inherited from the macro-context or describes the general environment or class of objects it belongs to.</class>
<class name="Absent_In_Text">The value is not written in the document at all.</class>
</classes>
</metric>

<metric name="Text_Precision">
<definition>Did the original text fragment contain semantic ambiguity, approximation, or qualitative estimation?</definition>
<classes>
<class name="Exact_Value">An exact, specific, or unambiguous value.</class>
<class name="Fuzzy_Or_Range">An approximate, estimated, or range value.</class>
</classes>
</metric>

<metric name="Value_Multiplicity">
<definition>Does the source text offer an array of alternative values within a single method?</definition>
<classes>
<class name="Single_Instance">The original text contains exactly one physical value, or the strict ontological nature of the field (for example, an explicit request for minimum, maximum, or a specific state) acted as a semantic filter, leaving exactly one logically valid candidate in the proposed array.</class>
<class name="Multiple_Instances">The text offers an array or pool of multiple values, each of which satisfies the semantics of the field, taking into account its strict ontology.</class>
</classes>
</metric>
</evaluation_metrics>

<input_data>
1. Raw text (.md).
2. Whole Ground Truth table (.csv).
3. Target Ground Truth row(s) batch (.csv).
4. Baseline prompt (.txt).
5. Schema (.yaml).
6. Generalized column’s semantic core analysis (.json).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below. Include `row_id` for each row/field entry. If a field has no valid candidates, return an empty array [] for that key.
<response_schema>
{
  "fields": [
    {
      "row_id": "Identifier of the target row (e.g., row_1)",
      "field_name": "Name of the target field",
      "candidates": [
        {
          "row_id": "Identifier of the target row (e.g., row_1)",
          "raw_text": "Strict quote from the source text containing the entity and its local context.",
          "candidate_value": "The exact raw value/substring of the candidate extracted from the text",
          "ground_truth": "The exact value from the GT cell",
          "Semantic_Core": "Standalone, domain-specific definition.",
          "Analytical_Method": "Standalone, domain-specific description, or 'Not_Specified'.",
          "System_Conditions": "Standalone, domain-specific description, or 'Not_Specified'.",
          "Hierarchy_Level_class": "Macro_System | Micro_Component | External_Environment",
          "Hierarchy_Level_description": "Standalone, domain-specific justification of the chosen class.",
          "Semantic_Binding_class": "Syntax_Direct | Context_Inherited | Absent_In_Text",
          "Semantic_Binding_description": "Standalone, domain-specific justification of the chosen class.",
          "Text_Precision_class": "Exact_Value | Fuzzy_Or_Range",
          "Text_Precision_description": "Standalone, domain-specific justification of the chosen class.",
          "Value_Multiplicity_class": "Single_Instance | Multiple_Instances",
          "Value_Multiplicity_description": "Standalone, domain-specific justification of the chosen class."
        }
      ]
    }
  ]
}
</response_schema>
</output_contract>"""
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Whole Ground Truth table as CSV string")
    generalized_core: str = dspy.InputField(desc="Column semantic core from positive analysis of this document or consolidated policy")
    target_gt_row: str = dspy.InputField(desc="Target Ground Truth row(s) batch as CSV string")

    analysis: NegativeColumnsOutput = dspy.OutputField(desc="Negative column analysis result")


class ColumnGeneralization(dspy.Signature):
    """<mission>
Synthesize consolidated column semantic cores, binding policies, and definitive column inclusion, exclusion, and transformation instructions by comparing accepted Ground Truth values against negative alternative candidates.
Refine and augment the user's baseline prompt while detecting Ground Truth errors, omissions, ingestion defects, and contradictions against the baseline prompt.
</mission>

<execution_constraints>
1. Synthesize `Semantic_Core_generalized`, `Analytical_and_Binding_Policy`, and imperative `Column_Inclusion_Instructions`, `Column_Exclusion_Instructions`, and `Transformation_Instructions`.
2. Honor all human expert rulings as immutable hard constraints.
3. REFINE & AUGMENT BASELINE PROMPT: Preserve non-conflicting field instructions from `baseline_prompt`. Never drop or contradict conditional/multi-step derivation mechanisms from `baseline_prompt` (such as assembling a field value from a shared template/scaffold and separate modifiers/substituents, unpacking relevant visual diagrams/tables, or invoking declared schema tools) solely because a small training batch did not contain those edge cases or showed canonicalized surface variations. Rules that preserve the raw or canonicalized format of structured records from VLM/table blocks must NOT block cross-checking and correcting obvious OCR/VLM recognition defects against explicit prose, nomenclature, or other sections/tables in the same document (only arbitrary manual rewriting or altering valid representations is prohibited).
4. BASELINE CONFLICT DETECTION: If empirical Ground Truth values/transformations contradict an explicit instruction in `baseline_prompt` (e.g., `baseline_prompt` specifies one unit, condition, or derivation rule while GT follows another), flag it in `Conflicts` with `conflict_type: "BASELINE_POLICY_CONFLICT"` and provide options for `OVERRIDE_BASELINE` (adopt GT behavior) and `KEEP_BASELINE` (preserve baseline prompt instruction).
5. If a negative candidate reveals that the Ground Truth cell has an error, typo, or extracted an invalid parameter (e.g. characterization optimum instead of experimental measurement), flag it as a `SUSPECTED_GT_ERROR` conflict.
6. If a value discrepancy appears caused by an OCR or VLM recognition defect in the ingested .md document, flag it as `TEXT_VS_GT_MISMATCH` or `SUSPECTED_INGESTION_DEFECT` so the user can patch .md or unpack visuals.
7. Formulate general semantic principles first, placing specific terms/compounds from training texts in parentheses as examples `(e.g., ...)` to prevent overfitting.
8. Write every instruction in the imperative mood (e.g., "Extract ...", "Exclude ...", "Convert ...").
9. Order `options` in each conflict in descending order of recommendation so the most strongly recommended resolution option is always placed first (`options[0]`).
10. Write all text strictly in English; never include CJK or other foreign-script characters.
</execution_constraints>

<synthesis_directives>
<directive name="Column_Inclusion_Instructions">
<definition>Set of instructions for including values for the column. Each instruction must be written in the imperative mood.</definition>
</directive>
<directive name="Column_Exclusion_Instructions">
<definition>Set of instructions for excluding values for the column. Each instruction must be written in the imperative mood.</definition>
</directive>
<directive name="Transformation_Instructions">
<definition>Description of target form and imperative steps needed to process the raw text into the final value and form. Each instruction must be written in the imperative mood.</definition>
</directive>
</synthesis_directives>

<input_data>
1. Positive columns analysis from different documents (.json).
2. Negative columns analysis from different documents (.json).
3. Baseline prompt (.txt).
4. Schema (.yaml).
5. Consolidated column policy (.json).
6. Expert rulings (.txt).
</input_data>

<output_contract>
Return response strictly matching the JSON structure below.
<response_schema>
{
  "Semantic_Core_generalized": "Standalone, domain-specific description of the semantic core.",
  "Analytical_and_Binding_Policy": "Policy on analytical methods, hierarchy level, binding expectations, and valid transformations.",
  "Instructions": {
    "Column_Inclusion_Instructions": [
      "Instruction 1",
      "Instruction 2"
    ],
    "Column_Exclusion_Instructions": [
      "Instruction 1",
      "Instruction 2"
    ],
    "Transformation_Instructions": [
      "Instruction 1",
      "Instruction 2"
    ]
  },
  "Conflicts": [
    {
      "conflict_id": "Unique string id",
      "scope": "column:<field_name>",
      "conflict_type": "BASELINE_POLICY_CONFLICT | TEXT_VS_GT_MISMATCH | SUSPECTED_INGESTION_DEFECT | SUSPECTED_GT_ERROR | CONDITIONAL_BRANCH_NEEDED | CROSS_DOC_CONTRADICTION",
      "document_id": "Document ID",
      "row_id": "row_id",
      "raw_text_quote": "Exact text quote",
      "gt_value": "GT cell value",
      "conflicting_value_or_rule": "Reason for contradiction or conflicting baseline prompt rule",
      "explanation": "Clear explanation of why this is a conflict (against baseline prompt, potential GT error, or ingestion/OCR hallucination in .md)",
      "options": [
        {
          "action_type": "OVERRIDE_BASELINE | KEEP_BASELINE | PATCH_GT_CELL | PATCH_INGESTED_MD | IGNORE_INGESTION_DEFECT | ADD_CONDITION | GENERALIZE | IGNORE_CASE",
          "label": "Short label",
          "proposed_rule_or_value": "Proposed instruction or STRICTLY the exact literal replacement cell value for PATCH_GT_CELL (never an explanatory sentence)"
        }
      ]
    }
  ]
}
</response_schema>
</output_contract>"""
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    field_name: str = dspy.InputField(desc="Name of the field being generalized")
    positive_columns_analyses: str = dspy.InputField(desc="JSON array of positive column analyses for a specific field across documents")
    negative_columns_analyses: str = dspy.InputField(desc="JSON array of negative column analyses for a specific field across documents")
    consolidated_policy: str = dspy.InputField(desc="Consolidated column semantic core and policy (.json)")
    expert_rulings: str = dspy.InputField(desc="Authoritative human expert rulings and constraints (.txt)")

    generalization: GeneralizedColumnsOutput = dspy.OutputField(desc="Generalized column instructions result")


class PromptHarmonization(dspy.Signature):
    """<mission>
Harmonize the assembled extraction prompt across all record-selection and column-specific sections.
Align cross-column dependencies, eliminate redundant repetitions, and ensure non-conflicting rules from the user's baseline prompt and authoritative expert rulings are seamlessly integrated.
</mission>

<execution_constraints>
1. Preserve the plain-text section structure separated by `---` (Preamble, Expert Rulings if present, RECORD SELECTION, each FIELD section in uppercase, and REASONING STYLE only if already present in `draft_prompt`).
2. CROSS-COLUMN ALIGNMENT: Ensure dependent fields (e.g. a kinetic parameter and its corresponding assay condition/substrate, or a value and its unit/relation) explicitly reference each other consistently.
3. BASELINE PRESERVATION: Integrate the domain persona, scope, and non-conflicting extraction guidelines from `baseline_prompt` into the appropriate sections of the prompt, explicitly retaining conditional assembly/derivation workflows (e.g., combining templates/scaffolds with separate modifiers or identifiers across figures/tables), visual-inspection directives, and usage rules for any tools declared in `extraction_schema`, while stripping external wrapper syntax (e.g., XML tags, code assignments, or raw JSON schema dumps) or conflicting output-format examples.
4. Do NOT dilute or delete specific Inclusion, Exclusion, or Transformation rules discovered for individual columns; only deduplicate verbatim redundancy and harmonize terminology.
5. CONTRADICTION CHECK: Explicitly verify that no field section contradicts another field section or RECORD SELECTION (e.g., if one section allows a category or metric type, another section must not exclude it). Resolve any such cross-section contradiction in favor of the schema and authoritative rulings. Apply specificity precedence when reconciling general format-preservation rules (e.g., `expert_rulings` stating "no normalization" or "preserve source form") with local resolution rules: if a table `footnote`, `superscript`, `caption`, or `legend` explicitly maps an abbreviated header or code to a full or qualified entity/condition/category designation, that local definition takes precedence over any general rule to preserve the raw form of standalone terms; otherwise preserve the source format. Explicitly separate the scopes of these two rules so they do not conflict.
6. GENERALIZE MECHANISMS: When rules mention resolving abbreviations or labels via table footnotes, superscripts, legends, or captions, keep the mechanism general (any footnote/superscript/index marker) rather than overfitting to one marker style.
7. Write the final harmonized prompt strictly in clean English; strip any accidental CJK or other foreign-script characters.
</execution_constraints>"""
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Original user baseline prompt (.txt)")
    expert_rulings: str = dspy.InputField(desc="Authoritative human expert rulings (.txt)")
    draft_prompt: str = dspy.InputField(desc="Draft assembled prompt containing all synthesized row and column rules")

    harmonization: PromptHarmonizationOutput = dspy.OutputField(desc="Harmonized final prompt and notes")
