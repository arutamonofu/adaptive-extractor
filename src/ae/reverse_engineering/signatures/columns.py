"""DSPy Signatures for Column-related reverse engineering tasks."""

import dspy
from ae.reverse_engineering.models import (
    PositiveColumnsOutput,
    ConsolidatedColumnsOutput,
    NegativeColumnsOutput,
    GeneralizedColumnsOutput,
)


class PositiveColumnAnalysis(dspy.Signature):
    """<mission>
Analyze each populated field within the target row. Compare the final table value with the source text to understand exactly what the data represents and how the annotator modified it.
</mission>

<execution_constraints>
1. Evaluate strictly the provided target GT row.
2. Analyze only populated fields. Categorically skip null/empty/NaN values.
3. The raw text extraction must be an exact, character-for-character substring of the source document containing the value and local context.
4. If a cell value is traceable in one form or another in several text fragments, prioritize local context, direct extraction, and the minimal number of transformation steps.
5. The response must be formulated strictly using the domain-specific terminology, taxonomy, and technical concepts native to the source text.
6. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
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
Return the response strictly matching the JSON structure below. Skip fields with null, empty, or NaN values in the GT row.
<response_schema>
{
  "fields": [
    {
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
      "Transformation_description": "Comprehensive detailed explanation of each identified mathematical, logical, lexical, syntactic, or other transformation step."
    }
  ]
}
</response_schema>
</output_contract>"""
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Whole Ground Truth table as CSV string")
    target_gt_row: str = dspy.InputField(desc="Target Ground Truth row as CSV string")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")

    analysis: PositiveColumnsOutput = dspy.OutputField(desc="Column analysis result")


class ColumnConsolidation(dspy.Signature):
    """<mission>
Review several specific definitions of a column and combine them into one precise, generalized definition.
</mission>

<execution_constraints>
1. Remove references to specific documents or substance names unless they define the core essence.
2. Do not make the meaning broader than the original description.
3. Pay attention to the name of the field. E.g., If it is for a "minimum" value, strictly keep the word "minimum" in your definition.
4. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
</execution_constraints>

<synthesis_directives>
<directive name="Semantic_Core_generalized">
<definition>What is the pure conceptual definition of the column's attribute, based on the provided local definitions?</definition>
</directive>
</synthesis_directives>

<input_data>
1. Column’s semantic core analysis from different documents (.json).
2. Baseline prompt (.txt).
3. Schema (.yaml).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below via Structured Outputs.
<response_schema>
{
  "Semantic_Core_generalized": "Standalone, domain-specific description of the semantic core."
}
</response_schema>
</output_contract>"""
    semantic_core_analyses: str = dspy.InputField(desc="JSON array containing semantic core analyses from different documents")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    field_name: str = dspy.InputField(desc="Name of the field being consolidated")

    consolidation: ConsolidatedColumnsOutput = dspy.OutputField(desc="Consolidated column core result")


class NegativeColumnAnalysis(dspy.Signature):
    """<mission>
Scan the source text to find and describe alternative or skipped values for the fields of the row that match the fields descriptions but were ignored in the Ground Truth table.
</mission>

<execution_constraints>
1. Evaluate strictly the provided target row of the table.
2. Any candidate must belong to the exact same main object that generated the target row. Do not extract values that belong to other objects or background systems.
3. Only extract exact synonyms or alternative measurements of the same parameter.
4. If there are multiple valid alternatives for a field, extract all of them.
5. The raw text extraction must be an exact, character-for-character substring of the source document containing the value and local context.
6. The response must be formulated strictly using the domain-specific terminology, taxonomy, and technical concepts native to the source text.
7. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
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
3. Target Ground Truth row (.csv).
4. Baseline prompt (.txt).
5. Schema (.yaml).
6. Generalized column’s semantic core analysis (.json).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below. If a field has no valid candidates, return an empty array [] for that key.
<response_schema>
{
  "fields": [
    {
      "field_name": "Name of the target field",
      "candidates": [
        {
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
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Whole Ground Truth table as CSV string")
    target_gt_row: str = dspy.InputField(desc="Target Ground Truth row as CSV string")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    generalized_core: str = dspy.InputField(desc="Generalized column semantic core from consolidation step")

    analysis: NegativeColumnsOutput = dspy.OutputField(desc="Negative column analysis result")


class ColumnGeneralization(dspy.Signature):
    """<mission>
Compare the accepted column values with the rejected ones to define exactly what a column means and how to extract it.
</mission>

<execution_constraints>
1. A true trap is a fundamentally different physical object or metric (ontological substitution). Different wording, slang, or descriptive text for the same object is NOT a trap (resolve these in the transformation).
2. Your synthesized rule must legitimize >90% of the positive column values and firmly reject 100% of the negative ones. If <10% accepted values contradicts your derived rule, isolate them into the anomaly log.
3. Synthesize a broad, generalized rule by replacing document-specific details, exact numbers, and narrow terms with fundamental, abstract domain categories.
4. Write every instruction in the imperative mood (e.g., "Extract ...", "Exclude ...", "Convert ..."). Do not use passive constructions or nominalisations.
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
</input_data>

<output_contract>
Return response strictly matching the JSON structure below.
<response_schema>
{
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
  "Anomalies": [
    {
      "source_reference": {
        "document_id": "Exact name of the source file (e.g., ange.201904751)",
        "positive_reference_ids": ["Array of identifiers from Phase 1 (e.g., row_5)", "or empty array"],
        "negative_reference_ids": ["Array of identifiers from Phase 3 (e.g., gap_row_2)", "or empty array"]
      },
      "anomaly_description": "Description of the logical contradiction."
    }
  ]
}
</response_schema>
</output_contract>"""
    positive_columns_analyses: str = dspy.InputField(desc="JSON array of positive column analyses for a specific field across documents")
    negative_columns_analyses: str = dspy.InputField(desc="JSON array of negative column analyses for a specific field across documents")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    field_name: str = dspy.InputField(desc="Name of the field being generalized")

    generalization: GeneralizedColumnsOutput = dspy.OutputField(desc="Generalized column instructions result")
