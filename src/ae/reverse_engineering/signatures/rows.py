"""DSPy Signatures for Row-related reverse engineering tasks."""

import dspy
from ae.reverse_engineering.models import (
    PositiveRowsOutput,
    ConsolidatedRowsOutput,
    NegativeRowsOutput,
    GeneralizedRowsOutput,
)


class PositiveRowAnalysis(dspy.Signature):
    """<mission>
Analyze the existing rows in the Ground Truth table alongside the source text.
</mission>

<execution_constraints>
1. Evaluate the table as a whole to find the general pattern for creating rows.
2. Only analyze the rows that are actually provided in the table. Do not guess or look for missing rows at this stage.
3. You must evaluate every single row provided in the data.
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
1. Raw text (.md).
2. Ground Truth table (.csv).
3. Baseline prompt (.txt).
4. Schema (.yaml).
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
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Ground Truth table as CSV string")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")

    analysis: PositiveRowsOutput = dspy.OutputField(desc="Row analysis result")


class RowConsolidation(dspy.Signature):
    """<mission>
Review several specific reasons explaining why individual data rows were extracted from various documents and combine them into one precise, generalized definition.
</mission>

<execution_constraints>
1. Synthesize a broad, generalized rule by replacing document-specific details, exact numbers, and narrow terms with fundamental, abstract domain categories (e.g., using 'catalyst type' or 'thermal treatment regime' instead of specific formulas or temperatures).
2. Do not make the meaning broader than the original description.
3. Do not include links, justifications or any formatting noise. Just the rule.
4. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
</execution_constraints>

<input_data>
1. Row instantiation trigger analysis array from different documents (.json).
2. Baseline prompt (.txt).
3. Schema (.yaml).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below.
<response_schema>
{
  "Row_Instantiation_Trigger_generalized": "Standalone, domain-specific description of the rule."
}
</response_schema>
</output_contract>"""
    row_trigger_analyses: str = dspy.InputField(desc="JSON array containing trigger descriptions from different documents")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")

    consolidation: ConsolidatedRowsOutput = dspy.OutputField(desc="Consolidated row trigger result")


class NegativeRowAnalysis(dspy.Signature):
    """<mission>
Scan the source text to find and describe items that match the row creation rule but were left out of the Ground Truth table.
</mission>

<execution_constraints>
1. Find all physical items in the text that fit the given row creation rule and then exclude anything that is already present in the GT array.
2. Create a row for each missed object you find.
3. Base your measurements only on what is explicitly written in the text. Do not guess why the annotator left them out.
4. The raw text extraction must be an exact, character-for-character substring of the source document containing the value and local context.
5. The response must be formulated strictly using the domain-specific terminology, taxonomy, and technical concepts native to the source text.
6. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
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
5. Generalized row instantiation analysis (.json).
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below via Structured Outputs. If no candidates exist, return an empty object {}.
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
    raw_text: str = dspy.InputField(desc="Full text content of the document (.md)")
    gt_table: str = dspy.InputField(desc="Ground Truth table as CSV string")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")
    generalized_trigger: str = dspy.InputField(desc="Generalized row trigger description from consolidation step")

    analysis: NegativeRowsOutput = dspy.OutputField(desc="Negative row analysis result")


class RowGeneralization(dspy.Signature):
    """<mission>
Compare the existing rows in the Ground Truth table with the ignored ones and write the final, absolute rule for creating a new row in this dataset.
</mission>

<execution_constraints>
1. Synthesize a broad, generalized rule by replacing document-specific details, exact numbers, and narrow terms with fundamental, abstract domain categories.
2. Your synthesized rule must legitimize >90% of the positive rows and firmly reject 100% of the negative ones. If <10% accepted rows contradicts your derived rule, isolate them into the anomaly log.
3. The response must be standalone, self-sufficient so that it is explicitly clear for an external reader who lacks access to the source context.
4. Write every instruction in the imperative mood (e.g., "Include ...", "Exclude ...", "Treat ..."). Do not use passive constructions or nominalisations.
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
</input_data>

<output_contract>
Return the response strictly matching the JSON structure below.
<response_schema>
{
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
  "Anomalies": [
    {
      "source_reference": "Document ID, gap_row_X / gt_row_Y",
      "anomaly_description": "Description of the logical contradiction."
    }
  ]
}
</response_schema>
</output_contract>"""
    positive_rows_analyses: str = dspy.InputField(desc="JSON array of positive row analyses from different documents")
    negative_rows_analyses: str = dspy.InputField(desc="JSON array of negative row analyses (gaps) from different documents")
    baseline_prompt: str = dspy.InputField(desc="Baseline extraction prompt (.txt)")
    extraction_schema: str = dspy.InputField(desc="Schema definition (.yaml)")

    generalization: GeneralizedRowsOutput = dspy.OutputField(desc="Generalized row instructions result")
