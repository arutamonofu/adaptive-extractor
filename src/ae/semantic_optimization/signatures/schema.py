"""DSPy Signature for domain-independent schema constraint inference in Reverse Engineering."""

import dspy

from ae.semantic_optimization.models import SchemaRefinementOutput


class SchemaConstraintInference(dspy.Signature):
    """<mission>
Analyze the task schema, extraction instructions, and empirical training column profiles to infer safe, non-overfitting JSON-Schema constraints (`description`, `choices`, `default_value`, and conservative `pattern`/`min_value`/`max_value`) for each field.
</mission>

<execution_constraints>
1. CLOSED ENUM VS OPEN VOCABULARY:
   - Set `is_closed_enum = true` ONLY if the field represents a genuinely finite, closed set of discrete categories, operators, flags, or assay/measurement types defined by the task ontology or instructions.
   - DOCUMENT SUPPORT GATE: NEVER set `is_closed_enum = true` if `active_docs_count < min_enum_doc_support` or if observed values in `column_profiles` keep appearing in only 1 document (`doc_count == 1`), because cross-document vocabulary saturation has not been established.
   - NEVER set `is_closed_enum = true` for open-vocabulary entities, identifiers, chemical structures/formulas, species/names, unit notations with cross-paper surface variants, free-text descriptions, or arbitrary numeric measurements, even if the training sample has only a few distinct values.
   - When `is_closed_enum = true`, `choices` MUST include ALL observed empirical values from `column_profiles` PLUS any other valid options explicitly stated in `baseline_prompt` or `generated_prompt` (e.g., if training data has `["=", ">"]` and the prompt also allows `"<"`, include `["=", ">", "<"]`).
2. DEFAULT VALUES:
   - Set `default_value` ONLY when the prompt or rules explicitly define a deterministic default when a value is absent in the document (e.g., "use '=' if no operator is stated"). Otherwise leave `default_value = null`.
3. CONSERVATIVE REGEX & BOUNDS:
   - Leave `pattern = null`, `min_value = null`, and `max_value = null` unless there is a strict, universal ontological invariant (such as a fixed symbol/code format or physical bound like 0..100%) that will never reject valid unseen test records.
4. DESCRIPTIONS:
   - Provide a concise, clear `description` for every field summarizing what should be extracted so it can be embedded directly in JSON Schema.
5. Preserve any existing constraints already defined in `extraction_schema`.
</execution_constraints>"""

    extraction_schema: str = dspy.InputField(desc="Current schema definition (.yaml)")
    baseline_prompt: str = dspy.InputField(desc="Original user baseline prompt (.txt)")
    generated_prompt: str = dspy.InputField(desc="Synthesized extraction prompt (.txt)")
    column_profiles: str = dspy.InputField(
        desc="JSON summary of empirical training GT statistics and synthesized column rules per field"
    )
    min_enum_doc_support: int = dspy.InputField(
        desc="Minimum active training document count required before closing a field as an enum"
    )

    refinement: SchemaRefinementOutput = dspy.OutputField(
        desc="Proposed schema constraints per field"
    )
