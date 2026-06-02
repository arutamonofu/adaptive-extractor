from typing import List, Optional
from ae.optimization.contrastive.models import AnalysisResult

DEFAULT_META_RULES = [
    "Literal extraction only: extract values exactly as they appear in the text without changing them.",
    "No calculations: do not calculate, compute, or extrapolate any values.",
    "No unit conversion: do not convert units (e.g., keep original temperature/pressure units).",
    "No inference: do not guess or infer missing values; extract only what is explicitly stated.",
    "Strict locality: extract information only from the provided text content, not from external knowledge.",
    "Missing values: if a value is missing or cannot be found, use null/None."
]


def build_three_level_prompt(
    analysis_result: AnalysisResult,
    meta_rules: Optional[List[str]] = None,
    task_description: str = "Extract structured experiments from the text according to the schema.",
) -> str:
    """Собирает трёхуровневый промпт из результатов контрастивного анализа.
    
    Промпт содержит следующие уровни:
    - [META]: Системные ограничения извлечения.
    - [ENTITY]: Правила отбора/фильтрации строк (какие сущности извлекать).
    - [SCHEMA]: Точные правила валидации и форматирования отдельных полей.
    """
    if meta_rules is None:
        meta_rules = DEFAULT_META_RULES

    prompt_parts = []
    prompt_parts.append(task_description.strip())
    prompt_parts.append("")
    
    prompt_parts.append("[META] (Системные ограничения)")
    for rule in meta_rules:
        prompt_parts.append(f"- {rule}")
    prompt_parts.append("")
    
    prompt_parts.append("[ENTITY] (Критерии фильтрации строк)")
    for rule in analysis_result.entity_level_rules:
        prompt_parts.append(f"- {rule.rule_text}")
    prompt_parts.append("")
    
    prompt_parts.append("[SCHEMA] (Правила валидации полей)")
    for rule in analysis_result.field_level_rules:
        field_label = rule.field_name if rule.field_name else "unknown"
        prompt_parts.append(f"* Поле {field_label}: {rule.rule_text}")
        
    return "\n".join(prompt_parts)
