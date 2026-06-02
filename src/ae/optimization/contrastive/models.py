import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class FieldSpecSummary(BaseModel):
    """Облегченное представление FieldSpec для контекста LLM."""
    model_config = ConfigDict(frozen=False)

    field_name: str
    field_type: str  # 'str', 'float', 'int', etc.
    required: bool
    choices: Optional[List[str]] = None

    @classmethod
    def from_field_spec(cls, name: str, spec: Any) -> "FieldSpecSummary":
        """Создает FieldSpecSummary из существующего FieldSpec в ae.core.tasks.config."""
        # spec.type может быть типом или строкой
        type_str = spec.type.__name__ if hasattr(spec.type, "__name__") else str(spec.type)
        return cls(
            field_name=name,
            field_type=type_str,
            required=spec.required,
            choices=spec.choices
        )


class AnalysisInput(BaseModel):
    """Входные данные для конвейера контрастивного анализа."""
    model_config = ConfigDict(frozen=False)

    document_id: str  # например, 'c6ra00963h'
    document_text: str  # полный текст статьи в формате markdown
    ground_truth_experiments: List[Dict[str, Any]]  # эксперименты из GT в виде словарей
    field_specs: Dict[str, FieldSpecSummary]  # имя поля -> краткая спецификация


class EntityObservation(BaseModel):
    """Единичное наблюдение о включении/исключении сущности (строки)."""
    model_config = ConfigDict(frozen=False)

    description: str  # описание наблюдения
    evidence: str  # цитата или ссылка из статьи
    included: bool  # True = сущность включена в GT, False = исключена (пропущена)


class FieldObservation(BaseModel):
    """Единичное наблюдение о качестве извлечения/форматирования поля."""
    model_config = ConfigDict(frozen=False)

    field_name: str
    observation_type: Literal['extraction_included', 'extraction_excluded', 'format_applied', 'format_rejected']
    description: str
    evidence: str  # цитата из текста или значение из GT
    gt_value: Optional[str] = None  # фактическое значение GT, если применимо


class DocumentAnalysis(BaseModel):
    """Полный результат анализа для одной пары документ-GT."""
    model_config = ConfigDict(frozen=False)

    document_id: str
    entity_observations: List[EntityObservation]
    field_observations: List[FieldObservation]
    summary: str  # краткое резюме ключевых находок


class VerifiedRule(BaseModel):
    """Правило со 100% консенсусом во всех проанализированных документах."""
    model_config = ConfigDict(frozen=False)

    rule_id: str  # автогенерируемый уникальный ID
    level: Literal['entity', 'field']  # ENTITY или SCHEMA уровень
    field_name: Optional[str] = None  # None для правил уровня сущности
    rule_text: str  # формулировка правила для промпта
    evidence_count: int = Field(..., gt=0)  # количество документов, подтверждающих правило
    evidence_examples: List[str]  # 2-3 примера цитат/доказательств


class Discrepancy(BaseModel):
    """Паттерн с консенсусом < 100%, требующий ручного разбора (Human Review)."""
    model_config = ConfigDict(frozen=False)

    discrepancy_id: str
    level: Literal['entity', 'field']
    field_name: Optional[str] = None
    problem_description: str
    consensus_ratio: float = Field(..., gt=0.0, lt=1.0)  # например, 0.8 означает 80% согласия
    variant_a: str  # одна интерпретация
    variant_b: str  # альтернативная интерпретация
    example_documents: List[str]  # ID документов, показывающих конфликт


class AnalysisResult(BaseModel):
    """Полный выходной результат конвейера контрастивного анализа."""
    model_config = ConfigDict(frozen=False)

    task_name: str
    analyzed_documents: int
    verified_rules: List[VerifiedRule]
    discrepancies: List[Discrepancy]
    timestamp: str

    @property
    def entity_level_rules(self) -> List[VerifiedRule]:
        """Удобное свойство: отфильтрованные правила уровня сущности."""
        return [rule for rule in self.verified_rules if rule.level == 'entity']

    @property
    def field_level_rules(self) -> List[VerifiedRule]:
        """Удобное свойство: отфильтрованные правила уровня полей."""
        return [rule for rule in self.verified_rules if rule.level == 'field']

    def has_discrepancies(self) -> bool:
        return len(self.discrepancies) > 0

    def get_rules_for_field(self, field_name: str) -> List[VerifiedRule]:
        return [rule for rule in self.verified_rules if rule.field_name == field_name]

    def to_json(self, path: Path) -> None:
        """Сериализация результата анализа в JSON файл с сохранением UTF-8."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "AnalysisResult":
        """Десериализация результата анализа из JSON файла."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


class HumanDecision(BaseModel):
    """Решение пользователя по конкретному расхождению."""
    model_config = ConfigDict(frozen=False)

    discrepancy_id: str
    action: Literal['accept_a', 'accept_b', 'custom_rule', 'skip']
    custom_rule_text: Optional[str] = None  # если action == 'custom_rule'


class ReviewSession(BaseModel):
    """Результат завершенной интерактивной сессии разбора расхождений."""
    model_config = ConfigDict(frozen=False)

    decisions: List[HumanDecision]
    resolved_rules: List[VerifiedRule]  # правила, созданные на основе решений пользователя
    skipped_count: int
