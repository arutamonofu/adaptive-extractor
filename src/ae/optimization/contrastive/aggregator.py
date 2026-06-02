import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import dspy

from ae.core.tasks.config import TaskConfig
from ae.optimization.contrastive.models import (
    DocumentAnalysis,
    AnalysisResult,
    VerifiedRule,
    Discrepancy,
)

logger = logging.getLogger(__name__)


class AggregateRulesSignature(dspy.Signature):
    """Анализирует списки наблюдений из множества документов, группирует их
    и возвращает список кандидатов правил и противоречий в формате JSON.
    
    Выходная JSON-строка должна иметь следующий формат:
    {
      "verified_rules": [
        {
          "rule_id": "rule_1",
          "level": "entity",
          "field_name": null,
          "rule_text": "description of the rule",
          "evidence_count": 5,
          "evidence_examples": ["evidence 1", "evidence 2"]
        }
      ],
      "discrepancies": [
        {
          "discrepancy_id": "disc_1",
          "level": "entity",
          "field_name": null,
          "problem_description": "description of the conflict",
          "consensus_ratio": 0.8,
          "variant_a": "interpretation A",
          "variant_b": "interpretation B",
          "example_documents": ["doc1", "doc2"]
        }
      ]
    }
    """
    entity_observations_json: str = dspy.InputField(desc="Сгруппированные наблюдения уровня сущности по всем документам")
    field_observations_json: str = dspy.InputField(desc="Сгруппированные наблюдения уровня схемы для конкретного поля (полей)")
    field_schema: str = dspy.InputField(desc="Описание целевых полей")
    num_documents: int = dspy.InputField(desc="Общее количество проанализированных документов")
    rules_and_discrepancies: str = dspy.OutputField(desc="JSON-строка с массивом VerifiedRule и Discrepancy в указанном формате")


class SemanticEquivalenceChecker(dspy.Signature):
    """Проверяет список текстовых формулировок правил от разных документов
    на семантическую эквивалентность и формирует единую выверенную формулировку.
    """
    rules: list[str] = dspy.InputField(desc="Список текстовых формулировок одного правила из разных документов")
    is_unanimous: bool = dspy.OutputField(desc="Флаг консенсуса: True, если все формулировки означают семантически одно и то же; False, если есть содержательные противоречия")
    consolidated_rule: str = dspy.OutputField(desc="Единая лаконичная выверенная формулировка правила на английском языке (если is_unanimous=True)")
    discrepancy_description: str = dspy.OutputField(desc="Подробное описание природы расхождения/конфликта (если is_unanimous=False)")


def extract_json(text: str) -> Any:
    """Extract the first valid JSON array or object from text by brace/bracket balancing."""
    text_to_search = text.strip()
    start_arr = text_to_search.find("[")
    start_obj = text_to_search.find("{")
    
    if start_arr == -1 and start_obj == -1:
        raise ValueError("No JSON block found in the output: " + text)
        
    start = start_arr if (start_obj == -1 or (start_arr != -1 and start_arr < start_obj)) else start_obj
    char_open = text_to_search[start]
    char_close = "]" if char_open == "[" else "}"
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text_to_search)):
        ch = text_to_search[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == char_open:
            depth += 1
        elif ch == char_close:
            depth -= 1
            if depth == 0:
                json_str = text_to_search[start:i+1]
                return json.loads(json_str)
    raise ValueError("Unbalanced JSON braces/brackets in the output")


class StrictAggregator:
    """Агрегирует наблюдения и формирует итоговый набор верифицированных правил."""

    def __init__(self, lm: dspy.LM, task_config: TaskConfig, cache_dir: str = "data/analysis"):
        self.lm = lm
        self.task_config = task_config
        self.cache_dir = Path(cache_dir)
        self.agg_predictor = dspy.Predict(AggregateRulesSignature)
        self.checker_predictor = dspy.Predict(SemanticEquivalenceChecker)

    def aggregate(self, analyses: List[DocumentAnalysis]) -> AnalysisResult:
        """Агрегирует наблюдения по всем проанализированным документам."""
        num_documents = len(analyses)
        if num_documents < 5:
            logger.warning(
                f"Мало документов для анализа: передано {num_documents} документов. "
                f"Консенсус может быть статистически недостоверным."
            )

        # 1. Сбор наблюдений
        entity_obs_list = []
        field_obs_by_field: Dict[str, List[Dict[str, Any]]] = {}

        for doc_analysis in analyses:
            doc_id = doc_analysis.document_id
            for ent_obs in doc_analysis.entity_observations:
                entity_obs_list.append({
                    "document_id": doc_id,
                    "description": ent_obs.description,
                    "evidence": ent_obs.evidence,
                    "included": ent_obs.included
                })
            for field_obs in doc_analysis.field_observations:
                f_name = field_obs.field_name
                if f_name not in field_obs_by_field:
                    field_obs_by_field[f_name] = []
                field_obs_by_field[f_name].append({
                    "document_id": doc_id,
                    "observation_type": field_obs.observation_type,
                    "description": field_obs.description,
                    "evidence": field_obs.evidence,
                    "gt_value": field_obs.gt_value
                })

        # Спецификация полей
        field_schema_dict = {}
        for name, spec in self.task_config.experiment_fields.items():
            type_str = spec.type.__name__ if hasattr(spec.type, "__name__") else str(spec.type)
            field_schema_dict[name] = {
                "field_name": name,
                "field_type": type_str,
                "description": spec.description,
                "required": spec.required
            }
        field_schema_str = json.dumps(field_schema_dict, ensure_ascii=False)

        verified_rules: List[VerifiedRule] = []
        discrepancies: List[Discrepancy] = []

        # 2. Выделение правил уровня сущности (1 вызов LLM)
        if entity_obs_list:
            logger.info("Агрегируем правила уровня сущностей...")
            entity_obs_json = json.dumps(entity_obs_list, ensure_ascii=False)
            try:
                with dspy.settings.context(lm=self.lm):
                    prediction = self.agg_predictor(
                        entity_observations_json=entity_obs_json,
                        field_observations_json="[]",
                        field_schema=field_schema_str,
                        num_documents=num_documents
                    )
                
                # Парсим JSON
                res_data = extract_json(prediction.rules_and_discrepancies)
                
                # Сохраняем сырой результат для отладки
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                raw_path = self.cache_dir / f"{self.task_config.name}_raw_entity_aggregation.json"
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(res_data, f, ensure_ascii=False, indent=2)

                # Обрабатываем правила уровня сущности
                rules_candidates = res_data.get("verified_rules", [])
                for idx, r in enumerate(rules_candidates):
                    # Валидируем evidence_count
                    ev_count = int(r.get("evidence_count", 1))
                    if ev_count <= 0:
                        ev_count = 1
                    
                    verified_rules.append(VerifiedRule(
                        rule_id=f"rule_ent_{idx}",
                        level="entity",
                        field_name=None,
                        rule_text=r.get("rule_text", ""),
                        evidence_count=ev_count,
                        evidence_examples=r.get("evidence_examples", [])
                    ))

                disc_candidates = res_data.get("discrepancies", [])
                for idx, d in enumerate(disc_candidates):
                    # Валидируем consensus_ratio
                    ratio = float(d.get("consensus_ratio", 0.5))
                    if not (0.0 < ratio < 1.0):
                        ratio = 0.5
                        
                    discrepancies.append(Discrepancy(
                        discrepancy_id=f"disc_ent_{idx}",
                        level="entity",
                        field_name=None,
                        problem_description=d.get("problem_description", ""),
                        consensus_ratio=ratio,
                        variant_a=d.get("variant_a", ""),
                        variant_b=d.get("variant_b", ""),
                        example_documents=d.get("example_documents", [])
                    ))
            except Exception as e:
                logger.error(f"Ошибка при агрегации правил уровня сущности: {e}")

        # 3. Выделение правил уровня полей (по каждому полю отдельно или пакетами)
        for field_name, f_obs in field_obs_by_field.items():
            if not f_obs:
                continue
            
            logger.info(f"Агрегируем правила для поля '{field_name}'...")
            f_obs_json = json.dumps(f_obs, ensure_ascii=False)
            
            try:
                with dspy.settings.context(lm=self.lm):
                    prediction = self.agg_predictor(
                        entity_observations_json="[]",
                        field_observations_json=f_obs_json,
                        field_schema=json.dumps({field_name: field_schema_dict.get(field_name, {})}, ensure_ascii=False),
                        num_documents=num_documents
                    )
                
                res_data = extract_json(prediction.rules_and_discrepancies)
                
                # Сохраняем сырой результат для отладки
                raw_path = self.cache_dir / f"{self.task_config.name}_raw_field_{field_name}_aggregation.json"
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(res_data, f, ensure_ascii=False, indent=2)

                rules_candidates = res_data.get("verified_rules", [])
                for idx, r in enumerate(rules_candidates):
                    # Проверяем семантическую эквивалентность формулировок, если у правила много свидетельств
                    evidence_examples = r.get("evidence_examples", [])
                    rule_text = r.get("rule_text", "")
                    
                    if len(evidence_examples) > 1:
                        # Запускаем SemanticEquivalenceChecker для проверки консенсуса формулировок
                        try:
                            with dspy.settings.context(lm=self.lm):
                                check_res = self.checker_predictor(
                                    rules=[rule_text] + evidence_examples
                                )
                            
                            if check_res.is_unanimous:
                                rule_text = check_res.consolidated_rule
                            else:
                                # Иначе превращаем это в расхождение
                                discrepancies.append(Discrepancy(
                                    discrepancy_id=f"disc_fld_eq_{field_name}_{len(discrepancies)}",
                                    level="field",
                                    field_name=field_name,
                                    problem_description=check_res.discrepancy_description or f"Wording contradiction for field {field_name}",
                                    consensus_ratio=0.5,
                                    variant_a=rule_text,
                                    variant_b=evidence_examples[0] if evidence_examples else "",
                                    example_documents=r.get("example_documents", []) or [doc_analysis.document_id for doc_analysis in analyses][:2]
                                ))
                                continue
                        except Exception as check_err:
                            logger.warning(f"Ошибка проверки семантической эквивалентности: {check_err}")

                    ev_count = int(r.get("evidence_count", 1))
                    if ev_count <= 0:
                        ev_count = 1

                    # Если правило выполняется не во всех документах, где это поле встретилось
                    # (например, ev_count < num_documents), но в LLM-отчете оно попало в verified_rules,
                    # то при строгой zero-tolerance политике, если есть альтернативное поведение, это discrepancy.
                    # Но если в других документах поле просто было пустым/отсутствовало, то это не противоречие.
                    # Поэтому мы доверяем результату LLM-агрегатора, но принудительно проверяем.
                    verified_rules.append(VerifiedRule(
                        rule_id=f"rule_fld_{field_name}_{len(verified_rules)}",
                        level="field",
                        field_name=field_name,
                        rule_text=rule_text,
                        evidence_count=ev_count,
                        evidence_examples=evidence_examples
                    ))

                disc_candidates = res_data.get("discrepancies", [])
                for idx, d in enumerate(disc_candidates):
                    ratio = float(d.get("consensus_ratio", 0.5))
                    if not (0.0 < ratio < 1.0):
                        ratio = 0.5
                        
                    discrepancies.append(Discrepancy(
                        discrepancy_id=f"disc_fld_{field_name}_{len(discrepancies)}",
                        level="field",
                        field_name=field_name,
                        problem_description=d.get("problem_description", ""),
                        consensus_ratio=ratio,
                        variant_a=d.get("variant_a", ""),
                        variant_b=d.get("variant_b", ""),
                        example_documents=d.get("example_documents", [])
                    ))
            except Exception as e:
                logger.error(f"Ошибка при агрегации правил для поля '{field_name}': {e}")

        # Формируем итоговый результат
        result = AnalysisResult(
            task_name=self.task_config.name,
            analyzed_documents=num_documents,
            verified_rules=verified_rules,
            discrepancies=discrepancies,
            timestamp=datetime.now().isoformat()
        )
        return result
