# Финальное Техническое Задание: Universal Row-Level Semantic Judge (Rev. 2)

## 0. Сводная таблица изменений

| Компонент | Изменение | Примечание |
|-----------|-----------|------------|
| `TaskConfig` | Добавить `@property field_descriptions` | Инкапсуляция извлечения описаний |
| `ExperimentMatcher.__init__` | Добавить `teacher_llm`, `field_descriptions`, `enable_semantic_judge` | Новые зависимости |
| `ExperimentMatcher._normalize_text` | Убрать `.lower()` | Case-sensitive сравнение |
| `ExperimentMatcher._compare_floats` | Заменить на `math.isclose(rel_tol=1e-9)` | Строгая проверка |
| `TaskMetric.__init__` | Добавить `teacher_llm`, `task_config`, `enable_semantic_judge` | Передача зависимостей |
| `optimize.py` | Передать `teacher_llm` и флаг в `TaskMetric` | Точка интеграции |
| Конфиг YAML | Добавить `evaluation.enable_semantic_judge` | Переключатель режима |

---

## 1. Модификация `TaskConfig` (src/aee/domain/tasks/config.py)

Добавить свойство для извлечения описаний полей:

```python
@property
def field_descriptions(self) -> Dict[str, str]:
    """Get field descriptions for semantic judge.
    
    Returns:
        Dictionary mapping field names to their descriptions.
        Only includes fields that have a description.
    """
    return {
        name: spec.description
        for name, spec in self.experiment_fields.items()
        if spec.description
    }
```

---

## 2. Модификация `ExperimentMatcher` (src/aee/domain/evaluation/matcher.py)

### 2.1. Обновление конструктора

```python
def __init__(
    self,
    fields_to_compare: List[str],
    float_tolerance: float,
    teacher_llm: Optional[Any] = None,
    field_descriptions: Optional[Dict[str, str]] = None,
    enable_semantic_judge: bool = True,
):
    """Initialize the ExperimentMatcher.
    
    Args:
        fields_to_compare: List of field names to compare between entities.
        float_tolerance: Tolerance for float comparisons (0.0 to 1.0).
                        Kept for backward compatibility, not used in strict mode.
        teacher_llm: DSPy LLM object for semantic judgment (optional).
        field_descriptions: Dictionary of field descriptions (optional).
        enable_semantic_judge: Flag to enable/disable semantic judge (default: True).
    """
    if not fields_to_compare:
        raise ValueError("fields_to_compare cannot be empty")
    if not 0 <= float_tolerance <= 1:
        raise ValueError("float_tolerance must be between 0 and 1")

    self.fields = fields_to_compare
    self.tolerance = float_tolerance  # Kept but unused
    self.teacher_llm = teacher_llm
    self.field_descriptions = field_descriptions or {}
    self.enable_semantic_judge = enable_semantic_judge
```

### 2.2. Изменение `_normalize_text` (Case-Sensitive)

```python
def _normalize_text(self, value: Any) -> str:
    """Normalize input values for comparison.
    
    Handles dash artifacts and whitespace. Case-sensitive.
    
    Args:
        value: Input value to normalize.
        
    Returns:
        Normalized string value (case-preserved).
    """
    if value is None:
        return ""
    
    # Convert to string, normalize dashes, remove whitespace
    # NOTE: No .lower() - case-sensitive comparison
    return self._RE_STRICT_CLEAN.sub("", str(value).translate(self._DASH_MAP))
```

### 2.3. Изменение `_compare_floats` (Strict Mode)

```python
import math

def _compare_floats(self, val_pred: float, val_gold: float) -> bool:
    """Compare two float values with strict tolerance.
    
    Uses math.isclose with relative tolerance 1e-9.
    
    Args:
        val_pred: Predicted float value.
        val_gold: Ground truth float value.
        
    Returns:
        True if values are close, False otherwise.
    """
    return math.isclose(val_pred, val_gold, rel_tol=1e-9)
```

### 2.4. Новые методы для Semantic Judge

#### Метод `_build_judge_prompt`

```python
def _build_judge_prompt(
    self,
    task_name: str,
    gt_json: Dict[str, Any],
    pred_json: Dict[str, Any],
    discrepancies: List[str],
) -> str:
    """Build prompt for semantic judge.
    
    Args:
        task_name: Name of the task (e.g., "nanozymes").
        gt_json: Ground truth experiment as dictionary (primitive types only).
        pred_json: Predicted experiment as dictionary (primitive types only).
        discrepancies: List of field names with mismatches.
        
    Returns:
        Formatted prompt string.
    """
    # Build schema context from field descriptions
    schema_lines = []
    for field_name, description in self.field_descriptions.items():
        schema_lines.append(f"- {field_name}: {description}")
    schema_context = "\n".join(schema_lines)
    
    # Build discrepancies list with values
    discrepancy_lines = []
    for field_name in discrepancies:
        gt_val = gt_json.get(field_name)
        pred_val = pred_json.get(field_name)
        gt_str = "null" if gt_val is None else str(gt_val)
        pred_str = "null" if pred_val is None else str(pred_val)
        discrepancy_lines.append(f"- {field_name}: GT='{gt_str}', Pred='{pred_str}'")
    discrepancies_text = "\n".join(discrepancy_lines)
    
    prompt = f"""You are an expert scientist evaluating an automated data extraction system.
Task: {task_name}

Schema Definition (Field Meanings):
{schema_context}

--- CONTEXT (Full Experiments) ---
Ground Truth (Reference):
{json.dumps(gt_json, indent=2)}

Predicted (Extraction):
{json.dumps(pred_json, indent=2)}

--- DISCREPANCIES TO EVALUATE ---
The following fields did not match strictly. Evaluate ONLY these fields based on the context above:
{discrepancies_text}

--- INSTRUCTIONS ---
For EACH discrepancy, determine if the Predicted value is SEMANTICALLY EQUIVALENT to Ground Truth.
Acceptable differences:
1. Implicit Values: Null in Predicted is logically implied by other fields in Context (e.g., width=null for cubic).
2. Physical Equivalence: Different units but same value (e.g., 50 uM == 0.05 mM).
3. Synonyms/Order: "H2O2 + TMB" == "TMB + H2O2" (unless roles differ).
4. Case Variations: "Cubic" == "cubic" for crystal systems.

Response Format: JSON ONLY where keys are field names and values are "YES" (Accept) or "NO" (Reject).
Example: {{"width": "YES", "reaction_type": "NO"}}

Respond with JSON only:"""
    
    return prompt
```

#### Метод `_call_semantic_judge`

```python
def _call_semantic_judge(
    self,
    task_name: str,
    gt_json: Dict[str, Any],
    pred_json: Dict[str, Any],
    discrepancies: List[str],
) -> Dict[str, str]:
    """Call semantic judge LLM and parse response.
    
    Args:
        task_name: Name of the task.
        gt_json: Ground truth experiment as dictionary.
        pred_json: Predicted experiment as dictionary.
        discrepancies: List of field names with mismatches.
        
    Returns:
        Dictionary mapping field names to "YES" or "NO".
        Empty dict if LLM call fails (fallback to strict).
    """
    if not self.enable_semantic_judge:
        logger.debug("[SemanticJudge] Disabled, skipping evaluation")
        return {}
    
    if self.teacher_llm is None:
        logger.warning("[SemanticJudge] teacher_llm not provided, skipping evaluation")
        return {}
    
    try:
        # Build prompt
        prompt = self._build_judge_prompt(task_name, gt_json, pred_json, discrepancies)
        
        # Call LLM via DSPy interface
        # DSPy LM returns list of strings, take first
        response = self.teacher_llm(prompt)
        response_text = response[0] if isinstance(response, list) else response
        
        # Parse JSON response
        verdicts = json.loads(response_text)
        
        # Validate and filter verdicts
        valid_verdicts = {}
        for field_name in discrepancies:
            if field_name in verdicts:
                # Only accept "YES", everything else is "NO"
                valid_verdicts[field_name] = "YES" if verdicts[field_name] == "YES" else "NO"
            else:
                # Field not in response, treat as NO
                valid_verdicts[field_name] = "NO"
        
        logger.info(
            f"[SemanticJudge] Checked {len(discrepancies)} fields -> "
            f"{{' + '.join(f'{k}: {v}' for k, v in valid_verdicts.items())}}"
        )
        
        return valid_verdicts
        
    except json.JSONDecodeError as e:
        logger.warning(f"[SemanticJudge] JSON parse error: {e}")
        return {}
    except Exception as e:
        logger.warning(f"[SemanticJudge] Failed: {e}")
        return {}
```

### 2.5. Обновление `_compute_stats` (ИСПРАВЛЕНО)

**Критическая правка:** Корректная логика начисления штрафов в зависимости от типа расхождения.

```python
def _compute_stats(
    self,
    pairs: List[Tuple[Optional[Any], Optional[Any]]],
    task_name: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate Micro-F1/Precision/Recall with semantic judge fallback.
    
    Args:
        pairs: List of aligned pairs (pred, gt).
        task_name: Optional task name for semantic judge context.
        
    Returns:
        Dict with precision, recall, and f1 scores.
    """
    tp, fp, fn = 0, 0, 0
    
    for pred, gold in pairs:
        # Case 3: False Negative (Missing Experiment)
        if pred is None and gold is not None:
            fn += sum(1 for f in self.fields if getattr(gold, f, None) is not None)
            continue
        
        # Case 2: False Positive (Hallucinated Experiment)
        if gold is None and pred is not None:
            fp += sum(1 for f in self.fields if getattr(pred, f, None) is not None)
            continue
        
        # Case 1: Aligned Experiment - Check field-wise
        strict_matches = []
        discrepancies = []
        
        for f in self.fields:
            val_p = getattr(pred, f, None)
            val_g = getattr(gold, f, None)
            
            if val_g is None and val_p is None:
                continue  # True Negative (Ignore)
            
            if val_g is not None and val_p is None:
                discrepancies.append(f)  # Missing value (Pure FN candidate)
            elif val_g is None and val_p is not None:
                discrepancies.append(f)  # Hallucinated value (Pure FP candidate)
            else:
                # Both present, check strict equality
                if self._is_match(val_p, val_g):
                    strict_matches.append(f)
                else:
                    discrepancies.append(f)  # Mismatch (FP + FN candidate)
        
        # Count strict matches as TP
        tp += len(strict_matches)
        
        # Handle discrepancies
        if discrepancies and self.enable_semantic_judge:
            # Convert to JSON for judge (ensure primitive types only)
            gt_json = {f: getattr(gold, f, None) for f in self.fields}
            pred_json = {f: getattr(pred, f, None) for f in self.fields}
            
            # Call semantic judge
            verdicts = self._call_semantic_judge(
                task_name=task_name or "unknown",
                gt_json=gt_json,
                pred_json=pred_json,
                discrepancies=discrepancies,
            )
            
            # Apply verdicts with correct penalty logic
            for field_name in discrepancies:
                verdict = verdicts.get(field_name, "NO")
                
                if verdict == "YES":
                    tp += 1  # Amnesty granted
                else:
                    # Вердикт NO - возвращаемся к исходной природе ошибки
                    val_p = getattr(pred, field_name, None)
                    val_g = getattr(gold, field_name, None)
                    
                    if val_p is None and val_g is not None:
                        fn += 1  # Pure Miss (модель промолчала)
                    elif val_p is not None and val_g is None:
                        fp += 1  # Pure Hallucination (модель придумала)
                    else:
                        # Mismatch (wrong value: модель ошиблась значением)
                        fp += 1
                        fn += 1
        else:
            # No discrepancies or judge disabled - apply strict penalties
            for field_name in discrepancies:
                val_p = getattr(pred, field_name, None)
                val_g = getattr(gold, field_name, None)
                
                if val_p is None and val_g is not None:
                    fn += 1  # Pure Miss
                elif val_p is not None and val_g is None:
                    fp += 1  # Pure Hallucination
                else:
                    # Mismatch
                    fp += 1
                    fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

### 2.6. Обновление публичных методов

```python
def get_optimization_score(
    self,
    preds: List[ExperimentEntity],
    gts: List[ExperimentEntity],
    task_name: Optional[str] = None,
) -> float:
    """Get optimization score (F1) for use in teleprompter.
    
    Args:
        preds: List of predicted experiment entities.
        gts: List of ground truth experiment entities.
        task_name: Optional task name for semantic judge context.
        
    Returns:
        F1 score.
    """
    pairs = self.align_pairs(preds, gts)
    return self._compute_stats(pairs, task_name)["f1"]

def get_detailed_report(
    self,
    preds: List[ExperimentEntity],
    gts: List[ExperimentEntity],
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Get detailed evaluation report.
    
    Args:
        preds: List of predicted experiment entities.
        gts: List of ground truth experiment entities.
        task_name: Optional task name for semantic judge context.
        
    Returns:
        Dict with detailed evaluation metrics.
    """
    pairs = self.align_pairs(preds, gts)
    stats = self._compute_stats(pairs, task_name)
    
    # Calculate per-field scores (existing logic, unchanged)
    field_scores = {}
    for field in self.fields:
        correct = 0
        total = 0
        for p, g in pairs:
            val_p = getattr(p, field, None) if p else None
            val_g = getattr(g, field, None) if g else None

            if val_g is not None:
                total += 1
                if self._is_match(val_p, val_g):
                    correct += 1
            elif val_p is not None:
                total += 1  # False Positive field

        field_scores[field] = correct / total if total > 0 else 1.0

    return {
        "f1": stats["f1"],
        "precision": stats["precision"],
        "recall": stats["recall"],
        "fields": field_scores,
        "counts": {"preds": len(preds), "gts": len(gts)}
    }
```

---

## 3. Модификация `TaskMetric` (src/aee/domain/evaluation/metrics.py)

### 3.1. Обновление конструктора

**Важно:** `field_descriptions` передаются явным аргументом, а не извлекаются внутри.

```python
class TaskMetric:
    """Task-specific evaluation metric with semantic judge support."""
    
    def __init__(
        self,
        task_config: Dict[str, Any],
        float_tolerance: float,
        teacher_llm: Optional[Any] = None,
        field_descriptions: Optional[Dict[str, str]] = None,
        enable_semantic_judge: bool = True,
    ) -> None:
        """Initialize the task metric.
        
        Args:
            task_config: Configuration dictionary for the task.
                        Must contain 'compare_fields' and 'name' keys.
            float_tolerance: Float tolerance (kept for compatibility).
            teacher_llm: DSPy LLM object for semantic judgment.
            field_descriptions: Dictionary of field descriptions (optional).
            enable_semantic_judge: Flag to enable/disable semantic judge.
        """
        self.matcher = ExperimentMatcher(
            fields_to_compare=task_config["compare_fields"],
            float_tolerance=float_tolerance,
            teacher_llm=teacher_llm,
            field_descriptions=field_descriptions or {},
            enable_semantic_judge=enable_semantic_judge,
        )
        self.fields_to_compare = task_config["compare_fields"]
        self.task_name = task_config.get("name", "unknown")
```

### 3.2. Обновление `__call__`

```python
def __call__(self, example: dspy.Example, prediction: dspy.Prediction, trace: Any = None) -> float:
    """Calculate the metric score for a prediction.
    
    Args:
        example: Ground truth example containing extracted_data.experiments.
        prediction: Predicted result containing extracted_data.experiments.
        trace: Optional trace information (unused).
        
    Returns:
        float: F1 score metric (0.0 to 1.0).
    """
    try:
        # Extract experiments from ground truth and prediction
        ground_truth_experiments = self._extract_experiments(example)
        predicted_experiments = self._extract_experiments(prediction)
        
        # Calculate detailed metrics using ExperimentMatcher
        report = self.matcher.get_detailed_report(
            predicted_experiments,
            ground_truth_experiments,
            task_name=self.task_name,
        )
        score = report["f1"]
        
        # Log detailed metrics if logger is enabled for INFO level
        if logger.isEnabledFor(logging.INFO):
            self._log_metrics(report)
        
        return score
        
    except (AttributeError, KeyError, TypeError) as e:
        logger.error(f"Error in metric calculation: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error in metric calculation: {e}")
        return 0.0
```

---

## 4. Интеграция в `optimize.py` (src/aee/interface/cli/optimize.py)

### 4.1. Обновление `setup_language_models`

Функция уже возвращает `teacher_lm`. Убедиться, что она не `None`.

### 4.2. Создание `TaskMetric` с зависимостями

Найти место создания `TaskMetric` (в `OptimizeAgentUseCase` или связанном коде) и обновить:

```python
# В OptimizeAgentUseCase или где создается метрика
task_metric = TaskMetric(
    task_config=request.task,  # Dict или TaskConfig
    float_tolerance=request.task.get("float_tolerance", 0.05),
    teacher_llm=request.teacher_lm,  # Передаем из запроса
    field_descriptions=request.task.get("field_descriptions", {}),  # Из TaskConfig
    enable_semantic_judge=settings.task.evaluation.enable_semantic_judge,
)
```

**Примечание:** Требуется проверить, где именно в коде создается `TaskMetric`, и обновить эту точку.

---

## 5. Конфигурация YAML

### 5.1. Обновление `config/systems/example.yaml`

Добавить в раздел `task.evaluation`:

```yaml
task:
  name: "nanozymes"
  initial_instruction_file: "config/initial_instructions/nanozymes_sota.txt"
  evaluation:
    compare_fields:
      - "formula"
      - "activity"
      # ... другие поля
    float_tolerance: 0.05
    enable_semantic_judge: true  # Новый флаг (default: true)
```

---

## 6. План реализации (Checklist)

### Этап 1: Подготовка (Day 1) ✅
- [x] Добавить `@property field_descriptions` в `TaskConfig`
- [x] Добавить флаг `enable_semantic_judge` в `Settings` (секция `task.evaluation`)
- [x] Обновить `ExperimentMatcher.__init__` с новыми параметрами

### Этап 2: Ядро Semantic Judge (Day 2-3) ✅
- [x] Изменить `_normalize_text` (убрать `.lower()`)
- [x] Изменить `_compare_floats` на `math.isclose`
- [x] Реализовать `_build_judge_prompt`
- [x] Реализовать `_call_semantic_judge` с try/except
- [x] Обновить `_compute_stats` с **исправленной** логикой штрафов

### Этап 3: Интеграция (Day 4) ✅
- [x] Обновить `TaskMetric.__init__`
- [x] Обновить `TaskMetric.__call__` с передачей `task_name`
- [x] Интегрировать передачу `teacher_llm` и `field_descriptions` из `optimize.py`
- [x] Добавить логирование с префиксом `[SemanticJudge]`

### Этап 4: Тестирование (Day 5-6) ✅
- [x] Unit-тест: Идеальное совпадение → LLM не вызывается
- [x] Unit-тест: Числовое расхождение → LLM YES → Score 1.0
- [x] Unit-тест: Числовое расхождение → LLM NO → Score падает
- [x] Unit-тест: Ошибка LLM → Fallback to Strict
- [x] Unit-тест: Case-sensitive проверка (`Co` != `co`)
- [x] Unit-тест: Missing value (`Pred=None`) → только FN, не FP+FN
- [x] Unit-тест: Hallucination (`GT=None`) → только FP, не FP+FN
- [ ] Integration-тест: Полный цикл optimize.py

---

## 7. Таблица штрафов (Reference)

| Ситуация | Pred | GT | Судья | Итог |
|----------|------|----|-------|------|
| Идеальное совпадение | `10` | `10` | — | TP |
| Missing Value | `None` | `10` | NO | FN |
| Missing Value | `None` | `10` | YES | TP |
| Hallucination | `10` | `None` | NO | FP |
| Hallucination | `10` | `None` | YES | TP |
| Mismatch | `5` | `10` | NO | FP + FN |
| Mismatch | `5` | `10` | YES | TP |

---

## 8. Статус реализации

**Реализация завершена:** ✅

### Изменённые файлы:
1. `src/aee/domain/tasks/config.py` — добавлено свойство `field_descriptions`
2. `src/aee/infrastructure/config/settings.py` — добавлен флаг `enable_semantic_judge`
3. `src/aee/domain/evaluation/matcher.py` — ядро Semantic Judge
4. `src/aee/domain/evaluation/metrics.py` — интеграция с TaskMetric
5. `src/aee/application/use_cases/optimize_agent.py` — передача зависимостей
6. `config/systems/example.yaml` — включён Semantic Judge
7. `config/systems/test_gemini-3-flash.yaml` — включён Semantic Judge
8. `tests/unit/domain/test_matcher.py` — 12 новых тестов

### Оставшиеся задачи:
- [ ] Integration test: полный цикл optimize.py (опционально)

---

**ТЗ утверждено. Реализация завершена.**
