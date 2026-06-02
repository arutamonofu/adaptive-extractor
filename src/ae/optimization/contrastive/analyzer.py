import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
import dspy
from pydantic import ValidationError

from ae.core.tasks.config import TaskConfig
from ae.optimization.contrastive.models import AnalysisInput, DocumentAnalysis, FieldSpecSummary

logger = logging.getLogger(__name__)


class AnalyzeDocumentSignature(dspy.Signature):
    """Анализирует научную статью в сравнении с Ground Truth данными её разметки.
    Выявляет правила, по которым строки включались в GT (entity-level),
    а также особенности извлечения и форматирования конкретных полей (field-level).
    """
    article_text: str = dspy.InputField(desc="Полный текст научной статьи в формате Markdown")
    ground_truth_json: str = dspy.InputField(desc="JSON-массив экспериментов из Ground Truth для этой статьи")
    field_schema: str = dspy.InputField(desc="Спецификация полей извлечения в формате JSON (имена, типы, ограничения)")
    analysis: DocumentAnalysis = dspy.OutputField(desc="Структурированный результат анализа (DocumentAnalysis)")


class LocalAnalyzer:
    """Анализирует отдельную пару документ-GT для извлечения наблюдений (observations) с гарантией валидности JSON и кэшированием."""

    def __init__(self, lm: dspy.LM, task_config: TaskConfig, cache_dir: str = "data/analysis", rate_limit_delay: float = 10.0):
        self.lm = lm  # Используется teacher LM из конфигурации (qwen3.5-397b-a17b)
        self.task_config = task_config
        self.cache_dir = Path(cache_dir)
        self.rate_limit_delay = rate_limit_delay
        # Использование Predict с Pydantic-аннотациями в сигнатуре
        self.predictor = dspy.Predict(AnalyzeDocumentSignature)

    async def analyze(self, input_data: AnalysisInput) -> DocumentAnalysis:
        """Выполняет LLM-анализ одного документа с поддержкой кэширования и повторных запросов."""
        task_name = self.task_config.name
        cache_path = self.cache_dir / f"{task_name}_map_{input_data.document_id}.json"
        
        # 1. Проверяем наличие валидного локального кэша перед вызовом LLM
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                validated_data = DocumentAnalysis.model_validate(cached_data)
                logger.info(f"Использован кэш для документа {input_data.document_id} из {cache_path}")
                return validated_data
            except Exception as e:
                logger.warning(
                    f"Кэш для документа {input_data.document_id} в {cache_path} поврежден или невалиден: {e}. "
                    f"Запускаем повторный анализ через LLM."
                )

        if self.rate_limit_delay > 0:
            logger.info(f"Ожидание {self.rate_limit_delay} секунд перед запросом к LLM для {input_data.document_id}...")
            await asyncio.sleep(self.rate_limit_delay)

        field_schema_str = json.dumps(
            {k: v.model_dump() for k, v in input_data.field_specs.items()},
            ensure_ascii=False
        )
        gt_json_str = json.dumps(input_data.ground_truth_experiments, ensure_ascii=False)
        
        max_retries = 2
        validation_error_feedback = ""
        
        for attempt in range(max_retries + 1):
            try:
                # Настраиваем параметры генерации с учетом контекста и потенциальной ошибки
                with dspy.settings.context(lm=self.lm):
                    if validation_error_feedback:
                        # В случае ошибки на предыдущем шаге, просим модель исправить её
                        prompt_instructions = (
                            f"\n[ВНИМАНИЕ]: Твой предыдущий ответ вызвал ошибку валидации:\n"
                            f"{validation_error_feedback}\n"
                            f"Пожалуйста, проанализируй ошибку, исправь структуру JSON и "
                            f"верни строго валидный JSON, соответствующий схеме DocumentAnalysis."
                        )
                        prediction = self.predictor(
                            article_text=input_data.document_text + prompt_instructions,
                            ground_truth_json=gt_json_str,
                            field_schema=field_schema_str
                        )
                    else:
                        prediction = self.predictor(
                            article_text=input_data.document_text,
                            ground_truth_json=gt_json_str,
                            field_schema=field_schema_str
                        )
                
                # Дополнительная строгая рантайм-валидация через Pydantic
                analysis_result = prediction.analysis
                if isinstance(analysis_result, DocumentAnalysis):
                    validated_data = analysis_result
                elif isinstance(analysis_result, dict):
                    validated_data = DocumentAnalysis.model_validate(analysis_result)
                elif isinstance(analysis_result, str):
                    from ae.optimization.contrastive.aggregator import extract_json
                    parsed_dict = extract_json(analysis_result)
                    validated_data = DocumentAnalysis.model_validate(parsed_dict)
                else:
                    validated_data = DocumentAnalysis.model_validate(analysis_result)
                
                # 2. Атомарное сохранение результата в кэш сразу после успешной генерации
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temp_path = cache_path.with_suffix(".tmp")
                try:
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(validated_data.model_dump(), f, ensure_ascii=False, indent=2)
                    temp_path.replace(cache_path)
                    logger.info(f"Анализ документа {input_data.document_id} успешно записан в кэш: {cache_path}")
                except Exception as e:
                    logger.error(f"Не удалось записать кэш для документа {input_data.document_id}: {e}")
                    if temp_path.exists():
                        temp_path.unlink()
                
                return validated_data
                
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Попытка {attempt + 1}/{max_retries + 1} для {input_data.document_id} "
                    f"завершилась ошибкой валидации: {e}"
                )
                validation_error_feedback = f"Ошибка валидации Pydantic/JSON: {str(e)}"
                
                if attempt == max_retries:
                    logger.error(
                        f"Не удалось получить валидный JSON для {input_data.document_id} после {max_retries} повторов."
                    )
                    raise e


class ContrastiveMapRunner:
    """Запускает Map-фазу пакетно для списка документов с ограничением конкурентности."""

    def __init__(self, analyzer: LocalAnalyzer, max_concurrent: int = 1):
        self.analyzer = analyzer
        self.max_concurrent = max_concurrent  # По умолчанию 1 из-за жестких rate-limits у API

    async def run_batch(
        self,
        inputs: List[AnalysisInput],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DocumentAnalysis]:
        """Последовательно или конкурентно (с семафором) запускает анализ документов."""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run_one(inp: AnalysisInput, idx: int):
            async with semaphore:
                try:
                    res = await self.analyzer.analyze(inp)
                    results.append(res)
                except Exception as e:
                    logger.error(f"Ошибка при анализе документа {inp.document_id}: {e}")
                    # Возвращаем частичный результат с описанием ошибки
                    results.append(DocumentAnalysis(
                        document_id=inp.document_id,
                        entity_observations=[],
                        field_observations=[],
                        summary=f"Ошибка анализа: {str(e)}"
                    ))
                finally:
                    if progress_callback:
                        progress_callback(len(results), len(inputs))

        # Запускаем таски
        tasks = [_run_one(inp, i) for i, inp in enumerate(inputs)]
        await asyncio.gather(*tasks)
        return results


def prepare_analysis_inputs(
    task_config: TaskConfig,
    document_ids: List[str],
    documents: Dict[str, str],            # doc_id -> markdown_text
    gt_data: Dict[str, List[Dict[str, Any]]],  # doc_id -> list of GT rows
) -> List[AnalysisInput]:
    """Преобразует сырые данные из репозиториев во входные структуры AnalysisInput."""
    field_specs = {
        name: FieldSpecSummary.from_field_spec(name, spec)
        for name, spec in task_config.experiment_fields.items()
    }
    
    inputs = []
    for doc_id in document_ids:
        if doc_id not in documents:
            logger.warning(f"Документ {doc_id} не найден в parsed_dir, пропускаем")
            continue
            
        doc_gt = gt_data.get(doc_id, [])
        inputs.append(AnalysisInput(
            document_id=doc_id,
            document_text=documents[doc_id],
            ground_truth_experiments=doc_gt,
            field_specs=field_specs
        ))
    return inputs
