# Руководство пользователя и разработчика

Практический справочник: настройка окружения, запуск конвейера (`CLI`), основные параметры `config/*.yaml`, расширение под новый домен, логи, MLflow и тестирование.

### Содержание
- [1. Окружение и патчи DSPy](#1-окружение-и-патчи-dspy)
- [2. Подготовка данных и справочник CLI](#2-подготовка-данных-и-справочник-cli)
- [3. Справочник конфигураций (`config/*.yaml`)](#3-справочник-конфигураций-configyaml)
- [4. Адаптация под новый домен и расширение](#4-адаптация-под-новый-домен-и-расширение)
- [5. Логи, MLflow, скрипты и тесты](#5-логи-mlflow-скрипты-и-тесты)

---

## 1. Окружение и патчи DSPy

### Переменные окружения (`.env`)

Скопируйте [`.env.example`](../.env.example) в `.env` и заполните ключи используемых провайдеров:

```dotenv
# Ключ Google Gemini API (для ae-ingest, parse_visuals и этапов с provider: "gemini")
GEMINI_API_KEY="your-gemini-api-key"

# Ключи OpenAI-совместимых API (для provider: "api")
OPENROUTER_API_KEY="sk-or-v1-your-key"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""

# URL серверов Ollama (ТОЛЬКО при provider: "ollama" в YAML)
OLLAMA_STUDENT_BASE_URL="http://localhost:11434"
OLLAMA_TEACHER_BASE_URL="http://localhost:11434"
OLLAMA_EXTRACTION_BASE_URL="http://localhost:11434"
OLLAMA_SEMANTIC_OPTIMIZATION_BASE_URL="http://localhost:11434"
OLLAMA_VISUAL_EXTRACTOR_BASE_URL="http://localhost:11434"

# Инфраструктура трекинга и кэширования
MLFLOW_TRACKING_URI="sqlite:///research/experiments/_runtime/mlflow.db"
DSPY_CACHE_DIR=".cache/dspy"
LITELLM_LOCAL_MODEL_COST_MAP=True
```

### Обязательные патчи DSPy MIPROv2

Перед запуском опциональной синтаксической оптимизации (`ae-optimize-syntactic`) примените оба патча совместимости к установленному пакету `dspy`:

```bash
python scripts/patch_dspy_mipro_zero_bootstrap.py
python scripts/patch_dspy_none_type.py
```

1. [`scripts/patch_dspy_mipro_zero_bootstrap.py`](../scripts/patch_dspy_mipro_zero_bootstrap.py) — исправляет `ValueError` при `max_bootstrapped_demos=0`, сохраняет заданные константы MIPROv2 и разрешает `GroundedProposer` использовать вручную размеченные примеры.
2. [`scripts/patch_dspy_none_type.py`](../scripts/patch_dspy_none_type.py) — устраняет `TypeError` на динамически собираемых Pydantic-сигнатурах в `dspy/propose/utils.py`.

Повторный запуск безопасен и проверяет статус патчей (`✓ ... already patched`).

---

## 2. Подготовка данных и справочник CLI

Перед запуском убедитесь, что пути в [`config/core.yaml`](../config/core.yaml) согласованы с файлами в `data/raw/`:
1. **PDF-статьи:** каталог `paths.pdf_dir` (по умолчанию `data/raw/pdf/`).
2. **Ground Truth CSV** *(для оптимизации)*: файл `paths.ground_truth_file`; колонка идентификатора (`pdf`, `filename`, `source`, `doi` или `document`) должна совпадать со stem-именем PDF.
3. **Разбиение на выборки (`splits.json`):** ключи `train`, `val`, `test`. Порядок ID в `train` сохраняется — первые `initial_docs` статей попадают в Anchor-итерацию.
4. **Схема (`config/schema.yaml`)** и **стартовый промпт (`config/baseline_instruction.txt`)**.

### Сводная таблица CLI-команд

Все 4 точки входа объявлены в [`pyproject.toml`](../pyproject.toml) и поддерживают `--config <DIR>` (по умолчанию `config/`):

| Команда | Скрипт-обёртка | Основные флаги | Назначение |
| :--- | :--- | :--- | :--- |
| `ae-ingest` | `scripts/ingest.py` | `--config`, `--overwrite`, `--from-stage`, `--split`, `--document-ids`, `--dry-run`, `[paths ...]` | Оцифровка PDF в Markdown с манифестами иллюстраций. |
| `ae-optimize-semantic run` | — | `--config`, `--steps`, `--resume`, `--non-interactive`, `--dry-run` | Выведение правил домена и ограничений схемы с опциональным HITL. |
| `ae-optimize-semantic status` | — | `--config` | Статус артефактов всех фаз семантической оптимизации. |
| `ae-optimize-syntactic` *(опц.)* | `scripts/optimize.py` | `--config`, `--run-name`, `--no-mlflow`, `--dry-run` | Тюнинг инструкций и `few-shot` (DSPy MIPROv2), сохранение агента. |
| `ae-extract` | `scripts/extract.py` | `--agent` *(обяз.)*, `--config`, `--split`, `--document-ids`, `--dry-run` | Пакетное извлечение сохранённым агентом в `paths.extracted_dir`. |

### Примеры запуска по этапам

#### 1. Ингестия (`ae-ingest`)
```bash
ae-ingest --dry-run
ae-ingest --split train
ae-ingest --document-ids acsomega.2c06142,j.bmcl.2018.03.046

# Допустимые значения --from-stage:
# parse_text, manifest_visuals, segment_and_crop_visuals_elements, segment_and_crop_visuals_atoms, parse_visuals
ae-ingest --from-stage manifest_visuals
```

> [!NOTE]
> Команда `ae-ingest` выполняет только первичный проход (текст, превью страниц и заглушки). Передача `--from-stage segment_and_crop_visuals_elements` или `parse_visuals` сбрасывает кэш отложенного разбора, чтобы при следующем вызове `parse_visuals` иллюстрации пересчитались заново.

#### 2. Семантическая оптимизация (`ae-optimize-semantic`)
```bash
ae-optimize-semantic run --dry-run
ae-optimize-semantic run
ae-optimize-semantic run --resume --non-interactive

# Запуск отдельных шагов (positive, consolidation, negative, generalization):
ae-optimize-semantic run --steps positive,consolidation
ae-optimize-semantic status
```
Результаты сохраняются в `data/interim/semantic_optimization/` (`generated_instruction.txt`, `generated_schema.yaml`, `expert_rulings.json`, `ground_truth_corrected.csv`, `iterations/`).

#### 3. Синтаксическая оптимизация *(опционально)* (`ae-optimize-syntactic`)
```bash
ae-optimize-syntactic --dry-run
ae-optimize-syntactic --run-name "benzimidazole_v1"
ae-optimize-syntactic --no-mlflow
```

> [!TIP]
> Шаг нужен только при тонкой подстройке под конкретную модель или подборе `few-shot`. Чтобы стартовать с инструкции этапа 2, укажите `paths.baseline_prompt_file: "data/interim/semantic_optimization/generated_instruction.txt"` в [`config/core.yaml`](../config/core.yaml). При `ae-extract` система автоматически подхватывает `generated_prompt_file` и совместимый `generated_schema_file` (если совпадает хэш базовой схемы у загружаемого агента).

#### 4. Пакетная экстракция (`ae-extract`)
```bash
ae-extract --agent data/processed/agents/agent_20261103_201253.json --split test --dry-run
ae-extract --agent data/processed/agents/agent_20261103_201253.json --split test
```
Итоговые JSON по каждой статье записываются в `data/processed/extracted/{doc_id}.json`.

---

## 3. Справочник конфигураций (`config/*.yaml`)

На старте `Settings.load()` автоматически объединяет все YAML из `config/` (кроме `schema.yaml`, который читается отдельно загрузчиком схем). Детальные комментарии по каждому параметру находятся внутри самих файлов:

| Файл | Назначение и ключевые секции |
| :--- | :--- |
| [`config/core.yaml`](../config/core.yaml) | Основные пути (`paths.*` — PDF, GT, `splits.json`, папка ингеста, промпты, агенты, выходы), лимиты дискового и RAM-кэша (`cache`), настройки `circuit_breaker`. |
| [`config/schema.yaml`](../config/schema.yaml) | Описание целевой таблицы: `name`, `fields` (тип, описание, `required`, `choices`, диапазоны, `validators`), `compare_fields`, `float_tolerance`, доменные `tools`. |
| [`config/baseline_instruction.txt`](../config/baseline_instruction.txt) | Базовая системная инструкция задачи (до оптимизации). |
| [`config/ingestion.yaml`](../config/ingestion.yaml) | Настройки `GeminiParser`: параллелизм (`max_documents`, `max_requests_per_document`), `thinking_level` по стадиям, векторный кроп (`crop`), RDKit-валидация SMILES (`molecule.validation`). |
| [`config/semantic_optimization.yaml`](../config/semantic_optimization.yaml) | Параметры `Anchor + Residual` (`initial_docs`, `step_size`, `f1_threshold`, `residual_enabled`, пороги `residual_skip_f1` и `residual_fallback_threshold`), `interactive` (HITL), `infer_schema_constraints`, `enable_semantic_judge` и модель Teacher. |
| [`config/syntactic_optimization.yaml`](../config/syntactic_optimization.yaml) | Параметры DSPy MIPROv2 (`num_candidates`, `num_trials`, `max_bootstrapped_demos`, `max_labeled_demos`, `minibatch`, `random_seed`) и модели `llm.student` / `llm.teacher`. |
| [`config/extraction.yaml`](../config/extraction.yaml) | Лимит шагов ReAct (`max_tool_iterations`), `agent_thinking_level` (`low`), пакетный вывод (`structured_output_batch_size`), `enable_sanity_checks` и рабочая модель. |

---

## 4. Адаптация под новый домен и расширение

Ядро (`ae.core`, `ae.extraction`, `ae.semantic_optimization`) доменно-независимо: адаптация под новую область науки сводится к конфигурации схемы и подключению плагинов без правки графов.

### Новая схема задачи (`config/schema.yaml`)

Задайте поля (`str`, `float`, `int`, `bool`), ограничения значений, `compare_fields` и подключаемые `validators`/`tools` (контракт описан в [`src/ae/core/schema/config.py`](../src/ae/core/schema/config.py)):

```yaml
name: catalytic_reactions
fields:
  catalyst:
    type: str
    description: "Формула или название катализатора"
    required: true
  yield_percent:
    type: float
    min_value: 0.0
    max_value: 100.0
  smiles:
    type: str
    required: false
    validators:
      - name: smiles
        on_error: retry
compare_fields: ["catalyst", "yield_percent", "smiles"]
float_tolerance: 0.05
tools:
  - name: assemble_markush_smiles
```

### Декларативные валидаторы полей (`FieldValidatorRegistry`)

Подключаются через `format: "<name>"` или список `validators:` с политиками `on_error: retry | nullify | warn` ([`validators.py`](../src/ae/core/schema/validators.py)):
- **Встроенные правила:** `regex`, `not_regex`, `forbidden_substrings`, `non_empty`, `length`, а также лениво загружаемый химический валидатор `smiles` ([`chemistry.py`](../src/ae/core/schema/domain_validators/chemistry.py)).
- **Свой валидатор и логика сравнения:** реализуйте функцию с возвратом `FieldValidationResult(valid, value, error)`, при желании добавьте `compare_values(val_a, val_b, **params)` для оценки в `RecordMatcher` и зарегистрируйте в `DEFAULT_VALIDATOR_REGISTRY` (или укажите `callable: "pkg.mod:fn"` в YAML).

### Доменные инструменты (`ExtractionToolRegistry`)

Для добавления инструмента агенту (по аналогии с `assemble_markush_smiles` в [`domain_tools/chemistry.py`](../src/ae/core/schema/domain_tools/chemistry.py)) опишите `RegisteredToolProvider` (`name`, `build_tool`, `is_applicable(doc_text)`, `policy_hint`) и зарегистрируйте его в `DEFAULT_TOOL_REGISTRY` ([`tools.py`](../src/ae/core/schema/tools.py)) или через `callable:` в `schema.yaml`.

### Подключение LLM-провайдера и парсера PDF

- **LLM-провайдер:** для DSPy-этапов унаследуйте класс от `BaseHTTPProvider` ([`provider.py`](../src/ae/core/llm/provider.py) — уже включает ретраи, rate limiting и `CircuitBreaker`), переопределите `_prepare_payload` и `_make_request` с записью `self._thread_local.last_usage` и добавьте в `LLM_PROVIDER_REGISTRY`; для LangGraph добавьте ветку в [`create_chat_model`](../src/ae/core/llm/factory.py).
- **Парсер PDF:** унаследуйтесь от `BaseParser` ([`base.py`](../src/ae/ingestion/parsers/base.py)) и зарегистрируйте в `PARSER_REGISTRY` ([`registry.py`](../src/ae/ingestion/parsers/registry.py)).

---

## 5. Логи, MLflow, скрипты и тесты

- **Логи LLM-вызовов:** пишутся в `research/experiments/_runtime/logs/llm/{semantic_optimization,syntactic_optimization,extraction}/` с расходом токенов и стоимостью.
- **Отладочные оверлеи ингестии:** bounding box'ы и сводки сохраняются в `data/interim/ingestion/{doc_id}/debug/`.
- **Дашборд MLflow** (логи `ae-optimize-syntactic` пишутся в `research/experiments/_runtime/mlflow.db`):
  ```bash
  mlflow ui --backend-store-uri sqlite:///research/experiments/_runtime/mlflow.db --port 5000
  ```
- **Полезные скрипты (`scripts/`):**
  - [`ingest.py`](../scripts/ingest.py), [`optimize.py`](../scripts/optimize.py), [`extract.py`](../scripts/extract.py) — тонкие CLI-обёртки;
  - [`fill_predictions.py`](../scripts/fill_predictions.py) — выгрузка сопоставленных предсказаний в CSV через `RecordMatcher`;
  - [`apply_gt_corrections.py`](../scripts/apply_gt_corrections.py) — применение принятых правок к Ground Truth CSV с проверкой RDKit;
  - [`patch_dspy_mipro_zero_bootstrap.py`](../scripts/patch_dspy_mipro_zero_bootstrap.py), [`patch_dspy_none_type.py`](../scripts/patch_dspy_none_type.py) — патчи DSPy;
  - [`slaunch_optimize_pilot.sh`](../scripts/slaunch_optimize_pilot.sh), [`slaunch_optimize_re.sh`](../scripts/slaunch_optimize_re.sh) — шаблоны SLURM-задач (перед использованием адаптируйте кластерные пути).
- **Тесты и статический анализ** (настройки в [`pyproject.toml`](../pyproject.toml)):
  ```bash
  pytest tests/unit -q
  pytest tests/integration -q
  pytest -m "not slow"
  ruff check src tests
  mypy src
  ```
