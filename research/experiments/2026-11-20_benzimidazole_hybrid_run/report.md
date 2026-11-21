# Аудит `ae.semantic_optimization` и прогона `benzimidazole_antibiotics` (`iter_1` vs `iter_2`)

- **Статус:** Завершён *(исторический технический аудит прогона `20261120_hybrid_run` до внесения исправлений)*
- **Реализация правок:** все доработки разделов 4–5 (Блоки 1–6: `A1–A5`, `B1–B4`, `C1–C5`, `Г1–Г2`) уже внедрены в код и проверены в контрольном прогоне [`2026-11-21_benzimidazole_post_audit_eval`](../2026-11-21_benzimidazole_post_audit_eval/report.md); 15 решений по пересмотру GT собраны в [`gt_audits/benzimidazole_antibiotics.csv`](../gt_audits/benzimidazole_antibiotics.csv).
- **Файлы:** [`report.md`](report.md), [`metrics.json`](metrics.json), [`config_snapshot/`](config_snapshot), [`result/generated_instruction.txt`](result/generated_instruction.txt), [`result/generated_schema.yaml`](result/generated_schema.yaml); локально (`.gitignore`) — [`logs/`](logs) (`re_20261120_182014.json` — 48 вызовов `qwen/qwen3.5-397b-a17b`; `extraction_20261120_182014.json` — 80 реальных вызовов `gemini-3.8-flash` + запись пост-валидации `log[57]`) и [`raw/`](raw).

> [!IMPORTANT]
> Отрывочные логи до ребалансировки сплитов (`014705`, `123943`, где на `val` оценивался `j.ejmech.2016.01.052` вместо `j.bmcl.2018.03.046`) удалены при чистке буфера `_runtime`; все метрики ниже проверены строго по логам `182014` и артефактам `raw/`.

---

## 1. Качество и динамика промпта (`iter_1` vs `iter_2`)

### 1.0. Сравнение с бейзлайном (`Baseline` vs `iter_2` на `val`)

Прогон базового промпта (`config/baseline_instruction.txt` + `config/schema.yaml`) на `val` (6 документов, 313 строк GT) с `SemanticJudge`:

| Документ (`val`) | GT строк | `Baseline` (+Judge) | `iter_2` (+Judge) | Δ (`iter_2` − Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| `acsomega.2c06142` | 16 | 1.0000 | 1.0000 | 0.0000 |
| `j.bmcl.2018.03.046` | 144 | 0.9762 | 0.9762 | 0.0000 |
| `j.ejmech.2019.03.026` | 54 | 0.9841 | 0.9841 | 0.0000 |
| `j.ejmech.2020.112996` | 78 | 0.9063 | **0.9936** | **+0.0873** |
| `rjc.2023.1638382` | 14 | 0.9524 | 0.9524 | 0.0000 |
| `s41598-022-21435-6` | 7 | 0.8571 | **1.0000** | **+0.1429** |
| **Micro-F1 (по форме, +Judge)** | **313** | 0.9576 | **0.9826** | **+0.0250** |
| **Macro-F1 (по форме, +Judge)** | **6** | 0.9460 | **0.9844** | **+0.0384** |
| Micro-F1 (по смыслу/фактам*) | 313 | 0.9951 | 0.9951 | 0.0000 |

*\*По смыслу (без штрафа за сокращения единственных штаммов в статье и дубль 8h из Table_4): у обоих вариантов ровно 8 фактических ошибок (7 ошибок VLM-ингеста в `smiles` + 1 FN `cid=28` в `rjc.2023.1638382`).*

Весь прирост от бейзлайна обеспечен полем `bacteria` (~0.7420 → **0.9172**, +55 строк); остальные 6 колонок совпадают (`compound_id`, `target_*`: 0.9936; `smiles`: 0.9713). Правило сохранения `± std` в `target_value` не отразилось в `val`-метриках из-за отсутствия строк с `±` на `val`.

### 1.1. Сводные метрики на `val` (`iter_1` vs `iter_2`)

Оптимизация завершилась досрочно на 2-й итерации (`f1_threshold: 0.98`):

| Документ (`val`) | GT строк | `iter_1` (+Judge) | `iter_2` (+Judge) | Δ F1 | `iter_1` Strict | `iter_2` Strict | Суть изменений |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `acsomega.2c06142` | 16 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | Стабильно 16/16 |
| `s41598-022-21435-6` | 7 | 0.8571 | **1.0000** | **+0.1429** | 0.8571 | **1.0000** | `S. aureus` → `Staphylococcus aureus ATCC 25923` |
| `j.ejmech.2020.112996` | 78 | 0.9121 | **0.9936** | **+0.0815** | 0.9121 | 0.9063* | `NR 100` → `MRSA NR 100` (48/48 Judge=YES) + FP `8h` |
| `j.bmcl.2018.03.046` | 144 | 0.9762 | 0.9762 | 0.0000 | 0.9762 | 0.9762 | Застрял на 24 `MRSA` (на попытке 1 раскрыл сноску, на ретрае откатился) |
| `j.ejmech.2019.03.026` | 54 | 1.0000 | 0.9841 | −0.0159 | 1.0000 | 0.9841 | Регрессия (6 `smiles`): 37–39 стали 3-пиридилом вместо 4-пиридила |
| `rjc.2023.1638382` | 14 | 0.9898 | 0.9524 | −0.0374 | 0.9898 | 0.9524 | Регрессия: потерян `cid=28` vs E. coli + баг ингеста `cid=15` |
| **Micro-F1** | **313** | 0.9635 | **0.9826** | **+0.0191** | — | — | Порог 0.98 преодолён |
| **Macro-F1** | **6** | 0.9559 | **0.9844** | **+0.0285** | — | — | Рост за счёт `s41598` и `112996` |

*\*В `112996` Strict показывает 0.9063 из-за расхождения `MRSA NR ...` vs `Staphylococcus aureus MRSA NR ...`, принятого `SemanticJudge` (48/48).*

### 1.2. Поколоночная динамика на `val`

| Колонка | `iter_1` (Strict) | `iter_1` (+Judge) | `iter_2` (Strict) | `iter_2` (+Judge) | Δ (+Judge) | Оставшиеся ошибки `iter_2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `compound_id` | 1.0000 | 1.0000 | 0.9936 | 0.9936 | −0.0064 | 1 FP (`112996` 8h), 1 FN (`rjc` 28) |
| `smiles` | **0.9968** | **0.9968** | 0.9713 | 0.9713 | −0.0255 | 6× `2019.03.026` (37–39), 1× `rjc` (15), 1 FP, 1 FN |
| `target_type` / `relation` / `value` / `units` | 1.0000 | 1.0000 | 0.9936 | 0.9936 | −0.0064 | 1 FP, 1 FN |
| `bacteria` | 0.7476 | 0.7476 | 0.7643 | **0.9172** | **+0.1696** | 24× `bmcl` (`MRSA` без `N315`), 1 FP, 1 FN |

### 1.3. Изменения промпта и схемы (`iter_1` → `iter_2`)

Сравнение [`iter_1/candidate_prompt.txt`](../../../data/interim/semantic_optimization/iterations/iter_1/04_generalization/candidate_prompt.txt) и [`generated_instruction.txt`](../../../data/interim/semantic_optimization/generated_instruction.txt), а также [`iter_1/candidate_schema.yaml`](../../../data/interim/semantic_optimization/iterations/iter_1/04_generalization/candidate_schema.yaml) и [`generated_schema.yaml`](../../../data/interim/semantic_optimization/generated_schema.yaml):

- **Улучшения:**
  1. Схема `target_units` исправила переобучение Итерации 1: [`iter_1/schema_refinement.json`](../../../data/interim/semantic_optimization/iterations/iter_1/04_generalization/schema_refinement.json#L68-L80) ошибочно закрыл `choices: [mM, µg/mL, µM]` (из-за чего отпадали 19 строк с `µmol mL-1`), а [`iter_2`](../../../data/interim/semantic_optimization/iterations/iter_2/04_generalization/schema_refinement.json#L68-L77) перевёл поле в `is_closed_enum: false`.
  2. `BACTERIA`: добавлены требования извлекать `MRSA`/`VRSA`, коды штаммов (`ATCC`, `MTCC`, `RCMB`, `N315`) и раскрывать сноски (+0.1429 на `s41598`, +0.0815 на `112996`).
  3. `TARGET_VALUE`: снято требование удалять незначащие нули, добавлено сохранение `± std`.
  4. `RECORD SELECTION` и `SMILES`: добавлен сбор MIC/pMIC из прозы, убрана ручная канонизация и закреплено использование `assemble_markush_smiles`.

- **Причины регрессий:**
  1. **Противоречие в `BACTERIA` (`j.bmcl.2018.03.046`):** шапка [`generated_instruction.txt`](../../../data/interim/semantic_optimization/generated_instruction.txt#L15) запрещала нормализацию форм, а конец ([L208–220](../../../data/interim/semantic_optimization/generated_instruction.txt#L208-L220)) требовал раскрывать сноски таблиц. На попытке 1 (`log[56]`) модель раскрыла сноску `<sup>b</sup>` в [`j.bmcl.2018.03.046.md`](../../../data/interim/ingestion/j.bmcl.2018.03.046/j.bmcl.2018.03.046.md#L559) (24/24), но упала на RDKit-валидации SMILES `4d` (из [L96](../../../data/interim/ingestion/j.bmcl.2018.03.046/j.bmcl.2018.03.046.md#L96)); на ретрае после [`_format_sanity_retry_feedback`](../../../src/ae/extraction/graph.py#L1273-L1280) исправила SMILES, но откатила `bacteria` обратно к `'MRSA'`.
  2. **Регрессия `j.ejmech.2019.03.026` (1.0000 → 0.9841):** в [`j.ejmech.2019.03.026.md`](../../../data/interim/ingestion/j.ejmech.2019.03.026/j.ejmech.2019.03.026.md#L84) VLM на `Scheme_1` выдал 3-пиридил, хотя проза IUPAC ([L603–611](../../../data/interim/ingestion/j.ejmech.2019.03.026/j.ejmech.2019.03.026.md#L603-L611)) указывает 4-пиридил; жёсткое правило [L11](../../../data/interim/semantic_optimization/generated_instruction.txt#L11) заставило слепо доверять VLM-реестру (плюс 2 тестовых вызова `compound_id="test"`).
  3. **Ошибки `rjc.2023.1638382` (0.9898 → 0.9524):** баг ингеста `cid=15` (`5-Cl-2-F` в [`Fig.-7`](../../../data/interim/ingestion/RJC.2023.1638382/RJC.2023.1638382.md#L133) против `2-chloro-6-fluorophenyl` в [прозе](../../../data/interim/ingestion/RJC.2023.1638382/RJC.2023.1638382.md#L105)) и пропуск `cid=28` (E. coli) в [L154](../../../data/interim/ingestion/RJC.2023.1638382/RJC.2023.1638382.md#L154) при исчерпании `max_tool_iterations=5`.
  4. **Мнимый FP `j.ejmech.2020.112996` (`8h`):** в `Table_1` ([L180](../../../data/interim/ingestion/j.ejmech.2020.112996/j.ejmech.2020.112996.md#L180)) указано `4 µg/mL`, а в `Table_4` ([L273–279](../../../data/interim/ingestion/j.ejmech.2020.112996/j.ejmech.2020.112996.md#L273-L279)) — `8 µg/mL` (подтверждено по PDF и добавлено в GT: 1768 → 1769 строк).

---

## 2. Эффективность архитектуры `ae.semantic_optimization`

### 2.1. Гибридный режим (`Anchor` + `Residual`), сэмплинг и carry-forward

| Этап (`re_20261120_182014.json`) | Документов | Вызовов LLM | Время | Оценка токенов (in / out) |
| :--- | :---: | :---: | :---: | :---: |
| Iter 1 (Anchor: 1A + 1B + Synth + HITL + Harm + Schema) | 2 | 33 | 4021.2 с (67.0 мин) | ~455k / ~110k |
| Iter 2 (Residual: 5 Residual + 4 Carry-Fwd + HITL + Harm + Schema) | 4 | 15 | 772.5 с (**12.9 мин**) | ~295k / ~29k |

На 4 документах `iter_2` обошлась 5 вызовами residual-анализа, а 4 из 8 областей правил перенесены через carry-forward за 0 вызовов (ускорение **~5.2×** при удвоении числа статей).

**Выявленные дефекты сплитов и стереохимии:**
1. [`load_all_splits` / `load_split`](../../../src/ae/core/storage/splits.py#L47-L100) возвращали неупорядоченный `set`, а [`cli.py:174`](../../../src/ae/semantic_optimization/cli.py#L174) сортировал ID по алфавиту — порядок массива `splits.json` терялся.
2. В [`chemistry.py:114`](../../../src/ae/core/schema/domain_validators/chemistry.py#L114) жёсткое `isomericSmiles=True` вызывало 33 ложные ошибки на плоских SMILES VLM-ингеста в [`ja.2017.70.md`](../../../data/interim/ingestion/ja.2017.70/ja.2017.70.md) (через [`matcher.py:682-688`](../../../src/ae/core/evaluation/matcher.py#L682-L688) и [`positive_analysis.py:228-259`](../../../src/ae/semantic_optimization/steps/positive_analysis.py#L228-L259)), засоряя HITL.
3. Принятие `SemanticJudge` форматов вроде `µg/mL == µg mL-1` скрывало ошибки `target_units` из `affected_scopes`.

### 2.2. Баги кэширования и `--resume`

| Узел / Механизм | Локация в коде | Выявленная проблема |
| :--- | :--- | :--- |
| `ResidualDocReport.input_hash` и `val_metrics.json` | [`residual_analysis.py:549-693`](../../../src/ae/semantic_optimization/steps/residual_analysis.py#L549-L693), [`graph.py:429-435`](../../../src/ae/semantic_optimization/graph.py#L429-L435) | Хэш не писался в skip/fallback отчёты, не включал `schema`/`ext_model` и считался по мутабельному `{doc}.md` (изменяемому [`service.py:70`](../../../src/ae/ingestion/service.py#L70)), срывая `--resume`. |
| `schema_refinement`, `generalization` и carry-forward | [`schema_refinement.py:226-247`](../../../src/ae/semantic_optimization/steps/schema_refinement.py#L226-L247), [`generalization.py:286-399`](../../../src/ae/semantic_optimization/steps/generalization.py#L286-L399) | Не передавался `cache_key` (`.sha256`); carry-forward не проверял новые `expert_rulings`. |
| `candidate_prompt.txt` и `02_consolidation` | [`generalization.py:517-541,647-659`](../../../src/ae/semantic_optimization/steps/generalization.py#L647-L659) | Отсутствовала проверка хэша входов и перезапись консолидации при рекомпуте. |
| Prompt Caching для Residual-сигнатур | [`provider.py:581-585`](../../../src/ae/core/llm/provider.py#L581-L585) | Отсутствовали разделители `mismatched_rows_context`, `missed_gt_rows`, `false_positive_rows`. |
| Нулевой cache hit в `structured_output` | [`extraction/graph.py:1183-1210`](../../../src/ae/extraction/graph.py#L1183-L1210) | 17/17 вызовов получали `cache_read = 0`; разделение префикса нужно для кэша между батчами и ретраями `structured_output_node`. |

### 2.3. Детекция конфликтов, HITL и `expert_rulings.json`

1. **Утечка мусорных литералов:** при `KEEP_BASELINE` в [`hitl.py:924-927`](../../../src/ae/semantic_optimization/hitl.py#L924-L927) голое значение `"0.202"` попало в [`expert_rulings.json:38`](../../../data/interim/semantic_optimization/expert_rulings.json#L38), породив ложный конфликт в `iter_2`.
2. **Пропуск конфликтов и неполная дедупликация:** [`hitl.py:411-456`](../../../src/ae/semantic_optimization/hitl.py#L411-L456) фильтровал только `KEEP_BASELINE`/`OVERRIDE_BASELINE`, игнорируя `GENERALIZE`/`ADD_CONDITION`, а [`generalization.py:433-494`](../../../src/ae/semantic_optimization/steps/generalization.py#L433-L494) не запускал 2-й раунд HITL для новых конфликтов после рекомпута (4 висящих аномалии в `iter_2`, включая `bacteria_footnote_expansion_002`).

### 2.4. Расход вызовов LLM, токенов и времени

Суммарное чистое время LLM-вызовов в `182014` составило 9,421.4 с (~2 ч 37 мин) без учёта ожидания HITL и `visual_prep`:

| Компонент | Модель | Вызовов | Время | Токены (in / cached / out / reasoning) |
| :--- | :--- | :---: | :---: | :--- |
| RE Iter 1 (Anchor, 2 docs) | `qwen3.5-397b` | 33 | 67.0 мин | `usage` не логировался (~455k in / ~110k out) |
| RE Iter 2 (Residual, 4 docs) | `qwen3.5-397b` | 15 | 12.9 мин | `usage` не логировался (~295k in / ~29k out) |
| Ext Iter 1 Val (6 docs) | `gemini-3.8-flash` | 28 (3 Judge) | 27.3 мин | 520.7k in (32.6% cached) / 221.7k out (85.4% reasoning) |
| Ext Iter 2 Residual (4 train) | `gemini-3.8-flash` | 21 (6 Judge) | 20.1 мин | 503.0k in (35.8% cached) / 150.8k out (83.6% reasoning) |
| Ext Iter 2 Val (6 docs) | `gemini-3.8-flash` | 31 (2 Judge)* | 29.7 мин* | 675.5k in (41.6% cached) / 246.0k out (81.9% reasoning) |
| **Итого Gemini Extraction** | `gemini-3.8-flash` | **80*** | **77.1 мин*** | **1.70M in (37.1% cached) / 618.5k out (83.6% reasoning)** |

*\*Без дубля `log[57]` от повторной записи ошибки валидации поверх `log[56]` в [`_invoke_with_retry`](../../../src/ae/extraction/graph.py#L705-L711).*

> [!CAUTION]
> **Главное узкое место — расход thinking-токенов на финальном `READY`-ходе `agent_node`:** на завершающем шаге маршрутизации модель сжигала 8–47k thinking-токенов за вызов (всего **194,682 reasoning-токенов и 23.0 мин = 29.8% времени фазы**) из-за подачи полного промпта и уровня `medium` в [`extraction/graph.py:618-642`](../../../src/ae/extraction/graph.py#L618-L642) и принудительного возврата в [`_route_after_tool`](../../../src/ae/extraction/graph.py#L1179-L1180).

Дополнительные потери: полный ретрай `j.bmcl.2018.03.046` из-за 1 опечатки SMILES (`4d`) обошёлся в 520 с и 74.8k токенов; при этом [`provider.py:217-262`](../../../src/ae/core/llm/provider.py#L217-L262) и [`visual_prep.py:82-135`](../../../src/ae/semantic_optimization/steps/visual_prep.py#L82-L135) не писали `usage` в логи истории.

### 2.5. Доменная независимость ядра

Точечные утечки химических и материаловедческих примеров обнаружены в [`visual_prep.py:28-33,174`](../../../src/ae/semantic_optimization/steps/visual_prep.py#L28-L33), [`signatures/columns.py:26`](../../../src/ae/semantic_optimization/signatures/columns.py#L26), [`signatures/rows.py:246`](../../../src/ae/semantic_optimization/signatures/rows.py#L246), [`signatures/residual.py:74`](../../../src/ae/semantic_optimization/signatures/residual.py#L74) и описании настройки в [`optimization_settings.py:219`](../../../src/ae/core/config/optimization_settings.py#L219).

---

## 3. Качество данных, ингеста и сплитов

- **`ground_truth_corrected.csv` vs исходный GT:** эквивалентны (1768 строк, дублей нет); в процессе аудита строка `8h` Table 4 для `j.ejmech.2020.112996` подтверждена по PDF и добавлена в [`Benzimidazole antibiotics_corrected.csv`](../../../data/raw/ground_truth/Benzimidazole%20antibiotics_corrected.csv) (1769 строк).
- **Ошибки ингеста на `val`:** 3-пиридил вместо 4-пиридила (`37–39` в `j.ejmech.2019.03.026`), `5-Cl-2-F` (`cid=15` в `RJC.2023.1638382`) и отсутствие `C=N` E/Z-стереохимии в `ja.2017.70`.
- **Риски отложенного `test` (6 документов, 287 строк):** 15 опечаток регистра `'µg/Ml'` в `0929867329666220407094430`, невиданные единицы `µM` и штамм `'MTCC 3160'` (`j.bmc.2014.09.008`), сокращения `'Ec'`/`'Sa'` (`s13065-018-0479-1`), сноска N315 в `cjoc.202200326` и 43 визуала в обзорной `antibiotics12071220` на `train`.

---

## 4. Приоритизированный план улучшений (реализовано в Блоках 1–6)

| Код | Проблема и решение | Точка входа |
| :---: | :--- | :--- |
| **A1** | Записывать `input_hash` во все ветки `ResidualDocReport`, считать хэш по каноническому манифесту стадии 02 | [`residual_analysis.py:549-693`](../../../src/ae/semantic_optimization/steps/residual_analysis.py#L549-L693), [`graph.py:429-435`](../../../src/ae/semantic_optimization/graph.py#L429-L435) |
| **A2** | Добавить `.sha256`-ключи кэша в `schema_refinement`, `generalization` и проверку новых `expert_rulings` при carry-forward | [`schema_refinement.py:226-247`](../../../src/ae/semantic_optimization/steps/schema_refinement.py#L226-L247), [`generalization.py:286-399`](../../../src/ae/semantic_optimization/steps/generalization.py#L286-L399) |
| **A3** | Расширить дедупликацию конфликтов на `GENERALIZE`/`ADD_CONDITION` и `IGNORE_CASE`; добавить 2-й раунд HITL после рекомпута | [`generalization.py:433-494`](../../../src/ae/semantic_optimization/steps/generalization.py#L433-L494), [`hitl.py:411-456`](../../../src/ae/semantic_optimization/hitl.py#L411-L456) |
| **A4** | Защитить `decision_text` проверкой осмысленности `_is_descriptive_rule_text` | [`hitl.py:920-927`](../../../src/ae/semantic_optimization/hitl.py#L920-L927) |
| **A5** | Ввести `load_split_ordered` в `splits.py` и сохранить порядок массива из `splits.json` | [`splits.py:47-100`](../../../src/ae/core/storage/splits.py#L47-L100), [`cli.py:174`](../../../src/ae/semantic_optimization/cli.py#L174) |
| **B1** | Задать в `PromptHarmonization` приоритет локальных сносок/легенд таблиц над общим сохранением сырой формы | [`signatures/columns.py:413`](../../../src/ae/semantic_optimization/signatures/columns.py#L413) |
| **B2** | Разрешить кросс-проверку VLM-структур против IUPAC и прозы статьи в `ColumnGeneralization` и Маркуш-туле | [`signatures/columns.py:317`](../../../src/ae/semantic_optimization/signatures/columns.py#L317), [`domain_tools/chemistry.py:18-23`](../../../src/ae/core/schema/domain_tools/chemistry.py#L18-L23) |
| **B3** | Сделать `isomeric` + `ignore_unassigned_stereo` параметрами плагина через `compare_field_values_via_validator` | [`validators.py`](../../../src/ae/core/schema/validators.py), [`domain_validators/chemistry.py:18-21,114`](../../../src/ae/core/schema/domain_validators/chemistry.py#L18-L21) |
| **B4** | Очистить доменные примеры из ядра RE (`visual_prep.py`, `signatures/`) | [`visual_prep.py:28-33`](../../../src/ae/semantic_optimization/steps/visual_prep.py#L28-L33), [`signatures/rows.py:246`](../../../src/ae/semantic_optimization/signatures/rows.py#L246) |
| **C1** | Перевести `agent_node` на `agent_thinking_level: "low"` и отсечь холостой `READY`-ход в `_route_after_tool` | [`extraction/graph.py:618-642,1179-1180`](../../../src/ae/extraction/graph.py#L618-L642), [`config/extraction.yaml:14`](../../../config/extraction.yaml#L14) |
| **C2** | Разделить статический документ и динамическую инструкцию в `_build_structured_messages` | [`extraction/graph.py:1183-1210`](../../../src/ae/extraction/graph.py#L1183-L1210) |
| **C3** | Логировать `usage` в `BaseLMProvider` (`OpenRouterLM`/`GeminiLM`), подключить логгер к `visual_prep` и убрать дубль `latency_s` | [`provider.py:217-262`](../../../src/ae/core/llm/provider.py#L217-L262), [`extraction/graph.py:705-711`](../../../src/ae/extraction/graph.py#L705-L711) |
| **C4** | Добавить Residual-разделители в `apply_prompt_caching` | [`provider.py:581-585`](../../../src/ae/core/llm/provider.py#L581-L585) |
| **C5** | Точечный JSON-ремонт только `failed_record_indices` вместо перегенерации всего документа | [`extraction/graph.py:1453-1470`](../../../src/ae/extraction/graph.py#L1453-L1470), [`sanity.py`](../../../src/ae/extraction/sanity.py) |
| **Г1–Г2** | Починить 15 опечаток `'µg/Ml'` в GT `test` и добавить подтверждённую строку `8h` в GT `val` (`1769` строк) | [`Benzimidazole antibiotics_corrected.csv`](../../../data/raw/ground_truth/Benzimidazole%20antibiotics_corrected.csv) |

---

## 5. Архитектурные гарантии и верификация API

- **Отсутствие хрупкости и доменная изоляция:** `RecordMatcher` и RE-узлы делегируют эквивалентность универсальному контракту `compare_field_values_via_validator` без хардкода SMILES/RDKit; правила B1/B2 и HITL-фильтры A3/A4 формулируются на структурных признаках (сноски, пересечение токенов, минимальная длина правила), а точечный ремонт C5 работает для любой Pydantic-модели из `schema.yaml`.
- **Верификация стека (`.venv`):** подтверждено, что в `langchain-google-genai == 4.4.2` параметр `thinking_level` через `bind_tools` маппится на `ThinkingConfig` по узлам графа без дублирования клиента, а `OpenRouterLM` и `GeminiLM` стабильно отдают метаданные `usage` и `cached_tokens` для записи в историю.
