# EXP-04: Защитные механизмы `SemanticOptimizationGraph` (2026-11-22)

- **Сплит:** `benzimidazole_antibiotics` (`train` 6 док. / 405 GT → `val` 6 док. / 314 GT)
- **Базовая точка:** `config/baseline_instruction.txt` + `config/schema.yaml` (и сравнение с [`2026-11-20_benzimidazole_hybrid_run`](../2026-11-20_benzimidazole_hybrid_run/report.md))
- **Статус:** Завершён
- **Артефакты:** [metrics.json](metrics.json), [generated_instruction.txt](result/generated_instruction.txt), [generated_schema.yaml](result/generated_schema.yaml)

## Что проверяли

Добавление защит в `SemanticOptimizationGraph` (`qwen/qwen3.5-397b-a17b` + `gemini-3.8-flash`):
1. `min_enum_doc_support = 4`: запрет закрывать поле в `enum`/`choices` по малому числу документов.
2. По-колоночный Pareto Auto-Rollback (`pareto_regression_threshold = 0.01`) и гармонизация правил `Contradiction-First`.
3. Фильтрация `CASE_PATCH` правил из промпта строк, `PU`-триггер и приоритизация HITL `P0–P3`.

## Результаты (`val`, 6 документов / 314 строк GT)

| Вариант | Micro-F1 (+Judge) | Macro-F1 (+Judge) | Strict Micro-F1 |
| :--- | :---: | :---: | :---: |
| `2026-11-20` (`iter_2`, hybrid run) | 0.9826 | 0.9844 | — |
| **SemOpt-Guarded (`EXP-04`)** | **0.9968** | **0.9762** | **0.9308** |

По колонкам (+Judge) в `EXP-04`: все поля кроме `bacteria` — `1.0000`, `bacteria` — `0.9777`. Между `iter_1` и `iter_2` сработал Pareto-откат для 6 из 7 колонок (оставлено обновление `target_relation`).

## Выводы и ограничения

1. Защиты встроены в `ae.semantic_optimization` (ADR-009) и удержали итоговый Semantic Micro-F1 на `0.9968` без регрессий отдельных колонок по semantic-метрике.
2. **Скрытая просадка `Strict F1`:** между `iter_1` и `iter_2` `strict_micro_f1` упал с `0.9645` до `0.9308` из-за обвала `strict_per_field_f1.bacteria` до `0.5159` — в `generated_instruction.txt` сохранились противоречивые правила для `BACTERIA`, а `SemanticJudge` сгладил все 144 расхождения в `j.bmcl.2018.03.046`.
3. **Непроверенные в живом прогоне ветки:** `CASE_PATCH` (`blocked_count = 0`), `PU`-триггер и `POT`-теги в этом прогоне ни разу не сработали (проверены только юнит-тестами); в `metrics.json` не записана телеметрия судьи (`judge_calls: 0`, `cost_usd_total: 0.0`), а решение Pareto-отката принималось непосредственно по `val`.
