# Реестр экспериментов Adaptive Extractor

Хронологический индекс завершённых экспериментов проекта (январь – ноябрь 2026). Регламент структуры `research/` и методология описаны в [`docs/methodology.md`](../../docs/methodology.md), хронология решений — в [`docs/journal.md`](../../docs/journal.md).

## Сводная таблица (12 экспериментов)

| Дата | Эксперимент | Семестр / Домен | Суть | Ключевые метрики | Отчёт и данные |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-01-30 | `2026-01-30_nanozymes_text_baseline_m1` | 1 сем. / `nanozymes` *(текст)* | Ручной M1 (0/1/2-shot) и первый прогон MIPROv2 | M1 1-shot F1 = 0.5811; MIPROv2: Val 0.69, Test 0.57 | [report](2026-01-30_nanozymes_text_baseline_m1/report.md) · [metrics](2026-01-30_nanozymes_text_baseline_m1/metrics.json) |
| 2026-03-15 | `2026-03-15_pdf_parsers_comparison` | 2 сем. / `nanozymes` *(40 статей)* | Сравнение 6 PDF-парсеров (`Docling`, `MinerU`, `Marker`, Gemini) | Выбран Gemini 3 Flash: 0 потерь, 95% без замечаний | [report](2026-03-15_pdf_parsers_comparison/report.md) · [metrics](2026-03-15_pdf_parsers_comparison/metrics.json) |
| 2026-04-15 | `2026-04-15_nanozymes_mipro_pilot` | 2 сем. / `nanozymes` *(текст)* | Пилот MIPROv2 (Qwen3.5-27B / 397B + Gemini 3 Flash) | eval_full: 0.7810 → **0.8492** (+6.82 п.п.) | [report](2026-04-15_nanozymes_mipro_pilot/report.md) · [metrics](2026-04-15_nanozymes_mipro_pilot/metrics.json) |
| 2026-07-01 | `2026-07-01_nanozymes_re_vs_mipro_benchmark` | 2 сем. / `nanozymes` *(PoC, N=15)* | Сравнение `nanoMINER`, `Manual`, `RE` и `RE + MIPROv2` | **RE Mean F1 = 0.8804** (`p_Holm = 0.00037` vs 0.6592) | [report](2026-07-01_nanozymes_re_vs_mipro_benchmark/report.md) · [metrics](2026-07-01_nanozymes_re_vs_mipro_benchmark/metrics.json) |
| 2026-10-29 | `2026-10-29_ocsr_gemini_benchmarks` | 3 сем. / `benzimidazole` *(OCSR)* | Бенчмарк 16 конфигураций Gemini (30 структур) и кропинг | Исключены Lite и high; кропинг Fig. 2 дал 4/4 и 6/6 | [report](2026-10-29_ocsr_gemini_benchmarks/report.md) · [metrics](2026-10-29_ocsr_gemini_benchmarks/metrics.json) |
| 2026-11-20 | `2026-11-20_benzimidazole_hybrid_run` | 3 сем. / `benzimidazole` *(MVP, val)* | Гибрид Anchor + Residual с HITL и технический аудит | Baseline 0.9576 → iter_2 **Micro-F1 = 0.9826** | [report](2026-11-20_benzimidazole_hybrid_run/report.md) · [metrics](2026-11-20_benzimidazole_hybrid_run/metrics.json) |
| 2026-11-21 | `2026-11-21_benzimidazole_post_audit_eval` | 3 сем. / `benzimidazole` *(MVP, val)* | Точечная перегенерация и контрольный прогон после правок | **Micro-F1 = 0.9949**, Macro = 0.9952, ускорение 1.78× | [report](2026-11-21_benzimidazole_post_audit_eval/report.md) · [metrics](2026-11-21_benzimidazole_post_audit_eval/metrics.json) |
| 2026-11-22 | `2026-11-22_exp01_ingestion_and_ocsr` | 3 сем. / `benzimidazole` *(EXP-01)* | Оптимизация ингестии, OCSR-оракул и фикс ADR-010 (`RJC #15`) | Fast-Sync: HTTP −62%, 0 сирот; 01A частично заблокирован `402` | [report](2026-11-22_exp01_ingestion_and_ocsr/report.md) · [metrics](2026-11-22_exp01_ingestion_and_ocsr/metrics.json) |
| 2026-11-22 | `2026-11-22_exp02_extraction_graph_efficiency` | 3 сем. / `benzimidazole` *(EXP-02, val)* | Tool-routing (`finalize_evidence`) и каноническая дедупликация | **Micro-F1 = 0.9964 ± 0.0008** (`n=3`), `draft_tokens = 0` | [report](2026-11-22_exp02_extraction_graph_efficiency/report.md) · [metrics](2026-11-22_exp02_extraction_graph_efficiency/metrics.json) |
| 2026-11-22 | `2026-11-22_exp03_evaluation_and_validators` | 3 сем. / `benzimidazole` *(EXP-03)* | Двухметрический `RecordMatcher`, `pint`/`simpleeval`/`POT`, судья | Replay 0.9949/0.9926; `gemma-4-31b-it` **κ = 0.9748** (21.2 с) | [report](2026-11-22_exp03_evaluation_and_validators/report.md) · [metrics](2026-11-22_exp03_evaluation_and_validators/metrics.json) |
| 2026-11-22 | `2026-11-22_exp04_semantic_opt_hardening` | 3 сем. / `benzimidazole` *(EXP-04)* | Pareto Auto-Rollback, `min_enum_doc_support`, `CASE_PATCH` | Semantic Micro-F1 = **0.9968**, но `strict bacteria = 0.5159` | [report](2026-11-22_exp04_semantic_opt_hardening/report.md) · [metrics](2026-11-22_exp04_semantic_opt_hardening/metrics.json) |
| 2026-11-22 | `2026-11-22_exp05_model_routing` | 3 сем. / `benzimidazole` *(EXP-05)* | Поузловой подбор моделей (треки 5.1–5.6) и маршрутная таблица | Вспомогательные узлы → `glm-5.3`; основной стек сохранён | [report](2026-11-22_exp05_model_routing/report.md) · [metrics](2026-11-22_exp05_model_routing/metrics.json) |

Решения `EXP-01..04` внедрены в `src/` и `config/` (ADR-006..010 в [`docs/journal.md`](../../docs/journal.md)); рекомендации `EXP-05` оставлены как резервная маршрутная карта без изменения `config/*.yaml`.

## Аудиты GT и граница хранения

- **Аудиты GT:** [`gt_audits/README.md`](gt_audits/README.md) — [`nanozymes.csv`](gt_audits/nanozymes.csv) (239 кейсов) и [`benzimidazole_antibiotics.csv`](gt_audits/benzimidazole_antibiotics.csv) (70 кейсов).
- **Хранение ([`.gitignore`](../../.gitignore)):** в Git входят `README.md`, `gt_audits/`, `report.md`, `metrics.json`, `tracks/*.json`, `config_snapshot/` и `result/`; буфер `_runtime/`, `logs/` и `raw/` остаются локально.
