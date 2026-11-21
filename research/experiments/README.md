# Реестр экспериментов Adaptive Extractor

Хронологический индекс всех завершённых экспериментов проекта (январь – ноябрь 2026). Регламент структуры `research/` и научная методология описаны в [`docs/methodology.md`](../../docs/methodology.md), а хронология решений — в [`docs/journal.md`](../../docs/journal.md).

## Сводная таблица экспериментов

| Дата | Эксперимент | Семестр / Домен | Суть | Ключевые метрики | Отчёт и метрики |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-01-30 | `2026-01-30_nanozymes_text_baseline_m1` | 1 сем. / `nanozymes` *(текст)* | Ручной M1 (0/1/2-shot) и первый прогон MIPROv2 | M1 1-shot F1 = 0.5811; MIPROv2 (70 триалов): Val 0.69, Test 0.57 | [`report.md`](2026-01-30_nanozymes_text_baseline_m1/report.md) · [`metrics.json`](2026-01-30_nanozymes_text_baseline_m1/metrics.json) |
| 2026-03-15 | `2026-03-15_pdf_parsers_comparison` | 2 сем. / `nanozymes` *(40 статей)* | Сравнение 6 PDF-парсеров (`Docling`, `MinerU`, `Marker`, Gemini) | Выбран Gemini 3 Flash: 0 структурных потерь, 95% без замечаний | [`report.md`](2026-03-15_pdf_parsers_comparison/report.md) · [`metrics.json`](2026-03-15_pdf_parsers_comparison/metrics.json) |
| 2026-04-15 | `2026-04-15_nanozymes_mipro_pilot` | 2 сем. / `nanozymes` *(текст)* | Пилот MIPROv2 на связке Qwen3.5-27B / 397B + Gemini 3 Flash | eval_full: **0.7810 → 0.8492** (+6.82 п.п.) | [`report.md`](2026-04-15_nanozymes_mipro_pilot/report.md) · [`metrics.json`](2026-04-15_nanozymes_mipro_pilot/metrics.json) |
| 2026-07-01 | `2026-07-01_nanozymes_re_vs_mipro_benchmark` | 2 сем. / `nanozymes` *(PoC, N=15)* | Статистическое сравнение `nanoMINER`, `Manual`, `RE` и `RE + MIPROv2` | **RE Mean F1 = 0.8804** (`p_Holm = 0.00037` vs 0.6592); RE+MIPROv2 0.8811 | [`report.md`](2026-07-01_nanozymes_re_vs_mipro_benchmark/report.md) · [`metrics.json`](2026-07-01_nanozymes_re_vs_mipro_benchmark/metrics.json) |
| 2026-10-29 | `2026-10-29_ocsr_gemini_benchmarks` | 3 сем. / `benzimidazole` *(OCSR)* | Бенчмарк 16 конфигураций Gemini (30 структур) и тест кропинга | Исключены Lite и high; кропинг Fig. 2 дал 0/4 → **4/4** и **6/6** | [`report.md`](2026-10-29_ocsr_gemini_benchmarks/report.md) · [`metrics.json`](2026-10-29_ocsr_gemini_benchmarks/metrics.json) |
| 2026-11-20 | `2026-11-20_benzimidazole_hybrid_run` | 3 сем. / `benzimidazole` *(MVP, val)* | Гибрид Anchor + Residual с HITL и технический аудит | Baseline 0.9576 → iter_2 **Micro-F1 = 0.9826** (Macro = 0.9844) | [`report.md`](2026-11-20_benzimidazole_hybrid_run/report.md) · [`metrics.json`](2026-11-20_benzimidazole_hybrid_run/metrics.json) |
| 2026-11-21 | `2026-11-21_benzimidazole_post_audit_eval` | 3 сем. / `benzimidazole` *(MVP, val)* | Точечная перегенерация и контрольный прогон после правок | **Micro-F1 = 0.9949**, **Macro-F1 = 0.9952**, ускорение **1.78×** | [`report.md`](2026-11-21_benzimidazole_post_audit_eval/report.md) · [`metrics.json`](2026-11-21_benzimidazole_post_audit_eval/metrics.json) |

## Аудит расхождений разметки (`GT Audits`) и границы хранения

- **Аудиты GT:** унифицированные таблицы ручной сверки эталона сведены в [`gt_audits/README.md`](gt_audits/README.md) — [`nanozymes.csv`](gt_audits/nanozymes.csv) (239 кейсов) и [`benzimidazole_antibiotics.csv`](gt_audits/benzimidazole_antibiotics.csv) (70 кейсов).
- **Граница хранения ([`.gitignore`](../../.gitignore)):** в Git отслеживаются `README.md`, `gt_audits/`, `_runtime/.gitkeep`, а внутри экспериментов — `report.md`, `metrics.json`, `config_snapshot/` и `result/`. Буфер `_runtime/`, тяжёлые JSON-логи LLM (`logs/`) и сырые дампы/ноутбуки (`raw/`) остаются локально.
