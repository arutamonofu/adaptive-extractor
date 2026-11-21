# Эксперимент: бенчмарк 2-го семестра — Reverse Engineering vs Baseline vs MIPROv2 (2026-07-01)

- **Статус:** Завершён *(2-й семестр, закрытие текстового PoC `nanozymes`)*
- **Задача и сплит:** `nanozymes` (`ChemX: nanozymes`, только текст, Gemini 3 Flash), пул обучения / `few-shot` 9 сложных статей (Score Decay), стратифицированный `val` из 15 статей (UMAP, кластеры 0–3)
- **Цель:** статистически проверить, превосходит ли автоматическое семантическое выведение правил (*Reverse Engineering*) + тюнинг MIPROv2 ручные промпты (`nanoMINER` и Manual).
- **Файлы:** [`report.md`](report.md), [`metrics.json`](metrics.json), [`config_snapshot/`](config_snapshot) (`optimization.yaml`, `initial_schema.yaml`, `baseline_manual_instruction.txt`), [`result/`](result) (`generated_schema.yaml`, `generated_instruction.txt`, `re_mipro_instruction.txt`).

---

## 1. Условия и дизайн эксперимента

- **Модели:** Student `qwen/qwen3.5-27b` (`temperature = 0.0`), Teacher `qwen/qwen3.5-397b-a17b` (`temperature = 1.0`).
- **Настройки MIPROv2 (`config_snapshot/optimization.yaml`):** `num_candidates = 9`, `num_trials = 27`, `minibatch = false`, `max_bootstrapped_demos = 0`, `max_labeled_demos = 1`, `random_seed = 42` ([`hyperparameters_analysis.ipynb`](../../notebooks/hyperparameters_analysis.ipynb)).
- **Контроль мощности и шума:** объём выборки N = 15 и подавление seed-дисперсии (< 10%) обоснованы в [`power_analysis.ipynb`](../../notebooks/power_analysis.ipynb) (методология в [`docs/methodology.md`](../../../docs/methodology.md#5-научная-методология-экспериментов-и-выбора-моделей)).

---

## 2. Ключевые метрики (4 группы сравнения)

| Группа / Пайплайн | Mean F1 | Median F1 | Примечание |
| :--- | :---: | :---: | :--- |
| `Baseline (nanoMINER)` | 0.6329 | 0.6452 | Промпт из оригинальной статьи `nanoMINER` |
| `Baseline (Manual)` | 0.6592 | 0.6500 | Вручную настроенный промпт под целевую схему (~8 КБ) |
| **Reverse-engineered** | **0.8804** | **0.9032** | Выведение скрытых правил и расширение инструкции (~40.6 КБ, **+22.1 п.п.** к Manual) |
| `Reverse-engineered + MIPROv2` | 0.8811 | 0.8750 | Поверх `Reverse-engineered` запущен байесовский поиск MIPROv2 |

*(Промежуточные замеры MIPROv2 перед финальной сводкой: прогон 2026-06-20 дал F1 = 0.8737; прогон 2026-06-21 показал рост 82.30% → 85.62% → 87.19% → 87.30% → 88.93% с 1 размеченным примером.)*

---

## 3. Статистическая проверка гипотез

| Тест и сравнение | Статистика / p | Поправка Холма | Вывод |
| :--- | :---: | :---: | :--- |
| **Критерий Фридмана** *(все 4 группы)* | χ² = 39.26, p = 1.53e-8 | — | H0 отвергнута (p < 0.05): различия значимы |
| **Уилкоксон:** Manual vs Reverse-engineered | p = 0.00006 | **p_Holm = 0.00037** | **Статистически значимый рост**, гипотеза PoC доказана |
| **Уилкоксон:** Reverse-engineered vs RE + MIPROv2 | p = 0.45994 | **p_Holm = 0.55009** | **Различие незначимо** (p > 0.05) |

---

## 4. Выводы

1. **Закрытие текстового PoC:** без ручного промпт-инжиниринга система вышла на **Mean F1 = 0.8804** (Median 0.9032), достоверно превзойдя оба ручных бейзлайна (0.6329 и 0.6592). Системный аудит 239 расхождений разметки по домену зафиксирован в [`gt_audits/nanozymes.csv`](../gt_audits/nanozymes.csv) ([`gt_audits/README.md`](../gt_audits/README.md)).
2. **Главный драйвер — семантическое выведение правил:** незначимость дополнительного прогона MIPROv2 поверх Reverse-engineered (`p_Holm = 0.55009`) легла в основу [ADR-002](../../../docs/journal.md#adr-002-семантическая-оптимизация-anchor--residual-с-hitl-как-основной-контур-и-опциональный-dspy-miprov2-2026-11-принято) — основным стал `ae-optimize-semantic`, а `ae-optimize-syntactic` переведён в опциональный режим.
3. **Переход к мультимодальности:** по итогам этапа проект переключён на домен `benzimidazole_antibiotics`.
