# Эксперимент: пилот MIPROv2 на текстовом корпусе `nanozymes` (2026-04-15)

- **Статус:** Завершён *(2-й семестр, первый пилот с `Qwen3.5` и парсингом через Gemini)*
- **Задача и сплит:** `nanozymes` (`ChemX: nanozymes`, только текст, `gemini-3-flash-preview`), `train_split = 9` (36 записей в `trainset.json`) / `val` (10 статей в `valset.json`; 15 в расширенной валидации)
- **Цель:** проверить рост качества при переходе на связку Qwen3.5-27B / 397B, чистый Markdown от Gemini 3 Flash и байесовский поиск DSPy MIPROv2.
- **Файлы:** [`report.md`](report.md), [`metrics.json`](metrics.json), [`config_snapshot/pilot.yaml`](config_snapshot/pilot.yaml), [`config_snapshot/nanozymes_pilot.txt`](config_snapshot/nanozymes_pilot.txt), [`result/optimized_instruction.txt`](result/optimized_instruction.txt).

---

## 1. Параметры запуска (`config_snapshot/pilot.yaml`)

- **Student LLM:** `qwen/qwen3.5-27b` (`OpenRouter`, `temperature = 0.0`, `top_p = 0.1`)
- **Teacher LLM:** `qwen/qwen3.5-397b-a17b` (`OpenRouter`, `temperature = 1.0`, `top_p = 0.9`)
- **MIPROv2:** `num_candidates = 8`, `num_trials = 25`, `max_bootstrapped_demos = 0`, `max_labeled_demos = 1`, `minibatch = true` (`minibatch_size = 4`), `random_seed = 42`.
- *Примечание по выборке:* в сохранённом запуске MLflow зафиксированы 10 примеров валидации (`optimization_20260415_032055`), а в расширенном сводном замере этапа — 72 кандидата и валидация 15 статей. Оба показателя отражены в [`metrics.json`](metrics.json).

---

## 2. Кривая оптимизации (`eval_full` из MLflow)

| Шаг полной оценки (`eval_full`) | F1 (%) | Прирост к бейзлайну | Примечание |
| :--- | :---: | :---: | :--- |
| step = 0 (Baseline) | 78.10 | — | Стартовый промпт (`nanozymes_pilot.txt`) |
| step = 1 (`eval_full_1`) | 82.28 | +4.18 п.п. | Раннее улучшение формулировок |
| **step = 2 (`eval_full_2`, Best)** | **84.92** | **+6.82 п.п.** | **Лучший результат пилота (0.781 → 0.849)** |
| step = 3 (`eval_full_3`) | 84.40 | +6.30 п.п. | — |
| step = 4 (`eval_full_4`) | 83.74 | +5.64 п.п. | — |
| step = 5 (`eval_full_5`) | 83.24 | +5.14 п.п. | — |

*(Итоговая пост-оценка в `nanozymes_pilot.meta.json` зафиксировала `f1 = 0.8101` на 10 примерах; все 25 минибатч-замеров от 71.02% до 98.90% сохранены в [`metrics.json`](metrics.json).)*

---

## 3. Выводы

- Переход на связку **Gemini-парсер + Qwen3.5 (27B / 397B)** поднял стартовый бейзлайн с 0.5811 (1-й семестр) до 0.7810, а оптимизация MIPROv2 дала прирост до **0.8492** (+6.82 п.п.).
- Анализ оставшихся ошибок показал, что синтаксический подбор не выводит скрытые правила разметки (реакция «субстрат + ко-субстрат», критерии включения, `syngony`), что привело к созданию этапа *Reverse Engineering* и бенчмарку [`2026-07-01_nanozymes_re_vs_mipro_benchmark`](../2026-07-01_nanozymes_re_vs_mipro_benchmark/report.md).
