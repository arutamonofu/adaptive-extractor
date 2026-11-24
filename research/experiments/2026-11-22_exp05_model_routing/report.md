# EXP-05: Подбор и маршрутизация моделей по узлам пайплайна (2026-11-22 / 2026-11-23)

- **Сплит:** `benzimidazole_antibiotics` (`train` 6/405, `val` 6/314, 30 OCSR-кропов `2026-10-29`; `test` не тронут)
- **Базовая точка:** `gemini-3.8-flash` (ингестия/экстракция) + `qwen/qwen3.5-397b-a17b` (семантическая оптимизация)
- **Статус:** Завершён; рекомендации в `config/*.yaml` **не применены**
- **Данные:** [metrics.json](metrics.json) и исходные замеры треков 5.1–5.6 в [`tracks/`](tracks)

## Что проверяли

Поэтапный поиск более дешёвых моделей для всех стадий пайплайна поверх замороженных входов `EXP-01..04`:
- **5.1** Ингестия и текстовые стадии (`01_parse_text`, `02_manifest`, `03_localize`, `05_decompose`, `T3.5`).
- **5.2** OCR таблиц (`parse_visuals.table`).
- **5.3** OCSR (`Tier 3A`) и rescue провальных кропов (`Tier 3B`).
- **5.4** ReAct-роутер инструментов (`agent_node`).
- **5.5** Структурированный вывод (`Tier 4B`) и локальный ремонт SMILES (`Tier 4C`).
- **5.6** Семантическая оптимизация (`Tier 4A` и `Tier 4B`).

## Сводная маршрутная таблица (цены 2026 г.)

| Узел | Рекомендуемая модель | Альтернатива | Качество / ключевой факт | Время | Цена (2026) | Источник |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `01_parse_text` | **gemini-3.8-flash** (keep) | `gpt-6-luna` (CER 0.048) | `0/10` искажений | — | `$0.00623 / стр` | `track51` |
| `02_manifest` | `glm-5.3-flash` | `gemini` baseline | visual recall `0.980`, orphan `0` | `75.1 с` (6 док.) | `$0.00090 / док` | `track51` |
| `03_localize` | **gemini-3.8-flash** (keep) | `glm-5.3-flash` (mIoU 0.575) | низкое согласие рамок у альтернатив | — | — | `track51` |
| `05_decompose` | `glm-5.3-flash` | `qwen3-vl-235b` | mode acc `1.0`, count sim `0.976` | `46.4 с` (18 комп.) | `$0.00047 / док` | `track51` |
| `T3.5 hint` | `glm-5.3-flash` | `gemma-4-31b-it` | set F1 `1.0` (`10/10`) | `10.3 с` (10 хинтов) | `$0.000021 / хинт` | `track51` |
| `parse_visuals.table` | **gemini-3.8-flash med** (keep) | `gemini-3.8-flash low` (−47%) | CellF1 `0.9996`; OpenRouter теряют сноски | `209.8 с` (4 табл.) | `$0.0309 / табл.` | `track52` |
| `OCSR (Tier 3A)` | **gemini-3.8-flash med** (keep) | `qwen3-vl-235b` | `30/30` valid, `29/30` exact | — | `$0.0128 / кроп` | `track53` |
| `rdkit rescue (Tier 3B)` | **gemini-3.8-flash + CodeExec** | `glm-5.3-flash` (2/6 exact) | `4/5` exact, `5/5` valid (6-й кроп `402`) | `98.3 с / кроп` (`n=5`) | `$0.0386 / кроп` | `track53` |
| `ReAct router` | **gemini-3.8-flash** (keep) | `deepseek-v4.1-flash` (`vr 0.944 / tool 0.833`) | e2e проверялся только на `glm+deepseek`: `0.9774` | `135.0 с` (6 док., deepseek) | `$0.00284 / док` (deepseek) | `track54` |
| `structured output (Tier 4B)` | `deepseek-v4.1-flash` (`low`, усл.) | `qwen3.8-flash` (`0.9968`) | val `0.9934` vs `0.9973`, но `RJC` medium `0.9121` | `49.2 с / док` (295.2 с) | `$0.00352 / док` | `track55` |
| `local repair (Tier 4C)` | `deepseek-v4.1-flash` | `gpt-6-luna` (`2/2`) | `2/2` SMILES восстановлены | `7.0 с` (2 кейса) | `$0.000923 / ремонт` | `track55` |
| `Tier 4A worker` | `deepseek-v4.1-flash` (усл.) | — | `2/2` валидных док. (`max_tokens=32768`) | `797.6 с` (2 док.) | `$0.0658 / док` | `track56` |
| `Tier 4B synthesis` | **qwen3.5-397b** (keep) | `qwen3.8-flash` (error) | у дешёвых моделей пустые ответы на `>32k` | `9126.1 с` (EXP-04) | — | `track56` |
| `SemanticJudge` | `google/gemma-4-31b-it` (EXP-03) | `gemini-3.8-flash med` | κ = `0.9748` с валидаторами (в `config/` не включён) | `21.2 с` (80 пар) | `$0.20 / $0.40` за 1M | EXP-03 |

*Примечание: `parse_visuals.chart` и `parse_visuals.molecule` отдельными руками не запускались; `gemini-3.8-flash minimal` в 5.5 отклонён API (`400`).*

## Ключевые итоги по трекам

1. **Тяжёлые мультимодальные узлы (`01_parse_text`, `03_localize`, `table`, `OCSR`) остаются на `gemini-3.8-flash`:** дешёвые VLM из OpenRouter теряют числа, сноски таблиц и точность SMILES (хотя на `Canary n=4` выглядели обманчиво хорошо). `gemini-3.8-flash low` в 5.2 сэкономил `47%` на 4 таблицах, но не проверен на Full-10.
2. **Почему `config/*.yaml` оставлены без изменений:**
   - В Треке 5.5 `deepseek-v4.1-flash (low)` уложился в допуск на `val` (`0.9934` при цене `19.8×` ниже `$0.0695/док`), но на предрегистрированной пробе с выровненным `effort=medium` на `RJC.2023.1638382` пропустил 2 строки (`0.9121` против `1.0000` у `gemini-3.8-flash`).
   - В Треке 5.4 `deepseek-v4.1-flash` не дотянул до `100%` по `domain_tool_recall` (`0.8333`), а единственная e2e-связка (`glm-5.3-flash + deepseek`) просела по Micro-F1 до `0.9774` (против `0.9973`).
   - В Треке 5.6 дешёвые модели падали по потолку вывода (`16k–32k`), а офлайн-скоринг `4b_qwen38_flash_reasoning` по 4 докам дал `strict-F1 = 0.8956`.
3. **Практический вывод:** дешёвый OpenRouter-роутинг подтверждён только как резерв для вспомогательных шагов (`02_manifest`, `05_decompose`, `T3.5`, `Tier 4C`), но не как замена основного стека `gemini-3.8-flash` + `qwen3.5-397b`.
