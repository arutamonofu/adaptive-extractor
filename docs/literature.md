# Литературный обзор и анализ SOTA

Систематизация **24 научных источников** (22 статьи, бенчмарка и обзора + 2 профильных отчёта по стратегиям `few-shot`), сравнительный анализ систем извлечения данных, методов оптимизации промптов и инструментов разбора PDF. Локальная библиотека PDF хранится в `research/references/literature/{extraction,optimization,few-shot}/` *(исключена из Git)*.

---

## 1. Системы извлечения научных данных (SOTA)

Подходы к автоматическому извлечению из статей разделяются на три класса:
1. **Правила и классический NER** (`ChemDataExtractor`): точны на узких шаблонах, но хрупки при смене вёрстки и домена.
2. **Zero-shot LLM**: гибки без разметки, но склонны к галлюцинациям и пропускам на сложных схемах.
3. **Мультиагентная оркестрация и декларативная оптимизация**: разделяют координатора, визуальные/валидационные инструменты и автоматический синтез промптов.

### Сравнение передовых реализаций

| Проект / Система | Предметная область и данные | Качество | Архитектура | Год |
| :--- | :--- | :--- | :--- | :---: |
| `nanoMINER` | Нанозимы: формулы, морфология, кристаллические системы, кинетика (Km, Vmax) | F1 = 0.80; F1(Km) = 0.94 | ReAct (GPT-4o): Vision (YOLO + GPT-4o) + NER (Mistral/Llama) | 2025 |
| `EnzyExtract` | Ферменты: kcat, Km, организмы, субстраты, pH и температура | F1(Km) = 0.80 | OCR (ResNet-18) + TableTransformer + fine-tuned GPT-4o-mini | 2025 |
| `GPT-4o` (Murton et al.) | Клинические исследования (РКИ): дизайн, демография, OS/PFS | F1(прост.) > 0.85; F1(сложн.) = 0.30 | Цепочка промптов (GPT-4o) | 2025 |

### Ключевые выводы SOTA

- **Барьеры переноса (`nanoMINER`):** классические мультиагентные решения требуют четырёх трудоёмких ручных операций при смене домена — переписывания промптов в модулях, жёсткого программирования схем, разметки и дообучения NER, а также обучения CV-детекторов.
- **Разделение ролей и иерархические таблицы:** оркестратор (`ReAct`) должен вызывать узкие инструменты (Vision, код, валидаторы), а для вложенных таблиц эффективен древовидный разбор заголовков (*TreeThinker* / *RealHiTBench*).
- **Прецизионность символов, HITL и верификация:** грубый OCR путает порядки величин (`m` vs `μ`), программная кросс-верификация подавляет галлюцинации, а работа эксперта в режиме коррекции (HITL) ускоряет подготовку разметки более чем на 50%.

---

## 2. Методы автоматической оптимизации промптов

| Алгоритм | Принцип работы | Задача | Бейзлайн | После оптимизации | Год |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`MIPROv2`** | Байесовский поиск по инструкциям и `few-shot`-наборам (*Teacher — Student*) | Детекция галлюцинаций LLM | Precision = 0.81 | **Precision = 0.86** | 2024 |
| `OPRO` | Итеративное переписывание инструкции LLM-оптимизатором | Логика и математика | Precision = 0.72 | Precision = 0.80 | 2023 |
| `Bootstrap RS` | Случайный поиск лучшего набора примеров без синтеза правил | Детекция галлюцинаций LLM | Macro F1 = 0.80 | Macro F1 = 0.81 | 2024 |

- Семантически близкие формулировки дают разброс качества свыше 10%, а оптимизатор находит неочевидные ограничения лучше ручного подбора.
- Из-за деградации внимания в середине длинного контекста (*Lost in the Middle*) ключевые примеры и правила нужно размещать ближе к краям контекстного окна (перед целевым документом).

---

## 3. Стратегии формирования `few-shot`-демонстраций

По итогам анализа литературы сформированы 4 критерия отбора примеров (реализованы в [`sampling_benzimidazole.ipynb`](../research/notebooks/sampling_benzimidazole.ipynb)):
1. **Полнота заполнения схемы (>80%):** гарантирует сигнал по каждому атрибуту и предотвращает пропуск редких полей.
2. **Структурное разнообразие:** контрастная вёрстка (сложные таблицы, нарративный текст, *Supporting Information*) важнее семантической близости статей.
3. **Покрытие граничных форматов:** научная нотация, спецсимволы Unicode и смешанные единицы обучают нормализации.
4. **Обучение на ошибках и дистракторах:** цепочка рассуждений с явной ссылкой на фрагмент текста (*evidence*) и объяснением отброшенных ложных кандидатов (*negative samples*) снижает FP.

---

## 4. Извлечение визуальных блоков из PDF и оцифровка графиков

### Семь классов проблем PDF-фигур

1. **Разные представления PDF:** текстовый вектор, встроенные растры, сканы и гибридные страницы ведут к потере осей или подписей при наивной обрезке.
2. **Нецелевые объекты:** графики и схемы перемешаны с ЯМР-спектрами, логотипами и graphical abstract.
3. **Разброс подписей:** нестандартные метки (`Fig.`, `Scheme`, `Extended Data`), расположение подписи на другой странице.
4. **Multi-panel / multipart:** панели A/B/C/D под общим caption — модель смешивает соседние молекулы и графики.
5. **Нарушенный порядок и дубли:** расхождение reading order и визуального порядка порождает дубли в манифесте.
6. **Ошибки OCR / bbox:** потеря мелких радикалов/индексов и обрезка меток панелей.
7. **Дефицит контекста:** расшифровка осей, штаммов и единиц часто находится только в тексте или SI.

### Сравнение парсеров и экстракторов фигур

| Решение | Тип и архитектура | Сильные стороны | Ключевые ограничения | Применимость |
| :--- | :--- | :--- | :--- | :--- |
| `Docling` | OSS layout-парсер (`PictureItem`) | Удобная модель документа и экспорт кропов | Нет декомпозиции панелей; теряет мелкие индексы | Отклонён после 1-го семестра |
| `MinerU` | ML-пайплайн (`PDF-Extract-Kit`) | Детальные `middle.json` и привязка `captions` | Тяжёлый GPU-стек, сдвиги таблиц и потери LaTeX | Отклонён после июльского теста |
| `Marker` | Surya OCR + опц. LLM | Быстрая генерация Markdown | OCR-ошибки в формулах и разрывы колонок | Не подходит как ядро |
| `GROBID` | Scholarly ML → TEI/XML | Сильная привязка научных `captions` и ссылок | Не отдаёт PNG-кропы без доработки | Избыточен для API-стека |
| `PDFFigures 2.0` | PDF-native геометрия + эвристики | Хороший бейзлайн для одиночных научных фигур | Устарел, не делит multi-panel схемы | Только как референс |
| `DeepFigures` | CNN-детектор + PDFFigures 2.0 | Классический research-бейзлайн | Нет production-обвязки | Не рекомендован |
| `Adobe PDF Extract` | Облачный проприетарный API | Готовые JSON, PNG и CSV/XLSX таблиц | Закрытый сервис, цена, vendor lock-in | Не воспроизводим в OSS |
| `Mistral OCR` | Облачный VLM/OCR-парсер | Удобный Markdown с плейсхолдерами и bbox | Для точных панелей нужен отдельный детектор | Ориентир архитектуры заглушек |

Настольные оцифровщики графиков **PlotDigitizer** и **Engauge** исключены из автоматического конвейера из-за отсутствия программного API — вместо них применена декомпозиция и векторный кроп в [`VisualParsingService`](../src/ae/ingestion/service.py) поверх `PyMuPDF` (до 1200 DPI) и `Gemini 3.8 Flash`.

---

## 5. Аннотированная библиография

Рабочий конспект и Q&A по источникам: [NotebookLM проекта](https://notebooklm.google.com/notebook/398c713a-7bf5-4091-a972-e9d12a60b2ec).

### 5.1. Извлечение данных и разбор таблиц (`research/references/literature/extraction/`)

1. **[2025-12]** *Optimizing Data Extraction from Materials Science Literature: A Study of Tools Using Large Language Models* — сравнение LLM-инструментов для извлечения ширины запрещённой зоны ([arXiv:2512.09370](https://www.arxiv.org/abs/2512.09370)).
2. **[2025-10]** *Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX* — бенчмарк `ChemX` для оценки агентов на нанозимах и малых молекулах ([arXiv:2510.00795v1](https://arxiv.org/abs/2510.00795v1)).
3. **[2025-10]** *Harnessing Large-Language Models for Efficient Data Extraction in Systematic Reviews* — роль промпт-инжиниринга в систематических обзорах ([PMC12559671](https://pmc.ncbi.nlm.nih.gov/articles/PMC12559671/)).
4. **[2025-08]** *A structure-oriented kinetics dataset of enzyme-substrate interactions* — агентный сбор кинетики фермент–субстрат ([Nature Sci. Data](https://www.nature.com/articles/s41597-025-05829-5)).
5. **[2025-08]** *Finding the dark matter: Large language model-based enzyme kinetic data extractor* (`EnzyExtract`) — OCR + TableTransformer + дообученный GPT-4o-mini ([PMC12355964](https://pmc.ncbi.nlm.nih.gov/articles/PMC12355964/)).
6. **[2025-06]** *Agent-based multimodal information extraction for nanomaterials* (`nanoMINER`) — базовая мультиагентная SOTA-система для нанозимов ([npj Comput. Mater.](https://www.nature.com/articles/s41524-025-01674-7)).
7. **[2025-06]** *RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark* — оценка LLM на иерархических таблицах и подход *TreeThinker* ([arXiv:2506.13405v1](https://arxiv.org/abs/2506.13405v1)).
8. **[2025-04]** *Data extraction methods for systematic review (semi)automation* — обзор полуавтоматических методов с человеком в контуре ([PMC8361807](https://pmc.ncbi.nlm.nih.gov/articles/PMC8361807/)).
9. **[2024-02]** *Structured information extraction from scientific text with large language models* — извлечение свойств полимеров и композитов ([Nat. Commun.](https://www.nature.com/articles/s41467-024-45563-x)).
10. **[2023-09]** *ReactionDataExtractor 2.0* — глубокое обучение для оцифровки схем химических реакций ([ACS JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00422)).
11. **[2016-09]** *ChemDataExtractor* — классический NLP-инструментарий извлечения химических сущностей ([ACS JCIM](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00207)).

### 5.2. Оптимизация промптов (`research/references/literature/optimization/`)

12. **[2024-12]** *A Comparative Study of DSPy Teleprompter Algorithms* — сравнение телепромптеров `DSPy` (`MIPROv2` vs `Bootstrap RS`) ([arXiv:2412.15298v1](https://arxiv.org/abs/2412.15298v1)).
13. **[2024-04]** *Large Language Models as Optimizers* (`OPRO`) — оптимизация инструкций через естественный язык ([arXiv:2309.03409](https://arxiv.org/abs/2309.03409)).

### 5.3. In-Context Learning и `few-shot` (`research/references/literature/few-shot/`)

14. **[2026-01]** *Lost in the Noise: How Reasoning Models Fail with Contextual Distractors* — фильтрация контекстного шума в примерах ([arXiv:2601.07226](https://arxiv.org/abs/2601.07226)).
15. **[2025-07]** *Failures Are the Stepping Stones to Success: Leveraging Negative Samples* — негативные примеры против галлюцинаций ([arXiv:2507.23211](https://arxiv.org/abs/2507.23211)).
16. **[2025-06]** *Agent-based multimodal information extraction for nanomaterials* — копия `nanoMINER` как база правил домена нанозимов ([Nature](https://www.nature.com/articles/s41524-025-01674-7)).
17. **[2025-06]** *The Role of Diversity in In-Context Learning* — приоритет структурного разнообразия над семантической близостью ([arXiv:2505.19426v2](https://arxiv.org/abs/2505.19426v2)).
18. **[2024-12]** *Curriculum Demonstration Selection for In-Context Learning* — отбор примеров по нарастающей сложности ([arXiv:2411.18126](https://arxiv.org/abs/2411.18126)).
19. **[2024-11]** *In-Context Compositional Generalization for Large Vision-Language Models* — разнообразие макетов и таблиц для VLM ([EMNLP 2024](https://aclanthology.org/2024.emnlp-main.996/)).
20. **[2024-02]** *In-Context Principle Learning from Mistakes* (`LEAP`) — обучение на сложных ошибках эффективнее простых позитивных кейсов ([arXiv:2402.05403](https://arxiv.org/abs/2402.05403)).
21. **[2023-12]** *Chain of Thought with Explicit Evidence Reasoning* — привязка выводов к цитатам текста ([EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.153/)).
22. **[2023-07]** *Lost in the Middle: How Language Models Use Long Contexts* — падение точности в середине длинного контекста ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)).
23. **Аналитический обзор:** `Beyond the Super-Example: A Hybrid Demonstration Strategy for Maximizing Reasoning Accuracy in Low-Shot Scientific Extraction.pdf` *(локально в `few-shot/`)*.
24. **Аналитический обзор:** `Complexity vs. Simplicity of Examples.pdf` *(локально в `few-shot/`)*.
