# AutoEvoExtractor

**A scientific data extraction system using Large Language Models with automatic prompt optimization.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-MIPROv2-green.svg)](https://github.com/stanfordnlp/dspy)

AutoEvoExtractor automatically extracts structured experimental data from scientific PDF documents using LLMs with automatic prompt optimization via DSPy's MIPROv2 algorithm.


## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli_reference.md) | Complete command reference |
| [Data Artifacts](docs/data_artifacts.md) | Data structure and file formats |
| [Configuration](docs/configuration.md) | YAML and environment variables |
| [Adding Tasks](docs/adding_tasks.md) | Creating new tasks (YAML) |
| [Architecture](docs/architecture.md) | System design |

---

## Scripts

The project provides three main scripts for the extraction pipeline:

| Script | Purpose |
|--------|---------|
| `scripts/extract.py` | Run batch extraction on documents using a trained agent |
| `scripts/optimize.py` | Optimize extraction agent using MIPROv2 (DSPy) |
| `scripts/generate_manual_agent.py` | Create manual agent from train_manual split examples |

### Common CLI Patterns

All scripts share common CLI conventions:

- `--config PATH` — Path to YAML configuration file (required)
- `--output PATH` — Override default output path (optional, script-specific)
- `--run-name NAME` — MLflow run name prefix (optimize.py only)
- `--no-mlflow` — Disable MLflow tracking (optimize.py only)

### Usage Examples

**Extract data from documents:**
```bash
python scripts/extract.py --config config/systems/dev.yaml --agent data/agents/nanozymes_v1.json
```

**Optimize agent with MIPROv2:**
```bash
python scripts/optimize.py --config config/systems/dev.yaml --run-name A1_high
```

**Generate manual agent:**
```bash
python scripts/generate_manual_agent.py --config config/systems/dev.yaml
```

---

## Установка

### Для разработки (рекомендуется)

Создайте conda-окружение со всеми зависимостями:

```bash
conda env create -f environment.yml
conda activate aee
```

**Что устанавливается:**
- Все зависимости через conda (стабильные бинарные пакеты)
- Jupyter-инструменты для интерактивной разработки
- Минимальный набор pip-пакетов (только что недоступно в conda)

### Для продакшн-развёртывания

Установите как Python-пакет:

```bash
pip install .
```

**Что устанавливается:**
- Только runtime-зависимости для работы пакета
- Без инструментов разработки (Jupyter и др.)

---

## Зависимости

Проект использует два файла зависимостей:

| Файл | Назначение | Когда использовать |
|------|------------|-------------------|
| `environment.yml` | Полная среда разработки | Разработка, эксперименты, Jupyter |
| `pyproject.toml` | Runtime-зависимости пакета | Развёртывание, `pip install .` |

**Почему два файла:**
- `environment.yml` обеспечивает воспроизводимость научной среды (conda + pip)
- `pyproject.toml` описывает минимальные требования для работы пакета

---

## Requirements

- **Python:** 3.11+
- **LLM:** Ollama (local) or OpenAI/Anthropic API
- **Core dependencies:** dspy-ai, pydantic, pandas, mlflow, docling

[Full list →](pyproject.toml)
