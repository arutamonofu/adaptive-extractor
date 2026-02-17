# AutoEvoExtractor

**A scientific data extraction system using Large Language Models with automatic prompt optimization.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-MIPROv2-green.svg)](https://github.com/stanfordnlp/dspy)

AutoEvoExtractor automatically extracts structured experimental data from scientific PDF documents using LLMs with automatic prompt optimization via DSPy's MIPROv2 algorithm.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/autoevoextractor.git
cd autoevoextractor

conda env create -f environment.yml
conda activate aee
pip install -e .
```

### Basic Workflow

#### Step 1: Data Preparation

**Place PDF files in `data/pdfs/` directory:**
```bash
mkdir -p data/pdfs
cp /path/to/your/papers/*.pdf data/pdfs/
```

**Create ground truth data** (`data/ground_truth/{task}.csv`):
```csv
filename,formula,activity,length,km_value,vmax_value
paper1.pdf,Cu-TEMPO,oxidation,10,0.05,100
paper2.pdf,Fe-TEMPO,oxidation,12,0.08,150
```

**Create data splits file** (`data/splits/nanozymes.json`):
```json
{
  "train": ["paper1", "paper2", "paper3"],
  "val": ["paper4"],
  "test": ["paper5", "paper6"]
}
```
> ⚠️ **Important:** Ensure all document IDs in splits file exist in ground truth CSV (without `.pdf` extension).

[**Learn more about data structure →**](docs/data_artifacts.md)

#### Step 2: Run Pipeline

```bash
# 1. Parse PDFs
python scripts/parse.py --config config/default.yaml data/pdfs/

# 2. Optimize agent (requires ground truth and splits file)
python scripts/optimize.py --task nanozymes --config config/default.yaml

# 3. Extract data from new documents
python scripts/predict.py \
    --config config/default.yaml \
    --agent data/agents/nanozymes_latest.json \
    --task nanozymes

# 4. Evaluate (optional)
python scripts/evaluate.py \
    --task nanozymes \
    --predictions data/predictions/nanozymes_predictions.json \
    --ground-truth data/ground_truth/nanozymes.csv
```

[**Full CLI reference →**](docs/cli_reference.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [**Quick Start**](#quick-start) | Installation and basic workflow |
| [**CLI Reference**](docs/cli_reference.md) | Complete reference for all commands and arguments |
| [**Data Artifacts**](docs/data_artifacts.md) | Data structure, pipeline, file formats |
| [**Configuration**](docs/configuration.md) | Complete YAML and environment variable reference |
| [**Adding Tasks**](docs/adding_tasks.md) | Step-by-step guide for new extraction tasks |
| [**API Usage**](docs/api_usage.md) | Using AutoEvoExtractor as a Python library |
| [**Architecture**](docs/architecture.md) | System design for developers |
| [**MLflow Integration**](docs/mlflow_integration.md) | Experiment tracking setup |
| [**Troubleshooting**](docs/troubleshooting.md) | Common issues and solutions |

---

## Project Structure

```
autoevoextractor/
├── src/aee/
│   ├── domain/           # Business logic: tasks, entities, evaluation
│   ├── application/      # Use cases and services
│   ├── infrastructure/   # LLM, parsers, storage, MLflow
│   ├── interface/        # CLI commands
│   └── shared/           # Exceptions, utilities
├── scripts/              # Entry points: parse.py, optimize.py, predict.py, generate_manual_agent.py
├── config/
│   ├── *.yaml            # YAML configurations
│   └── initial_instructions/  # Initial instructions for optimization
└── data/                 # Project data
    ├── pdf/              # Source PDF files (place your PDFs here)
    ├── parsed/           # Parsed JSON documents (created by parse.py)
    ├── ground_truth/     # CSV annotations for training (created by you)
    ├── splits/           # Task-specific splits (created by you)
    │   └── nanozymes.json
    ├── agents/           # Trained agents (created by optimize.py)
    └── predictions/      # Extraction results (created by predict.py)
```

---

## Adding New Tasks

The system is extensible via plugins. Add a new extraction task in 4 steps:

```bash
mkdir -p src/aee/domain/tasks/mytask
```

1. **Data models** (`models.py`) — Pydantic models for experiments
2. **DSPy signature** (`signature.py`) — LLM extraction schema
3. **Row converter** (`converters.py`) — CSV row → experiment
4. **Task plugin** (`__init__.py`) — register the task

[**Full tutorial →**](docs/adding_tasks.md)

---

## Configuration

### Environment Variables

```bash
# LLM (Ollama)
export LLM__STUDENT__OLLAMA__OLLAMA_BASE_URL="http://localhost:11434"
export LLM__STUDENT__MODEL="llama3.2:3b"

# LLM (OpenAI/Anthropic)
export LLM__STUDENT__USE_OLLAMA=false
export LLM__STUDENT__MODEL="gpt-4"
export LLM__STUDENT__NON_OLLAMA__API_KEY="sk-..."

# Paths
export PATHS__PARSED_DIR="data/parsed"
export PATHS__AGENTS_DIR="data/agents"

# Optimization
export OPTIMIZATION__NUM_TRIALS="50"
export OPTIMIZATION__USE_CACHE="true"

# Logging
export PROJECT__LOG_LEVEL="DEBUG"
```

### YAML Configs

- `config/default.yaml` — production settings
- `config/default_fast.yaml` — fast optimization (testing)

[**Full configuration reference →**](docs/configuration.md)

---

## Requirements

- **Python:** 3.11+
- **LLM:** Ollama (local) or OpenAI/Anthropic API
- **Parsers:** Docling or Marker (optional)
- **Tracking:** MLflow (optional)

### Core Dependencies

| Package | Version |
|---------|---------|
| dspy-ai | >=2.5.0,<3.0.0 |
| pydantic | >=2.7.1,<3.0.0 |
| pandas | >=2.0.0,<3.0.0 |
| mlflow | >=2.10.0,<3.0.0 |
| streamlit | >=1.28.0,<2.0.0 |
