# AutoEvoExtractor

**A scientific data extraction system using Large Language Models with automatic prompt optimization.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-MIPROv2-green.svg)](https://github.com/stanfordnlp/dspy)

AutoEvoExtractor automatically extracts structured experimental data from scientific PDF documents using LLMs with automatic prompt optimization via DSPy's MIPROv2 algorithm.

---

## Quick Start

### Installation

```bash
git clone https://github.com/ai-chem/AutoEvoExtractor.git
cd autoevoextractor

conda env create -f environment.yml
conda activate aee
pip install -e .
```

### Basic Workflow

> **Example below is for the nanozymes task.** For other tasks, adjust file paths and field names accordingly.

#### Step 0: Configure LLM Provider

**Quick Setup with `.env` file:**

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Environment (dev, test, prod)
AEE_ENV=dev

# LLM API Keys (required for non-Ollama providers)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ollama URLs
OLLAMA_STUDENT_BASE_URL=http://localhost:11434
OLLAMA_TEACHER_BASE_URL=http://localhost:11434

# MLflow Tracking
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Logging
LOG_LEVEL=INFO
```

**For Ollama (local):**
```bash
# Set Ollama server URL (if different from default)
export OLLAMA_STUDENT_BASE_URL="http://localhost:11434"
export OLLAMA_TEACHER_BASE_URL="http://localhost:11434"

# Pull required models
ollama pull mistral-small3.1-24b-128k:latest
ollama pull gpt-oss:120b
```

**For OpenAI/Anthropic API:**
```bash
export LLM__STUDENT__USE_OLLAMA=false
export LLM__STUDENT__MODEL="gpt-4"
export LLM__STUDENT__NON_OLLAMA__API_KEY="sk-..."
```

#### Step 1: Data Preparation

**Place PDF files in `data/pdf/` directory:**
```bash
mkdir -p data/pdf
cp /path/to/your/papers/*.pdf data/pdf/
```

**Create ground truth data** (`data/ground_truth/nanozymes.csv`):
```csv
filename,formula,activity,syngony,surface,length,km_value,km_unit,vmax_value,vmax_unit,ph,temperature
paper1.pdf,Cu-TEMPO,oxidase,amorphous,PEG,10,0.05,mM,100,mM/s,7.0,25.0
paper2.pdf,Fe3O4,peroxidase,cubic,naked,12,0.08,mM,150,mM/s,6.5,30.0
paper3.pdf,Au,oxidase,hexagonal,citrate,15,0.12,mM,200,mM/s,7.5,20.0
```

**Required CSV columns:**
- `filename` — PDF filename (must match file in `data/pdf/`)
- `formula` — Chemical formula (required)
- `activity` — Catalytic activity type (required)
- Other fields depend on your task definition (see `task.yaml`)

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
# 1. Parse PDFs (PDF directory configured via paths.pdf_dir in YAML config)
python scripts/parse.py --config default.yaml

# Parse with overwrite (re-parse existing files)
python scripts/parse.py --config default.yaml --overwrite

# 2. Optimize agent (requires ground truth and splits file)
python scripts/optimize.py --config default.yaml

# Optimize with custom run name
python scripts/optimize.py --config default.yaml --run-name "A1_temp0.5"

# Optimize without MLflow tracking
python scripts/optimize.py --config default.yaml --no-mlflow

# Fast test optimization (fewer trials, smaller dataset)
python scripts/optimize.py --config default_fast.yaml

# 3. Extract data from new documents
python scripts/extract.py \
    --config default.yaml \
    --agent data/agents/nanozymes_latest.json

# 4. Create manual agent from examples (optional)
python scripts/generate_manual_agent.py --config default.yaml

# 5. Evaluate (optional)
# Evaluation is performed manually or custom scripts.
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
| [**Adding Tasks**](docs/adding_tasks.md) | Step-by-step guide for new extraction tasks (YAML & Python) |
| [**Architecture**](docs/architecture.md) | System design for developers |
| [**MLflow Integration**](docs/mlflow_integration.md) | Experiment tracking setup |
| [**DSPy MIPRO Patch**](docs/dspy_mipro_threshold_patch.md) | DSPy MIPRO threshold patch documentation |

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
├── scripts/              # Entry points
│   ├── parse.py          # Parse PDFs to JSON
│   ├── optimize.py       # Optimize agent via MIPROv2
│   ├── extract.py        # Extract data using trained agent
│   ├── generate_manual_agent.py  # Create manual agent from examples
│   └── patch_dspy_mipro_threshold.py  # DSPy MIPRO patch
├── config/
│   ├── *.yaml            # YAML configurations (default.yaml, default_fast.yaml)
│   └── initial_instructions/  # Initial instructions for optimization
│       └── nanozymes_sota.txt
└── data/                 # Project data
    ├── pdf/              # Source PDF files (place your PDFs here)
    ├── parsed/           # Parsed JSON documents (created by parse.py)
    ├── ground_truth/     # CSV annotations for training (created by you)
    ├── splits/           # Task-specific splits (created by you)
    │   └── nanozymes.json
    ├── agents/           # Trained agents (created by optimize.py)
    └── extractions/      # Extraction results (created by extract.py)
```

---

## Adding New Tasks

AutoEvoExtractor supports **two approaches** for adding new extraction tasks:

### Approach 1: YAML-Based Configuration (Recommended)

Define your task declaratively in a YAML file. The system automatically generates Pydantic models, DSPy signatures, and row converters.

**Time Estimate**: 15-30 minutes for a simple task

```bash
mkdir -p src/aee/domain/tasks/mytask
```

Create `src/aee/domain/tasks/mytask/task.yaml`:

```yaml
name: mytask
description: Extract my domain-specific experiments

# Fields to extract
fields:
  field_name:
    type: str                    # str, int, float, bool
    description: "Field description"
    required: true
    alt_names:
      - alternative_name

# Evaluation settings
compare_fields:
  - field_name
float_tolerance: 0.05            # 5% tolerance for float comparisons
```

[**Full YAML tutorial →**](docs/adding_tasks.md#step-by-step-guide-yaml-approach)

### Approach 2: Python-Based Plugin (Advanced)

Create a custom task plugin in Python for complex scenarios requiring custom validation logic or non-standard field types.

```bash
mkdir -p src/aee/domain/tasks/mytask
```

1. **Data models** (`models.py`) — Pydantic models for experiments
2. **DSPy signature** (`signature.py`) — LLM extraction schema
3. **Row converter** (`converters.py`) — CSV row → experiment
4. **Task plugin** (`__init__.py`) — register the task

**Time Estimate**: 2-4 hours for a complete task

[**Full Python tutorial →**](docs/adding_tasks.md#step-by-step-guide-python-approach)

---

## Configuration

### YAML Configs

Configuration files are stored in `config/` directory:

| File | Description |
|------|-------------|
| `default.yaml` | Production configuration with all required fields |

**Configuration Priority** (highest to lowest):

1. **Environment variables** (`.env` file, `AEE__*` overrides)
2. **CLI arguments** (`--config`, `--overwrite`, etc.)
3. **YAML configuration files** (`config/default.yaml`, `config/<env>.yaml`)
4. **Internal defaults**

> ⚠️ **Important:** API keys (OpenAI, Anthropic, Gemini) must be set via environment variables only — never in YAML files.

**Key configuration sections:**

```yaml
# LLM settings
llm:
  student:
    use_ollama: true           # Use Ollama (true) or API provider (false)
    model: "mistral-small3.1-24b-128k:latest"
    timeout: 600
    enable_cache: true         # Cache LLM responses

# Optimization settings
optimization:
  num_trials: 70               # Number of optimization trials
  use_cache: true              # Cache LLM responses during optimization
  verbose: true                # Verbose logging

# Task settings
task:
  name: "nanozymes"
  initial_instruction_file: "config/initial_instructions/nanozymes_sota.txt"
```

All LLM, optimization, and infrastructure parameters are **explicitly required** in YAML configuration. Only safe path defaults are provided.

### Environment Variables

Any setting can be overridden via environment variables (double underscore for nested):

```bash
# ===== LLM Configuration =====
# Use non-Ollama provider
export LLM__STUDENT__USE_OLLAMA=false
export LLM__STUDENT__MODEL="gpt-4"
export LLM__STUDENT__NON_OLLAMA__API_KEY="sk-..."

# Ollama server URLs (separate for student and teacher)
export OLLAMA_STUDENT_BASE_URL="http://localhost:11434"
export OLLAMA_TEACHER_BASE_URL="http://localhost:11434"

# ===== Paths =====
export PATHS__PDF_DIR="data/my_pdfs"
export PATHS__PARSED_DIR="data/my_parsed"
export PATHS__SPLITS_FILE="data/splits/mytask.json"

# ===== Optimization =====
export OPTIMIZATION__NUM_TRIALS="50"
export OPTIMIZATION__USE_CACHE="true"
export OPTIMIZATION__VERBOSE="true"

# ===== Task =====
export TASK__NAME="nanozymes"
export TASK__INITIAL_INSTRUCTION_FILE="config/initial_instructions/nanozymes_sota.txt"

# ===== MLflow Tracking =====
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# ===== Logging =====
export LOG_LEVEL="DEBUG"
export PROJECT__LOG_LEVEL="DEBUG"
```

**Note:** API keys should **only** be set via environment variables, never in YAML files.

[**Full configuration reference →**](docs/configuration.md)

---

## Testing

The project uses pytest for testing with unit, integration, and end-to-end (e2e) test suites.

### Run Tests

```bash
# Run all tests
pytest

# Run unit tests with coverage
pytest tests/unit -v --cov=src/aee --cov-report=term-missing

# Run integration tests (excluding slow tests)
pytest tests/integration -v -m "not slow"

# Run end-to-end tests
pytest tests/e2e -v

# Run specific test file
pytest tests/unit/domain/test_task_loader.py -v
```

### Test Structure

| Directory | Purpose |
|-----------|---------|
| `tests/unit/` | Unit tests for individual components |
| `tests/integration/` | Integration tests for component interactions |
| `tests/e2e/` | End-to-end workflow tests |

[**Testing Guide →**](docs/TESTING_GUIDE.md)

---

## Requirements

- **Python:** 3.11+
- **LLM:** Ollama (local) or OpenAI/Anthropic API
- **Parsers:** Docling or Marker (optional)
- **Tracking:** MLflow (optional)

### Core Dependencies

| Package | Version |
|---------|---------|
| dspy-ai | >=2.4.4,<4.0.0 |
| pydantic | >=2.7.1,<3.0.0 |
| pydantic-settings | >=2.0.0,<3.0.0 |
| pandas | >=2.0.0,<3.0.0 |
| numpy | >=1.24.0,<3.0.0 |
| mlflow | >=2.10.0,<3.0.0 |
| docling | >=2.0.0,<3.0.0 |
| marker-pdf | >=1.0.0,<2.0.0 |

**Full dependency list:** See [pyproject.toml](pyproject.toml) or [environment.yml](environment.yml)
