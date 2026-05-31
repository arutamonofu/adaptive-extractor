# Adaptive Extractor

**A scientific data extraction system using Large Language Models with automatic prompt optimization.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-MIPROv2-green.svg)](https://github.com/stanfordnlp/dspy)

Adaptive Extractor automatically extracts structured experimental data from scientific PDF documents using LLMs with automatic prompt optimization via DSPy's MIPROv2 algorithm.

## Installation

```bash
# Clone the repository and install the package
pip install -e .

# Or set up using micromamba/conda
conda env create -f environment.yml
```

## Workflow

The system operates in three main stages:

1. **Ingest (Parse PDF to Markdown):**
   ```bash
   ae-parse
   ```
   *Uses Gemini API or local layout parsers to convert raw PDFs in `data/pdf/` to clean Markdown structure with visual anchors under `data/parsed/`.*

2. **Optimize (Self-Optimizing Prompts):**
   ```bash
   ae-optimize
   ```
   *Optimizes signature instructions and few-shot examples using Bayesian Optimization (DSPy MIPROv2) against ground truth labels under `data/ground_truth/`.*

3. **Extract (Batch Extract Structured Data):**
   ```bash
   ae-extract --agent data/agents/nanozymes_pilot.json
   ```
   *Runs data extraction using a serialized trained agent and outputs structured extractions to JSON under `data/extractions/`.*

## Configuration

System parameters and LLM settings are managed via YAML files in the `config/` directory:
- `core.yaml` — Global settings, active task, LLM provider configurations (API, Ollama, etc.), and cache thresholds.
- `ingestion.yaml` — Parser configurations and visual layout parameters.
- `extraction.yaml` — Agent matching rules and batch sizes.
- `optimization.yaml` — MIPROv2 hyperparameters (evaluation metrics, number of trials, few-shot parameters).
- `tasks/` — Directory containing target extraction schemas (e.g., `nanozymes/initial_schema.yaml`).

## Documentation

- [MIPROv2 Algorithm Analysis](docs/dspy_miprov2.md) — Breakdown of the prompt optimization stages.
- [DSPy Threshold Bug Fix](docs/dspy_miprov2_threshold_patch.md) — Verification and patching details for metric filtering.
- [MLflow Integration](docs/mlflow_integration.md) — Tracking experiments, trials, and metrics.