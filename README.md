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
   *Converts raw PDFs in `data/pdf/` to clean Markdown structure under `data/parsed/`. Supports both standard text parsing and visual-informed chart/table reconstruction.*
   *See [Ingestion Module Architecture & Workings](docs/ingestion_module.md).*

2. **Optimize (Self-Optimizing Prompts):**
   ```bash
   ae-optimize
   ```
   *Optimizes signature instructions and few-shot examples using Bayesian Optimization (DSPy MIPROv2) against ground truth labels under `data/ground_truth/`.*
   *See [Optimization Module Architecture & Workings](docs/optimization_module.md).*

3. **Extract (Batch Extract Structured Data):**
   ```bash
   ae-extract --agent data/agents/nanozymes_pilot.json
   ```
   *Runs batch data extraction using a serialized pre-trained agent and outputs structured extractions to JSON under `data/extractions/`.*
   *See [Extraction Module Architecture & Workings](docs/extraction_module.md).*

## Configuration

System parameters and LLM settings are managed via YAML files in the `config/` directory:
- `core.yaml` — Global settings, active task, LLM provider configurations, and cache thresholds.
- `ingestion.yaml` — Parser configurations and visual layout parameters.
- `extraction.yaml` — Agent matching rules and batch sizes.
- `optimization.yaml` — MIPROv2 hyperparameters (evaluation metrics, number of trials, few-shot parameters).
- `tasks/` — Directory containing target extraction schemas (e.g., `nanozymes/initial_schema.yaml`).

## Documentation

*   **Functional Modules**:
    *   [Ingestion Module](docs/ingestion_module.md) — Converting PDFs into structured Markdown with visual anchors and tables.
    *   [Optimization Module](docs/optimization_module.md) — Automated prompt tuning using DSPy MIPROv2 and MLflow tracking.
    *   [Extraction Module](docs/extraction_module.md) — Batch processing of documents using reconstructed agents and CoT reasoning.
*   **Deep Dives**:
    *   [MIPROv2 Algorithm Analysis](docs/dspy_miprov2.md) — Breakdown of the prompt optimization stages.
    *   [DSPy MIPROv2 Patches](docs/dspy_miprov2_patches.md) — Verification and patching details for metric thresholding and zero-shot support.
    *   [MLflow Integration](docs/mlflow_integration.md) — Tracking experiments, trials, and metrics.