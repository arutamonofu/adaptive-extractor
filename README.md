# AutoEvoExtractor

**A scientific data extraction system using Large Language Models with automatic prompt optimization.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
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

## Requirements

- **Python:** 3.12+
- **LLM providers:** Ollama, HuggingFace Transformers (local), or API (OpenRouter, OpenAI, Anthropic, Gemini)
- **PDF parser:** Gemini API (primary) or Marker (local, GPU, optional)
- **GPU server:** NVIDIA GPU with CUDA 12.x support (A6000 recommended)

### Critical optimizations for local Transformers inference

When running models like **Qwen3.5-27B** locally, these three packages are required for acceptable performance (~5-7 tok/sec vs ~2 tok/sec without them):

| Package | Purpose | What happens if missing |
|---------|---------|------------------------|
| `flash-attn` | FlashAttention-2 kernel | Falls back to slower sdpa attention |
| `causal-conv1d` | SSM/linear attention fast path | **75% of layers** use slow torch conv1d |
| `bitsandbytes` | 4-bit quantization (NF4) | Model won't fit in GPU memory |

These are auto-detected by `transformers` — no code changes needed. They are listed in `pyproject.toml` and installed automatically with `[dev,quant]` extras.

After installation, verify kernel status:
```bash
python scripts/benchmark_inference.py --config config/systems/example.yaml
```

## Installation

### Option 1: Conda (local development with Jupyter)

```bash
conda env create -f environment.yml
conda activate aee
```

This installs all dependencies including Jupyter notebooks. The `environment.yml` uses `pyproject.toml` as the single source of truth for Python packages.

> **Important (NFS/home environments):** If your conda env is on an NFS-mounted home directory, you may need to set `LD_LIBRARY_PATH` so that `causal-conv1d` and Triton kernels can find the correct libraries:
> ```bash
> export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
> ```

### Option 2: pip venv (GPU server, A6000)

```bash
python3.12 -m venv /opt/aee/env
source /opt/aee/env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,quant]"     # includes flash-attn, causal-conv1d, bitsandbytes
```

> **Note:** Do not install `[notebook]` extras on the server.

### Optional dependency groups

| Group | Purpose | Install command |
|-------|---------|-----------------|
| `dev` | Testing, linting, type checking | `-e ".[dev]"` |
| `quant` | Model quantization (4bit/8bit) | `-e ".[dev,quant]"` |
| `notebook` | Jupyter, matplotlib, seaborn | `-e ".[dev,notebook]"` |
| `marker` | Marker PDF parser (GPU, requires `transformers<5.0`) | See note below |

> **Note:** `transformers`, `flash-attn`, `causal-conv1d`, and `accelerate` are in the base dependency list — they install regardless of extras. The `[quant]` group only adds `bitsandbytes` for 4-bit/8-bit quantization support.

## Quick Start

```bash
# 1. Copy and configure environment variables
cp .env.example .env

# 2. Parse a PDF to Markdown (using Gemini)
aee-parse --file paper.pdf --config config/systems/example.yaml

# 3. Run extraction with a local LLM
aee-extract --config config/systems/example.yaml --task nanozymes

# 4. Optimize the agent
aee-optimize --config config/systems/example.yaml --task nanozymes
```

## Project Structure

```
├── src/aee/                  # Package source
│   ├── domain/               # Entities, tasks, evaluation
│   ├── application/          # Use cases, services
│   ├── infrastructure/       # LLM providers, parsers, storage
│   ├── interface/            # CLI
│   └── shared/               # Shared utilities
├── config/                   # YAML configurations
│   ├── systems/              # System configs (models, providers)
│   └── tasks/                # Task definitions
├── scripts/                  # Utility scripts
│   ├── benchmark_inference.py  # Performance benchmark (tokens/sec, kernel check)
│   ├── diagnose_performance.py # Deep diagnostic suite (prefill/decode analysis)
│   ├── parse.py                # CLI wrapper
│   ├── extract.py              # CLI wrapper
│   └── optimize.py             # CLI wrapper
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
├── pyproject.toml            # Package definition and dependencies
├── environment.yml           # Conda environment (local dev)
└── constraints.txt           # NCCL conflict resolution (conda only)
```
