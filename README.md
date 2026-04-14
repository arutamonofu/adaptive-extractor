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