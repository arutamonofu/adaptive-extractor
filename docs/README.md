# AutoEvoExtractor Documentation

Complete documentation for AutoEvoExtractor - a scientific data extraction system using LLMs with automatic prompt optimization.

---

## Getting Started

New to AutoEvoExtractor? Start here:

1. **[Main README](../README.md)** - Installation and quick start
2. **[Configuration](configuration.md)** - Set up your environment
3. **[Troubleshooting](troubleshooting.md)** - Common issues

---

## Documentation Index

### For Users

| Document | Description |
|----------|-------------|
| [Configuration](configuration.md) | Complete YAML and environment variable reference |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |

### For Developers

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, layers, and patterns |
| [Adding Tasks](adding_tasks.md) | Step-by-step guide for new extraction tasks |
| [API Usage](api_usage.md) | Using AutoEvoExtractor as a Python library |

### For Researchers

| Document | Description |
|----------|-------------|
| [MLflow Integration](mlflow_integration.md) | Experiment tracking with DSPy autologging |

---

## Quick Reference

### Common Workflows

**Parse → Optimize → Predict:**
```bash
python scripts/parse.py --config default.yaml
python scripts/optimize.py --config default.yaml
python scripts/extract.py --agent nanozymes_latest.json --config default.yaml
```

**Add a new task:**
1. Define models → `domain/tasks/mytask/models.py`
2. Create signature → `domain/tasks/mytask/signature.py`
3. Write converter → `domain/tasks/mytask/converters.py`
4. Register task → `domain/tasks/mytask/__init__.py`

See [Adding Tasks Guide](adding_tasks.md) for details.

### Environment Variables

```bash
# LLM (Ollama)
export OLLAMA_STUDENT_BASE_URL="http://localhost:11434"
export OLLAMA_TEACHER_BASE_URL="http://localhost:11434"
export LLM__STUDENT__MODEL="llama3.2:3b"

# LLM (API provider)
export LLM__STUDENT__USE_OLLAMA=false
export LLM__STUDENT__MODEL="gpt-4"
export OPENAI_API_KEY="sk-..."

# Paths
export PATHS__SPLITS_FILE="data/splits/mytask.json"

# Optimization
export OPTIMIZATION__NUM_TRIALS="50"
export OPTIMIZATION__USE_CACHE="true"

# Cache
export CACHE__DISK_SIZE_LIMIT_BYTES="30000000000"

# Circuit Breaker
export CIRCUIT_BREAKER__FAILURE_THRESHOLD="5"
```

---

## External Resources

- **[DSPy Framework](https://dspy-docs.vercel.app/)** - Prompt optimization
- **[Docling](https://github.com/DS4SD/docling)** - PDF parsing
- **[Ollama](https://ollama.com/)** - Local LLM inference
- **[MLflow](https://mlflow.org/docs/latest/llms/dspy/index.html)** - DSPy tracking
