# MLflow Integration

MLflow integration for DSPy experiment tracking.

## Overview

Adaptive Extractor uses `mlflow.dspy` for automatic tracking:

- DSPy program calls and predictions
- Prompt template evolution
- Optimization trials and metrics
- Trained agent models

---

## Features

### Automatic Logging

`mlflow.dspy.autolog()` captures:
- Every DSPy program invocation
- Prompt templates and changes
- Input/output examples
- MIPROv2 trial metrics

### Model Logging

`mlflow.dspy.log_model()` serializes:
- Optimized prompts and demonstrations
- Internal DSPy state
- Signature definitions

---

## Configuration

### Enable/Disable

```bash
# Disable MLflow tracking
ae-optimize --no-mlflow
# or: python scripts/optimize.py --no-mlflow

# Enable (default)
ae-optimize
# or: python scripts/optimize.py
```

### Run Naming

```bash
# Experiment A1: Prompt quality comparison
ae-optimize --run-name "A1_high"
ae-optimize --run-name "A1_low"

# Experiment A2: Teacher temperature comparison
ae-optimize --run-name "A2_temp1.0"
```

System appends timestamp automatically (e.g., `A1_high_20260217_143022`).

### Tracking URI

Set via environment:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

Or in YAML config:

```yaml
mlflow_tracking_uri: "sqlite:///mlflow.db"
```

---

## Viewing Results

```bash
# Start MLflow UI
mlflow ui

# Open browser
open http://localhost:5000
```

Navigate to experiment: `{task_name}/optimization` (e.g., `nanozymes/optimization`)

---

## Tracked Metrics

Per optimization trial:
- F1 score
- Precision
- Recall
- Trial number
- Configuration parameters

Per agent:
- Model version
- Instruction hash
- Training data size
- Final metrics

---

## DSPy Cache

DSPy responses are cached automatically:

```bash
# Cache location (default)
export DSPY_CACHE_DIR="${HOME}/.dspy_cache"

# View cache
ls ~/.dspy_cache/
```

Cache is reused across optimization runs for efficiency.
