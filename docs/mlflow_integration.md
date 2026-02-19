# MLflow Integration for DSPy

AutoEvoExtractor integrates MLflow with native DSPy support for comprehensive experiment tracking.

---

## Overview

The project uses MLflow's specialized DSPy integration (`mlflow.dspy`) to automatically track:

- DSPy program calls and predictions
- Prompt template evolution during optimization
- Optimization trials and metrics
- Trained agent models with proper serialization

This is more powerful than generic MLflow logging because it understands DSPy's internal structure.

---

## Features

### Automatic Logging (DSPy Autolog)

By default, `mlflow.dspy.autolog()` automatically captures:

- Every DSPy program invocation
- Prompt templates and their changes
- Input/output examples during optimization
- MIPROv2 trial metrics

### Model Logging

`mlflow.dspy.log_model()` properly serializes DSPy models, including:

- Optimized prompts and demonstrations
- Internal DSPy state
- Signature definitions

---

## Configuration

### Enable/Disable Tracking

```bash
# Disable MLflow tracking
python scripts/optimize.py --no-mlflow

# Enable (default)
python scripts/optimize.py
```

**Note:** DSPy autologging is enabled by default when creating `ExperimentTracker`. Disabling is only possible through code (parameter `enable_dspy_autolog=False` in the constructor).

### Run Naming

For sequential experiments (e.g., A1 → A2 → A3), use the `--run-name` argument to set a meaningful prefix:

```bash
# Experiment A1: Prompt quality comparison
python scripts/optimize.py --config default.yaml --run-name "A1_high"
python scripts/optimize.py --config default.yaml --run-name "A1_low"

# Experiment A2: Teacher temperature comparison
python scripts/optimize.py --config default.yaml --run-name "A2_temp1.0"
python scripts/optimize.py --config default.yaml --run-name "A2_temp0.5"

# Experiment A3: Parameter set comparison
python scripts/optimize.py --config default.yaml --run-name "A3_full"
python scripts/optimize.py --config default.yaml --run-name "A3_limited"
```

The system automatically appends a timestamp to ensure unique run names (e.g., `A1_high_20260217_143022`).

All runs are logged to a single MLflow experiment: `{task_name}/optimization` (e.g., `nanozymes/optimization`).

### Tracking URI

Set via environment variable or configuration:

```bash
# Local SQLite database (default)
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Remote MLflow server
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
```

This setting can also be specified in YAML configuration via `mlflow_tracking_uri`.

---

## Usage

### Viewing Experiments

Launch the MLflow UI:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` to view:

- All optimization runs
- Metrics over time
- Saved models and artifacts
- Configuration parameters

### Programmatic Access

For programmatic use via Python API:

```python
from aee.application.services import ExperimentTracker

# Create tracker (DSPy autologging enabled by default)
tracker = ExperimentTracker(
    experiment_name="nanozyme_optimization",
    tracking_uri="sqlite:///mlflow.db",
)

# Start experiment
with tracker.start_run(run_name="trial_1"):
    # DSPy operations are automatically logged
    
    # Log metrics
    tracker.log_metrics({"f1": 0.85, "precision": 0.82})
    
    # Log artifacts
    tracker.log_artifact(Path("agent.json"))
```

To disable DSPy autologging (manual logging only):

```python
tracker = ExperimentTracker(
    experiment_name="test",
    enable_dspy_autolog=False,
)
```

---

## Architecture

### ExperimentTracker

The service is located at `src/aee/application/services/experiment_tracker.py`.

**Key Methods:**

- `start_run(run_name)` — start a new experiment
- `log_params(params)` — log parameters
- `log_metrics(metrics)` — log metrics
- `log_artifact(path)` — log files
- `log_dspy_model(model)` — log DSPy model with proper serialization
- `log_optimization_results(...)` — convenient logging of optimization results

### Integration in Optimization Workflow

`OptimizeAgentUseCase` automatically:

1. Enables DSPy autologging when the run starts
2. Logs configuration parameters
3. Tracks all MIPROv2 operations
4. Saves the final model via `log_dspy_model()`
5. Logs metrics and artifacts

---

## Requirements

MLflow with DSPy support requires:

```
mlflow>=2.10.0
dspy-ai>=2.5.0
```

When using an older MLflow version, the system automatically falls back to regular artifact logging.

---

## Troubleshooting

### "DSPy autologging not available"

**Cause:** MLflow version < 2.10.0

**Solution:** Upgrade MLflow:
```bash
pip install --upgrade mlflow
```

### "Failed to log DSPy model"

**Cause:** Model serialization issue (e.g., non-serializable objects like thread locks)

**Solution:** The system automatically falls back to logging as a JSON artifact if DSPy logging fails.

### Experiments not appearing in UI

**Check tracking URI:**
```bash
echo $MLFLOW_TRACKING_URI
```

**Verify database location:**
```bash
ls -lh mlflow.db
```

---

## Best Practices

1. **Use CLI for optimization** — all tracking settings are enabled by default
2. **Use meaningful run names** — helps identify experiments later
3. **Compare runs in UI** — use MLflow UI to compare metrics across trials
4. **Configure tracking URI for production** — use a remote server for team collaboration

---

## Example: Complete Optimization via CLI

```bash
# Basic optimization (MLflow enabled by default)
python scripts/optimize.py --config default.yaml

# With named run
python scripts/optimize.py --config default.yaml --run-name "A1_temp0.5"

# Without MLflow tracking
python scripts/optimize.py --config default.yaml --no-mlflow
```

All optimization parameters are automatically logged to MLflow, including:
- MIPROv2 configuration (num_trials, seed, num_candidates, etc.)
- LLM parameters (model, temperature)
- Final metrics (F1, precision, recall)
- Optimized agent (as both DSPy model and JSON artifact)

---

## Related Documentation

- [CLI Reference](cli_reference.md) — complete command reference
- [Configuration Guide](configuration.md) — YAML settings and environment variables
- [Architecture](architecture.md) — system design for developers
- [README](README.md) — quick start guide

---

## References

- [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
