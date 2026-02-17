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

When enabled, `mlflow.dspy.autolog()` automatically captures:

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

### Disable DSPy Autologging

By default, DSPy autologging is enabled for comprehensive tracking. To disable it:

```bash
# Disable DSPy autologging (still logs metrics and artifacts)
python scripts/optimize.py --task nanozymes --no-dspy-autolog
```

**When to disable autologging:**
- Debugging specific DSPy operations
- Reducing log verbosity for large experiments
- Using custom logging logic

### Run Naming

For sequential experiments (e.g., A1 → A2 → A3), use the `--run-name` argument to set a meaningful prefix:

```bash
# Experiment A1: Prompt quality comparison
python scripts/optimize.py --task nanozymes --run-name "A1_high"
python scripts/optimize.py --task nanozymes --run-name "A1_low"

# Experiment A2: Teacher temperature comparison
python scripts/optimize.py --task nanozymes --run-name "A2_temp1.0"
python scripts/optimize.py --task nanozymes --run-name "A2_temp0.5"

# Experiment A3: Parameter set comparison
python scripts/optimize.py --task nanozymes --run-name "A3_full"
python scripts/optimize.py --task nanozymes --run-name "A3_limited"
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

```python
from aee.application.services import ExperimentTracker

# Create tracker with DSPy autologging enabled
tracker = ExperimentTracker(
    experiment_name="nanozyme_optimization",
    enable_dspy_autolog=True,  # Enable automatic DSPy tracking
)

# Start a run
with tracker.start_run(run_name="trial_1"):
    # DSPy operations are automatically logged
    optimized_agent = optimize_my_agent()

    # Log metrics
    tracker.log_metrics({"f1": 0.85, "precision": 0.82})

    # Log DSPy model with proper serialization
    tracker.log_dspy_model(optimized_agent, name="agent")

    # Regular artifacts still work
    tracker.log_artifact(Path("agent.json"))
```

### Disabling DSPy Autolog

If you only want manual logging:

```python
tracker = ExperimentTracker(
    experiment_name="test",
    enable_dspy_autolog=False,  # Disable automatic tracking
)
```

---

## Implementation Details

### ExperimentTracker Service

Located at `src/aee/application/services/experiment_tracker.py`

**Key Methods:**

- `enable_dspy_autolog()` - Enable automatic DSPy tracking
- `disable_dspy_autolog()` - Disable automatic tracking
- `log_dspy_model()` - Log DSPy model with proper serialization
- `log_optimization_results()` - Convenience method for complete optimization logging

### Integration in Optimization Workflow

The `OptimizeAgentUseCase` automatically:

1. Enables DSPy autologging when tracker is provided
2. Logs configuration parameters
3. Tracks all MIPROv2 operations
4. Saves final model using `log_dspy_model()`
5. Logs metrics and artifacts

---

## Requirements

MLflow with DSPy support requires:

```
mlflow>=2.10.0
dspy-ai>=2.5.0
```

If using an older MLflow version, the system gracefully falls back to regular artifact logging.

---

## Troubleshooting

### "DSPy autologging not available"

**Cause:** MLflow version < 2.10.0

**Solution:** Upgrade MLflow:
```bash
pip install --upgrade mlflow
```

### "Failed to log DSPy model"

**Cause:** Model serialization issue

**Fallback:** The system automatically logs as regular artifact if DSPy logging fails.

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

1. **Enable autolog for optimization** - Let MLflow capture all DSPy operations automatically
2. **Use meaningful run names** - Helps identify experiments later
3. **Log both DSPy model and JSON artifact** - Provides compatibility and flexibility
4. **Set tags for filtering** - Use `tracker.set_tag()` for task name, model version, etc.
5. **Compare runs in UI** - Use MLflow UI to compare metrics across trials

---

## Example: Complete Optimization with Tracking

```python
from pathlib import Path
from aee.application.services import ExperimentTracker, AgentManager, DatasetBuilder
from aee.application.use_cases import OptimizeAgentRequest, OptimizeAgentUseCase
from aee.domain.tasks import get_task

# Setup tracker with DSPy autologging
tracker = ExperimentTracker(
    experiment_name="nanozyme_optimization",
    tracking_uri="sqlite:///mlflow.db",
    enable_dspy_autolog=True,
)

# Create use case
use_case = OptimizeAgentUseCase(
    dataset_builder=builder,
    agent_manager=manager,
    gt_repo=gt_repo,
    tracker=tracker,  # Pass tracker here
)

# Execute optimization
request = OptimizeAgentRequest(
    task=get_task("nanozymes"),
    gt_path=Path("data/ground_truth/nanozymes.csv"),
    splits_dir=Path("data/splits"),
    task_name="nanozymes",
    student_lm=student_lm,
    num_trials=20,
)

response = use_case.execute(request)

# All DSPy operations are automatically tracked in MLflow!
```

---

## Related Documentation

- [Configuration Guide](configuration.md) - General configuration options
- [Architecture](architecture.md) - System design and layers
- [Troubleshooting](troubleshooting.md) - Common issues

---

## References

- [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
