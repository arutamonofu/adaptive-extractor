# Adding New Extraction Tasks

This guide walks you through adding a new extraction task to AutoEvoExtractor. We'll use a hypothetical "Protein Structures" task as an example.

## Overview

Adding a new task requires:
1. **Define the domain model** (experiment structure)
2. **Create DSPy signature** (LLM extraction schema)
3. **Implement row converter** (CSV to experiment mapping)
4. **Create task plugin** (tie everything together)
5. **Prepare ground truth data** (CSV with examples)
6. **Test and optimize** (verify and improve)

**Time Estimate**: 2-4 hours for a complete task

## Prerequisites

- Basic understanding of Python and Pydantic
- Familiarity with the domain you're extracting
- Sample ground truth data (even just 5-10 examples)
- Understanding of the AutoEvoExtractor architecture (see `docs/architecture.md`)

## Step-by-Step Guide

### Step 1: Define Your Domain Model

Create a new directory for your task:

```bash
mkdir -p src/aee/domain/tasks/proteins
touch src/aee/domain/tasks/proteins/__init__.py
```

Create `src/aee/domain/tasks/proteins/models.py`:

```python
"""Protein structure experiment data models."""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field

from aee.domain.entities import Experiment


class ProteinExperiment(Experiment):
    """Represents a single protein structure experiment."""

    # Required fields
    protein_name: str = Field(
        ...,
        description="Name of the protein (e.g., 'Cytochrome C', 'Hemoglobin')"
    )
    structure_method: str = Field(
        ...,
        description="Method used: X-ray crystallography, NMR, cryo-EM"
    )

    # Optional fields
    resolution: Optional[float] = Field(
        None,
        description="Resolution in Angstroms"
    )
    pdb_id: Optional[str] = Field(
        None,
        description="PDB database ID (e.g., '1A2B')"
    )
    organism: Optional[str] = Field(
        None,
        description="Source organism"
    )
    temperature: Optional[float] = Field(
        None,
        description="Experiment temperature in Kelvin"
    )


class ProteinExtractionOutput(BaseModel):
    """Container for extracted protein experiments."""

    experiments: List[ProteinExperiment] = Field(default_factory=list)
```

**Design Tips**:
- Always add `from __future__ import annotations` at the top for Python 3.8+ compatibility
- Use clear, descriptive field names
- Add detailed descriptions (helps the LLM understand)
- Mark required fields with `...`, optional with `None`
- Include validation where appropriate
- Consider units (Angstroms, Kelvin, etc.)
- Use specific type hints (List[Experiment] not list)

### Step 2: Create DSPy Signature

Create `src/aee/domain/tasks/proteins/signature.py`:

```python
"""DSPy signature for protein structure extraction."""

from __future__ import annotations

from typing import Type

import dspy

from .models import ProteinExtractionOutput


def create_protein_signature(instruction: str) -> Type[dspy.Signature]:
    """Create a ProteinSignature class with the given instruction.

    Args:
        instruction: The initial instruction text loaded from a configuration file.

    Returns:
        A DSPy Signature class configured with the instruction.
    """
    if not instruction or not instruction.strip():
        raise ValueError("Initial instruction for ProteinSignature cannot be empty")

    class ProteinSignature(dspy.Signature):
        """Extract protein structure experiments from text."""

        __doc__ = instruction

        text: str = dspy.InputField(
            desc="Scientific text containing protein structure data"
        )
        data: ProteinExtractionOutput = dspy.OutputField(
            desc="Extract ALL protein structure experiments mentioned."
        )

    return ProteinSignature
```

**Signature Tips**:
- Use a factory function that accepts `instruction: str`
- The instruction is loaded from `config/initial_instructions/` at runtime
- Provide clear input/output field descriptions
- Validate that instruction is not empty
- This allows testing different instructions without code changes

### Step 3: Implement Row Converter

Create `src/aee/domain/tasks/proteins/converters.py`:

```python
"""Converters for protein structure data."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .models import ProteinExperiment

logger = logging.getLogger(__name__)


def row_to_protein(row: pd.Series) -> Optional[ProteinExperiment]:
    """Convert a pandas Series row to a ProteinExperiment.

    This function loads ground truth data from CSV files.

    Args:
        row: Pandas Series containing experiment data

    Returns:
        ProteinExperiment object or None if required fields are missing
    """
    # Extract required fields
    protein_name = row.get("protein_name") or row.get("protein") or row.get("name")
    if not protein_name:
        logger.debug("Missing required 'protein_name' field")
        return None

    structure_method = (
        row.get("structure_method")
        or row.get("method")
        or row.get("technique")
    )
    if not structure_method:
        logger.debug("Missing required 'structure_method' field")
        return None

    # Extract optional fields with type conversion
    resolution = None
    if "resolution" in row and pd.notna(row["resolution"]):
        try:
            resolution = float(row["resolution"])
        except (ValueError, TypeError):
            logger.debug(f"Could not convert resolution: {row['resolution']}")

    # Build experiment
    try:
        return ProteinExperiment(
            protein_name=str(protein_name),
            structure_method=str(structure_method),
            resolution=resolution,
            pdb_id=row.get("pdb_id"),
            organism=row.get("organism"),
            temperature=_safe_float(row.get("temperature")),
        )
    except Exception as e:
        logger.warning(f"Failed to create ProteinExperiment: {e}")
        return None


def _safe_float(value) -> Optional[float]:
    """Safely convert value to float."""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
```

**Converter Tips**:
- Handle multiple possible column names (flexibility)
- Validate required fields early
- Use safe type conversions
- Log warnings for debugging
- Return `None` for invalid rows
- Match CSV column names to model fields

### Step 4: Create Task Plugin

Create `src/aee/domain/tasks/proteins/__init__.py`:

```python
"""Protein structure extraction task plugin."""

from __future__ import annotations

from typing import Callable, List, Type

import dspy
from pydantic import BaseModel

from aee.domain.tasks.base import TaskDefinition
from aee.domain.tasks import register_task

from .models import ProteinExperiment, ProteinExtractionOutput
from .signature import create_protein_signature
from .converters import row_to_protein


class ProteinTask(TaskDefinition):
    """Task definition for protein structure extraction."""

    def __init__(self, initial_instruction: str):
        """Initialize the task with an instruction.

        Args:
            initial_instruction: The instruction text loaded from configuration.
        """
        if not initial_instruction or not initial_instruction.strip():
            raise ValueError("ProteinTask requires non-empty initial_instruction")

        self._initial_instruction = initial_instruction
        self._signature_class = create_protein_signature(initial_instruction)

    @property
    def name(self) -> str:
        return "proteins"

    @property
    def description(self) -> str:
        return "Extract protein structure experiments from scientific papers"

    @property
    def signature(self) -> Type[dspy.Signature]:
        return self._signature_class

    @property
    def output_model(self) -> Type[BaseModel]:
        return ProteinExtractionOutput

    @property
    def experiment_model(self) -> Type[BaseModel]:
        return ProteinExperiment

    @property
    def row_converter(self) -> Callable:
        return row_to_protein

    @property
    def compare_fields(self) -> List[str]:
        """Fields to use for matching predicted vs. ground truth experiments."""
        return ["protein_name", "structure_method", "resolution"]

    @property
    def initial_instruction(self) -> str:
        """The initial instruction used for this task."""
        return self._initial_instruction

    @property
    def instruction_metadata(self) -> dict:
        """Metadata about the initial instruction for reproducibility tracking.
        
        Returns:
            Dictionary containing instruction hash and length.
        """
        import hashlib
        instruction_hash = hashlib.sha256(self._initial_instruction.encode()).hexdigest()[:12]
        return {
            "instruction_hash": instruction_hash,
            "instruction_length": len(self._initial_instruction),
        }


# Note: Task registration happens after loading instruction from config.
# See scripts/optimize.py for the pattern.
__all__ = [
    "ProteinTask",
    "ProteinExperiment",
    "ProteinExtractionOutput",
    "create_protein_signature",
    "row_to_protein",
]
```

**Plugin Tips**:
- Choose a unique task name (lowercase, no spaces)
- Accept `initial_instruction` in constructor (loaded from config)
- Write a clear description
- Select compare_fields carefully (used for evaluation)
- Export key classes for testing
- The instruction is loaded by `scripts/optimize.py` from `config/initial_instructions/`
- Implement `instruction_metadata` property for reproducibility tracking (hash + length)

**Example: Loading Initial Instruction from Config**:

```python
from pathlib import Path
from aee.infrastructure.config.instruction_loader import InstructionLoader
from aee.domain.tasks import register_task
from aee.domain.tasks.proteins import ProteinTask

# Load instruction from config file
config_dir = Path("config")
loader = InstructionLoader(config_dir=config_dir)
instruction = loader.load("initial_instructions/proteins_v1.txt")

# Create and register task with loaded instruction
task = ProteinTask(initial_instruction=instruction)
register_task(task)

# Verify registration
from aee.domain.tasks import get_task
task = get_task("proteins")
print(f"Task registered: {task.name}")
print(f"Instruction hash: {task.instruction_metadata['instruction_hash']}")
```

### Step 5: Prepare Ground Truth Data

Create `data/ground_truth/proteins.csv`:

```csv
filename,protein_name,structure_method,resolution,pdb_id,organism,temperature
paper1.pdf,Cytochrome C,X-ray crystallography,1.9,1A2B,Saccharomyces cerevisiae,100
paper1.pdf,Hemoglobin,X-ray crystallography,2.1,1A3N,Homo sapiens,100
paper2.pdf,Green Fluorescent Protein,X-ray crystallography,1.6,1GFL,Aequorea victoria,298
paper3.pdf,Lysozyme,NMR,,2LZM,Gallus gallus,298
paper4.pdf,Insulin,cryo-EM,3.2,,Bos taurus,77
```

**Ground Truth Tips**:
- Use `filename` column (without .pdf for IDs)
- Include both required and optional fields
- Leave optional fields empty (not "N/A" or "None")
- Include variety of cases (different methods, organisms, etc.)
- Start with 10-20 examples, expand to 50+ for best results
- Ensure accuracy - ground truth quality affects training!

### Step 6: Create Data Split

```bash
python -c "
from pathlib import Path
from aee.infrastructure.storage import DataSplitRepository

# Get document IDs from ground truth
import pandas as pd
df = pd.read_csv('data/ground_truth/proteins.csv')
doc_ids = df['filename'].str.replace('.pdf', '').unique().tolist()

# Create 80/20 split
repo = DataSplitRepository()
split = repo.create_random_split(
    documents=doc_ids,
    train_ratio=0.8,
    seed=42
)

# Save split
repo.save_splits(split, Path('data/splits/proteins.json'))
print(f'Created split: {len(split[\"train\"])} train, {len(split[\"test\"])} test')
"
```

### Step 7: Test Your Task

Create `tests/unit/domain/test_tasks_protein.py`:

```python
"""Unit tests for ProteinTask plugin."""

import pytest
import pandas as pd

from aee.domain.tasks import get_task
from aee.domain.tasks.proteins import (
    ProteinTask,
    ProteinExperiment,
    row_to_protein,
)


@pytest.mark.unit
class TestProteinTask:
    """Tests for ProteinTask plugin."""

    @pytest.fixture
    def protein_task(self):
        """Provide a protein task instance."""
        return ProteinTask()

    def test_task_registered(self):
        """Test task is auto-registered."""
        task = get_task("proteins")
        assert isinstance(task, ProteinTask)

    def test_task_name(self, protein_task):
        """Test task name is correct."""
        assert protein_task.name == "proteins"

    def test_task_validation(self, protein_task):
        """Test task validation passes."""
        protein_task.validate()  # Should not raise

    def test_row_converter(self):
        """Test row converter produces valid experiments."""
        row = pd.Series({
            "protein_name": "Cytochrome C",
            "structure_method": "X-ray crystallography",
            "resolution": 1.9,
            "pdb_id": "1A2B",
        })

        experiment = row_to_protein(row)

        assert experiment is not None
        assert experiment.protein_name == "Cytochrome C"
        assert experiment.structure_method == "X-ray crystallography"
        assert experiment.resolution == 1.9

    def test_row_converter_missing_required(self):
        """Test row converter handles missing required fields."""
        row = pd.Series({
            "resolution": 1.9,  # Missing protein_name and method
        })

        experiment = row_to_protein(row)

        assert experiment is None  # Should return None

    def test_experiment_model(self):
        """Test ProteinExperiment model validation."""
        exp = ProteinExperiment(
            protein_name="Hemoglobin",
            structure_method="NMR",
            resolution=2.1,
        )

        assert exp.protein_name == "Hemoglobin"
        assert exp.structure_method == "NMR"
```

Run tests:
```bash
pytest tests/unit/domain/test_tasks_protein.py -v
```

### Step 8: Create Initial Instruction

Create `config/initial_instructions/proteins_v1.txt`:

```txt
You are a helpful assistant specializing in [domain]. Your task is to analyze scientific articles and extract detailed information about [entity] experiments.

For each experiment mentioned in the text, extract:
- [Field 1] (required): Description of what to extract
- [Field 2] (required): Description
- [Field 3] (optional): Description

IMPORTANT: Extract each experiment separately. If the same [entity] is studied with different parameters, create separate entries.

Be precise with numerical values and units. Use null for missing information.
```

**Instruction Tips**:
- Store instructions as plain `.txt` files (not YAML) to avoid escaping issues
- Be explicit about required vs. optional fields
- Provide guidance on handling missing data and edge cases
- Keep it concise - DSPy will add examples during optimization
- You can create multiple instruction variants for A/B testing

Update your config (`config/proteins.yaml`):

```yaml
task:
  name: "proteins"
  initial_instruction_file: "initial_instructions/proteins_v1.txt"
  evaluation:
    float_tolerance: 0.05
    compare_fields:
      - "protein_name"
      - "structure_method"
      - "resolution"
```

**Note on Prompts vs. Instructions**:
- **Instruction**: Base guidance you provide (this file)
- **Prompt**: Instruction + examples (what DSPy MIPROv2 generates during optimization)
- The system optimizes both to create effective prompts for your task

### Step 9: Optimize Agent for Your Task

```bash
# Create config for your task (optional)
cp config/default.yaml config/proteins.yaml
# Edit config/proteins.yaml to adjust parameters

# Run optimization
python scripts/optimize.py \
    --task proteins \
    --config config/proteins.yaml \
    --trials 20
```

This will:
1. Load your ground truth data
2. Train an agent using MIPROv2 optimization
3. Evaluate on validation set
4. Save the optimized agent with metadata

### Step 10: Run Predictions

```bash
# Find the latest agent
ls -lt data/agents/proteins_*.json | head -1

# Run batch prediction
python scripts/predict.py \
    --agent data/agents/proteins_v1_2024-01-15.json \
    --task proteins \
    --output data/predictions/proteins_predictions.json
```

### Step 11: Evaluate Results

```bash
python scripts/evaluate.py \
    --task proteins \
    --predictions data/predictions/proteins_predictions.json \
    --ground-truth data/ground_truth/proteins.csv
```

This will show:
- Precision, Recall, F1 scores
- Per-field accuracy
- Confusion matrix
- Missed experiments

## Advanced Topics

### Custom Matching Logic

If default field matching isn't sufficient, create custom matcher:

```python
# src/aee/domain/tasks/proteins/matching.py

from aee.domain.evaluation.matcher import ExperimentMatcher


class ProteinMatcher(ExperimentMatcher):
    """Custom matcher for protein experiments."""

    def matches(self, pred, gt) -> bool:
        """Custom matching logic."""
        # Exact protein name match
        if pred.protein_name.lower() != gt.protein_name.lower():
            return False

        # Fuzzy resolution match (within 0.5 Å)
        if pred.resolution and gt.resolution:
            if abs(pred.resolution - gt.resolution) > 0.5:
                return False

        return True
```

Then use in your task:

```python
class ProteinTask(TaskDefinition):
    # ... other methods ...

    def create_matcher(self) -> ExperimentMatcher:
        """Override to use custom matcher."""
        return ProteinMatcher(compare_fields=self.compare_fields)
```

### Multi-Table Extraction

If your task requires extracting structured tables:

```python
class ProteinExperiment(Experiment):
    # ... other fields ...

    kinetic_data: Optional[List[KineticMeasurement]] = Field(
        None,
        description="Time-series kinetic measurements"
    )


class KineticMeasurement(BaseModel):
    time: float  # seconds
    signal: float  # intensity
```

### Validation Rules

Add domain-specific validation:

```python
from pydantic import field_validator


class ProteinExperiment(Experiment):
    # ... fields ...

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        """Ensure resolution is positive and reasonable."""
        if v is not None and (v <= 0 or v > 10):
            raise ValueError("Resolution must be between 0 and 10 Angstroms")
        return v

    @field_validator("pdb_id")
    @classmethod
    def validate_pdb_id(cls, v):
        """Ensure PDB ID format is correct."""
        if v and not (len(v) == 4 and v[0].isdigit()):
            raise ValueError("PDB ID must be 4 characters (e.g., '1A2B')")
        return v
```

## Troubleshooting

### Task Not Found

**Error**: `TaskNotFoundError: Task 'proteins' not found`

**Solution**: Ensure your task is imported somewhere:
```python
# In src/aee/__init__.py or a script
from aee.domain.tasks.proteins import ProteinTask  # This triggers registration
```

### Low Extraction Accuracy

**Causes**:
1. **Insufficient ground truth**: Add more examples (aim for 50+)
2. **Ambiguous signature**: Make instructions more specific
3. **Poor compare_fields**: Choose fields that truly identify unique experiments
4. **Model limitations**: Try a more capable LLM (e.g., Claude vs. smaller models)
5. **LLM configuration**: Incorrect Ollama URL or API settings

**Solutions**:
- Expand ground truth data
- Refine DSPy signature with clearer instructions
- Adjust compare_fields or create custom matcher
- Increase `num_trials` in optimization
- Verify LLM connection with `export OLLAMA_BASE_URL` if using custom Ollama server

### Validation Errors

**Error**: `ValidationError: field required`

**Solution**: Check your CSV has all required field columns, or make fields optional in your model.

### Import Errors

**Error**: `ImportError: cannot import name 'ProteinTask'`

**Solution**: Check `__init__.py` exports and that files are in the right location.

## Best Practices

1. **Start Small**: Begin with 10-15 ground truth examples to validate your approach
2. **Iterate**: Test → Evaluate → Refine → Repeat
3. **Clear Instructions**: Spend time crafting good DSPy signature descriptions
4. **Flexible Converters**: Handle variations in CSV column names
5. **Test Thoroughly**: Write unit tests for your models and converters
6. **Document**: Add docstrings explaining domain-specific concepts
7. **Version Control**: Commit after each working milestone

## Example Tasks

For reference, see the existing nanozymes task:
- `src/aee/domain/tasks/nanozymes/`

This provides a complete, production-ready example.

## Getting Help

- Architecture questions: See `docs/architecture.md`
- Issues: https://github.com/anthropics/autoevoextractor/issues
