# Adding New Extraction Tasks

This guide walks you through adding a new extraction task to AutoEvoExtractor.

## Overview

AutoEvoExtractor supports **two approaches** for adding tasks:

### Approach 1: YAML-Based Configuration (Recommended)
Define your task declaratively in a YAML file. The system automatically generates:
- Pydantic models for experiments
- DSPy signatures for LLM extraction
- Row converters for CSV loading

**Time Estimate**: 15-30 minutes for a simple task

### Approach 2: Python-Based Plugin (Advanced)
Create a custom task plugin in Python for complex scenarios requiring:
- Custom validation logic
- Non-standard field types
- Specialized row converters
- Domain-specific matching logic

**Time Estimate**: 2-4 hours for a complete task

## Which Approach to Choose?

**Use YAML (Approach 1) if:**
- Your fields are standard types (str, int, float, bool)
- You need basic validation (required/optional, choices, min/max)
- You want to get started quickly

**Use Python (Approach 2) if:**
- You need custom type coercion (e.g., parsing complex strings)
- You have domain-specific validation rules
- You need to override default behavior

## Prerequisites

### For YAML Approach
- Basic understanding of YAML syntax
- Familiarity with the domain you're extracting
- Sample ground truth data (10-20 examples to start)

### For Python Approach
- All of the above, plus:
- Basic understanding of Python and Pydantic
- Understanding of the AutoEvoExtractor architecture (see `docs/architecture.md`)

## Step-by-Step Guide (YAML Approach)

### Step 1: Create Task Directory and YAML Configuration

Create a new directory for your task:

```bash
mkdir -p src/aee/domain/tasks/{task_name}
```

Create `src/aee/domain/tasks/{task_name}/task.yaml`:

```yaml
# Task configuration for {task_name} extraction
# Example: src/aee/domain/tasks/proteins/task.yaml

name: proteins
description: >
  Extract protein structure experiments from scientific papers,
  including protein names, structure methods, resolution, and source organisms.

version: 1.0.0
tags:
  - biology
  - proteins
  - structural_biology

# Instruction for DSPy signature (loaded at optimization time)
# Option 1: Path to instruction file (recommended)
instruction_file: config/initial_instructions/proteins_v1.txt

# Option 2: Inline instruction (alternative)
# initial_instruction: |
#   You are a helpful assistant specializing in protein structures...

# Fields to extract from documents
fields:
  protein_name:
    type: str
    description: "Name of the protein (e.g., 'Cytochrome C', 'Hemoglobin')"
    required: true
    alt_names:
      - protein
      - name
      - protein_id

  structure_method:
    type: str
    description: "Method used: X-ray crystallography, NMR, cryo-EM"
    required: true
    alt_names:
      - method
      - technique
      - experimental_method

  resolution:
    type: float
    description: "Resolution in Angstroms"
    required: false
    min_value: 0
    alt_names:
      - resolution_angstrom
      - angstrom

  pdb_id:
    type: str
    description: "PDB database ID (e.g., '1A2B')"
    required: false
    pattern: "^[0-9][A-Za-z0-9]{3}$"  # Optional regex validation

  organism:
    type: str
    description: "Source organism"
    required: false
    alt_names:
      - species
      - source_organism

  temperature:
    type: float
    description: "Experiment temperature in Kelvin"
    required: false
    min_value: 0
    max_value: 1000

# Fields to compare during evaluation (must exist in fields)
compare_fields:
  - protein_name
  - structure_method
  - resolution

# Tolerance for float comparisons (0.05 = 5%)
float_tolerance: 0.05

# Row converter configuration - maps CSV columns to fields
row_converter:
  protein_name:
    - protein_name
    - protein
    - name
  structure_method:
    - structure_method
    - method
    - technique
  resolution:
    - resolution
    - resolution_angstrom
  pdb_id:
    - pdb_id
    - pdb
  organism:
    - organism
    - species
  temperature:
    - temperature
    - temp
    - temperature_k

# Model names (optional, defaults will be used if not specified)
experiment_model_name: ProteinExperiment
output_model_name: ProteinExtractionOutput

# Additional metadata
metadata:
  author: Your Name
  created: 2026-02-19
  license: MIT
```

**Field Specification Reference**:

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| `type` | str | Python type: `str`, `int`, `float`, `bool` | Yes |
| `description` | str | Human-readable description for LLM | Yes |
| `required` | bool | Whether field is required | No (default: true) |
| `default` | any | Default value for optional fields | No |
| `choices` | list[str] | Valid choices (creates Literal type) | No |
| `min_value` | float | Minimum for numeric fields | No |
| `max_value` | float | Maximum for numeric fields | No |
| `pattern` | str | Regex pattern for string fields | No |
| `alt_names` | list[str] | Alternative column names in CSV | No |

**Design Tips**:
- Use clear, descriptive field names (snake_case)
- Add detailed descriptions (helps the LLM understand what to extract)
- Mark fields as `required: false` if they might be missing
- Use `alt_names` for flexibility in CSV column naming
- Use `choices` for categorical fields (creates Literal types)
- Consider units in descriptions (Angstroms, Kelvin, etc.)

### Step 2: Create Initial Instruction

Create `config/initial_instructions/proteins_v1.txt`:

```txt
You are a helpful assistant specializing in protein structure analysis. Your task is to analyze scientific articles and extract detailed information about protein structure experiments.

For each experiment mentioned in the text, extract:
- protein_name (required): The name of the protein being studied (e.g., "Cytochrome C", "Hemoglobin", "Green Fluorescent Protein")
- structure_method (required): The experimental method used: X-ray crystallography, NMR spectroscopy, or cryo-EM
- resolution (optional): The resolution in Angstroms if mentioned
- pdb_id (optional): PDB database identifier if available (format: 4 characters like "1A2B")
- organism (optional): The source organism or species
- temperature (optional): The experiment temperature in Kelvin

IMPORTANT GUIDELINES:
1. Extract each experiment separately. If the same protein is studied with different methods or conditions, create separate entries.
2. Be precise with numerical values and units.
3. Use null/None for missing information - do not invent values.
4. Pay attention to the context - only extract actual experimental data, not hypothetical or proposed experiments.
```

**Instruction Tips**:
- Be explicit about required vs. optional fields
- Provide concrete examples of expected values
- Include guidance on handling missing data
- Mention edge cases (same protein, different conditions)
- Keep it concise - DSPy will add examples during optimization
- Store as plain `.txt` files (not YAML) to avoid escaping issues

**Note on Prompts vs. Instructions**:
- **Instruction**: Base guidance you provide in the `.txt` file
- **Prompt**: Instruction + examples (DSPy MIPROv2 generates during optimization)
- The system optimizes both to create effective prompts for your task

### Step 3: Prepare Ground Truth Data

Create `data/ground_truth/proteins.csv`:

```csv
filename,protein_name,structure_method,resolution,pdb_id,organism,temperature
paper1,Cytochrome C,X-ray crystallography,1.9,1A2B,Saccharomyces cerevisiae,100
paper1,Hemoglobin,X-ray crystallography,2.1,1A3N,Homo sapiens,100
paper2,Green Fluorescent Protein,X-ray crystallography,1.6,1GFL,Aequorea victoria,298
paper3,Lysozyme,NMR,,2LZM,Gallus gallus,298
paper4,Insulin,cryo-EM,3.2,,Bos taurus,77
```

**Ground Truth Tips**:
- Use `filename` column (document ID, with or without `.pdf` extension - both are supported)
- Include both required and optional fields
- Leave optional fields empty (not "N/A" or "None")
- Include variety of cases (different methods, organisms, etc.)
- Start with 10-20 examples, expand to 50+ for best results
- Ensure accuracy - ground truth quality affects training!
- Column names can match `alt_names` from YAML (flexible mapping)

### Step 4: Create Data Split

```bash
python -c "
from pathlib import Path
from aee.infrastructure.storage import DataSplitRepository
import pandas as pd

# Get document IDs from ground truth
df = pd.read_csv('data/ground_truth/proteins.csv')
doc_ids = df['filename'].unique().tolist()

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

### Step 5: Test Your Task

Test that the task loads and validates correctly:

```python
from pathlib import Path
from aee.domain.tasks import load_task_from_yaml

# Load task from YAML
yaml_path = Path("src/aee/domain/tasks/proteins/task.yaml")
task = load_task_from_yaml(yaml_path)

# Validate task
try:
    task.validate()
    print(f"✓ Task '{task.name}' is valid")
    print(f"  Description: {task.description}")
    print(f"  Fields: {len(task.experiment_fields)}")
    print(f"  Compare fields: {task.compare_fields}")
except Exception as e:
    print(f"✗ Validation failed: {e}")

# Test loading instruction
try:
    instruction = task.config.get_instruction()
    print(f"✓ Instruction loaded: {len(instruction)} chars")
    print(f"  Hash: {task.config.get_instruction_hash()}")
except FileNotFoundError as e:
    print(f"✗ Instruction file not found: {e}")
```

### Step 6: Optimize Agent for Your Task

```bash
# Create environment config (optional)
cp config/default.yaml config/proteins.yaml
# Edit config/proteins.yaml to adjust parameters if needed

# Run optimization
python scripts/optimize.py \
    --task proteins \
    --config proteins.yaml
```

This will:
1. Load your ground truth data
2. Load the initial instruction from `config/initial_instructions/`
3. Train an agent using MIPROv2 optimization
4. Evaluate on validation set
5. Save the optimized agent with metadata (including instruction hash)

### Step 7: Run Extractions

```bash
# Find the latest agent
ls -lt data/agents/proteins_*.json | head -1

# Run batch extraction
python scripts/extract.py \
    --config proteins.yaml \
    --agent proteins_v1_2024-01-15.json \
    --task proteins
```

### Step 8: Evaluate Results

Evaluation is performed automatically during optimization. For manual evaluation:

```python
from aee.domain.tasks import get_task
from aee.domain.evaluation.metrics import TaskMetric

task = get_task("proteins")
metric = TaskMetric(
    task_name=task.name,
    compare_fields=task.compare_fields,
    float_tolerance=task.float_tolerance,
)

# Calculate metrics on your extractions
# ... (see docs/evaluation.md for details)
```

## Python Approach (Advanced)

For complex tasks requiring custom logic, use the Python plugin approach.

### When to Use Python

- Custom type coercion (e.g., parsing "10^-5 M" to float)
- Domain-specific validation (e.g., PDB ID format)
- Complex row conversion logic
- Custom experiment matching during evaluation

### Step-by-Step (Python)

See the existing nanozymes task for a complete example:
- `src/aee/domain/tasks/nanozymes/models.py` - Pydantic models
- `src/aee/domain/tasks/nanozymes/signature.py` - DSPy signature
- `src/aee/domain/tasks/nanozymes/converters.py` - Row converter
- `src/aee/domain/tasks/nanozymes/__init__.py` - Task definition
- `src/aee/domain/tasks/nanozymes/task.yaml` - YAML config (still used!)

**Key Difference**: Even with Python approach, you still use YAML for task configuration. The Python code provides custom models and converters that override the dynamically generated ones.

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

Then use in your task (Python approach only):

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

### Validation Rules (Python Approach)

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

### Instruction Versioning

Track instruction changes for reproducibility:

```bash
# Create new instruction version
cp config/initial_instructions/proteins_v1.txt \
   config/initial_instructions/proteins_v2.txt

# Edit v2 with improvements
# Update task.yaml to point to new instruction
# Re-run optimization
python scripts/optimize.py --task proteins --config proteins.yaml

# Agent metadata will include instruction hash for tracking
```

## Troubleshooting

### Task Not Found

**Error**: `TaskNotFoundError: Task 'proteins' not found`

**Solutions**:
1. Ensure YAML file exists: `src/aee/domain/tasks/proteins/task.yaml`
2. Check task name in YAML matches the `--task` argument
3. Verify YAML is valid: `python -c "from aee.domain.tasks import load_task_from_yaml; load_task_from_yaml('src/aee/domain/tasks/proteins/task.yaml')"`

### Instruction File Not Found

**Error**: `FileNotFoundError: Instruction file not found: config/initial_instructions/proteins_v1.txt`

**Solutions**:
1. Create the instruction file
2. Check path is relative to project root
3. Use absolute path in YAML if needed: `instruction_file: /full/path/to/config/initial_instructions/proteins_v1.txt`

### Validation Errors

**Error**: `TaskConfig validation failed` or `TaskValidationError`

**Common causes**:
- `compare_fields` reference fields not in `fields`
- Missing required properties (name, description)
- Both `initial_instruction` and `instruction_file` specified
- Invalid field types or constraints

**Solution**: Check error message for specific validation failures.

### Low Extraction Accuracy

**Causes**:
1. **Insufficient ground truth**: Add more examples (aim for 50+)
2. **Ambiguous instruction**: Make instructions more specific
3. **Poor compare_fields**: Choose fields that truly identify unique experiments
4. **Model limitations**: Try a more capable LLM
5. **LLM configuration**: Incorrect Ollama URL or API settings

**Solutions**:
- Expand ground truth data
- Refine instruction with clearer guidance
- Adjust compare_fields
- Increase `num_trials` in optimization
- Verify LLM connection

### Row Converter Issues

**Error**: Experiments return `None` or missing fields

**Solutions**:
1. Check CSV column names match `alt_names` in YAML
2. Verify required fields have values in CSV
3. Check type conversions (float, int) are valid
4. Enable debug logging: `export LOG_LEVEL=DEBUG`

### Import Errors (Python Approach)

**Error**: `ImportError: cannot import name 'ProteinTask'`

**Solutions**:
1. Check `__init__.py` exports
2. Verify files are in correct location
3. Ensure task module is imported somewhere (triggers registration)

## Best Practices

1. **Start with YAML**: Begin with YAML approach, switch to Python only if needed
2. **Iterate Quickly**: Test → Evaluate → Refine → Repeat
3. **Clear Instructions**: Spend time crafting good initial instructions
4. **Flexible Converters**: Use `alt_names` for CSV column flexibility
5. **Test Thoroughly**: Validate task before running optimization
6. **Version Control**: Commit after each working milestone
7. **Track Instructions**: Use instruction hashing for reproducibility
8. **Start Small**: 10-15 ground truth examples to validate approach
9. **Expand Gradually**: Add more examples as you refine the task
10. **Document**: Add comments in YAML for complex fields

## Example Tasks

For reference, see the existing tasks:
- **YAML-based**: `src/aee/domain/tasks/nanozymes/task.yaml`
- **Python-based**: `src/aee/domain/tasks/nanozymes/` (full implementation)

## Getting Help

- Architecture questions: See `docs/architecture.md`
- Configuration: See `docs/configuration.md`
- Issues: https://github.com/ai-chem/AutoEvoExtractor/issues
