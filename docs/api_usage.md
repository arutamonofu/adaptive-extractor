# API Usage Guide

How to use AutoEvoExtractor programmatically as a Python library.

## Overview

While AutoEvoExtractor provides CLI commands for common workflows, you can also use it as a library in your own Python scripts, notebooks, or applications. This guide shows how to use the core components directly.

## Table of Contents

- [Document Parsing](#document-parsing)
- [Task Registry](#task-registry)
- [Agent Management](#agent-management)
- [Dataset Building](#dataset-building)
- [Batch Prediction](#batch-prediction)
- [Evaluation](#evaluation)
- [Custom Workflows](#custom-workflows)

---

## Document Parsing

### Parse a Single PDF

```python
from pathlib import Path
from aee.infrastructure.parsers import DoclingParser

# Create parser
parser = DoclingParser(
    device="cpu",              # "cpu", "cuda", or "mps"
    num_threads=4,             # Number of CPU threads
    do_ocr=True,               # Enable OCR
    do_table_structure=True    # Extract table structure
)

# Parse PDF
pdf_path = Path("data/pdfs/paper.pdf")
document = parser.parse(pdf_path)

# Access parsed data
print(f"Document ID: {document.doc_id}")
print(f"Text length: {len(document.text)} characters")
print(f"Metadata: {document.metadata}")
```

### Parse Multiple PDFs

```python
from pathlib import Path
from aee.infrastructure.parsers import DoclingParser
from aee.infrastructure.storage import DocumentRepository

# Setup
parser = DoclingParser(device="cpu", num_threads=4)
doc_repo = DocumentRepository(parsed_dir=Path("data/parsed"))

# Parse directory
pdf_dir = Path("data/pdfs")
for pdf_path in pdf_dir.glob("*.pdf"):
    print(f"Parsing {pdf_path.name}...")

    # Parse
    document = parser.parse(pdf_path)

    # Save
    doc_repo.save(document)
    print(f"Saved to {document.doc_id}.json")
```

### Configure Parser

```python
from pathlib import Path
from aee.infrastructure.parsers import DoclingParser, MarkerParser

# Docling parser with custom settings
docling_parser = DoclingParser(
    device="cuda",              # Use GPU
    num_threads=8,              # More threads
    do_ocr=True,                # Enable OCR
    do_table_structure=True     # Extract tables
)

# Or use Marker parser
marker_parser = MarkerParser(
    device="cuda"               # "cpu" or "cuda"
)

document = docling_parser.parse(Path("data/pdfs/paper.pdf"))
```

---

## Task Registry

### List Available Tasks

```python
from aee.domain.tasks import get_global_registry

# Get all registered tasks
registry = get_global_registry()
tasks = registry.list_tasks()
for task in tasks:
    print(f"- {task.name}: {task.description}")
```

### Get a Specific Task

```python
from aee.domain.tasks import get_task

# Get task by name (task must be registered beforehand)
task = get_task("nanozymes")

print(f"Task: {task.name}")
print(f"Compare fields: {task.compare_fields}")
print(f"Description: {task.description}")
```

**Note:** Tasks are typically registered when their modules are imported. For example, `NanozymeTask` is registered when you import from `aee.domain.tasks.nanozymes`.

### Create Custom Task at Runtime

```python
from aee.domain.tasks.base import TaskDefinition
from aee.domain.tasks import register_task
import dspy
from pydantic import BaseModel
from typing import List

# Define models
class MyExperiment(BaseModel):
    field1: str
    field2: float

class MyOutput(BaseModel):
    experiments: List[MyExperiment]

# Define signature
class MySignature(dspy.Signature):
    """Extract my data from text."""
    text: str = dspy.InputField()
    data: MyOutput = dspy.OutputField()

# Define task
class MyTask(TaskDefinition):
    @property
    def name(self) -> str:
        return "mytask"

    @property
    def signature(self):
        return MySignature

    @property
    def output_model(self):
        return MyOutput

    @property
    def experiment_model(self):
        return MyExperiment

    @property
    def row_converter(self):
        def converter(row):
            return MyExperiment(
                field1=row.get("field1"),
                field2=float(row.get("field2", 0))
            )
        return converter

    @property
    def compare_fields(self) -> List[str]:
        return ["field1"]

# Register
register_task(MyTask())

# Now available
task = get_task("mytask")
```

---

## Agent Management

### Load Latest Agent

```python
from pathlib import Path
from aee.infrastructure.storage import AgentRepository

# Setup repository
agent_repo = AgentRepository(agents_dir=Path("data/agents"))

# Get latest agent for task
task_name = "nanozymes"
agent_path = agent_repo.get_latest(task_name)

if agent_path:
    agent, metadata = agent_repo.load(agent_path)
    print(f"Loaded agent from {agent_path}")
    print(f"F1 Score: {metadata.metrics.get('f1', 0):.3f}")
    print(f"Created: {metadata.created_at}")
    print(f"Task: {metadata.task_name}")
else:
    print(f"No agent found for task '{task_name}'")
```

### Load Specific Agent

```python
from pathlib import Path
from aee.infrastructure.storage import AgentRepository

agent_repo = AgentRepository(agents_dir=Path("data/agents"))

# Load specific agent by path
agent_path = Path("data/agents/nanozymes_sota_2026-02-17.json")
agent, metadata = agent_repo.load(agent_path)

print(f"Agent loaded: {metadata.task_name}")
print(f"Metrics: {metadata.metrics}")
```

### Save Agent with Metadata

```python
from pathlib import Path
from aee.infrastructure.storage import AgentRepository, AgentMetadata
from datetime import datetime

# Setup
agent_repo = AgentRepository(agents_dir=Path("data/agents"))

# Create metadata
metadata = AgentMetadata(
    task_name="nanozymes",
    created_at=datetime.now().isoformat(),
    model_version="1.0.0",
    metrics={"f1": 0.85, "precision": 0.87, "recall": 0.83},
    config_snapshot={
        "llm": {"model": "llama3.2:3b"},
        "optimization": {"num_trials": 20}
    },
    description="Optimized with 50 training examples"
)

# Save agent (agent must be a DSPy module)
path = agent_repo.save(
    agent=my_agent,
    task_name="nanozymes",
    metadata=metadata
)
print(f"Agent saved to {path}")
```

### List Agents

```python
from pathlib import Path
from aee.infrastructure.storage import AgentRepository

agent_repo = AgentRepository(agents_dir=Path("data/agents"))

# List all agents for a task
agents = agent_repo.list_agents(
    task_name="nanozymes",
    sort_by="created_at"  # or "metrics.f1"
)

for agent_path in agents:
    _, metadata = agent_repo.load(agent_path)
    print(f"{agent_path.name}")
    print(f"  F1: {metadata.metrics.get('f1', 0):.3f}")
    print(f"  Created: {metadata.created_at}")
```

---

## Dataset Building

### Build Training Dataset

```python
from pathlib import Path
from aee.application.services import DatasetBuilder
from aee.infrastructure.storage import (
    GroundTruthRepository,
    DocumentRepository,
    DataSplitRepository
)
from aee.domain.tasks import get_task

# Setup repositories
gt_repo = GroundTruthRepository(gt_dir=Path("data/ground_truth"))
doc_repo = DocumentRepository(parsed_dir=Path("data/parsed"))
split_repo = DataSplitRepository()

# Build dataset
dataset_builder = DatasetBuilder(gt_repo, doc_repo, split_repo)

task = get_task("nanozymes")
splits_dir = Path("data/splits")
task_name = "nanozymes"

dataset = dataset_builder.build_from_split(
    task=task,
    gt_path=Path("data/ground_truth/nanozymes.csv"),
    splits_dir=splits_dir,
    task_name=task_name,
    split_name="train",
    limit=50
)

print(f"Built dataset with {len(dataset)} examples")

# Access examples
for example in dataset[:3]:
    print(f"Document: {example.doc_id}")
    print(f"Experiments: {len(example.ground_truth)}")
    print(f"Text preview: {example.text[:100]}...")
    print()
```

### Create Data Split

```python
from pathlib import Path
from aee.infrastructure.storage import DataSplitRepository
import pandas as pd

# Load ground truth to get document IDs
gt_path = Path("data/ground_truth/nanozymes.csv")
df = pd.read_csv(gt_path)
doc_ids = df["filename"].str.replace(".pdf", "").unique().tolist()

print(f"Found {len(doc_ids)} unique documents")

# Create split
split_repo = DataSplitRepository()
splits = split_repo.create_random_split(
    documents=doc_ids,
    train_ratio=0.8,
    seed=42
)

# Save
splits_path = Path("data/splits/nanozymes.json")
split_repo.save_splits(splits, splits_path)

print(f"Train: {len(splits['train'])} documents")
print(f"Test: {len(splits['test'])} documents")
```

---

## Batch Prediction

### Predict on New Documents

```python
from pathlib import Path
from aee.application.use_cases import BatchPredictionUseCase, BatchPredictionRequest
from aee.infrastructure.storage import (
    AgentRepository,
    DocumentRepository,
    PredictionRepository
)
from aee.domain.tasks import get_task

# Setup
task = get_task("nanozymes")
agent_repo = AgentRepository(agents_dir=Path("data/agents"))
doc_repo = DocumentRepository(parsed_dir=Path("data/parsed"))
pred_repo = PredictionRepository(predictions_dir=Path("data/predictions"))

# Load agent
agent_path = agent_repo.get_latest(task.name)
agent, metadata = agent_repo.load(agent_path)

# Create request
request = BatchPredictionRequest(
    agent=agent,
    task=task,
    document_ids=["paper1", "paper2", "paper3"],  # or None for all
    parsed_dir=Path("data/parsed")
)

# Execute prediction
use_case = BatchPredictionUseCase(
    agent_repo=agent_repo,
    document_repo=doc_repo,
    prediction_repo=pred_repo
)

response = use_case.execute(request)

if response.success:
    print(f"Predicted {response.num_predictions} documents")
    print(f"Saved to {response.output_path}")

    # Access predictions
    for doc_id, experiments in response.predictions.items():
        print(f"\n{doc_id}:")
        for exp in experiments:
            print(f"  - {exp.formula}, {exp.activity}")
else:
    print(f"Prediction failed: {response.error_message}")
```

### Single Document Prediction

```python
from pathlib import Path
from aee.infrastructure.agents import UniversalExtractor
from aee.infrastructure.llm import setup_student
from aee.infrastructure.storage import DocumentRepository
from aee import settings

# Setup LLM
student_lm = setup_student(settings)

# Create extractor
from aee.domain.tasks import get_task
task = get_task("nanozymes")
extractor = UniversalExtractor(signature=task.signature)

# Load document
doc_repo = DocumentRepository(parsed_dir=Path("data/parsed"))
doc_id = "paper1"
document = doc_repo.load(doc_id)

# Predict
try:
    result = extractor(text=document.text)
    experiments = result.data.experiments

    print(f"Extracted {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.formula}: {exp.activity}")
except Exception as e:
    print(f"Extraction failed: {e}")
```

### Using AgentManager (Optional)

```python
from pathlib import Path
from aee.application.services import AgentManager
from aee.infrastructure.storage import AgentRepository

# AgentManager provides higher-level agent operations
agent_repo = AgentRepository(agents_dir=Path("data/agents"))
agent_manager = AgentManager(agent_repo)

# Get latest agent path
agent_path = agent_manager.get_latest_agent("nanozymes")

if agent_path:
    agent, metadata = agent_manager.load_agent(agent_path)
    print(f"Loaded agent with F1={metadata.metrics.get('f1', 0):.3f}")
```

---

## Evaluation

### Evaluate Predictions

```python
from pathlib import Path
from aee.domain.evaluation import ExperimentMatcher, TaskMetric
from aee.infrastructure.storage import (
    GroundTruthRepository,
    PredictionRepository
)
from aee.domain.tasks import get_task

# Setup
task = get_task("nanozymes")
gt_repo = GroundTruthRepository(gt_dir=Path("data/ground_truth"))
pred_repo = PredictionRepository(predictions_dir=Path("data/predictions"))

# Load ground truth experiments
ground_truth = gt_repo.load_experiments(
    task=task,
    document_ids=["paper1", "paper2"]
)

# Load predictions
predictions = pred_repo.load_predictions(
    Path("data/predictions/nanozymes_predictions.json")
)

# Create matcher with task-specific settings
matcher = ExperimentMatcher(
    compare_fields=task.compare_fields,
    tolerance=task.float_tolerance  # or custom value like 0.05
)

# Compute metrics
metric = TaskMetric(matcher=matcher)
results = metric.compute_all(predictions, ground_truth)

# Display results
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")
print(f"True Positives: {results['true_positives']}")
print(f"False Positives: {results['false_positives']}")
print(f"False Negatives: {results['false_negatives']}")
```

### Custom Evaluation

```python
from aee.domain.evaluation import ExperimentMatcher, TaskMetric

# Create custom matcher with specific fields
matcher = ExperimentMatcher(
    compare_fields=["formula", "activity"],  # Only these fields
    tolerance=0.1  # More lenient for floats
)

metric = TaskMetric(matcher=matcher)

# Evaluate per-document
for doc_id in ["paper1", "paper2"]:
    pred = predictions.get(doc_id, [])
    gt = ground_truth.get(doc_id, [])

    results = metric.compute_all({doc_id: pred}, {doc_id: gt})

    print(f"\n{doc_id}:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1: {results['f1']:.3f}")
```

---

## Custom Workflows

### End-to-End Pipeline

```python
from pathlib import Path
from aee.domain.tasks import get_task
from aee.infrastructure.parsers import DoclingParser
from aee.infrastructure.storage import DocumentRepository, AgentRepository
from aee.infrastructure.agents import UniversalExtractor
from aee.infrastructure.llm import setup_student
from aee import settings

def extract_from_pdf(pdf_path: Path, task_name: str, agent_path: Path):
    """Complete pipeline: PDF -> Parsing -> Extraction -> Results"""

    # 1. Parse PDF
    print("Parsing PDF...")
    parser = DoclingParser(device="cpu")
    document = parser.parse(pdf_path)

    # 2. Load task and agent
    print("Loading agent...")
    task = get_task(task_name)
    agent_repo = AgentRepository(agents_dir=Path("data/agents"))
    agent, metadata = agent_repo.load(agent_path)

    # 3. Setup LLM
    student_lm = setup_student(settings)

    # 4. Extract using the agent
    print("Extracting data...")
    result = agent(text=document.text)
    experiments = result.data.experiments

    # 5. Return results
    print(f"Extracted {len(experiments)} experiments")
    return experiments

# Use it
experiments = extract_from_pdf(
    pdf_path=Path("data/pdfs/new_paper.pdf"),
    task_name="nanozymes",
    agent_path=Path("data/agents/nanozymes_sota_2026-02-17.json")
)

for exp in experiments:
    print(f"Formula: {exp.formula}")
    print(f"Activity: {exp.activity}")
    print(f"Km: {exp.km_value} {exp.km_unit}")
    print()
```

### Jupyter Notebook Example

```python
# In a Jupyter notebook
from pathlib import Path
from aee.domain.tasks import get_task
from aee.infrastructure.storage import DocumentRepository, AgentRepository
from aee.infrastructure.llm import setup_student
from aee import settings
import pandas as pd

# Setup
task = get_task("nanozymes")
doc_repo = DocumentRepository(parsed_dir=Path("data/parsed"))
agent_repo = AgentRepository(agents_dir=Path("data/agents"))

# Load agent
agent_path = agent_repo.get_latest(task.name)
agent, metadata = agent_repo.load(agent_path)

# Setup LLM
student_lm = setup_student(settings)

# Process multiple documents
results = []
for doc_id in ["paper1", "paper2", "paper3"]:
    document = doc_repo.load(doc_id)
    result = agent(text=document.text)

    for exp in result.data.experiments:
        results.append({
            "document": doc_id,
            "formula": exp.formula,
            "activity": exp.activity,
            "km_value": exp.km_value,
            "vmax_value": exp.vmax_value
        })

# Create DataFrame
df = pd.DataFrame(results)
display(df)

# Save to CSV
df.to_csv("extracted_data.csv", index=False)
```

### Batch Processing with Progress

```python
from pathlib import Path
from tqdm import tqdm
from aee.infrastructure.parsers import DoclingParser
from aee.infrastructure.storage import DocumentRepository

def batch_parse_pdfs(pdf_dir: Path, output_dir: Path):
    """Parse all PDFs in directory with progress bar"""

    parser = DoclingParser(device="cpu")
    doc_repo = DocumentRepository(parsed_dir=output_dir)

    pdf_files = list(pdf_dir.glob("*.pdf"))

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            # Parse
            document = parser.parse(pdf_path)

            # Save
            doc_repo.save(document)

        except Exception as e:
            print(f"\nError parsing {pdf_path.name}: {e}")
            continue

    print(f"\nParsed {len(pdf_files)} PDFs")

# Use it
batch_parse_pdfs(
    pdf_dir=Path("data/pdfs"),
    output_dir=Path("data/parsed")
)
```

---

## Configuration in Code

### Load Settings

```python
from aee import settings as global_settings
from pathlib import Path

# Use global settings (loaded from environment and config)
print(f"Student model: {global_settings.llm.student.model}")
print(f"Parsed dir: {global_settings.paths.parsed_dir}")
print(f"Num trials: {global_settings.optimization.num_trials}")

# Or create custom settings from YAML
import yaml
with open(Path("config/default.yaml")) as f:
    config_dict = yaml.safe_load(f)

# Access configuration values
print(f"Student model: {config_dict['llm']['student']['model']}")
```

### Create Settings in Code

```python
from aee.infrastructure.config.settings import Settings
from pathlib import Path

# Load custom settings from YAML file
custom_settings = Settings.load(config_path=Path("config/default_fast.yaml"))

print(f"Student model: {custom_settings.llm.student.model}")
print(f"Num trials: {custom_settings.optimization.num_trials}")
```

### Load Initial Instruction

```python
from aee.infrastructure.config.instruction_loader import InstructionLoader
from pathlib import Path

# Load instruction for task
loader = InstructionLoader(config_dir=Path("config"))
instruction = loader.load("initial_instructions/nanozymes_sota.txt")

print(f"Instruction length: {len(instruction)} characters")
print(f"Instruction hash: {loader.compute_hash(instruction)}")

# Load with metadata
metadata = loader.load_with_metadata("initial_instructions/nanozymes_sota.txt")
print(f"File: {metadata['instruction_file']}")
print(f"Hash: {metadata['instruction_hash']}")
```

---

## Best Practices

### Error Handling

```python
from aee.shared.exceptions import (
    TaskNotFoundError,
    ParserError,
    AgentNotFoundError,
    UseCaseExecutionError
)

# Task not found
try:
    from aee.domain.tasks import get_task
    task = get_task("nonexistent")
except TaskNotFoundError as e:
    print(f"Task error: {e}")

# Parsing error
from aee.infrastructure.parsers import DoclingParser
parser = DoclingParser()
try:
    document = parser.parse(Path("invalid.pdf"))
except ParserError as e:
    print(f"Parsing error: {e}")

# Agent not found
from aee.infrastructure.storage import AgentRepository
agent_repo = AgentRepository(agents_dir=Path("data/agents"))
try:
    agent, metadata = agent_repo.load(Path("nonexistent.json"))
except AgentNotFoundError as e:
    print(f"Agent error: {e}")

# Use case execution error
from aee.application.use_cases import OptimizeAgentUseCase
try:
    response = use_case.execute(request)
except UseCaseExecutionError as e:
    print(f"Use case error: {e}")
```

### Logging

```python
import logging
from aee import setup_logging
from pathlib import Path

# Setup logging
setup_logging(log_level="DEBUG", log_dir=Path("logs"))

# Get logger
logger = logging.getLogger(__name__)

# Use it
logger.info("Starting extraction...")
logger.debug(f"Processing document: {doc_id}")
logger.warning("No experiments found")
logger.error(f"Extraction failed: {error}")
```

### LLM Cache Management

```python
from aee.infrastructure.llm import setup_student
from aee import settings

# For optimization: enable cache (default)
student_lm = setup_student(settings, enable_cache=True)

# For predictions: disable cache for fresh results
student_lm = setup_student(settings, enable_cache=False)

# Note: Cache is global state managed by DSPy
# Once enabled, it affects all subsequent LLM calls
```

### Working with Data Splits

```python
from pathlib import Path
from aee.infrastructure.storage import DataSplitRepository
import pandas as pd

# Create data split
gt_path = Path("data/ground_truth/nanozymes.csv")
df = pd.read_csv(gt_path)
doc_ids = df["filename"].str.replace(".pdf", "").unique().tolist()

split_repo = DataSplitRepository()
splits = split_repo.create_random_split(
    documents=doc_ids,
    train_ratio=0.8,
    seed=42
)

# Save splits
split_repo.save_splits(splits, Path("data/splits/nanozymes.json"))

# Load splits later
splits = split_repo.load_splits(Path("data/splits/nanozymes.json"))
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")
```

---

## Additional Resources

- [Architecture Documentation](architecture.md) - System design details
- [Configuration Reference](configuration.md) - All config options
- [Troubleshooting Guide](troubleshooting.md) - Common issues
- [Adding Tasks](adding_tasks.md) - Create custom tasks
