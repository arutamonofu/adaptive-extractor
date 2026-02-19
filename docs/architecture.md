# AutoEvoExtractor Architecture

Technical documentation for developers contributing to or integrating with AutoEvoExtractor.

> **New to the project?** Start with the [Main README](../README.md) for installation and quick start.

## Design Principles

1. **Separation of Concerns**: Business logic, data access, and external integrations are cleanly separated
2. **Dependency Inversion**: High-level modules depend on abstractions, not implementations
3. **Testability**: Each layer can be tested independently
4. **Extensibility**: New extraction tasks via plugin system (TaskDefinition or YAML-based TaskConfig)
5. **Maintainability**: Clear boundaries between components
6. **Pragmatism**: Functional API preferred for simple operations, classes for complex state

## 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ INTERFACE LAYER                                             │
│ - CLI commands (thin wrappers)                              │
│ - Entry points for users                                    │
│ - Environment-based config (AEE_ENV)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ APPLICATION LAYER                                           │
│ - Use Cases: OptimizeAgent, BatchPrediction, ParseDocuments │
│ - Services: AgentManager, DatasetBuilder, ExperimentTracker │
│           DataValidator                                     │
│ - DTOs: Defined inline in use case modules                  │
│ - Orchestrates workflows and coordinates domain objects     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ DOMAIN LAYER                                                │
│ - Task Plugin System (TaskDefinition ABC, TaskConfig, YAML) │
│ - BaseAgent ABC (abstraction for extraction agents)         │
│ - Domain Entities (Experiments, Documents, Extractions)     │
│ - Evaluation Logic (ExperimentMatcher, TaskMetric)          │
│ - Pure business rules, no infrastructure dependencies       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ INFRASTRUCTURE LAYER                                        │
│ - Agents: UniversalExtractor (DSPy ChainOfThought)          │
│ - LLM Providers: OllamaLM with Circuit Breaker protection   │
│ - Document Parsers: DoclingParser, MarkerParser             │
│ - Storage: Repository classes + Functional API              │
│ - Config: Pydantic Settings, YAML, Environment variables    │
│ - Tracking: MLflow with DSPy autolog integration            │
│ - InstructionLoader: Load optimization instructions         │
└─────────────────────────────────────────────────────────────┘
```

## Layer Details

### 1. Interface Layer (`src/aee/interface/`)

**Purpose**: Thin entry points that translate user commands into application use cases.

**Structure**:
```
interface/
├── cli/
│   ├── parse.py      # Document parsing command
│   ├── extract.py    # Batch extraction command
│   ├── optimize.py   # Agent optimization command
│   └── common.py     # Shared CLI utilities
```

**Responsibilities**:
- Parse command-line arguments
- Load configuration
- Create dependencies (dependency injection)
- Invoke use cases with proper requests
- Display results to users

**Key Pattern**: Each CLI command is ~50-100 lines that delegate to use cases.

**Example**:
```python
# scripts/optimize.py (9 lines)
from aee.interface.cli.optimize import optimize_command
if __name__ == "__main__":
    sys.exit(optimize_command())

# src/aee/interface/cli/optimize.py (~200 lines)
def optimize_command(argv=None):
    args = parse_arguments(argv)
    config = load_configuration(args.config)

    # Create dependencies
    task = get_task(args.task)
    agent_repo = AgentRepository(agents_dir=config.paths.agents)
    agent_manager = AgentManager(agent_repo)

    # Build request
    request = OptimizeAgentRequest(
        task=task,
        gt_path=config.paths.ground_truth,
        # ... more fields
    )

    # Execute use case
    use_case = OptimizeAgentUseCase(agent_manager, ...)
    response = use_case.execute(request)

    # Display results
    print(f"Agent saved to: {response.agent_path}")
    print(f"F1 Score: {response.final_score:.3f}")
```

### 2. Application Layer (`src/aee/application/`)

**Purpose**: Orchestrates workflows and coordinates domain objects. Contains business logic for multi-step operations.

**Structure**:
```
application/
├── use_cases/
│   ├── optimize_agent.py      # Agent optimization workflow
│   ├── predict_batch.py       # Batch extraction workflow (formerly extract_batch)
│   └── parse_documents.py     # Document parsing workflow
└── services/
    ├── agent_manager.py       # Agent lifecycle management
    ├── dataset_builder.py     # Dataset creation
    ├── experiment_tracker.py  # MLflow tracking facade
    └── data_validator.py      # Data validation service
```

**Note**: DTOs (Request/Response classes) are defined inline in their respective use case modules:
- `OptimizeAgentRequest/Response` in `optimize_agent.py`
- `BatchPredictionRequest/Response` in `predict_batch.py`
- `ParseDocumentsRequest/Response` in `parse_documents.py`

#### Use Cases

Use cases represent complete business workflows with clear inputs and outputs.

**Pattern**:
```python
@dataclass
class OptimizeAgentRequest:
    """Input for optimization workflow."""
    task: TaskDefinition
    gt_path: Path
    split_path: Path
    num_trials: int
    # ... more fields

@dataclass
class OptimizeAgentResponse:
    """Output from optimization workflow."""
    success: bool
    agent_path: Path
    final_score: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class OptimizeAgentUseCase:
    def __init__(self, agent_manager, dataset_builder, tracker, ...):
        # Dependency injection
        self.agent_manager = agent_manager
        self.dataset_builder = dataset_builder
        # ...

    def execute(self, request: OptimizeAgentRequest) -> OptimizeAgentResponse:
        """Execute optimization workflow."""
        # 1. Load ground truth
        # 2. Prepare datasets
        # 3. Create metric
        # 4. Run optimization
        # 5. Save agent
        # 6. Track results
        # 7. Return response
```

**Benefits**:
- Testable (mock dependencies)
- Reusable (CLI, API, notebooks)
- Clear contracts (Request/Response)
- Proper error handling

#### Services

Services provide high-level operations that use multiple repositories or domain objects.

**AgentManager** (`agent_manager.py`):
- Save agents with metadata
- Load latest/best agent
- Get agent history
- Compare agents

**DatasetBuilder** (`dataset_builder.py`):
- Build training datasets
- Build evaluation datasets
- Get dataset statistics
- Handle data splitting

**ExperimentTracker** (`experiment_tracker.py`):
- MLflow integration with native DSPy support
- Automatic DSPy operation tracking via autolog
- Log parameters, metrics, and DSPy models
- Track artifacts and model serialization
- Simplified interface for experiment tracking

**DataValidator** (`data_validator.py`):
- Validate data splits against ground truth
- Check for overlapping documents in train/val splits
- Verify dataset quality before optimization
- Pre-flight validation for OptimizeAgentUseCase
- Log validation results with structured output

### 3. Domain Layer (`src/aee/domain/`)

**Purpose**: Core business logic and domain concepts. No dependencies on infrastructure or external libraries (except Pydantic for models).

**Structure**:
```
domain/
├── agents/
│   └── base.py          # BaseAgent ABC for extraction agents
├── entities/
│   ├── document.py      # ProcessedDocument, DocumentMetadata
│   ├── experiment.py    # Base Experiment class
│   └── extraction.py    # ExtractionResult, ExtractionOutput
├── tasks/
│   ├── base.py          # TaskDefinition ABC
│   ├── config.py        # TaskConfig, FieldSpec, RowConverterConfig
│   ├── registry.py      # TaskRegistry
│   ├── dynamic_models.py     # Dynamic Pydantic model generation
│   ├── dynamic_wrapper.py    # ConfigBackedTask wrapper
│   ├── loader.py        # YAML loading utilities
│   ├── signature.py     # DSPy signature generation
│   └── nanozymes/       # Nanozyme task plugin (YAML-based)
│       ├── task.yaml    # Task configuration manifest
│       ├── models.py    # NanozymeExperiment (if custom models needed)
│       └── __init__.py  # Task registration
└── evaluation/
    ├── matcher.py       # ExperimentMatcher
    └── metrics.py       # TaskMetric
```

#### BaseAgent ABC

New in v0.4: The `BaseAgent` abstract base class in `domain/agents/base.py` provides a clean abstraction for extraction agents, allowing the application layer to depend on abstractions rather than DSPy-specific implementations.

```python
class BaseAgent(ABC):
    @abstractmethod
    def forward(self, document_text: str) -> Any:
        """Execute the extraction pipeline."""
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent to a file."""
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent from a file."""
```

#### Task Plugin System (v0.4: YAML-based)

The task plugin system is the core extensibility mechanism. Starting from v0.4, tasks can be defined using YAML configuration files, eliminating the need for boilerplate Python code.

**Two Approaches**:

1. **Classic (TaskDefinition ABC)** - Define task in Python code
2. **Modern (TaskConfig + YAML)** - Define task in YAML, models generated dynamically

**TaskDefinition ABC** (`base.py`):
```python
class TaskDefinition(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique task identifier (e.g., 'nanozymes')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the task."""

    @property
    @abstractmethod
    def signature(self) -> Type[dspy.Signature]:
        """DSPy signature for LLM extraction."""

    @property
    @abstractmethod
    def output_model(self) -> Type[BaseModel]:
        """Pydantic model for extraction output."""

    @property
    @abstractmethod
    def experiment_model(self) -> Type[BaseModel]:
        """Pydantic model for individual experiments."""

    @property
    @abstractmethod
    def row_converter(self) -> Callable:
        """Function to convert CSV row to experiment."""

    @property
    @abstractmethod
    def compare_fields(self) -> List[str]:
        """Fields to use for experiment matching."""

    def validate(self) -> None:
        """Validate task completeness."""
```

**TaskConfig (YAML-based)** (`config.py`):
```yaml
# domain/tasks/nanozymes/task.yaml
name: "nanozymes"
description: "Extract nanozyme experiment data"
fields:
  - name: "formula"
    type: "str"
    description: "Chemical formula"
  - name: "activity"
    type: "float"
    description: "Enzymatic activity"
  - name: "length"
    type: "float"
    description: "Nanoparticle length (nm)"
compare_fields:
  - "formula"
  - "activity"
  - "length"
float_tolerance: 0.05
instruction_file: "config/initial_instructions/nanozymes_sota.txt"
```

**ConfigBackedTask** (`dynamic_wrapper.py`):
Wrapper class that creates a `TaskDefinition` from a `TaskConfig`, dynamically generating:
- Pydantic models (experiment, output)
- DSPy signature
- Row converter function

**TaskRegistry** (`registry.py`):
```python
class TaskRegistry:
    def register(self, task: TaskDefinition, validate: bool = True):
        """Register a TaskDefinition instance."""

    def register_config(self, config: TaskConfig, validate: bool = True):
        """Register a TaskConfig."""

    def register_from_yaml(self, yaml_path: str | Path, validate: bool = True):
        """Load and register a task from YAML file."""

    def get(self, task_name: str) -> TaskDefinition:
        """Get a registered task (returns ConfigBackedTask for YAML tasks)."""

    def list_tasks() -> List[TaskDefinition]:
        """List all registered tasks."""

# Global registry access
def get_task(name: str) -> TaskDefinition:
    return _registry.get(name)
```

**Example Task Plugin (YAML-based)**:
```yaml
# domain/tasks/nanozymes/task.yaml
name: "nanozymes"
description: "Extract nanozyme experiment data from scientific papers"
fields:
  - name: "formula"
    type: "str"
    required: true
  - name: "activity"
    type: "float"
    required: true
  - name: "length"
    type: "float"
    description: "Nanoparticle length in nanometers"
compare_fields:
  - "formula"
  - "activity"
  - "length"
float_tolerance: 0.05
instruction_file: "config/initial_instructions/nanozymes_sota.txt"
```

The task is auto-registered when loaded via `load_task_from_yaml()` or `load_and_register_task()`.

### 4. Infrastructure Layer (`src/aee/infrastructure/`)

**Purpose**: External integrations and technical implementations. Depends on external libraries and services.

**Structure**:
```
infrastructure/
├── agents/
│   ├── extractor.py   # UniversalExtractor (DSPy ChainOfThought wrapper)
│   └── __init__.py
├── cache/
│   └── __init__.py    # DSPy cache configuration (memory + disk)
├── config/
│   ├── settings.py         # Pydantic Settings with env var support
│   ├── instruction_loader.py # Load optimization instructions with metadata
│   ├── logging.py          # Logging configuration
│   ├── environments.py     # AEE_ENV-based environment loading
│   └── __init__.py
├── llm/
│   ├── provider.py    # OllamaLM with Circuit Breaker protection
│   ├── circuit_breaker.py  # Circuit breaker pattern implementation
│   └── __init__.py    # create_lm, setup_student, setup_teacher
├── parsers/
│   ├── base.py        # BaseParser ABC
│   ├── parsers.py     # DoclingParser, MarkerParser
│   ├── cleaning.py    # TextCleaner
│   └── __init__.py    # get_parser()
├── storage/
│   ├── agents.py       # AgentRepository + Functional API (agents_fn)
│   ├── agents_fn.py    # save_agent, load_agent, list_agents (preferred)
│   ├── ground_truth.py # GroundTruthRepository
│   ├── ground_truth_fn.py # load_ground_truth (functional API)
│   ├── extractions.py  # ExtractionRepository
│   ├── documents.py    # DocumentRepository
│   ├── splits.py       # DataSplitRepository
│   ├── splits_fn.py    # load_split, validate_splits (functional API)
│   ├── migrations.py   # AgentMigrator, GroundTruthMigrator
│   └── __init__.py
└── tracking/
    └── __init__.py    # ExperimentTracker with DSPy autolog
```

**Key Changes in v0.4**:
- **Functional API** preferred for storage operations (simpler, more testable)
- **Circuit Breaker** pattern for LLM calls (prevents cascade failures)
- **InstructionLoader** for managing optimization instructions with hash tracking
- **Migrations** support for evolving data formats
- Removed `optimization/` module (MIPROv2 used directly in use cases)

#### Functional API for Storage (New in v0.4)

New code should prefer the functional API over class-based repositories:

```python
# Functional API (preferred)
from aee.infrastructure.storage import save_agent, load_agent, list_agents

# Save an agent
agent_path = save_agent(
    agent=agent_dict,
    task_name="nanozymes",
    agents_dir=Path("data/agents"),
    metrics={"f1": 0.85},
    config_snapshot={"num_trials": 10},
)

# Load an agent
agent, metadata = load_agent(agent_path)

# List agents
agents = list_agents(agents_dir, task_name="nanozymes")
```

**Benefits**:
- Simpler to use and test
- No state management overhead
- Composable with other functions
- Clearer data flow

**Class-based repositories** (`AgentRepository`, `GroundTruthRepository`, etc.) are maintained for backward compatibility but new code should use the functional API.

#### Circuit Breaker Pattern

The `CircuitBreaker` class in `llm/circuit_breaker.py` protects against cascade failures from LLM providers:

```python
from aee.infrastructure.llm import CircuitBreaker

circuit_breaker = CircuitBreaker(
    failure_threshold=5,       # Open after 5 failures
    reset_timeout=60.0,        # Wait 60s before half-open
    half_open_max_calls=3,     # Allow 3 test calls in half-open
    name="ollama-mistral",
)

# Used internally by OllamaLM
llm = OllamaLM(config, circuit_breaker=circuit_breaker)
```

**States**:
- **Closed**: Normal operation, requests pass through
- **Open**: Circuit tripped, requests fail immediately
- **Half-Open**: Testing if provider recovered

#### InstructionLoader

Load optimization instructions with metadata tracking:

```python
from aee.infrastructure.config.instruction_loader import InstructionLoader

loader = InstructionLoader(config_dir=Path("config"))
metadata = loader.load_with_metadata("initial_instructions/nanozymes_sota.txt")

instruction = metadata["instruction"]
instruction_hash = metadata["instruction_hash"]  # SHA256 (first 12 chars)
```

Used for tracking which instruction was used during agent optimization.

#### Repository Pattern (Classic)

Repositories provide clean data access interfaces. Still available for backward compatibility:

```python
@dataclass
class AgentMetadata:
    task_name: str
    created_at: str
    model_version: str
    metrics: Dict[str, float]
    config_snapshot: Dict[str, Any]
    git_commit: Optional[str] = None
    description: Optional[str] = None
    initial_instruction_file: Optional[str] = None
    instruction_hash: Optional[str] = None

class AgentRepository:
    def __init__(self, agents_dir: Path):
        self.agents_dir = agents_dir

    def save(self, agent, task_name, metadata) -> Path:
        """Save agent with metadata."""
        # Auto-version: task_v1_2024-01-15.json
        # Save .json + .meta.json side-by-side

    def load(self, path) -> Tuple[Any, AgentMetadata]:
        """Load agent with metadata."""

    def get_latest(self, task_name) -> Optional[Path]:
        """Get most recent agent."""

    def list_agents(self, task_name=None, sort_by="created_at") -> List[Path]:
        """List agents with filtering."""
```

**Benefits**:
- Data access logic centralized
- Easy to mock for testing
- Versioning and metadata built-in
- Clean separation from business logic

## Data Flow Examples

### Example 1: Document Parsing

```
User Command
    │
    └──> CLI (parse.py)
            │
            ├──> Load Settings (AEE_ENV or --config)
            │
            └──> ParseDocumentsUseCase
                    │
                    ├──> get_parser("docling") [Infrastructure]
                    ├──> DocumentRepository [Infrastructure]
                    └──> ProcessedDocument [Domain]
                            │
                            └──> Save to data/parsed/
```

### Example 2: Agent Optimization (v0.4)

```
User Command
    │
    └──> CLI (optimize.py)
            │
            ├──> Load Settings + Task Config (YAML)
            │       │
            │       ├──> AEE_ENV=dev → config/dev.yaml
            │       └──> domain/tasks/nanozymes/task.yaml
            │
            └──> OptimizeAgentUseCase
                    │
                    ├──> InstructionLoader ← Load initial instruction
                    │       └──> instruction_hash (SHA256, first 12 chars)
                    │
                    ├──> DataValidator [Application] ← Pre-flight check
                    │       ├──> load_split (Functional API)
                    │       └──> validate_ground_truth
                    │
                    ├──> DatasetBuilder [Application]
                    │       ├──> GroundTruthRepository [Infrastructure]
                    │       ├──> DocumentRepository [Infrastructure]
                    │       └──> ConfigBackedTask [Domain]
                    │
                    ├──> MIPROv2 Optimizer (DSPy, direct usage)
                    │       ├──> Student LM (OllamaLM + CircuitBreaker)
                    │       └──> Teacher LM (OllamaLM + CircuitBreaker)
                    │
                    ├──> AgentManager [Application]
                    │       └──> save_agent (Functional API)
                    │           └──> data/agents/nanozymes_v1_*.json + .meta.json
                    │
                    └──> ExperimentTracker [Application]
                            ├──> MLflow with DSPy autolog
                            └──> log_dspy_model (native serialization)
```

### Example 3: Batch Extraction (v0.4)

```
User Command
    │
    └──> CLI (extract.py --agent nanozymes_v1_*.json)
            │
            ├──> Load Settings + Task Config
            │
            ├──> Setup LLM Cache + Circuit Breaker
            │
            └──> BatchPredictionUseCase
                    │
                    ├──> AgentManager.load_agent
                    │       └──> load_agent (Functional API)
                    │
                    ├──> DocumentRepository.list_documents
                    │
                    └──> For each document:
                            │
                            ├──> UniversalExtractor(document_text)
                            │       └──> DSPy ChainOfThought
                            │
                            └──> ExtractionRepository.save
                                    └──> data/extractions/{doc_id}_result.json
```

### Example 4: Adding a New Task (YAML-based, v0.4+)

**Modern Approach (recommended)**:
```
1. Create Task YAML
   └──> domain/tasks/proteins/task.yaml
        ├──> name, description
        ├──> fields (type, description, required)
        ├──> compare_fields
        └──> instruction_file

2. (Optional) Add Custom Models
   └──> domain/tasks/proteins/models.py
        ├──> ProteinExperiment (if custom logic needed)
        └──> ProteinExtractionOutput

3. Task Auto-Registration
   └──> CLI loads task automatically via ConfigBackedTask
   └──> All use cases, services, repos work automatically!
```

**Classic Approach (still supported)**:
```
1. Create Task Plugin [Domain]
   └──> domain/tasks/proteins/
        ├── models.py (ProteinExperiment)
        ├── signature.py (ProteinSignature)
        ├── converters.py (row_to_protein)
        └── __init__.py (ProteinTask)

2. Register Task
   └──> from aee.domain.tasks import register_task
        register_task(ProteinTask())

3. Use Existing Infrastructure
   └──> All use cases, services, repos work automatically!
```

## Design Patterns Used

### 1. Repository Pattern
Abstracts data storage and retrieval.
- **Where**: `infrastructure/storage/`
- **Why**: Separates business logic from data access

### 2. Use Case Pattern
Encapsulates business workflows with clear contracts.
- **Where**: `application/use_cases/`
- **Why**: Testable, reusable, clear boundaries

### 3. Service Layer
Provides high-level operations using multiple repositories.
- **Where**: `application/services/`
- **Why**: Reusable business operations

### 4. Dependency Injection
Components receive dependencies rather than creating them.
- **Where**: All layers
- **Why**: Testability, flexibility, decoupling

### 5. Strategy Pattern
Task plugins define different extraction strategies.
- **Where**: `domain/tasks/`
- **Why**: Easy to add new tasks without modifying existing code

### 6. Abstract Base Classes
Define interfaces for implementations.
- **Where**: `TaskDefinition`, `BaseParser`, etc.
- **Why**: Ensures consistency, enables validation

### 7. DTO Pattern
Request/Response objects for use cases.
- **Where**: `application/dto/`
- **Why**: Clear contracts, validation, type safety

## Configuration Management

Configuration is managed through multiple layers with clear precedence:

### Configuration Priority (highest → lowest)

1. **Environment Variables** (`.env` file, processed by pydantic-settings)
   - API keys (secrets): `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
   - Infrastructure URLs: `OLLAMA_STUDENT_BASE_URL`, `OLLAMA_TEACHER_BASE_URL`, `MLFLOW_TRACKING_URI`
   - Application settings: `LOG_LEVEL`, `AEE_ENV`

2. **CLI Arguments** (`--config` flag)
   - Custom YAML configuration file path
   - Highest priority for file-based config

3. **AEE_ENV Environment Selector**
   - `AEE_ENV=dev` → loads `config/dev.yaml`
   - `AEE_ENV=prod` → loads `config/prod.yaml`
   - Default: `AEE_ENV=dev` → falls back to `config/default.yaml`

4. **YAML Configuration Files** (`config/`)
   - `default.yaml` - Base production configuration
   - `dev.yaml`, `prod.yaml` - Environment-specific overrides
   - Custom configs via `--config` flag

5. **Internal Defaults** (in `Settings` class)
   - Fallback values for safe infrastructure paths
   - Minimal defaults to prevent crashes

### Environment Variables

**Secrets (required, must be in `.env` only)**:
```bash
# API Keys for non-Ollama providers
GEMINI_API_KEY="your-gemini-key"
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"

# Infrastructure URLs (environment-specific)
OLLAMA_STUDENT_BASE_URL="http://localhost:11434"
OLLAMA_TEACHER_BASE_URL="http://localhost:11435"
MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
DSPY_CACHE_DIR="~/.dspy_cache"
```

**Application Settings (can override via `.env`)**:
```bash
# Use double underscore for nested settings
LLM__STUDENT__MODEL="mistral-small3.1-24b-128k:latest"
LLM__STUDENT__TEMPERATURE=0.0
PROJECT__LOG_LEVEL="DEBUG"
OPTIMIZATION__NUM_TRIALS=50
PATHS__AGENTS_DIR="data/agents"
```

### YAML Configuration Structure

```yaml
# config/default.yaml
project:
  name: "autoevoextractor"
  log_level: "INFO"

llm:
  student:
    use_ollama: true
    model: "mistral-small3.1-24b-128k:latest"
    temperature: 0.0
    enable_cache: true
    ollama:
      num_ctx: 64000
      num_predict: 2048

paths:
  pdf_dir: "data/pdf"
  parsed_dir: "data/parsed"
  ground_truth_dir: "data/ground_truth"
  splits_file: "data/splits/nanozymes.json"
  agents_dir: "data/agents"
  extractions_dir: "data/extractions"

optimization:
  num_trials: 70
  train_split: 20
  total_load: 20
  num_candidates: 10
  max_bootstrapped_demos: 1
  minibatch: false
  metric_threshold: 1.0
  init_temperature: 0.5
  random_seed: 42
  use_cache: true
  verbose: true

task:
  name: "nanozymes"
  initial_instruction_file: "config/initial_instructions/nanozymes_sota.txt"
  evaluation:
    float_tolerance: 0.05
    compare_fields:
      - "formula"
      - "activity"
      - "length"

extraction:
  enable_cache: false

cache:
  disk_size_limit_bytes: 30000000000  # 30 GB
  memory_max_entries: 1000000

circuit_breaker:
  failure_threshold: 8
  reset_timeout: 30.0
  half_open_max_calls: 1
```

### Security Notes

- **API keys MUST be set via environment variables only** (never in YAML files)
- **Infrastructure URLs** (OLLAMA_*, MLFLOW_*) should be in `.env` for environment portability
- **Application parameters** belong in YAML files for version control
- **SecretStr type** used for sensitive fields with safe `__repr__` implementation

### Loading Configuration

```python
from aee.infrastructure.config.settings import Settings
from aee.infrastructure.config.environments import load_settings_for_environment

# Method 1: Auto-load based on AEE_ENV
settings = load_settings_for_environment()

# Method 2: Load with custom config file
settings = Settings.load(config_path=Path("config/custom.yaml"))

# Method 3: Load with CLI-provided path (used in CLI commands)
settings = Settings.load(config_path=args.config)
```

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Test each layer independently
- Mock dependencies
- Fast execution

### Integration Tests (`tests/integration/`)
- Test component interaction
- Use real implementations (not mocks)
- Test full workflows

> **Note:** Test suite is under development. Structure follows the organization below.

### Test Organization
```
tests/
├── unit/
│   ├── domain/         # Domain logic tests
│   ├── application/    # Use cases and services
│   ├── infrastructure/ # Repositories, parsers, LLM
│   └── interface/      # CLI tests
├── integration/
│   ├── test_parse_pipeline.py
│   ├── test_predict_pipeline.py
│   └── test_optimize_pipeline.py
└── fixtures/
    ├── sample_pdfs/
    ├── sample_parsed/
    └── sample_ground_truth/
```

## Backward Compatibility

To maintain compatibility with existing code:

1. **Compatibility Wrappers**: Old import paths still work
   ```python
   # Old way (still works with deprecation warning)
   from aee.ingestion import DoclingParser

   # New way
   from aee.infrastructure.parsers import DoclingParser
   ```

2. **Deprecated Modules**: Marked with warnings
   - `aee.ingestion` → `aee.infrastructure.parsers`
   - `aee.llm` → `aee.infrastructure.llm`
   - `aee.utils.io` → `aee.infrastructure.storage`

3. **Data Compatibility**: All existing data formats unchanged
   - Agent JSON files
   - Parsed documents
   - Extractions
   - Ground truth CSV

## Evolution from v0.3 to v0.4

Version 0.4 introduces significant architectural improvements while maintaining backward compatibility.

### Major Changes

#### 1. Functional API for Storage (New Preferred Pattern)

**v0.3** (still supported):
```python
from aee.infrastructure.storage import AgentRepository

repo = AgentRepository(agents_dir=Path("data/agents"))
repo.save(agent, task_name, metadata)
```

**v0.4** (recommended):
```python
from aee.infrastructure.storage import save_agent, load_agent

agent_path = save_agent(
    agent=agent_dict,
    task_name="nanozymes",
    agents_dir=Path("data/agents"),
    metrics={"f1": 0.85},
)
agent, metadata = load_agent(agent_path)
```

**Why**: Simpler, more testable, no state management overhead.

#### 2. YAML-based Task Configuration

**v0.3** (Classic approach):
```python
# domain/tasks/nanozymes/__init__.py
class NanozymeTask(TaskDefinition):
    @property
    def name(self) -> str:
        return "nanozymes"
    # ... 100+ lines of boilerplate
```

**v0.4** (YAML-based):
```yaml
# domain/tasks/nanozymes/task.yaml
name: "nanozymes"
description: "Extract nanozyme experiment data"
fields:
  - name: "formula"
    type: "str"
  - name: "activity"
    type: "float"
compare_fields: ["formula", "activity"]
```

**Why**: 10x less code, easier to add new tasks, dynamic model generation.

#### 3. BaseAgent ABC in Domain Layer

**New in v0.4**:
```python
# domain/agents/base.py
class BaseAgent(ABC):
    @abstractmethod
    def forward(self, document_text: str) -> Any: ...
    
    @abstractmethod
    def save(self, path: str) -> None: ...
    
    @abstractmethod
    def load(self, path: str) -> None: ...
```

**Why**: Clean abstraction, application layer no longer depends on DSPy-specific implementations.

#### 4. Circuit Breaker for LLM Calls

**New in v0.4**:
```python
from aee.infrastructure.llm import CircuitBreaker

circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    reset_timeout=30.0,
)
llm = OllamaLM(config, circuit_breaker=circuit_breaker)
```

**Why**: Prevents cascade failures when LLM provider is unavailable.

#### 5. InstructionLoader with Hash Tracking

**New in v0.4**:
```python
from aee.infrastructure.config.instruction_loader import InstructionLoader

loader = InstructionLoader(config_dir=Path("config"))
metadata = loader.load_with_metadata("initial_instructions/nanozymes_sota.txt")
# metadata["instruction_hash"] → SHA256 (first 12 chars)
```

**Why**: Track which instruction was used during optimization for reproducibility.

#### 6. Environment-based Configuration (AEE_ENV)

**v0.3**: Manual config file selection
**v0.4**: `AEE_ENV` environment variable

```bash
# .env file
AEE_ENV=dev

# CLI automatically loads config/dev.yaml or falls back to default.yaml
python scripts/optimize.py
```

**Why**: Easier environment switching, better for CI/CD.

#### 7. Removed Modules

- `infrastructure/optimization/` - MIPROv2 now used directly in `OptimizeAgentUseCase`
- `domain/evaluation/scoring.py` - Scoring logic moved to `TaskMetric`

### Migration Guide

**For existing code**:
1. No breaking changes - all v0.3 code continues to work
2. Gradually migrate to Functional API for storage operations
3. Consider YAML-based task configuration for new tasks

**For new features**:
1. Use Functional API (`save_agent`, `load_agent`)
2. Use YAML for task definitions
3. Leverage `CircuitBreaker` for LLM calls
4. Use `InstructionLoader` for optimization instructions

### Version Compatibility Matrix

| Feature | v0.3 | v0.4 | Notes |
|---------|------|------|-------|
| TaskDefinition ABC | ✓ | ✓ | Still supported |
| TaskConfig (YAML) | ✗ | ✓ | New in v0.4 |
| AgentRepository | ✓ | ✓ | Delegates to functional API |
| save_agent() | ✗ | ✓ | New in v0.4 |
| CircuitBreaker | ✗ | ✓ | New in v0.4 |
| BaseAgent ABC | ✗ | ✓ | New in v0.4 |
| AEE_ENV | ✗ | ✓ | New in v0.4 |
| InstructionLoader | ✗ | ✓ | New in v0.4 |

## Performance Considerations

### 1. Caching (Two-Level)

**Memory Cache** (session-level):
- Fast in-memory caching for repeated calls
- Configurable max entries (default: 1,000,000)
- Global state via `dspy.configure_cache()`

**Disk Cache** (persistent across runs):
- Persistent caching for LLM responses
- Configurable size limit (default: 30 GB)
- Located in `DSPY_CACHE_DIR` (default: `~/.dspy_cache`)

```python
# Enable in config/default.yaml
llm:
  student:
    enable_cache: true
  teacher:
    enable_cache: true

cache:
  disk_size_limit_bytes: 30000000000  # 30 GB
  memory_max_entries: 1000000
```

### 2. Circuit Breaker Pattern

**New in v0.4**: Protects against cascade failures from LLM providers.

```python
# Configuration
circuit_breaker:
  failure_threshold: 8        # Open after 8 failures
  reset_timeout: 30.0         # Wait 30s before half-open
  half_open_max_calls: 1      # Allow 1 test call

# Usage (internal, automatic)
llm = OllamaLM(config, circuit_breaker=circuit_breaker)
```

**Benefits**:
- Fast failure when provider is down
- Automatic recovery testing
- Prevents resource exhaustion

### 3. Lazy Loading

- Modules loaded only when needed
- LLM providers initialized on first use
- Parsers created per-request (not global singletons)

### 4. Batch Processing

- Efficient batch operations in repositories
- `DocumentRepository.load_all()` loads all documents in single pass
- `BatchPredictionUseCase` processes documents sequentially with progress tracking

### 5. Type Annotations

- Modern Python 3.10+ type hints
- Better IDE support and autocomplete
- Runtime type validation via Pydantic

### 6. Functional API Performance

- No class instantiation overhead
- Simpler call stack
- Easier to optimize and profile

### 7. LLM Connection Management

- Reuse LLM connections for multiple requests
- Configurable timeouts and retries
- Rate limiting via `rate_limit_delay`

### 8. MLflow Optimization

- DSPy autolog with selective tracing
- Model serialization via native DSPy save (avoids pickle issues)
- Artifact logging as fallback

## Security Considerations

1. **Input Validation**: Pydantic models validate all inputs
2. **Path Sanitization**: File paths validated before use
3. **Error Handling**: Sensitive information not leaked in errors
4. **Secrets Management**: API keys exclusively from environment variables
5. **Type Safety**: Comprehensive type annotations prevent runtime errors
6. **Configuration Isolation**: Sensitive values never hardcoded in source

## Future Extensibility

The architecture is designed to easily support:

1. **New Extraction Tasks**: Via plugin system
2. **New LLM Providers**: Via provider interface
3. **New Document Parsers**: Via parser interface
4. **REST API**: Use cases can be wrapped in API endpoints
5. **Web UI**: Same use cases, different interface
6. **Multiple Languages**: Separate interface layer for each

## Adding New Features

See `docs/adding_tasks.md` for step-by-step guide to adding new extraction tasks.
