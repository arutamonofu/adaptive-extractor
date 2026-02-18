# AutoEvoExtractor Architecture

Technical documentation for developers contributing to or integrating with AutoEvoExtractor.

> **New to the project?** Start with the [Main README](../README.md) for installation and quick start.

## Design Principles

1. **Separation of Concerns**: Business logic, data access, and external integrations are cleanly separated
2. **Dependency Inversion**: High-level modules depend on abstractions, not implementations
3. **Testability**: Each layer can be tested independently
4. **Extensibility**: New extraction tasks via plugin system
5. **Maintainability**: Clear boundaries between components

## 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ INTERFACE LAYER                                             │
│ - CLI commands (thin wrappers)                              │
│ - Entry points for users                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ APPLICATION LAYER                                           │
│ - Use Cases: OptimizeAgent, BatchExtraction, ParseDocuments │
│ - Services: AgentManager, DatasetBuilder, ExperimentTracker │
│ - Orchestrates workflows and coordinates domain objects     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ DOMAIN LAYER                                                │
│ - Task Plugin System (TaskDefinition, TaskRegistry)         │
│ - Domain Entities (Experiments, Documents, Extractions)     │
│ - Evaluation Logic (Matchers, Metrics, Scoring)             │
│ - Pure business rules, no infrastructure dependencies       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│ INFRASTRUCTURE LAYER                                        │
│ - Agents (DSPy module implementations)                      │
│ - LLM Providers (Ollama, OpenAI)                            │
│ - Document Parsers (Docling, Marker)                        │
│ - Repositories (Storage for agents, data, extractions)      │
│ - MLflow Tracking, DSPy Optimization Wrappers               │
│ - Config (Settings, logging, environments)                  │
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
│   ├── extract_batch.py       # Batch extraction workflow
│   └── parse_documents.py     # Document parsing workflow
├── services/
│   ├── agent_manager.py       # Agent lifecycle management
│   ├── dataset_builder.py     # Dataset creation
│   ├── experiment_tracker.py  # MLflow tracking facade
│   └── data_validator.py      # Data validation service
└── dto/                       # Request/Response DTOs (defined in use_case files)
    # Note: DTOs are defined inline in their respective use case modules
    # - OptimizeAgentRequest/Response in optimize_agent.py
    # - BatchPredictionRequest/Response in extract_batch.py
    # - ParseDocumentsRequest/Response in parse_documents.py
```

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
├── entities/
│   ├── document.py      # ProcessedDocument, DocumentMetadata
│   ├── experiment.py    # Base Experiment class
│   └── extraction.py    # ExtractionResult, ExtractionOutput
├── tasks/
│   ├── base.py          # TaskDefinition ABC
│   ├── registry.py      # TaskRegistry
│   └── nanozymes/       # Nanozyme task plugin
│       ├── models.py    # NanozymeExperiment
│       ├── signature.py # DSPy signature
│       ├── converters.py # CSV to experiment conversion
│       └── config.py    # Task-specific config
└── evaluation/
    ├── matcher.py       # ExperimentMatcher
    ├── metrics.py       # TaskMetric
    └── scoring.py       # Scoring algorithms
```

### 4. Cross-Cutting Modules

These modules support all layers and don't fit strictly into one layer.

**Shared** (`src/aee/shared/`):
- Custom exception hierarchy
- Utilities used across all layers
- No dependencies on other layers (dependency inversion)

#### Task Plugin System

The task plugin system is the core extensibility mechanism.

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
        # Checks name, signature, models, etc.
```

**TaskRegistry** (`registry.py`):
```python
class TaskRegistry:
    def register(self, task: TaskDefinition, validate: bool = True):
        """Register a task with validation."""

    def get(self, task_name: str) -> TaskDefinition:
        """Get a registered task."""

    def list_tasks() -> List[TaskDefinition]:
        """List all registered tasks."""

# Global registry
_registry = TaskRegistry()

def get_task(name: str) -> TaskDefinition:
    return _registry.get(name)
```

**Example Task Plugin** (`nanozymes/`):
```python
class NanozymeTask(TaskDefinition):
    @property
    def name(self) -> str:
        return "nanozymes"

    @property
    def description(self) -> str:
        return "Extract nanozyme experiment data from scientific papers"

    @property
    def signature(self):
        return NanozymeSignature

    @property
    def output_model(self):
        return NanozymeExtractionOutput

    @property
    def experiment_model(self):
        return NanozymeExperiment

    @property
    def row_converter(self):
        return row_to_nanozyme

    @property
    def compare_fields(self) -> List[str]:
        return ["formula", "activity", "length"]

# Auto-register on import
register_task(NanozymeTask())
```

### 4. Infrastructure Layer (`src/aee/infrastructure/`)

**Purpose**: External integrations and technical implementations. Depends on external libraries and services.

**Structure**:
```
infrastructure/
├── agents/
│   ├── extractor.py   # UniversalExtractor DSPy module
│   └── __init__.py    # Agent implementations
├── cache/
│   └── __init__.py    # LLM response cache configuration
├── config/
│   ├── settings.py    # Pydantic settings management
│   ├── logging.py     # Logging configuration
│   ├── environments.py # Environment-specific settings
│   └── __init__.py
├── llm/
│   ├── provider.py    # OllamaLM, create_lm, setup functions
│   └── __init__.py    # LLM provider factory
├── parsers/
│   ├── base.py        # BaseParser ABC
│   ├── parsers.py     # DoclingParser, MarkerParser
│   ├── cleaning.py    # TextCleaner
│   └── __init__.py    # get_parser()
├── storage/
│   ├── agents.py      # AgentRepository with metadata
│   ├── ground_truth.py # GroundTruthRepository
│   ├── extractions.py  # ExtractionRepository
│   ├── documents.py    # DocumentRepository
│   ├── splits.py       # DataSplitRepository
│   └── __init__.py
├── optimization/
│   └── __init__.py    # MIPROv2 optimization wrappers
└── tracking/
    └── __init__.py    # MLflow integration
```

#### Repository Pattern

Repositories provide clean data access interfaces.

**Example - AgentRepository**:
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
            ├──> ParseDocumentsUseCase
            │       │
            │       ├──> get_parser("docling") [Infrastructure]
            │       ├──> DocumentRepository [Infrastructure]
            │       └──> ProcessedDocument [Domain]
            │
            └──> Display results to user
```

### Example 2: Agent Optimization

```
User Command
    │
    └──> CLI (optimize.py)
            │
            ├──> OptimizeAgentUseCase
            │       │
            │       ├──> DataValidator [Application] ← Pre-flight check
            │       │       ├──> DataSplitRepository [Infrastructure]
            │       │       └──> GroundTruthRepository [Infrastructure]
            │       │
            │       ├──> DatasetBuilder [Application]
            │       │       ├──> GroundTruthRepository [Infrastructure]
            │       │       ├──> DocumentRepository [Infrastructure]
            │       │       └──> TaskDefinition [Domain]
            │       │
            │       ├──> MIPROv2 Optimizer [Infrastructure]
            │       │       └──> DSPy + LLM Provider
            │       │
            │       ├──> AgentManager [Application]
            │       │       └──> AgentRepository [Infrastructure]
            │       │
            │       └──> ExperimentTracker [Application]
            │               └──> MLflow [Infrastructure]
            │
            └──> Display results to user
```

### Example 3: Adding a New Task

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

1. **Settings Classes** (`infrastructure/config/settings.py`):
   - Type-safe configuration with Pydantic
   - Hierarchical structure: project, paths, llm, optimization, parsing, task
   - Environment-specific configs via `environments.py`
   - Validation built-in
   - Python 3.8+ compatible type hints

2. **Environment Variables**:
   - Override any configuration setting
   - Nested settings use double underscore: `LLM__STUDENT__MODEL`
   - Key variables:
     - `LLM__STUDENT__OLLAMA__OLLAMA_BASE_URL` - Ollama server URL
     - `LLM__STUDENT__NON_OLLAMA__API_KEY` - API key for non-Ollama LLMs
     - `LLM__STUDENT__USE_OLLAMA` - Toggle between Ollama and API providers
     - `PROJECT__LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
     - `PATHS__AGENTS_DIR` - Override agents directory
     - `OPTIMIZATION__NUM_TRIALS` - Number of optimization trials

3. **YAML Config Files** (`config/`):
   - `default.yaml` - Production configuration with all required fields explicitly set
   - `default_fast.yaml` - Fast optimization settings (fewer trials)
   - Custom configs can be specified via `--config` flag

**Configuration Precedence** (highest to lowest):
1. Environment variables (`.env` file)
2. Custom YAML file (if specified via --config)
3. Default YAML file (`config/default.yaml`)
4. Internal defaults (minimal - only for safe infrastructure paths)

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

## Performance Considerations

1. **Lazy Loading**: Modules loaded only when needed
2. **Caching**: DSPy cache for LLM calls via `src/aee/infrastructure/cache/`
   - Memory cache for session-level caching
   - Disk cache for persistent caching across runs
   - Configurable via `enable_cache` in LLM settings
   - Global state managed by `dspy.configure_cache()`
3. **Batch Processing**: Efficient batch operations in repositories
4. **Streaming**: Large file processing uses streaming when possible
5. **Type Annotations**: Modern Python 3.8+ annotations for better performance
6. **Connection Pooling**: Reuse LLM connections for multiple requests

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
