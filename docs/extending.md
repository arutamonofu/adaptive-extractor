# Extending Adaptive Extractor

This guide provides step-by-step instructions on how to extend the Adaptive Extractor system with new task schemas, LLM backend providers, and document parsers.

---

## 1. Adding a New Task Schema

Adaptive Extractor dynamically configures its extraction target based on YAML task schemas located in `config/tasks/`.

### Step 1: Create the Task Directory
Create a new directory under `config/tasks/` named after your task (use lowercase and underscores):
```bash
mkdir -p config/tasks/my_new_task
```

### Step 2: Define the Schema (`schema.yaml`)
Create `config/tasks/my_new_task/schema.yaml` defining the fields, types, and constraints for extraction.

```yaml
name: my_new_task
description: "Extraction of catalyst parameters and reactor conditions"

fields:
  material:
    type: str
    description: "The chemical formula of the catalyst material"
  temperature:
    type: float
    description: "Reaction temperature in degrees Celsius"
  pressure:
    type: float
    description: "Reactor pressure in bar"
  yield:
    type: float
    description: "Product yield in percent"
  selectivity:
    type: float
    description: "Product selectivity in percent"
```

#### Supported Field Types and Settings:
- `type`: `str`, `float`, `int`, `bool`, `list[str]`, `list[float]`, `list[int]`.
- `description`: A clear, textual explanation used by the LLM to understand what it should extract.

### Step 3: Define the Baseline Prompt (`baseline_instruction.txt`)
Create `config/tasks/my_new_task/baseline_instruction.txt` containing the initial system instructions and guidelines for the extraction agent. This text will be loaded as the base instruction during optimization.

```text
You are an information extraction system specialized in chemical catalysis.
Your task is to extract structured experimental data about catalyst materials and reaction conditions.

CRITICAL EXTRACTION POLICY:
1. Extract values exactly as reported in the text.
2. Do not normalize or convert units.
3. If a value is not explicitly mentioned, return null.
```

### Step 4: Activate the Task
Update the active task name in `config/core.yaml`:
```yaml
task:
  name: my_new_task
```

---

## 2. Adding a New LLM Backend Provider

LLM backends are decoupled from external API wrappers to prevent JSON serialization errors during optimization. All providers must implement the `LMProvider` Protocol.

### Step 1: Implement the `LMProvider` Class
Create your provider class. It must implement the `LMProvider` PEP-544 Protocol methods (like `__call__`, `deepcopy`, `reset_copy`, and `clear_history`).

Open [provider.py](file:///home/arutamonofu/dev/study/adaptive-extractor/src/ae/core/llm/provider.py) and implement your class, for example, a custom provider for an internal inference service:

```python
class MyCustomLM(BaseLMProvider):
    """Custom LLM provider calling an internal API endpoint."""

    def __init__(self, config: LLMInstanceConfig, circuit_breaker: Optional[CircuitBreaker] = None):
        super().__init__(config, circuit_breaker)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        # Initialize your custom API client here

    def _call_api(self, prompt: str, **kwargs) -> str:
        # Perform the actual HTTP post request or client call
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            **kwargs
        }
        # e.g., response = requests.post("https://my-internal-llm/generate", json=payload)
        # return response.json()["text"]
        return "Dummy output"
```

### Step 2: Register the Provider in the Registry
In `src/ae/core/llm/provider.py`, register your provider with the global `LLM_PROVIDER_REGISTRY`:

```python
# Register default providers
LLM_PROVIDER_REGISTRY.register("ollama", OllamaLM)
LLM_PROVIDER_REGISTRY.register("api", OpenRouterLM)
LLM_PROVIDER_REGISTRY.register("custom", MyCustomLM)  # Registered!
```

### Step 3: Configure the Provider in `core.yaml`
Update your active models in `config/core.yaml` to use your new provider:

```yaml
llm:
  student:
    provider: custom
    model: my-custom-model-7b
    temperature: 0.1
  teacher:
    provider: custom
    model: my-custom-model-70b
    temperature: 0.7
```

---

## 3. Adding a New Ingestion Document Parser

The ingestion pipeline converts raw scientific PDFs into structured Markdown documents under `data/interim/ingestion/`.

### Step 1: Implement the Parser Class
All parsers must inherit from `BaseParser` in `src/ae/ingestion/base_parser.py`.

Create your parser class (e.g., in a new file `src/ae/ingestion/custom_parser.py` or directly in `parsers.py`):

```python
from typing import Dict, Any
from pathlib import Path
from ae.ingestion.base_parser import BaseParser

class CustomPDFParser(BaseParser):
    """A parser that uses a custom extraction library."""

    def __init__(self, config: Any):
        # config is an instance of your custom config class
        self.config = config

    def parse(self, pdf_path: Path) -> str:
        """Parse a PDF document and return structured Markdown text."""
        logger.info(f"Parsing {pdf_path} using CustomPDFParser")
        # Extract text/tables and format them into clean Markdown
        markdown_text = f"# Parsed {pdf_path.name}\n\nContent..."
        return markdown_text
```

### Step 2: Define a Configuration Schema
Add a Pydantic settings config class in `src/ae/core/config/settings.py` (or parser-specific config):
```python
from pydantic import BaseModel

class CustomParserConfig(BaseModel):
    enabled: bool = False
    api_endpoint: str = "http://localhost:8080"
```

### Step 3: Register the Parser in `ParserRegistry`
Open [parsers.py](file:///home/arutamonofu/dev/study/adaptive-extractor/src/ae/ingestion/parsers.py) and register your new parser:

```python
from ae.ingestion.custom_parser import CustomPDFParser, CustomParserConfig

# Register default parsers
PARSER_REGISTRY.register("gemini", GeminiParser, GeminiParserConfig)
PARSER_REGISTRY.register("gemini_visual", AEVisualParser, AEVisualParserConfig)
PARSER_REGISTRY.register("custom", CustomPDFParser, CustomParserConfig)  # Registered!
```

### Step 4: Configure Parser in `ingestion.yaml`
Add your new parser configurations in `config/ingestion.yaml`:
```yaml
parsing:
  active_parser: custom
  custom:
    api_endpoint: "http://localhost:8080"
```