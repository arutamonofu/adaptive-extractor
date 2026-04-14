# MIPROv2: Detailed Analysis of the Prompt Optimization Algorithm

**Version:** DSPy MIPROv2 (dspy/teleprompt/mipro_optimizer_v2.py)
**Analysis Date:** 2026-03-10

---

## Overview

**MIPROv2** (Model-based Instruction and Prompt Optimization) is a teleprompter from the DSPy library that automatically optimizes instructions and few-shot demonstrations for language models using **Bayesian optimization**.

### Three Main Stages

```
compile() →
  1. _bootstrap_fewshot_examples()      # Bootstrap few-shot examples
  2. _propose_instructions()            # Generate instruction candidates
  3. _optimize_prompt_parameters()      # Bayesian Optimization with Optuna
```

---

## MIPROv2 Architecture

### Classes and Components

| Component | File | Description |
|-----------|------|-------------|
| `MIPROv2` | `dspy/teleprompt/mipro_optimizer_v2.py:69` | Main optimizer |
| `GroundedProposer` | `dspy/propose/grounded_proposer.py:435` | Instruction generator |
| `GenerateModuleInstruction` | `dspy/propose/grounded_proposer.py:270` | Module instruction generation |
| `DescribeProgram` | `dspy/propose/grounded_proposer.py:33` | Signature for program description |
| `DescribeModule` | `dspy/propose/grounded_proposer.py:52` | Signature for module description |
| `DatasetDescriptor` | `dspy/propose/dataset_summary_generator.py:24` | Signature for dataset description |

---

## Step 1: Bootstrap Few-Shot Examples

**Method:** `_bootstrap_fewshot_examples()` (lines 770-850)

### Purpose

Creating sets of demonstration examples (input → output) for:
- Using as few-shot examples in the prompt
- Informing instruction generation

### Algorithm

```python
demo_candidates = create_n_fewshot_demo_sets(
    student=program,
    num_candidate_sets=num_fewshot_candidates,
    trainset=trainset,
    max_labeled_demos=LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT,  # 0
    max_bootstrapped_demos=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,  # 3
    metric=self.metric,
    teacher=teacher,  # TeacherWrapper
    teacher_settings=self.teacher_settings,
    seed=seed,
    rng=self.rng,
)
```

### Process

1. **Initialization:** Create `num_fewshot_candidates` empty sets
2. **Bootstrap:** For each example from trainset:
   - Teacher model generates the correct answer
   - Example with answer is added to the set
3. **Validation:** Examples are filtered by quality metric
4. **Result:** `demo_candidates[predictor_i][demo_set_i][example_i]`

### Constants

```python
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3  # Examples in context
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0       # No manual labeling
```

---

## Step 2: Generate Instruction Candidates

**Method:** `_propose_instructions()` (lines 852-920)

### Purpose

Generate `num_instruct_candidates` alternative instructions for each program predictor using:
- Dataset summary
- Program description
- Module description
- Few-shot examples
- Random tips

### GroundedProposer Configuration

```python
proposer = GroundedProposer(
    program=program,
    trainset=trainset,
    prompt_model=self.prompt_model,      # teacher_lm for generation
    view_data_batch_size=10,
    program_aware=True,                  # Describe program
    use_dataset_summary=True,            # Describe dataset
    use_task_demos=True,                 # Use few-shot
    num_demos_in_context=3,              # 3 examples in context
    use_tip=True,                        # Use tips
    set_tip_randomly=True,               # Random tip selection
    use_instruct_history=False,          # No instruction history
    set_history_randomly=False,
    verbose=self.verbose,
    rng=self.rng,
    init_temperature=self.init_temperature,
)
```

### 2.1 Dataset Summary Generation

**File:** `dspy/propose/dataset_summary_generator.py`

#### Signature: DatasetDescriptor

```python
class DatasetDescriptor(dspy.Signature):
    """Given several examples from a dataset please write observations
    about trends that hold for most or all of the samples.
    Some areas you may consider in your observations: topics, content,
    syntax, conciseness, etc.
    It will be useful to make an educated guess as to the nature of the
    task this dataset will enable. Don't be afraid to be creative"""

    examples = dspy.InputField(desc="Sample data points from the dataset")
    observations = dspy.OutputField(desc="Somethings that holds true for most or all of the data you observed")
```

#### Algorithm: `create_dataset_summary()`

```python
def create_dataset_summary(trainset, view_data_batch_size, prompt_model):
    # Step 1: First view_data_batch_size examples
    upper_lim = min(len(trainset), view_data_batch_size)
    observation = dspy.Predict(DatasetDescriptor, n=1, temperature=1.0)(
        examples=order_input_keys_in_string(trainset[0:upper_lim].__repr__())
    )
    observations = observation["observations"]

    # Step 2: Iterative addition of observations by batches
    max_calls = 10
    calls = 0
    for b in range(view_data_batch_size, len(trainset), view_data_batch_size):
        calls += 1
        if calls >= max_calls:
            break
        upper_lim = min(len(trainset), b + view_data_batch_size)
        output = dspy.Predict(DatasetDescriptorWithPriorObservations, n=1, temperature=1.0)(
            prior_observations=observations,
            examples=order_input_keys_in_string(trainset[b:upper_lim].__repr__())
        )
        # Update observations

    # Step 3: Final summarization
    summary = dspy.Predict(ObservationSummarizer, n=1, temperature=1.0)(
        observations=observations
    )
    return summary.summary
```

#### Example Output (from logs)

```
[[ ## observations ## ]]
- The documents are scientific research articles, most of which follow
  a conventional structure: Title → Author list → Abstract → Introduction
  → Experimental/Materials & Methods → Results and Discussion → Conclusions
  → Acknowledgements → References
- Each article includes a DOI, citation details and a statement about
  the licensing (e.g., Creative Commons)
- Chemical formulas are written with HTML‑style tags (<sub> for subscripts,
  <sup> for superscripts) and occasionally with LaTeX notation ($...$)
- The core experimental focus is on nanomaterials (e.g., MoO₃ nanobelts,
  Co₃O₄ nanocrystals, AuNC‑Cu²⁺ complexes) that are characterized as
  peroxidase‑like, catalase‑like, or oxidase‑like nanozymes
- Typical characterization techniques listed include XRD, TEM/HR‑TEM,
  STEM, XPS, EPR, Raman, UV/Vis‑NIR spectroscopy, FT‑IR, ICP‑MS, DLS
- Kinetic data are repeatedly reported: Michaelis–Menten parameters
  (Kₘ, Vₘₐₓ, K_cat, K_cat/Kₘ) for substrates such as TMB, H₂O₂, or glucose
[[ ## completed ## ]]

[[ ## summary ## ]]
The dataset comprises scientific articles on nanomaterial‑based nanozymes
that follow a standard research‑article format and include comprehensive
metadata (DOI, citations, licensing). Each paper details extensive
physicochemical characterizations (XRD, TEM, XPS, spectroscopy, DFT) and
reports kinetic parameters (Kₘ, Vₘₐₓ, K_cat) for peroxidase‑, catalase‑,
or oxidase‑like activities, often under varied pH, temperature, and
photo‑enhanced conditions, with colorimetric assay results (e.g., TMB
oxidation) and accompanying biocompatibility/in‑vivo studies.
[[ ## completed ## ]]
```

---

### 2.2 Program Source Code Extraction

**File:** `dspy/propose/utils.py`, function `get_dspy_source_code()`

#### Algorithm

```python
def get_dspy_source_code(module):
    header = []
    base_code = ""

    # Step 1: Extract the module's own code (except Predict/ChainOfThought)
    if not type(module).__name__ == "Predict" and not type(module).__name__ == "ChainOfThought":
        try:
            base_code = inspect.getsource(type(module))
        except TypeError:
            # For Jupyter notebook
            obj = type(module)
            cell_code = "".join(inspect.linecache.getlines(new_getfile(obj)))
            class_code = extract_symbols(cell_code, obj.__name__)[0][0]
            base_code = str(class_code)

    # Step 2: Recursively traverse module attributes
    completed_set = set()
    for attribute in module.__dict__.keys():
        try:
            iterable = iter(getattr(module, attribute))
        except TypeError:
            iterable = [getattr(module, attribute)]

        for item in iterable:
            # Skip non-hashable objects (module history)
            try:
                hash(item)
            except TypeError:
                continue

            # Step 3: Extract signature (signatures with instructions)
            if isinstance(item, Parameter):
                if hasattr(item, "signature") and item.signature is not None:
                    sig_name = item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig"
                    if sig_name not in completed_set:
                        try:
                            header.append(inspect.getsource(item.signature))
                        except (TypeError, OSError):
                            header.append(str(item.signature))
                        completed_set.add(sig_name)

            # Step 4: Recursively process nested dspy.Module
            if isinstance(item, dspy.Module):
                code = get_dspy_source_code(item).strip()
                if code not in completed_set:
                    header.append(code)
                    completed_set.add(code)

            completed_set.add(item)

    return "\n\n".join(header) + "\n\n" + base_code
```

#### What is Extracted

| Component | Extraction Method | Example |
|-----------|-------------------|---------|
| **Signature class** | `inspect.getsource(item.signature)` | `StringSignature(document_text -> reasoning, extracted_data instructions="...")` |
| **Module class** | `inspect.getsource(type(module))` | `class UniversalExtractor(BaseAgent, dspy.Module): ...` |
| **Nested modules** | Recursive call | All program submodules |
| **Signature fields** | Via `__pydantic_parent_namespace__` | `document_text = Field(annotation=str required=True)` |

#### Example Output (from logs)

```python
SOURCE CODE: StringSignature(document_text -> reasoning, extracted_data
    instructions="You are helpful assistant in chemistry, specializing in nanozymes. Your task is to analyze scientific articles and extract detailed information about various experiments with nanozymes..."
    document_text = Field(annotation=str required=True json_schema_extra={'desc': 'Full text content of the scientific article or document.', '__dspy_field_type': 'input', 'prefix': 'Document Text:'})
    reasoning = Field(annotation=str required=True json_schema_extra={'prefix': "Reasoning: Let's think step by step in order to", 'desc': '${reasoning}', '__dspy_field_type': 'output'})
    extracted_data = Field(annotation=ExtractionOutput required=True json_schema_extra={'desc': 'Extracted nanozymes experiments as structured data.', '__dspy_field_type': 'output', 'prefix': 'Extracted Data:'})
)

class UniversalExtractor(BaseAgent, dspy.Module, metaclass=UniversalExtractorMeta):
    """Task-agnostic extraction agent.

    Wraps a specific task signature (e.g., Nanozymes) with Chain-of-Thought reasoning.
    """

    def __init__(self, signature_class: Type[dspy.Signature]):
        """Initialize the UniversalExtractor.

        Args:
            signature_class: The DSPy signature defining input/output fields and instructions.
        """
        BaseAgent.__init__(self)
        dspy.Module.__init__(self)
        self.prog = dspy.ChainOfThought(signature_class)

    def forward(self, document_text: str) -> dspy.Prediction:
        """Execute the extraction pipeline.

        Args:
            document_text: The full content of the document (Markdown/HTML hybrid).

        Returns:
            dspy.Prediction: Contains 'reasoning' (str) and 'extracted_data' (Pydantic model).
        """
        return self.prog(document_text=document_text)

    def save(self, path: str) -> None:
        """Save the agent to a file."""
        dspy.Module.save(self, path)

    def load(self, path: str) -> None:
        """Load the agent from a file."""
        dspy.Module.load(self, path)
```

---

### 2.3 Program Description Generation

**Signature:** `DescribeProgram` (lines 33-50)

```python
class DescribeProgram(dspy.Signature):
    """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work."""

    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.OutputField(
        desc="Describe what task the program is designed to solve, and how it goes about solving this task.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )
```

#### Algorithm (from `forward()` method of `GenerateModuleInstruction`)

```python
if self.program_aware:
    program_description = strip_prefix(
        self.describe_program(
            program_code=self.program_code_string,
            program_example=task_demos,
        ).program_description
    )
```

#### What is Analyzed

| Input | Source | Content |
|-------|--------|---------|
| `program_code` | `get_dspy_source_code()` | Source code of Signature + Module |
| `program_example` | `task_demos` | Few-shot examples (input/output) |

#### What is Generated

**Output:** Text description including:
- What task the program solves
- How it works (step-by-step)
- Why this structure is useful

#### Example Output (from logs)

```
[[ ## program_description ## ]]
**Task the program is designed to solve**

The program is a domain‑specific information extraction pipeline for
scientific literature about nanozyme experiments. Its goal is to read
the full text of a chemistry article, locate every experiment that
reports a nanozyme (nanoparticle with enzyme‑like activity), and output
a structured representation of each experiment that includes all relevant
physicochemical and kinetic parameters (formula, crystal system, size,
surface chemistry, synthesis polymer, surfactant, Michaelis‑Menten
constants, reaction substrates, concentration ranges, pH, temperature, etc.).

The extraction must obey strict rules:
* Every experiment is reported independently – no cross‑experiment
  references are allowed in the output.
* Numerical values must retain the original units.
* Kinetic parameters (Km, Vmax, etc.) must be linked to the correct
  reaction track (whether H₂O₂ is the substrate or the co‑substrate).
* Only experiments that contain a kinetic assay are kept; all other
  listed attributes are still required.
* The model must not rely on any pre‑existing knowledge about nanozymes
  – it must base its answer solely on the supplied document.

**How the program works**

1. Signature definition (StringSignature)
   Defines the interface for the extraction task.
   - document_text – input field containing the raw article.
   - reasoning – output field where the model records a chain‑of‑thought
     explanation ("Let's think step by step…") that justifies its
     extraction decisions.
   - extracted_data – output field typed as ExtractionOutput, a
     Pydantic‑style model that will hold the list of extracted
     nanozyme‑experiment objects.
   - The instructions string embeds a detailed prompt that tells the
     model exactly what to look for, the list of attributes to capture,
     the logical rules for assigning reaction types, and the critical
     extraction rules (units, separate tracks, etc.).

2. UniversalExtractor class
   - Inherits from BaseAgent (provides generic agent utilities) and
     dspy.Module (DSPy's modular component system).
   - Its constructor receives the signature class (here the
     StringSignature) and builds a Chain‑of‑Thought (CoT) program
     (self.prog = dspy.ChainOfThought(signature_class)). The CoT wrapper
     automatically prepends the "Reasoning:" prefix, asks the language
     model to first think through the document, then produce the final
     structured output.
   - The forward method is the entry point: it receives the raw
     document_text, forwards it to the CoT program, and returns a
     dspy.Prediction containing both the reasoning string and the
     extracted_data model.
   - save and load methods allow persisting the configured extractor
     to disk and re‑loading it later, leveraging DSPy's built-in
     serialization.

3. Execution flow
   - Input: a scientific article (plain text, Markdown, or HTML).
   - Prompt generation: DSPy combines the user-provided instructions
     with the article text and the CoT template, forming a single prompt
     sent to the underlying LLM.
   - Chain‑of‑Thought reasoning: the LLM first writes a step‑by‑step
     rationale (e.g., "Identify each experimental section, locate the
     formula Fe₃O₄, read the size …, decide whether H₂O₂ is substrate …").
   - Extraction: after the reasoning, the LLM emits a JSON‑compatible
     representation that matches ExtractionOutput, filling in every
     required field for each experiment.
   - Output: the Prediction object is returned; downstream code can
     read prediction.reasoning for auditability and
     prediction.extracted_data for downstream analysis.

**Why this design is useful**

- Task‑agnostic wrapper: UniversalExtractor can be reused for other
  domains by swapping the signature (different instructions and output
  schema) while keeping the same CoT infrastructure.
- Transparency: The explicit reasoning field lets users verify that the
  model followed the prescribed logic (important for scientific data
  extraction).
- Strict compliance: By embedding the detailed extraction rules directly
  in the prompt, the system enforces unit fidelity, correct reaction‑track
  assignment, and separation of experiments, which are critical for
  high‑quality curated datasets.

In summary, the program implements a specialized, rule‑driven
information‑extraction pipeline for nanozyme experimental data,
leveraging DSPy's signature‑based prompting and chain‑of‑thought
reasoning to produce accurate, auditable, and structured outputs
from raw scientific texts.
[[ ## completed ## ]]
```

---

### 2.4 Module Description Generation

**Signature:** `DescribeModule` (lines 52-72)

```python
class DescribeModule(dspy.Signature):
    """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline."""

    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.InputField(
        desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )
    module = dspy.InputField(
        desc="The module in the program that we want to describe.",
        prefix="MODULE:",
    )
    module_description = dspy.OutputField(
        desc="Description of the module's role in the broader program.",
        prefix="MODULE DESCRIPTION:",
    )
```

#### Module Code Formation

```python
# From forward() method of GenerateModuleInstruction (lines 223-234)
inputs = []
outputs = []
for field_name, field in get_signature(program.predictors()[pred_i]).fields.items():
    dspy_field_type = field.json_schema_extra.get("__dspy_field_type")
    if dspy_field_type == "input":
        inputs.append(field_name)
    else:
        outputs.append(field_name)

module_code = f"{program.predictors()[pred_i].__class__.__name__}({', '.join(inputs)}) -> {', '.join(outputs)}"
# Example: "UniversalExtractor(document_text) -> reasoning, extracted_data"
```

#### Algorithm

```python
module_description = self.describe_module(
    program_code=self.program_code_string,
    program_description=program_description,
    program_example=task_demos,
    module=module_code,
    max_depth=10,
).module_description
```

#### Example Output (from logs)

```
[[ ## module_description ## ]]
The **Predict** module is the core execution component of the extraction
pipeline. It receives the raw document_text (the full scientific article)
as its sole input and forwards this text to the underlying Chain‑of‑Thought
program (self.prog).

The Chain‑of‑Thought wrapper first asks the language model to generate a
step‑by‑step reasoning trace that explains how it identified each nanozyme
experiment, interpreted the experimental parameters, and assigned reaction
tracks.

After the reasoning is produced, the model emits the structured extracted_data
that conforms to the ExtractionOutput schema, containing a separate object
for every nanozyme experiment with all required attributes (formula, activity,
size, kinetic constants, etc.).

Thus, the Predict module orchestrates the end‑to‑end transformation from
unstructured article text to a transparent reasoning narrative and a
fully‑structured, rule‑compliant data representation.
[[ ## completed ## ]]
```

---

### 2.5 TIPS (Generation Prompts)

**File:** `dspy/propose/grounded_proposer.py`, line 72

```python
TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
}
```

#### Random Tip Selection

```python
# From propose_instructions_for_program() (lines 383-393)
if self.set_tip_randomly:
    if self.verbose:
        print("Using a randomly generated configuration for our grounded proposer.")
    selected_tip_key = self.rng.choice(list(TIPS.keys()))
    selected_tip = TIPS[selected_tip_key]
    self.use_tip = bool(selected_tip)
    if self.verbose:
        print(f"Selected tip: {selected_tip_key}")
```

#### Influence on Generation

| Tip | Effect |
|-----|--------|
| `none` | No additional instructions |
| `creative` | Encourages creative, non-standard formulations |
| `simple` | Requires clarity and conciseness |
| `description` | Requires detail and informativeness |
| `high_stakes` | Adds a high-stakes scenario |
| `persona` | Adds a role ("You are a ...") |

**In your logs:**
```
Using a randomly generated configuration for our grounded proposer.
Selected tip: persona
```

---

### 2.6 New Instruction Generation

**Signature:** `GenerateSingleModuleInstruction` (dynamically created)

#### Dynamic Class Generation

```python
# Function generate_instruction_class() (lines 81-163)
def generate_instruction_class(
    use_dataset_summary=True,
    program_aware=True,
    use_task_demos=True,
    use_instruct_history=True,
    use_tip=True,
):
    class GenerateSingleModuleInstruction(dspy.Signature):
        """Use the information below to learn about a task that we are
        trying to solve using calls to an LM, then generate a new
        instruction that will be used to prompt a Language Model to
        better solve the task."""

        if use_dataset_summary:
            dataset_description = dspy.InputField(
                desc="A description of the dataset that we are using.",
                prefix="DATASET SUMMARY:",
            )

        if program_aware:
            program_code = dspy.InputField(
                format=str,
                desc="Language model program designed to solve a particular task.",
                prefix="PROGRAM CODE:",
            )
            program_description = dspy.InputField(
                desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
                prefix="PROGRAM DESCRIPTION:",
            )
            module = dspy.InputField(
                desc="The module to create an instruction for.",
                prefix="MODULE:",
            )
            module_description = dspy.InputField(
                desc="Description of the module to create an instruction for.",
                prefix="MODULE DESCRIPTION:",
            )

        task_demos = dspy.InputField(
            format=str,
            desc="Example inputs/outputs of our module.",
            prefix="TASK DEMO(S):",
        )

        if use_instruct_history:
            previous_instructions = dspy.InputField(
                format=str,
                desc="Previous instructions we've attempted, along with their associated scores.",
                prefix="PREVIOUS INSTRUCTIONS:",
            )

        basic_instruction = dspy.InputField(
            format=str,
            desc="Basic instruction.",
            prefix="BASIC INSTRUCTION:",
        )

        if use_tip:
            tip = dspy.InputField(
                format=str,
                desc="A suggestion for how to go about generating the new instruction.",
                prefix="TIP:",
            )

        proposed_instruction = dspy.OutputField(
            desc="Propose an instruction that will be used to prompt a Language Model to perform this task.",
            prefix="PROPOSED INSTRUCTION:",
        )

    return dspy.Predict(GenerateSingleModuleInstruction)
```

#### Instruction Generation Algorithm

```python
# From propose_instruction_for_predictor() (lines 417-437)

# Step 1: Create instruction history (if used)
instruction_history = create_predictor_level_history_string(
    program, pred_i, trial_logs, MAX_INSTRUCT_IN_HISTORY
)

# Step 2: Create instruction generator
instruction_generator = GenerateModuleInstruction(
    program_code_string=self.program_code_string,
    use_dataset_summary=self.use_dataset_summary,
    program_aware=self.program_aware,
    use_task_demos=self.use_task_demos and demo_candidates,
    use_instruct_history=self.use_instruct_history and instruction_history,
    use_tip=self.use_tip,
    verbose=self.verbose
)

# Step 3: Copy model with unique rollout_id (cache bypass)
rollout_lm = self.prompt_model.copy(
    rollout_id=self.rng.randint(0, 10**9),
    temperature=self.init_temperature,
)

# Step 4: Generate instruction
with dspy.context(lm=rollout_lm):
    proposed_instruction = instruction_generator(
        demo_candidates=demo_candidates,
        pred_i=pred_i,
        demo_set_i=demo_set_i,
        program=program,
        data_summary=self.data_summary,
        previous_instructions=instruction_history,
        num_demos_in_context=self.num_demos_in_context,
        tip=tip,
    ).proposed_instruction

# Step 5: Clean prefix
return strip_prefix(proposed_instruction)
```

#### Task Demos Formation

```python
# From forward() method of GenerateModuleInstruction (lines 189-207)
def gather_examples_from_sets(candidate_sets, max_examples):
    """Helper function to gather up to augmented examples from given sets."""
    count = 0
    for candidate_set in candidate_sets:
        for example in candidate_set:
            if "augmented" in example.keys():
                fields_to_use = get_signature(program.predictors()[pred_i]).fields
                yield create_example_string(fields_to_use, example)
                count += 1
                if count >= max_examples:
                    return

if self.use_task_demos:
    # Combine current and adjacent sets
    adjacent_sets = (
        [demo_candidates[pred_i][demo_set_i]] +
        demo_candidates[pred_i][demo_set_i + 1:] +
        demo_candidates[pred_i][:demo_set_i]
    )

    # Gather examples up to the required count
    example_strings = gather_examples_from_sets(adjacent_sets, num_demos_in_context)
    task_demos = "\n\n".join(example_strings) + "\n\n"

# Default to no demos provided
if not task_demos.strip() or demo_set_i == 0:
    task_demos = "No task demos provided."
```

#### Example Output (from logs)

```
[[ ## proposed_instruction ## ]]
You are a **research‑oriented chemistry assistant specialized in nanozyme
experiments**. Your job is to read the full text of a scientific article
(plain text, Markdown, or HTML) and extract **every individual experiment
that reports kinetic data for a nanozyme**.

For each experiment you must produce **two separate, self‑contained objects**
(one for each reaction track) that include **all of the following fields**
exactly as they appear in the paper, preserving the original units:

| Field | Description |
|-------|-------------|
| **formula** | Chemical formula of the nanoparticle (e.g., `Fe3O4`). |
| **activity** | Enzyme‑like activity (peroxidase, oxidase, catalase, laccase, etc.). |
| **syngony** | Crystal system (cubic, hexagonal, tetragonal, monoclinic, orthorhombic, trigonal, amorphous, triclinic). |
| **size** | Length, width, depth, or single diameter/size (include unit, e.g., `12 nm`). |
| **surface_chemistry** | Surface coating or functionalization (default "naked" if none). |
| **polymer_used** | Polymer employed during synthesis (or "none"). |
| **surfactant** | Surfactant used (or "none"). |
| **molar_mass** | Molar mass of the nanozyme (include unit). |
| **reaction_track** | Either `H2O2_substrate` (H₂O₂ concentration varies, TMB constant) or `H2O2_cosubstrate` (TMB varies, H₂O₂ constant). |
| **reaction_type** | Exact substrate + co‑substrate notation as written in the article (e.g., `TMB+H2O2` or `H2O2+TMB`). |
| **C_min** | Minimum substrate concentration used for kinetic assay (mM). |
| **C_max** | Maximum substrate concentration used for kinetic assay (mM). |
| **C_const** | Constant concentration of the co‑substrate (mM). |
| **nanoparticle_concentration** | Concentration of nanozyme in the assay (mg/mL). |
| **Km** | Michaelis constant (include unit, typically mM). |
| **Vmax** | Maximum reaction rate (include unit, e.g., `µM s⁻¹`). |
| **temperature** | Temperature at which the assay was performed (°C). |
| **pH** | pH of the assay buffer. |
| **additional_notes** | Any other quantitative or qualitative details that are explicitly reported. |

### Extraction Rules
1. **One experiment = one nanozyme + one kinetic assay.** Do not merge
   data from different experiments; each object must be independent.
2. **Only include experiments that contain ki... [truncated]
```

---

### 2.7 Complete Instruction Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. get_dspy_source_code(module)                                 │
│    ├─ Signature: StringSignature(instructions="...")            │
│    │   - document_text: str -> reasoning: str                   │
│    │   - extracted_data: ExtractionOutput                       │
│    └─ Module: class UniversalExtractor(...)                     │
│        - __init__(signature_class)                              │
│        - forward(document_text) -> Prediction                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. DescribeProgram (LLM call, temperature=1.0)                  │
│    Input:                                                       │
│      - program_code: source code of Signature + Module          │
│      - program_example: few-shot examples (task_demos)          │
│    Output:                                                      │
│      - "domain-specific information extraction pipeline..."     │
│      - "The program reads chemistry articles and extracts..."   │
│      - "How it works: 1) Signature definition, 2) CoT..."       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. DescribeModule (LLM call, temperature=1.0)                   │
│    Input:                                                       │
│      - program_code: source code                                │
│      - program_description: from step 2                         │
│      - module: "UniversalExtractor(document_text) -> ..."       │
│    Output:                                                      │
│      - "Predict module is the core execution component..."      │
│      - "Receives document_text, forwards to ChainOfThought..."  │
│      - "Generates reasoning trace + structured output..."       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. GenerateSingleModuleInstruction (LLM call, temperature=1.0)  │
│    Input:                                                       │
│      - dataset_description: dataset analysis                    │
│      - program_code: source code                                │
│      - program_description: task description                    │
│      - module: predictor signature                              │
│      - module_description: module role                          │
│      - task_demos: few-shot examples (3 pcs.)                   │
│      - tip: "persona"                                           │
│      - basic_instruction: current instruction                   │
│    Output:                                                      │
│      - "You are a research-oriented chemistry assistant..."     │
│      - Table of 20 fields for extraction                        │
│      - Extraction Rules                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 3: Bayesian Optimization of Prompt Parameters

**Method:** `_optimize_prompt_parameters()` (lines 922+)

### Purpose

Search for the optimal combination of:
- Instructions for each predictor
- Few-shot sets for each predictor

using **Bayesian optimization** (Optuna TPE Sampler).

### Optuna Configuration

```python
import optuna

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(),  # Tree-structured Parzen Estimator
    direction="maximize"
)
```

### Parameter Space

```python
def _get_param_distributions(self, program, instruction_candidates, demo_candidates):
    from optuna.distributions import CategoricalDistribution

    param_distributions = {}

    for i in range(len(instruction_candidates)):
        # Instruction for predictor i
        param_distributions[f"{i}_predictor_instruction"] = CategoricalDistribution(
            range(len(instruction_candidates[i]))
        )

        # Few-shot set for predictor i (if available)
        if demo_candidates:
            param_distributions[f"{i}_predictor_demos"] = CategoricalDistribution(
                range(len(demo_candidates[i]))
            )

    return param_distributions
```

### Optimization Algorithm

```python
for trial_num in range(num_trials):
    # Step 1: Select parameter combination via Optuna
    trial = study.ask(param_distributions)

    # Step 2: Apply selected instructions and demos
    chosen_params, raw_chosen_params = _select_and_insert_instructions_and_demos(
        program, instruction_candidates, demo_candidates, trial, trial_logs, trial_num
    )

    # Step 3: Evaluate on minibatch (35 examples)
    score = evaluate_candidate_program(
        num_samples=minibatch_size,
        valset=valset,
        program=program,
        evaluate=evaluate,
        rng=self.rng
    ).score

    # Step 4: Update Optuna
    study.tell(trial, score)

    # Step 5: Periodic full evaluation
    if trial_num % minibatch_full_eval_steps == 0:
        best_score, best_program, total_eval_calls = _perform_full_evaluation(
            trial_num, adjusted_num_trials, param_score_dict,
            fully_evaled_param_combos, evaluate, valset, trial_logs,
            total_eval_calls, score_data, best_score, best_program,
            study, instruction_candidates, demo_candidates
        )
```

### Minibatch vs Full Evaluation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `minibatch_size` | 35 | Examples for fast trial evaluation |
| `minibatch_full_eval_steps` | 5 | Step for full evaluation on entire valset |

### Trial Logging

```python
trial_logs[trial_num] = {
    "score": score,
    "total_eval_calls_so_far": total_eval_calls,
    f"{i}_predictor_instruction": instruction_idx,
    f"{i}_predictor_demos": demos_idx,
}
```

---

## Configuration in AutoEvoExtractor Project

**File:** `src/aee/application/use_cases/optimize_agent.py`, lines 397-415

```python
teleprompter = MIPROv2(
    metric=metric,
    prompt_model=request.teacher_lm,      # gpt-oss:120b for instruction generation
    task_model=request.student_lm,        # for executing trials
    teacher_settings={"lm": request.teacher_lm},
    auto=None,  # Intentionally None: custom values
    num_candidates=request.num_candidates,    # 3
    max_bootstrapped_demos=request.max_bootstrapped_demos,
    max_labeled_demos=request.max_labeled_demos,
    seed=request.seed,
    init_temperature=request.init_temperature,  # 1.0
    verbose=request.verbose,
    metric_threshold=request.metric_threshold,
    num_threads=1
)

# Run optimization
optimized_agent = teleprompter.compile(
    base_agent,
    trainset=trainset,
    valset=valset,
    teacher=teacher_module,
    num_trials=request.num_trials,
    max_bootstrapped_demos=request.max_bootstrapped_demos,
    max_labeled_demos=request.max_labeled_demos,
    seed=request.seed,
    minibatch=request.minibatch,
    minibatch_size=35,
    minibatch_full_eval_steps=5,
    program_aware_proposer=True,
    data_aware_proposer=True,
    view_data_batch_size=10,
    tip_aware_proposer=True,
    fewshot_aware_proposer=True,
)
```

---

## Key MIPROv2 Constants

```python
# From mipro_optimizer_v2.py (lines 36-42)
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3  # Examples in context
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0       # No manual labeling
MIN_MINIBATCH_SIZE = 50                       # Minimum minibatch size

AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}

# From grounded_proposer.py (line 68)
MAX_INSTRUCT_IN_HISTORY = 5  # Maximum instructions in history
```

---

## Design Decisions and Rationale

### 1. Program-Aware Proposer

**Why:** LLM understands the **task context**, not just an abstract instruction.

**What it provides:**
- Instruction is generated with understanding of **architecture** (ChainOfThought, Signature)
- LLM sees **data flows** (input → reasoning → extracted_data)
- Generated instruction considers **system constraints**

### 2. Module Description

**Why:** Understanding the **role of a specific predictor** in the pipeline.

**What it provides:**
- Instruction becomes **module-specific**, not general
- LLM understands what **output is expected** (reasoning + extracted_data)
- Can optimize instructions for **multi-module programs**

### 3. Dataset Summary

**Why:** Task contextualization based on **patterns in data**.

**What it provides:**
- Instruction reflects **real dataset characteristics**
- LLM understands **data format** (scientific articles, HTML tags, formulas)
- **Domain-relevant** instructions are generated

### 4. Random Tips

**Why:** Adding **diversity** to instruction generation.

**What it provides:**
- Different instruction candidates have different **stylistic accents**
- **Instruction space coverage** is increased
- Non-obvious but effective formulations are found

### 5. Rollout ID for Cache Bypass

**Why:** Ensuring **diversity** during generation.

```python
rollout_lm = self.prompt_model.copy(
    rollout_id=self.rng.randint(0, 10**9),  # Unique ID
    temperature=self.init_temperature,       # 1.0 for diversity
)
```

**What it provides:**
- Each instruction generation is **unique** (not cached)
- At temperature 1.0, **maximum diversity** is ensured

---

## Glossary

| Term | Definition |
|------|------------|
| **Signature** | DSPy class defining input/output and instruction for LLM |
| **Predictor** | DSPy module (e.g., Predict, ChainOfThought) with signature |
| **Bootstrap** | Generation of correct answers by teacher model for trainset |
| **Few-shot demos** | Input/output examples for use in prompt |
| **Grounded Proposing** | Instruction generation based on code, data, and examples |
| **TPE Sampler** | Tree-structured Parzen Estimator — Bayesian optimization algorithm |
| **Minibatch evaluation** | Fast evaluation on valset subset (35 examples) |
| **Full evaluation** | Full evaluation on entire valset |

---

## Source Code References

| Component | Path |
|-----------|------|
| MIPROv2 | `dspy/teleprompt/mipro_optimizer_v2.py` |
| GroundedProposer | `dspy/propose/grounded_proposer.py` |
| Dataset Summary | `dspy/propose/dataset_summary_generator.py` |
| Utils | `dspy/propose/utils.py` |
| Teleprompt Utils | `dspy/teleprompt/utils.py` |
| Optimize Agent | `src/aee/application/use_cases/optimize_agent.py` |

---

## Appendices

### A. Example Complete Prompt for Instruction Generation

```
DATASET SUMMARY:
The dataset comprises scientific articles on nanomaterial-based nanozymes...

PROGRAM CODE:
StringSignature(document_text -> reasoning, extracted_data
    instructions="You are helpful assistant in chemistry..."
    ...
)

class UniversalExtractor(BaseAgent, dspy.Module):
    ...

PROGRAM DESCRIPTION:
The program is a domain-specific information extraction pipeline...

MODULE:
UniversalExtractor(document_text) -> reasoning, extracted_data

MODULE DESCRIPTION:
The Predict module is the core execution component...

TASK DEMO(S):
Document Text: <article about Fe3O4 nanozymes>
Reasoning: Let's think step by step...
Extracted Data: {"experiments": [...]}

TIP:
Include a persona that is relevant to the task in the instruction
(ie. "You are a ...")

BASIC INSTRUCTION:
You are helpful assistant in chemistry, specializing in nanozymes...

PREVIOUS INSTRUCTIONS:
(empty, since use_instruct_history=False)

PROPOSED INSTRUCTION:
[Generated by LLM]
```

### B. Data Flow in MIPROv2

```
┌──────────────────────────────────────────────────────────────┐
│                    MIPROv2.compile()                         │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Step 1: Bootstrap│ │ Step 2: Propose  │ │ Step 3: Optimize │
│ Few-Shot         │ │ Instructions     │ │ Parameters       │
│                  │ │                  │ │                  │
│ • Teacher model  │ │ • Dataset        │ │ • Optuna TPE     │
│ • Trainset →     │ │   Summary        │ │ • Minibatch      │
│   demo_candidates│ │ • Program        │ │   evaluation     │
│ • N sets of      │ │   Description    │ │ • Full eval      │
│   demonstrations │ │ • Module         │ │   every 5 steps  │
│                  │ │   Description    │ │ • Best           │
│                  │ │ • Tips (persona) │ │   combination    │
│                  │ │ • N instructions │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### C. Instruction Generation Timeline

```
Time     Component                    Action
────     ─────────                    ──────
00:00    MIPROv2                      Start _propose_instructions()
00:01    GroundedProposer.__init__    Initialize proposer
00:02    create_dataset_summary()     Dataset analysis (batch by batch)
00:15    get_dspy_source_code()       Extract program source code
00:16    DescribeProgram (LLM)        Generate program description
00:30    DescribeModule (LLM)         Generate module description
00:35    select_tip()                 Select random tip (persona)
00:36    GenerateSingleModule...      Assemble complete prompt
00:37    LLM (gpt-oss:120b)           Generate new instruction
00:50    strip_prefix()               Clean prefix
00:51    return                       Return instruction
```

---

## Semantic Judge Model Usage

### Model Roles in AutoEvoExtractor

The system uses two distinct LLM models with different responsibilities:

| Model | Role | Usage |
|-------|------|-------|
| **student_llm** | Semantic Judge | Evaluates extraction quality during evaluation |
| **teacher_llm** | MIPROv2 Optimization | Generates instruction candidates, bootstraps demonstrations |

### Semantic Judge (student_llm)

The semantic judge uses the **student model** to evaluate discrepancies between extracted data and ground truth during evaluation. It:

- Called by `ExperimentMatcher._call_semantic_judge()` when strict comparison fails
- Analyzes JSON representations of predicted vs. ground truth experiments
- Returns YES/NO verdicts on whether discrepancies represent actual errors or acceptable variations
- Falls back to strict penalties if student_llm is unavailable or fails

**Flow:** `TaskMetric → ExperimentMatcher → Semantic Judge (student_llm)`

### MIPROv2 Optimization (teacher_llm)

The teacher model is used exclusively during the optimization phase:

- **prompt_model**: Generates instruction candidates via `GroundedProposer`
- **task_model**: Not used directly (student runs trials)
- **teacher**: Bootstraps few-shot demonstrations via `TeacherWrapper`

**Flow:** `MIPROv2 → GroundedProposer (teacher_lm) + TeacherWrapper (teacher_lm)`

### Architecture Rationale

Using the student model for semantic judgment ensures:
- Evaluation reflects the actual capabilities of the deployed model
- No dependency on teacher model availability during inference/evaluation
- Consistent scoring across optimization trials and final validation
