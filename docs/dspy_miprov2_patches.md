# DSPy MIPROv2 Optimization Patches

To run prompt optimization successfully under various configurations (such as zero-shot optimization or dynamic signatures), DSPy MIPROv2 requires two critical patches.

---

## 1. Zero-Shot & Labeled-Only Optimization Bug Fix ([patch_dspy_mipro_zero_bootstrap.py](file:///home/arutamonofu/dev/study/adaptive-extractor/scripts/patch_dspy_mipro_zero_bootstrap.py))

### Problem
When setting `max_bootstrapped_demos = 0` (focusing on pure instruction tuning or labeled-only few-shot without auto-generated few-shot examples), the optimization process crashed or failed to include examples in the prompt templates.

### Root Cause
1. **Randrange Error**: In `dspy/teleprompt/utils.py`, when `max_bootstrapped_demos=0`, the code executed `rng.randint(min_num_samples, max_bootstrapped_demos)` which translates to `randint(1, 0)`, raising a `ValueError: empty range for randrange() (1, 0)`.
2. **Constants Override**: In `mipro_optimizer_v2.py`, internal constants were overriding the zero values, enforcing 3 bootstrapped demos.
3. **Demos Pollution**: Demos were not properly cleared from predictors after bootstrap compilation, polluting zero-shot runs.
4. **Proposer Filtering**: In `dspy/propose/grounded_proposer.py`, the `gather_examples_from_sets` helper only gathered examples if they contained the `"augmented"` key, which excluded hand-labeled examples from being used in the prompt proposer.

### Solution
The patch:
1. Modifies `utils.py` to skip bootstrapping entirely if `max_bootstrapped_demos <= 0` to prevent the `randint` error.
2. Corrects the constant check and clears the predictor demos in `mipro_optimizer_v2.py` for true zero-shot.
3. Modifies `grounded_proposer.py` to allow both bootstrapped (augmented) and hand-labeled examples to be used in meta-prompts.

---

## 2. Dynamic Signature 'NoneType' Bug Fix ([patch_dspy_none_type.py](file:///home/arutamonofu/dev/study/adaptive-extractor/scripts/patch_dspy_none_type.py))

### Problem
When using dynamic signatures in DSPy during optimization, the process crashed with:
```text
TypeError: 'NoneType' object is not subscriptable
```

### Root Cause
In `dspy/propose/utils.py`, the code attempted to access `item.signature.__pydantic_parent_namespace__["signature_name"]` without checking if `__pydantic_parent_namespace__` is `None` (which is often the case with dynamic signatures).

### Solution
The patch adds a safe check to verify `__pydantic_parent_namespace__` is not `None` and contains the `"signature_name"` key before indexing it.

---

## 3. Applying the Patches

All patches can be applied using the scripts in the `scripts/` directory:

```bash
# Apply zero-shot support patch (updates utils.py, mipro_optimizer_v2.py, grounded_proposer.py)
python scripts/patch_dspy_mipro_zero_bootstrap.py

# Apply 'NoneType' fix for dynamic signatures
python scripts/patch_dspy_none_type.py
```

### Verification

To verify if the patches are applied:
```bash
# Run the patch scripts directly. They check if the files are already patched.
python scripts/patch_dspy_mipro_zero_bootstrap.py
# Output should contain: "✓ utils.py already patched..." / "✓ mipro_optimizer_v2.py already patched..."

python scripts/patch_dspy_none_type.py
# Output should contain: "✓ File already patched!"
```

---

## 4. Impact

After applying these patches:
* **Zero-Shot & Labeled-Only Support**: Optimization works correctly with `max_bootstrapped_demos: 0`, allowing pure instruction tuning or labeled-only few-shot runs.
* **Dynamic Signatures**: Optimization functions seamlessly with dynamically generated signatures.
* **Enhanced Robustness**: Prompt tuning becomes significantly more reliable, robust, and cost-efficient.
