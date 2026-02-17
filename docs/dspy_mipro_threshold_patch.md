# DSPy MIPROv2 metric_threshold Bug Fix

## Problem

When optimizing with `metric_threshold = 1.0`, MIPROv2 was selecting examples with metric 0.8 as "full traces", even though they should have been filtered out.

### Error Message
```
Bootstrapped 2 full traces after 2 examples for up to 1 rounds, amounting to 2 attempts.
```

## Root Cause

In the file `dspy/teleprompt/utils.py`, the function `create_n_fewshot_demo_sets()`, when creating bootstrap examples for `seed == -1` (unshuffled few-shot), the `metric_threshold` parameter was **not being passed** to `BootstrapFewShot`:

```python
# BEFORE (incorrect)
elif seed == -1:
    # unshuffled few-shot
    program = BootstrapFewShot(
        metric=metric,
        max_errors=max_errors,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        teacher_settings=teacher_settings,
        max_rounds=max_rounds,
        # ← metric_threshold was NOT passed!
    )
```

This meant that for one of the candidate demonstration sets (specifically the unshuffled few-shot), the metric threshold was being ignored.

## Solution

Added `metric_threshold` parameter passing to `BootstrapFewShot` for all cases:

```python
# AFTER (correct)
elif seed == -1:
    # unshuffled few-shot
    # PATCHED: Added metric_threshold parameter (was missing, causing threshold to be ignored)
    teleprompter = BootstrapFewShot(
        metric=metric,
        max_errors=max_errors,
        metric_threshold=metric_threshold,  # ← ADDED
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        teacher_settings=teacher_settings,
        max_rounds=max_rounds,
    )
```

## Applying the Patch

```bash
python scripts/patch_dspy_mipro_threshold.py
```

## Verification

After applying the patch, verify the file has been modified:

```bash
grep -A 10 "elif seed == -1:" $(python -c "import dspy; print(dspy.__file__.replace('__init__.py', 'teleprompt/utils.py'))")
```

You should see:
```python
elif seed == -1:
    # unshuffled few-shot
    # PATCHED: Added metric_threshold parameter (was missing, causing threshold to be ignored)
    teleprompter = BootstrapFewShot(
        metric=metric,
        max_errors=max_errors,
        metric_threshold=metric_threshold,
        ...
```

## Impact on Optimization

After applying the patch:
- Examples with metrics below `metric_threshold` will be correctly filtered out in **all** bootstrap demonstration sets
- The number of "full traces" should decrease (only examples with metric >= threshold)
- Optimized agent quality may improve as only high-quality demonstrations will be used

## Notes

- The patch is applied to the installed DSPy library in the current environment
- If you reinstall DSPy, you'll need to reapply the patch
- Recommended to add patch application to your environment setup script

## Related Issues

- DSPy issue: (link to GitHub issue if exists)
- Internal bug tracker: (if applicable)
