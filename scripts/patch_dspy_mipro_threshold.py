#!/usr/bin/env python
"""Patch for DSPy MIPROv2 metric_threshold bug.

This script patches the create_n_fewshot_demo_sets function in DSPy to properly
pass metric_threshold to BootstrapFewShot in all cases.

Bug: When seed == -1 (unshuffled few-shot), metric_threshold was not being passed
to BootstrapFewShot, causing the threshold to be ignored for that candidate set.

Issue: https://github.com/stanfordnlp/dspy/issues/...
"""

import os
import re
from pathlib import Path


def patch_dspy_utils():
    """Patch the DSPy utils.py file to fix metric_threshold bug."""
    
    # Find DSPy installation
    import dspy
    dspy_dir = Path(dspy.__file__).parent
    utils_path = dspy_dir / "teleprompt" / "utils.py"
    
    print(f"Patching DSPy utils.py at: {utils_path}")
    
    # Read the original file
    with open(utils_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if already patched
    if "# PATCHED: Added metric_threshold parameter" in content:
        print("✓ File already patched!")
        return True
    
    # Find and replace the seed == -1 case
    # Original code (lines ~382-391):
    # elif seed == -1:
    #     # unshuffled few-shot
    #     program = BootstrapFewShot(
    #         metric=metric,
    #         max_errors=max_errors,
    #         max_bootstrapped_demos=max_bootstrapped_demos,
    #         max_labeled_demos=max_labeled_demos,
    #         teacher_settings=teacher_settings,
    #         max_rounds=max_rounds,
    #     )
    
    old_code = """        elif seed == -1:
            # unshuffled few-shot
            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = program.compile(student, teacher=teacher, trainset=trainset_copy)"""
    
    new_code = """        elif seed == -1:
            # unshuffled few-shot
            # PATCHED: Added metric_threshold parameter (was missing, causing threshold to be ignored)
            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = teleprompter.compile(student, teacher=teacher, trainset=trainset_copy)"""
    
    if old_code not in content:
        print("✗ Could not find the code section to patch!")
        print("The DSPy version may have changed.")
        return False
    
    # Apply the patch
    patched_content = content.replace(old_code, new_code)
    
    # Write the patched file
    with open(utils_path, "w", encoding="utf-8") as f:
        f.write(patched_content)
    
    print("✓ Successfully patched DSPy utils.py!")
    print("\nChanges made:")
    print("  - Added metric_threshold=metric_threshold to BootstrapFewShot in seed==-1 case")
    print("  - Changed 'program = BootstrapFewShot' to 'teleprompter = BootstrapFewShot' for consistency")
    
    return True


if __name__ == "__main__":
    success = patch_dspy_utils()
    exit(0 if success else 1)
