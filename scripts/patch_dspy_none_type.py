#!/usr/bin/env python
"""Patch for DSPy 'NoneType' object is not subscriptable error.

This script patches dspy/propose/utils.py to check if __pydantic_parent_namespace__
is None (which is the case for dynamic signatures) before attempting to index it.
"""

from pathlib import Path


def patch_dspy_propose_utils():
    """Patch the DSPy propose/utils.py file."""
    # Find DSPy installation
    import dspy
    dspy_dir = Path(dspy.__file__).parent
    utils_path = dspy_dir / "propose" / "utils.py"

    print(f"Patching DSPy propose/utils.py at: {utils_path}")

    # Read the original file
    with open(utils_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already patched
    if "# PATCHED: Safe check for __pydantic_parent_namespace__" in content:
        print("✓ File already patched!")
        return True

    # Original code around line 175:
    # if hasattr(item, "signature") and item.signature is not None and item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig" not in completed_set:

    old_code = '            if isinstance(item, Parameter):\n                if hasattr(item, "signature") and item.signature is not None and item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig" not in completed_set:'

    new_code = """            if isinstance(item, Parameter):
                # PATCHED: Safe check for __pydantic_parent_namespace__ (can be None in dynamic signatures)
                has_ns = (
                    hasattr(item, "signature")
                    and item.signature is not None
                    and getattr(item.signature, "__pydantic_parent_namespace__", None) is not None
                )
                if (
                    has_ns
                    and "signature_name" in item.signature.__pydantic_parent_namespace__
                    and item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig" not in completed_set
                ):"""

    if old_code not in content:
        print("✗ Could not find the code section to patch!")
        print("The DSPy version may have changed.")
        return False

    # Apply the patch
    patched_content = content.replace(old_code, new_code)

    # Write the patched file
    with open(utils_path, "w", encoding="utf-8") as f:
        f.write(patched_content)

    print("✓ Successfully patched DSPy propose/utils.py!")
    return True


if __name__ == "__main__":
    success = patch_dspy_propose_utils()
    exit(0 if success else 1)
