#!/usr/bin/env python
"""Run batch extraction on documents.

This is a thin wrapper that delegates to the CLI module.
"""

import sys

from _bootstrap import add_src_to_path

add_src_to_path()

from ae.extraction.cli import extract_command

if __name__ == "__main__":
    sys.exit(extract_command())
