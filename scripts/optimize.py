#!/usr/bin/env python
"""Optimize extraction agents using MIPROv2.

This is a thin wrapper that delegates to the CLI module.
"""

import sys

from _bootstrap import add_src_to_path

add_src_to_path()

from ae.optimization.cli import optimize_command

if __name__ == "__main__":
    sys.exit(optimize_command())
