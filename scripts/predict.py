#!/usr/bin/env python
"""Run batch predictions on documents.

This is a thin wrapper that delegates to the CLI module.
"""

import sys

from aee.interface.cli.predict import predict_command

if __name__ == "__main__":
    sys.exit(predict_command())
