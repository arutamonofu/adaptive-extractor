#!/usr/bin/env python
"""Parse PDF documents into structured format.

This is a thin wrapper that delegates to the CLI module.
"""

from _bootstrap import add_src_to_path

add_src_to_path()

from ae.ingestion.cli import main

if __name__ == "__main__":
    main()
