#!/usr/bin/env python3
"""Backward-compatible wrapper to the new package CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eaf_twin.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    main()
