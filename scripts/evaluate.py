#!/usr/bin/env python3
"""Convenience entry point; identical arguments to the root evaluate.py."""
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
runpy.run_path(str(root / "evaluate.py"), run_name="__main__")
