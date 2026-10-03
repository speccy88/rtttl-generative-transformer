#!/usr/bin/env python3
"""Run from the project directory: python train.py --config configs/local_2060s.yaml."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rtttl_gen.training import main
if __name__ == "__main__":
    main()
