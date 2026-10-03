#!/usr/bin/env python3
"""Render generation songs.jsonl into WAV files and a local HTML listener."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtttl_gen.audio import create_listening_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to generated songs.jsonl")
    parser.add_argument("--output", required=True, help="New directory for WAV and index.html")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = create_listening_batch(records, args.output, args.sample_rate, args.max_seconds)
    print(json.dumps({"output": args.output, "audio": results}, indent=2))


if __name__ == "__main__":
    main()
