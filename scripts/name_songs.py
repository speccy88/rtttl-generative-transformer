#!/usr/bin/env python3
"""Name an existing generated batch locally, preserving the source directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from generate import add_naming_arguments, apply_optional_naming, new_output_dir, write_generation_outputs
from rtttl_gen.audio import create_listening_batch
from rtttl_gen.rtttl import parse_rtttl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing generated directory or songs.jsonl")
    parser.add_argument("--output", help="New output directory; source is never overwritten")
    parser.add_argument("--audio", action="store_true", help="Render a listening page showing full titles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    add_naming_arguments(parser)
    parser.set_defaults(name_songs=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source = Path(args.input)
    source = source / "songs.jsonl" if source.is_dir() else source
    raw = source.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not records:
        parser.error("Input batch contains no records")
    songs = [parse_rtttl(record["rtttl"]) for record in records if record.get("valid")]
    output = new_output_dir(args.output)
    write_generation_outputs(output, records, songs)
    started = time.perf_counter()
    records, songs, naming = apply_optional_naming(records, songs, args)
    metrics = write_generation_outputs(output, records, songs)
    if args.audio:
        create_listening_batch(records, output / "listening")
    metadata = {"status": "executed", "operation": "name_existing_songs",
                "source": str(source.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
                "arguments": vars(args), "naming": naming,
                "duration_seconds": time.perf_counter() - started,
                "utc": datetime.now(timezone.utc).isoformat()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "naming": naming, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
