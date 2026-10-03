#!/usr/bin/env python3
"""Paired, no-rejection comparison of legacy and repetition-aware sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from generate import generate_batch, new_output_dir, resolve_device, training_reference, write_generation_outputs
from rtttl_gen.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num-songs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bpm", type=int, default=120, help="0 samples tempo")
    parser.add_argument("--min-events", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    train_songs, train_ids = training_reference(args.data, checkpoint)
    output = new_output_dir(args.output, parent="evaluations")
    settings = dict(num_songs=args.num_songs, seed=args.seed, bpm=args.bpm or None,
                    min_events=args.min_events, max_events=args.max_events,
                    temperature=0.8, top_k=20, top_p=0.95, device=device)
    controls = {
        "legacy": dict(repetition_penalty=1.0, repetition_window=16, max_pitch_run=0,
                       max_motif_repeats=0, max_motif_length=16),
        "guarded": dict(repetition_penalty=1.2, repetition_window=16, max_pitch_run=4,
                        max_motif_repeats=3, max_motif_length=16),
    }
    report = {"status": "executed", "protocol": "Same checkpoint and per-song seeds; all attempts retained; no rejection or cherry-picking; no weight updates.",
              "settings": settings, "controls": controls,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "dataset_sha256": checkpoint["dataset_sha256"],
              "runtime": {"device": device, "torch": torch.__version__, "python": platform.python_version()},
              "arms": {}}
    for arm, options in controls.items():
        started = time.perf_counter()
        print(f"Generating {arm}: {args.num_songs} songs on {device}", flush=True)
        records, songs = generate_batch(model, tokenizer, train_songs, train_ids, **settings, **options)
        arm_output = new_output_dir(output / arm)
        metrics = write_generation_outputs(arm_output, records, songs)
        report["arms"][arm] = {"metrics": metrics, "seconds": time.perf_counter() - started}
        (output / "comparison.json").write_text(json.dumps({**report, "status": "in_progress"}, indent=2) + "\n")
        print(json.dumps({"arm": arm, "degenerate_pct": metrics["degenerate_generation_pct"],
                          "repetition": metrics["repetition"]}), flush=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"COMPARISON={output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
