#!/usr/bin/env python3
"""Evaluate one executed checkpoint; outputs never contain unexecuted results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from generate import add_naming_arguments, add_sampling_arguments, apply_optional_naming, generate_batch, new_output_dir, resolve_device, sampling_kwargs, training_reference, write_generation_outputs
from rtttl_gen.audio import create_listening_batch
from rtttl_gen.dataset import load_songs
from rtttl_gen.evaluation import compare_distributions, predictive_metrics
from rtttl_gen.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output", help="New output directory; existing outputs are protected")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, help="Defaults to the checkpoint's training context")
    parser.add_argument("--max-eval-songs", type=int, help="Explicit diagnostic subset, reported in metrics; default evaluates entire splits")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    add_sampling_arguments(parser)
    add_naming_arguments(parser)
    args = parser.parse_args()
    if args.max_eval_songs is not None and args.max_eval_songs < 1:
        parser.error("--max-eval-songs must be positive")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model_type = checkpoint["model_type"]
    context = args.context_length or checkpoint.get("model_config", {}).get("context_length") or checkpoint.get("config", {}).get("data", {}).get("context_length", 256)
    train_songs, train_ids = training_reference(args.data, checkpoint)
    data = Path(args.data)
    validation, test = load_songs(data / "val.jsonl"), load_songs(data / "test.jsonl")
    available_counts = {"validation": len(validation), "test": len(test)}
    if args.max_eval_songs:
        validation, test = validation[:args.max_eval_songs], test[:args.max_eval_songs]
    output = new_output_dir(args.output, parent="evaluations")
    started = time.perf_counter()
    predictive = {name: predictive_metrics(model, tokenizer, songs, context, device=device, batch_size=args.batch_size) for name, songs in (("validation", validation), ("test", test))}
    records, songs = generate_batch(model, tokenizer, train_songs, train_ids, **sampling_kwargs(args), device=device)
    if args.name_songs:
        write_generation_outputs(output, records, songs)
    records, songs, naming = apply_optional_naming(records, songs, args)
    generated = write_generation_outputs(output, records, songs)
    distribution = compare_distributions(test, songs, output / "plots")
    if args.audio:
        create_listening_batch(records, output / "listening")
    parameters = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) if hasattr(model, "parameters") else None
    # The model's checkpoint contains measured training time; no speed estimate.
    metrics = {"status": "executed", "model_type": model_type, "parameters": parameters, "trainable_parameters": trainable, "vocabulary_size": tokenizer.vocab_size, "context_length": context, "predictive": predictive, "generation": generated, "distributions": distribution, "training_seconds": checkpoint.get("training_seconds"), "evaluation_seconds": time.perf_counter() - started, "evaluation_scope": "diagnostic subset" if args.max_eval_songs else "full provided validation/test splits", "available_split_songs": available_counts, "checkpoint": str(Path(args.checkpoint).resolve()), "training_reference_songs": len(train_songs), "arguments": vars(args), "runtime": {"device": device, "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None, "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(), "utc": datetime.now(timezone.utc).isoformat()}}
    metrics["dataset_sha256_verified"] = checkpoint["dataset_sha256"]
    metrics["naming"] = naming
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    row = {"model": model_type, "parameters": parameters, "test_cross_entropy_nats": predictive["test"]["cross_entropy_nats"], "test_perplexity": predictive["test"]["perplexity"], "rtttl_validity_pct": generated["rtttl_validity_pct"], "unique_generations_pct": generated["unique_generations_pct"], "exact_memorization_pct": generated["exact_training_match_pct"], "training_seconds": checkpoint.get("training_seconds"), "evaluation_scope": metrics["evaluation_scope"], "test_songs": len(test), "generated_songs": args.num_songs, "checkpoint": args.checkpoint}
    with (output / "comparison_row.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps({"output": str(output), "comparison": row}, indent=2))


if __name__ == "__main__":
    main()
