#!/usr/bin/env python3
"""Generate structured RTTTL songs from a trained neural or n-gram checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from rtttl_gen.audio import create_listening_batch
from rtttl_gen.dataset import load_records
from rtttl_gen.evaluation import degeneracy_flags, generated_metrics, validate_song
from rtttl_gen.generation import generate_tokens
from rtttl_gen.rtttl import song_from_dict
from rtttl_gen.similarity import SimilarityIndex
from rtttl_gen.training import dataset_fingerprint, load_checkpoint


def new_output_dir(requested: str | Path | None, parent: str = "generated") -> Path:
    path = Path(requested) if requested else Path(parent) / (datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    return requested


def training_reference(data: str | Path, checkpoint: dict | None = None) -> tuple[list, list[str]]:
    path = Path(data)
    if checkpoint is not None:
        expected = checkpoint.get("dataset_sha256")
        actual = dataset_fingerprint(path)
        if expected is None or expected != actual:
            raise ValueError("Dataset fingerprints do not match the checkpoint. Supply the exact processed split files used for this run; scoring another corpus as 'training data' would mislabel memorization.")
    records = load_records(path / "train.jsonl" if path.is_dir() else path)
    # A smoke run may use only the first N training songs. Unseen records must
    # not be called training-set matches merely because they share a split file.
    subset = (checkpoint or {}).get("config", {}).get("data", {}).get("max_train_songs")
    if subset:
        records = records[:int(subset)]
    if not records:
        raise ValueError("Training reference is empty; novelty cannot be measured")
    return [song_from_dict(record["song"]) for record in records], [str(record["id"]) for record in records]


def generate_batch(model, tokenizer, train_songs, train_ids, *, num_songs: int = 20, temperature: float = 0.8, top_k: int = 20, top_p: float = 0.95, min_events: int = 8, max_events: int = 96, bpm: int | None = None, seed: int = 42, device: str = "cpu") -> tuple[list[dict], list]:
    if num_songs < 1:
        raise ValueError("num_songs must be positive")
    index = SimilarityIndex(train_songs, ids=train_ids)
    records, valid_songs = [], []
    for i in range(num_songs):
        sample_seed = seed + i
        output = generate_tokens(model, tokenizer, min_events=min_events, max_events=max_events, temperature=temperature, top_k=top_k, top_p=top_p, bpm=bpm, seed=sample_seed, device=device)
        ids = output["token_ids"]
        record = {"id": f"sample_{i + 1:04d}", "seed": sample_seed, "token_ids": ids, "tokens": [tokenizer.tokens[token] for token in ids], "forced_eos": bool(output.get("forced_eos", False)), "valid": False, "rtttl": None, "similarity": {}, "degeneracy_flags": []}
        try:
            song = tokenizer.decode(ids, name=f"Generated{i + 1:04d}")
            valid, rtttl, error = validate_song(song)
            record.update(valid=valid, rtttl=rtttl, event_count=len(song.events), bpm=song.bpm, validation_error=error)
            record["degeneracy_flags"] = degeneracy_flags(song, min_events)
            if valid:
                record["similarity"] = index.nearest(song)
                valid_songs.append(song)
        except (ValueError, TypeError, IndexError) as exc:
            record["validation_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records, valid_songs


def write_generation_outputs(output: Path, records: list[dict], songs: list) -> dict:
    (output / "songs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (output / "songs.rtttl").write_text("\n".join(r["rtttl"] for r in records if r.get("valid")) + "\n", encoding="utf-8")
    metrics = generated_metrics(records, songs)
    (output / "generation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-songs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-events", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=96)
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--seed", type=int, default=42)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed", help="Dataset directory containing train.jsonl, used for novelty comparisons")
    parser.add_argument("--output", help="New output directory; existing directories are never overwritten")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audio", action="store_true", help="Create WAV files and listening/index.html")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    add_sampling_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device=device)
    train_songs, train_ids = training_reference(args.data, checkpoint)
    output = new_output_dir(args.output)
    started = time.perf_counter()
    records, songs = generate_batch(model, tokenizer, train_songs, train_ids, **{key: getattr(args, key) for key in ("num_songs", "temperature", "top_k", "top_p", "min_events", "max_events", "bpm", "seed")}, device=device)
    metrics = write_generation_outputs(output, records, songs)
    if args.audio:
        create_listening_batch(records, output / "listening")
    metadata = {"arguments": vars(args), "device": device, "torch_version": torch.__version__, "python_version": platform.python_version(), "model_type": checkpoint.get("model_type"), "duration_seconds": time.perf_counter() - started, "training_reference_songs": len(train_songs), "dataset_sha256_verified": checkpoint["dataset_sha256"], "status": "executed", "utc": datetime.now(timezone.utc).isoformat()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
