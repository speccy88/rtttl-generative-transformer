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
from rtttl_gen.batch_plan import plan_batch
from rtttl_gen.dataset import load_records
from rtttl_gen.evaluation import degeneracy_flags, generated_metrics, validate_song
from rtttl_gen.generation import generate_tokens
from rtttl_gen.guidance import PROFILE_NAMES
from rtttl_gen.rtttl import song_from_dict
from rtttl_gen.similarity import SimilarityIndex
from rtttl_gen.training import dataset_fingerprint, load_checkpoint, select_device


def new_output_dir(requested: str | Path | None, parent: str = "generated") -> Path:
    path = Path(requested) if requested else Path(parent) / (datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_device(requested: str) -> str:
    return str(select_device(requested))


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


def generate_batch(model, tokenizer, train_songs, train_ids, *, num_songs: int = 20, temperature: float = 0.8, top_k: int = 20, top_p: float = 0.95, min_events: int = 8, max_events: int = 96, bpm: int | None = None, bpm_range: tuple[int, int] | None = None, seed: int = 42, device: str = "cpu", repetition_penalty: float = 1.2, repetition_window: int = 16, max_pitch_run: int = 4, max_motif_repeats: int = 3, max_motif_length: int = 16, profile: str | None = None, tonic: str | None = None, mode: str | None = None) -> tuple[list[dict], list]:
    if num_songs < 1:
        raise ValueError("num_songs must be positive")
    plan = plan_batch(num_songs, seed=seed, profile=profile, bpm=bpm, bpm_range=bpm_range,
                      tonic=tonic, mode=mode)
    index = SimilarityIndex(train_songs, ids=train_ids)
    records, valid_songs = [], []
    for i in range(num_songs):
        sample_seed = seed + i
        settings = plan[i]
        output = generate_tokens(model, tokenizer, min_events=min_events, max_events=max_events, temperature=temperature, top_k=top_k, top_p=top_p, bpm=settings["bpm"], seed=sample_seed, device=device, repetition_penalty=repetition_penalty, repetition_window=repetition_window, max_pitch_run=max_pitch_run, max_motif_repeats=max_motif_repeats, max_motif_length=max_motif_length, profile=settings["profile"], tonic=settings["tonic"], mode=settings["mode"])
        ids = output["token_ids"]
        record = {"id": f"sample_{i + 1:04d}", "seed": sample_seed, "token_ids": ids, "tokens": [tokenizer.tokens[token] for token in ids], "forced_eos": bool(output.get("forced_eos", False)), "valid": False, "rtttl": None, "similarity": {}, "degeneracy_flags": []}
        record["repetition_interventions"] = output["repetition_interventions"]
        record["generation_settings"] = dict(settings)
        try:
            song = tokenizer.decode(ids, name=f"Generated{i + 1:04d}")
            valid, rtttl, error = validate_song(song)
            record.update(valid=valid, rtttl=rtttl, event_count=len(song.events), bpm=song.bpm, validation_error=error)
            record["generation_settings"]["bpm"] = song.bpm
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
    tempo = parser.add_mutually_exclusive_group()
    tempo.add_argument("--bpm", type=int, help="Fix the tempo of every melody")
    tempo.add_argument("--bpm-range", type=int, nargs=2, metavar=("MIN", "MAX"), help="Choose a seeded tempo per melody in this inclusive range")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Recent pitch/rest logit penalty; 1 disables. Durations are unaffected")
    parser.add_argument("--repetition-window", type=int, default=16, help="Recent events considered for the soft pitch penalty")
    parser.add_argument("--max-pitch-run", type=int, default=4, help="Maximum consecutive identical pitches/rests; 0 disables")
    parser.add_argument("--max-motif-repeats", type=int, default=3, help="Copies of a pitch motif allowed before breaking its continuation; 0 disables")
    parser.add_argument("--max-motif-length", type=int, default=16, help="Longest pitch motif checked, in events (minimum 2)")
    parser.add_argument("--profile", choices=[*PROFILE_NAMES, "mixed"], help="Melody guidance; mixed balances styles, tempos and keys across the batch")
    parser.add_argument("--tonic", help="Key preference, e.g. C, F#, Bb; default C for one profile, varied for mixed")
    parser.add_argument("--mode", choices=["major", "natural-minor"], help="Scale preference; defaults to each profile's mode")


def sampling_kwargs(args: argparse.Namespace) -> dict:
    return {key: getattr(args, key) for key in (
        "num_songs", "temperature", "top_k", "top_p", "min_events", "max_events",
        "bpm", "bpm_range", "seed", "repetition_penalty", "repetition_window", "max_pitch_run",
        "max_motif_repeats", "max_motif_length", "profile", "tonic", "mode")}


def add_naming_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name-songs", action="store_true", help="After generation, use a small embedded local LLM to name melodies (optional naming dependencies)")
    parser.add_argument("--name-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model ID or local model directory")
    parser.add_argument("--name-revision", help="Optional model revision; the default model uses a pinned commit")
    parser.add_argument("--name-device", choices=["auto", "cuda", "mps", "xpu", "cpu"], default="auto")
    parser.add_argument("--name-offline", action="store_true", help="Use cached/local model files only; never download")
    parser.add_argument("--name-cache", help="Optional Hugging Face model cache directory")


def apply_optional_naming(records: list[dict], songs: list, args: argparse.Namespace) -> tuple[list[dict], list, dict]:
    """The caller saves melodies first; optional naming failure keeps them usable."""
    if not args.name_songs:
        return records, songs, {"status": "disabled"}
    if not any(record.get("valid") for record in records):
        return records, songs, {"status": "skipped", "reason": "No valid songs"}
    from rtttl_gen.local_llm import LocalTitleModel
    from rtttl_gen.naming import name_records
    namer = None
    try:
        print("Naming melodies with a local model (first use may download weights)...", file=sys.stderr, flush=True)
        namer = LocalTitleModel(model_id=args.name_model, revision=args.name_revision,
                                device=args.name_device, offline=args.name_offline,
                                cache_dir=args.name_cache)
        named, named_songs, report = name_records(records, namer, seed=args.seed)
        report["model"] = namer.metadata()
        if report.get("failed"):
            print(f"Naming warning: {report['failed']} melodies kept fallback titles; see metadata.", file=sys.stderr)
        return named, named_songs, report
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:1000]}"
        print(f"Naming unavailable; melodies preserved. {error}", file=sys.stderr)
        return records, songs, {"status": "unavailable", "error": error,
                                "named": 0, "failed": sum(bool(r.get("valid")) for r in records)}
    finally:
        if namer is not None:
            namer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed", help="Dataset directory containing train.jsonl, used for novelty comparisons")
    parser.add_argument("--output", help="New output directory; existing directories are never overwritten")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audio", action="store_true", help="Create WAV files and listening/index.html")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    add_sampling_arguments(parser)
    add_naming_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device=device)
    train_songs, train_ids = training_reference(args.data, checkpoint)
    output = new_output_dir(args.output)
    started = time.perf_counter()
    records, songs = generate_batch(model, tokenizer, train_songs, train_ids, **sampling_kwargs(args), device=device)
    if args.name_songs:
        write_generation_outputs(output, records, songs)  # preserve music even if naming is interrupted
    records, songs, naming = apply_optional_naming(records, songs, args)
    metrics = write_generation_outputs(output, records, songs)
    if args.audio:
        create_listening_batch(records, output / "listening")
    metadata = {"arguments": vars(args), "device": device, "torch_version": torch.__version__, "python_version": platform.python_version(), "model_type": checkpoint.get("model_type"), "duration_seconds": time.perf_counter() - started, "training_reference_songs": len(train_songs), "dataset_sha256_verified": checkpoint["dataset_sha256"], "status": "executed", "utc": datetime.now(timezone.utc).isoformat()}
    metadata["naming"] = naming
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
