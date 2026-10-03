#!/usr/bin/env python3
"""Dataset-free local RTTTL generation from the released safetensors weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch
from safetensors.torch import load_model

from batch_plan import plan_batch
from diagnostics import degeneracy_flags, repetition_statistics
from generation import generate_tokens
from options import add_naming_arguments, add_sampling_arguments, apply_optional_naming, sampling_kwargs
from rtttl import encode_rtttl, parse_rtttl
from tokenizer import EventTokenizer
from transformer import TransformerLM


def select_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this PyTorch installation")
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError("Device must be auto, cpu, cuda or mps")
    return requested


def load_local_model(directory: str | Path, device: str = "auto"):
    """Read local JSON/safetensors, preserve tied weights, and select hardware."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    tokenizer = EventTokenizer.from_dict(json.loads((directory / "tokenizer.json").read_text()))
    if config.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("Model and tokenizer vocabulary sizes do not match")
    model = TransformerLM(**config)
    load_model(model, str(directory / "model.safetensors"), strict=True, device="cpu")
    model.to(select_device(device)).eval()
    return model, tokenizer


def generate_records(model, tokenizer, *, device: str = "cpu", **kwargs) -> tuple[list, list]:
    settings = plan_batch(kwargs["num_songs"], seed=kwargs["seed"], profile=kwargs["profile"],
                          bpm=kwargs["bpm"], bpm_range=kwargs["bpm_range"],
                          tonic=kwargs["tonic"], mode=kwargs["mode"])
    sampling = {key: value for key, value in kwargs.items()
                if key not in ("num_songs", "seed", "profile", "bpm", "bpm_range", "tonic", "mode")}
    records, songs = [], []
    for index, planned in enumerate(settings):
        seed = kwargs["seed"] + index
        result = generate_tokens(model, tokenizer, **sampling, device=device, seed=seed,
                                 profile=planned["profile"], bpm=planned["bpm"],
                                 tonic=planned["tonic"], mode=planned["mode"])
        song = tokenizer.decode(result["token_ids"], name=f"Generated{index + 1:02d}")
        line = encode_rtttl(song)
        parsed = parse_rtttl(line)
        if parsed.events != song.events or parsed.bpm != song.bpm:
            raise RuntimeError("Generated RTTTL failed its round-trip check")
        records.append(dict(id=f"sample_{index + 1:04d}", seed=seed, valid=True, rtttl=line,
            event_count=len(song.events), bpm=song.bpm, token_ids=result["token_ids"],
            tokens=[tokenizer.tokens[token] for token in result["token_ids"]],
            forced_eos=result["forced_eos"], repetition_interventions=result["repetition_interventions"],
            generation_settings={**planned, "bpm":song.bpm},
            degeneracy_flags=degeneracy_flags(song, kwargs["min_events"]),
            repetition_statistics=repetition_statistics(song)))
        songs.append(song)
    return records, songs


def write_songs(directory: Path, records: list) -> None:
    (directory / "songs.rtttl").write_text("\n".join(r["rtttl"] for r in records) + "\n")
    (directory / "songs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=2)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="New RTTTL file; existing files are refused")
    destination.add_argument("--output-dir", type=Path, help="New folder for RTTTL, metadata and optional audio")
    parser.add_argument("--audio", action="store_true", help="Create a listening page; requires --output-dir")
    add_sampling_arguments(parser)
    add_naming_arguments(parser)
    parser.set_defaults(num_songs=5)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.audio and args.output_dir is None:
        parser.error("--audio requires --output-dir")
    for path in (args.output, args.output_dir):
        if path is not None and path.exists():
            parser.error("Output already exists; choose a new path")
    # Validate planning before loading weights or creating output folders.
    plan_batch(args.num_songs, seed=args.seed, profile=args.profile, bpm=args.bpm,
               bpm_range=args.bpm_range, tonic=args.tonic, mode=args.mode)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model, tokenizer = load_local_model(args.model_dir, device)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records, songs = generate_records(model, tokenizer, device=device, **sampling_kwargs(args))
    if args.output_dir is not None:
        write_songs(args.output_dir, records)
    elif args.output is not None:
        # Save melody before the optional naming model is loaded.
        with args.output.open("x") as f:
            f.write("\n".join(r["rtttl"] for r in records) + "\n")
    records, songs, naming = apply_optional_naming(records, songs, args)
    if args.output_dir is not None:
        write_songs(args.output_dir, records)
        if args.audio:
            from audio import create_listening_batch
            create_listening_batch(records, args.output_dir / "listening")
        metadata = dict(device=device, python_version=platform.python_version(), torch_version=torch.__version__,
                        duration_seconds=time.perf_counter() - started, naming=naming,
                        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                        training_reference="Unavailable in this dataset-free release; originality was not evaluated.")
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        text = "\n".join(r["rtttl"] for r in records) + "\n"
        if args.output is None:
            print(text, end="")
        elif args.name_songs:
            args.output.write_text(text)
    forced = sum(r["forced_eos"] for r in records)
    print(f"Generated {len(records)} valid songs on {device}; {forced} ended at the event limit. "
          "Training-reference originality was not evaluated.", file=sys.stderr)


if __name__ == "__main__":
    main()
