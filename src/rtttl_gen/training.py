"""Small, readable training loop shared by the Transformer and GRU.

Losses are computed over the entire vocabulary (no grammar mask), ignoring PAD
only. Gradient accumulation divides by the *actual number of target tokens*.
Checkpoints are trusted local Python files: never load untrusted .pt files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .dataset import SequenceDataset, load_songs
from .tokenizer import EventTokenizer
from .transformer import TransformerLM
from .recurrent import GRULM
from .ngram import NGramLM


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        # Set before a CUDA BLAS operation for CUDA reproducibility where supported.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Top-level function, so Windows spawn-based DataLoaders can pickle it."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot use CUDA.")
    if requested.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but this PyTorch installation cannot use the Apple GPU. Install an MPS-enabled PyTorch on a supported Mac; CPU fallback is not automatic for an explicit MPS request.")
    return torch.device(requested)


def _model(model_type: str, config: dict[str, Any], vocab_size: int):
    args = dict(config)
    args.pop("type", None)
    args.pop("vocab_size", None)
    if model_type == "transformer":
        return TransformerLM(vocab_size=vocab_size, **args)
    if model_type == "gru":
        args.pop("n_heads", None)
        return GRULM(vocab_size=vocab_size, **args)
    if model_type == "ngram":
        return NGramLM(order=args.get("order", 5), alpha=args.get("alpha", 0.1))
    raise ValueError(f"Unknown model type: {model_type}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dataset_fingerprint(directory: Path) -> dict[str, str]:
    """Reject resumes against changed splits or vocabulary, even at a new path."""
    result = {}
    for filename in ("train.jsonl", "val.jsonl", "test.jsonl", "tokenizer.json"):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/prepare_dataset.py first.")
        result[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _new_run(output_root: str | Path, model_type: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    for serial in range(10000):
        run = root / f"{stamp}_{model_type}_{serial:03d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue
    raise RuntimeError("Unable to reserve a fresh run directory.")


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    # Atomic replacement avoids a partially-written last checkpoint after interruption.
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a checkpoint made by this package; do not pass an untrusted file."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    tokenizer = EventTokenizer.from_dict(checkpoint["tokenizer"])
    model = _model(checkpoint["model_type"], checkpoint["model_config"], tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.eval()
    return model, tokenizer, checkpoint


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: str | torch.device, pad_id: int) -> dict[str, float | int]:
    """Token-weighted, unmasked-vocabulary teacher-forced cross entropy."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        x, targets = batch["input_ids"].to(device), batch["targets"].to(device)
        logits = model(x).float()
        total_nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                    ignore_index=pad_id, reduction="sum").item()
        total_tokens += int((targets != pad_id).sum())
    model.train(was_training)
    if not total_tokens:
        raise ValueError("Evaluation split has no target tokens.")
    ce = total_nll / total_tokens
    return {"cross_entropy": ce, "perplexity": math.exp(min(ce, 700)), "token_count": total_tokens}


def evaluate_ngram_loader(model, loader: DataLoader, pad_id: int) -> dict[str, float | int]:
    """Score each unmasked target with its actual window context (including BPM).

    Cache scalar context/target NLLs rather than dense 940-entry probability
    vectors; repeated transitions are cheap without inflating memory.
    """
    nll, count = 0.0, 0
    cache: dict[tuple[tuple[int, ...], int], float] = {}
    order_context = max(0, model.order - 1)
    for batch in loader:
        for x, targets in zip(batch["input_ids"].tolist(), batch["targets"].tolist()):
            for index, target in enumerate(targets):
                if target == pad_id:
                    continue
                context = tuple(x[max(0, index + 1 - order_context):index + 1]) if order_context else ()
                key = (context, target)
                if key not in cache:
                    cache[key] = -model.log_probability(context, target)
                nll += cache[key]
                count += 1
    if count == 0:
        raise ValueError("Evaluation split has no target tokens.")
    ce = nll / count
    return {"cross_entropy": ce, "perplexity": math.exp(min(ce, 700)), "token_count": count}


def evaluate_ngram(model, sequences: list[list[int]]) -> dict[str, float | int]:
    """Convenience scorer for complete, unpadded song token sequences."""
    examples = [{"input_ids": torch.tensor(seq[:-1]), "targets": torch.tensor(seq[1:])}
                for seq in sequences if len(seq) > 1]
    return evaluate_ngram_loader(model, DataLoader(examples, batch_size=1), pad_id=0)


def _sample(model, tokenizer: EventTokenizer, config: dict, path: Path, step: int, device) -> None:
    from .generation import generate_tokens
    from .rtttl import encode_rtttl
    settings = config.get("samples", {})
    # Sampling must never advance the RNG used for training/dropout.
    rng = _rng_state()
    was_training = model.training if isinstance(model, torch.nn.Module) else None
    try:
        if was_training is not None:
            model.eval()
        with path.open("a", encoding="utf-8") as output:
            for index in range(int(settings.get("count", 2))):
                result = generate_tokens(model, tokenizer,
                    min_events=int(settings.get("min_events", 8)),
                    max_events=int(settings.get("max_events", 32)),
                    temperature=float(settings.get("temperature", 0.9)),
                    top_k=int(settings.get("top_k", 20)), top_p=1.0,
                    bpm=settings.get("bpm"), seed=int(config.get("seed", 42)) + step * 100 + index,
                    device=str(device))
                song = tokenizer.decode(result["token_ids"], name=f"step{step}_{index}")
                output.write(f"step={step} forced_eos={result['forced_eos']}\n{encode_rtttl(song)}\n")
    finally:
        _restore_rng(rng)
        if was_training is not None:
            model.train(was_training)


def _amp(device: torch.device, setting: str):
    setting = setting.lower()  # YAML may parse unquoted off as False
    if device.type != "cuda" or setting in ("off", "none", "false"):
        return None
    if setting == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if setting == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested but not supported; select mixed_precision: auto or fp16.")
        return torch.bfloat16
    if setting == "fp16":
        return torch.float16
    raise ValueError(f"Unknown mixed_precision setting: {setting}")


def _loader(dataset, cfg: dict, *, training: bool, epoch: int = 0):
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)) + epoch)
    workers = int(cfg.get("training", {}).get("num_workers", 0))
    return DataLoader(dataset, batch_size=int(cfg["training"].get("batch_size", 16)),
                      shuffle=training, generator=generator, num_workers=workers,
                      worker_init_fn=seed_worker,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


def train(config: str | Path | dict[str, Any], resume: str | Path | None = None,
          init_checkpoint: str | Path | None = None) -> Path:
    """Train in a new directory. A resume preserves its parent run unchanged.

    max_steps is the planned total optimizer updates; stop_after_steps optionally
    interrupts this invocation at an update boundary for a reproducible resume.
    init_checkpoint copies compatible weights only, starting a fresh optimizer,
    schedule, RNG stream, and validation history; it is not an exact resume.
    All relative paths are relative to the working directory (run from project root).
    """
    if resume is not None and init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive; choose exact resume or a new warm-start run.")
    if isinstance(config, (str, Path)):
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    else:
        cfg = json.loads(json.dumps(_jsonable(config)))
    tc = cfg.setdefault("training", {})
    seed = int(cfg.get("seed", 42))
    seed_everything(seed, bool(cfg.get("deterministic", True)))
    torch.set_num_threads(int(tc.get("cpu_threads", min(4, os.cpu_count() or 1))))
    device = select_device(str(tc.get("device", "auto")))
    data_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    fingerprints = dataset_fingerprint(data_dir)
    tokenizer = EventTokenizer.from_dict(json.loads((data_dir / "tokenizer.json").read_text()))
    songs = load_songs(data_dir / "train.jsonl")
    validation_songs = load_songs(data_dir / "val.jsonl")
    subset = cfg.get("data", {}).get("max_train_songs")
    if subset:
        songs = songs[:int(subset)]
    subset_val = cfg.get("data", {}).get("max_val_songs")
    if subset_val:
        validation_songs = validation_songs[:int(subset_val)]
    if not songs or not validation_songs:
        raise ValueError("Training and validation splits must both contain songs.")
    model_config = dict(cfg.get("model", {}))
    model_type = model_config.pop("type", "transformer")
    source = None
    if resume or init_checkpoint:
        if model_type == "ngram":
            raise ValueError("N-gram fitting is immediate; resume and init_checkpoint apply to neural models only.")
        source = torch.load(Path(resume or init_checkpoint), map_location="cpu", weights_only=False)
        if source.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format.")
        resume_fields = {"optimizer_state_dict", "scaler_state_dict", "scheduler_state_dict", "rng_state",
                         "next_epoch", "next_batch", "global_step", "best_val_loss", "stale_evaluations",
                         "schedule_steps", "warmup_steps", "config"}
        if resume and (source.get("inference_only") or not resume_fields.issubset(source)):
            raise ValueError("This checkpoint lacks full training state and cannot be resumed exactly. Use --init-checkpoint to copy its weights into a new training run with a fresh optimizer and schedule.")
        if source.get("dataset_sha256") != fingerprints:
            raise ValueError("Dataset changed since checkpoint: split/tokenizer SHA256 mismatch.")
        if (source.get("model_type") != model_type or source.get("model_config") != model_config
                or source.get("tokenizer") != tokenizer.to_dict()):
            raise ValueError("Checkpoint initialization requires the same model architecture and vocabulary.")
    initialization_checkpoint = (str(Path(init_checkpoint).resolve()) if init_checkpoint
                                 else source.get("initialization_checkpoint") if source else None)
    run = _new_run(cfg.get("output_root", "runs"), model_type)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _write_json(run / "dataset_sha256.json", fingerprints)
    _write_json(run / "provenance.json", {"parent_checkpoint": str(Path(resume).resolve()) if resume else None,
                "initialization_checkpoint": initialization_checkpoint,
                "utc_started": datetime.now(timezone.utc).isoformat(), "dataset_path": str(data_dir.resolve())})
    started = time.perf_counter()
    if model_type == "ngram":
        if resume:
            raise ValueError("N-gram fitting is immediate; resume applies to neural models only.")
        model = _model(model_type, model_config, tokenizer.vocab_size)
        # Use the SAME window construction as neural training, preserving metric comparability.
        context = int(cfg.get("data", {}).get("context_length", 256))
        train_data = SequenceDataset(songs, tokenizer, context,
                        augment_semitones=int(cfg.get("data", {}).get("augment_semitones", 0)), seed=seed)
        val_data = SequenceDataset(validation_songs, tokenizer, context, seed=seed)
        # Fit complete songs, so tempo is counted exactly once and long songs do
        # not introduce artificial BOS/BPM training transitions. Metrics below
        # use the shared windows, including their masked repeated BPM targets.
        if cfg.get("data", {}).get("augment_semitones", 0):
            from .augment import legal_transpositions, transpose
            limit = int(cfg["data"]["augment_semitones"])
            fit_songs = [transpose(song, random.Random(seed + 9176 * index).choice(
                legal_transpositions(song, max_semitones=limit))) for index, song in enumerate(songs)]
        else:
            fit_songs = songs
        model.fit([tokenizer.encode(song) for song in fit_songs], vocab_size=tokenizer.vocab_size)
        # n-gram has only one fitting pass. For an architecture comparison use
        # augment_semitones=0 for every model; neural augmentation changes each epoch.
        train_metrics = evaluate_ngram_loader(model, DataLoader(train_data, batch_size=32), tokenizer.pad_id)
        val_metrics = evaluate_ngram_loader(model, DataLoader(val_data, batch_size=32), tokenizer.pad_id)
        elapsed = time.perf_counter() - started
        model_config["context_length"] = context
        checkpoint = {"format_version": 1, "model_type": model_type, "model_config": model_config,
                      "model_state_dict": model.state_dict(), "tokenizer": tokenizer.to_dict(), "config": cfg,
                      "dataset_sha256": fingerprints, "epoch": 1, "global_step": 1,
                      "best_val_loss": val_metrics["cross_entropy"], "training_seconds": elapsed}
        _save_checkpoint(run / "checkpoint_best.pt", checkpoint)
        _save_checkpoint(run / "checkpoint_last.pt", checkpoint)
        fields = ["epoch", "global_step", "train_loss", "train_perplexity", "val_loss", "val_perplexity", "learning_rate", "elapsed_seconds"]
        with (run / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            writer.writerow(dict(zip(fields, [1, 1, train_metrics["cross_entropy"], train_metrics["perplexity"], val_metrics["cross_entropy"], val_metrics["perplexity"], 0, elapsed])))
        _sample(model, tokenizer, cfg, run / "samples.txt", 1, device)
        _write_json(run / "summary.json", {"model_type": model_type, "parameters": None,
            "training_seconds": elapsed, "train": train_metrics, "validation": val_metrics,
            "context_length": context, "vocab_size": tokenizer.vocab_size,
            "stored_count_entries": model.parameter_count()})
        print(f"RUN_DIR={run}", flush=True)
        return run

    context = int(model_config.get("context_length", 256))
    train_data = SequenceDataset(songs, tokenizer, context,
                    augment_semitones=int(cfg.get("data", {}).get("augment_semitones", 0)), seed=seed)
    val_data = SequenceDataset(validation_songs, tokenizer, context, seed=seed)
    val_loader = _loader(val_data, cfg, training=False)
    model = _model(model_type, model_config, tokenizer.vocab_size).to(device)
    if init_checkpoint:
        model.load_state_dict(source["model_state_dict"])
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc.get("learning_rate", 3e-4)),
                                 weight_decay=float(tc.get("weight_decay", 0.01)), betas=(0.9, 0.95))
    amp_dtype = _amp(device, str(tc.get("mixed_precision", "auto")))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    accum = int(tc.get("gradient_accumulation_steps", 1))
    epochs = int(tc.get("epochs", 50))
    if accum < 1 or epochs < 1:
        raise ValueError("gradient_accumulation_steps and epochs must be positive.")
    batches_per_epoch = len(_loader(train_data, cfg, training=True))
    planned_steps = math.ceil(batches_per_epoch / accum) * epochs
    max_steps = int(tc.get("max_steps", 0) or 0)
    if max_steps:
        planned_steps = min(planned_steps, max_steps)
    schedule_steps = planned_steps
    warmup = min(int(tc.get("warmup_steps", 100)), max(0, schedule_steps // 10))
    min_lr_ratio = float(tc.get("min_lr_ratio", 0.1))
    parent = None
    start_epoch, start_batch, global_step, best_val, stale = 0, 0, 0, float("inf"), 0
    patience_anchor = float("inf")
    running_nll, running_tokens = 0.0, 0
    prior_seconds = 0.0
    if resume:
        parent = source
        old_cfg = parent["config"]
        for key in ("seed", "data"):
            # Dataset path may move on RunPod; compare data options without that path.
            old, new = old_cfg.get(key), cfg.get(key)
            if key == "data":
                old = {k:v for k,v in (old or {}).items() if k != "processed_dir"}
                new = {k:v for k,v in (new or {}).items() if k != "processed_dir"}
            if old != new:
                raise ValueError(f"Resume requires unchanged {key} settings.")
        for key in ("batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
                    "min_lr_ratio", "early_stopping_min_delta", "gradient_clip"):
            if old_cfg["training"].get(key) != tc.get(key):
                raise ValueError(f"Resume requires unchanged training.{key}; start a new run for a new experiment.")
        model.load_state_dict(parent["model_state_dict"])
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        # CPU/bf16 checkpoints have no fp16 scaler state. Initialize a fresh
        # scaler when crossing precision/hardware; otherwise preserve it exactly.
        if scaler.is_enabled() and parent["scaler_state_dict"]:
            scaler.load_state_dict(parent["scaler_state_dict"])
        start_epoch, start_batch = parent["next_epoch"], parent["next_batch"]
        global_step, best_val, stale = parent["global_step"], parent["best_val_loss"], parent["stale_evaluations"]
        patience_anchor = parent.get("patience_anchor", best_val)
        running_nll, running_tokens = parent.get("running_nll", 0.0), parent.get("running_tokens", 0)
        prior_seconds = parent.get("training_seconds", 0.0)
        schedule_steps = parent["schedule_steps"]
        warmup = parent["warmup_steps"]
    def lr_factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, schedule_steps - warmup)))
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if parent:
        scheduler.load_state_dict(parent["scheduler_state_dict"])
        # LambdaLR construction changes optimizer LR; restore checkpoint LR exactly.
        for group, saved_group in zip(optimizer.param_groups, parent["optimizer_state_dict"]["param_groups"]):
            group["lr"] = saved_group["lr"]
        _restore_rng(parent["rng_state"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # AdamW FP32 weights + gradients + two optimizer moments: about 16 bytes/parameter.
    metadata = {"model_type": model_type, "parameters": param_count, "trainable_parameters": trainable_count,
                "vocab_size": tokenizer.vocab_size, "context_length": context,
                "max_events_per_window": (context - 2) // 2,
                "train_songs": len(songs), "validation_songs": len(validation_songs),
                "train_windows": len(train_data), "validation_windows": len(val_data),
                "planned_optimizer_steps": planned_steps, "schedule_steps": schedule_steps,
                "fp32_adamw_parameter_state_MiB": param_count * 16 / 1024**2,
                "memory_note": "Parameter-state estimate excludes activations, attention workspaces, logits, CUDA context and allocator. Measure peak memory on your GPU.",
                "device": str(device), "mixed_precision": str(amp_dtype),
                "torch_version": torch.__version__, "parent_checkpoint": str(resume) if resume else None,
                "initialization_checkpoint": initialization_checkpoint}
    _write_json(run / "model_info.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    writer_tb = None
    if tc.get("tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer_tb = SummaryWriter(str(run / "tensorboard"))
        except ImportError:
            print("TensorBoard unavailable; CSV logging remains enabled.", flush=True)
    fields = ["epoch", "global_step", "train_loss", "train_perplexity", "val_loss", "val_perplexity", "learning_rate", "train_target_tokens", "val_target_tokens", "elapsed_seconds", "peak_cuda_MiB"]
    metrics_path = run / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    invocation_steps = 0
    stop_after = int(tc.get("stop_after_steps", 0) or 0)
    interval = int(tc.get("eval_every_steps", 0) or 0)
    sample_every = int(tc.get("sample_every_evaluations", 1))
    eval_count = 0
    final_validation = None
    stop_reason = "epochs_completed"
    optimizer.zero_grad(set_to_none=True)
    def validate_and_save(next_epoch: int, next_batch: int, epoch_label: int):
        nonlocal best_val, patience_anchor, stale, eval_count, final_validation, running_nll, running_tokens
        final_validation = evaluate_loss(model, val_loader, device, tokenizer.pad_id)
        val_loss = float(final_validation["cross_entropy"])
        improved = val_loss < best_val
        significant = val_loss < patience_anchor - float(tc.get("early_stopping_min_delta", 0.0))
        stale = 0 if significant else stale + 1
        if significant:
            patience_anchor = val_loss
        if improved:
            best_val = val_loss
        train_loss = running_nll / running_tokens if running_tokens else 0.0
        elapsed = prior_seconds + time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        row = dict(zip(fields, [epoch_label, global_step, train_loss, math.exp(min(train_loss, 700)),
            val_loss, final_validation["perplexity"], optimizer.param_groups[0]["lr"],
            running_tokens, final_validation["token_count"], elapsed, peak]))
        with metrics_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"step={global_step} epoch={epoch_label} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_ppl={final_validation['perplexity']:.3f}", flush=True)
        if writer_tb:
            for key in ("train_loss", "val_loss", "val_perplexity", "learning_rate"):
                writer_tb.add_scalar(key, row[key], global_step)
        eval_count += 1
        if sample_every > 0 and eval_count % sample_every == 0:
            _sample(model, tokenizer, cfg, run / "samples.txt", global_step, device)
        running_nll, running_tokens = 0.0, 0
        checkpoint = {"format_version": 1, "model_type": model_type, "model_config": model_config,
            "model_state_dict": model.state_dict(), "tokenizer": tokenizer.to_dict(), "config": cfg,
            "initialization_checkpoint": initialization_checkpoint,
            "dataset_sha256": fingerprints, "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _rng_state(), "epoch": epoch_label, "next_epoch": next_epoch, "next_batch": next_batch,
            "global_step": global_step, "best_val_loss": best_val, "stale_evaluations": stale,
            "patience_anchor": patience_anchor,
            "schedule_steps": schedule_steps, "warmup_steps": warmup, "running_nll": running_nll,
            "running_tokens": running_tokens, "training_seconds": elapsed, "model_info": metadata}
        _save_checkpoint(run / "checkpoint_last.pt", checkpoint)
        if improved:
            _save_checkpoint(run / "checkpoint_best.pt", checkpoint)
        elif not (run / "checkpoint_best.pt").exists() and parent:
            # Preserve inherited best weights, not the current less-good weights.
            inherited_best = Path(resume).parent / "checkpoint_best.pt"
            if inherited_best.exists():
                import shutil
                shutil.copy2(inherited_best, run / "checkpoint_best.pt")
            else:
                # No best sibling available: do not mislabel a last checkpoint as best.
                (run / "BEST_CHECKPOINT_NOTE.txt").write_text("No improvement in resumed run and parent best checkpoint was not provided. Use the original best checkpoint.\n")
    completed = False
    for epoch in range(start_epoch, epochs):
        if hasattr(train_data, "set_epoch"):
            train_data.set_epoch(epoch)
        loader = _loader(train_data, cfg, training=True, epoch=epoch)
        accumulated_tokens = 0
        microbatches = 0
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            if max_steps and global_step >= max_steps:
                completed = True; stop_reason = "max_steps"; break
            model.train()
            x, targets = batch["input_ids"].to(device), batch["targets"].to(device)
            token_count = int((targets != tokenizer.pad_id).sum())
            if token_count == 0:
                continue
            with torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext():
                logits = model(x)
                loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                          ignore_index=tokenizer.pad_id, reduction="sum")
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Non-finite loss; last saved checkpoint is retained.")
            # Keep the backward loss near an average to avoid gratuitous fp16
            # overflow, then correct gradients for the actual non-PAD token count.
            loss_normalizer = int(tc.get("batch_size", 16)) * context * accum
            scaler.scale(loss_sum / loss_normalizer).backward()
            accumulated_tokens += token_count
            running_nll += loss_sum.item()
            running_tokens += token_count
            microbatches += 1
            if microbatches < accum and batch_index + 1 < len(loader):
                continue
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(loss_normalizer / accumulated_tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.get("gradient_clip", 1.0)))
            scale_before = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            # An fp16 overflow skips the optimizer step and must also skip scheduler advancement.
            successful_update = scaler.get_scale() >= scale_before
            if successful_update:
                scheduler.step()
                global_step += 1
                invocation_steps += 1
            optimizer.zero_grad(set_to_none=True)
            accumulated_tokens, microbatches = 0, 0
            end_epoch = batch_index + 1 == len(loader)
            hit_max = bool(max_steps and global_step >= max_steps)
            hit_stop = bool(stop_after and invocation_steps >= stop_after)
            should_validate = end_epoch or hit_max or hit_stop or (successful_update and interval > 0 and global_step % interval == 0)
            if should_validate:
                validate_and_save(epoch + 1 if end_epoch else epoch, 0 if end_epoch else batch_index + 1, epoch + 1)
                patience = int(tc.get("early_stopping_patience", 10))
                if patience > 0 and stale >= patience:
                    completed = True; stop_reason = "early_stopping"
            if hit_max or hit_stop or completed:
                completed = True
                if hit_stop:
                    stop_reason = "stop_after_steps"
                elif hit_max:
                    stop_reason = "max_steps"
                break
        start_batch = 0
        if completed:
            break
    if writer_tb:
        writer_tb.close()
    if not (run / "checkpoint_last.pt").exists():
        raise ValueError("No training steps remain in this configuration. Increase epochs/max_steps to resume; parent files were preserved.")
    _write_json(run / "summary.json", {**metadata, "global_step": global_step,
        "training_seconds": prior_seconds + time.perf_counter() - started, "stop_reason": stop_reason,
        "best_validation_loss": best_val, "final_validation": final_validation,
        "peak_cuda_MiB": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0})
    print(f"RUN_DIR={run}", flush=True)
    return run


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train or fine-tune a Transformer, GRU, or n-gram model.")
    parser.add_argument("--config", required=True, help="YAML config, interpreted relative to current project directory")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", help="Trusted full training checkpoint; exact resume creates a NEW run directory")
    source.add_argument("--init-checkpoint", help="Trusted neural checkpoint; copy weights into a NEW run with a fresh optimizer and schedule")
    parser.add_argument("--stop-after-steps", type=int, help="Stop this invocation after N updates; checkpoint can resume")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.stop_after_steps is not None:
        cfg.setdefault("training", {})["stop_after_steps"] = args.stop_after_steps
    train(cfg, resume=args.resume, init_checkpoint=args.init_checkpoint)


if __name__ == "__main__":
    main()
