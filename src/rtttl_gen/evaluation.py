"""Predictive, validity, diversity, and distribution metrics for RTTTL models.

Cross entropy is measured on the *unmasked* vocabulary. Grammar constraints are
only a generation aid; they do not artificially improve reported perplexity.
Similarity scores are heuristic diagnostics, never proof of originality.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .rtttl import Song, encode_rtttl, parse_rtttl


def entropy(values: Iterable[Any]) -> float:
    """Empirical Shannon entropy in bits; empty inputs have entropy zero."""
    counts = Counter(values)
    count = sum(counts.values())
    return -sum((n / count) * math.log2(n / count) for n in counts.values()) if count else 0.0


def event_key(event: Any) -> tuple:
    return (event.pitch, event.duration, event.dotted)


def song_key(song: Song) -> tuple:
    """Exact generated-sequence identity includes tempo, but excludes title."""
    return (song.bpm, tuple(event_key(e) for e in song.events))


def degeneracy_flags(song: Song, min_events: int = 1) -> list[str]:
    """Conservative listening/review flags; these are not syntax errors."""
    flags: list[str] = []
    keys = [event_key(e) for e in song.events]
    pitches = [e.pitch for e in song.events if e.pitch is not None]
    if len(keys) < min_events:
        flags.append("too_short")
    if not pitches:
        flags.append("all_rests")
    if len(pitches) >= 8 and len(set(pitches)) == 1:
        flags.append("single_pitch")
    if len(keys) >= 8 and max(Counter(keys).values()) / len(keys) >= 0.9:
        flags.append("dominant_event_at_least_90pct")
    if len(keys) >= 12:
        for period in range(1, 5):
            if sum(keys[i] == keys[i % period] for i in range(len(keys))) / len(keys) >= 0.9:
                flags.append("short_repeated_motif")
                break
    return flags


def validate_song(song: Song) -> tuple[bool, str | None, str | None]:
    """Encode and independently re-parse, checking exact musical equivalence."""
    try:
        rtttl = encode_rtttl(song)
        reparsed = parse_rtttl(rtttl)
        if song_key(song) != song_key(reparsed):
            return False, rtttl, "encoder/parser changed events or BPM"
        return True, rtttl, None
    except (ValueError, TypeError, AttributeError) as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, dict):
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported model output {type(output)!r}")


@torch.no_grad()
def predictive_metrics(
    model: Any,
    tokenizer: Any,
    songs: Sequence[Song],
    context_length: int,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Score every target once using the same song-aware windows as training.

    This uses the same causal next-token objective as training, without grammar
    masks. Windows start with BOS/BPM; songs are never concatenated. Repeated BPM
    targets are masked, and EOS is present only at the actual song ending.
    n-grams use the same window boundaries.
    The denominator is tokens, not batches or songs. Empty splits are explicit.
    """
    if context_length < 1 or batch_size < 1:
        raise ValueError("context_length and batch_size must be positive")
    if hasattr(model, "eval"):
        model.eval()
    from .dataset import SequenceDataset
    dataset = SequenceDataset(songs, tokenizer, context_length, augment_semitones=0, seed=42)
    pad_id = tokenizer.pad_id
    chunks = [(item["input_ids"].tolist(), [t if t != pad_id else -100 for t in item["targets"].tolist()]) for item in dataset]
    total_nll, token_count = 0.0, 0
    if hasattr(model, "next_logits"):
        ngram_cache: dict[tuple, float] = {}
        for inputs, targets in chunks:
            for i, target in enumerate(targets):
                if target == -100:
                    continue
                context = inputs[:i + 1]
                if hasattr(model, "log_probability"):
                    cache_key = (tuple(context[-getattr(model, "context_length", len(context)):]), target)
                    if cache_key not in ngram_cache:
                        ngram_cache[cache_key] = -model.log_probability(context, target)
                    total_nll += ngram_cache[cache_key]
                else:
                    logits = torch.as_tensor(model.next_logits(context), dtype=torch.float64)
                    total_nll -= float(F.log_softmax(logits, dim=-1)[target].item())
                token_count += 1
    else:
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            width = max(len(x) for x, _ in batch)
            pad_id = getattr(tokenizer, "pad_id", 0)
            inputs = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
            targets = torch.full((len(batch), width), -100, dtype=torch.long, device=device)
            for row, (x, y) in enumerate(batch):
                inputs[row, :len(x)] = torch.tensor(x, device=device)
                targets[row, :len(y)] = torch.tensor(y, device=device)
            logits = _logits(model(inputs)).float()
            total_nll += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum", ignore_index=-100).item())
            token_count += sum(sum(t != -100 for t in y) for _, y in batch)
    cross_entropy = total_nll / token_count if token_count else None
    perplexity = math.exp(cross_entropy) if cross_entropy is not None and cross_entropy < 700 else None
    return {
        "cross_entropy_nats": cross_entropy,
        "perplexity": perplexity,
        "target_tokens": token_count,
        "songs": len(songs),
        "context_windows": len(chunks),
        "context_length": context_length,
        "grammar_masked": False,
        "window_policy": "training SequenceDataset; complete event windows; BOS/BPM prefix; duplicate BPM targets masked; EOS only true ending; every original target once",
    }


def distribution_data(songs: Sequence[Song]) -> dict[str, list[float | int]]:
    data: dict[str, list] = {k: [] for k in ("pitch", "interval", "duration_beats", "bpm", "event_length", "rest_fraction", "note_range", "dotted_fraction")}
    for song in songs:
        notes = [e.pitch for e in song.events if e.pitch is not None]
        data["pitch"].extend(notes)
        # Intervals are between consecutive pitched notes, skipping rests.
        data["interval"].extend(b - a for a, b in zip(notes, notes[1:]))
        data["duration_beats"].extend(4 / e.duration * (1.5 if e.dotted else 1) for e in song.events)
        data["bpm"].append(song.bpm)
        data["event_length"].append(len(song.events))
        data["rest_fraction"].append(sum(e.pitch is None for e in song.events) / max(1, len(song.events)))
        data["note_range"].append(max(notes) - min(notes) if notes else 0)
        data["dotted_fraction"].append(sum(e.dotted for e in song.events) / max(1, len(song.events)))
    return data


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    ordered = sorted(values)
    n = len(ordered)
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    return {"count": n, "min": min(values), "mean": sum(values) / n, "median": median, "max": max(values)}


def _js_divergence(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    """Jensen-Shannon divergence in bits, range [0, 1], using exact categories."""
    if not left or not right:
        return None
    a, b = Counter(left), Counter(right)
    result = 0.0
    for key in a.keys() | b.keys():
        p, q = a[key] / len(left), b[key] / len(right)
        midpoint = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def generated_metrics(records: Sequence[dict], songs: Sequence[Song]) -> dict[str, Any]:
    """records includes failures; songs includes successfully decoded valid songs."""
    count = len(records)
    valid_count = sum(bool(r.get("valid")) for r in records)
    unique = len({song_key(song) for song in songs})
    similarities = [r["similarity"]["score"] for r in records if r.get("valid") and isinstance(r.get("similarity", {}).get("score"), (int, float))]
    exact = sum(bool(r.get("similarity", {}).get("exact_event_match")) for r in records if r.get("valid"))
    transposed = sum(bool(r.get("similarity", {}).get("transposition_match")) for r in records if r.get("valid"))
    def percent(n: float, d: float) -> float | None:
        return 100 * n / d if d else None
    return {
        "attempted_songs": count,
        "valid_songs": valid_count,
        "rtttl_validity_pct": percent(valid_count, count),
        "unique_generated_sequences": unique,
        "unique_generations_pct": percent(unique, valid_count),
        "duplicate_generation_pct": percent(valid_count - unique, valid_count),
        "exact_training_match_pct": percent(exact, valid_count),
        "transposition_family_match_pct": percent(transposed, valid_count),
        "nearest_similarity": _summary(similarities),
        "pitch_entropy_bits": entropy(e.pitch for song in songs for e in song.events if e.pitch is not None),
        "event_entropy_bits": entropy(event_key(e) for song in songs for e in song.events),
        "degenerate_generation_pct": percent(sum(bool(r.get("degeneracy_flags")) for r in records if r.get("valid")), valid_count),
        "forced_eos_pct": percent(sum(bool(r.get("forced_eos")) for r in records), count),
        "similarity_labels": dict(Counter(r.get("similarity", {}).get("label", "not evaluated") for r in records)),
        "denominator_note": "Validity/forced EOS use all attempts; uniqueness, memorization and degeneracy use valid songs. Exact diversity identity includes BPM and resolved events, excludes title.",
    }


def compare_distributions(real_songs: Sequence[Song], generated_songs: Sequence[Song], output_dir: Path | str) -> dict[str, Any]:
    """Write raw metric data and an eight-panel comparison plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real, generated = distribution_data(real_songs), distribution_data(generated_songs)
    result = {
        "real_song_count": len(real_songs), "generated_song_count": len(generated_songs),
        "real": {key: _summary(values) for key, values in real.items()},
        "generated": {key: _summary(values) for key, values in generated.items()},
        "categorical_js_divergence_bits": {key: _js_divergence(real[key], generated[key]) for key in ("pitch", "interval", "duration_beats", "bpm")},
        "definitions": {"duration_beats": "quarter-note beats including dotted multiplier; includes rests", "interval": "semitones between consecutive pitched notes, skipping rests", "note_range": "max minus min MIDI pitch per song; zero for songs with no notes", "js_divergence": "0 identical categorical distributions, 1 disjoint; descriptive, no significance claim"},
    }
    (output_dir / "distribution_data.json").write_text(json.dumps({"real": real, "generated": generated}, indent=2), encoding="utf-8")
    (output_dir / "distribution_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    titles = {"pitch": "Pitch (MIDI)", "interval": "Pitch interval (semitones)", "duration_beats": "Duration (quarter-note beats)", "bpm": "Tempo (BPM)", "event_length": "Song length (events)", "rest_fraction": "Rest fraction per song", "note_range": "Note range (semitones)", "dotted_fraction": "Dotted fraction per song"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, titles.items()):
        left, right = real[key], generated[key]
        if not left and not right:
            ax.text(0.5, 0.5, "No observations", ha="center", transform=ax.transAxes)
        elif key in ("pitch", "interval", "duration_beats", "bpm"):
            categories = sorted(set(left) | set(right))
            indices = np.arange(len(categories))
            for values, displacement, color, label in ((left, -0.2, "#2763a6", "Held-out real"), (right, 0.2, "#c66423", "Generated")):
                counts = Counter(values)
                ax.bar(indices + displacement, [counts[x] / max(1, len(values)) for x in categories], width=0.4, color=color, label=label)
            stride = max(1, math.ceil(len(categories) / 12))
            ax.set_xticks(indices[::stride], [f"{x:g}" for x in categories[::stride]], rotation=45)
        else:
            combined = left + right
            lo, hi = min(combined), max(combined)
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            bins = np.linspace(lo, hi, min(20, max(3, int(math.sqrt(len(combined))))) + 1)
            for values, color, label in ((left, "#2763a6", "Held-out real"), (right, "#c66423", "Generated")):
                if values:
                    ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), alpha=0.55, color=color, label=label)
        ax.set_title(title)
        ax.set_ylabel("Fraction")
        ax.grid(axis="y", alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Musical distribution comparison — {len(real_songs)} held-out / {len(generated_songs)} generated songs")
    fig.savefig(output_dir / "musical_distributions.png", dpi=160)
    plt.close(fig)
    return result
