"""Grammar-constrained autoregressive generation for all three models."""
from __future__ import annotations

import math
import random
from typing import Any

import torch

from .guidance import make_guidance
from .tokenizer import EventTokenizer


def repetition_filter(logits: torch.Tensor, allowed_ids: list[int], pitches: list[int],
                      repetition_penalty: float, repetition_window: int,
                      max_pitch_run: int, max_motif_repeats: int,
                      max_motif_length: int) -> tuple[torch.Tensor, list[int], dict[str, int]]:
    """Adjust an event-start decision, before top-k/nucleus truncation.

    History contains pitch/rest IDs, never duration or structural tokens. A
    repeated rhythm alone is normal music and must not be penalized. Motifs
    use the full generated history even when the neural context has slid.
    Counts describe filtering steps, not how often the sampled winner changed.
    """
    counts = dict(penalty_steps=0, pitch_run_blocks=0, motif_blocks=0)
    if not pitches:
        return logits, allowed_ids, counts
    blocked: set[int] = set()
    if max_pitch_run and len(pitches) >= max_pitch_run:
        if all(p == pitches[-1] for p in pitches[-max_pitch_run:]):
            blocked.add(pitches[-1])
            counts["pitch_run_blocks"] = 1
    if max_motif_repeats:
        for period in range(2, min(max_motif_length, len(pitches) // max_motif_repeats) + 1):
            motif = pitches[-period:]
            if len(set(motif)) > 1 and pitches[-period * max_motif_repeats:] == motif * max_motif_repeats:
                blocked.add(motif[0])
                counts["motif_blocks"] = 1
    allowed = [token for token in allowed_ids if token not in blocked]
    if repetition_penalty != 1:
        # Clone so callers' logits and cached n-gram distributions stay intact.
        logits = logits.detach().float().cpu().clone()
        recent = list(set(pitches[-repetition_window:]) & set(allowed))
        if recent:
            scores = logits[recent]
            logits[recent] = torch.where(scores < 0, scores * repetition_penalty,
                                        scores / repetition_penalty)
            counts["penalty_steps"] = 1
    return logits, allowed, counts


def neural_context(ids: list[int], context_length: int) -> list[int]:
    """Keep BOS/BPM and an event-aligned suffix, matching training prefixes.

    For long generations the suffix slides; training uses disjoint excerpts.
    It is therefore still an approximation to unlimited-context prediction.
    A trailing pitch is retained so its duration can be predicted next.
    """
    if context_length < 4:
        raise ValueError("Generation context_length must be at least 4 tokens")
    if len(ids) <= context_length:
        return ids
    body = ids[2:]
    start = max(0, len(body) - (context_length - 2))
    start += start % 2  # event beginnings have even offsets inside the body
    return ids[:2] + body[start:]


def sample_token(logits: torch.Tensor, allowed_ids: list[int], generator: torch.Generator,
                 temperature: float = 0.8, top_k: int = 20, top_p: float = 0.95) -> int:
    """Filter within legal support, then sample. CPU RNG is portable."""
    if not allowed_ids:
        raise ValueError("No legal next token")
    scores = logits.detach().float().cpu()[allowed_ids] / temperature
    if not torch.isfinite(scores).all():
        raise ValueError("Model produced non-finite legal-token logits")
    if top_k > 0 and top_k < scores.numel():
        # Keep exactly k entries even when logits tie.
        kept = torch.topk(scores, top_k).indices
        filtered = torch.full_like(scores, -torch.inf)
        filtered[kept] = scores[kept]
        scores = filtered
    if top_p < 1:
        sorted_scores, order = scores.sort(descending=True)
        cumulative = sorted_scores.softmax(dim=0).cumsum(dim=0)
        remove = cumulative > top_p
        # Retain the token that crosses the threshold and at least one token.
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[order[remove]] = -torch.inf
    probabilities = scores.softmax(dim=0)
    selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return int(allowed_ids[selected])


@torch.no_grad()
def generate_tokens(model: Any, tokenizer: EventTokenizer, min_events: int = 8,
                    max_events: int = 128, temperature: float = 0.8,
                    top_k: int = 20, top_p: float = 0.95, bpm: int | None = None,
                    seed: int = 42, device: str | torch.device = "cpu", *,
                    repetition_penalty: float = 1.2, repetition_window: int = 16,
                    max_pitch_run: int = 4, max_motif_repeats: int = 3,
                    max_motif_length: int = 16, profile: str | None = None,
                    tonic: str = "C", mode: str | None = None) -> dict[str, Any]:
    if not 1 <= min_events <= max_events:
        raise ValueError("Require 1 <= min_events <= max_events")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("top_k >= 0 and 0 < top_p <= 1 are required")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError("repetition_penalty must be finite and >= 1")
    for name, value, minimum in (("repetition_window", repetition_window, 1),
                                  ("max_pitch_run", max_pitch_run, 0),
                                  ("max_motif_repeats", max_motif_repeats, 0),
                                  ("max_motif_length", max_motif_length, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_motif_repeats == 1:
        raise ValueError("max_motif_repeats must be 0 (disabled) or >= 2")
    guidance = make_guidance(tokenizer, profile, tonic, mode)
    if guidance is not None and bpm is None:
        # A separate local RNG varies tempo without consuming melody sampling
        # draws or altering the application's global random state.
        bpm = random.Random(seed ^ 0x54454D504F).randint(*guidance.tempo_range)
    if bpm is not None and (isinstance(bpm, bool) or not isinstance(bpm, int)
                            or not tokenizer.MIN_BPM <= bpm <= tokenizer.MAX_BPM):
        raise ValueError(f"BPM must be an integer in {tokenizer.MIN_BPM}..{tokenizer.MAX_BPM}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = [tokenizer.bos_id]
    pitches: list[int] = []
    interventions = dict(penalty_steps=0, pitch_run_blocks=0, motif_blocks=0)
    forced_eos = False
    neural = isinstance(model, torch.nn.Module)
    was_training = bool(model.training) if neural else False
    if neural:
        model.eval()
    try:
        if bpm is not None:
            ids.append(tokenizer.token_to_id[f"<BPM_{bpm}>"])
        # Each iteration adds exactly one legal token. A completed event budget
        # forces EOS; this is reported instead of pretending natural termination.
        while ids[-1] != tokenizer.eos_id:
            allowed = tokenizer.allowed_next(ids, min_events=min_events, max_events=max_events)
            if allowed == [tokenizer.eos_id]:
                ids.append(tokenizer.eos_id)
                forced_eos = True
                break
            if neural:
                context = neural_context(ids, model.context_length)
                tensor = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(tensor)[0, -1]
            else:
                logits = model.next_logits(ids)
            if guidance is not None:
                logits = guidance.apply(logits, allowed, pitches)
            if len(ids) >= 2 and (len(ids) - 2) % 2 == 0:
                logits, allowed, counts = repetition_filter(
                    logits, allowed, pitches, repetition_penalty, repetition_window,
                    max_pitch_run, max_motif_repeats, max_motif_length)
                for key, count in counts.items():
                    interventions[key] += count
            token = sample_token(logits, allowed, generator, temperature, top_k, top_p)
            ids.append(token)
            if token in tokenizer.event_start_ids:
                pitches.append(token)
        tokenizer.decode(ids)  # A postcondition, independent of the sampler.
        return {"token_ids": ids, "forced_eos": forced_eos,
                "repetition_interventions": interventions}
    finally:
        if neural:
            model.train(was_training)
