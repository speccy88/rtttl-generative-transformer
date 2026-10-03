"""Grammar-constrained autoregressive generation for all three models."""
from __future__ import annotations

import math
from typing import Any

import torch

from .tokenizer import EventTokenizer


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
                    seed: int = 42, device: str | torch.device = "cpu") -> dict[str, Any]:
    if not 1 <= min_events <= max_events:
        raise ValueError("Require 1 <= min_events <= max_events")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("top_k >= 0 and 0 < top_p <= 1 are required")
    if bpm is not None and (isinstance(bpm, bool) or not isinstance(bpm, int)
                            or not tokenizer.MIN_BPM <= bpm <= tokenizer.MAX_BPM):
        raise ValueError(f"BPM must be an integer in {tokenizer.MIN_BPM}..{tokenizer.MAX_BPM}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = [tokenizer.bos_id]
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
            ids.append(sample_token(logits, allowed, generator, temperature, top_k, top_p))
        tokenizer.decode(ids)  # A postcondition, independent of the sampler.
        return {"token_ids": ids, "forced_eos": forced_eos}
    finally:
        if neural:
            model.train(was_training)
