"""Interpolated token n-gram baseline for the shared event representation.

With two tokens per event, order=5 conditions on approximately two events.
The unigram has additive smoothing. Each observed higher-order context uses
Witten--Bell interpolation with its shorter suffix; probabilities sum to one
over the complete vocabulary. Generation applies the same grammar as neural
models, but predictive cross entropy uses this unmasked distribution.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import math
from typing import Any, Iterable, Sequence

import torch


class NGramLM:
    def __init__(self, order: int = 5, alpha: float = 0.1) -> None:
        if order < 1 or alpha <= 0:
            raise ValueError("order >= 1 and alpha > 0 are required")
        self.order = order
        self.alpha = alpha
        self.vocab_size = 0
        self.counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.context_length = max(1, order - 1)
        self._base_probabilities: torch.Tensor | None = None
        self._context_totals: dict[tuple[int, ...], tuple[int, int]] = {}
        self._logits_cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()

    def fit(self, sequences: Iterable[Sequence[int]], vocab_size: int | None = None) -> "NGramLM":
        self.counts = defaultdict(Counter)
        max_token = -1
        for sequence in sequences:
            sequence = [int(x) for x in sequence]
            if sequence:
                max_token = max(max_token, max(sequence))
            if any(t < 0 for t in sequence):
                raise ValueError("Token IDs cannot be negative")
            # BOS is context, never a supervised target. No cross-song transitions.
            for i in range(1, len(sequence)):
                target = sequence[i]
                for length in range(min(self.order - 1, i) + 1):
                    context = tuple(sequence[i - length:i]) if length else ()
                    self.counts[context][target] += 1
        self.vocab_size = max_token + 1 if vocab_size is None else vocab_size
        if self.vocab_size < 1 or max_token >= self.vocab_size or not self.counts.get(()):
            raise ValueError("Need nonempty training targets and a vocabulary covering their IDs")
        self._prepare_prediction()
        return self

    def _prepare_prediction(self) -> None:
        self._context_totals = {context: (sum(counts.values()), len(counts))
                                for context, counts in self.counts.items()}
        probabilities = torch.full((self.vocab_size,), self.alpha, dtype=torch.float64)
        for token, count in self.counts[()].items():
            probabilities[token] += count
        self._base_probabilities = probabilities / probabilities.sum()
        self._logits_cache.clear()

    def log_probability(self, context: Sequence[int], target: int) -> float:
        """Scalar equivalent of next_logits; avoids dense vectors for scoring."""
        if self._base_probabilities is None:
            raise ValueError("Fit or load the n-gram model first")
        if not 0 <= target < self.vocab_size:
            raise ValueError("Target ID is outside the vocabulary")
        probability = float(self._base_probabilities[target])
        context = tuple(int(x) for x in context[-self.context_length:])
        for length in range(1, min(self.order - 1, len(context)) + 1):
            key = context[-length:]
            counts = self.counts.get(key)
            if counts:
                total, types = self._context_totals[key]
                probability = (counts.get(target, 0) + types * probability) / (total + types)
        return math.log(probability)

    def next_logits(self, context: Sequence[int]) -> torch.Tensor:
        """Return normalized log probabilities (usable as sampling logits)."""
        if not self.counts.get(()) or self.vocab_size < 1:
            raise ValueError("Fit or load the n-gram model first")
        context = tuple(int(x) for x in context[-self.context_length:])
        cached = self._logits_cache.get(context)
        if cached is not None:
            self._logits_cache.move_to_end(context)
            return cached.clone()
        assert self._base_probabilities is not None
        probabilities = self._base_probabilities.clone()
        for length in range(1, min(self.order - 1, len(context)) + 1):
            key = context[-length:]
            counts = self.counts.get(key)
            if not counts:
                continue
            total, types = self._context_totals[key]
            denominator = total + types
            probabilities *= types / denominator
            for token, count in counts.items():
                probabilities[token] += count / denominator
        logits = probabilities.log().float()
        self._logits_cache[context] = logits
        if len(self._logits_cache) > 2048:
            self._logits_cache.popitem(last=False)
        return logits.clone()

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": 1, "order": self.order, "alpha": self.alpha,
                "vocab_size": self.vocab_size,
                "counts": [{"context": list(context), "targets": [[token, n] for token, n in sorted(counts.items())]}
                           for context, counts in sorted(self.counts.items())]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != 1:
            raise ValueError("Unsupported n-gram state format")
        self.__init__(order=int(state["order"]), alpha=float(state["alpha"]))
        self.vocab_size = int(state["vocab_size"])
        for row in state["counts"]:
            context = tuple(int(x) for x in row["context"])
            self.counts[context] = Counter({int(token): int(n) for token, n in row["targets"]})
        if self.vocab_size < 1 or not self.counts.get(()):
            raise ValueError("Invalid empty n-gram state")
        self._prepare_prediction()

    def parameter_count(self, trainable_only: bool = False) -> int:
        """Stored count entries, not neural parameters. Report separately."""
        return sum(len(counts) for counts in self.counts.values())


# Friendly compatibility name for callers referring to the baseline generically.
NGram = NGramLM
