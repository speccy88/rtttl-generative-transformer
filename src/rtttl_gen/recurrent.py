"""A deliberately simple causal GRU comparison using the same vocabulary."""
from __future__ import annotations

import torch
from torch import nn

from .transformer import count_parameters


class GRULM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 192,
                 n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if min(vocab_size, context_length, d_model, n_layers) < 1:
            raise ValueError("Model dimensions must be positive")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.config = dict(vocab_size=vocab_size, context_length=context_length,
                           d_model=d_model, n_layers=n_layers, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.output_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context_length:
            raise ValueError(f"ids must be [batch, length] with length 1..{self.context_length}")
        x = self.input_dropout(self.embedding(ids))
        x, _ = self.gru(x)
        return self.lm_head(self.output_dropout(x))

    def parameter_count(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only)
