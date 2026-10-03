"""Small GPT-style decoder, initialized from scratch. Right-pad batches."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # PyTorch uses its supported optimized kernel; CPU uses a normal fallback.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output_dropout(self.projection(y))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 192,
                 n_layers: int = 4, n_heads: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        if min(vocab_size, context_length, d_model, n_layers, n_heads) < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.config = dict(vocab_size=vocab_size, context_length=context_length,
                           d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._initialize)
        # Sharing input/output weights reduces parameters and regularizes the model.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError("ids must have shape [batch, token_length]")
        length = ids.shape[1]
        if not 1 <= length <= self.context_length:
            raise ValueError(f"Token length must be 1..{self.context_length}")
        positions = torch.arange(length, device=ids.device)
        x = self.embedding_dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def parameter_count(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only)
