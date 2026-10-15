"""Feed-forward Transformer (FFT) blocks as used in FastSpeech / FastSpeech 2.

Each block = multi-head self-attention + a 2-layer 1-D convolutional position-wise
network (kernel sizes [9, 1] in the paper), each followed by residual + LayerNorm.
"""
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """(B,) lengths -> (B, L) bool mask, True at PADDED positions."""
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


def sinusoid_table(n_position: int, d_hid: int) -> torch.Tensor:
    pos = np.arange(n_position)[:, None]
    i = np.arange(d_hid)[None, :]
    angle = pos / np.power(10000, 2 * (i // 2) / d_hid)
    table = np.zeros((n_position, d_hid), dtype=np.float32)
    table[:, 0::2] = np.sin(angle[:, 0::2])
    table[:, 1::2] = np.cos(angle[:, 1::2])
    return torch.from_numpy(table)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_k = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_drop = dropout
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_head, self.d_k).permute(2, 0, 3, 1, 4)
        keep = (~pad_mask)[:, None, None, :]                       # (B,1,1,L) True = attend
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep, dropout_p=self.attn_drop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.norm(x + self.drop(self.fc(out)))


class PositionwiseConv(nn.Module):
    def __init__(self, d_model: int, d_inner: int, kernel_sizes: Sequence[int], dropout: float = 0.1):
        super().__init__()
        k1, k2 = kernel_sizes
        self.w1 = nn.Conv1d(d_model, d_inner, k1, padding=(k1 - 1) // 2)
        self.w2 = nn.Conv1d(d_inner, d_model, k2, padding=(k2 - 1) // 2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w2(F.relu(self.w1(x.transpose(1, 2)))).transpose(1, 2)
        return self.norm(x + self.drop(y))


class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_inner, kernel_sizes, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, d_model, dropout)
        self.ffn = PositionwiseConv(d_model, d_inner, kernel_sizes, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, pad_mask).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = self.ffn(x).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x


class FFTStack(nn.Module):
    """N FFT blocks with sinusoidal positions added at the input."""

    def __init__(self, d_model, n_layers, n_head, d_inner, kernel_sizes, dropout, max_len=2000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pos", sinusoid_table(max_len + 1, d_model), persistent=False)
        self.layers = nn.ModuleList(
            [FFTBlock(d_model, n_head, d_inner, kernel_sizes, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, add_pos: bool = True) -> torch.Tensor:
        L = x.size(1)
        if add_pos:
            pos = self.pos[:L] if L <= self.pos.size(0) else sinusoid_table(L, self.d_model).to(x.device)
            x = x + pos.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x
