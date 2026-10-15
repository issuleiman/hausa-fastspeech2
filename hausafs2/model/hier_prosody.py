"""Hierarchical prosody encoder: syllable -> word -> phrase (+ utterance).

Phoneme states are mean-pooled into syllable, word and phrase *nodes*; each level
is processed by its own small FFT stack (so syllables can attend to syllables,
words to words, phrases to phrases -> context at that prosodic granularity);
the node states are then broadcast back to their phonemes and added through
zero-initialised projections (the module starts as an identity).

Tokens whose index is -1 for a level (e.g. pauses have no syllable or word) are
excluded from that level's pooling and receive no broadcast from it. Pauses do
belong to a phrase, so they see phrase context (useful for pause duration).
"""
from typing import Dict, Sequence

import torch
import torch.nn as nn

from .layers import FFTStack, lengths_to_mask  # noqa: F401


def segment_mean(x: torch.Tensor, idx: torch.Tensor, n_seg: int):
    """Mean-pool x (B,N,D) into segments given idx (B,N) with -1 = excluded.
    Returns pooled (B,n_seg,D) and member counts (B,n_seg)."""
    B, N, D = x.shape
    valid = (idx >= 0).to(x.dtype)
    safe = idx.clamp(min=0)
    pooled = x.new_zeros(B, n_seg, D)
    count = x.new_zeros(B, n_seg)
    pooled.scatter_add_(1, safe.unsqueeze(-1).expand(-1, -1, D), x * valid.unsqueeze(-1))
    count.scatter_add_(1, safe, valid)
    return pooled / count.clamp(min=1).unsqueeze(-1), count


def segment_broadcast(seg: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Inverse of segment_mean: gather segment states back to tokens (zeros for idx<0)."""
    D = seg.size(-1)
    g = seg.gather(1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, D))
    return g * (idx >= 0).unsqueeze(-1).to(seg.dtype)


class HierarchicalProsodyEncoder(nn.Module):
    LEVEL_KEYS: Dict[str, str] = {"syllable": "syl_idx", "word": "word_idx", "phrase": "phrase_idx"}

    def __init__(self, d_model: int, n_head: int, d_inner: int, kernel: Sequence[int] = (3, 1),
                 dropout: float = 0.1, n_layers: int = 1,
                 levels: Sequence[str] = ("syllable", "word", "phrase")):
        super().__init__()
        self.levels = list(levels)
        self.stacks = nn.ModuleDict({
            lv: FFTStack(d_model, n_layers, n_head, d_inner, kernel, dropout) for lv in self.levels})
        self.level_emb = nn.ParameterDict({
            lv: nn.Parameter(torch.zeros(d_model)) for lv in self.levels})
        self.proj = nn.ModuleDict({lv: nn.Linear(d_model, d_model) for lv in self.levels})
        self.utt_proj = nn.Linear(d_model, d_model)
        for m in list(self.proj.values()) + [self.utt_proj]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, batch, pad_mask: torch.Tensor) -> torch.Tensor:
        out = h
        for lv in self.levels:
            idx = batch[self.LEVEL_KEYS[lv]].masked_fill(pad_mask, -1)
            n_seg = max(int(idx.max().item()) + 1, 1)
            nodes, count = segment_mean(h, idx, n_seg)
            node_pad = count == 0
            nodes = (nodes + self.level_emb[lv]).masked_fill(node_pad.unsqueeze(-1), 0.0)
            nodes = self.stack_forward(lv, nodes, node_pad)
            out = out + self.proj[lv](segment_broadcast(nodes, idx))
        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)
        utt = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        out = out + self.utt_proj(utt).unsqueeze(1)
        return self.norm(out).masked_fill(pad_mask.unsqueeze(-1), 0.0)

    def stack_forward(self, lv, nodes, node_pad):
        # A sample with zero nodes at some level (e.g. a pause-only sequence) is
        # fully padded; give attention one dummy visible key to avoid NaNs.
        all_pad = node_pad.all(dim=1)
        if all_pad.any():
            node_pad = node_pad.clone()
            node_pad[all_pad, 0] = False
        return self.stacks[lv](nodes, node_pad).masked_fill(node_pad.unsqueeze(-1), 0.0)
