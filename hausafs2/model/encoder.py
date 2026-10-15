"""Linguistic encoder: factored embeddings of Hausa phonological features + FFT stack.

Input features per token (all small categorical spaces, embedded and summed):
  phone, vowel length, TONE, syllable weight, role in syllable, boundary strength,
  position of syllable-in-word / word-in-phrase / phrase-in-utterance, and a
  *downstep counter* computed from the tone sequence (see `downstep_counts`).

Extras for robustness with un-annotated text
  * tone dropout (training only): whole words (or utterances) have their tones
    replaced with UNKNOWN so the model learns to cope with unmarked orthography;
  * a tone probe head that predicts H/L/F from the encoder states, trained on the
    positions whose tone was hidden; at inference `predict_tone=True` fills UNKNOWN
    tones with the probe's prediction and re-encodes (two-pass).
"""
import torch
import torch.nn as nn

from ..frontend import symbols as S
from .layers import FFTStack


def downstep_counts(tone: torch.Tensor, phrase_idx: torch.Tensor, max_count: int = S.MAX_DOWNSTEP) -> torch.Tensor:
    """Number of Low-introducing tones (L, or F whose second half is low) seen so far
    *within the current phrase*, excluding the current token. Hausa high tones are
    progressively lowered after each low (downdrift / downstep); this counter lets
    the model condition pitch on that history. Resets at each phrase.

    tone, phrase_idx: (B, N) long. Returns (B, N) long in [0, max_count].
    """
    trig = ((tone == S.TONE_L) | (tone == S.TONE_F)).long()
    c_excl = torch.cumsum(trig, dim=1) - trig
    change = torch.ones_like(phrase_idx, dtype=torch.bool)
    change[:, 1:] = phrase_idx[:, 1:] != phrase_idx[:, :-1]
    start_val = torch.where(change, c_excl, torch.zeros_like(c_excl))
    base = torch.cummax(start_val, dim=1).values        # c_excl is non-decreasing -> cummax = latest start
    return (c_excl - base).clamp(0, max_count)


def apply_tone_dropout(tone, word_idx, p_word: float, p_utt: float):
    """Replace known tones by UNKNOWN for random words / whole utterances."""
    B, N = tone.shape
    known = (tone != S.TONE_NONE) & (tone != S.TONE_UNK)
    n_words = int(word_idx.max().item()) + 1 if word_idx.numel() else 1
    r_word = torch.rand(B, max(n_words, 1), device=tone.device)
    per_tok = r_word.gather(1, word_idx.clamp(min=0)) < p_word
    per_utt = (torch.rand(B, 1, device=tone.device) < p_utt)
    drop = known & (per_tok | per_utt)
    return torch.where(drop, torch.full_like(tone, S.TONE_UNK), tone)


ALL_FEATURES = ["length", "tone", "weight", "role", "boundary", "syl_pos", "word_pos", "phrase_pos"]


class LinguisticEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_head: int, d_inner: int, kernel, dropout: float,
                 max_len: int = 2000, feat_init_std: float = 0.3, use_downstep: bool = True,
                 features=None):
        super().__init__()
        self.use_downstep = use_downstep
        # which categorical streams are fed to the encoder (ablation switch; [] = phone ids only)
        self.features = list(ALL_FEATURES if features is None else features)
        self.phone_emb = nn.Embedding(S.N_PHONES, d_model, padding_idx=S.PAD_ID)
        sizes = dict(length=S.N_LENGTHS, tone=S.N_TONES, weight=S.N_WEIGHTS, role=S.N_ROLES,
                     boundary=S.N_BOUNDARIES, syl_pos=S.N_POS, word_pos=S.N_POS,
                     phrase_pos=S.N_POS, downstep=S.N_DOWNSTEP)
        self.feat_emb = nn.ModuleDict({k: nn.Embedding(v, d_model) for k, v in sizes.items()})
        for emb in self.feat_emb.values():
            nn.init.normal_(emb.weight, std=feat_init_std)
        self.stack = FFTStack(d_model, n_layers, n_head, d_inner, kernel, dropout, max_len)
        self.tone_probe = nn.Linear(d_model, 3)       # H / L / F

    def embed(self, batch, tone_in):
        x = self.phone_emb(batch["phone"])
        for name in self.features:
            ids = tone_in if name == "tone" else batch[name]
            x = x + self.feat_emb[name](ids)
        if self.use_downstep:
            x = x + self.feat_emb["downstep"](downstep_counts(tone_in, batch["phrase_idx"]))
        return x

    def forward(self, batch, tone_in, pad_mask):
        x = self.embed(batch, tone_in).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return self.stack(x, pad_mask)

    def fill_unknown_tones(self, enc_out, tone_in):
        """Replace UNKNOWN tones by the probe's argmax (H/L/F)."""
        pred = self.tone_probe(enc_out).argmax(-1) + S.TONE_H
        return torch.where(tone_in == S.TONE_UNK, pred, tone_in)
