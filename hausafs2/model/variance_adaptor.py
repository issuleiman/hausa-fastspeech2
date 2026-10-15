"""Variance adaptor: duration, pitch and energy (FastSpeech 2, Sec. 2.3), adapted for tone.

Pitch representation  (cfg.pitch.mode)
--------------------------------------
contour (default)  Per-phoneme pitch contour of K points of speaker-normalised
                   log-F0, predicted at phoneme level; frames inside a phoneme
                   read it back by linear interpolation. A falling tone therefore
                   keeps its slope inside a single long vowel, which phoneme-level
                   averaging would destroy. K=1 == phoneme-level (reference repo).
frame              Frame-level pitch predicted after length regulation (the
                   original paper's setting).

Energy  (cfg.energy.mode)  'phoneme' (average per phoneme, added before LR) or 'frame'.

Both are quantised into `n_bins` bins and embedded, as in the paper.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .layers import lengths_to_mask


class VariancePredictor(nn.Module):
    """2x [Conv1D -> ReLU -> LayerNorm -> Dropout] + Linear (paper Fig. 1c)."""

    def __init__(self, d_in: int, filter_size: int, kernel: int, dropout: float, out_dim: int = 1,
                 squeeze: bool = True):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(d_in, filter_size, kernel, padding=pad)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel, padding=pad)
        self.ln1, self.ln2 = nn.LayerNorm(filter_size), nn.LayerNorm(filter_size)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(filter_size, out_dim)
        self.out_dim, self.squeeze = out_dim, squeeze and out_dim == 1

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
        h = self.drop(self.ln1(torch.relu(self.conv1((x * keep).transpose(1, 2)).transpose(1, 2))))
        h = h * keep                       # padded positions must not leak into conv2 (batch invariance)
        h = self.drop(self.ln2(torch.relu(self.conv2(h.transpose(1, 2)).transpose(1, 2))))
        out = self.linear(h)
        if self.squeeze:
            return out.squeeze(-1).masked_fill(pad_mask, 0.0)
        return out.masked_fill(pad_mask.unsqueeze(-1), 0.0)


def length_regulate(h: torch.Tensor, dur: torch.Tensor, max_len: Optional[int] = None):
    """Vectorised length regulator (no Python loop over tokens).

    h (B,N,D); dur (B,N) non-negative ints (0 for padding).
    Returns expanded (B,T,D), mel_len (B,), token_idx (B,T), valid (B,T) bool,
    rel (B,T) relative position in [0,1] of each frame inside its token.
    """
    B, N, D = h.shape
    csum = torch.cumsum(dur, dim=1)
    mel_len = csum[:, -1]
    T = int(max_len) if max_len is not None else int(mel_len.max().item())
    T = max(T, 1)
    t = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T).contiguous()
    idx = torch.searchsorted(csum.contiguous(), t, right=True).clamp(max=N - 1)
    valid = t < mel_len.unsqueeze(1)
    out = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)) * valid.unsqueeze(-1).to(h.dtype)
    start = (csum - dur).gather(1, idx)
    d_at = dur.gather(1, idx).clamp(min=1)
    rel = ((t - start).to(h.dtype) + 0.5) / d_at.to(h.dtype)
    return out, mel_len, idx, valid, rel.clamp(0.0, 1.0)


def contour_to_frames(contour: torch.Tensor, idx: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """contour (B,N,K) -> frame-level values (B,T) by linear interpolation of the K
    points (located at relative positions j/(K-1)) inside each token."""
    B, N, K = contour.shape
    c = contour.gather(1, idx.unsqueeze(-1).expand(-1, -1, K))            # (B,T,K)
    if K == 1:
        return c.squeeze(-1)
    pos = rel * (K - 1)
    lo = pos.floor().long().clamp(0, K - 1)
    hi = (lo + 1).clamp(max=K - 1)
    w = pos - lo.to(pos.dtype)
    v_lo = c.gather(-1, lo.unsqueeze(-1)).squeeze(-1)
    v_hi = c.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
    return v_lo * (1 - w) + v_hi * w


class VarianceAdaptor(nn.Module):
    def __init__(self, d_model: int, vp_cfg: dict, pitch_cfg: dict, energy_cfg: dict,
                 pitch_range, energy_range):
        super().__init__()
        f, k, dr = vp_cfg["filter"], vp_cfg["kernel"], vp_cfg["dropout"]
        self.pitch_mode = pitch_cfg.get("mode", "contour")
        self.energy_mode = energy_cfg.get("mode", "phoneme")
        assert self.pitch_mode in ("contour", "frame") and self.energy_mode in ("phoneme", "frame")
        self.K = int(pitch_cfg.get("n_points", 3)) if self.pitch_mode == "contour" else 1
        n_bins_p = int(pitch_cfg.get("n_bins", 256))
        n_bins_e = int(energy_cfg.get("n_bins", 256))

        self.duration_predictor = VariancePredictor(d_model, f, k, dr)
        self.pitch_predictor = VariancePredictor(d_model, f, k, dr, out_dim=self.K if self.pitch_mode == "contour" else 1,
                                                 squeeze=self.pitch_mode != "contour")   # contour keeps (B,N,K) even for K=1
        self.energy_predictor = VariancePredictor(d_model, f, k, dr)

        self.register_buffer("pitch_bins", torch.linspace(float(pitch_range[0]), float(pitch_range[1]), n_bins_p - 1))
        self.register_buffer("energy_bins", torch.linspace(float(energy_range[0]), float(energy_range[1]), n_bins_e - 1))
        self.pitch_emb = nn.Embedding(n_bins_p, d_model)
        self.energy_emb = nn.Embedding(n_bins_e, d_model)

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor, src_pad: torch.Tensor, batch: Dict[str, torch.Tensor],
                controls: Optional[dict] = None, max_len: Optional[int] = None,
                pitch_cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        c = dict(duration=1.0, pitch_shift=0.0, pitch_range=1.0, energy_shift=0.0)
        c.update(controls or {})
        # Each variance is teacher-forced only if its target is in the batch, so e.g. GT durations with
        # *predicted* pitch/energy can be used to evaluate the pitch/energy predictors (see scripts/compare_arms.py).
        t_dur, t_pit, t_en = "duration" in batch, "pitch" in batch, "energy" in batch
        valid_src = ~src_pad

        log_d = self.duration_predictor(h, src_pad)
        if t_dur:
            dur = batch["duration"].long()
        else:
            dur = torch.clamp(torch.round((torch.exp(log_d) - 1.0) * c["duration"]), min=0).long()
        dur = dur * valid_src.long()

        # All phoneme-level predictors read the same encoder output (parallel, paper Fig. 1b).
        h0 = h
        hp = h0 if pitch_cond is None else h0 + pitch_cond      # explicit tone/downstep skip (contour mode)
        p_pred = self.pitch_predictor(hp, src_pad) if self.pitch_mode == "contour" else None
        e_pred = None
        if self.energy_mode == "phoneme":
            e_pred = self.energy_predictor(h0, src_pad)
            e_in = batch["energy"] if t_en else e_pred + c["energy_shift"]
            h = h0 + self.energy_emb(torch.bucketize(e_in, self.energy_bins))   # added before LR

        x, mel_len, tok_idx, valid, rel = length_regulate(h, dur, max_len)
        mel_pad = ~valid

        if self.pitch_mode == "contour":
            if t_pit:
                p_in = batch["pitch"]
            else:
                p_in = p_pred * c["pitch_range"] + c["pitch_shift"]
            frame_p = contour_to_frames(p_in, tok_idx, rel)
            x = x + self.pitch_emb(torch.bucketize(frame_p, self.pitch_bins))
        else:   # frame-level pitch predicted on the expanded sequence
            p_pred = self.pitch_predictor(x, mel_pad)
            p_in = batch["pitch"] if t_pit else p_pred * c["pitch_range"] + c["pitch_shift"]
            x = x + self.pitch_emb(torch.bucketize(p_in, self.pitch_bins))

        if self.energy_mode == "frame":
            e_pred = self.energy_predictor(x, mel_pad)
            e_in = batch["energy"] if t_en else e_pred + c["energy_shift"]
            x = x + self.energy_emb(torch.bucketize(e_in, self.energy_bins))

        x = x.masked_fill(mel_pad.unsqueeze(-1), 0.0)
        return dict(x=x, mel_len=mel_len, mel_pad=mel_pad, log_dur=log_d, dur=dur,
                    pitch=p_pred, energy=e_pred, tok_idx=tok_idx, rel=rel)
