"""FastSpeech 2 loss + Hausa-specific auxiliary terms.

total = w_mel*L1(mel) + w_postnet*L1(mel_post) + w_dur*MSE(log(d+1))
      + w_pitch*MSE(pitch) + w_energy*MSE(energy)
      + w_tone*CE(tone probe, on tone-hidden vowels)           [optional]
      + w_len*hinge(long-vs-short predicted log-duration)       [optional, default off]

Pitch/energy losses ignore pause tokens and zero-duration tokens. Pitch error on
tone-bearing vowels is up-weighted (`tone_bearing_weight`) since vowels carry the
lexical tone, while consonant/pause pitch is mostly interpolation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..frontend import symbols as S


def _wmean(sq: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (sq * w).sum() / w.sum().clamp(min=1.0)


class HausaLoss(nn.Module):
    def __init__(self, loss_cfg: dict, pitch_mode: str = "contour", energy_mode: str = "phoneme"):
        super().__init__()
        w = dict(mel=1.0, postnet=1.0, duration=1.0, pitch=1.0, energy=1.0, tone=0.1, length=0.0)
        w.update(loss_cfg.get("weights", {}))
        self.w = w
        self.tone_bearing_weight = float(loss_cfg.get("tone_bearing_weight", 2.0))
        self.length_margin = float(loss_cfg.get("length_margin", 0.2))
        self.pitch_mode, self.energy_mode = pitch_mode, energy_mode

    def forward(self, out: dict, batch: dict):
        mel_pad = out["mel_pad"]
        src_pad = out["src_pad"]
        T = min(out["mel"].size(1), batch["mel"].size(1))
        mv = (~mel_pad[:, :T]).unsqueeze(-1).float()
        tgt = batch["mel"][:, :T]
        n_mels = tgt.size(-1)
        denom = mv.sum().clamp(min=1.0) * n_mels
        l_mel = ((out["mel"][:, :T] - tgt).abs() * mv).sum() / denom
        l_post = ((out["mel_post"][:, :T] - tgt).abs() * mv).sum() / denom

        kind = batch["kind"]
        dur_t = batch["duration"]
        src_valid = (~src_pad).float()
        # duration (log domain, +1 so zero-length pauses are representable)
        l_dur = _wmean((out["log_dur"] - torch.log(dur_t.float() + 1.0)) ** 2, src_valid)

        tok_ok = src_valid * (kind != S.KIND_P).float() * (dur_t > 0).float()
        is_vowel = (kind == S.KIND_V).float()
        tok_w = tok_ok * (1.0 + (self.tone_bearing_weight - 1.0) * is_vowel)

        if self.pitch_mode == "contour":
            sq = ((out["pitch"] - batch["pitch"]) ** 2).mean(-1)
            l_pitch = _wmean(sq, tok_w)
        else:
            Tf = out["pitch"].size(1)
            fk = kind.gather(1, out["tok_idx"][:, :Tf])
            fw = (~mel_pad[:, :Tf]).float() * (fk != S.KIND_P).float()
            fw = fw * (1.0 + (self.tone_bearing_weight - 1.0) * (fk == S.KIND_V).float())
            l_pitch = _wmean((out["pitch"] - batch["pitch"][:, :Tf]) ** 2, fw)

        if self.energy_mode == "phoneme":
            l_energy = _wmean((out["energy"] - batch["energy"]) ** 2, tok_ok)
        else:
            Tf = out["energy"].size(1)
            fk = kind.gather(1, out["tok_idx"][:, :Tf])
            fw = (~mel_pad[:, :Tf]).float() * (fk != S.KIND_P).float()
            l_energy = _wmean((out["energy"] - batch["energy"][:, :Tf]) ** 2, fw)

        # tone probe: only where the tone was hidden from the encoder but is known in truth
        tone_gt, tone_in = batch["tone"], out["tone_in"]
        hid = (tone_in == S.TONE_UNK) & (tone_gt >= S.TONE_H) & (tone_gt <= S.TONE_F) & (~src_pad)
        if hid.any():
            l_tone = F.cross_entropy(out["tone_logits"][hid], tone_gt[hid] - S.TONE_H)
        else:
            l_tone = out["mel"].new_zeros(())

        l_len = self.length_contrast(out, batch) if self.w["length"] > 0 else out["mel"].new_zeros(())

        total = (self.w["mel"] * l_mel + self.w["postnet"] * l_post + self.w["duration"] * l_dur
                 + self.w["pitch"] * l_pitch + self.w["energy"] * l_energy
                 + self.w["tone"] * l_tone + self.w["length"] * l_len)
        return dict(total=total, mel=l_mel, postnet=l_post, duration=l_dur, pitch=l_pitch,
                    energy=l_energy, tone=l_tone, length=l_len)

    def length_contrast(self, out, batch):
        """Hinge: per utterance, mean predicted log-duration of long monophthong vowels
        should exceed that of short vowels by `length_margin` (log domain)."""
        ph = batch["phone"]
        mono = (ph != S.PHONE2ID["ai"]) & (ph != S.PHONE2ID["au"]) & (batch["kind"] == S.KIND_V)
        valid = mono & (~out["src_pad"]) & (batch["duration"] > 0)
        lg = out["log_dur"]
        lm, sm = (valid & (batch["length"] == 1)).float(), (valid & (batch["length"] == 0)).float()
        has = (lm.sum(1) > 0) & (sm.sum(1) > 0)
        if not has.any():
            return lg.new_zeros(())
        mean_l = (lg * lm).sum(1) / lm.sum(1).clamp(min=1)
        mean_s = (lg * sm).sum(1) / sm.sum(1).clamp(min=1)
        return F.relu(self.length_margin - (mean_l - mean_s))[has].mean()


@torch.no_grad()
def diagnostics(out: dict, batch: dict) -> dict:
    """Linguistic diagnostics: tone-probe accuracy on hidden tones and predicted vs. true
    long/short vowel duration ratio."""
    res = {}
    tone_gt, tone_in = batch["tone"], out["tone_in"]
    hid = (tone_in == S.TONE_UNK) & (tone_gt >= S.TONE_H) & (tone_gt <= S.TONE_F) & (~out["src_pad"])
    if hid.any():
        pred = out["tone_logits"].argmax(-1) + S.TONE_H
        res["tone_probe_acc"] = (pred[hid] == tone_gt[hid]).float().mean().item()
    ph = batch["phone"]
    mono = (ph != S.PHONE2ID["ai"]) & (ph != S.PHONE2ID["au"]) & (batch["kind"] == S.KIND_V)
    valid = mono & (~out["src_pad"]) & (batch["duration"] > 0)
    lng, sht = valid & (batch["length"] == 1), valid & (batch["length"] == 0)
    if lng.any() and sht.any():
        pd = torch.exp(out["log_dur"]) - 1.0
        td = batch["duration"].float()
        res["long_short_ratio_pred"] = (pd[lng].mean() / pd[sht].mean().clamp(min=1e-3)).item()
        res["long_short_ratio_true"] = (td[lng].mean() / td[sht].mean().clamp(min=1e-3)).item()
    return res
