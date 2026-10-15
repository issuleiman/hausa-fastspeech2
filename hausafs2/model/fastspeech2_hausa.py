"""HausaFastSpeech2: FastSpeech 2 with explicit Hausa tone, vowel-length, syllable-weight
and hierarchical-prosody modelling.

  tokens ──► LinguisticEncoder ──► HierarchicalProsodyEncoder ──(+speaker)──►
         VarianceAdaptor (duration / tone-aware pitch contour / energy) ──►
         FFT mel decoder ──► Linear ──► (+PostNet) ──► mel
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..frontend import symbols as S
from .encoder import LinguisticEncoder, apply_tone_dropout, downstep_counts
from .hier_prosody import HierarchicalProsodyEncoder
from .layers import FFTStack, lengths_to_mask
from .postnet import PostNet
from .variance_adaptor import VarianceAdaptor

TOKEN_KEYS = ["phone", "length", "tone", "role", "weight", "kind", "boundary",
              "syl_pos", "word_pos", "phrase_pos", "syl_idx", "word_idx", "phrase_idx"]


class HausaFastSpeech2(nn.Module):
    def __init__(self, cfg: dict, stats: dict):
        """cfg: the `model` section of the yaml config.
        stats: dict(pitch_range=[lo,hi], energy_range=[lo,hi], n_mels=int, n_speakers=int)."""
        super().__init__()
        self.cfg = cfg
        d = cfg["hidden"]
        enc, dec = cfg["encoder"], cfg["decoder"]
        max_len = cfg.get("max_len", 2000)

        self.encoder = LinguisticEncoder(
            d, enc["layers"], enc["heads"], enc["conv_filter"], enc["conv_kernel"], enc["dropout"],
            max_len=max_len, use_downstep=cfg.get("use_downstep", True),
            features=cfg.get("input_features"))

        hier = cfg.get("hier", {})
        self.hier = None
        if hier.get("enabled", True):
            self.hier = HierarchicalProsodyEncoder(
                d, hier.get("heads", 2), hier.get("conv_filter", 2 * d), hier.get("conv_kernel", [3, 1]),
                hier.get("dropout", 0.1), hier.get("layers", 1),
                hier.get("levels", ["syllable", "word", "phrase"]))

        self.variance_adaptor = VarianceAdaptor(
            d, cfg["variance_predictor"], cfg["pitch"], cfg["energy"],
            stats["pitch_range"], stats["energy_range"])

        self.decoder = FFTStack(d, dec["layers"], dec["heads"], dec["conv_filter"], dec["conv_kernel"],
                                dec["dropout"], max_len)
        self.mel_linear = nn.Linear(d, stats["n_mels"])
        pn = cfg.get("postnet", {})
        self.postnet = PostNet(stats["n_mels"], pn.get("channels", 512), pn.get("layers", 5),
                               pn.get("kernel", 5)) if pn.get("enabled", True) else None

        self.speaker_emb = self.speaker_dec = None
        if cfg.get("multi_speaker", False):
            n = int(stats["n_speakers"])
            self.speaker_emb = nn.Embedding(n, d)
            if cfg.get("speaker_to_decoder", True):
                self.speaker_dec = nn.Embedding(n, d)

        # Explicit tone -> pitch path: tone (+downstep) embeddings are added directly to the pitch
        # predictor's input, so F0 does not depend on tone surviving the deep encoder.
        self.pitch_tone_skip = bool(cfg.get("pitch_tone_skip", True)) and cfg["pitch"].get("mode", "contour") == "contour"
        if self.pitch_tone_skip:
            self.skip_tone = nn.Embedding(S.N_TONES, d)
            self.skip_down = nn.Embedding(S.N_DOWNSTEP, d)
            nn.init.normal_(self.skip_tone.weight, std=0.3)
            nn.init.normal_(self.skip_down.weight, std=0.3)

        tone_cfg = cfg.get("tone", {})
        self.tone_dropout_word = float(tone_cfg.get("dropout_word", 0.0))   # see docs/RESULTS.md: dropout costs accuracy
        self.tone_dropout_utt = float(tone_cfg.get("dropout_utt", 0.0))

    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor], controls: Optional[dict] = None,
                predict_tone: bool = False, hide_tones: bool = False) -> Dict[str, torch.Tensor]:
        """batch holds token features (+ `speaker`, `src_len`). If it also holds
        `duration`/`pitch`/`energy` they are used as conditioning (teacher forcing,
        as in the paper); otherwise predicted values are used (inference)."""
        N = batch["phone"].size(1)
        src_pad = lengths_to_mask(batch["src_len"], N)
        tone_gt = batch["tone"]
        tone_in = tone_gt
        if hide_tones:                                     # simulate fully unmarked text
            known = (tone_gt != S.TONE_NONE)
            tone_in = torch.where(known, torch.full_like(tone_gt, S.TONE_UNK), tone_gt)
        elif self.training and (self.tone_dropout_word > 0 or self.tone_dropout_utt > 0):
            tone_in = apply_tone_dropout(tone_gt, batch["word_idx"], self.tone_dropout_word,
                                         self.tone_dropout_utt)

        enc = self.encoder(batch, tone_in, src_pad)
        tone_logits = self.encoder.tone_probe(enc)
        if predict_tone:                                   # two-pass: fill unknown tones, re-encode
            tone_in = self.encoder.fill_unknown_tones(enc, tone_in)
            enc = self.encoder(batch, tone_in, src_pad)
        h = self.hier(enc, batch, src_pad) if self.hier is not None else enc
        if self.speaker_emb is not None:
            h = h + self.speaker_emb(batch["speaker"]).unsqueeze(1)

        max_len = batch["mel"].size(1) if "mel" in batch else None
        cond = None
        if self.pitch_tone_skip:
            cond = self.skip_tone(tone_in) + self.skip_down(downstep_counts(tone_in, batch["phrase_idx"]))
        va = self.variance_adaptor(h, src_pad, batch, controls, max_len, pitch_cond=cond)
        x = va["x"]
        if self.speaker_dec is not None:
            x = x + self.speaker_dec(batch["speaker"]).unsqueeze(1)
        mel_pad = va["mel_pad"]
        x = self.decoder(x, mel_pad).masked_fill(mel_pad.unsqueeze(-1), 0.0)
        mel = self.mel_linear(x)
        mel_post = mel + self.postnet(mel, mel_pad) if self.postnet is not None else mel
        mel_post = mel_post.masked_fill(mel_pad.unsqueeze(-1), 0.0)
        return dict(mel=mel, mel_post=mel_post, src_pad=src_pad, tone_in=tone_in,
                    tone_logits=tone_logits, **{k: v for k, v in va.items() if k != "x"})


def batch_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
