"""Text -> mel -> wav.

    python -m hausafs2.synthesize --ckpt output/hausa_single/last.pt \
        --text "Sannu da zuwa." --speaker spkA --vocoder hifigan \
        --hifigan_ckpt vocoder_assets/generator_universal.pth.tar.zip \
        --hifigan_config vocoder_assets/hifigan_config.json --out out.wav

Controls (FastSpeech 2's variance controllability, expressed in speaker-relative units):
  --speed        speaking-rate factor (>1 faster)
  --pitch_ratio  multiplicative F0 shift (1.2 = 20 % higher register)
  --pitch_range  scales tone excursions around the speaker's mean (>1 exaggerates tone)
  --energy_ratio loudness factor
  --predict_tone fill UNKNOWN tones from the model's tone head (for unmarked text)
"""
import argparse
import json
import math
import os
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch

from .data.audio import MelProcessor
from .frontend import HausaFrontend, Utterance
from .frontend import symbols as S
from .model import HausaFastSpeech2, batch_to_device
from .model.fastspeech2_hausa import TOKEN_KEYS
from .model.variance_adaptor import contour_to_frames


def load_checkpoint(path: str, device="cpu"):
    from .train import build_model
    ck = torch.load(path, map_location=device)
    model = build_model(ck["cfg"], ck["stats"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["cfg"], ck["stats"]


def utterance_to_batch(utt: Utterance, speaker_id: int = 0, device="cpu") -> dict:
    arr = utt.arrays()
    b = {k: torch.from_numpy(arr[k]).unsqueeze(0) for k in TOKEN_KEYS}
    b["src_len"] = torch.tensor([len(utt)])
    b["speaker"] = torch.tensor([speaker_id])
    return batch_to_device(b, device)


@torch.no_grad()
def synthesize_mel(model: HausaFastSpeech2, frontend: HausaFrontend, text: str, stats: dict,
                   speaker: Optional[str] = None, speed: float = 1.0, pitch_ratio: float = 1.0,
                   pitch_range: float = 1.0, energy_ratio: float = 1.0, predict_tone: bool = True,
                   device="cpu") -> Dict:
    utt = frontend(text)
    if len(utt) == 0:
        raise ValueError("Empty input after normalisation")
    spk_name = speaker or next(iter(stats["speakers"]))
    sp = stats["speakers"][spk_name]
    controls = dict(
        duration=1.0 / speed,
        pitch_shift=math.log(pitch_ratio) / sp["pitch_std"],       # F0 ratio -> shift in z units
        pitch_range=pitch_range,
        energy_shift=math.log(energy_ratio) / sp["energy_std"] if stats.get("log_energy", True) else 0.0)
    batch = utterance_to_batch(utt, sp["id"], device)
    out = model(batch, controls=controls, predict_tone=predict_tone)
    mel = out["mel_post"][0, : int(out["mel_len"][0])].cpu().numpy()

    # per-frame F0 in Hz (for plotting / inspection), de-normalised with the speaker's own stats
    z = out["pitch"]
    if z.dim() == 3:
        z = contour_to_frames(z * pitch_range + controls["pitch_shift"], out["tok_idx"], out["rel"])
    f0 = np.exp(z[0, : mel.shape[0]].cpu().numpy() * sp["pitch_std"] + sp["pitch_mean"])
    return dict(utt=utt, mel=mel, f0=f0, dur=out["dur"][0].cpu().numpy(),
                tone_used=out["tone_in"][0].cpu().numpy(), pitch_tokens=out["pitch"][0].cpu().numpy(),
                speaker=spk_name)


def plot_result(res: dict, path: str, hop=256, sr=22050):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mel, f0, utt, dur = res["mel"], res["f0"], res["utt"], res["dur"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    a1.imshow(mel.T, origin="lower", aspect="auto", cmap="magma")
    a1.set_ylabel("mel bin")
    a2.plot(np.arange(len(f0)), f0, lw=1.5)
    a2.set_ylabel("F0 (Hz)")
    a2.set_xlabel(f"frame (hop {hop})")
    pos = 0
    for t, d, tone in zip(utt.tokens, dur, res["tone_used"]):
        if d > 0:
            for ax in (a1, a2):
                ax.axvline(pos, color="w" if ax is a1 else "gray", lw=0.3, alpha=0.5)
            if t.kind == S.KIND_V:
                a2.text(pos + d / 2, a2.get_ylim()[1], S.TONE_NAMES[int(tone)], ha="center", va="top", fontsize=9, color="C3")
                a1.text(pos + d / 2, mel.shape[1] - 2, t.label, ha="center", va="top", fontsize=7, color="w")
        pos += int(d)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--vocoder", choices=["hifigan", "griffinlim"], default="hifigan")
    ap.add_argument("--hifigan_ckpt", default="vocoder_assets/generator_universal.pth.tar.zip")
    ap.add_argument("--hifigan_config", default="vocoder_assets/hifigan_config.json")
    ap.add_argument("--tone_mode", default=None, help="override frontend tone_mode")
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--pitch_ratio", type=float, default=1.0)
    ap.add_argument("--pitch_range", type=float, default=1.0)
    ap.add_argument("--energy_ratio", type=float, default=1.0)
    ap.add_argument("--no_predict_tone", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, stats = load_checkpoint(a.ckpt, dev)
    fe_cfg = cfg.get("frontend", {})
    fe = HausaFrontend(a.tone_mode or fe_cfg.get("tone_mode", "auto"), a.lexicon or fe_cfg.get("lexicon_path"))
    res = synthesize_mel(model, fe, a.text, stats, a.speaker, a.speed, a.pitch_ratio, a.pitch_range,
                         a.energy_ratio, not a.no_predict_tone, dev)
    print(res["utt"].describe())
    mp = MelProcessor.from_config(cfg["audio"])
    if a.vocoder == "hifigan":
        from .vocoder import load_hifigan, mel_to_wav
        wav = mel_to_wav(load_hifigan(a.hifigan_ckpt, a.hifigan_config, dev), res["mel"])
    else:
        wav = mp.griffin_lim(res["mel"])
    sf.write(a.out, wav, mp.sr)
    if a.plot:
        plot_result(res, os.path.splitext(a.out)[0] + ".png", mp.hop, mp.sr)
    print(f"wrote {a.out} ({len(wav) / mp.sr:.2f}s)")


if __name__ == "__main__":
    main()
