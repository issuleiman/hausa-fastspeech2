"""HiFi-GAN generator (inference only).

Architecture adapted from https://github.com/jik876/hifi-gan (MIT, Jungil Kong et al.)
via https://github.com/ming024/FastSpeech2 (MIT, Chung-Ming Chien), so that their
released generator checkpoints (LJSpeech / universal) load unchanged. Mel settings
must match: 22.05 kHz, n_fft 1024, hop 256, win 1024, 80 mels, fmin 0, fmax 8000.
"""
import io
import json
import warnings
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self


def _weight_norm(m, legacy: bool):
    if legacy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nn.utils.weight_norm(m)
    return nn.utils.parametrizations.weight_norm(m)


def _init(m, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(0.0, std)


def _pad(k, d=1):
    return int((k * d - d) / 2)


class ResBlock(nn.Module):
    def __init__(self, ch, k=3, dil=(1, 3, 5), legacy=True):
        super().__init__()
        wn = lambda m: _weight_norm(m, legacy)  # noqa: E731
        self.convs1 = nn.ModuleList([wn(Conv1d(ch, ch, k, 1, dilation=d, padding=_pad(k, d))) for d in dil])
        self.convs2 = nn.ModuleList([wn(Conv1d(ch, ch, k, 1, dilation=1, padding=_pad(k, 1))) for _ in dil])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE)) + x
        return x


class Generator(nn.Module):
    def __init__(self, h, legacy: bool = True):
        super().__init__()
        wn = lambda m: _weight_norm(m, legacy)  # noqa: E731
        self.num_kernels, self.num_ups = len(h.resblock_kernel_sizes), len(h.upsample_rates)
        c0 = h.upsample_initial_channel
        self.conv_pre = wn(Conv1d(h.num_mels if "num_mels" in h else 80, c0, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            wn(ConvTranspose1d(c0 // 2 ** i, c0 // 2 ** (i + 1), k, u, padding=(k - u) // 2))
            for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes))])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = c0 // 2 ** (i + 1)
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d, legacy))
        self.conv_post = wn(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(_init)
        self.conv_post.apply(_init)

    def forward(self, x):                       # x: (B, 80, T) natural-log mel
        x = self.conv_pre(x)
        for i in range(self.num_ups):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            xs = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


def _remap_legacy_keys(sd):
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_g"):
            k = k[:-len("weight_g")] + "parametrizations.weight.original0"
        elif k.endswith(".weight_v"):
            k = k[:-len("weight_v")] + "parametrizations.weight.original1"
        out[k] = v
    return out


def load_hifigan(ckpt_path: str, config_path: str, device="cpu", legacy=None) -> Generator:
    """Load a HiFi-GAN generator from `.pth.tar` or the `.zip` released by ming024/FastSpeech2."""
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    if legacy is None:
        legacy = hasattr(nn.utils, "weight_norm")
    gen = Generator(h, legacy).to(device)
    if ckpt_path.endswith(".zip"):
        with zipfile.ZipFile(ckpt_path) as z:
            name = [n for n in z.namelist() if not n.endswith("/")][0]
            ck = torch.load(io.BytesIO(z.read(name)), map_location=device)
    else:
        ck = torch.load(ckpt_path, map_location=device)
    sd = ck["generator"] if "generator" in ck else ck
    gen.load_state_dict(sd if legacy else _remap_legacy_keys(sd))
    return gen.eval()


@torch.no_grad()
def mel_to_wav(gen: Generator, mel_log, max_wav_value: float = 32768.0):
    """mel_log: (T, 80) numpy/tensor -> float32 numpy waveform in [-1, 1]."""
    m = torch.as_tensor(mel_log, dtype=torch.float32).T.unsqueeze(0).to(next(gen.parameters()).device)
    return gen(m).squeeze().cpu().numpy().astype("float32")
