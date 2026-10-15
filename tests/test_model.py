import copy

import numpy as np
import pytest
import torch

from hausafs2.frontend import HausaFrontend
from hausafs2.frontend import symbols as S
from hausafs2.model import HausaFastSpeech2, HausaLoss
from hausafs2.model.encoder import apply_tone_dropout, downstep_counts
from hausafs2.model.fastspeech2_hausa import TOKEN_KEYS
from hausafs2.model.hier_prosody import segment_broadcast, segment_mean
from hausafs2.model.variance_adaptor import contour_to_frames, length_regulate

torch.manual_seed(0)


def small_cfg(pitch_mode="contour", energy_mode="phoneme", multi=False, skip=True, hier=True):
    return dict(
        hidden=32, encoder=dict(layers=1, heads=2, conv_filter=64, conv_kernel=[9, 1], dropout=0.0),
        decoder=dict(layers=1, heads=2, conv_filter=64, conv_kernel=[9, 1], dropout=0.0),
        hier=dict(enabled=hier, layers=1, heads=2, conv_filter=64, dropout=0.0),
        variance_predictor=dict(filter=32, kernel=3, dropout=0.0),
        pitch=dict(mode=pitch_mode, n_points=3, n_bins=32), energy=dict(mode=energy_mode, n_bins=32),
        postnet=dict(channels=16, layers=2), multi_speaker=multi, pitch_tone_skip=skip,
        tone=dict(dropout_word=0.0, dropout_utt=0.0))


STATS = dict(pitch_range=[-3, 3], energy_range=[-3, 3], n_mels=20, n_speakers=2)


def make_batch(texts, K=3, pitch_mode="contour", energy_mode="phoneme", seed=0):
    g = torch.Generator().manual_seed(seed)
    fe = HausaFrontend("auto")
    arrs = [fe(t).arrays() for t in texts]
    N = max(len(a["phone"]) for a in arrs)
    b = {}
    for k in TOKEN_KEYS:
        pad = -1 if k.endswith("_idx") else 0
        b[k] = torch.tensor(np.stack([np.pad(a[k], (0, N - len(a[k])), constant_values=pad) for a in arrs]))
    b["src_len"] = torch.tensor([len(a["phone"]) for a in arrs])
    B = len(texts)
    valid = torch.arange(N)[None] < b["src_len"][:, None]
    dur = torch.randint(1, 5, (B, N), generator=g) * valid
    b["duration"] = dur
    T = int(dur.sum(1).max())
    b["mel"] = torch.randn(B, T, 20, generator=g)
    b["pitch"] = torch.randn(B, N, K, generator=g) if pitch_mode == "contour" else torch.randn(B, T, generator=g)
    b["energy"] = torch.randn(B, N, generator=g) if energy_mode == "phoneme" else torch.randn(B, T, generator=g)
    b["speaker"] = torch.zeros(B, dtype=torch.long)
    return b


def test_length_regulate_matches_reference_loop():
    h = torch.randn(3, 5, 4)
    dur = torch.tensor([[2, 0, 3, 1, 0], [1, 1, 1, 1, 1], [0, 0, 4, 0, 0]])
    out, mel_len, idx, valid, rel = length_regulate(h, dur)
    assert mel_len.tolist() == [6, 5, 4]
    for b in range(3):
        ref = torch.repeat_interleave(h[b], dur[b], dim=0)
        assert torch.allclose(out[b, : len(ref)], ref)
        assert out[b, len(ref):].abs().sum() == 0
        assert valid[b].sum() == len(ref)
    # relative position increases from ~0 to ~1 within a token
    assert rel[2, 0] < 0.2 and rel[2, 3] > 0.8


def test_contour_interpolation_is_linear_and_K1_is_constant():
    h = torch.zeros(1, 2, 1)
    dur = torch.tensor([[4, 2]])
    _, _, idx, valid, rel = length_regulate(h, dur)
    c = torch.tensor([[[0.0, 1.0], [5.0, 5.0]]])
    f = contour_to_frames(c, idx, rel)[0]
    assert torch.all(f[:3] >= f[:3].min()) and f[0] < f[1] < f[2] < f[3]       # rises across token 0
    assert torch.allclose(f[4:6], torch.tensor([5.0, 5.0]))
    c1 = torch.tensor([[[2.0], [7.0]]])
    assert torch.allclose(contour_to_frames(c1, idx, rel)[0], torch.tensor([2., 2, 2, 2, 7, 7]))


def test_segment_pool_and_broadcast():
    x = torch.tensor([[[1.0], [3.0], [10.0], [100.0]]])
    idx = torch.tensor([[0, 0, 1, -1]])
    pooled, cnt = segment_mean(x, idx, 2)
    assert pooled.flatten().tolist() == [2.0, 10.0] and cnt.flatten().tolist() == [2, 1]
    back = segment_broadcast(pooled, idx).flatten().tolist()
    assert back == [2.0, 2.0, 10.0, 0.0]


def test_downstep_counter_resets_per_phrase():
    L, H, F, N_ = S.TONE_L, S.TONE_H, S.TONE_F, S.TONE_NONE
    tone = torch.tensor([[H, L, H, F, H, H, L, H]])
    phrase = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]])
    got = downstep_counts(tone, phrase).tolist()[0]
    assert got == [0, 0, 1, 1, 2, 0, 0, 1]


def test_tone_dropout_only_hides_known_tones():
    tone = torch.tensor([[S.TONE_NONE, S.TONE_H, S.TONE_L, S.TONE_UNK]])
    word = torch.tensor([[0, 0, 1, 1]])
    out = apply_tone_dropout(tone, word, p_word=1.0, p_utt=0.0)
    assert out.tolist() == [[S.TONE_NONE, S.TONE_UNK, S.TONE_UNK, S.TONE_UNK]]
    assert apply_tone_dropout(tone, word, 0.0, 0.0).equal(tone)


@pytest.mark.parametrize("pm,em,multi,skip,hier", [
    ("contour", "phoneme", False, True, True), ("contour", "frame", True, False, True),
    ("frame", "phoneme", False, True, False), ("frame", "frame", True, True, True)])
def test_forward_backward_and_inference(pm, em, multi, skip, hier):
    m = HausaFastSpeech2(small_cfg(pm, em, multi, skip, hier), STATS)
    b = make_batch(["gida ruwa, kai.", "sarki"], pitch_mode=pm, energy_mode=em)
    b["speaker"] = torch.tensor([0, 1])
    out = m(b)
    L = HausaLoss({"weights": {"length": 0.1}}, pm, em)(out, b)
    assert torch.isfinite(L["total"])
    L["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert out["mel_post"].shape == b["mel"].shape
    m.eval()
    inf = {k: v for k, v in b.items() if k not in ("duration", "mel", "pitch", "energy")}
    with torch.no_grad():
        o = m(inf, predict_tone=True)
    assert torch.isfinite(o["mel_post"]).all() and o["mel_post"].size(2) == 20


def test_padding_invariance_in_eval():
    """A sample must give identical output alone and when padded inside a bigger batch."""
    m = HausaFastSpeech2(small_cfg(), STATS).eval()
    texts = ["sarki gida", "kai"]
    big = make_batch(texts)
    solo = {k: (v[1:2] if torch.is_tensor(v) else v) for k, v in big.items()}
    n = int(big["src_len"][1])
    solo = {k: (v[:, :n] if k in TOKEN_KEYS or k in ("duration", "pitch", "energy") else v) for k, v in solo.items()}
    T = int(solo["duration"].sum())
    solo["mel"] = big["mel"][1:2, :T]
    with torch.no_grad():
        a, b = m(big), m(solo)
    T = b["mel_post"].size(1)
    assert torch.allclose(a["mel_post"][1, :T], b["mel_post"][0], atol=1e-4)
    assert a["mel_post"][1, T:].abs().sum() == 0


def test_tone_changes_pitch_prediction_path():
    """Changing the tone input must change predictions (tone is wired into the model)."""
    m = HausaFastSpeech2(small_cfg(), STATS).eval()
    fe = HausaFrontend("auto")
    ba, bb = make_batch(["ɓàal"]), make_batch(["ɓáal"])
    for k in ("duration", "mel", "pitch", "energy"):
        bb[k] = ba[k]
    with torch.no_grad():
        pa, pb = m(ba)["pitch"], m(bb)["pitch"]
    assert not torch.allclose(pa, pb)


def test_length_contrast_hinge():
    cfg = small_cfg()
    m = HausaFastSpeech2(cfg, STATS)
    b = make_batch(["ɓaa ɓa"])
    out = m(b)
    lf = HausaLoss({"weights": {"length": 1.0}, "length_margin": 5.0}, "contour", "phoneme")
    assert lf.length_contrast(out, b) > 0                       # margin 5 cannot be met by an untrained net
    lf0 = HausaLoss({"weights": {"length": 1.0}, "length_margin": -50.0}, "contour", "phoneme")
    assert lf0.length_contrast(out, b) == 0


def test_speed_control_scales_duration():
    m = HausaFastSpeech2(small_cfg(), STATS).eval()
    b = make_batch(["gida ruwa kai"])
    inf = {k: v for k, v in b.items() if k not in ("duration", "mel", "pitch", "energy")}
    with torch.no_grad():
        d1 = m(inf, controls=dict(duration=1.0))["dur"].sum()
        d2 = m(inf, controls=dict(duration=3.0))["dur"].sum()
    assert d2 >= d1


@pytest.mark.parametrize("K", [1, 2, 5])
def test_contour_mode_supports_any_K_including_phoneme_level(K):
    """Regression: K=1 (phoneme-level pitch) used to crash because the predictor squeezed its last dim."""
    cfg = small_cfg()
    cfg["pitch"]["n_points"] = K
    m = HausaFastSpeech2(cfg, STATS)
    b = make_batch(["gida ruwa, kai."], K=K)
    out = m(b)
    assert out["pitch"].shape == b["pitch"].shape
    HausaLoss({}, "contour", "phoneme")(out, b)["total"].backward()


def test_baseline_switches_run_and_ignore_disabled_features():
    cfg = small_cfg(hier=False, skip=False)
    cfg.update(input_features=[], use_downstep=False)
    m = HausaFastSpeech2(cfg, STATS).eval()
    a, b = make_batch(["ɓàal"]), make_batch(["ɓáal"])
    for k in ("duration", "mel", "pitch", "energy"):
        b[k] = a[k]
    with torch.no_grad():
        assert torch.allclose(m(a)["mel_post"], m(b)["mel_post"])        # tone input is ignored when disabled


def test_partial_teacher_forcing_uses_predicted_pitch_when_absent():
    m = HausaFastSpeech2(small_cfg(), STATS).eval()
    b = make_batch(["gida ruwa"])
    feed = {k: v for k, v in b.items() if k not in ("pitch", "energy")}
    with torch.no_grad():
        out = m(feed)
    assert out["mel_post"].shape == b["mel"].shape                       # GT durations -> same length
