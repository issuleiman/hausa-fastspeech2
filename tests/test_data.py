import os

import numpy as np
import pytest
import torch

from hausafs2.data.alignment import AlignmentError, durations_from_intervals, read_textgrid_intervals
from hausafs2.data.mfa import build_dictionary, canonical_words, prepare
from hausafs2.data.preprocess import fit_durations, phoneme_contours, phoneme_mean
from hausafs2.frontend import HausaFrontend
from hausafs2.model.variance_adaptor import contour_to_frames, length_regulate

SR, HOP = 22050, 256
F = HOP / SR


def iv(spec):
    """spec: list of (label, n_frames) -> contiguous intervals"""
    t, out = 0.0, []
    for lab, n in spec:
        out.append((t, t + n * F, lab))
        t += n * F
    return out


def test_alignment_basic_with_pause_and_trim():
    u = HausaFrontend("auto")("ba, ka")
    labels, pz = u.labels, [t.kind == 2 for t in u.tokens]
    assert labels == ["b", "a", "pm", "k", "a"]
    intervals = iv([("", 13), ("b", 5), ("a", 7), ("", 10), ("k", 4), ("a", 8), ("", 13)])
    d, s, e = durations_from_intervals(labels, pz, intervals, SR, HOP)
    assert d.tolist() == [5, 7, 10, 4, 8] and (s, e) == (13, 13 + 34)


def test_pause_without_silence_gets_zero():
    u = HausaFrontend("auto")("ba ka")
    d, *_ = durations_from_intervals(u.labels, [t.kind == 2 for t in u.tokens],
                                     iv([("b", 5), ("a", 7), ("k", 4), ("a", 8)]), SR, HOP)
    assert d.tolist() == [5, 7, 0, 4, 8]


def test_intra_word_silence_goes_to_previous_phone():
    u = HausaFrontend("auto")("ba")
    d, *_ = durations_from_intervals(u.labels, [False, False], iv([("b", 5), ("", 3), ("a", 7)]), SR, HOP)
    assert d.tolist() == [8, 7]


def test_alignment_label_mismatch_raises():
    u = HausaFrontend("auto")("ba")
    with pytest.raises(AlignmentError):
        durations_from_intervals(u.labels, [False, False], iv([("b", 5), ("e", 7)]), SR, HOP)


def test_alignment_gap_raises():
    """Regression: dropped silence intervals used to silently shift all later phones."""
    u = HausaFrontend("auto")("ba, ka")
    gappy = iv([("b", 5), ("a", 7), ("k", 4), ("a", 8)])      # silence interval (comma pause) missing
    gappy = [gappy[0], gappy[1], (gappy[1][1] + 10 * F, gappy[1][1] + 14 * F, "k"), (gappy[1][1] + 14 * F, gappy[1][1] + 22 * F, "a")]
    with pytest.raises(AlignmentError, match="gap"):
        durations_from_intervals(u.labels, [t.kind == 2 for t in u.tokens], gappy, SR, HOP)


def test_textgrid_reader_keeps_silence_intervals(tmp_path):
    """Regression: tgt drops empty-label intervals unless include_empty_intervals=True."""
    import tgt
    tg = tgt.core.TextGrid()
    tier = tgt.core.IntervalTier(name="phones")
    for a, b, lab in [(0, .1, ""), (.1, .2, "b"), (.2, .3, ""), (.3, .4, "a")]:
        tier.add_interval(tgt.core.Interval(a, b, lab))
    tg.add_tier(tier)
    p = os.path.join(tmp_path, "x.TextGrid")
    tgt.io.write_to_file(tg, p, format="long")
    got = read_textgrid_intervals(p)
    assert [g[2] for g in got] == ["", "b", "", "a"]


def test_fit_durations_trims_from_end():
    assert fit_durations(np.array([3, 4, 5]), 10).tolist() == [3, 4, 3]
    assert fit_durations(np.array([3, 4, 5]), 5).tolist() == [3, 2, 0]
    assert fit_durations(np.array([3, 4]), 20).tolist() == [3, 4]


def test_phoneme_contour_roundtrip_for_linear_ramp():
    """A falling F0 inside one long vowel must survive K-point summarisation (K=3 interpolation)."""
    dur = np.array([6, 12, 5])
    frames = np.concatenate([np.full(6, 1.0), np.linspace(1.0, -1.0, 12), np.full(5, -1.0)]).astype(np.float32)
    c = phoneme_contours(frames, dur, 3)
    assert c[1, 0] > 0.7 and c[1, 2] < -0.7 and abs(c[1, 1]) < 0.15
    rec = contour_to_frames(torch.tensor(c)[None], *length_regulate(torch.zeros(1, 3, 1), torch.tensor(dur)[None])[2:5:2])[0].numpy()
    assert np.abs(rec[6:18] - frames[6:18]).max() < 0.25          # smooth fall reproduced
    assert np.abs(phoneme_mean(frames, dur)[1]) < 0.1            # K=1 (phoneme mean) would lose the fall


def test_phoneme_contour_zero_duration_is_safe():
    c = phoneme_contours(np.ones(4, dtype=np.float32), np.array([0, 4, 0]), 3)
    assert c[0].sum() == 0 and c[2].sum() == 0 and np.allclose(c[1], 1)


def test_mfa_dictionary_matches_frontend_labels(tmp_path):
    texts = ["Sàaká gídaa, 'yan kai.", "ruwaa sabbi"]
    d = build_dictionary(texts)
    fe = HausaFrontend("none")
    for t in texts:
        from_dict = [p for w in canonical_words(t) for p in d[w]]
        from_fe = [x for x in fe(t).labels if x not in ("sp", "pm", "sil")]
        assert from_dict == from_fe


def test_prepare_mfa_writes_files(tmp_path):
    import soundfile as sf
    corpus = tmp_path / "corpus"
    (corpus / "wavs" / "s1").mkdir(parents=True)
    sf.write(corpus / "wavs" / "s1" / "u1.wav", np.zeros(2205, dtype=np.float32), SR)
    (corpus / "metadata.csv").write_text("u1|s1|Gídaa kai.\n", encoding="utf-8")
    n = prepare(str(corpus), str(tmp_path / "mfa"))
    assert n == 2
    assert (tmp_path / "mfa" / "s1" / "u1.lab").read_text(encoding="utf-8").strip() == "gidaa kai"
    assert "gidaa\tg i d aː" in (tmp_path / "mfa" / "hausa.dict").read_text(encoding="utf-8")
