import numpy as np
import pytest

from hausafs2.data.alignment import durations_from_intervals
from hausafs2.data.ctc_align import chars_for_tokens, spans_to_intervals
from hausafs2.data.hf_export import decode_audio, export_examples
from hausafs2.frontend import HausaFrontend

SR, HOP = 22050, 256


class FakeDecoder:                                  # mimics datasets>=4 AudioDecoder
    class S:
        def __init__(self, a):
            self.data, self.sample_rate = a, 16000

    def __init__(self, a):
        self.a = a

    def get_all_samples(self):
        return self.S(self.a[None, :])


def mk(n_sec, spk, text="Ina kwana", fmt="dict"):
    a = (0.1 * np.sin(np.linspace(0, 300, int(16000 * n_sec)))).astype(np.float32)
    return dict(audio=dict(array=a, sampling_rate=16000) if fmt == "dict" else FakeDecoder(a),
                text=text, speaker_id=spk)


def test_decode_audio_both_formats():
    for fmt in ("dict", "dec"):
        arr, sr = decode_audio(mk(1.0, 1, fmt=fmt)["audio"])
        assert arr.ndim == 1 and sr == 16000 and len(arr) == 16000


def test_export_filters_and_limits(tmp_path):
    ex = [mk(0.5, 1), mk(2.0, 1), mk(2.0, 1), mk(2.0, 2), mk(20.0, 2), mk(2.0, 2, text=""), mk(2.0, 3, fmt="dec")]
    rows = export_examples(ex, str(tmp_path), n=20, max_per_speaker=2)
    assert [r[1] for r in rows] == ["spk1", "spk1", "spk2", "spk3"]      # too short/long/empty skipped, cap per speaker
    meta = (tmp_path / "metadata.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(meta) == 4 and meta[0].count("|") == 2
    import soundfile as sf
    assert sf.info(tmp_path / "wavs" / "spk1" / "spk1_0000.wav").samplerate == 22050


def test_chars_for_tokens_romanisation():
    u = HausaFrontend("none")("ɓaƙaa ƴa shi")
    chars, idx = chars_for_tokens(u.labels, [t.kind == 2 for t in u.tokens])
    assert "".join(chars).startswith("bakaa"[:3]) and all(c.isalpha() or c == "'" for s in chars for c in s)
    assert len(chars) == len(idx) == sum(t.kind != 2 for t in u.tokens)


def test_spans_to_intervals_roundtrip_through_alignment():
    u = HausaFrontend("none")("ba ka")
    labels, pz = u.labels, [t.kind == 2 for t in u.tokens]
    chars, idx = chars_for_tokens(labels, pz)
    nonp = [labels[i] for i in idx]
    # per-character starts (seconds): b a | k a with a 0.5 s pause between words
    spans = [(0.10, 0.12), (0.14, 0.16), (0.70, 0.72), (0.74, 0.80)]
    iv = spans_to_intervals(chars, spans, nonp, total_end=0.9)
    d, s, e = durations_from_intervals(labels, pz, iv, SR, HOP)
    assert all(x >= 0 for x in d) and d[labels.index("sp")] > 0        # the long gap became the pause token
    assert d[1] <= round(0.25 * SR / HOP) + 1                           # vowel capped, not stretched over the pause
    assert (e - s) == d.sum()


def test_spans_to_intervals_requires_matching_char_count():
    with pytest.raises(AssertionError):
        spans_to_intervals(["ba"], [(0.0, 0.1)], ["b"], 1.0)
