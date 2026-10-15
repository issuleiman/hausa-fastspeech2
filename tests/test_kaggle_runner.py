import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hke", ROOT / "kaggle" / "hausa_kaggle_experiment.py")
hke = importlib.util.module_from_spec(_spec)
sys.modules["hke"] = hke
_spec.loader.exec_module(hke)


# ------------------------------------------------------------------ alignment maths
def test_ctc_viterbi_recovers_known_alignment():
    V, blank = 6, 0
    targets = [1, 2, 2, 3]                     # repeated symbol needs a blank between
    frames = [0, 1, 1, 0, 2, 0, 2, 2, 0, 3, 3, 0]
    logp = np.full((len(frames), V), np.log(0.01))
    for t, s in enumerate(frames):
        logp[t, s] = np.log(0.95)
    path, ext = hke.ctc_viterbi(logp, targets, blank)
    spans, score = hke.path_to_char_spans(path, ext, logp, 0.02)
    assert [round(a / 0.02) for a, _ in spans] == [1, 4, 6, 9]
    assert [round(b / 0.02) for _, b in spans] == [3, 5, 8, 11]
    assert score > np.log(0.5)


def test_ctc_viterbi_infeasible_returns_none():
    logp = np.log(np.full((3, 5), 0.2))
    path, _ = hke.ctc_viterbi(logp, [1, 2, 3, 4], 0)
    assert path is None


def test_tokens_to_vocab_chars_fallbacks():
    vocab = {c: i for i, c in enumerate("abdikstwy'|")}          # no ɓ ɗ ƙ ƴ in this vocabulary
    labels = ["ɓ", "aː", "bː", "ai", "sh", "ƙ", "sp", "ʔ"]
    out = hke.tokens_to_vocab_chars(labels, [False, False, False, False, False, False, True, False], vocab)
    assert out == ["b", "a", "bb", "ai", "s", "k", "'"]         # sh -> 's' (h missing), pause skipped
    with pytest.raises(ValueError):
        hke.tokens_to_vocab_chars(["ɓ"], [False], {"z": 0})


# ------------------------------------------------------------------ selection / stats / state
def _fake_meta(n_spk=6, per=40, seed=0):
    import pandas as pd
    rng = np.random.RandomState(seed)
    rows = []
    shared = [f"ina kwana {chr(97 + i)}{chr(97 + j)}" for i in range(6) for j in range(6)]
    for s in range(1, n_spk + 1):
        for k in range(per):
            rows.append(dict(speaker_id=s, text=shared[rng.randint(len(shared))] if k < 20 else f"jumla {s} {k} ta musamman",
                             duration=float(rng.uniform(2, 6)), gender="male" if s % 2 else "female"))
    return pd.DataFrame(rows)


def test_choose_subset_is_text_disjoint_and_gender_balanced():
    from hausafs2.frontend import HausaFrontend
    P = dict(min_dur=1.5, max_dur=10, utts_per_speaker=12, test_per_speaker=4, n_speakers=4)
    sel, table = hke.choose_subset(_fake_meta(), P, HausaFrontend("none"), [])
    tr, te = sel[sel.role == "train"], sel[sel.role == "test"]
    assert set(tr.norm) & set(te.norm) == set()                  # no sentence appears in both
    assert sel.speaker.nunique() == 4
    g = sel.drop_duplicates("speaker").gender.value_counts()
    assert g["male"] == 2 and g["female"] == 2
    assert len(te) == 16 and sel.id.is_unique and te.id.str.contains("_te_").all()
    assert len(table) == 6


def test_choose_subset_force_speakers():
    from hausafs2.frontend import HausaFrontend
    P = dict(min_dur=1.5, max_dur=10, utts_per_speaker=5, test_per_speaker=2, n_speakers=99)
    sel, _ = hke.choose_subset(_fake_meta(), P, HausaFrontend("none"), [2, 5])
    assert sorted(sel.speaker.unique()) == [2, 5]


def test_merge_states_done_wins_and_progress_advances():
    a = {"stages": {"align": {"done": False}}, "runs": {"r1": {"status": "running", "step": 100},
                                                        "r2": {"status": "done", "step": 500}}, "updated": 1}
    b = {"stages": {"align": {"done": True, "ok": 3}, "subset": {"done": True}},
         "runs": {"r1": {"status": "running", "step": 300}, "r2": {"status": "running", "step": 900},
                  "r3": {"status": "done", "step": 5}}, "updated": 2, "repo_commit": "abc"}
    m = hke.merge_states(a, b)
    assert m["stages"]["align"]["done"] and m["stages"]["subset"]["done"]
    assert m["runs"]["r1"]["step"] == 300 and m["runs"]["r2"]["status"] == "done" and "r3" in m["runs"]
    assert m["repo_commit"] == "abc" and m["updated"] == 2


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.RandomState(0)
    shifted = hke.paired_bootstrap(rng.normal(-0.5, 0.3, 200))
    null = hke.paired_bootstrap(rng.normal(0.0, 0.3, 200))
    assert shifted["hi"] < 0 and null["lo"] < 0 < null["hi"]


def test_localhub_roundtrip_and_pending(tmp_path):
    hub = hke.LocalHub(tmp_path / "hub")
    f = tmp_path / "x.txt"
    f.write_text("hi")
    assert hub.put_many([(f, "a/b/x.txt")]) and hub.list("a/") == ["a/b/x.txt"]
    out = tmp_path / "y.txt"
    assert hub.get("a/b/x.txt", out) and out.read_text() == "hi"
    assert not hub.get("nope", out)
    hub.pending["a/b/x.txt"] = str(f)
    assert hub.flush_pending() and not hub.pending


def test_hfhub_rate_limit_defers_instead_of_blocking(tmp_path, monkeypatch):
    class FakeApi:
        def create_commit(self, **kw):
            raise RuntimeError("429 Too Many Requests: rate limit for repository commits")
    hub = hke.HfHub.__new__(hke.HfHub)
    hub.repo, hub.token, hub.api, hub.pending = "u/r", "t", FakeApi(), {}
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"1")
    monkeypatch.setattr(hke.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("must not sleep")))
    assert hub.put_many([(f, "runs/x/last.pt")], "m") is False
    assert hub.pending == {"runs/x/last.pt": str(f)}


# ------------------------------------------------------------------ end-to-end with simulated sessions
class SimulatedCrash(Exception):
    pass


class OracleAligner:
    """Returns the known synthetic alignment (keyed by token labels) and counts calls."""

    def __init__(self, table):
        self.table, self.calls = table, 0

    def align(self, w16, labels, is_pause):
        self.calls += 1
        return self.table[tuple(labels)], 0.0


def _build_fake_world(tmp: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from hausafs2.data.alignment import read_textgrid_intervals
    from hausafs2.data.synthetic import write_corpus
    from hausafs2.frontend import HausaFrontend
    corpus, align, _, _ = write_corpus(str(tmp / "syn"), n_per_speaker=16, n_words=30, seed=0, p_new=0.5)
    fe = HausaFrontend("auto")
    rows, table = [], {}
    for line in (Path(corpus) / "metadata.csv").read_text(encoding="utf-8").splitlines():
        uid, spk, text = line.split("|", 2)
        wav = (Path(corpus) / "wavs" / spk / f"{uid}.wav").read_bytes()
        import soundfile as sf
        info = sf.info(io.BytesIO(wav))
        rows.append(dict(audio={"bytes": wav, "path": f"{uid}.wav"}, text=text, speaker_id=1 if spk == "spkA" else 2,
                         duration=info.duration, gender="male" if spk == "spkA" else "female"))
        table[tuple(fe(" ".join(text.split())).labels)] = read_textgrid_intervals(
            str(Path(align) / spk / f"{uid}.TextGrid"))
    d = tmp / "parquet"
    d.mkdir()
    half = len(rows) // 2
    for i, chunk in enumerate((rows[:half], rows[half:])):
        t = pa.table({"audio": pa.array([r["audio"] for r in chunk], type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                      "text": [r["text"] for r in chunk], "speaker_id": pa.array([r["speaker_id"] for r in chunk], pa.int64()),
                      "gender": [r["gender"] for r in chunk], "duration": pa.array([r["duration"] for r in chunk], pa.float32())})
        pq.write_table(t, d / f"train-0000{i}.parquet", row_group_size=8)
    return hke.LocalParquetSource(d), table


def _P():
    P = hke.get_params("smoke")
    P.update(n_speakers=2, utts_per_speaker=10, test_per_speaker=3, min_dur=0.3, max_dur=30.0, steps=40, seg=20,
             hidden=32, enc_layers=1, dec_layers=1, conv_filter=64, vp_filter=32, n_bins=32, postnet_channels=16,
             postnet_layers=2, val_size=3, ckpt_upload_min=0.0, n_samples=1, seeds=[1234],
             arms=["fs2_original", "hausa_full"])
    return P


@pytest.mark.slow
def test_three_sessions_stop_crash_resume(tmp_path):
    source, table = _build_fake_world(tmp_path)
    hub_dir = tmp_path / "hub"                                     # the only thing that survives between sessions

    def session(n, budget_hours=None, crash=None):
        al = OracleAligner(table)
        exp = hke.Experiment(_P(), hke.LocalHub(hub_dir), source, al, tmp_path / f"disk{n}", ROOT,
                             device="cpu", budget_hours=budget_hours, force_speakers=[])
        if crash:
            def hook(rid, step):
                if rid.startswith(crash[0]) and step == crash[1]:
                    raise SimulatedCrash("kaggle died")
            exp.crash_hook = hook
        return exp, al

    # session 1: runs out of time right after subset selection (budget < safety margin)
    e1, a1 = session(1, budget_hours=0.0)
    assert e1.run() == "stopped" and a1.calls == 0
    assert (hub_dir / "exp/exp1/subset.tar").exists()

    # session 2: new empty disk; aligns + preprocesses; crashes in the middle of the 2nd model
    e2, a2 = session(2, crash=("hausa_full", 20))
    with pytest.raises(SimulatedCrash):
        e2.run()
    n_aligned = a2.calls
    assert n_aligned >= 20 and not any(ev[0] == "resume" for ev in e2.events)
    state = json.loads((hub_dir / "exp/exp1/state.json").read_text())
    assert state["stages"]["align"]["done"] and state["stages"]["preprocess"]["done"]
    done = [r for r in state["runs"].values() if r["status"] == "done"]
    running = [r for r in state["runs"].values() if r["status"] == "running"]
    assert len(done) == 1 and done[0]["arm"] == "fs2_original" and running[0]["step"] == 20
    assert any(str(x).endswith("last.pt") for x in hub_dir.rglob("*"))

    # session 3: new empty disk again; must NOT redo alignment/subset/preprocess/finished run
    e3, a3 = session(3)
    assert e3.run() == "complete"
    assert a3.calls == 0                                            # alignment not repeated
    resumes = [ev for ev in e3.events if ev[0] == "resume"]
    assert len(resumes) == 1 and resumes[0][1].startswith("hausa_full") and resumes[0][2] == 20
    state = json.loads((hub_dir / "exp/exp1/state.json").read_text())
    assert sum(r["status"] == "done" for r in state["runs"].values()) == 2
    files = set(map(str, (p.relative_to(hub_dir) for p in hub_dir.rglob("*") if p.is_file())))
    assert "exp/exp1/report/report.md" in files and "exp/exp1/report/results_table.tex" in files
    assert "exp/exp1/samples.tar" in files
    assert not any(f.endswith("last.pt") for f in files)            # resume checkpoints cleaned after completion
    assert sum(f.endswith("result.json") for f in files) == 2
    rep = (hub_dir / "exp/exp1/report/report.md").read_text()
    assert "fs2_original" in rep and "hausa_full" in rep and "Paired bootstrap" in rep
    res = json.loads(next(p for p in hub_dir.rglob("result.json")).read_text())
    assert res["n_test"] == 6 and all(np.isfinite(res["metrics"][m]) for m in hke.METRICS if m != "pitch_corr")


# ------------------------------------------------------------------ Hub wrapper error semantics (stubbed network)
def test_hfhub_get_distinguishes_missing_from_network_error(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError
    hub = hke.HfHub.__new__(hke.HfHub)
    hub.repo, hub.token, hub.api, hub.pending = "u/r", "t", None, {}
    monkeypatch.setattr(hke.time, "sleep", lambda s: None)

    def missing(**kw):
        raise EntryNotFoundError("404 Entry Not Found")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert hub.get("state.json", tmp_path / "s.json") is False          # absent -> False (start fresh)

    def flaky(**kw):
        raise ConnectionError("connection reset")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", flaky)
    with pytest.raises(RuntimeError):                                      # network trouble -> raise, NEVER 'absent'
        hub.get("state.json", tmp_path / "s.json")


def test_hfhub_put_many_builds_one_commit(tmp_path):
    calls = []

    class FakeApi:
        def create_commit(self, **kw):
            calls.append(kw)
    hub = hke.HfHub.__new__(hke.HfHub)
    hub.repo, hub.token, hub.api, hub.pending = "u/r", "t", FakeApi(), {}
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("1"); b.write_text("2")
    assert hub.put_many([(a, "x/a"), (b, "x/b")], "msg")
    assert len(calls) == 1 and [o.path_in_repo for o in calls[0]["operations"]] == ["x/a", "x/b"]
    assert calls[0]["repo_type"] == "model" and calls[0]["repo_id"] == "u/r"


# ------------------------------------------------------------------ MMS aligner glue with stub model/processor
def test_mms_aligner_glue_with_stub_model():
    import torch
    from hausafs2.data.alignment import durations_from_intervals
    from hausafs2.frontend import HausaFrontend
    fe = HausaFrontend("none")
    utt = fe("ɓa ka")
    labels, pz = utt.labels, [t.kind == 2 for t in utt.tokens]
    vocab = {"<pad>": 0, "|": 1, "a": 2, "b": 3, "k": 4, "'": 5}
    chars = hke.tokens_to_vocab_chars(labels, pz, vocab)
    assert chars == ["b", "a", "k", "a"]
    # frames: each char for 2 frames; 20 blank frames (400 ms) between words. NOTE: gaps shorter than the
    # vowel cap (0.25 s) are absorbed into the preceding vowel - a documented limitation of CTC alignment
    seq = [0, 3, 3, 2, 2] + [0] * 20 + [4, 4, 2, 2, 0]                    # a 400 ms pause between the words
    logits = torch.full((1, len(seq), len(vocab)), -8.0)
    for t, s in enumerate(seq):
        logits[0, t, s] = 8.0

    class Proc:
        class tokenizer:
            pad_token_id = 0
            @staticmethod
            def get_vocab():
                return vocab
        def __call__(self, wav, sampling_rate, return_tensors):
            class O: input_values = torch.zeros(1, len(wav))
            return O()

    class Model:
        def __call__(self, x):
            class O: pass
            o = O(); o.logits = logits
            return o
        def parameters(self):
            return iter([])

    al = hke.MMSAligner("cpu")
    al.proc, al.model, al.vocab, al.blank, al.dtype = Proc(), Model(), vocab, 0, torch.float32
    wav = np.zeros(int(len(seq) * 320), dtype=np.float32)               # 16 kHz, 20 ms per frame
    iv, score = al.align(wav, labels, pz)
    d, s, e = durations_from_intervals(labels, pz, iv, 22050, 256)
    assert d[labels.index("sp")] > 0 and (d >= 0).all() and score > np.log(0.9)
    assert [l for _, _, l in iv if l] == [l for l, p in zip(labels, pz) if not p]


# ------------------------------------------------------------------ git clone + commit pinning
def test_setup_repo_pins_recorded_commit(tmp_path, monkeypatch):
    import subprocess
    src = tmp_path / "origin"
    src.mkdir()
    sh = lambda c, cwd=src: subprocess.run(c, shell=True, check=True, cwd=cwd, capture_output=True, text=True)
    sh("git init -q -b main && git config user.email a@b.c && git config user.name t")
    (src / "hausafs2").mkdir()
    (src / "hausafs2" / "v.txt").write_text("one")
    sh("git add -A && git commit -q -m one")
    c1 = sh("git rev-parse HEAD").stdout.strip()
    monkeypatch.setattr(hke, "GITHUB_URL", str(src))
    monkeypatch.setattr(hke, "GITHUB_REF", "main")
    monkeypatch.setattr(hke, "GITHUB_TOKEN", "")
    dest = tmp_path / "clone"
    assert hke.setup_repo(dest, None) == c1                               # first session records the commit
    (src / "hausafs2" / "v.txt").write_text("two")                        # author pushes new code later
    sh("git commit -q -am two")
    c2 = sh("git rev-parse HEAD").stdout.strip()
    assert c2 != c1
    assert hke.setup_repo(dest, c1) == c1                                 # later session stays on the recorded commit
    assert (dest / "hausafs2" / "v.txt").read_text() == "one"
    monkeypatch.setattr(hke, "PIN_TO_RECORDED_COMMIT", False)
    assert hke.setup_repo(dest, c1) == c2                                 # unpinned follows the branch
