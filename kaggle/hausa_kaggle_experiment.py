# =============================================================================
#  HAUSA-FASTSPEECH2  -  KAGGLE EXPERIMENT RUNNER   (single cell, fully resumable)
# =============================================================================
#  Paste this whole file into ONE Kaggle notebook cell (GPU on, Internet on) and run it.
#  Start with MODE = "smoke" (20 clips, ~15 min) to verify everything, then MODE = "full".
#
#  What it does (every stage is checkpointed to YOUR private Hugging Face repo, so a new
#  Kaggle session - or a crash, or a closed laptop - continues where it stopped):
#     1  clone your GitHub repo (the hausa-fastspeech2 code)         -> pinned commit recorded
#     2  select speakers/utterances from the dataset (text-disjoint test split)
#     3  forced-align with MMS CTC (rough, see README limitations)    -> resumable per clip
#     4  preprocess (mel / per-speaker F0 / energy / contours)
#     5  train + evaluate every arm x seed (baselines + ablations)    -> resumable per 1000 steps
#     6  report: tables (csv / LaTeX), paired-bootstrap CIs, plot, listening-test samples
#
#  RESUME: just run the same cell again. State lives in HF repo  <HF_REPO_ID>/exp/<EXP_NAME>/.
# =============================================================================
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# 0. USER CONFIG  -  edit this block only
# =============================================================================
HF_TOKEN = "PASTE_YOUR_NEW_HF_WRITE_TOKEN_HERE"      # or add a Kaggle Secret named HF_TOKEN and leave this
HF_REPO_ID = "suleiman2003/hausa-fs2-interspeech"  # private repo, created automatically
GITHUB_URL = "https://github.com/issuleiman/hausa-fastspeech2.git"
GITHUB_REF = "main"              # branch, tag or commit. For the paper: use a TAG (e.g. "v1.0")
GITHUB_TOKEN = ""                # only needed if the GitHub repo is private (a fine-grained read-only PAT)
PIN_TO_RECORDED_COMMIT = True    # later sessions check out the exact commit used by the first session

DATASET_REPO = "suleiman2003/unified-hausa-speech"
EXP_NAME = "exp1"                # change to start an independent experiment in the same HF repo
MODE = "smoke"                   # "smoke" (20 clips, tiny model)  |  "full"
FORCE_SPEAKERS: List[int] = []   # e.g. [3, 17, 25, 40]; empty = automatic (balanced gender, most data)
TONE_LEXICON_PATH = ""           # optional tone/length lexicon .tsv (path inside the repo or on disk)
SQUASH_HISTORY = False           # True: after each finished run, squash the HF repo history to free storage used by old checkpoint versions (IRREVERSIBLE)
SESSION_HOURS = 0                # 0 = automatic (8.5 h interactive, 11 h when run via "Save & Run All"); else set e.g. 8.0
OVERRIDES: Dict[str, Any] = {}   # e.g. {"steps": 20000, "seeds": [1234, 4321, 2024], "arms": ["fs2_original", "hausa_full"]}
# =============================================================================


def get_params(mode: str) -> Dict[str, Any]:
    if mode == "smoke":
        P = dict(n_speakers=2, utts_per_speaker=10, test_per_speaker=2, min_dur=1.5, max_dur=10.0,
                 seeds=[1234], steps=60, seg=30, batch_size=8, hidden=96, enc_layers=2, dec_layers=2,
                 conv_filter=256, vp_filter=96, postnet_channels=64, postnet_layers=3, n_bins=128,
                 warmup=20, peak_lr=2e-3, arms=["fs2_original", "hausa_full"], val_size=4,
                 align_drop_frac=0.0, n_samples=2, upload_models=True, ckpt_upload_min=1.0, num_workers=0)
    elif mode == "full":
        P = dict(n_speakers=10, utts_per_speaker=400, test_per_speaker=30, min_dur=1.5, max_dur=12.0,
                 seeds=[1234, 4321], steps=12000, seg=1000, batch_size=16, hidden=256, enc_layers=4,
                 dec_layers=6, conv_filter=1024, vp_filter=256, postnet_channels=512, postnet_layers=5,
                 n_bins=256, warmup=2000, peak_lr=1e-3, arms="ALL", val_size=100, align_drop_frac=0.05,
                 n_samples=10, upload_models=True, ckpt_upload_min=25.0, num_workers=2)
    else:
        raise ValueError("MODE must be 'smoke' or 'full'")
    P.update(OVERRIDES)
    P["mode"] = mode
    return P


# ---- model arms: baselines + cumulative ladder + leave-one-out + isolated contributions -----------
GROUPS = {"pos": ["boundary", "syl_pos", "word_pos", "phrase_pos"],      # prosodic position / boundary
          "syl": ["weight", "role", "length"],                            # syllable structure
          "len": ["length"], "tone": ["tone"]}


def _arm(groups=(), K=1, hier=False, pitch_mode="contour", tone_stuff=False) -> Dict[str, Any]:
    feats: List[str] = []
    for g in groups:
        feats += GROUPS[g]
    return dict(input_features=feats, hier=dict(enabled=hier), use_downstep=tone_stuff,
                pitch_tone_skip=tone_stuff, pitch=dict(mode=pitch_mode, n_points=K))


ARMS: Dict[str, Dict[str, Any]] = {
    # -- baselines (original FastSpeech 2 recipes) --
    "fs2_original":   _arm(),                                      # phone ids, phoneme-level pitch (ming024-style)
    "fs2_framepitch": _arm(pitch_mode="frame"),                    # phone ids, frame-level pitch (paper)
    # -- proposed model --
    "hausa_full":     _arm(["pos", "syl"], K=3, hier=True),
    # -- cumulative ladder (add one component at a time) --
    "ladder_pos":     _arm(["pos"]),
    "ladder_syl":     _arm(["pos", "syl"]),
    "ladder_contour": _arm(["pos", "syl"], K=3),                   # = full without hierarchy
    # -- leave-one-out from the full model --
    "loo_no_pos":     _arm(["syl"], K=3, hier=True),
    "loo_no_syl":     _arm(["pos"], K=3, hier=True),
    "loo_no_contour": _arm(["pos", "syl"], K=1, hier=True),
    # -- isolated contributions / sensitivity --
    "contour_only":   _arm([], K=3),
    "hier_only":      _arm([], hier=True),
    "sens_K5":        _arm(["pos", "syl"], K=5, hier=True),
}
TONE_ARMS: Dict[str, Dict[str, Any]] = {      # only added when the data really carries tone information
    "fs2_tonelength":      _arm(["tone", "len"]),
    "full_tone":           _arm(["pos", "syl", "tone"], K=3, hier=True, tone_stuff=True),
    "full_tone_noskip":    _arm(["pos", "syl", "tone"], K=3, hier=True, tone_stuff=False),
}
MAIN_ARMS = ["fs2_original", "hausa_full"]    # models kept on the Hub + used for listening samples
REFERENCE_ARMS = ["fs2_original", "hausa_full"]
METRICS = ["mel_l1", "mcd_proxy", "pitch_rmse", "pitch_corr", "dur_mae", "rate_err"]
HIGHER_BETTER = {"pitch_corr"}

# =============================================================================
# 1. small utilities
# =============================================================================
IS_KAGGLE = os.path.isdir("/kaggle/working")
WORK_ROOT = Path("/kaggle/working") if IS_KAGGLE else Path("./kaggle_work").resolve()
SAFETY_S = 600          # keep this many seconds free at the end of a session for final uploads


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


def resolve_token() -> str:
    tok = HF_TOKEN
    if not tok or tok.startswith("PASTE_"):
        tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = ""
    if not tok:
        raise RuntimeError("No Hugging Face token. Paste a NEW write token into HF_TOKEN at the top of this cell "
                           "(or add a Kaggle Secret named HF_TOKEN).")
    return tok


class StopSession(Exception):
    """Raised when the session time budget is nearly used up; state is saved, just re-run the cell."""


class Budget:
    def __init__(self, hours: float):
        self.t0, self.limit = time.time(), hours * 3600

    def left(self) -> float:
        return self.limit - (time.time() - self.t0)

    def check(self, need_s: float = 0.0):
        if self.left() < need_s + SAFETY_S:
            raise StopSession(f"{self.left() / 60:.0f} min left in this session")


def default_budget_hours() -> float:
    if SESSION_HOURS:
        return float(SESSION_HOURS)
    if not IS_KAGGLE:
        return 1e6
    # 'Save & Run All (Commit)' sessions may run up to 12 h, interactive GPU sessions 9 h.
    return 11.0 if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() in ("batch", "commit") else 8.5


# =============================================================================
# 2. persistent store (Hugging Face Hub)  +  a local fake used for testing
# =============================================================================
def is_rate_limit(exc: Exception) -> bool:
    m = str(exc).lower()
    return "429" in m or "rate limit" in m or "repository commits" in m


class HubBase:
    pending: Dict[str, str]

    def put_many(self, items: List[Tuple[Path, str]], message: str = "update") -> bool: ...
    def get(self, remote: str, local: Path, force: bool = False) -> bool: ...
    def list(self, prefix: str = "") -> List[str]: ...
    def delete(self, remote: str) -> None: ...
    def squash(self) -> None: ...

    def put(self, local: Path, remote: str) -> bool:
        return self.put_many([(local, remote)], remote)

    def exists(self, remote: str) -> bool:
        return remote in self.list(remote)

    def flush_pending(self, max_wait_s: float = 0.0) -> bool:
        """Retry uploads deferred by a rate limit / network error. True when nothing is left pending."""
        t0 = time.time()
        while self.pending:
            items = [(Path(l), r) for r, l in self.pending.items() if Path(l).exists()]
            self.pending = {}
            if not items or self.put_many(items, "deferred upload"):
                break
            if time.time() - t0 >= max_wait_s:
                return False
            time.sleep(60)
        return not self.pending


class HfHub(HubBase):
    def __init__(self, repo_id: str, token: str):
        from huggingface_hub import HfApi
        self.repo, self.token, self.api = repo_id, token, HfApi(token=token)
        self.pending = {}

    def ensure(self):
        who = self.api.whoami()
        log(f"Hugging Face: authenticated as {who.get('name')}")
        self.api.create_repo(self.repo, repo_type="model", private=True, exist_ok=True)
        log(f"Hugging Face repo ready (private): {self.repo}")

    def put_many(self, items, message="update") -> bool:
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(l)) for l, r in items]
        for attempt in range(5):
            try:
                self.api.create_commit(repo_id=self.repo, repo_type="model", operations=ops,
                                       commit_message=message[:200])
                return True
            except Exception as exc:
                if is_rate_limit(exc):
                    log(f"  [hub] rate-limited; deferring {len(items)} upload(s) (compute continues)")
                    for l, r in items:
                        self.pending[r] = str(l)
                    return False
                wait = min(90, 2 ** attempt + random.random())
                log(f"  [hub] upload failed ({type(exc).__name__}: {str(exc)[:120]}); retry in {wait:.0f}s")
                time.sleep(wait)
        for l, r in items:
            self.pending[r] = str(l)
        return False

    def get(self, remote, local, force=False) -> bool:
        from huggingface_hub import hf_hub_download
        last = None
        for attempt in range(3):
            tmp = tempfile.mkdtemp()
            try:
                p = hf_hub_download(repo_id=self.repo, filename=remote, repo_type="model", token=self.token,
                                    local_dir=tmp, force_download=force)
                Path(local).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(p, local)
                return True
            except Exception as exc:
                msg = str(exc)
                if "404" in msg or "Entry Not Found" in msg or "EntryNotFound" in type(exc).__name__:
                    return False
                last = exc
                time.sleep(2 ** attempt * 3)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Could not download {remote} from the Hub: {last}")   # never confuse 'error' with 'absent'

    def list(self, prefix="") -> List[str]:
        return [f for f in self.api.list_repo_files(self.repo, repo_type="model") if f.startswith(prefix)]

    def delete(self, remote):
        try:
            self.api.delete_file(path_in_repo=remote, repo_id=self.repo, repo_type="model",
                                 commit_message=f"delete {remote}")
        except Exception as exc:
            log(f"  [hub] delete {remote} skipped: {str(exc)[:100]}")

    def squash(self):
        try:   # drops old LFS versions of overwritten checkpoints -> keeps storage small (irreversible)
            self.api.super_squash_history(repo_id=self.repo, repo_type="model")
        except Exception as exc:
            log(f"  [hub] squash skipped: {str(exc)[:100]}")


class LocalHub(HubBase):
    """Directory-backed stand-in for the Hub (used by the offline tests)."""

    def __init__(self, root: Path):
        self.root, self.pending = Path(root), {}
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure(self):
        pass

    def put_many(self, items, message="update") -> bool:
        for l, r in items:
            dst = self.root / r
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            shutil.copy2(l, tmp)
            os.replace(tmp, dst)
        return True

    def get(self, remote, local, force=False) -> bool:
        src = self.root / remote
        if not src.exists():
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local)
        return True

    def list(self, prefix="") -> List[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                      if p.is_file() and str(p.relative_to(self.root)).startswith(prefix))

    def delete(self, remote):
        (self.root / remote).unlink(missing_ok=True)

    def squash(self):
        pass


# ---- experiment state (JSON on the hub) ---------------------------------------------------------
def merge_states(a: dict, b: dict) -> dict:
    """Union of two state dicts: a finished stage/run always wins; otherwise the more advanced one wins."""
    out = copy.deepcopy(a)
    out.setdefault("stages", {})
    out.setdefault("runs", {})
    for st, v in b.get("stages", {}).items():
        cur = out["stages"].get(st)
        if cur is None or (v.get("done") and not cur.get("done")):
            out["stages"][st] = copy.deepcopy(v)
    for rid, v in b.get("runs", {}).items():
        cur = out["runs"].get(rid)
        if cur is None:
            out["runs"][rid] = copy.deepcopy(v)
        elif cur.get("status") != "done" and (v.get("status") == "done" or v.get("step", 0) > cur.get("step", 0)):
            out["runs"][rid] = copy.deepcopy(v)
    for k, v in b.items():
        if k not in ("stages", "runs") and k not in out:
            out[k] = copy.deepcopy(v)
    out["updated"] = max(a.get("updated", 0), b.get("updated", 0))
    return out


class State:
    def __init__(self, hub: HubBase, local: Path, remote: str):
        self.hub, self.local, self.remote = hub, local, remote
        self.d: Dict[str, Any] = {"stages": {}, "runs": {}, "updated": 0}

    def load(self):
        loaded = []
        if self.local.exists():
            loaded.append(json.loads(self.local.read_text(encoding="utf-8")))
        tmp = self.local.with_suffix(".remote")
        if self.hub.get(self.remote, tmp, force=True):
            loaded.append(json.loads(tmp.read_text(encoding="utf-8")))
            tmp.unlink(missing_ok=True)
        for s in loaded:
            self.d = merge_states(self.d, s)
        log(f"state loaded: stages done={[k for k, v in self.d['stages'].items() if v.get('done')]}, "
            f"runs done={sum(v.get('status') == 'done' for v in self.d['runs'].values())}/{len(self.d['runs'])}")

    def save(self, push: bool = True, extra: Optional[List[Tuple[Path, str]]] = None):
        self.d["updated"] = time.time()
        atomic_json(self.local, self.d)
        if push:
            self.hub.put_many([(self.local, self.remote)] + (extra or []), "state")

    def stage_done(self, name: str) -> bool:
        return bool(self.d["stages"].get(name, {}).get("done"))

    def mark(self, name: str, **kw):
        self.d["stages"][name] = {"done": True, "time": time.time(), **kw}


# =============================================================================
# 3. forced alignment helpers (pure functions are unit-tested offline)
# =============================================================================
def ctc_viterbi(logp, targets: List[int], blank: int):
    """CTC forced alignment (Viterbi). logp: (T,V) log-probabilities. Returns (path, ext) where path[t] is
    the index into the blank-interleaved target sequence `ext`, or None if the alignment is infeasible."""
    import numpy as np
    T, L = logp.shape[0], len(targets)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    need = L + sum(1 for i in range(1, L) if targets[i] == targets[i - 1])
    if T < need:
        return None, ext
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, blank]
    dp[0, 1] = logp[0, ext[1]]
    allow2 = np.zeros(S, dtype=bool)
    allow2[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    for t in range(1, T):
        prev = dp[t - 1]
        p1 = np.concatenate([[NEG], prev[:-1]])
        p2 = np.where(allow2, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        stack = np.stack([prev, p1, p2])
        idx = stack.argmax(0)
        dp[t] = stack.max(0) + logp[t, ext]
        bp[t] = idx
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path, ext


def path_to_char_spans(path, ext, logp, sec_per_frame: float):
    """-> list of (start_s, end_s) per target character, and mean log-prob of emitting frames (confidence)."""
    import numpy as np
    L = (len(ext) - 1) // 2
    spans = []
    for k in range(L):
        fr = np.where(path == 2 * k + 1)[0]
        spans.append((fr[0] * sec_per_frame, (fr[-1] + 1) * sec_per_frame))
    emit = path % 2 == 1
    score = float(np.mean(logp[np.arange(len(path))[emit], ext[path[emit]]])) if emit.any() else -99.0
    return spans, score


_FALLBACK = {"ɓ": ["b"], "ɗ": ["d"], "ƙ": ["k"], "ƴ": ["y"], "ʔ": ["'", "ʼ", "ʾ", "’"]}


def tokens_to_vocab_chars(labels: List[str], is_pause: List[bool], vocab: Dict[str, int]) -> List[str]:
    """One character string per NON-pause token, using only characters of the aligner's vocabulary.
    Hausa letters (ɓ ɗ ƙ ƴ) are used when the model knows them, else romanised. Long vowels = 1 char,
    geminate consonants = 2 chars, diphthongs/digraphs = their letters."""
    def pick(c: str) -> Optional[str]:
        if c in vocab:
            return c
        for alt in _FALLBACK.get(c, []):
            if alt in vocab:
                return alt
        return None

    out = []
    for lab, p in zip(labels, is_pause):
        if p:
            continue
        long = lab.endswith("ː")
        base = lab.replace("ː", "")
        src = list(base)
        if long and base[0] not in "aeiou":
            src = src + src                       # geminate consonant
        chars = [pick(c) for c in src]
        s = "".join(c for c in chars if c)
        if not s:
            fb = pick("ʔ")
            if not fb:
                raise ValueError(f"no aligner character available for token {lab!r}")
            s = fb
        out.append(s)
    return out


class MMSAligner:
    """Character-level CTC forced alignment with Meta MMS-1B-all (Hausa adapter). Rough (~20 ms, peaky)."""

    def __init__(self, device: str, model_id: str = "facebook/mms-1b-all", lang: str = "hau"):
        self.device, self.model_id, self.lang, self.model = device, model_id, lang, None

    def _load(self):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        log(f"loading aligner {self.model_id} [{self.lang}] (first use downloads ~4 GB) ...")
        self.proc = AutoProcessor.from_pretrained(self.model_id, target_lang=self.lang)
        m = Wav2Vec2ForCTC.from_pretrained(self.model_id, target_lang=self.lang, ignore_mismatched_sizes=True)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = m.to(self.device, dtype=self.dtype).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id
        log(f"aligner ready; {len(self.vocab)} symbols; Hausa letters known: "
            f"{[c for c in 'ɓɗƙƴ' if c in self.vocab]}")

    def align(self, wav16k, labels: List[str], is_pause: List[bool]):
        import torch
        from hausafs2.data.ctc_align import spans_to_intervals
        if self.model is None:
            self._load()
        chars = tokens_to_vocab_chars(labels, is_pause, self.vocab)
        ids = [self.vocab[c] for s in chars for c in s]
        x = self.proc(wav16k, sampling_rate=16000, return_tensors="pt").input_values.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            logits = self.model(x).logits[0].float()
        logp = torch.log_softmax(logits, -1).cpu().numpy()
        path, ext = ctc_viterbi(logp, ids, self.blank)
        if path is None:
            raise ValueError("audio too short for the transcript (CTC infeasible)")
        dur = len(wav16k) / 16000
        spans, score = path_to_char_spans(path, ext, logp, dur / logp.shape[0])
        nonpause = [l for l, p in zip(labels, is_pause) if not p]
        return spans_to_intervals(chars, spans, nonpause, total_end=dur), score


# =============================================================================
# 4. dataset access: HF parquet source (+ local source for tests) and subset selection
# =============================================================================
class HfParquetSource:
    """Reads the dataset's parquet shards. Metadata via column-projected range reads (no audio downloaded)."""
    COLS = ["speaker_id", "text", "duration", "gender"]

    def __init__(self, repo: str, token: str, prefix: str = "train"):
        from huggingface_hub import HfApi
        self.repo, self.token, self.prefix, self.api = repo, token, prefix, HfApi(token=token)

    def files(self) -> List[str]:
        fs = [f for f in self.api.list_repo_files(self.repo, repo_type="dataset") if f.endswith(".parquet")]
        sel = sorted(f for f in fs if Path(f).name.startswith(self.prefix) or f"/{self.prefix}" in f)
        if not sel:
            raise RuntimeError(f"no '{self.prefix}' parquet files found in {self.repo}: {fs[:5]}")
        return sel

    def read_meta(self, f: str):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        try:
            with HfFileSystem(token=self.token).open(f"datasets/{self.repo}/{f}", "rb") as fh:
                return pq.read_table(fh, columns=self.COLS).to_pandas()
        except Exception as exc:
            log(f"  range-read failed for {f} ({type(exc).__name__}); downloading whole shard")
            p = self.fetch(f)
            try:
                return pq.read_table(p, columns=self.COLS).to_pandas()
            finally:
                self.release(p)

    def fetch(self, f: str) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=self.repo, filename=f, repo_type="dataset", token=self.token))

    def release(self, p: Path):
        for q in {str(p), os.path.realpath(p)}:
            try:
                os.remove(q)
            except OSError:
                pass


class LocalParquetSource:
    COLS = HfParquetSource.COLS

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def files(self):
        return sorted(p.name for p in self.dir.glob("*.parquet"))

    def read_meta(self, f):
        import pyarrow.parquet as pq
        return pq.read_table(self.dir / f, columns=self.COLS).to_pandas()

    def fetch(self, f):
        return self.dir / f

    def release(self, p):
        pass


def norm_for_dedupe(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def text_ok(fe, text: str) -> bool:
    if any(ch.isdigit() for ch in text):
        return False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            u = fe(text)
        except Exception:
            return False
    return not w and len(u) >= 3


def choose_subset(meta, P: Dict[str, Any], fe, force_speakers: List[int], seed: int = 20260101):
    """Pick speakers and utterances. Test texts are removed from every train row (text-disjoint split),
    which matters because the same prompt is often read by many speakers."""
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(seed)
    df = meta[(meta.duration >= P["min_dur"]) & (meta.duration <= P["max_dur"])].copy()
    df["norm"] = df.text.map(norm_for_dedupe)
    df = df[df.norm.str.len() >= 3]
    need = P["utts_per_speaker"] + P["test_per_speaker"]
    counts = df.groupby("speaker_id").size()
    spk_table = (meta.groupby("speaker_id").agg(clips=("text", "size"), hours=("duration", lambda d: d.sum() / 3600),
                                                 gender=("gender", "first")).reset_index())
    cand = [int(s) for s in force_speakers] if force_speakers else \
        [int(s) for s in counts[counts >= need].sort_values(ascending=False).index]
    if not force_speakers:
        gender_of = df.groupby("speaker_id").gender.first()
        males = [s for s in cand if gender_of[s] == "male"]
        females = [s for s in cand if gender_of[s] == "female"]
        others = [s for s in cand if s not in males and s not in females]
        chosen: List[int] = []
        for i in range(max(len(males), len(females), len(others))):
            for lst in (females, males, others):
                if i < len(lst) and len(chosen) < P["n_speakers"]:
                    chosen.append(lst[i])
        cand = chosen
    if not cand:
        raise RuntimeError("no speaker has enough usable utterances; lower utts_per_speaker/test_per_speaker")
    pools = {}
    for s in cand:
        sub = df[df.speaker_id == s].drop_duplicates("norm")
        sub = sub[sub.text.map(lambda t: text_ok(fe, t))]
        pools[s] = sub.iloc[rng.permutation(len(sub))]
    rows, test_norms = [], set()
    for s, pool in pools.items():                                   # pass 1: test rows
        take = pool.head(P["test_per_speaker"])
        test_norms |= set(take.norm)
        rows += [dict(r._asdict(), role="test", speaker=s) for r in take.itertuples()]
    for s, pool in pools.items():                                   # pass 2: train rows, texts disjoint from test
        take = pool[~pool.norm.isin(test_norms)].head(P["utts_per_speaker"])
        rows += [dict(r._asdict(), role="train", speaker=s) for r in take.itertuples()]
    out = pd.DataFrame(rows)
    ctr: Dict[Tuple[int, str], int] = defaultdict(int)
    ids = []
    for r in out.itertuples():
        k = ctr[(r.speaker, r.role)]
        ctr[(r.speaker, r.role)] += 1
        ids.append(f"s{r.speaker}_te_{k:04d}" if r.role == "test" else f"s{r.speaker}_{k:04d}")
    out["id"] = ids
    return out, spk_table


# =============================================================================
# 5. evaluation + statistics
# =============================================================================
def _mcd_proxy(a, b) -> float:
    """MCD-like distance (dB) between two natural-log mel spectrograms via DCT cepstra (a PROXY, not WORLD-MCD)."""
    import numpy as np
    from scipy.fft import dct
    ca = dct(a, type=2, norm="ortho", axis=1)[:, 1:13]
    cb = dct(b, type=2, norm="ortho", axis=1)[:, 1:13]
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0 * ((ca - cb) ** 2).sum(1)).mean())


def _corr(x, y) -> float:
    import numpy as np
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_model(model, cfg: dict, pre_dir: str, split_file: str, device: str, bs: int = 16) -> Dict[str, Any]:
    """Ground-truth durations, PREDICTED pitch/energy (so the pitch/energy predictors are what is measured).
    Returns per-utterance metric lists plus their means."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from hausafs2.data.dataset import HausaTTSDataset, collate
    from hausafs2.frontend import symbols as S
    from hausafs2.model import batch_to_device
    from hausafs2.model.variance_adaptor import contour_to_frames
    pm, em = cfg["model"]["pitch"]["mode"], cfg["model"]["energy"]["mode"]
    ds = HausaTTSDataset(pre_dir, split_file, pm, em, cfg["model"]["pitch"].get("n_points"))
    per = {m: [] for m in METRICS}
    model.eval()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate):
            b = batch_to_device(b, device)
            out = model({k: v for k, v in b.items() if k not in ("pitch", "energy")})
            z = out["pitch"]
            z = contour_to_frames(z, out["tok_idx"], out["rel"]) if z.dim() == 3 else z
            kind_f = b["kind"].gather(1, out["tok_idx"])
            pred_dur = torch.clamp(torch.round(torch.exp(out["log_dur"]) - 1), min=0)
            for i in range(b["mel"].size(0)):
                T = int(b["mel_len"][i])
                mel_t, mel_p = b["mel"][i, :T].cpu().numpy(), out["mel_post"][i, :T].cpu().numpy()
                per["mel_l1"].append(float(np.abs(mel_t - mel_p).mean()))
                per["mcd_proxy"].append(_mcd_proxy(mel_t, mel_p))
                fm = (kind_f[i, :T] != S.KIND_P).cpu().numpy()
                zt, zp = b["pitch_frame"][i, :T].cpu().numpy()[fm], z[i, :T].cpu().numpy()[fm]
                per["pitch_rmse"].append(float(np.sqrt(((zt - zp) ** 2).mean())))
                per["pitch_corr"].append(_corr(zt, zp))
                n = int(b["src_len"][i])
                nonp = (b["kind"][i, :n] != S.KIND_P).cpu().numpy()
                gd, pd_ = b["duration"][i, :n].cpu().numpy().astype(float), pred_dur[i, :n].cpu().numpy()
                per["dur_mae"].append(float(np.abs(gd - pd_)[nonp].mean()))
                per["rate_err"].append(float(abs(math.log((pd_.sum() + 1) / (gd.sum() + 1)))))
    return dict(per_utt=per, mean={m: float(np.nanmean(v)) for m, v in per.items()}, n_utts=len(per["mel_l1"]))


def paired_bootstrap(diff, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    import numpy as np
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), p=float(min(1.0, max(p, 1.0 / n_boot))), n=len(d))


def make_report(results: List[dict], out_dir: Path) -> Dict[str, Any]:
    """Tables (csv/LaTeX/markdown), paired-bootstrap vs reference arms, bar plot."""
    import numpy as np
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    by_arm: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)
    rows = []
    for arm, rs in by_arm.items():
        row = {"arm": arm, "seeds": len(rs), "params_M": round(rs[0]["params"] / 1e6, 2)}
        for m in METRICS:
            v = np.array([r["metrics"][m] for r in rs])
            row[m] = float(v.mean())
            row[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    order = list(ARMS) + list(TONE_ARMS)
    df = pd.DataFrame(rows).sort_values("arm", key=lambda s: s.map(lambda a: order.index(a) if a in order else 99))
    df.to_csv(out_dir / "results_by_arm.csv", index=False)

    def per_utt(arm, metric):
        return np.mean([r["per_utt"][metric] for r in by_arm[arm]], axis=0)

    stats_rows = []
    for ref in REFERENCE_ARMS:
        if ref not in by_arm:
            continue
        for arm in by_arm:
            if arm == ref:
                continue
            for m in METRICS:
                a, b = per_utt(arm, m), per_utt(ref, m)
                if len(a) != len(b):
                    continue
                bs = paired_bootstrap(a - b)
                better = (bs["hi"] < 0) if m not in HIGHER_BETTER else (bs["lo"] > 0)
                worse = (bs["lo"] > 0) if m not in HIGHER_BETTER else (bs["hi"] < 0)
                stats_rows.append(dict(arm=arm, vs=ref, metric=m, diff=bs["mean"], ci_lo=bs["lo"], ci_hi=bs["hi"],
                                       p=bs["p"], verdict="better" if better else ("worse" if worse else "n.s.")))
    sdf = pd.DataFrame(stats_rows)
    sdf.to_csv(out_dir / "paired_bootstrap_vs_reference.csv", index=False)

    cols = ["arm", "seeds", "params_M"] + METRICS
    with open(out_dir / "results_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            cells = [str(r["arm"]).replace("_", "\\_"), str(int(r["seeds"])), f"{r['params_M']:.2f}"]
            cells += [f"{r[m]:.3f}$\\pm${r[m + '_sd']:.3f}" for m in METRICS]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("# Results (mean over seeds, +- sd). Lower is better except pitch_corr.\n\n")
        try:
            f.write(df[cols].round(3).to_markdown(index=False))
        except Exception:
            f.write(df[cols].round(3).to_string(index=False))
        f.write("\n\n# Paired bootstrap (utterance level, 95% CI of arm - reference)\n\n")
        f.write(sdf.round(4).to_string(index=False) if len(sdf) else "(reference arms missing)")
        f.write("\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, m in zip(axs, ["mel_l1", "pitch_rmse", "dur_mae"]):
            ax.bar(range(len(df)), df[m], yerr=df[m + "_sd"], capsize=2)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["arm"], rotation=70, ha="right", fontsize=8)
            ax.set_title(m)
        fig.tight_layout()
        fig.savefig(out_dir / "results_bars.png", dpi=130)
        plt.close(fig)
    except Exception as exc:
        log(f"plot skipped: {exc}")
    return dict(table=df, stats=sdf)


# =============================================================================
# 6. the experiment
# =============================================================================
def make_tar(src: Path, tar_path: Path, exclude: Tuple[str, ...] = ()):
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(src, arcname=src.name, filter=lambda ti: None if any(e in ti.name for e in exclude) else ti)


def extract_tar(tar_path: Path, dest_parent: Path):
    dest_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_parent)


class Experiment:
    def __init__(self, P: Dict[str, Any], hub: HubBase, source, aligner, work: Path, repo_dir: Path,
                 device: Optional[str] = None, budget_hours: Optional[float] = None,
                 force_speakers: Optional[List[int]] = None, exp_name: str = EXP_NAME):
        self.P, self.hub, self.source, self.aligner = P, hub, source, aligner
        self.exp = exp_name
        self.work, self.repo_dir = Path(work) / exp_name, Path(repo_dir)
        self.hubdir = f"exp/{exp_name}"
        self.force_speakers = list(force_speakers if force_speakers is not None else FORCE_SPEAKERS)
        self.budget = Budget(budget_hours if budget_hours is not None else default_budget_hours())
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dirs = {k: self.work / k for k in ("subset", "align", "pre", "runs", "report", "samples")}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.state = State(hub, self.work / "state.json", f"{self.hubdir}/state.json")
        self.events: List[tuple] = []          # (kind, ...) - used by the tests
        self.crash_hook: Optional[Callable[[str, int], None]] = None
        self.seg_seconds = 0.0

    # ------------------------------------------------------------------ helpers
    def fetch_tar(self, remote: str, dest: Path) -> bool:
        tmp = self.work / "_dl" / Path(remote).name
        if not self.hub.get(f"{self.hubdir}/{remote}", tmp):
            return False
        if dest.exists():
            shutil.rmtree(dest)
        extract_tar(tmp, dest.parent)
        tmp.unlink(missing_ok=True)
        return True

    def push_tar(self, src: Path, remote: str, exclude=(), extra=()):
        tmp = self.work / "_up" / Path(remote).name
        make_tar(src, tmp, exclude)
        ok = self.hub.put_many([(tmp, f"{self.hubdir}/{remote}")] + list(extra), remote)
        return ok

    def _read_meta_csv(self) -> List[Tuple[str, str, str]]:
        rows = []
        for line in (self.dirs["subset"] / "metadata.csv").read_text(encoding="utf-8").splitlines():
            if line.strip():
                u, s, t = line.split("|", 2)
                rows.append((u, s, t))
        return rows

    # ------------------------------------------------------------------ stage 1: subset
    def stage_subset(self):
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile as sf
        sub = self.dirs["subset"]
        if self.state.stage_done("subset"):
            if not (sub / "metadata.csv").exists():
                log("subset: restoring from Hub")
                if not self.fetch_tar("subset.tar", sub):
                    raise RuntimeError("state says subset is done but subset.tar is missing on the Hub")
            return
        from hausafs2.frontend import HausaFrontend
        log("stage 1/5: selecting speakers/utterances (metadata scan, no audio downloaded) ...")
        import pandas as pd
        metas = []
        for f in self.source.files():
            m = self.source.read_meta(f)
            m["file"], m["row"] = f, np.arange(len(m))
            metas.append(m)
        meta = pd.concat(metas, ignore_index=True)
        log(f"  metadata: {len(meta)} clips, {meta.speaker_id.nunique()} speakers")
        sel, spk_table = choose_subset(meta, self.P, HausaFrontend("none"), self.force_speakers)
        spk_table.sort_values("clips", ascending=False).to_csv(self.work / "speaker_table.csv", index=False)
        sel.drop(columns=["norm"]).to_csv(self.work / "selection.csv", index=False)
        log(f"  selected {int((sel.role == 'train').sum())} train + {int((sel.role == 'test').sum())} test clips from "
            f"{sel.speaker.nunique()} speakers: {sorted(sel.speaker.unique().tolist())}")
        for f, grp in sel.groupby("file"):
            local = self.source.fetch(f)
            pf = pq.ParquetFile(str(local))
            want = {int(r.row): r for r in grp.itertuples()}
            offset = 0
            for g in range(pf.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                here = [r for r in want if offset <= r < offset + n]
                if here:
                    col = pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
                    for r in here:
                        rec = want[r]
                        a = col[r - offset]
                        wav, sr = sf.read(io.BytesIO(bytes(a["bytes"])), dtype="float32", always_2d=False)
                        if wav.ndim > 1:
                            wav = wav.mean(1)
                        d = sub / "wavs" / f"s{rec.speaker}"
                        d.mkdir(parents=True, exist_ok=True)
                        sf.write(str(d / f"{rec.id}.wav"), wav, sr, subtype="PCM_16")
                offset += n
            self.source.release(local)
        with open(sub / "metadata.csv", "w", encoding="utf-8") as fh:
            for r in sel.itertuples():
                fh.write(f"{r.id}|s{r.speaker}|{' '.join(str(r.text).split()).replace('|', ' ')}\n")
        self.push_tar(sub, "subset.tar", extra=[(self.work / "speaker_table.csv", f"{self.hubdir}/speaker_table.csv"),
                                                (self.work / "selection.csv", f"{self.hubdir}/selection.csv")])
        self.state.mark("subset", n=len(sel))
        self.state.save()

    # ------------------------------------------------------------------ stage 2: alignment
    def stage_align(self):
        import soundfile as sf
        import librosa
        from hausafs2.data.alignment import write_textgrid
        from hausafs2.frontend import HausaFrontend
        ad = self.dirs["align"]
        prog_path = ad / "progress.json"
        if self.state.stage_done("align"):
            if not (ad / "progress.json").exists():
                log("align: restoring from Hub")
                self.fetch_tar("align.tar", ad)
            return
        if not prog_path.exists():
            self.fetch_tar("align.tar", ad)
        ad.mkdir(parents=True, exist_ok=True)
        progress: Dict[str, dict] = json.loads(prog_path.read_text()) if prog_path.exists() else {}
        rows = self._read_meta_csv()
        fe = HausaFrontend("auto", TONE_LEXICON_PATH or None)
        todo = [r for r in rows if r[0] not in progress]
        log(f"stage 2/5: aligning {len(todo)} clips ({len(rows) - len(todo)} already done)")
        last_sync, dirty = time.time(), False

        def sync(final=False):
            atomic_json(prog_path, progress)
            ok = self.push_tar(ad, "align.tar")
            log(f"  align checkpoint {'pushed' if ok else 'deferred'}: {len(progress)}/{len(rows)}")

        try:
            for i, (uid, spk, text) in enumerate(todo):
                self.budget.check()
                try:
                    utt = fe(text)
                    wav, sr = sf.read(str(self.dirs["subset"] / "wavs" / spk / f"{uid}.wav"), dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(1)
                    w16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
                    iv, score = self.aligner.align(w16, utt.labels, [t.kind == 2 for t in utt.tokens])
                    (ad / spk).mkdir(parents=True, exist_ok=True)
                    write_textgrid(str(ad / spk / f"{uid}.TextGrid"), iv)
                    progress[uid] = {"ok": True, "score": float(score)}
                except Exception as exc:
                    progress[uid] = {"ok": False, "err": f"{type(exc).__name__}: {str(exc)[:150]}"}
                dirty = True
                self.events.append(("aligned", uid))
                if (i + 1) % 25 == 0 or time.time() - last_sync > 600:
                    sync()
                    last_sync, dirty = time.time(), False
        finally:
            if dirty:
                sync()
        ok = {k: v for k, v in progress.items() if v["ok"]}
        bad = [k for k, v in progress.items() if not v["ok"]]
        log(f"  alignment finished: {len(ok)} ok, {len(bad)} failed")
        if bad:
            log("  e.g. failures:", [progress[b]["err"] for b in bad[:3]])
        drop = int(len(ok) * float(self.P["align_drop_frac"]))
        if drop > 0:                                            # discard the least confident alignments
            worst = sorted(ok, key=lambda k: ok[k]["score"])[:drop]
            for uid in worst:
                tg = next(ad.glob(f"*/{uid}.TextGrid"), None)
                if tg:
                    tg.rename(tg.with_suffix(".rejected"))
            log(f"  dropped {drop} least-confident alignments")
        atomic_json(prog_path, progress)
        self.state.mark("align", ok=len(ok), failed=len(bad), dropped=drop)
        self.push_tar(ad, "align.tar")
        self.state.save()

    # ------------------------------------------------------------------ stage 3: preprocess
    def stage_preprocess(self):
        pre = self.dirs["pre"]
        if self.state.stage_done("preprocess"):
            if not (pre / "stats.json").exists():
                log("preprocess: restoring from Hub")
                if not self.fetch_tar("pre.tar", pre):
                    raise RuntimeError("state says preprocess is done but pre.tar is missing on the Hub")
            return
        from hausafs2.data.preprocess import Preprocessor
        log("stage 3/5: preprocessing (mel, per-speaker F0/energy, contours) ...")
        cfg = {"dataset": dict(corpus_dir=str(self.dirs["subset"]), alignment_dir=str(self.dirs["align"]),
                               out_dir=str(pre), val_size=self.P["val_size"], test_id_substring="_te_"),
               "audio": dict(sampling_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80,
                             fmin=0, fmax=8000),
               "frontend": dict(tone_mode="auto", lexicon_path=TONE_LEXICON_PATH or None),
               "model": dict(pitch=dict(mode="contour", n_points=3, f0_min=60, f0_max=700, min_voiced_frames=5),
                             energy=dict(mode="phoneme", log=True))}
        if pre.exists():
            shutil.rmtree(pre)
        stats = Preprocessor(cfg).run()
        skipped = json.loads((pre / "skipped.json").read_text())
        has_tone = self._tone_fraction(pre) > 0.3
        log(f"  tone information in data: {'YES' if has_tone else 'no (all tones unknown) -> tone arms disabled'}")
        self.push_tar(pre, "pre.tar", exclude=("/raw",))
        self.state.mark("preprocess", n_skipped=len(skipped), has_tone=has_tone, n_speakers=stats["n_speakers"])
        self.state.save()

    @staticmethod
    def _tone_fraction(pre: Path) -> float:
        import numpy as np
        from hausafs2.frontend import symbols as S
        known = tot = 0
        for f in list((pre / "feat").glob("*.npz"))[:300]:
            z = np.load(f)
            v = z["kind"] == S.KIND_V
            tot += int(v.sum())
            known += int((v & (z["tone"] >= S.TONE_H) & (z["tone"] <= S.TONE_F)).sum())
        return known / max(tot, 1)

    # ------------------------------------------------------------------ stage 4: runs
    def arm_table(self) -> Dict[str, Dict[str, Any]]:
        arms = dict(ARMS)
        if self.state.d["stages"].get("preprocess", {}).get("has_tone"):
            arms.update(TONE_ARMS)
        want = self.P["arms"]
        if want != "ALL":
            arms = {k: v for k, v in arms.items() if k in want}
        return arms

    def build_cfg(self, arm: str, patch: dict, seed: int) -> dict:
        P = self.P
        base = dict(
            dataset=dict(out_dir=str(self.dirs["pre"])),
            model=dict(hidden=P["hidden"], max_len=2000,
                       encoder=dict(layers=P["enc_layers"], heads=2, conv_filter=P["conv_filter"], conv_kernel=[9, 1], dropout=0.2),
                       decoder=dict(layers=P["dec_layers"], heads=2, conv_filter=P["conv_filter"], conv_kernel=[9, 1], dropout=0.2),
                       hier=dict(enabled=True, layers=1, heads=2, conv_filter=2 * P["hidden"], conv_kernel=[3, 1],
                                 dropout=0.1, levels=["syllable", "word", "phrase"]),
                       variance_predictor=dict(filter=P["vp_filter"], kernel=3, dropout=0.5),
                       pitch=dict(mode="contour", n_points=1, n_bins=P["n_bins"]),
                       energy=dict(mode="phoneme", n_bins=P["n_bins"], log=True),
                       tone=dict(dropout_word=0.0, dropout_utt=0.0),
                       postnet=dict(enabled=True, channels=P["postnet_channels"], layers=P["postnet_layers"], kernel=5),
                       multi_speaker=True, speaker_to_decoder=True),
            loss=dict(weights=dict(mel=1.0, postnet=1.0, duration=1.0, pitch=1.0, energy=1.0, tone=0.0, length=0.0),
                      tone_bearing_weight=1.0),
            train=dict(out_dir="", batch_size=P["batch_size"], total_steps=P["steps"], warmup_steps=P["warmup"],
                       peak_lr=P["peak_lr"], betas=[0.9, 0.98], eps=1e-9, grad_clip=1.0, grad_accum=1,
                       log_every=max(10, P["seg"] // 5), val_every=P["seg"], save_every=10 ** 9, keep_last=1,
                       num_workers=P["num_workers"], seed=seed))
        deep_merge(base["model"], patch)
        base["arm"] = arm
        return base

    def run_id(self, arm: str, seed: int, cfg: dict) -> str:
        sig = dict(model=cfg["model"], loss=cfg["loss"], steps=self.P["steps"], bs=self.P["batch_size"],
                   lr=self.P["peak_lr"], warm=self.P["warmup"])
        return f"{arm}__s{seed}__{short_hash(sig)}"

    def run_one(self, arm: str, patch: dict, seed: int):
        import torch
        from hausafs2.model import count_parameters
        from hausafs2.train import build_model, train
        cfg = self.build_cfg(arm, patch, seed)
        rid = self.run_id(arm, seed, cfg)
        st = self.state.d["runs"].get(rid, {})
        if st.get("status") == "done":
            return
        out = self.dirs["runs"] / rid
        out.mkdir(parents=True, exist_ok=True)
        cfg["train"]["out_dir"] = str(out)
        last, rdir = out / "last.pt", f"{self.hubdir}/runs/{rid}"
        if not last.exists() and self.hub.get(f"{rdir}/last.pt", last):
            log(f"  [{rid}] restored checkpoint from the Hub")
        step = int(torch.load(last, map_location="cpu")["step"]) if last.exists() else 0
        if step:
            self.events.append(("resume", rid, step))
            log(f"  [{rid}] resuming at step {step}/{self.P['steps']}")
        t_start, last_upload, seg = time.time(), time.time(), self.P["seg"]

        def upload_ckpt(final=False):
            self.state.d["runs"][rid] = dict(status="running", step=step, arm=arm, seed=seed)
            self.state.save(push=False)
            return self.hub.put_many([(last, f"{rdir}/last.pt"), (self.state.local, self.state.remote)],
                                     f"{rid} step {step}")

        while step < self.P["steps"]:
            self.budget.check(need_s=1.3 * self.seg_seconds)
            target = min(self.P["steps"], step + seg)
            c = copy.deepcopy(cfg)
            c["train"]["seed"] = seed + 7919 * (step // seg)          # different batch order every segment
            t0 = time.time()
            hist = train(c, resume=str(last) if last.exists() else None, max_steps=target, device=self.device,
                         log=lambda m: log(f"    {m[:170]}"))
            step = target
            self.seg_seconds = max(self.seg_seconds, time.time() - t0)
            with open(out / "train_log.jsonl", "a", encoding="utf-8") as fh:
                for h in hist:
                    fh.write(json.dumps(h, default=str) + "\n")
            due = (time.time() - last_upload) / 60 >= self.P["ckpt_upload_min"] or step >= self.P["steps"]
            if due:
                upload_ckpt()
                last_upload = time.time()
            if self.crash_hook:
                self.crash_hook(rid, step)

        ck = torch.load(last, map_location="cpu")
        model = build_model(cfg, ck["stats"])
        model.load_state_dict(ck["model"])
        model.to(self.device)
        ev = evaluate_model(model, cfg, str(self.dirs["pre"]), "test.txt", self.device)
        val = evaluate_model(model, cfg, str(self.dirs["pre"]), "val.txt", self.device)
        res = dict(run_id=rid, arm=arm, seed=seed, steps=self.P["steps"], params=count_parameters(model),
                   metrics=ev["mean"], per_utt=ev["per_utt"], val_metrics=val["mean"], n_test=ev["n_utts"],
                   train_minutes=round((time.time() - t_start) / 60, 1), cfg=cfg["model"])
        rpath = out / "result.json"
        atomic_json(rpath, res)
        items = [(rpath, f"{rdir}/result.json")]
        if arm in MAIN_ARMS and self.P["upload_models"]:
            mpath = out / "model.pt"
            torch.save(dict(model=ck["model"], cfg=ck["cfg"], stats=ck["stats"], step=step), mpath)
            items.append((mpath, f"{rdir}/model.pt"))
        self.state.d["runs"][rid] = dict(status="done", step=step, arm=arm, seed=seed,
                                         metrics=ev["mean"], params=res["params"])
        self.state.save(push=False)
        items.append((self.state.local, self.state.remote))
        self.hub.put_many(items, f"{rid} done")
        self.hub.delete(f"{rdir}/last.pt")
        if SQUASH_HISTORY:
            self.hub.squash()
        last.unlink(missing_ok=True)
        for p in out.glob("ckpt_*.pt"):
            p.unlink()
        log(f"  [{rid}] DONE  " + "  ".join(f"{k}={v:.3f}" for k, v in ev["mean"].items()))
        del model
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def stage_runs(self):
        arms = self.arm_table()
        plan = [(a, p, s) for s in self.P["seeds"] for a, p in arms.items()]     # all arms for seed 1 first
        log(f"stage 4/5: {len(plan)} runs planned ({len(arms)} arms x {len(self.P['seeds'])} seeds), "
            f"{self.P['steps']} steps each")
        for i, (a, p, s) in enumerate(plan, 1):
            log(f"--- run {i}/{len(plan)}: {a} seed {s}")
            try:
                self.run_one(a, p, s)
            except StopSession:
                raise
            except Exception as exc:
                if type(exc).__name__ == "SimulatedCrash":
                    raise
                tb = traceback.format_exc()
                log(f"  run failed: {type(exc).__name__}: {str(exc)[:200]}\n{tb[-600:]}")
                self.state.d["runs"][f"FAILED::{a}::{s}"] = dict(status="failed", err=str(exc)[:300])
                self.state.save(push=False)

    # ------------------------------------------------------------------ stage 5: report + samples
    def collect_results(self) -> List[dict]:
        out = []
        for rid, st in self.state.d["runs"].items():
            if st.get("status") != "done":
                continue
            p = self.dirs["runs"] / rid / "result.json"
            if not p.exists():
                self.hub.get(f"{self.hubdir}/runs/{rid}/result.json", p)
            if p.exists():
                out.append(json.loads(p.read_text()))
        return out

    def stage_report(self):
        results = self.collect_results()
        if not results:
            log("no finished runs yet; nothing to report")
            return
        rep = self.dirs["report"]
        make_report(results, rep)
        items = [(p, f"{self.hubdir}/report/{p.name}") for p in rep.glob("*") if p.is_file()]
        self.hub.put_many(items, "report")
        log(f"report written ({len(results)} runs): {rep}/report.md , results_table.tex , results_by_arm.csv")
        try:
            print((rep / "results_by_arm.csv").read_text()[:3000])
        except Exception:
            pass

    def stage_samples(self):
        if self.state.d["stages"].get("samples", {}).get("done"):
            return
        import soundfile as sf
        from hausafs2.data.audio import MelProcessor
        from hausafs2.frontend import HausaFrontend
        from hausafs2.synthesize import load_checkpoint, synthesize_mel
        from hausafs2.vocoder import load_hifigan, mel_to_wav
        arms = [a for a in MAIN_ARMS if a in self.arm_table()]
        rows = [l.split("|", 2) for l in (self.dirs["pre"] / "test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = rows[: self.P["n_samples"]]
        voc_zip = self.repo_dir / "vocoder_assets" / "generator_universal.pth.tar.zip"
        voc_cfg = self.repo_dir / "vocoder_assets" / "hifigan_config.json"
        if not voc_zip.exists():
            log("samples skipped: vocoder weights missing (vocoder_assets/ in your repo)")
            return
        gen = load_hifigan(str(voc_zip), str(voc_cfg), self.device)
        mp = MelProcessor(sr=22050)
        sd = self.dirs["samples"]
        seed0 = self.P["seeds"][0]
        fe = HausaFrontend("auto", TONE_LEXICON_PATH or None)
        for arm in arms:
            cfg = self.build_cfg(arm, self.arm_table()[arm], seed0)
            rid = self.run_id(arm, seed0, cfg)
            mp_path = self.dirs["runs"] / rid / "model.pt"
            if not mp_path.exists() and not self.hub.get(f"{self.hubdir}/runs/{rid}/model.pt", mp_path):
                log(f"  samples: no model for {arm}; skipped")
                continue
            model, mcfg, stats = load_checkpoint(str(mp_path), self.device)
            (sd / arm).mkdir(parents=True, exist_ok=True)
            for uid, spk, text in rows:
                r = synthesize_mel(model, fe, text, stats, spk, predict_tone=False, device=self.device)
                sf.write(str(sd / arm / f"{uid}.wav"), mel_to_wav(gen, r["mel"]), mp.sr)
        (sd / "ground_truth").mkdir(parents=True, exist_ok=True)
        for uid, spk, text in rows:
            src = self.dirs["subset"] / "wavs" / spk / f"{uid}.wav"
            if src.exists():
                shutil.copy2(src, sd / "ground_truth" / f"{uid}.wav")
        (sd / "texts.tsv").write_text("\n".join("\t".join(r) for r in rows), encoding="utf-8")
        self.push_tar(sd, "samples.tar")
        self.state.mark("samples", n=len(rows))
        self.state.save()

    # ------------------------------------------------------------------ driver
    def run(self):
        log(f"=== experiment '{self.exp}'  mode={self.P['mode']}  device={self.device}  "
            f"session budget {self.budget.limit / 3600:.1f} h ===")
        self.state.load()
        try:
            self.stage_subset()
            self.stage_align()
            self.stage_preprocess()
            self.stage_runs()
            self.stage_report()
            self.stage_samples()
            log("=== ALL STAGES COMPLETE ===")
            return "complete"
        except StopSession as exc:
            log(f"=== SESSION BUDGET REACHED ({exc}). Progress is saved. Start a NEW session and run this cell again. ===")
            return "stopped"
        finally:
            try:
                self.state.save(push=False)
                if not self.hub.flush_pending(max_wait_s=0):
                    log("  some uploads are still pending (rate limit); they will be retried on the next run")
                self.state.save(push=True)
            except Exception as exc:
                log(f"final state push failed: {exc}")


# =============================================================================
# 7. environment setup + entry point
# =============================================================================
def sh(cmd: str, check: bool = True):
    return subprocess.run(cmd, shell=True, check=check)


def setup_environment():
    need = {"pyworld": "pyworld", "tgt": "tgt", "soundfile": "soundfile", "librosa": "librosa", "yaml": "pyyaml",
            "transformers": "transformers", "huggingface_hub": "huggingface_hub", "pyarrow": "pyarrow",
            "matplotlib": "matplotlib", "scipy": "scipy", "pandas": "pandas", "tabulate": "tabulate"}
    missing = [p for m, p in need.items() if importlib.util.find_spec(m) is None]
    if missing:
        log("pip install", missing)
        sh(f"{sys.executable} -m pip install -q {' '.join(missing)}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_repo(dest: Path, recorded_commit: Optional[str]) -> str:
    url = GITHUB_URL
    if GITHUB_TOKEN and url.startswith("https://"):
        url = url.replace("https://", f"https://{GITHUB_TOKEN}@")
    if not dest.exists():
        sh(f"git clone -q {url} {dest}")
    else:
        sh(f"git -C {dest} fetch -q --all --tags", check=False)
    ref = recorded_commit if (recorded_commit and PIN_TO_RECORDED_COMMIT) else GITHUB_REF
    sh(f"git -C {dest} checkout -q {ref}")
    if not recorded_commit or not PIN_TO_RECORDED_COMMIT:
        sh(f"git -C {dest} pull -q --ff-only", check=False)
    commit = subprocess.run(f"git -C {dest} rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, str(dest))
    return commit


def main():
    setup_environment()
    import torch
    token = resolve_token()
    P = get_params(MODE)
    log(f"MODE={MODE}  params: " + ", ".join(f"{k}={P[k]}" for k in ("n_speakers", "utts_per_speaker", "steps", "seeds")))
    if not torch.cuda.is_available():
        log("WARNING: no GPU detected (Kaggle: Settings -> Accelerator -> GPU T4/P100). CPU will be extremely slow.")
    hub = HfHub(HF_REPO_ID, token)
    hub.ensure()
    # pin the code version used by the first session
    st_tmp = WORK_ROOT / "_peek_state.json"
    recorded = None
    if hub.get(f"exp/{EXP_NAME}/state.json", st_tmp, force=True):
        recorded = json.loads(st_tmp.read_text()).get("repo_commit")
    repo_dir = WORK_ROOT / "hausa-fastspeech2"
    commit = setup_repo(repo_dir, recorded)
    log(f"code commit: {commit}" + ("  (pinned from first session)" if recorded else ""))
    import transformers
    exp = Experiment(P, hub, HfParquetSource(DATASET_REPO, token), MMSAligner("cuda" if torch.cuda.is_available() else "cpu"),
                     WORK_ROOT, repo_dir)
    exp.state.load()
    if not exp.state.d.get("repo_commit"):
        exp.state.d.update(repo_commit=commit, mode=MODE, params=P, dataset=DATASET_REPO, started=time.time(),
                           versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                         python=sys.version.split()[0]))
        exp.state.save()
    elif exp.state.d.get("params") != P:
        log("NOTE: params differ from the first session of this experiment; finished runs are kept, "
            "new run ids (config hash) are used for changed settings. Use a new EXP_NAME for a clean start.")
    return exp.run()


if __name__ == "__main__":
    main()
