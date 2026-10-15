#!/usr/bin/env python
"""Smoke-test the pipeline (and compare arms) on ~20 clips of a Hugging Face Hausa dataset.

    pip install datasets torchaudio
    python scripts/hf_smoke_20.py --dataset suleiman2003/unified-hausa-speech --n 20 --out data/hf20 --steps 600

Stages: download+export -> text audit -> CTC forced alignment (MMS_FA, rough!) -> preprocess -> compare_arms.

READ THIS FIRST
* The sandbox this was written in had no access to huggingface.co, so stages 1 and 3 (network / MMS weights) are
  UNTESTED end to end; their pure logic is unit-tested (tests/test_hf_tools.py). Report errors and I will fix them.
* The dataset text is plain orthography: no tone marks, no vowel-length marks. Tone/length inputs are therefore
  UNKNOWN/short almost everywhere, so the tone-aware features cannot help much here without a tone lexicon.
* 20 clips cannot train a usable TTS model. This checks that the pipeline runs and the losses move; do NOT read the
  arm comparison on 20 clips as evidence of quality.
* CTC alignments are rough (~20 ms, peaky). Use MFA on the full dataset for real training.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hausafs2.data.alignment import write_textgrid  # noqa: E402
from hausafs2.data.hf_export import export_examples  # noqa: E402
from hausafs2.frontend import HausaFrontend  # noqa: E402
from hausafs2.frontend.validate import corpus_report  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="suleiman2003/unified-hausa-speech")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/hf20")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--tone_mode", default="auto")
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--skip_compare", action="store_true")
    a = ap.parse_args()
    corpus, align, pre = (os.path.join(a.out, d) for d in ("corpus", "alignments", "pre"))

    # 1. download + export (streaming, shuffled so several speakers appear)
    from datasets import load_dataset
    ds = load_dataset(a.dataset, split=a.split, streaming=True).shuffle(seed=0, buffer_size=500)
    fe = HausaFrontend(a.tone_mode, a.lexicon)

    def clean(stream):                       # skip texts our frontend cannot represent faithfully
        for ex in stream:
            t = ex.get("text") or ""
            if any(ch.isdigit() for ch in t):
                continue                     # digits are not expanded -> transcript/audio mismatch
            yield ex

    rows = export_examples(clean(ds), corpus, n=a.n)
    print(f"exported {len(rows)} clips from {len({r[1] for r in rows})} speaker(s)")

    # 2. text audit
    utts = [fe(r[2]) for r in rows]
    rep = corpus_report(utts)
    print("annotation audit:", {k: v for k, v in rep.items() if not k.startswith("weight")})

    # 3. rough CTC alignment -> TextGrids
    import librosa
    from hausafs2.data.ctc_align import align_utterance
    ok = 0
    for (uid, spk, text), u in zip(rows, utts):
        try:
            wav, _ = librosa.load(os.path.join(corpus, "wavs", spk, f"{uid}.wav"), sr=16000)
            iv = align_utterance(wav, u.labels, [t.kind == 2 for t in u.tokens])
            os.makedirs(os.path.join(align, spk), exist_ok=True)
            write_textgrid(os.path.join(align, spk, f"{uid}.TextGrid"), iv)
            ok += 1
        except Exception as e:                                    # keep going; report
            print(f"  align failed for {uid}: {type(e).__name__}: {e}")
    print(f"aligned {ok}/{len(rows)}")

    # 4. preprocess (smoke config; multi-speaker if >1 speaker)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "smoke.yaml")))
    cfg["dataset"].update(corpus_dir=corpus, alignment_dir=align, out_dir=pre, val_size=max(2, a.n // 5))
    cfg["frontend"].update(tone_mode=a.tone_mode, lexicon_path=a.lexicon)
    cfg["model"]["multi_speaker"] = len({r[1] for r in rows}) > 1
    cfg_path = os.path.join(a.out, "config.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    from hausafs2.data.preprocess import Preprocessor
    Preprocessor(cfg).run()

    if a.skip_compare:
        return
    # 5. compare arms (tiny data: pipeline check only)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compare_arms.py"), "--pre", pre,
                    "--config", cfg_path, "--out", os.path.join(a.out, "arms"), "--steps", str(a.steps)], check=True)
    print("results:", os.path.join(a.out, "arms"))


if __name__ == "__main__":
    main()
