#!/usr/bin/env python
"""Audit tone/length annotation of a corpus BEFORE training.

    python scripts/audit_corpus.py --metadata data/hausa_corpus/metadata.csv --tone_mode auto [--lexicon lexicon/hausa_tone_lexicon.tsv]

Prints how many vowels have known/unknown tone, long/short counts, syllable weights and
linguistic warnings (e.g. falling tone on a light syllable).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hausafs2.frontend import HausaFrontend  # noqa: E402
from hausafs2.frontend.symbols import TONE_NAMES  # noqa: E402
from hausafs2.frontend.validate import check_utterance, corpus_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--metadata", required=True)
ap.add_argument("--tone_mode", default="auto")
ap.add_argument("--lexicon", default=None)
ap.add_argument("--show", type=int, default=10, help="print this many warnings")
a = ap.parse_args()
fe = HausaFrontend(a.tone_mode, a.lexicon)
utts = [fe(l.rstrip("\n").split("|", 2)[2]) for l in open(a.metadata, encoding="utf-8") if l.strip()]
rep = corpus_report(utts)
tot_v = sum(v for k, v in rep.items() if k.startswith("tone_"))
print(f"{len(utts)} utterances, {tot_v} vowels")
for i, n in enumerate(TONE_NAMES):
    if f"tone_{i}" in rep:
        print(f"  tone {n}: {rep[f'tone_{i}']} ({100 * rep[f'tone_{i}'] / tot_v:.1f}%)")
print(f"  long vowels {rep.get('long_vowels', 0)} | short vowels {rep.get('short_vowels', 0)}")
print(f"  warnings {rep.get('msg_warn', 0)} | info {rep.get('msg_info', 0)}")
shown = 0
for u in utts:
    for lvl, m in check_utterance(u):
        if lvl == "warn" and shown < a.show:
            print("  WARN:", m)
            shown += 1
if rep.get("tone_4", 0) / max(tot_v, 1) > 0.5:
    print("\nMore than half of the vowels have UNKNOWN tone. The model can still be trained (tone dropout /"
          " tone head), but tone accuracy will be limited: see README 'Tone annotation'.")
