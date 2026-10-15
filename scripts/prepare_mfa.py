#!/usr/bin/env python
"""Prepare MFA inputs (see hausafs2/data/mfa.py).

    python scripts/prepare_mfa.py --corpus_dir data/hausa_corpus --out_dir data/mfa_input
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hausafs2.data.mfa import prepare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpus_dir", required=True)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
n = prepare(a.corpus_dir, a.out_dir)
print(f"wrote {n} dictionary entries to {os.path.join(a.out_dir, 'hausa.dict')}")
print("Next (check `mfa train --help` for the flags of your MFA version):")
print(f"  mfa train --output_directory data/hausa_alignments {a.out_dir} {a.out_dir}/hausa.dict hausa_acoustic.zip")
