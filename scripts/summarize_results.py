#!/usr/bin/env python
"""Summarise results/results_seed*.json (written by scripts/compare_arms.py)."""
import glob
import json
import os
import numpy as np

here = os.path.join(os.path.dirname(__file__), "..", "results")
R = {}
for f in sorted(glob.glob(os.path.join(here, "results_seed*.json"))):
    for arm, v in json.load(open(f)).items():
        R.setdefault(arm, []).append(v)
metrics = ["mel_l1", "pitch_rmse", "rmse_H", "rmse_L", "rmse_F", "dur_mae", "ls_ratio"]
order = ["fs2_baseline", "fs2_tone_length", "tone_length_K3", "hausa_no_tonedrop_no_hier", "hausa_no_tonedrop",
         "hausa_tonedrop_low", "hausa_full"]
for split in ("val", "test_novel", "test_unmarked"):
    print(f"\n=== {split}: mean over the seeds available per arm (n); lower is better except ls_ratio (true ~2.0)")
    print(f"{'arm':27s}{'n':>2s}" + "".join(f"{m:>11s}" for m in metrics) + f"{'params':>10s}")
    for arm in order:
        vs = [v[split] for v in R.get(arm, []) if v.get(split)]
        if vs:
            print(f"{arm:27s}{len(vs):>2d}" + "".join(f"{np.mean([v[m] for v in vs]):11.3f}" for m in metrics)
                  + f"{R[arm][0]['params']:10d}")
