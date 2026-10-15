#!/usr/bin/env python
"""Head-to-head: original-style FastSpeech 2 vs. the Hausa extension, same data / seed / budget.

Arms (all implemented with switches of the SAME code, so the comparison is controlled):
  fs2_baseline    phone ids only (no tone/length/weight/boundary/position inputs), no hierarchy, no
                  downstep, phoneme-level pitch (K=1), no tone head   -> original FastSpeech 2 recipe
  fs2_tone_length baseline + tone and vowel-length inputs (what one gets by writing tone/length into the
                  phone symbols, the usual trick for tonal languages)  -> a STRONG baseline
  hausa_full      everything: + weight/role/boundary/position features, downstep counter, hierarchical
                  prosody encoder, K=3 pitch contour, tone->pitch skip, tone dropout + tone head

Evaluation (held-out `val` and held-out `test` = utterances made of NEVER-SEEN words), with ground-truth
durations but PREDICTED pitch and energy, so pitch/energy predictor quality is actually measured:
  mel_l1        L1 to the true log-mel
  pitch_rmse    frame-level RMSE of speaker-normalised log-F0 (z) on non-pause frames
  rmse_H/L/F    same, restricted to vowel frames of each tone
  dur_mae       |predicted - true| duration in frames (non-pause tokens)
  ls_ratio      predicted long/short vowel duration ratio (true ~2)
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hausafs2.data.dataset import HausaTTSDataset, collate  # noqa: E402
from hausafs2.frontend import symbols as S  # noqa: E402
from hausafs2.model import batch_to_device  # noqa: E402
from hausafs2.model.variance_adaptor import contour_to_frames  # noqa: E402
from hausafs2.train import build_model, train  # noqa: E402

COMMON_OFF = dict(hier=dict(enabled=False), use_downstep=False, pitch_tone_skip=False,
                  pitch=dict(n_points=1), tone=dict(dropout_word=0.0, dropout_utt=0.0))
ARMS = {
    "fs2_baseline": dict(model=dict(input_features=[], **COMMON_OFF),
                         loss=dict(weights=dict(tone=0.0), tone_bearing_weight=1.0)),
    "fs2_tone_length": dict(model=dict(input_features=["length", "tone"], **COMMON_OFF),
                            loss=dict(weights=dict(tone=0.0), tone_bearing_weight=1.0)),
    "hausa_full": dict(model=dict(), loss=dict()),
    # --- diagnostic ablations (why is the full model worse on H/L levels?) ---
    "hausa_no_tonedrop": dict(model=dict(tone=dict(dropout_word=0.0, dropout_utt=0.0)),
                              loss=dict(weights=dict(tone=0.0))),
    "hausa_tonedrop_low": dict(model=dict(tone=dict(dropout_word=0.05, dropout_utt=0.0))),
    "tone_length_K3": dict(model=dict(input_features=["length", "tone"], hier=dict(enabled=False),
                                      use_downstep=False, pitch_tone_skip=False,
                                      tone=dict(dropout_word=0.0, dropout_utt=0.0)),
                           loss=dict(weights=dict(tone=0.0), tone_bearing_weight=1.0)),
    "hausa_no_tonedrop_no_hier": dict(model=dict(hier=dict(enabled=False),
                                                 tone=dict(dropout_word=0.0, dropout_utt=0.0)),
                                      loss=dict(weights=dict(tone=0.0))),
}
MAIN_ARMS = ["fs2_baseline", "fs2_tone_length", "hausa_full"]


def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
    return a


@torch.no_grad()
def evaluate(model, cfg, stats, pre, split, unmarked=False):
    pm, em = cfg["model"]["pitch"]["mode"], cfg["model"]["energy"]["mode"]
    ds = HausaTTSDataset(pre, split, pm, em, cfg["model"]["pitch"]["n_points"])
    model.eval()
    acc = dict(l1=0.0, n_mel=0, se=0.0, n_p=0, dur=0.0, n_d=0)
    tone_se = {1: [0.0, 0], 2: [0.0, 0], 3: [0.0, 0]}
    dl_long, dl_short = [], []
    for b in DataLoader(ds, batch_size=16, collate_fn=collate):
        b = batch_to_device(b, "cpu")
        feed = {k: v for k, v in b.items() if k not in ("pitch", "energy")}       # GT durations, predicted rest
        # unmarked=True simulates plain orthography: every tone hidden, filled in by the model's tone head
        out = model(feed, hide_tones=True, predict_tone=True) if unmarked else model(feed)
        T = b["mel"].size(1)
        valid = ~out["mel_pad"][:, :T]
        acc["l1"] += float(((out["mel_post"][:, :T] - b["mel"]).abs().mean(-1) * valid).sum())
        acc["n_mel"] += int(valid.sum())
        z = out["pitch"]
        z = contour_to_frames(z, out["tok_idx"], out["rel"]) if z.dim() == 3 else z
        z = z[:, :T]
        kind_f = b["kind"].gather(1, out["tok_idx"][:, :T])
        tone_f = b["tone"].gather(1, out["tok_idx"][:, :T])
        fm = valid & (kind_f != S.KIND_P)
        err = (z - b["pitch_frame"][:, :T]) ** 2
        acc["se"] += float((err * fm).sum()); acc["n_p"] += int(fm.sum())
        for t in (1, 2, 3):
            m = fm & (tone_f == t) & (kind_f == S.KIND_V)
            tone_se[t][0] += float((err * m).sum()); tone_se[t][1] += int(m.sum())
        tok_ok = (~out["src_pad"]) & (b["kind"] != S.KIND_P)
        pd = torch.clamp(torch.round(torch.exp(out["log_dur"]) - 1), min=0)
        acc["dur"] += float(((pd - b["duration"].float()).abs() * tok_ok).sum()); acc["n_d"] += int(tok_ok.sum())
        mono = tok_ok & (b["kind"] == S.KIND_V) & ~((b["phone"] == S.PHONE2ID["ai"]) | (b["phone"] == S.PHONE2ID["au"]))
        pdur = torch.exp(out["log_dur"]) - 1
        dl_long += pdur[mono & (b["length"] == 1)].tolist(); dl_short += pdur[mono & (b["length"] == 0)].tolist()
    r = dict(mel_l1=acc["l1"] / acc["n_mel"], pitch_rmse=(acc["se"] / acc["n_p"]) ** .5,
             dur_mae=acc["dur"] / acc["n_d"], ls_ratio=float(np.mean(dl_long) / np.mean(dl_short)))
    for t, n in zip((1, 2, 3), "HLF"):
        r[f"rmse_{n}"] = (tone_se[t][0] / max(tone_se[t][1], 1)) ** .5
    return {k: round(float(v), 3) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="preprocessed dir with train/val/test.txt")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "smoke.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--arms", nargs="+", default=MAIN_ARMS)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--add_unmarked", action="store_true", help="re-evaluate saved checkpoints in unmarked-text mode")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = yaml.safe_load(open(a.config))
    res_path = os.path.join(a.out, f"results_seed{a.seed}.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for arm in a.arms:
        if arm in results and a.add_unmarked:                        # evaluation only, no training
            cfg = merge(copy.deepcopy(base), copy.deepcopy(ARMS[arm]))
            cfg["dataset"]["out_dir"] = a.pre
            ck = torch.load(os.path.join(a.out, f"{arm}_s{a.seed}", "last.pt"))
            model = build_model(cfg, ck["stats"]); model.load_state_dict(ck["model"])
            results[arm]["val_unmarked"] = evaluate(model, cfg, ck["stats"], a.pre, "val.txt", True)
            results[arm]["test_unmarked"] = evaluate(model, cfg, ck["stats"], a.pre, "test.txt", True)
            json.dump(results, open(res_path, "w"), indent=1)
            print(arm, "unmarked", json.dumps(results[arm]["test_unmarked"]), flush=True)
            continue
        if arm in results:
            continue
        cfg = merge(copy.deepcopy(base), copy.deepcopy(ARMS[arm]))
        cfg["dataset"]["out_dir"] = a.pre
        cfg["train"].update(out_dir=os.path.join(a.out, f"{arm}_s{a.seed}"), total_steps=a.steps, seed=a.seed)
        print(f"\n##### ARM {arm} (seed {a.seed})", flush=True)
        t0 = time.time()
        train(cfg, max_steps=a.steps, log=lambda m: print(m[:160], flush=True) if ("VAL" in m or "parameters" in m) else None)
        ck = torch.load(os.path.join(cfg["train"]["out_dir"], "last.pt"))
        model = build_model(cfg, ck["stats"]); model.load_state_dict(ck["model"])
        n_par = sum(p.numel() for p in model.parameters())
        results[arm] = dict(params=n_par, train_s=round(time.time() - t0),
                            val=evaluate(model, cfg, ck["stats"], a.pre, "val.txt"),
                            test_novel=(evaluate(model, cfg, ck["stats"], a.pre, "test.txt")
                                        if os.path.exists(os.path.join(a.pre, "test.txt")) else None))
        print(arm, json.dumps(results[arm]), flush=True)
        json.dump(results, open(res_path, "w"), indent=1)
    print("\nDONE")


if __name__ == "__main__":
    main()
