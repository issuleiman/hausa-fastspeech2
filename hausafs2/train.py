"""Training loop for HausaFastSpeech2.

    python -m hausafs2.train -c configs/hausa_single.yaml [--resume output/.../last.pt]

Optimiser follows the paper: Adam(beta1=0.9, beta2=0.98, eps=1e-9) with the
Transformer warm-up / inverse-sqrt-decay learning-rate schedule (Vaswani et al.),
expressed through a peak learning rate.
"""
import argparse
import glob
import json
import math
import os
import random
import time
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import BucketBatchSampler, HausaTTSDataset, collate
from .model import HausaFastSpeech2, HausaLoss, batch_to_device, count_parameters, diagnostics


def noam_lr(step: int, peak_lr: float, warmup: int) -> float:
    step = max(step, 1)
    return peak_lr * min(step / warmup, math.sqrt(warmup / step))


def build_model(cfg: dict, stats: dict) -> HausaFastSpeech2:
    return HausaFastSpeech2(cfg["model"], dict(
        pitch_range=stats["pitch_range"], energy_range=stats["energy_range"],
        n_mels=stats["n_mels"], n_speakers=stats["n_speakers"]))


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> dict:
    model.eval()
    sums, n, diag = {}, 0, {}
    for b in loader:
        b = batch_to_device(b, device)
        out = model(b)
        L = loss_fn(out, b)
        for k, v in L.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        # fully unmarked-text probe accuracy (all tones hidden)
        out_h = model(b, hide_tones=True)
        for k, v in {**diagnostics(out, b), **{k: v for k, v in diagnostics(out_h, b).items() if k == "tone_probe_acc"}}.items():
            diag.setdefault(k, []).append(v)
        n += 1
    model.train()
    res = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
    res.update({f"val_{k}": float(np.mean(v)) for k, v in diag.items()})
    return res


def train(cfg: dict, resume: Optional[str] = None, max_steps: Optional[int] = None,
          device: Optional[str] = None, log: Callable = print) -> list:
    tc = cfg["train"]
    seed = tc.get("seed", 1234)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pre = cfg["dataset"]["out_dir"]
    pm, em = cfg["model"]["pitch"].get("mode", "contour"), cfg["model"]["energy"].get("mode", "phoneme")
    K = cfg["model"]["pitch"].get("n_points")
    train_ds = HausaTTSDataset(pre, "train.txt", pm, em, K)
    val_ds = HausaTTSDataset(pre, "val.txt", pm, em, K)
    stats = train_ds.stats
    bs = tc["batch_size"]
    train_loader = DataLoader(train_ds, batch_sampler=BucketBatchSampler(train_ds, bs, True, seed=seed),
                              collate_fn=collate, num_workers=tc.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate)

    model = build_model(cfg, stats).to(device)
    loss_fn = HausaLoss(cfg.get("loss", {}), pm, em)
    opt = torch.optim.Adam(model.parameters(), lr=tc["peak_lr"], betas=tuple(tc["betas"]), eps=tc["eps"])
    out_dir = tc["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log(f"model parameters: {count_parameters(model) / 1e6:.2f}M | device {device} | "
        f"train {len(train_ds)} / val {len(val_ds)} utterances")

    step = 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        log(f"resumed from {resume} at step {step}")

    total = max_steps if max_steps is not None else tc["total_steps"]
    accum = tc.get("grad_accum", 1)
    history, t0, micro = [], time.time(), 0
    model.train()
    done = False
    while not done:
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            out = model(batch)
            L = loss_fn(out, batch)
            (L["total"] / accum).backward()
            micro += 1
            if micro % accum:
                continue
            step += 1
            for g in opt.param_groups:
                g["lr"] = noam_lr(step, tc["peak_lr"], tc["warmup_steps"])
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad(set_to_none=True)

            if step % tc["log_every"] == 0 or step == 1:
                rec = dict(step=step, lr=opt.param_groups[0]["lr"], t=round(time.time() - t0, 1),
                           **{k: round(float(v.detach()), 4) for k, v in L.items()})
                history.append(rec)
                log(" ".join(f"{k}={v}" for k, v in rec.items()))
            if step % tc["val_every"] == 0 or step == total:
                v = validate(model, val_loader, loss_fn, device)
                v["step"] = step
                history.append(v)
                log("VAL " + " ".join(f"{k}={round(x, 4) if isinstance(x, float) else x}" for k, x in v.items()))
            if step % tc["save_every"] == 0 or step == total:
                _save(model, opt, step, cfg, stats, out_dir, tc.get("keep_last", 3))
            if step >= total:
                done = True
                break
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return history


def _save(model, opt, step, cfg, stats, out_dir, keep_last):
    ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg, stats=stats)
    path = os.path.join(out_dir, f"ckpt_{step}.pt")
    torch.save(ck, path)
    torch.save(ck, os.path.join(out_dir, "last.pt"))
    for old in sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")), key=os.path.getmtime)[:-keep_last]:
        os.remove(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
