"""Generic, resumable training loop shared by every model in the study.

AdamW + linear warm-up + cosine decay, mixed precision, gradient accumulation and
clipping. A checkpoint ``best_<metric>.pt`` is kept for every monitored metric and
training stops once *none* of them has improved for ``patience`` epochs.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from .config import TrainConfig
from .metrics import macro_auc_ap, per_source_macro_auc


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Forward passes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, loader, dev, activation: str = "sigmoid"):
    """Returns (targets, probabilities, sources)."""
    model.eval()
    ys, ps, ss = [], [], []
    amp = dev.type == "cuda"
    for x, y, src in tqdm(loader, desc="  predict", leave=False):
        with torch.autocast(dev.type, enabled=amp):
            out = model(x.to(dev, non_blocking=True))
        out = out.float()
        ps.append((torch.softmax(out, 1)[:, 1] if activation == "softmax" else torch.sigmoid(out)).cpu())
        ys.append(y if torch.is_tensor(y) else torch.as_tensor(y))
        ss += list(src)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), np.array(ss)


def evaluate_multilabel(model, loader, dev):
    y, p, s = predict(model, loader, dev, "sigmoid")
    auc, ap = macro_auc_ap(y, p)
    return {"macro_auc": auc, "macro_ap": ap, **per_source_macro_auc(y, p, s)}, (y, p, s)


def evaluate_binary(model, loader, dev):
    y, p, s = predict(model, loader, dev, "softmax")
    pred = (p >= 0.5).astype(int)
    m = {"val_acc": accuracy_score(y, pred), "val_f1": f1_score(y, pred, zero_division=0),
         "val_auc": roc_auc_score(y, p)}
    for src in np.unique(s):
        k = s == src
        if len(np.unique(y[k])) > 1:
            m[f"auc_{src}"] = roc_auc_score(y[k], p[k])
    return m, (y, p, s)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def _cosine_with_warmup(opt, steps_per_epoch: int, cfg: TrainConfig):
    total = cfg.epochs * steps_per_epoch
    warm = cfg.warmup_epochs * steps_per_epoch

    def f(step):
        if step < warm:
            return step / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def fit(model: nn.Module, train_loader, val_loader, criterion: nn.Module,
        evaluate: Callable, out_dir: str | Path, cfg: TrainConfig,
        monitors: Sequence[str] = ("macro_ap",), resume: bool = True) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    model.to(dev)
    criterion.to(dev)
    accum = max(1, cfg.grad_accum)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = _cosine_with_warmup(opt, max(1, math.ceil(len(train_loader) / accum)), cfg)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")

    def save(path, epoch, hist):
        torch.save({"epoch": epoch, "history": hist, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
                    "scaler": scaler.state_dict()}, path)

    start, hist = 0, []
    best = {m: -math.inf for m in monitors}
    if resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sch.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, hist = ck["epoch"] + 1, ck["history"]
        for m in monitors:
            best[m] = max([h.get(m, -math.inf) for h in hist] or [-math.inf])
        print(f"resumed at epoch {start}: " + ", ".join(f"best {m}={v:.4f}" for m, v in best.items()))

    patience = 0
    for epoch in range(start, cfg.epochs):
        model.train()
        t0, running, n_seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"epoch {epoch:02d}")
        for i, (x, y, _) in bar:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = criterion(model(x), y) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum * x.size(0)
            n_seen += x.size(0)
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()
            bar.set_postfix(loss=f"{running / n_seen:.4f}")

        metrics, _ = evaluate(model, val_loader, dev)
        row = {"epoch": epoch, "train_loss": running / max(1, n_seen),
               "lr": sch.get_last_lr()[0], "secs": round(time.time() - t0, 1), **metrics}
        hist.append(row)
        print(f"e{epoch:02d} loss {row['train_loss']:.4f} | " +
              " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        with open(out / "train_log.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0 or f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        save(out / "latest.pt", epoch, hist)

        improved = False
        for m in monitors:
            v = metrics.get(m, float("nan"))
            if not math.isnan(v) and v > best[m]:
                best[m], improved = v, True
                save(out / f"best_{m}.pt", epoch, hist)
                print(f"   ** new best {m}={v:.4f}")
        patience = 0 if improved else patience + 1
        if patience >= cfg.patience:
            print(f"early stop at epoch {epoch}")
            break
    print("done: " + ", ".join(f"best {m}={v:.4f}" for m, v in best.items()))
    return hist
