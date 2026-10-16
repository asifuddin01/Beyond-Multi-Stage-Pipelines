"""Evaluation metrics. ``score_all`` is the single scoring routine used for every
row of the ablation table, so all configurations are directly comparable."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

THRESHOLD_GRID = np.linspace(0.05, 0.95, 37)


def best_f1_threshold(y: np.ndarray, p: np.ndarray, grid=THRESHOLD_GRID) -> float:
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)


def tune_thresholds(y_val: np.ndarray, p_val: np.ndarray, min_pos: int = 10) -> np.ndarray:
    """Per-class max-F1 thresholds on *validation*. Classes with fewer than
    ``min_pos`` validation positives keep the default 0.5."""
    th = np.full(y_val.shape[1], 0.5)
    for j in range(y_val.shape[1]):
        if y_val[:, j].sum() >= min_pos:
            th[j] = best_f1_threshold(y_val[:, j], p_val[:, j])
    return th


def macro_auc_ap(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    cols = [j for j in range(y.shape[1]) if len(np.unique(y[:, j])) > 1]
    if not cols:
        return float("nan"), float("nan")
    return (float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in cols])),
            float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in cols])))


def score_all(y: np.ndarray, prob: np.ndarray, th: np.ndarray | float, tag: str = "") -> dict:
    """Image-level and label-level metrics over all targets with >=1 test positive.

    detection = fraction of true disease instances that are predicted positive.
    """
    y = y.astype(int)
    b = (prob >= th).astype(int)
    mask = [j for j in range(y.shape[1]) if y[:, j].sum() > 0]
    well = [j for j in mask if y[:, j].sum() >= 10]
    f1s = {j: f1_score(y[:, j], b[:, j], zero_division=0) for j in mask}
    inter = ((b == 1) & (y == 1)).sum(1)
    union = ((b == 1) | (y == 1)).sum(1)
    total = int(y[:, mask].sum())
    found = int(((b[:, mask] == 1) & (y[:, mask] == 1)).sum())
    return {
        "config": tag,
        "exact_acc": float((b == y).all(axis=1).mean()),
        "jaccard": float(np.where(union > 0, inter / np.maximum(union, 1), 1.0).mean()),
        "micro_f1": float(f1_score(y[:, mask].ravel(), b[:, mask].ravel(), zero_division=0)),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "macro_f1_well": float(np.mean([f1s[j] for j in well])) if well else float("nan"),
        "macro_auc": float(np.mean([roc_auc_score(y[:, j], prob[:, j]) for j in mask
                                    if len(np.unique(y[:, j])) > 1])),
        "macro_ap": float(np.mean([average_precision_score(y[:, j], prob[:, j]) for j in mask])),
        "detection": found / max(1, total),
    }


def per_class_table(y: np.ndarray, prob: np.ndarray, th: np.ndarray, names: Sequence[str]):
    import pandas as pd
    b = (prob >= th).astype(int)
    rows = []
    for j, c in enumerate(names):
        if y[:, j].sum() == 0:
            continue
        rows.append({"disease": c, "n_true": int(y[:, j].sum()),
                     "f1": f1_score(y[:, j], b[:, j], zero_division=0),
                     "ap": average_precision_score(y[:, j], prob[:, j]),
                     "auc": roc_auc_score(y[:, j], prob[:, j]) if len(np.unique(y[:, j])) > 1 else np.nan,
                     "thr": float(np.broadcast_to(th, (len(names),))[j]),
                     "pred_pos": int(b[:, j].sum())})
    return pd.DataFrame(rows).set_index("disease").sort_values("n_true", ascending=False)


def per_source_macro_auc(y: np.ndarray, p: np.ndarray, sources: np.ndarray) -> dict:
    """Macro-AUC inside each source. Large gaps between sources flag a dataset shortcut."""
    out = {}
    for s in np.unique(sources):
        m = sources == s
        aucs = [roc_auc_score(y[m, j], p[m, j]) for j in range(y.shape[1])
                if len(np.unique(y[m, j])) > 1]
        if aucs:
            out[f"macroauc_{s}"] = float(np.mean(aucs))
    return out
