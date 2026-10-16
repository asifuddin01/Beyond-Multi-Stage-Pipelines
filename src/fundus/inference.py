"""Test-time augmentation, ensembling and the three-stage cascade."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import FundusDataset, eval_transform, make_loader
from .engine import predict
from .labels import GKEY, GROUP_COLS, GROUPS, RARE_COL, TARGETS


# --------------------------------------------------------------------------- #
#  Test-time augmentation
# --------------------------------------------------------------------------- #
def tta_views(x: torch.Tensor, n: int = 4) -> list[torch.Tensor]:
    """Label-preserving views of a fundus batch (N, C, H, W):
    identity, horizontal flip, vertical flip, both flips (+ rotations for n=8)."""
    views = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
    if n >= 8:
        r1, r3 = torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])
        views += [r1, r3, torch.flip(r1, [3]), torch.flip(r3, [3])]
    return views[:n]


@torch.no_grad()
def predict_tta(model, loader, dev, n_views: int = 4):
    """Sigmoid probabilities averaged over ``n_views`` augmented views."""
    model.eval()
    ys, ps = [], []
    amp = dev.type == "cuda"
    for x, y, _ in tqdm(loader, desc=f"  TTA x{n_views}", leave=False):
        x = x.to(dev, non_blocking=True)
        acc = 0
        for v in tta_views(x, n_views):
            with torch.autocast(dev.type, enabled=amp):
                acc = acc + torch.sigmoid(model(v).float())
        ps.append((acc / n_views).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def sweep_blend_weight(p_a_val: np.ndarray, p_b_val: np.ndarray, y_val: np.ndarray,
                       grid=np.linspace(0, 1, 21)) -> float:
    """Weight w for ``w * A + (1 - w) * B`` maximising validation macro-AP
    (scored at a uniform 0.5 threshold, as in the paper)."""
    from .metrics import score_all
    best_w, best = 0.5, -1.0
    for w in grid:
        s = score_all(y_val, w * p_a_val + (1 - w) * p_b_val, 0.5)["macro_ap"]
        if s > best:
            best, best_w = s, float(w)
    return best_w


# --------------------------------------------------------------------------- #
#  Three-stage cascade
# --------------------------------------------------------------------------- #
def run_cascade(test_df: pd.DataFrame, img_dir, s1_model, s2_model, s3_models: dict,
                s1_threshold: float, s2_thresholds: dict, s3_thresholds: dict,
                dev, batch_size: int = 8, img_size: int = 384, num_workers: int = 4):
    """Stage 1 -> Stage 2 -> Stage 3 with hard gating.

    Images rejected by a gate receive probability 0 for every downstream target,
    exactly as a deployed cascade would behave.

    Returns ``final_prob`` / ``final_pred`` (images x 39 targets) and the
    per-stage tables ``s1``, ``s2`` for the loss-cascade analysis.
    """
    tf = eval_transform(img_size)

    def loader(df):
        ds = FundusDataset(df, img_dir, tf, label_cols=TARGETS)
        return ds, make_loader(ds, batch_size, train=False, num_workers=num_workers)

    names = test_df["image"].tolist()
    final_prob = pd.DataFrame(0.0, index=names, columns=TARGETS)

    # Stage 1 - healthy vs unhealthy (softmax, index 1 = unhealthy)
    ds, dl = loader(test_df)
    _, p1, _ = predict(s1_model, dl, dev, activation="softmax")
    s1 = pd.DataFrame({"image": ds.names, "p_unhealthy": p1})
    s1["pred_unhealthy"] = (s1.p_unhealthy >= s1_threshold).astype(int)

    # Stage 2 - multi-label group routing on Stage-1 survivors
    surv = test_df[test_df.image.isin(s1.loc[s1.pred_unhealthy == 1, "image"])]
    ds, dl = loader(surv)
    _, p2, _ = predict(s2_model, dl, dev)
    s2 = pd.DataFrame(p2, columns=[f"p_{g}" for g in GROUP_COLS])
    s2.insert(0, "image", ds.names)
    for g in GROUP_COLS:
        s2[f"pred_{g}"] = (s2[f"p_{g}"] >= s2_thresholds[g]).astype(int)

    # G_Rare is decided at Stage 2 (too few examples for a Stage-3 model)
    final_prob.loc[s2.image, RARE_COL] = s2[f"p_{RARE_COL}"].values
    final_pred = pd.DataFrame(0, index=names, columns=TARGETS)
    final_pred.loc[s2.image, RARE_COL] = s2[f"pred_{RARE_COL}"].values

    # Stage 3 - one fine-grained classifier per routed group
    for key, model in s3_models.items():
        gcol, classes = GKEY[key], GROUPS[GKEY[key]]
        routed = test_df[test_df.image.isin(s2.loc[s2[f"pred_{gcol}"] == 1, "image"])]
        if routed.empty:
            continue
        ds = FundusDataset(routed, img_dir, tf, label_cols=classes)
        dl = make_loader(ds, batch_size, train=False, num_workers=num_workers)
        _, p3, _ = predict(model, dl, dev)
        th = np.array([s3_thresholds[key].get(c, 0.5) for c in classes])
        final_prob.loc[ds.names, classes] = p3
        final_pred.loc[ds.names, classes] = (p3 >= th).astype(int)
    return final_prob, final_pred, s1, s2


def loss_cascade(test_df: pd.DataFrame, final_pred: pd.DataFrame,
                 s1: pd.DataFrame, s2: pd.DataFrame) -> dict:
    """Attribute every missed true disease instance to the stage that lost it."""
    kept = set(s1.loc[s1.pred_unhealthy == 1, "image"])
    routed = {g: set(s2.loc[s2[f"pred_{g}"] == 1, "image"]) for g in GROUPS}
    out = {"lost_stage1": 0, "lost_stage2": 0, "lost_stage3": 0, "found": 0}
    for _, r in test_df.set_index("image").iterrows():
        img = r.name
        for g, members in GROUPS.items():
            for d in members:
                if r[d] != 1:
                    continue
                if img not in kept:
                    out["lost_stage1"] += 1
                elif img not in routed[g]:
                    out["lost_stage2"] += 1
                elif final_pred.loc[img, d] == 1:
                    out["found"] += 1
                else:
                    out["lost_stage3"] += 1
        if r[RARE_COL] == 1:          # G_Rare is decided entirely at Stage 2
            if img not in kept:
                out["lost_stage1"] += 1
            elif final_pred.loc[img, RARE_COL] == 1:
                out["found"] += 1
            else:
                out["lost_stage2"] += 1
    out["total"] = sum(out.values())
    return out
