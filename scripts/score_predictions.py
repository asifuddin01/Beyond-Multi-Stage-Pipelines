#!/usr/bin/env python
"""Score released predictions against the test ground truth (no GPU needed).

    python scripts/score_predictions.py \
        --gt  $FUNDUS_DATA_ROOT/test/merged_test.csv \
        --prob results/final_ensemble/test_prob_ensemble_flat_lc_w055_tta.csv \
        --per-disease results/final_ensemble/test_per_disease_ensemble_flat_lc.csv

Thresholds are read from the ``thr`` column of the per-disease file, so the
binary predictions are reproduced exactly. ``--subset`` restricts scoring to the
class subsets used for the literature comparison (``mured15`` or ``rfmid26``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fundus.labels import GROUPS, RARE, RARE_COL, TARGETS  # noqa: E402
from fundus.metrics import score_all  # noqa: E402

SUBSETS = {
    "mured15": "DR MH ODC TSLN ARMD DN MYA BRVO ODP CRVO RS ODE LS CSR CRS".split(),
    "rfmid26": ("DR ARMD MH DN MYA BRVO TSLN ERM LS MS CSR ODC CRVO TV AH ODP ST AION PT RT "
                "RS CRS EDN RPEC MHL RP").split(),
}


def ground_truth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
    for c in TARGETS[:-1] + RARE:
        if c not in df.columns:
            df[c] = 0
    df[RARE_COL] = (df[RARE].sum(axis=1) > 0).astype(int)
    return df.set_index(df["image"].astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="test CSV with image + disease columns")
    ap.add_argument("--prob", required=True, help="probability CSV (<target>_prob columns)")
    ap.add_argument("--per-disease", required=True, help="CSV with a 'thr' column per target")
    ap.add_argument("--subset", choices=sorted(SUBSETS), default=None)
    a = ap.parse_args()

    prob = pd.read_csv(a.prob).set_index("image")
    prob.columns = [c.removesuffix("_prob") for c in prob.columns]
    thr = pd.read_csv(a.per_disease).set_index("disease")["thr"].reindex(TARGETS).fillna(0.5)
    gt = ground_truth(a.gt).loc[prob.index]

    cols = SUBSETS[a.subset] if a.subset else TARGETS
    y = gt[cols].values.astype(int)
    p = prob[cols].values
    res = score_all(y, p, thr[cols].values, a.subset or "all 39 targets")
    print(pd.Series(res).to_string())


if __name__ == "__main__":
    main()
