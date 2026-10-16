"""Data loading, label derivation and the shared preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .labels import GROUPS, RARE, RARE_COL, DISEASES, source_of

# ImageNet statistics (the normalisation of the timm Swin-V2 checkpoint).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# --------------------------------------------------------------------------- #
#  Preprocessing
# --------------------------------------------------------------------------- #
class RetinaSquareCrop:
    """Crop to the bounding box of the illuminated retina, then zero-pad the short
    side so the fundus is centred in a square. No retinal tissue is discarded."""

    def __init__(self, tol: int = 7):
        self.tol = tol

    def __call__(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img.convert("RGB"))
        h, w = a.shape[:2]
        mask = a.mean(2) > self.tol
        if mask.sum() == 0:                      # all-black safety net
            s = min(h, w)
            y, x = (h - s) // 2, (w - s) // 2
            return Image.fromarray(a[y:y + s, x:x + s])
        ys, xs = np.where(mask)
        crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        top, left = (side - ch) // 2, (side - cw) // 2
        sq = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return Image.fromarray(sq)


def train_transform(img_size: int = 384) -> T.Compose:
    return T.Compose([
        RetinaSquareCrop(),
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(img_size: int = 384) -> T.Compose:
    return T.Compose([
        RetinaSquareCrop(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


# --------------------------------------------------------------------------- #
#  Label tables
# --------------------------------------------------------------------------- #
def load_split(csv_path: str | Path, img_col: str = "image") -> pd.DataFrame:
    """Read a merged split CSV and derive every column the models need.

    Required columns: ``image`` (identifier without extension), ``Normal``
    (1 = healthy) and one 0/1 column per native disease label. Missing disease
    columns are created as zeros, so a split need not contain every class.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
    df[img_col] = df[img_col].astype(str)
    for d in DISEASES + RARE:
        if d not in df.columns:
            df[d] = 0
    df[RARE_COL] = (df[RARE].sum(axis=1) > 0).astype(int)
    for g, members in GROUPS.items():
        df[g] = (df[members].sum(axis=1) > 0).astype(int)
    df["unhealthy"] = 1 - df["Normal"].astype(int)
    df["source"] = df[img_col].map(source_of)
    return df


def unhealthy_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["unhealthy"] == 1].reset_index(drop=True)


def source_balanced_negatives(df: pd.DataFrame, classes: Sequence[str],
                              ratio: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Return positives for ``classes`` plus negatives drawn *per source* in
    proportion to that source's positives.

    When a label is almost exclusive to one source (e.g. media haze in RFMiD,
    cataract in ODIR) a classifier can learn the camera instead of the lesion.
    Equalising the positive rate inside every source removes that shortcut.
    """
    pos = df[df[list(classes)].sum(axis=1) > 0]
    pool = df[df[list(classes)].sum(axis=1) == 0]
    rng = np.random.RandomState(seed)
    picks = []
    for src, grp in pos.groupby("source"):
        avail = pool[pool["source"] == src]
        take = min(int(round(len(grp) * ratio)), len(avail))
        if take > 0:
            picks.append(avail.sample(n=take, random_state=rng.randint(1_000_000)))
    neg = pd.concat(picks) if picks else pool.iloc[0:0]
    return pd.concat([pos, neg]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
def _index_images(img_dir: Path) -> dict[str, Path]:
    return {f.stem: f for f in img_dir.iterdir() if f.is_file() and f.suffix in IMG_EXTS}


class FundusDataset(Dataset):
    """Returns ``(image, target, source)``.

    ``label_cols`` gives a float multi-hot target; ``class_col`` gives an integer
    class index (used by the binary Stage-1 softmax model).
    """

    def __init__(self, df: pd.DataFrame, img_dir: str | Path, transform,
                 label_cols: Sequence[str] | None = None, class_col: str | None = None,
                 img_col: str = "image"):
        assert (label_cols is None) != (class_col is None), "give label_cols OR class_col"
        index = _index_images(Path(img_dir))
        keep = df[img_col].isin(index.keys())
        missing = (~keep).sum()
        if missing:
            print(f"[warn] {missing} rows have no image on disk, e.g. "
                  f"{df.loc[~keep, img_col].head(3).tolist()}")
        df = df[keep].reset_index(drop=True)
        self.names = df[img_col].tolist()
        self.paths = [index[n] for n in self.names]
        self.sources = df["source"].tolist()
        self.transform = transform
        if label_cols is not None:
            self.labels = df[list(label_cols)].values.astype("float32")
            self.multilabel = True
        else:
            self.labels = df[class_col].values.astype("int64")
            self.multilabel = False

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.paths[i]).convert("RGB"))
        y = torch.from_numpy(self.labels[i]) if self.multilabel else int(self.labels[i])
        return img, y, self.sources[i]


def make_loader(ds: Dataset, batch_size: int, train: bool, num_workers: int = 4) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      drop_last=train and len(ds) > batch_size,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def pos_weight(labels: np.ndarray, clip: float = 50.0) -> torch.Tensor:
    """Per-class BCE positive weight neg/pos, clipped to ``[1, clip]``."""
    pos = labels.sum(0)
    neg = len(labels) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, clip), dtype=torch.float)
