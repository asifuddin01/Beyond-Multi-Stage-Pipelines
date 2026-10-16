"""Paths and hyper-parameters.

Every path is derived from one data root so the code runs on any machine:

    export FUNDUS_DATA_ROOT=/path/to/merged_dataset

Expected layout (see ``data/README.md``)::

    merged_dataset/
        train/merged_train.csv   train/images/
        val/merged_val.csv       val/images/
        test/merged_test.csv     test/images/
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_NAME = "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"
IMG_SIZE = 384
SEED = 42


@dataclass
class Paths:
    data_root: Path = field(default_factory=lambda: Path(
        os.environ.get("FUNDUS_DATA_ROOT", "data/merged_dataset")))
    run_root: Path = field(default_factory=lambda: Path(
        os.environ.get("FUNDUS_RUN_ROOT", "runs")))

    def csv(self, split: str) -> Path:
        return self.data_root / split / f"merged_{split}.csv"

    def images(self, split: str) -> Path:
        return self.data_root / split / "images"

    def run(self, name: str) -> Path:
        p = self.run_root / name
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class TrainConfig:
    """Shared optimisation protocol (held constant across architectures)."""
    model_name: str = MODEL_NAME
    img_size: int = IMG_SIZE
    pretrained: bool = True
    drop_rate: float = 0.1
    drop_path: float = 0.1
    epochs: int = 30
    batch_size: int = 4
    grad_accum: int = 2           # effective batch = batch_size * grad_accum (= 8)
    lr: float = 2e-5
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    patience: int = 7
    pos_weight_clip: float = 50.0
    num_workers: int = 4
    seed: int = SEED


# Per-model settings used in the paper.
STAGE1 = TrainConfig(epochs=60, lr=1e-5, warmup_epochs=4, patience=8)
STAGE2 = TrainConfig(epochs=30, lr=2e-5, patience=7)
STAGE3 = TrainConfig(epochs=30, lr=2e-5, patience=7)
FLAT = TrainConfig(epochs=30, lr=2e-5, patience=7)
LABEL_CORR = TrainConfig(epochs=25, lr=2e-5, grad_accum=4, patience=6)
