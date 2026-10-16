"""Loss functions. BCE with per-class ``pos_weight`` is the protocol for every
reported model; focal and asymmetric losses were tried in the Stage-2 ablation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCE(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma, self.pos_weight = gamma, pos_weight

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight,
                                                reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * ce).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss (Ridnik et al., ICCV 2021)."""

    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = targets * torch.log(xs_pos.clamp(min=self.eps)) + \
            (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gp * targets + self.gn * (1 - targets)
        return -(loss * (1 - pt) ** gamma).mean()


def make_criterion(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalBCE(pos_weight=pos_weight)
    if name == "asl":
        return AsymmetricLoss()
    raise ValueError(f"unknown loss {name!r}")
