"""Model definitions.

* ``build_classifier`` - Swin-V2-B with a linear head (Stage 1/2/3 and the flat model).
* ``LabelCorrelationNet`` - Swin-V2-B whose linear head is replaced by 39 learned
  label queries and a two-layer transformer decoder (cross-attention to image
  tokens, self-attention among labels).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .config import TrainConfig


def build_classifier(n_out: int, cfg: TrainConfig) -> nn.Module:
    return timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=n_out,
                             drop_rate=cfg.drop_rate, drop_path_rate=cfg.drop_path)


def _tokens(feat: torch.Tensor, dim: int) -> torch.Tensor:
    """Flatten a backbone feature map to (B, N, C) for NHWC (Swin) or NCHW (CNN)."""
    if feat.dim() == 4:
        if feat.shape[-1] == dim:            # NHWC
            return feat.flatten(1, 2)
        return feat.flatten(2).transpose(1, 2)  # NCHW
    return feat                               # already (B, N, C)


class LabelCorrelationNet(nn.Module):
    def __init__(self, n_labels: int, cfg: TrainConfig, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(cfg.model_name, pretrained=cfg.pretrained,
                                          num_classes=0, global_pool="")
        self.feat_dim = self.backbone.num_features
        self.proj = nn.Linear(self.feat_dim, d_model)
        self.label_queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = self.proj(_tokens(self.backbone.forward_features(x), self.feat_dim))
        q = self.label_queries.unsqueeze(0).expand(x.size(0), -1, -1)
        z = self.decoder(tgt=q, memory=memory)      # labels attend to image AND each other
        return self.head(self.norm(z)).squeeze(-1)  # (B, n_labels) logits


class GeM(nn.Module):
    """Generalised-mean pooling over tokens (Stage-2 ablation run D)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class GeMClassifier(nn.Module):
    def __init__(self, n_out: int, cfg: TrainConfig):
        super().__init__()
        self.backbone = timm.create_model(cfg.model_name, pretrained=cfg.pretrained,
                                          num_classes=0, global_pool="",
                                          drop_path_rate=cfg.drop_path)
        self.feat_dim = self.backbone.num_features
        self.pool = GeM()
        self.fc = nn.Linear(self.feat_dim, n_out)

    def forward(self, x):
        return self.fc(self.pool(_tokens(self.backbone.forward_features(x), self.feat_dim)))


def load_weights(model: nn.Module, ckpt_path, device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()
