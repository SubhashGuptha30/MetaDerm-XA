"""
Image-only baseline model — Phase 4.
EfficientNet-B2 (or any timm backbone) with a simple classification head.
"""

import timm
import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """
    Pretrained CNN backbone + classification head.
    Image-only — no metadata.

    Args:
        backbone_name: timm model identifier.
        pretrained: load ImageNet weights.
        num_classes: 1 for binary (BCEWithLogitsLoss).
        dropout: classifier dropout.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # Infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B,) — raw logit
        """
        feat = self.backbone(x)          # (B, feat_dim)
        return self.classifier(feat).squeeze(-1)  # (B,)
