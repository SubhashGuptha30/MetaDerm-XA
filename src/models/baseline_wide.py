"""
Parameter-matched baseline model — Q1 Strengthening Task 1.

Widened MLP head to match Field-Token's ~8.96M parameters.
This is a vision-only model (no metadata) with additional capacity
to control for the parameter-count confound.
"""

import timm
import torch
import torch.nn as nn


class BaselineWideModel(nn.Module):
    """
    Pretrained CNN backbone + widened classification head.
    Image-only — no metadata. Parameter-matched to Field-Token (~8.96M).

    The wider MLP head (feat_dim→512→256→128→1) adds ~1M parameters
    to match the cross-attention module's overhead.
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

        # Widened classifier to match Field-Token param count (~8.96M)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
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
