"""
Image-only baselines: EfficientNet-B2 + MLP head (narrow) and a parameter-matched wide head.
Metadata arguments are accepted and ignored so every model shares one call signature.
"""

import torch
import torch.nn as nn

from src.models.vision_backbone import create_backbone


class BaselineModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b2", pretrained: bool = True,
                 dropout: float = 0.4, wide: bool = False) -> None:
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        if wide:  # ≈ parameter-matched to field-token (AUDIT: capacity control)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(768, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.25),
                nn.Linear(128, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1),
            )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)  # (B, C)

    def fuse(self, feats, metadata_cont=None, site_idx=None) -> torch.Tensor:
        return self.classifier(feats).squeeze(-1)

    def forward(self, image, metadata_cont=None, site_idx=None) -> torch.Tensor:
        return self.fuse(self.encode_image(image))
