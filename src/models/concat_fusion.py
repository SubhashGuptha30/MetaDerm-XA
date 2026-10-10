"""
Late-fusion baseline: pooled image features concatenated with the shared metadata embedding.
"""

import torch
import torch.nn as nn

from src.models.metadata_encoder_model import MetadataEncoderModel
from src.models.vision_backbone import create_backbone


class ConcatFusionModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b2", pretrained: bool = True,
                 continuous_dim: int = 5, num_sites: int = 7, feature_dim: int = 256,
                 dropout: float = 0.4) -> None:
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained, num_classes=0, global_pool="avg")
        img_feat_dim = self.backbone.num_features
        self.meta_encoder = MetadataEncoderModel(continuous_dim=continuous_dim, num_sites=num_sites,
                                                 output_dim=feature_dim, dropout=dropout * 0.5)
        self.fusion_head = nn.Sequential(
            nn.Linear(img_feat_dim + feature_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)  # (B, C)

    def fuse(self, feats, metadata_cont, site_idx) -> torch.Tensor:
        meta = self.meta_encoder(metadata_cont, site_idx)
        return self.fusion_head(torch.cat([feats, meta], dim=-1)).squeeze(-1)

    def forward(self, image, metadata_cont, site_idx) -> torch.Tensor:
        return self.fuse(self.encode_image(image), metadata_cont, site_idx)
