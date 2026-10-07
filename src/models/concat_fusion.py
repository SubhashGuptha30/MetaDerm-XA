"""
Naive concatenation fusion baseline — Phase 5.
Flattens pooled image features + metadata vector → MLP → logit.
This is the "what everyone else does" comparison point.
"""

import timm
import torch
import torch.nn as nn

from src.models.metadata_encoder_model import MetadataEncoderModel


class ConcatFusionModel(nn.Module):
    """
    Naive fusion: global-pooled image features concatenated with metadata MLP output.

    Args:
        backbone_name: timm model identifier.
        pretrained: load ImageNet weights.
        continuous_dim: dimension of continuous metadata features.
        num_sites: vocabulary size for site embedding.
        feature_dim: shared latent dimension.
        dropout: dropout rate.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        continuous_dim: int = 5,
        num_sites: int = 7,
        feature_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        # Image backbone (global pool → feature vector)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            img_feat_dim = self.backbone(dummy).shape[-1]

        # Metadata encoder
        self.meta_encoder = MetadataEncoderModel(
            continuous_dim=continuous_dim,
            num_sites=num_sites,
            output_dim=feature_dim,
            dropout=dropout * 0.5,
        )

        # Fusion MLP
        fused_dim = img_feat_dim + feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        image: torch.Tensor,           # (B, 3, H, W)
        metadata_cont: torch.Tensor,   # (B, continuous_dim)
        site_idx: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        img_feat = self.backbone(image)                       # (B, img_feat_dim)
        meta_feat = self.meta_encoder(metadata_cont, site_idx)  # (B, feature_dim)
        fused = torch.cat([img_feat, meta_feat], dim=-1)     # (B, fused_dim)
        return self.fusion_head(fused).squeeze(-1)            # (B,)
