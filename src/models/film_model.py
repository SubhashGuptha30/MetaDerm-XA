"""
FiLM conditioning (Perez et al., 2018) of the final CNN feature map, in the spirit of
MetaBlock (Pacheco & Krohling, 2021, IEEE JBHI). gamma is sigmoid-bounded (restricted FiLM).

    feat_map (B,C,H,W) -> gamma(meta) * feat_map + beta(meta) -> GAP -> MLP -> logit
"""

import torch
import torch.nn as nn

from src.models.metadata_encoder_model import MetadataEncoderModel
from src.models.vision_backbone import create_backbone


class FiLMLayer(nn.Module):
    def __init__(self, meta_dim: int, num_channels: int) -> None:
        super().__init__()
        self.gamma_net = nn.Sequential(nn.Linear(meta_dim, num_channels), nn.Sigmoid())
        self.beta_net = nn.Linear(meta_dim, num_channels)
        nn.init.ones_(self.gamma_net[0].bias)  # gamma starts near sigmoid(1)≈0.73
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, feature_map: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(metadata)[..., None, None]
        beta = self.beta_net(metadata)[..., None, None]
        return gamma * feature_map + beta


class FiLMModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b2", pretrained: bool = True,
                 continuous_dim: int = 5, num_sites: int = 7, feature_dim: int = 256,
                 dropout: float = 0.4) -> None:
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained, num_classes=0, global_pool="")
        self.native_channels = self.backbone.num_features
        self.meta_encoder = MetadataEncoderModel(continuous_dim=continuous_dim, num_sites=num_sites,
                                                 output_dim=feature_dim, dropout=dropout * 0.5)
        self.film = FiLMLayer(meta_dim=feature_dim, num_channels=self.native_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.native_channels, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)  # (B, C, H, W)

    def fuse(self, feats, metadata_cont, site_idx) -> torch.Tensor:
        meta = self.meta_encoder(metadata_cont, site_idx)
        pooled = self.film(feats, meta).mean(dim=(2, 3))
        return self.classifier(pooled).squeeze(-1)

    def forward(self, image, metadata_cont, site_idx) -> torch.Tensor:
        return self.fuse(self.encode_image(image), metadata_cont, site_idx)
