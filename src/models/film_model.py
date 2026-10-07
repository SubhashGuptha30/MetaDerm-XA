"""
FiLM (Feature-wise Linear Modulation) conditioning model.

Implements the standard approach for conditioning CNN features on metadata,
as used in MetaBlock (Kharazmi et al. 2021, JBHI) and related works.

Instead of cross-attention, FiLM learns channel-wise affine transforms
(gamma, beta) from metadata and applies them to spatial feature maps:

    output = gamma * features + beta

This is the key competing baseline for field-token cross-attention.

References:
    - Perez et al. (2018). "FiLM: Visual Reasoning with a General
      Conditioning Layer." AAAI.
    - Kharazmi et al. (2021). "MetaBlock: CNN-based Feature Learning
      with Metadata." JBHI.
"""

import timm
import torch
import torch.nn as nn

from src.models.metadata_encoder_model import MetadataEncoderModel


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Takes metadata features and generates per-channel scale (gamma) and
    shift (beta) parameters that modulate the CNN feature map.

    Args:
        meta_dim: dimension of the metadata feature vector.
        num_channels: number of channels in the feature map to modulate.
    """

    def __init__(self, meta_dim: int, num_channels: int) -> None:
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(meta_dim, num_channels),
            nn.Sigmoid(),  # Bound gamma to [0, 1] for stability
        )
        self.beta_net = nn.Linear(meta_dim, num_channels)

        # Initialize gamma close to 1 (identity) and beta close to 0
        nn.init.ones_(self.gamma_net[0].bias)
        nn.init.zeros_(self.beta_net.bias)

    def forward(
        self,
        feature_map: torch.Tensor,  # (B, C, H, W)
        metadata: torch.Tensor,     # (B, meta_dim)
    ) -> torch.Tensor:
        gamma = self.gamma_net(metadata)  # (B, C)
        beta = self.beta_net(metadata)    # (B, C)

        # Reshape for broadcasting: (B, C) → (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * feature_map + beta  # (B, C, H, W)


class FiLMModel(nn.Module):
    """
    FiLM-conditioned CNN model for skin lesion classification.

    Architecture:
        Image → EfficientNet-B2 (spatial features) → FiLM(metadata) → GAP → Classifier

    The metadata modulates the CNN feature map via channel-wise affine
    transforms BEFORE global average pooling, allowing metadata to
    selectively amplify or suppress feature channels.

    Args:
        backbone_name: timm model identifier.
        pretrained: load ImageNet weights.
        continuous_dim: dimension of continuous metadata features.
        num_sites: vocabulary size for site embedding.
        feature_dim: metadata encoder output dimension.
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

        # Image backbone — keep spatial features (no global pool)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # Keep spatial map (B, C, H, W)
        )

        # Infer native channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            self.native_channels = out.shape[1]  # e.g. 1408 for EB2

        # Metadata encoder
        self.meta_encoder = MetadataEncoderModel(
            continuous_dim=continuous_dim,
            num_sites=num_sites,
            output_dim=feature_dim,
            dropout=dropout * 0.5,
        )

        # FiLM layer: modulates backbone's spatial features
        self.film = FiLMLayer(
            meta_dim=feature_dim,
            num_channels=self.native_channels,
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.native_channels, feature_dim),
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
        # Extract spatial CNN features
        feat_map = self.backbone(image)  # (B, C, H, W) e.g. (B, 1408, 7, 7)

        # Encode metadata
        meta_feat = self.meta_encoder(metadata_cont, site_idx)  # (B, feature_dim)

        # Apply FiLM conditioning
        modulated = self.film(feat_map, meta_feat)  # (B, C, H, W)

        # Global pool → classify
        pooled = self.gap(modulated).squeeze(-1).squeeze(-1)  # (B, C)
        return self.classifier(pooled).squeeze(-1)  # (B,)
