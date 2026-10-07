"""
Full cross-attention multi-modal model — Phase 5. Core architecture.

Pipeline:
    Image → VisionBackbone → spatial tokens (B, N, D)
    Metadata → MetadataEncoderModel → latent vector (B, D)
    CrossAttentionFusion → metadata-conditioned tokens (B, N, D)
    ClassifierHead → logit (B,)
"""

import torch
import torch.nn as nn

from src.models.classifier_head import ClassifierHead
from src.models.cross_attention_fusion import CrossAttentionFusion
from src.models.metadata_encoder_model import MetadataEncoderModel
from src.models.vision_backbone import VisionBackbone


class MultiModalModel(nn.Module):
    """
    Cross-attention multi-modal skin lesion classifier.

    Args:
        backbone_name: timm backbone identifier.
        pretrained: load ImageNet weights.
        freeze_early: freeze early backbone layers.
        feature_dim: shared token dimension.
        num_attn_heads: attention heads in cross-attention block.
        continuous_dim: metadata continuous feature dimension.
        num_sites: site embedding vocabulary size.
        dropout: global dropout rate.
        pool: 'mean' or 'max' pooling in classifier head.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        freeze_early: bool = False,
        feature_dim: int = 256,
        num_attn_heads: int = 8,
        continuous_dim: int = 5,
        num_sites: int = 7,
        dropout: float = 0.3,
        pool: str = "mean",
    ) -> None:
        super().__init__()

        self.vision_backbone = VisionBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_early=freeze_early,
            feature_dim=feature_dim,
        )

        self.meta_encoder = MetadataEncoderModel(
            continuous_dim=continuous_dim,
            num_sites=num_sites,
            output_dim=feature_dim,
            dropout=dropout * 0.5,
        )

        self.cross_attention = CrossAttentionFusion(
            feature_dim=feature_dim,
            num_heads=num_attn_heads,
            dropout=dropout * 0.5,
        )

        self.head = ClassifierHead(
            input_dim=feature_dim,
            hidden_dim=128,
            dropout=dropout,
            pool=pool,
        )

    def forward(
        self,
        image: torch.Tensor,           # (B, 3, H, W)
        metadata_cont: torch.Tensor,   # (B, continuous_dim)
        site_idx: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        """Returns: logits (B,) — raw pre-sigmoid values."""
        tokens = self.vision_backbone(image)                    # (B, N, D)
        meta_vec = self.meta_encoder(metadata_cont, site_idx)  # (B, D)
        fused = self.cross_attention(tokens, meta_vec)          # (B, N, D)
        logits = self.head(fused)                               # (B,)
        return logits

    @property
    def cross_attn_weights(self):
        """Returns last attention weight tensor (B, heads, N, 1) for XAI."""
        return self.cross_attention.last_attn_weights
