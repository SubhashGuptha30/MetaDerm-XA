"""
Ablated field-token model — K-token ablation study.

Uses AblatedMetadataEncoder to produce K∈{1,2,3} metadata tokens,
keeping the same VisionBackbone + CrossAttentionFusion + ClassifierHead.
"""

import torch
import torch.nn as nn

from src.models.classifier_head import ClassifierHead
from src.models.cross_attention_fusion import CrossAttentionFusion
from src.models.ablated_metadata_encoder import AblatedMetadataEncoder
from src.models.vision_backbone import VisionBackbone


class AblatedFieldTokenModel(nn.Module):
    """
    Field-token model with configurable number of metadata tokens (K).

    K=1: single merged metadata token (like cross-attention baseline)
    K=2: [demographic, site]
    K=3: [age+miss, sex, site]
    K=4: full model (use FieldTokenMultiModalModel instead)
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        freeze_early: bool = False,
        feature_dim: int = 256,
        num_attn_heads: int = 8,
        num_meta_tokens: int = 2,
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

        self.meta_encoder = AblatedMetadataEncoder(
            num_tokens=num_meta_tokens,
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
        image: torch.Tensor,
        metadata_cont: torch.Tensor,
        site_idx: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.vision_backbone(image)
        meta_tokens = self.meta_encoder(metadata_cont, site_idx)
        fused = self.cross_attention(tokens, meta_tokens)
        return self.head(fused)
