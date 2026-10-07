"""
Quantum fusion model — Phase 6.
Cross-attention model with the classifier head swapped for a QuantumInspiredBlock.
Everything else is identical to MultiModalModel for fair ablation comparison.
"""

import torch
import torch.nn as nn

from src.models.cross_attention_fusion import CrossAttentionFusion
from src.models.metadata_encoder_model import MetadataEncoderModel
from src.models.quantum_inspired_layer import QuantumInspiredBlock
from src.models.vision_backbone import VisionBackbone


class QuantumFusionModel(nn.Module):
    """
    Identical to MultiModalModel except the ClassifierHead is replaced by
    a QuantumInspiredBlock — only one variable changes for ablation validity.

    Args: same as MultiModalModel.
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

        # ← ONLY change vs MultiModalModel
        self.quantum_head = QuantumInspiredBlock(
            input_dim=feature_dim,
            hidden_dim=128,
            dropout=dropout,
        )

    def forward(
        self,
        image: torch.Tensor,
        metadata_cont: torch.Tensor,
        site_idx: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.vision_backbone(image)
        meta_vec = self.meta_encoder(metadata_cont, site_idx)
        fused = self.cross_attention(tokens, meta_vec)
        pooled = fused.mean(dim=1)                    # (B, D) — mean pool
        return self.quantum_head(pooled)              # (B,)

    @property
    def cross_attn_weights(self):
        return self.cross_attention.last_attn_weights
