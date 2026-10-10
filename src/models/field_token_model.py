"""
Token cross-attention models: image spatial tokens (queries) attend to metadata tokens
(keys/values) inside one post-norm transformer block, then mean-pool → MLP → logit.

    Z  = Proj(Flatten(EffNet(x)))                      (B, N=49, D)
    M  = MetaTokenizer(meta)                           (B, K, D)
    Z' = LN(Z + MHA(Q=Z, K=M, V=M))
    Z''= LN(Z' + FFN(Z'))
    y  = MLP(mean_n Z'')

meta_mode:
    'single'  — shared metadata MLP → 1 token (K=1; softmax is trivially 1)      [Cross-Attn]
    'field'   — one token per field: age, sex, site, missingness (K=4)            [Field-Token]
    'ablated' — field-token tokenizer with merged groups, K ∈ {1,2,3}             [K-ablation]
"""

import torch
import torch.nn as nn

from src.models.ablated_metadata_encoder import AblatedMetadataEncoder
from src.models.classifier_head import ClassifierHead
from src.models.cross_attention_fusion import CrossAttentionFusion
from src.models.field_token_metadata_encoder import FieldTokenMetadataEncoder
from src.models.metadata_encoder_model import MetadataEncoderModel
from src.models.vision_backbone import VisionBackbone


class TokenCrossAttentionModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b2", pretrained: bool = True,
                 freeze_early: bool = False, feature_dim: int = 256, num_attn_heads: int = 8,
                 num_sites: int = 7, dropout: float = 0.4, meta_mode: str = "field",
                 num_meta_tokens: int = 4) -> None:
        super().__init__()
        self.meta_mode = meta_mode
        self.vision_backbone = VisionBackbone(backbone_name=backbone_name, pretrained=pretrained,
                                              freeze_early=freeze_early, feature_dim=feature_dim)
        if meta_mode == "single":
            self.meta_encoder = MetadataEncoderModel(num_sites=num_sites, output_dim=feature_dim,
                                                     dropout=dropout * 0.5)
        elif meta_mode == "field":
            self.meta_encoder = FieldTokenMetadataEncoder(num_sites=num_sites, output_dim=feature_dim,
                                                          dropout=dropout * 0.5)
        elif meta_mode == "ablated":
            self.meta_encoder = AblatedMetadataEncoder(num_tokens=num_meta_tokens, num_sites=num_sites,
                                                       output_dim=feature_dim, dropout=dropout * 0.5)
        else:
            raise ValueError(meta_mode)
        self.cross_attention = CrossAttentionFusion(feature_dim=feature_dim, num_heads=num_attn_heads,
                                                    dropout=dropout * 0.5)
        self.head = ClassifierHead(input_dim=feature_dim, hidden_dim=128, dropout=dropout, pool="mean")

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_backbone(image)  # (B, N, D)

    def meta_tokens(self, metadata_cont, site_idx) -> torch.Tensor:
        return self.meta_encoder(metadata_cont, site_idx)  # (B, D) or (B, K, D)

    def fuse(self, feats, metadata_cont, site_idx, meta_tokens=None) -> torch.Tensor:
        if meta_tokens is None:
            meta_tokens = self.meta_tokens(metadata_cont, site_idx)
        return self.head(self.cross_attention(feats, meta_tokens))

    def forward(self, image, metadata_cont, site_idx) -> torch.Tensor:
        return self.fuse(self.encode_image(image), metadata_cont, site_idx)

    @property
    def cross_attn_weights(self):
        """Last attention tensor (B, heads, N, K)."""
        return self.cross_attention.last_attn_weights
