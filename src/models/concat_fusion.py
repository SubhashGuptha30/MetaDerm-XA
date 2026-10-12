"""
Late-fusion baseline: pooled image features concatenated with the shared metadata embedding.

Optional shortcut-mitigation baselines (off by default, so the original model is unchanged):
  meta_dropout > 0 : modality dropout (Neverova et al., 2016) - the metadata embedding of a training sample is
                     zeroed with this probability.
  ogm_alpha > 0    : OGM (Peng et al., CVPR 2022) - a linear fusion head lets each modality's contribution be
                     scored; the gradients of the dominant modality's encoder are scaled down by
                     1 - tanh(alpha * (ratio - 1)).
"""

import torch
import torch.nn as nn

from src.models.metadata_encoder_model import MetadataEncoderModel
from src.models.vision_backbone import create_backbone


class ConcatFusionModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b2", pretrained: bool = True,
                 continuous_dim: int = 5, num_sites: int = 7, feature_dim: int = 256,
                 dropout: float = 0.4, meta_dropout: float = 0.0, ogm_alpha: float = 0.0) -> None:
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained, num_classes=0, global_pool="avg")
        img_feat_dim = self.backbone.num_features
        self.meta_encoder = MetadataEncoderModel(continuous_dim=continuous_dim, num_sites=num_sites,
                                                 output_dim=feature_dim, dropout=dropout * 0.5)
        self.meta_dropout, self.ogm_alpha = meta_dropout, ogm_alpha
        if ogm_alpha > 0:  # OGM needs a head that is linear in each modality
            self.img_dim = img_feat_dim
            self.fusion_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(img_feat_dim + feature_dim, 1))
        else:
            self.fusion_head = nn.Sequential(
                nn.Linear(img_feat_dim + feature_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1),
            )
        self._cache = None
        self._coef = (1.0, 1.0)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)  # (B, C)

    def fuse(self, feats, metadata_cont, site_idx) -> torch.Tensor:
        meta = self.meta_encoder(metadata_cont, site_idx)
        if self.training and self.meta_dropout > 0:
            keep = (torch.rand(meta.size(0), 1, device=meta.device) >= self.meta_dropout).to(meta.dtype)
            meta = meta * keep
        if self.training and self.ogm_alpha > 0:
            self._cache = (feats, meta)
        return self.fusion_head(torch.cat([feats, meta], dim=-1)).squeeze(-1)

    def forward(self, image, metadata_cont, site_idx) -> torch.Tensor:
        return self.fuse(self.encode_image(image), metadata_cont, site_idx)

    def extra_loss(self, y: torch.Tensor) -> torch.Tensor:
        """No extra loss; computes the OGM modulation coefficients from this batch (called by train.py)."""
        if self.ogm_alpha > 0 and self._cache is not None:
            with torch.no_grad():
                feats, meta = (t.float() for t in self._cache)
                w = self.fusion_head[1].weight[0].float()
                b = self.fusion_head[1].bias.float() / 2
                sign = 2 * y.float() - 1  # score of the true class for each modality alone
                s_img = torch.sigmoid(sign * (feats @ w[:self.img_dim] + b)).sum()
                s_meta = torch.sigmoid(sign * (meta @ w[self.img_dim:] + b)).sum()
                ratio = (s_img / s_meta.clamp_min(1e-6)).item()
                k = lambda r: 1 - float(torch.tanh(torch.tensor(self.ogm_alpha * (r - 1)))) if r > 1 else 1.0  # noqa: E731
                self._coef = (k(ratio), k(1 / max(ratio, 1e-6)))
            self._cache = None
        return torch.zeros((), device=y.device)

    def after_backward(self) -> None:
        """Scale the encoder gradients of the dominant modality (OGM)."""
        if self.ogm_alpha <= 0:
            return
        for module, c in ((self.backbone, self._coef[0]), (self.meta_encoder, self._coef[1])):
            if c != 1.0:
                for p in module.parameters():
                    if p.grad is not None:
                        p.grad.mul_(c)
