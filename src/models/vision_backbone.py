"""
Shared vision backbone — Phase 5.
Returns spatial feature tokens (B, N_tokens, C) suitable for cross-attention.
"""

import logging
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def create_backbone(backbone_name: str, pretrained: bool, **kwargs) -> nn.Module:
    """timm.create_model, but loads ImageNet weights from $METADERM_PRETRAINED_DIR/<name>.pth
    when present (Kaggle jobs run without internet)."""
    if backbone_name.startswith("vit"):  # ViTs (e.g. DINOv2, native 518 px) run at our 224 px input
        kwargs.setdefault("img_size", 224)
        if kwargs.get("global_pool") == "avg":  # avg pooling would replace the pretrained final norm; use CLS
            kwargs["global_pool"] = "token"
    local = Path(os.environ.get("METADERM_PRETRAINED_DIR", "pretrained")) / f"{backbone_name}.pth"
    if pretrained and local.exists():
        model = timm.create_model(backbone_name, pretrained=False, **kwargs)
        missing, unexpected = model.load_state_dict(torch.load(local, map_location="cpu"), strict=False)
        assert not missing, f"missing keys loading {local}: {missing[:5]}"
        log.info(f"Loaded local pretrained weights {local} (ignored {len(unexpected)} head keys)")
        return model
    return timm.create_model(backbone_name, pretrained=pretrained, **kwargs)

SUPPORTED_BACKBONES = [
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b4",
    "resnet50",
    "swin_tiny_patch4_window7_224",
    "vit_small_patch16_224",
]


class VisionBackbone(nn.Module):
    """
    Pretrained backbone that outputs spatial feature tokens for cross-attention.

    For CNN-based models (EfficientNet, ResNet): the final spatial feature map
    is reshaped to (B, H*W, C).

    For transformer models (Swin, ViT): patch tokens are returned directly.

    Args:
        backbone_name: timm model name.
        pretrained: whether to load ImageNet pretrained weights.
        freeze_early: if True, freeze all but the last 2 stages.
        feature_dim: output feature dimension (projected to this if needed).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        freeze_early: bool = False,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim

        # Load backbone without classification head
        self.backbone = create_backbone(
            backbone_name,
            pretrained,
            num_classes=0,       # removes classifier head
            global_pool="",      # removes global pooling → keeps spatial map
        )

        # Infer the backbone's native feature channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            # out shape: (1, C, H, W) for CNNs, (1, N, C) for transformers
            if out.dim() == 4:
                native_channels = out.shape[1]
                self._is_cnn = True
            elif out.dim() == 3:
                native_channels = out.shape[2]
                self._is_cnn = False
            else:
                raise ValueError(f"Unexpected backbone output shape: {out.shape}")

        log.info(
            f"Backbone '{backbone_name}': native_channels={native_channels}, "
            f"is_cnn={self._is_cnn}"
        )

        # Project to common feature_dim
        self.proj = nn.Sequential(
            nn.Linear(native_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        if freeze_early:
            self._freeze_early_layers()

    def _freeze_early_layers(self) -> None:
        """Freeze all backbone params except last two named children."""
        children = list(self.backbone.named_children())
        freeze_up_to = max(0, len(children) - 2)
        for i, (name, module) in enumerate(children):
            if i < freeze_up_to:
                for param in module.parameters():
                    param.requires_grad = False
        log.info(f"Frozen first {freeze_up_to}/{len(children)} backbone stages.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            tokens: (B, N_tokens, feature_dim)
        """
        feat = self.backbone(x)  # (B, C, H, W) or (B, N, C)

        if self._is_cnn:
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)

        tokens = self.proj(feat)  # (B, N, feature_dim)
        return tokens
