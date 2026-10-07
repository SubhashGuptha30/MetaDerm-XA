"""
Grad-CAM — Phase 7.
Generates gradient-weighted class activation maps for CNN backbones.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM for the vision backbone's final convolutional layer.

    Works with any model that has a `vision_backbone.backbone` attribute
    (MultiModalModel, QuantumFusionModel) or a bare `backbone` (BaselineModel).

    Args:
        model: trained PyTorch model.
        target_layer_name: dot-separated attribute path to the target conv layer,
            e.g. 'vision_backbone.backbone.blocks.6' or 'backbone.conv_head'.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _get_layer(self) -> torch.nn.Module:
        """Resolves dot-separated attribute path."""
        obj = self.model
        for attr in self.target_layer_name.split("."):
            obj = getattr(obj, attr)
        return obj

    def _register_hooks(self) -> None:
        layer = self._get_layer()

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()

    def generate(
        self,
        image: torch.Tensor,          # (1, 3, H, W)
        metadata_cont: Optional[torch.Tensor] = None,
        site_idx: Optional[torch.Tensor] = None,
        model_type: str = "cross_attention",
    ) -> np.ndarray:
        """
        Returns a Grad-CAM heatmap as a numpy array of shape (H, W), values in [0, 1].
        """
        self.model.eval()

        image = image.requires_grad_(True)

        # Forward pass
        if model_type == "baseline":
            logits = self.model(image)
        else:
            logits = self.model(image, metadata_cont, site_idx)

        # Backward on the positive class logit
        self.model.zero_grad()
        logits.backward(torch.ones_like(logits))

        # Grad-CAM computation
        grads = self._gradients   # (1, C, H, W)
        acts = self._activations  # (1, C, H, W)

        if grads is None or acts is None:
            log.warning("No gradients captured — check target_layer_name.")
            return np.zeros((224, 224))

        # Global average pool over spatial dims
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam.astype(np.float32)


def overlay_cam_on_image(
    image_np: np.ndarray,  # (H, W, 3) uint8
    cam: np.ndarray,       # (H', W') float [0, 1] — may differ from image size
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blends Grad-CAM heatmap over original image. Returns (H, W, 3) uint8."""
    import matplotlib.cm as cm

    # Resize CAM to match original image if dimensions differ
    h, w = image_np.shape[:2]
    if cam.shape[0] != h or cam.shape[1] != w:
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    else:
        cam_resized = cam

    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam_resized)[:, :, :3]  # (H, W, 3) float [0, 1]
    heatmap = (heatmap * 255).astype(np.uint8)

    blended = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return blended


def get_default_target_layer(model_type: str, backbone_name: str) -> str:
    """Returns a sensible default target layer name for common backbones."""
    if "efficientnet" in backbone_name:
        if model_type == "baseline":
            return "backbone.conv_head"
        else:
            return "vision_backbone.backbone.conv_head"
    elif "resnet" in backbone_name:
        if model_type == "baseline":
            return "backbone.layer4"
        else:
            return "vision_backbone.backbone.layer4"
    elif "swin" in backbone_name or "vit" in backbone_name:
        log.warning(
            "Grad-CAM on transformer models is approximate — "
            "consider using attention rollout instead."
        )
        return "vision_backbone.backbone.layers.3"
    else:
        raise ValueError(f"Unknown backbone for default target layer: {backbone_name}")
