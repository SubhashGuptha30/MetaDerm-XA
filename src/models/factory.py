"""Single model factory: cfg['model_type'] → nn.Module. All models share forward(image, meta, site)
and the encode_image()/fuse() split used by src/probes.py."""

import torch.nn as nn

MODEL_TYPES = ("baseline", "baseline_wide", "concat_fusion", "film",
               "cross_attention", "field_token", "field_token_ablated")
IMAGE_ONLY = ("baseline", "baseline_wide")


def build_model(cfg: dict) -> nn.Module:
    t = cfg["model_type"]
    common = dict(backbone_name=cfg.get("backbone", "efficientnet_b2"),
                  pretrained=cfg.get("pretrained", True), dropout=cfg.get("dropout", 0.4))
    if t in IMAGE_ONLY:
        from src.models.baseline import BaselineModel
        return BaselineModel(wide=t == "baseline_wide", **common)
    if t == "concat_fusion":
        from src.models.concat_fusion import ConcatFusionModel
        return ConcatFusionModel(feature_dim=cfg.get("feature_dim", 256), **common)
    if t == "film":
        from src.models.film_model import FiLMModel
        return FiLMModel(feature_dim=cfg.get("feature_dim", 256), **common)
    if t in ("cross_attention", "field_token", "field_token_ablated"):
        from src.models.field_token_model import TokenCrossAttentionModel
        mode = {"cross_attention": "single", "field_token": "field", "field_token_ablated": "ablated"}[t]
        return TokenCrossAttentionModel(feature_dim=cfg.get("feature_dim", 256),
                                        num_attn_heads=cfg.get("num_attn_heads", 8),
                                        freeze_early=cfg.get("freeze_early", False),
                                        meta_mode=mode, num_meta_tokens=cfg.get("num_meta_tokens", 4),
                                        **common)
    raise ValueError(f"Unknown model_type: {t}")
