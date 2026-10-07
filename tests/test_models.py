"""
Unit tests for all model architectures — Phase 4–6.
Verifies forward pass shapes without real data.
"""

import pytest
import torch

BATCH = 2
IMG_H = IMG_W = 224
CONT_DIM = 5
NUM_SITES = 7


@pytest.fixture
def dummy_batch():
    image = torch.randn(BATCH, 3, IMG_H, IMG_W)
    meta_cont = torch.randn(BATCH, CONT_DIM)
    site_idx = torch.randint(0, NUM_SITES, (BATCH,))
    return image, meta_cont, site_idx


class TestBaselineModel:
    def test_output_shape(self, dummy_batch):
        from src.models.baseline import BaselineModel
        model = BaselineModel(backbone_name="efficientnet_b0", pretrained=False)
        model.eval()
        image, _, _ = dummy_batch
        with torch.no_grad():
            out = model(image)
        assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"

    def test_output_is_finite(self, dummy_batch):
        from src.models.baseline import BaselineModel
        model = BaselineModel(backbone_name="efficientnet_b0", pretrained=False)
        model.eval()
        image, _, _ = dummy_batch
        with torch.no_grad():
            out = model(image)
        assert torch.isfinite(out).all(), "Non-finite values in baseline output"


class TestVisionBackbone:
    def test_token_shape_cnn(self, dummy_batch):
        from src.models.vision_backbone import VisionBackbone
        model = VisionBackbone(backbone_name="efficientnet_b0", pretrained=False, feature_dim=128)
        model.eval()
        image, _, _ = dummy_batch
        with torch.no_grad():
            tokens = model(image)
        assert tokens.dim() == 3, f"Expected 3D tensor, got {tokens.dim()}"
        assert tokens.shape[0] == BATCH
        assert tokens.shape[2] == 128, f"Expected feature_dim=128, got {tokens.shape[2]}"

    def test_tokens_finite(self, dummy_batch):
        from src.models.vision_backbone import VisionBackbone
        model = VisionBackbone(backbone_name="efficientnet_b0", pretrained=False, feature_dim=64)
        model.eval()
        image, _, _ = dummy_batch
        with torch.no_grad():
            tokens = model(image)
        assert torch.isfinite(tokens).all()


class TestMetadataEncoderModel:
    def test_output_shape(self, dummy_batch):
        from src.models.metadata_encoder_model import MetadataEncoderModel
        model = MetadataEncoderModel(continuous_dim=CONT_DIM, num_sites=NUM_SITES, output_dim=128)
        model.eval()
        _, meta_cont, site_idx = dummy_batch
        with torch.no_grad():
            out = model(meta_cont, site_idx)
        assert out.shape == (BATCH, 128)

    def test_output_finite(self, dummy_batch):
        from src.models.metadata_encoder_model import MetadataEncoderModel
        model = MetadataEncoderModel(continuous_dim=CONT_DIM, num_sites=NUM_SITES, output_dim=64)
        model.eval()
        _, meta_cont, site_idx = dummy_batch
        with torch.no_grad():
            out = model(meta_cont, site_idx)
        assert torch.isfinite(out).all()


class TestCrossAttentionFusion:
    def test_output_same_shape_as_tokens(self):
        from src.models.cross_attention_fusion import CrossAttentionFusion
        model = CrossAttentionFusion(feature_dim=64, num_heads=4)
        model.eval()
        tokens = torch.randn(BATCH, 49, 64)   # e.g. 7x7 spatial map
        meta_vec = torch.randn(BATCH, 64)
        with torch.no_grad():
            out = model(tokens, meta_vec)
        assert out.shape == tokens.shape, f"Shape mismatch: {out.shape} vs {tokens.shape}"

    def test_attention_weights_stored(self):
        from src.models.cross_attention_fusion import CrossAttentionFusion
        model = CrossAttentionFusion(feature_dim=64, num_heads=4)
        model.eval()
        tokens = torch.randn(BATCH, 49, 64)
        meta_vec = torch.randn(BATCH, 64)
        with torch.no_grad():
            _ = model(tokens, meta_vec)
        assert model.last_attn_weights is not None


class TestMultiModalModel:
    def test_forward_shape(self, dummy_batch):
        from src.models.multimodal_model import MultiModalModel
        model = MultiModalModel(backbone_name="efficientnet_b0", pretrained=False, feature_dim=64, num_attn_heads=4)
        model.eval()
        image, meta_cont, site_idx = dummy_batch
        with torch.no_grad():
            out = model(image, meta_cont, site_idx)
        assert out.shape == (BATCH,)

    def test_no_nans_in_output(self, dummy_batch):
        from src.models.multimodal_model import MultiModalModel
        model = MultiModalModel(backbone_name="efficientnet_b0", pretrained=False, feature_dim=64, num_attn_heads=4)
        model.eval()
        image, meta_cont, site_idx = dummy_batch
        with torch.no_grad():
            out = model(image, meta_cont, site_idx)
        assert torch.isfinite(out).all()


class TestConcatFusionModel:
    def test_forward_shape(self, dummy_batch):
        from src.models.concat_fusion import ConcatFusionModel
        model = ConcatFusionModel(backbone_name="efficientnet_b0", pretrained=False, feature_dim=64)
        model.eval()
        image, meta_cont, site_idx = dummy_batch
        with torch.no_grad():
            out = model(image, meta_cont, site_idx)
        assert out.shape == (BATCH,)


class TestQuantumInspiredLayer:
    def test_forward_shape(self):
        from src.models.quantum_inspired_layer import QuantumInspiredLinear
        layer = QuantumInspiredLinear(in_features=64, out_features=32)
        x = torch.randn(BATCH, 64)
        out = layer(x)
        assert out.shape == (BATCH, 32)

    def test_output_finite(self):
        from src.models.quantum_inspired_layer import QuantumInspiredLinear
        layer = QuantumInspiredLinear(in_features=64, out_features=32)
        x = torch.randn(BATCH, 64)
        out = layer(x)
        assert torch.isfinite(out).all()

    def test_gate_bounded(self):
        from src.models.quantum_inspired_layer import QuantumInspiredLinear
        import torch.nn.functional as F
        layer = QuantumInspiredLinear(10, 5)
        gate = torch.sigmoid(layer.gate)
        assert 0 < gate.item() < 1


class TestQuantumFusionModel:
    def test_forward_shape(self, dummy_batch):
        from src.models.quantum_fusion_model import QuantumFusionModel
        model = QuantumFusionModel(backbone_name="efficientnet_b0", pretrained=False, feature_dim=64, num_attn_heads=4)
        model.eval()
        image, meta_cont, site_idx = dummy_batch
        with torch.no_grad():
            out = model(image, meta_cont, site_idx)
        assert out.shape == (BATCH,)

    def test_same_backbone_as_cross_attention(self):
        """Confirms quantum model and cross-attn model share identical backbone class."""
        from src.models.multimodal_model import MultiModalModel
        from src.models.quantum_fusion_model import QuantumFusionModel
        from src.models.vision_backbone import VisionBackbone

        qm = QuantumFusionModel(backbone_name="efficientnet_b0", pretrained=False, feature_dim=64, num_attn_heads=4)
        cm = MultiModalModel(backbone_name="efficientnet_b0", pretrained=False, feature_dim=64, num_attn_heads=4)
        assert type(qm.vision_backbone) == type(cm.vision_backbone)
        assert type(qm.meta_encoder) == type(cm.meta_encoder)
        assert type(qm.cross_attention) == type(cm.cross_attention)
