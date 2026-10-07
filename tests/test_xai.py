"""
Unit tests for XAI components — Phase 7.
Tests counterfactual engine and metric utilities without real model weights.
"""

import numpy as np
import pytest
import torch

from src.data.metadata_encoder import MetadataEncoder
from src.utils.metrics import (
    auroc_bootstrap_ci,
    compute_metrics,
    delong_test,
    find_optimal_threshold,
)
from src.xai.counterfactual_engine import sweep_age, sweep_site, sweep_sex


# ─── Metrics Tests ───────────────────────────────────────────────────────────

class TestMetrics:
    @pytest.fixture
    def perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
        return y_true, y_prob

    def test_auroc_perfect(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        m = compute_metrics(y_true, y_prob)
        assert m["auroc"] > 0.95, f"Expected AUROC>0.95, got {m['auroc']}"

    def test_all_keys_present(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        m = compute_metrics(y_true, y_prob)
        for key in ["auroc", "recall", "precision", "f1", "tp", "fp", "fn", "tn"]:
            assert key in m, f"Missing key: {key}"

    def test_recall_is_one_for_perfect(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m["recall"] == 1.0

    def test_bootstrap_ci_bounds(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 25 + [1] * 25)
        y_prob = np.concatenate([rng.uniform(0, 0.4, 25), rng.uniform(0.6, 1.0, 25)])
        pt, lo, hi = auroc_bootstrap_ci(y_true, y_prob, n_bootstrap=200)
        assert lo <= pt <= hi, f"CI bounds invalid: [{lo}, {hi}] around {pt}"

    def test_delong_same_model_p_near_1(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        z, p = delong_test(y_true, y_prob, y_prob)
        assert abs(z) < 1e-6, f"Expected z≈0 for identical models, got {z}"

    def test_optimal_threshold_range(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        t = find_optimal_threshold(y_true, y_prob, metric="youden")
        assert 0.0 <= t <= 1.0, f"Threshold out of range: {t}"

    def test_metrics_values_in_range(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        m = compute_metrics(y_true, y_prob)
        assert 0 <= m["auroc"] <= 1
        assert 0 <= m["recall"] <= 1
        assert 0 <= m["precision"] <= 1
        assert 0 <= m["f1"] <= 1


# ─── Counterfactual Engine Tests ─────────────────────────────────────────────

class FakeModel(torch.nn.Module):
    """Fake model that returns logit proportional to age_norm (index 0 of cont)."""

    def forward(self, image, metadata_cont=None, site_idx=None):
        # Simple: logit = 2 * age_norm (more age = higher risk)
        return metadata_cont[:, 0] * 2.0


@pytest.fixture
def fake_setup():
    encoder = MetadataEncoder()
    encoder.age_mean = 50.0
    encoder.age_std = 15.0
    encoder._fitted = True

    model = FakeModel()
    image = torch.zeros(1, 3, 224, 224)
    base_cont = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    site_idx = 0
    device = torch.device("cpu")

    return model, image, base_cont, site_idx, encoder, device


class TestCounterfactualEngine:
    def test_age_sweep_length(self, fake_setup):
        model, image, base_cont, site_idx, encoder, device = fake_setup
        result = sweep_age(model, image, base_cont, site_idx, encoder, device,
                           model_type="cross_attention", age_range=(20, 80), step=10)
        expected = list(range(20, 81, 10))
        assert result["ages"] == expected
        assert len(result["probs"]) == len(expected)

    def test_age_sweep_monotonic_for_linear_model(self, fake_setup):
        """FakeModel is linear in age, so probs should be monotonically non-decreasing."""
        model, image, base_cont, site_idx, encoder, device = fake_setup
        result = sweep_age(model, image, base_cont, site_idx, encoder, device,
                           age_range=(10, 90), step=10)
        probs = result["probs"]
        # Sigmoid of linear: should be strictly increasing
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 1e-6, (
                f"Non-monotonic age sensitivity at index {i}: {probs[i]:.4f} > {probs[i+1]:.4f}"
            )

    def test_site_sweep_length(self, fake_setup):
        model, image, base_cont, site_idx, encoder, device = fake_setup
        result = sweep_site(model, image, base_cont, encoder, device)
        assert len(result["sites"]) == encoder.num_sites
        assert len(result["probs"]) == encoder.num_sites

    def test_sex_sweep_length(self, fake_setup):
        model, image, base_cont, site_idx, encoder, device = fake_setup
        result = sweep_sex(model, image, base_cont, site_idx, encoder, device)
        assert len(result["sexes"]) == 3
        assert len(result["probs"]) == 3

    def test_probs_in_range(self, fake_setup):
        model, image, base_cont, site_idx, encoder, device = fake_setup
        result = sweep_age(model, image, base_cont, site_idx, encoder, device, age_range=(10, 90), step=20)
        for p in result["probs"]:
            assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"
