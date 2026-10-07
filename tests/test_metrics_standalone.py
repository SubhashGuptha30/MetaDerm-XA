"""
Standalone metrics tests — no torch dependency.
Can run without PyTorch installed.
"""

import numpy as np
import pytest

from src.utils.metrics import (
    auroc_bootstrap_ci,
    bootstrap_ci,
    compute_metrics,
    delong_test,
    find_optimal_threshold,
)


class TestMetricsStandalone:
    @pytest.fixture
    def perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
        return y_true, y_prob

    def test_auroc_perfect(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        m = compute_metrics(y_true, y_prob)
        assert m["auroc"] > 0.95

    def test_all_keys_present(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        m = compute_metrics(y_true, y_prob)
        for key in ["auroc", "recall", "precision", "f1", "tp", "fp", "fn", "tn"]:
            assert key in m

    def test_recall_is_one_for_perfect(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        assert m["recall"] == 1.0

    def test_bootstrap_ci_bounds(self):
        # Use a larger balanced sample to avoid degenerate bootstrap draws
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 25 + [1] * 25)
        y_prob = np.concatenate([rng.uniform(0, 0.4, 25), rng.uniform(0.6, 1.0, 25)])
        pt, lo, hi = auroc_bootstrap_ci(y_true, y_prob, n_bootstrap=200)
        assert lo <= pt <= hi, f"CI invalid: [{lo:.3f}, {hi:.3f}] around {pt:.3f}"

    def test_delong_same_model_z_near_zero(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        z, p = delong_test(y_true, y_prob, y_prob)
        assert abs(z) < 1e-6

    def test_optimal_threshold_in_range(self, perfect_prediction):
        y_true, y_prob = perfect_prediction
        t = find_optimal_threshold(y_true, y_prob, metric="youden")
        assert 0.0 <= t <= 1.0

    def test_metrics_values_in_range(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        m = compute_metrics(y_true, y_prob)
        assert 0 <= m["auroc"] <= 1
        assert 0 <= m["recall"] <= 1
        assert 0 <= m["f1"] <= 1
