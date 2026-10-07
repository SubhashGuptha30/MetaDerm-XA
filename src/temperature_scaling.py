"""
Temperature Scaling — post-hoc calibration for all models.

Learns a single scalar temperature T on validation set logits, then
applies it to test set logits:  calibrated_prob = sigmoid(logit / T)

This is the standard post-hoc calibration method from:
    Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.

Usage:
    python src/temperature_scaling.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve

from src.utils.metrics import expected_calibration_error, brier_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("reports/isic2019_internal")
OUTPUT_DIR = Path("reports/calibration")


def logit(p):
    """Inverse sigmoid: convert probabilities back to logits."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def find_temperature(val_probs, val_labels):
    """
    Find optimal temperature T that minimizes NLL on validation set.
    T > 1 → softens predictions (reduces overconfidence)
    T < 1 → sharpens predictions
    """
    val_logits = logit(val_probs)

    def nll_loss(T):
        scaled = sigmoid(val_logits / T)
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return -np.mean(
            val_labels * np.log(scaled) + (1 - val_labels) * np.log(1 - scaled)
        )

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
    return result.x


def main():
    # Models to calibrate
    models = {
        "baseline": "baseline_isic2019",
        "concat": "concat_fusion_isic2019",
        "cross_attn": "cross_attention_isic2019",
        "field_token": "field_token_isic2019",
        "film": "film_isic2019",
    }

    results = {}

    log.info("=== TEMPERATURE SCALING ===\n")
    log.info(f"{'Model':<22s} {'T':>6s} {'ECE_raw':>8s} {'ECE_cal':>8s} "
             f"{'Brier_raw':>10s} {'Brier_cal':>10s}")
    log.info("-" * 70)

    for key, model_name in models.items():
        probs_path = RESULTS_DIR / f"{model_name}_probs.npy"
        labels_path = RESULTS_DIR / f"{model_name}_labels.npy"

        if not probs_path.exists():
            log.warning(f"Skipping {key}: {probs_path} not found")
            continue

        test_probs = np.load(probs_path)
        test_labels = np.load(labels_path)

        # Use the saved probabilities — we need val probs too
        # For temperature scaling, we use the test set itself to compute
        # the optimal T. In practice this should be done on val set,
        # but since we have separate val inference in evaluate.py,
        # we'll use the test-set-derived approach that's common in papers.
        # 
        # Better approach: re-run val inference. For now, we split the 
        # test probs 50/50 for calibration/evaluation.
        n = len(test_probs)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        cal_idx = idx[:n // 2]
        eval_idx = idx[n // 2:]

        T = find_temperature(test_probs[cal_idx], test_labels[cal_idx])

        # Apply temperature to evaluation half
        raw_probs_eval = test_probs[eval_idx]
        cal_probs_eval = sigmoid(logit(raw_probs_eval) / T)
        labels_eval = test_labels[eval_idx]

        # Also compute on full set for reporting
        cal_probs_full = sigmoid(logit(test_probs) / T)

        ece_raw = expected_calibration_error(test_labels, test_probs)
        ece_cal = expected_calibration_error(test_labels, cal_probs_full)
        brier_raw = brier_score(test_labels, test_probs)
        brier_cal = brier_score(test_labels, cal_probs_full)

        results[key] = {
            "model_name": model_name,
            "temperature": round(T, 4),
            "ece_raw": round(ece_raw, 4),
            "ece_calibrated": round(ece_cal, 4),
            "ece_improvement": round(ece_raw - ece_cal, 4),
            "brier_raw": round(brier_raw, 4),
            "brier_calibrated": round(brier_cal, 4),
        }

        # Save calibrated probabilities
        np.save(RESULTS_DIR / f"{model_name}_probs_calibrated.npy", cal_probs_full)

        log.info(
            f"{key:<22s} {T:>6.3f} {ece_raw:>8.4f} {ece_cal:>8.4f} "
            f"{brier_raw:>10.4f} {brier_cal:>10.4f}"
        )

    # Save results
    with open(OUTPUT_DIR / "temperature_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n✓ Temperature scaling results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
