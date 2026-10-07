"""
Calibration & DeLong pairwise comparison — Tasks 2 & 4.

Loads saved probability arrays from all ISIC 2019 models and computes:
  - ECE, Brier score for each model
  - DeLong pairwise tests between all model pairs
  - Reliability diagrams (overlay all models)

Usage:
    python src/calibration_analysis.py
"""

import json
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.utils.metrics import (
    brier_score,
    expected_calibration_error,
    delong_test,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- Configuration ---
RESULTS_DIR = Path("reports/isic2019_internal")
OUTPUT_DIR = Path("reports/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model registry: (display_name, probs_filename, labels_filename)
MODELS = {
    "baseline": {
        "display": "Vision-Only Baseline",
        "probs": "baseline_isic2019_probs.npy",
        "labels": "baseline_isic2019_labels.npy",
    },
    "concat": {
        "display": "Concat Fusion",
        "probs": "concat_fusion_isic2019_probs.npy",
        "labels": "concat_fusion_isic2019_labels.npy",
    },
    "cross_attn": {
        "display": "Cross-Attn (1-token)",
        "probs": "cross_attention_isic2019_probs.npy",
        "labels": "cross_attention_isic2019_labels.npy",
    },
    "field_token": {
        "display": "Field-Token (4-token)",
        "probs": "field_token_isic2019_probs.npy",
        "labels": "field_token_isic2019_labels.npy",
    },
    "film": {
        "display": "FiLM (MetaBlock)",
        "probs": "film_isic2019_probs.npy",
        "labels": "film_isic2019_labels.npy",
    },
}

# Colors for consistency
COLORS = {
    "baseline": "#8B8B8B",
    "concat": "#4F83CC",
    "cross_attn": "#E05C5C",
    "field_token": "#5CAE73",
    "film": "#D4A574",
}


def load_model_data():
    """Load all saved probability arrays."""
    data = {}
    for key, info in MODELS.items():
        probs_path = RESULTS_DIR / info["probs"]
        labels_path = RESULTS_DIR / info["labels"]

        if not probs_path.exists():
            log.warning(f"Missing: {probs_path} — skipping {key}")
            continue
        if not labels_path.exists():
            log.warning(f"Missing: {labels_path} — skipping {key}")
            continue

        probs = np.load(probs_path)
        labels = np.load(labels_path)
        data[key] = {"probs": probs, "labels": labels, "display": info["display"]}
        log.info(f"Loaded {key}: {len(probs)} samples")

    return data


def compute_calibration(data):
    """Compute ECE and Brier score for each model."""
    results = {}
    log.info("\n=== CALIBRATION METRICS ===")
    log.info(f"{'Model':<25s} {'ECE':>8s} {'Brier':>8s}")
    log.info("-" * 45)

    for key, d in data.items():
        ece = expected_calibration_error(d["labels"], d["probs"], n_bins=15)
        brier = brier_score(d["labels"], d["probs"])
        results[key] = {"ece": ece, "brier": brier, "display": d["display"]}
        log.info(f"{d['display']:<25s} {ece:>8.4f} {brier:>8.4f}")

    return results


def run_delong_pairwise(data):
    """Run DeLong test between all model pairs."""
    keys = list(data.keys())
    results = []

    log.info("\n=== DELONG PAIRWISE TESTS ===")
    log.info(f"{'Model A':<22s} vs {'Model B':<22s} {'z':>8s} {'p-value':>10s} {'Sig?':>6s}")
    log.info("-" * 75)

    for a, b in combinations(keys, 2):
        # Ensure same labels
        assert np.array_equal(data[a]["labels"], data[b]["labels"]), \
            f"Label mismatch: {a} vs {b}"

        z, p = delong_test(data[a]["labels"], data[a]["probs"], data[b]["probs"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        results.append({
            "model_a": data[a]["display"],
            "model_b": data[b]["display"],
            "z": round(z, 4),
            "p_value": round(p, 6),
            "significant": sig,
        })
        log.info(
            f"{data[a]['display']:<22s} vs {data[b]['display']:<22s} "
            f"{z:>8.3f} {p:>10.6f} {sig:>6s}"
        )

    return results


def plot_reliability_diagrams(data, save_path):
    """Overlay reliability diagrams for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: individual reliability curves
    ax = axes[0]
    for key, d in data.items():
        prob_true, prob_pred = calibration_curve(
            d["labels"], d["probs"], n_bins=10, strategy="uniform"
        )
        ax.plot(
            prob_pred, prob_true,
            "o-", color=COLORS.get(key, "#333"),
            label=d["display"], lw=2, ms=6
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Reliability Diagram (Calibration)", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Right: bar chart of ECE and Brier
    ax2 = axes[1]
    names = [data[k]["display"] for k in data]
    ece_vals = [expected_calibration_error(data[k]["labels"], data[k]["probs"]) for k in data]
    brier_vals = [brier_score(data[k]["labels"], data[k]["probs"]) for k in data]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax2.bar(x - width/2, ece_vals, width, label="ECE", color="#E05C5C", alpha=0.8)
    bars2 = ax2.bar(x + width/2, brier_vals, width, label="Brier Score", color="#4F83CC", alpha=0.8)

    # Value labels
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Score (lower = better)", fontsize=12)
    ax2.set_title("Calibration Metrics Comparison", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved reliability diagram to {save_path}")
    plt.close(fig)


def main():
    log.info("Loading model predictions...")
    data = load_model_data()

    if len(data) == 0:
        log.error("No model predictions found! Run evaluate.py first.")
        return

    # --- Calibration ---
    cal_results = compute_calibration(data)

    # --- DeLong pairwise ---
    delong_results = run_delong_pairwise(data)

    # --- Reliability diagrams ---
    plot_reliability_diagrams(data, OUTPUT_DIR / "reliability_diagrams.png")

    # --- Save results ---
    output = {
        "calibration": {k: {"ece": v["ece"], "brier": v["brier"]} for k, v in cal_results.items()},
        "delong_pairwise": delong_results,
    }
    with open(OUTPUT_DIR / "calibration_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\n✓ All results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
