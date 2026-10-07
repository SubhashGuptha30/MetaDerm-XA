"""
Metrics utilities — Phase 8.
AUROC, recall, precision, F1, bootstrap CIs, DeLong test.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes full binary classification metric suite.

    Args:
        y_true: binary ground truth (0/1).
        y_prob: predicted probabilities (after sigmoid).
        threshold: decision threshold for precision/recall/F1.

    Returns:
        dict with keys: auroc, ap, recall, specificity, precision, f1,
                        brier_score, ece, threshold.
    """
    auroc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calibration metrics
    brier = brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)

    return {
        "auroc": auroc,
        "ap": ap,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "brier_score": brier,
        "ece": ece,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold,
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier score: mean squared error of probability predictions.
    Lower is better. Range [0, 1]. Perfect = 0.
    """
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) — weighted average of per-bin
    |accuracy - confidence| gaps.

    Lower is better. Perfect calibration = 0.
    Uses equal-width binning (standard for ECE reporting).

    Args:
        y_true: binary ground truth (0/1).
        y_prob: predicted probabilities.
        n_bins: number of equal-width bins (default 15, as in Guo et al. 2017).

    Returns:
        ECE as a float in [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        # Include left edge for the first bin
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any scalar metric.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except Exception:
            pass  # skip degenerate samples (all one class)

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    point = metric_fn(y_true, y_prob)
    return float(point), float(lower), float(upper)


def auroc_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    return bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap, ci)


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> Tuple[float, float]:
    """
    DeLong's test for statistical comparison of two AUROC values.
    Returns (z_statistic, p_value).

    Implementation follows: DeLong et al. (1988), Biometrics.
    Simplified closed-form approximation.
    """

    def _placement_values(scores, labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        n_pos, n_neg = len(pos), len(neg)
        V10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        V01 = np.array([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])
        return V10, V01, n_pos, n_neg

    V10_a, V01_a, n1, n2 = _placement_values(y_prob_a, y_true)
    V10_b, V01_b, _, _ = _placement_values(y_prob_b, y_true)

    theta_a = np.mean(V10_a)
    theta_b = np.mean(V10_b)

    # Covariance matrix of [AUROC_a, AUROC_b]
    S10_aa = np.var(V10_a, ddof=1) if n1 > 1 else 0.0
    S10_bb = np.var(V10_b, ddof=1) if n1 > 1 else 0.0
    S10_ab = np.cov(V10_a, V10_b)[0, 1] if n1 > 1 else 0.0

    S01_aa = np.var(V01_a, ddof=1) if n2 > 1 else 0.0
    S01_bb = np.var(V01_b, ddof=1) if n2 > 1 else 0.0
    S01_ab = np.cov(V01_a, V01_b)[0, 1] if n2 > 1 else 0.0

    var_diff = (S10_aa + S10_bb - 2 * S10_ab) / n1 + (S01_aa + S01_bb - 2 * S01_ab) / n2

    if var_diff <= 0:
        log.warning("DeLong variance is non-positive — returning p=1.0")
        return 0.0, 1.0

    z = (theta_a - theta_b) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "youden",
) -> float:
    """
    Find decision threshold that maximizes a metric.
    metric: 'youden' (J = sensitivity + specificity - 1) or 'f1'.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if metric == "youden":
        best_idx = np.argmax(tpr - fpr)
    elif metric == "f1":
        f1s = [
            f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        best_idx = int(np.argmax(f1s))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return float(thresholds[best_idx])
