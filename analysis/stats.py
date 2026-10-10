"""
Offline statistics for MetaDerm-XA v2 (CPU only, operates on saved logits).

Conventions
  * Scores are logits; probabilities = sigmoid(logit).
  * Thresholds and calibration maps are fitted on the `calib` split only and then frozen.
  * ECE: 10 equal-width bins (as stated in the paper).
  * Confidence intervals: percentile bootstrap. For ISIC 2020 resampling is clustered by
    patient_id (group_id); ISIC 2019 test has no patient IDs → image-level.
  * Multi-seed comparisons: two-level bootstrap (resample seeds, then images), so the
    interval reflects both training randomness and test-set sampling.
"""

from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ----------------------------------------------------------------------------- DeLong
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def fast_delong(y: np.ndarray, scores: np.ndarray):
    """Sun & Xu (2014) fast DeLong. scores: (n_models, n). Returns (aucs, covariance)."""
    y = np.asarray(y).astype(bool)
    scores = np.atleast_2d(scores)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(scores[r][np.r_[np.where(y)[0], np.where(~y)[0]]]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y, a, b):
    """Two-sided DeLong test for AUROC(a) − AUROC(b). Returns (diff, z, p)."""
    aucs, cov = fast_delong(y, np.vstack([a, b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        return float(diff), 0.0, 1.0
    z = diff / np.sqrt(var)
    return float(diff), float(z), float(2 * stats.norm.sf(abs(z)))


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------- metrics
def ece(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            tot += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(tot)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def sens_at_spec(y, s, spec: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def spec_at_sens(y, s, sens: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr >= sens
    return float(1 - fpr[ok].min()) if ok.any() else 0.0


def threshold_youden(y, s) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_for_sens(y, s, sens: float) -> float:
    """Largest threshold whose sensitivity on (y, s) is ≥ sens."""
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr >= sens)])


def confusion_metrics(y, s, thr: float) -> Dict[str, float]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s) >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn), "balanced_acc": (sens + spec) / 2,
            "flag_rate": float(pred.mean())}


def discrimination(y, s) -> Dict[str, float]:
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "sens@spec90": sens_at_spec(y, s, 0.90), "sens@spec95": sens_at_spec(y, s, 0.95),
            "spec@sens90": spec_at_sens(y, s, 0.90)}


# ----------------------------------------------------------------------------- calibration
def fit_temperature(y, logit) -> float:
    y, z = np.asarray(y, float), np.asarray(logit, float)

    def nll(t):
        p = np.clip(sigmoid(z / t), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(optimize.minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def fit_platt(y, logit):
    """p = sigmoid(a·logit + b), fitted by maximum likelihood (handles intercept shift)."""
    y, z = np.asarray(y, float), np.asarray(logit, float)

    def nll(ab):
        p = np.clip(sigmoid(ab[0] * z + ab[1]), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    a, b = optimize.minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead").x
    return float(a), float(b)


def calibration_slope_intercept(y, p):
    """Logistic recalibration of logit(p): slope (ideal 1) and calibration-in-the-large intercept (ideal 0)."""
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    a, b = fit_platt(y, z)
    return a, b


# ----------------------------------------------------------------------------- bootstrap
def _resample_idx(rng, n: int, clusters: Optional[np.ndarray]):
    if clusters is None:
        return rng.integers(0, n, n)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[i] for i in pick])


def bootstrap(fn: Callable[[np.ndarray], float], n: int, clusters=None, B: int = 2000, seed: int = 0):
    """fn(idx) → statistic on the resampled indices. Returns array of B replicates (NaNs dropped)."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        idx = _resample_idx(rng, n, clusters)
        try:
            reps.append(fn(idx))
        except ValueError:  # single-class resample
            continue
    return np.asarray(reps, dtype=float)


def ci(reps: np.ndarray, alpha: float = 0.05):
    return float(np.nanpercentile(reps, 100 * alpha / 2)), float(np.nanpercentile(reps, 100 * (1 - alpha / 2)))


def two_level_auc_diff(y, scores_a: np.ndarray, scores_b: np.ndarray, clusters=None, B: int = 2000, seed: int = 0):
    """scores_*: (n_seeds, n). Each replicate resamples seeds (independently per model) and
    images, then compares the mean-over-seeds AUROC. Returns (point diff, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    point = np.mean([roc_auc_score(y, s) for s in scores_a]) - np.mean([roc_auc_score(y, s) for s in scores_b])
    reps = []
    for _ in range(B):
        idx = _resample_idx(rng, len(y), clusters)
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        sa = scores_a[rng.integers(0, len(scores_a), len(scores_a))]
        sb = scores_b[rng.integers(0, len(scores_b), len(scores_b))]
        reps.append(np.mean([roc_auc_score(yi, s[idx]) for s in sa]) -
                    np.mean([roc_auc_score(yi, s[idx]) for s in sb]))
    reps = np.asarray(reps)
    lo, hi = ci(reps)
    p = float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))
    return float(point), lo, hi, p
