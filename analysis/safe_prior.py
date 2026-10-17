"""
SAFE-Prior test-time procedure and comparison table (CPU, saved predictions only).

Every model is reduced to an image evidence term f and a metadata prior g (both log-odds):
  * trained SAFE-Prior runs:  f = fused logit - g_s(m), g_s read from the frozen buffers in model.pt
  * post-hoc SAFE-Prior:      f = image-only ensemble logit, g = metadata-only logistic regression, combined
                              with weights fitted on the calibration split (no retraining)
Then
  1. bounded prior   g_B = c + B * tanh((g - c) / B), c = median prior on the calibration split.
                     B is chosen on the calibration split to maximise the worst age-band sensitivity at the
                     90%-sensitivity threshold (ties -> larger AUROC). Guarantee: an image with
                     f >= t - c + B is flagged whatever the metadata.
  2. decisions use the bounded score z = f + g_B with the calibration-split threshold;
     probabilities shown to a clinician use z + delta, delta re-estimated on the unlabelled target set by EM
     (Saerens et al. 2002). delta shifts every score equally, so it cannot change decisions or AUROC.

    python -m analysis.safe_prior          # -> reports/v2/safe_main.csv, paper/generated/safe_main.tex
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis import collect as C
from analysis import stats as S
from analysis.report import save

SPLITS = ["calib", "test", "external", "pad"]
BOUNDS = [0.0, 0.25, 0.5, 1.0, 2.0, np.inf]
AGE_BANDS = [-1, 30, 50, 70, 200]


def age_band(split):
    a = C.manifest(split)["age_approx"]
    return pd.cut(a, AGE_BANDS, labels=False).fillna(2).astype(int).values


def trained_parts(model):
    """(f, g) per split for a trained SAFE-Prior model: seed-ensemble evidence and the frozen prior."""
    from src.data.metadata_encoder import MetadataEncoder
    enc = MetadataEncoder().fit(C.manifest("train"))
    seed = C.seeds(model)[0]
    sd = torch.load(C.RUNS / model / f"seed{seed}" / "model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    w_cont, w_site, b = (sd[k].numpy() for k in ("prior_w_cont", "prior_w_site", "prior_b"))
    out = {}
    for s in SPLITS:
        cont, site = enc.transform(C.manifest(s))
        g = cont @ w_cont + w_site[site] + b[0]
        out[s] = (C.logit_matrix(model, s).mean(0) - g, g)
    return out


def posthoc_parts(image_model="baseline"):
    """(f, g) per split: image-only ensemble + metadata LR, weighted on the calibration split."""
    from analysis.report import metadata_only_scores
    f = {s: C.logit_matrix(image_model, s).mean(0) for s in SPLITS}
    g = {s: np.asarray(metadata_only_scores(s), float) for s in SPLITS}
    st = LogisticRegression(C=1e6).fit(np.c_[f["calib"], g["calib"]], C.manifest("calib")["target"])
    a, b = st.coef_[0]
    return {s: (a * f[s] + st.intercept_[0], b * g[s]) for s in SPLITS}


def bounded(g, c, B):
    return np.full_like(g, c) if B == 0 else (g if np.isinf(B) else c + B * np.tanh((g - c) / B))


def worst_band_sens(y, z, thr, band):
    sens = [(z[(band == k) & (y == 1)] >= thr).mean() for k in range(4) if ((band == k) & (y == 1)).sum() >= 5]
    return min(sens)


SAFETY_EPS = 0.05  # pre-specified: metadata may cost the worst age band at most 5 points of sensitivity


def choose_bound(parts):
    """Largest-AUROC bound B on the calibration split among those whose worst age-band sensitivity (at the
    90%-sensitivity threshold, refitted per B) is within SAFETY_EPS of using no metadata (B = 0)."""
    f, g = parts["calib"]
    y, band, c = C.manifest("calib")["target"].values, age_band("calib"), np.median(g)
    scores = []
    for B in BOUNDS:
        z = f + bounded(g, c, B)
        thr = S.threshold_for_sens(y, z, 0.90)
        scores.append((worst_band_sens(y, z, thr, band), roc_auc_score(y, z), B))
    floor = scores[0][0] - SAFETY_EPS
    return max((auc, B) for w, auc, B in scores if w >= floor)[1], c


def em_intercept(z):
    """Saerens EM with fixed likelihood ratios; returns the implied logit shift of the class prior."""
    p0 = S.sigmoid(z)
    pi_src = pi = p0.mean()
    for _ in range(500):
        w1, w0 = pi / pi_src, (1 - pi) / (1 - pi_src)
        q = p0 * w1 / (p0 * w1 + (1 - p0) * w0)
        if abs(q.mean() - pi) < 1e-9:
            break
        pi = q.mean()
    return np.log(pi / (1 - pi)) - np.log(pi_src / (1 - pi_src))


def evaluate(name, z_by_split, adapt=False):
    yc = C.manifest("calib")["target"].values
    thr = S.threshold_for_sens(yc, z_by_split["calib"], 0.90)
    row = {"Method": name}
    for s, lab in [("test", "ISIC19"), ("external", "ISIC20"), ("pad", "PAD")]:
        y, z = C.manifest(s)["target"].values, z_by_split[s]
        m = S.confusion_metrics(y, z, thr)
        p = S.sigmoid(z + (em_intercept(z) if adapt else 0.0))
        row[f"{lab} AUROC"] = f"{roc_auc_score(y, z):.3f}"
        row[f"{lab} sens"] = f"{m['sensitivity']:.3f}"
        row[f"{lab} spec"] = f"{m['specificity']:.3f}"
        row[f"{lab} ECE"] = f"{S.ece(y, p):.3f}"
    man = C.manifest("test")
    y, z, band = man["target"].values, z_by_split["test"], age_band("test")
    row["Young miss"] = f"{1 - (z[(band == 0) & (y == 1)] >= thr).mean():.3f}"
    acral = (man["anatom_site_general_challenge"] == "palms/soles").values & (y == 1)
    row["Acral miss"] = f"{1 - (z[acral] >= thr).mean():.3f}"
    row["Worst age sens"] = f"{worst_band_sens(y, z, thr, band):.3f}"
    return row


def main():
    rows = []
    for bb, p in [("EfficientNet-B2", ""), ("DINOv2 ViT-B/14", "dino_")]:
        out = []
        for m in [p + m for m in ["baseline", "concat_fusion", "film", "field_token", "concat_moddrop", "concat_ogm"]]:
            if C.seeds(m):
                out.append(evaluate(C.LABEL.get(m, m), {s: C.logit_matrix(m, s).mean(0) for s in SPLITS}))
        variants = [("SAFE-Prior post-hoc", posthoc_parts(p + "baseline"))] if C.seeds(p + "baseline") else []
        variants += [(f"SAFE-Prior ({m})", trained_parts(m)) for m in [p + "safe_prior", p + "safe_prior_hsic"]
                     if C.seeds(m)]
        for name, parts in variants:
            out.append(evaluate(name + ", prior removed (f only)", {s: f for s, (f, g) in parts.items()}))
            out.append(evaluate(name + ", unbounded", {s: f + g for s, (f, g) in parts.items()}))
            B, c = choose_bound(parts)
            out.append(evaluate(f"{name}, bounded B={B:g}", {s: f + bounded(g, c, B) for s, (f, g) in parts.items()},
                                adapt=True))
        rows += [{"Backbone": bb, **r} for r in out]
    save("safe_main", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
