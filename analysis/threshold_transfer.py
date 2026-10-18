"""
Few-label operating-point transfer to a new site (CPU, saved predictions only).

The source threshold (90% sensitivity on the calibration split) loses sensitivity at new sites because melanoma
scores shift down while benign scores barely move. With k labelled melanomas from the target site we move the
threshold by the shift of target positives relative to source positives:

  * HL shift        t = t_src + median_{i,j}(tp_i - sp_j)                       (Hodges-Lehmann point estimate)
  * HL lower bound  t = t_src + one-sided 90% Mann-Whitney lower bound on the shift (conservative)
  * loc-scale       t = median(tp) + b (t_src - median(sp)), b = shrunk ratio of spreads (K0 pseudo-labels)
  * loc-scale cert. 10th percentile of the loc-scale threshold over a bootstrap of the k labels
Every method is run on raw logits and on per-site quantile-normalised scores (label-free; needed when the
backbone compresses scores, e.g. DINOv2). Baselines: source threshold, minimum of the k labelled scores, and the oracle threshold (all target labels).
Metrics are computed on the target images not used as labels, over R random draws of the k melanomas.

    python -m analysis.threshold_transfer            # -> reports/v2/threshold_transfer.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis import collect as C
from analysis import stats as S
from analysis.report import save

TARGET, LEVEL, K0, R, B_BOOT = 0.90, 0.90, 5, 500, 200
KS = [2, 3, 5, 10, 20]
MODELS = ["baseline", "concat_moddrop", "safe_prior", "dino_baseline_hl", "dino_concat_moddrop_hl", "dino_safe_prior_hl"]


def hl_shift(tp, sp, level=LEVEL):
    """Hodges-Lehmann shift of tp vs sp and its one-sided lower confidence bound (Mann-Whitney inversion,
    normal approximation for the critical rank)."""
    d = np.sort((np.asarray(tp)[:, None] - np.asarray(sp)[None, :]).ravel())
    m, n = len(tp), len(sp)
    c = int(np.floor(m * n / 2 - norm.ppf(level) * np.sqrt(m * n * (m + n + 1) / 12)))
    return float(np.median(d)), float(d[max(c, 0)])


def loc_scale_thr(tp, t_src, med_s, sd_s, k0=K0):
    """Target median plus the source offset rescaled by the (shrunk towards 1) ratio of positive-score spreads."""
    k = len(tp)
    b = np.exp(k / (k + k0) * np.log(max(np.std(tp), 1e-3) / sd_s)) if k > 1 else 1.0
    return float(np.median(tp) + b * (t_src - med_s))


def quantile_normalise(x):
    """Probit of each score's mid-rank within its own site's unlabelled scores. Benign images dominate every site
    (prevalence ~2%), so this re-expresses scores relative to the site's benign distribution; label-free."""
    r = np.sort(x)
    p = (np.searchsorted(r, x, "left") + np.searchsorted(r, x, "right")) / 2 / len(r)
    return norm.ppf(np.clip(p, 0.5 / len(r), 1 - 0.5 / len(r)))


def run_model(model, rng, space="raw"):
    z = {s: C.logit_matrix(model, s).mean(0) for s in ["calib", "external", "pad"]}
    if space == "quantile":
        z = {s: quantile_normalise(v) for s, v in z.items()}
    y = {s: C.manifest(s)["target"].values for s in z}
    sp = z["calib"][y["calib"] == 1]
    t_src = S.threshold_for_sens(y["calib"], z["calib"], TARGET)
    med_s, sd_s = np.median(sp), sp.std()
    rows = []
    for s, lab in [("external", "ISIC20"), ("pad", "PAD")]:
        pos = np.where(y[s] == 1)[0]
        orc = S.confusion_metrics(y[s], z[s], S.threshold_for_sens(y[s], z[s], TARGET))
        rows.append({"Model": C.LABEL.get(model, model), "Space": space, "Target": lab, "k": "all", "Method": "Oracle",
                     "Sens": orc["sensitivity"], "Spec": orc["specificity"], "P(sens>=0.9)": np.nan})
        for k in KS:
            res = {}
            for _ in range(R):
                idx = rng.choice(pos, k, replace=False)
                keep = np.ones(len(y[s]), bool)
                keep[idx] = False
                tp = z[s][idx]
                hl, lo = hl_shift(tp, sp)
                boot = [loc_scale_thr(rng.choice(tp, k), t_src, med_s, sd_s) for _ in range(B_BOOT)]
                thr = {"Source threshold": t_src, "Min of k labels": tp.min(), "HL shift": t_src + hl,
                       "HL lower bound": t_src + lo, "Loc-scale": loc_scale_thr(tp, t_src, med_s, sd_s),
                       "Loc-scale certified": float(np.quantile(boot, 1 - LEVEL))}
                for name, t in thr.items():
                    m = S.confusion_metrics(y[s][keep], z[s][keep], t)
                    res.setdefault(name, []).append((m["sensitivity"], m["specificity"]))
            for name, v in res.items():
                v = np.asarray(v)
                rows.append({"Model": C.LABEL.get(model, model), "Space": space, "Target": lab, "k": k, "Method": name,
                             "Sens": v[:, 0].mean(), "Spec": v[:, 1].mean(),
                             "P(sens>=0.9)": (v[:, 0] >= TARGET).mean()})
    return rows


def main():
    rng = np.random.default_rng(0)
    rows = [r for m in MODELS if C.seeds(m) for sp in ("raw", "quantile") for r in run_model(m, rng, sp)]
    save("threshold_transfer", pd.DataFrame(rows).round(3))


if __name__ == "__main__":
    main()
