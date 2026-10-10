"""
Builds every results table of the paper from saved logits (CPU only).

    python -m analysis.report                 # all sections
    python -m analysis.report --only main shift

Outputs: reports/v2/<section>.csv and paper/generated/<section>.tex (included by main.tex).
Each section skips itself (with a message) if its inputs do not exist yet.
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis import collect as C
from analysis import stats as S

OUT = Path("reports/v2")
TEX = Path("paper/generated")
B = 2000


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None or np.isnan(s) else f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def fmt_ci(p, lo, hi, d=3):
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def save(name: str, df: pd.DataFrame, tex_cols=None, caption=None):
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    t = df[tex_cols] if tex_cols else df
    body = t.to_latex(index=False, escape=False, column_format="l" + "c" * (t.shape[1] - 1))
    body = re.sub(r"(?<!\\)%", r"\\%", body)  # escape=False keeps $\pm$ etc.; a bare % would start a comment
    (TEX / f"{name}.tex").write_text(body)
    print(f"--- {name} ---\n{t.to_string(index=False)}\n")


def clusters(split):
    return C.manifest(split)["group_id"].values if split in ("external", "pad") else None


def available(models):
    return [m for m in models if C.seeds(m)]


# ============================================================================ sections
def sec_data():
    info = json.loads((C.SPLITS / "preprocess_info.json").read_text())
    rows = [{"Split": k, "Images": v["n"], "Melanoma": v["n_pos"], "Prevalence (%)": f"{100 * v['prevalence']:.1f}",
             "Groups": v["n_groups"], "Age missing": v["age_missing"], "Sex unknown": v["sex_unknown"],
             "Site unknown": v["site_unknown"]} for k, v in info["splits"].items()]
    save("data", pd.DataFrame(rows))


def metadata_only_scores(split):
    """Logistic regression on the 5 metadata features + one-hot site (fit on train)."""
    from src.data.metadata_encoder import MetadataEncoder
    tr = C.manifest("train")
    enc = MetadataEncoder().fit(tr)

    def X(df):
        cont, site = enc.transform(df)
        return np.hstack([cont, np.eye(len(enc.site_categories))[site]])

    lr = LogisticRegression(max_iter=2000).fit(X(tr), tr.target)
    return lr.decision_function(X(C.manifest(split)))


def sec_main(split):
    y = C.manifest(split)["target"].values
    cl = clusters(split)
    rows = []
    mo = metadata_only_scores(split)
    lo, hi = S.ci(S.bootstrap(lambda i: roc_auc_score(y[i], mo[i]), len(y), cl, B))
    d = S.discrimination(y, mo)
    rows.append({"Model": C.LABEL["metadata_only"], "Seeds": "--", "AUROC": fmt(d["auroc"]),
                 "AUROC (ens.) [95% CI]": fmt_ci(d["auroc"], lo, hi), "AUPRC": fmt(d["auprc"]),
                 "Sens@Spec95": fmt(d["sens@spec95"]), "Spec@Sens90": fmt(d["spec@sens90"])})
    for m in available(C.MAIN_MODELS):
        L = C.logit_matrix(m, split)
        per = pd.DataFrame([S.discrimination(y, s) for s in L])
        ens = L.mean(0)
        lo, hi = S.ci(S.bootstrap(lambda i: roc_auc_score(y[i], ens[i]), len(y), cl, B))
        rows.append({"Model": C.LABEL[m], "Seeds": len(L),
                     "AUROC": fmt(per.auroc.mean(), per.auroc.std(ddof=1)),
                     "AUROC (ens.) [95% CI]": fmt_ci(roc_auc_score(y, ens), lo, hi),
                     "AUPRC": fmt(per.auprc.mean(), per.auprc.std(ddof=1)),
                     "Sens@Spec95": fmt(per["sens@spec95"].mean(), per["sens@spec95"].std(ddof=1)),
                     "Spec@Sens90": fmt(per["spec@sens90"].mean(), per["spec@sens90"].std(ddof=1))})
    save(f"main_{split}", pd.DataFrame(rows))


def sec_pairwise(split):
    y = C.manifest(split)["target"].values
    models = available(C.MAIN_MODELS)
    rows = []
    for a, b in itertools.combinations(models, 2):
        La, Lb = C.logit_matrix(a, split), C.logit_matrix(b, split)
        d, lo, hi, p = S.two_level_auc_diff(y, Lb, La, clusters(split), B=B)
        # per-seed DeLong on matched seeds, reported as count of significant wins
        common = sorted(set(C.seeds(a)) & set(C.seeds(b)))
        dl = [S.delong_test(y, C.logits(b, split)[s], C.logits(a, split)[s]) for s in common]
        rows.append({"A": C.LABEL[a], "B": C.LABEL[b], "dAUROC (B-A)": d, "lo": lo, "hi": hi, "p": p,
                     "DeLong sig. seeds (B>A / B<A)": f"{sum(x[2] < .05 and x[0] > 0 for x in dl)}/"
                                                       f"{sum(x[2] < .05 and x[0] < 0 for x in dl)} of {len(dl)}"})
    df = pd.DataFrame(rows)
    df["p (Holm)"] = S.holm(df.p.values)
    df["$\\Delta$AUROC (B$-$A) [95\\% CI]"] = [fmt_ci(r["dAUROC (B-A)"], r.lo, r.hi) for _, r in df.iterrows()]
    df["p"] = df.p.map(lambda v: f"{v:.3f}")
    df["p (Holm)"] = df["p (Holm)"].map(lambda v: f"{v:.3f}")
    save(f"pairwise_{split}", df, ["A", "B", "$\\Delta$AUROC (B$-$A) [95\\% CI]", "p", "p (Holm)",
                                   "DeLong sig. seeds (B>A / B<A)"])


def sec_operating():
    """Thresholds fitted on calib, applied unchanged to test and external."""
    rows = []
    yc = C.manifest("calib")["target"].values
    for m in available(C.MAIN_MODELS):
        for rule in ["sens90", "youden"]:
            for split in ["test", "external"]:
                y = C.manifest(split)["target"].values
                res = []
                for s in C.seeds(m):
                    zc = C.logits(m, "calib")[s]
                    thr = S.threshold_for_sens(yc, zc, 0.90) if rule == "sens90" else S.threshold_youden(yc, zc)
                    res.append(S.confusion_metrics(y, C.logits(m, split)[s], thr))
                r = pd.DataFrame(res)
                rows.append({"Model": C.LABEL[m], "Threshold rule": rule, "Split": split,
                             "Sensitivity": fmt(r.sensitivity.mean(), r.sensitivity.std(ddof=1)),
                             "Specificity": fmt(r.specificity.mean(), r.specificity.std(ddof=1)),
                             "PPV": fmt(r.ppv.mean(), r.ppv.std(ddof=1)),
                             "Flag rate": fmt(r.flag_rate.mean(), r.flag_rate.std(ddof=1))})
    save("operating_points", pd.DataFrame(rows))


def sec_calibration():
    rows = []
    yc = C.manifest("calib")["target"].values
    for m in available(C.MAIN_MODELS):
        for split in ["test", "external"]:
            y = C.manifest(split)["target"].values
            acc = {k: [] for k in ["ece_raw", "ece_temp", "ece_platt", "brier_raw", "brier_temp", "brier_platt",
                                   "T", "slope", "intercept"]}
            for s in C.seeds(m):
                zc, z = C.logits(m, "calib")[s], C.logits(m, split)[s]
                T = S.fit_temperature(yc, zc)
                a, b = S.fit_platt(yc, zc)
                for tag, p in [("raw", S.sigmoid(z)), ("temp", S.sigmoid(z / T)), ("platt", S.sigmoid(a * z + b))]:
                    acc[f"ece_{tag}"].append(S.ece(y, p))
                    acc[f"brier_{tag}"].append(S.brier(y, p))
                sl, ic = S.calibration_slope_intercept(y, S.sigmoid(z))
                acc["T"].append(T)
                acc["slope"].append(sl)
                acc["intercept"].append(ic)
            r = {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan) for k, v in acc.items()}
            prev = y.mean()
            rows.append({"Model": C.LABEL[m], "Split": split,
                         "ECE raw": fmt(*r["ece_raw"]), "ECE temp.": fmt(*r["ece_temp"]),
                         "ECE Platt": fmt(*r["ece_platt"]), "Brier raw": fmt(*r["brier_raw"]),
                         "Brier Platt": fmt(*r["brier_platt"]), "Brier ref.": fmt(prev * (1 - prev)),
                         "T": fmt(*r["T"], d=2), "Cal. slope": fmt(*r["slope"], d=2),
                         "Cal. intercept": fmt(*r["intercept"], d=2)})
    save("calibration", pd.DataFrame(rows))


def sec_shift():
    """Internal→external AUROC drop and reliance on metadata (withheld / shuffled metadata)."""
    rows = []
    for m in available(C.MAIN_MODELS):
        r = {"Model": C.LABEL[m]}
        for split, tag in [("test", "int"), ("external", "ext")]:
            y = C.manifest(split)["target"].values
            auc = np.array([roc_auc_score(y, z) for z in C.logit_matrix(m, split)])
            r[f"AUROC {tag}"] = fmt(auc.mean(), auc.std(ddof=1))
            r[f"_{tag}"] = auc
            if m not in ("baseline", "baseline_wide"):
                wh, sh = [], []
                for s in C.seeds(m):
                    pr = C.probes(m, s, split)
                    if pr is None:
                        continue
                    wh.append(roc_auc_score(y, pr["withheld"]))
                    sh.append(roc_auc_score(y, pr["shuffled"]))
                if wh:
                    r[f"AUROC withheld {tag}"] = fmt(np.mean(wh), np.std(wh, ddof=1) if len(wh) > 1 else np.nan)
                    r[f"AUROC shuffled {tag}"] = fmt(np.mean(sh), np.std(sh, ddof=1) if len(sh) > 1 else np.nan)
        drop = 100 * (r["_int"].mean() - r["_ext"].mean()) / r["_int"].mean()
        r["Rel. drop (%)"] = f"{drop:.1f}"
        rows.append(r)
    df = pd.DataFrame(rows).drop(columns=["_int", "_ext"])
    save("shift", df.fillna("--"))


def _monotone_frac(age_logits, tol=1e-4):
    return float((np.diff(age_logits, axis=1) >= -tol).all(1).mean())


def sec_sensitivity():
    rows = []
    y = C.manifest("test")["target"].values
    for m in available(["concat_fusion", "cross_attention", "film", "field_token"]):
        acc = {k: [] for k in ["mono_mel", "mono_all", "step_mel", "range_mel", "dp_age_mel",
                               "site_range_mel", "sex_fm_mel"]}
        for s in C.seeds(m):
            pr = C.probes(m, s, "test")
            if pr is None or "age" not in pr:
                continue
            mel = y == 1
            age = pr["age"]
            pa = S.sigmoid(age[mel])
            acc["mono_mel"].append(_monotone_frac(age[mel]))
            acc["mono_all"].append(_monotone_frac(age))
            acc["step_mel"].append(float((np.diff(age[mel], axis=1) > 0).mean()))
            acc["range_mel"].append(float((pa.max(1) - pa.min(1)).mean()))
            acc["dp_age_mel"].append(float((pa[:, -1] - pa[:, 0]).mean()))
            ps = S.sigmoid(pr["site"][mel]).mean(0)
            acc["site_range_mel"].append(float(ps.max() - ps.min()))
            px = S.sigmoid(pr["sex"][mel]).mean(0)
            acc["sex_fm_mel"].append(float(px[1] - px[0]))
        if not acc["mono_mel"]:
            continue
        g = lambda k, d=3: fmt(np.mean(acc[k]), np.std(acc[k], ddof=1) if len(acc[k]) > 1 else np.nan, d)  # noqa
        rows.append({"Model": C.LABEL[m], "Monotone cases (mel.)": g("mono_mel"),
                     "Monotone cases (all)": g("mono_all"), "Increasing steps (mel.)": g("step_mel"),
                     "Mean range $p$ (mel.)": g("range_mel"), "$\\Delta p$ age 10$\\to$85": g("dp_age_mel"),
                     "Site range $\\bar p$": g("site_range_mel"), "$\\bar p$ F$-$M": g("sex_fm_mel")})
    if rows:
        save("sensitivity", pd.DataFrame(rows))


def sec_faithfulness():
    rows = []
    for m in available(["field_token", "field_token_k2", "field_token_k3"]):
        for k in (5, 10):
            stats_ = []
            for s in C.seeds(m):
                pr = C.probes(m, s, "test")
                if pr is None or f"faith_top{k}" not in pr:
                    continue
                d_top = np.abs(pr[f"faith_top{k}"] - pr["orig"])
                d_rand = np.abs(pr[f"faith_rand{k}"] - pr["orig"])
                d_bot = np.abs(pr[f"faith_bottom{k}"] - pr["orig"])
                w = sps.wilcoxon(d_top, d_rand, alternative="two-sided")
                stats_.append((d_top.mean(), d_rand.mean(), d_bot.mean(), (d_top > d_rand).mean(), w.pvalue))
            if stats_:
                a = np.array(stats_)
                rows.append({"Model": C.LABEL[m], "k": k, "|dlogit| top": fmt(a[:, 0].mean(), a[:, 0].std(ddof=1)),
                             "|dlogit| random": fmt(a[:, 1].mean(), a[:, 1].std(ddof=1)),
                             "|dlogit| bottom": fmt(a[:, 2].mean(), a[:, 2].std(ddof=1)),
                             "frac top>random": fmt(a[:, 3].mean(), a[:, 3].std(ddof=1)),
                             "seeds p<0.05": f"{int((a[:, 4] < 0.05).sum())}/{len(a)}"})
    if rows:
        save("faithfulness", pd.DataFrame(rows))


def sec_ablation():
    y = C.manifest("test")["target"].values
    ye = C.manifest("external")["target"].values
    rows = []
    for m in available(C.ABLATION_MODELS):
        a = np.array([roc_auc_score(y, z) for z in C.logit_matrix(m, "test")])
        e = np.array([roc_auc_score(ye, z) for z in C.logit_matrix(m, "external")])
        rows.append({"Model": C.LABEL[m], "Seeds": len(a), "AUROC test": fmt(a.mean(), a.std(ddof=1)),
                     "AUROC external": fmt(e.mean(), e.std(ddof=1))})
    if rows:
        save("ablation", pd.DataFrame(rows))


def sec_unk():
    man = C.manifest("test")
    known = (man["UNK"] == 0).values
    y = man["target"].values
    rows = []
    for m in available(C.MAIN_MODELS):
        L = C.logit_matrix(m, "test")
        a = np.array([roc_auc_score(y, z) for z in L])
        k = np.array([roc_auc_score(y[known], z[known]) for z in L])
        rows.append({"Model": C.LABEL[m], f"AUROC all (n={len(y)})": fmt(a.mean(), a.std(ddof=1)),
                     f"AUROC known-label (n={known.sum()})": fmt(k.mean(), k.std(ddof=1)),
                     "Delta": fmt((k - a).mean(), (k - a).std(ddof=1))})
    save("unk", pd.DataFrame(rows))


def sec_subgroups():
    man = C.manifest("test")
    y = man["target"].values
    groups = {
        "Sex": man["sex"].values,
        "Age": pd.cut(man["age_approx"], [-1, 30, 50, 70, 200], labels=["<=30", "31-50", "51-70", ">70"]).astype(str).values,
        "Site": man["anatom_site_general_challenge"].values,
    }
    rows = []
    for m in available(C.MAIN_MODELS):
        ens = C.logit_matrix(m, "test").mean(0)
        for gname, g in groups.items():
            for val in sorted(set(g)):
                mask = g == val
                if mask.sum() < 50 or y[mask].min() == y[mask].max() or y[mask].sum() < 10:
                    continue
                reps = S.bootstrap(lambda i: roc_auc_score(y[mask][i], ens[mask][i]), int(mask.sum()), None, 500)
                lo, hi = S.ci(reps)
                rows.append({"Model": C.LABEL[m], "Group": gname, "Value": val, "n": int(mask.sum()),
                             "n_mel": int(y[mask].sum()), "AUROC [95% CI]": fmt_ci(roc_auc_score(y[mask], ens[mask]), lo, hi)})
    save("subgroups", pd.DataFrame(rows))


def sec_pitfalls(models=("baseline", "film", "field_token")):
    """Conference paper: the same predictions evaluated under a naive vs a correct protocol."""
    rows = []
    for m in available(models):
        for split in ("test", "external"):
            # 1. duplicates kept vs removed
            C.INCLUDE_DUPLICATES = True
            y_all, L_all = C.manifest(split)["target"].values, C.logit_matrix(m, split)
            C.INCLUDE_DUPLICATES = False
            y, L = C.manifest(split)["target"].values, C.logit_matrix(m, split)
            a_all = np.array([roc_auc_score(y_all, z) for z in L_all])
            a = np.array([roc_auc_score(y, z) for z in L])
            rows.append({"Model": C.LABEL[m], "Pitfall": "duplicates kept", "Split": split,
                         "Naive": fmt(a_all.mean()), "Correct": fmt(a.mean())})
            # 2. best seed vs seed mean
            rows.append({"Model": C.LABEL[m], "Pitfall": "best of seeds (AUROC)", "Split": split,
                         "Naive": fmt(a.max()), "Correct": fmt(a.mean(), a.std(ddof=1))})
            # 3. threshold for 90% sensitivity and temperature fitted on the evaluated split vs on calib
            yc = C.manifest("calib")["target"].values
            sens_n, sens_c, ece_n, ece_c = [], [], [], []
            for s in C.seeds(m):
                z, zc = C.logits(m, split)[s], C.logits(m, "calib")[s]
                sens_n.append(S.confusion_metrics(y, z, S.threshold_for_sens(y, z, 0.90))["sensitivity"])
                sens_c.append(S.confusion_metrics(y, z, S.threshold_for_sens(yc, zc, 0.90))["sensitivity"])
                ece_n.append(S.ece(y, S.sigmoid(z / S.fit_temperature(y, z))))
                ece_c.append(S.ece(y, S.sigmoid(z / S.fit_temperature(yc, zc))))
            rows.append({"Model": C.LABEL[m], "Pitfall": "threshold fitted on test (sens. at 90% target)",
                         "Split": split, "Naive": fmt(np.mean(sens_n)), "Correct": fmt(np.mean(sens_c), np.std(sens_c))})
            rows.append({"Model": C.LABEL[m], "Pitfall": "temperature fitted on test (ECE)", "Split": split,
                         "Naive": fmt(np.mean(ece_n)), "Correct": fmt(np.mean(ece_c), np.std(ece_c))})
    # 4. single-seed DeLong vs two-level bootstrap, image-only vs each fusion model
    for m in available([x for x in models if x != "baseline"]):
        if not C.seeds("baseline"):
            break
        for split in ("test", "external"):
            y = C.manifest(split)["target"].values
            common = sorted(set(C.seeds(m)) & set(C.seeds("baseline")))
            ps = [S.delong_test(y, C.logits(m, split)[s], C.logits("baseline", split)[s])[2] for s in common]
            _, lo, hi, p = S.two_level_auc_diff(y, C.logit_matrix(m, split), C.logit_matrix("baseline", split),
                                                clusters(split), B=B)
            rows.append({"Model": f"{C.LABEL[m]} vs image-only", "Pitfall": "single-seed DeLong (seeds p<0.05)",
                         "Split": split, "Naive": f"{sum(p_ < .05 for p_ in ps)}/{len(ps)}",
                         "Correct": f"p={p:.3f} [{lo:+.3f}, {hi:+.3f}]"})
    if rows:
        save("pitfalls", pd.DataFrame(rows))


def sec_transfer(models=("baseline", "film", "field_token"), budgets=(25, 50, 100, 200, 500, 1000),
                 reps=200, target_sens=0.90):
    """Conference paper: does a calib-fitted 90%-sensitivity operating point transfer to a new domain, and how
    many labelled target images restore it? Recalibration sample = random patients from the target set;
    evaluation = the remaining target images. Uses the seed-ensemble logit."""
    yc = C.manifest("calib")["target"].values
    rows = []
    for m in available(models):
        zc = C.logit_matrix(m, "calib").mean(0)
        thr_c = S.threshold_for_sens(yc, zc, target_sens)
        a_c, b_c = S.fit_platt(yc, zc)
        for split in ("test", "external", "pad"):
            man = C.manifest(split)
            y, z, g = man["target"].values, C.logit_matrix(m, split).mean(0), man["group_id"].values
            cm = S.confusion_metrics(y, z, thr_c)
            rows.append({"Model": C.LABEL[m], "Target": split, "n labelled": 0, "n mel. (mean)": 0,
                         "Sensitivity": fmt(cm["sensitivity"]), "Specificity": fmt(cm["specificity"]),
                         "|sens-0.90|": fmt(abs(cm["sensitivity"] - target_sens)),
                         "ECE (Platt)": fmt(S.ece(y, S.sigmoid(a_c * z + b_c)))})
            groups = np.unique(g)
            rng = np.random.default_rng(0)
            for n in budgets:
                if n >= len(y) // 2:
                    continue
                res = []
                for _ in range(reps):
                    pick = np.isin(g, rng.permutation(groups)[:max(1, int(len(groups) * n / len(y)))])
                    ev = ~pick
                    if y[pick].sum() < 2 or y[ev].sum() < 2:  # cannot fit a sensitivity threshold
                        continue
                    thr = S.threshold_for_sens(y[pick], z[pick], target_sens)
                    a, b = S.fit_platt(y[pick], z[pick])
                    c2 = S.confusion_metrics(y[ev], z[ev], thr)
                    res.append((pick.sum(), y[pick].sum(), c2["sensitivity"], c2["specificity"],
                                S.ece(y[ev], S.sigmoid(a * z[ev] + b))))
                if not res:
                    continue
                r = np.array(res)
                rows.append({"Model": C.LABEL[m], "Target": split, "n labelled": int(r[:, 0].mean()),
                             "n mel. (mean)": f"{r[:, 1].mean():.1f}",
                             "Sensitivity": fmt(r[:, 2].mean(), r[:, 2].std(ddof=1)),
                             "Specificity": fmt(r[:, 3].mean(), r[:, 3].std(ddof=1)),
                             "|sens-0.90|": fmt(np.abs(r[:, 2] - target_sens).mean()),
                             "ECE (Platt)": fmt(r[:, 4].mean(), r[:, 4].std(ddof=1)),
                             "usable reps": f"{len(r)}/{reps}"})
    if rows:
        save("transfer", pd.DataFrame(rows).fillna("--"))


def sec_lr():
    """Stage-1 learning-rate selection (validation AUROC only). Returns {model: best_lr}."""
    rows = []
    for p in sorted(C.RUNS.glob("tune_*_lr*/seed42/summary.json")):
        s = json.loads(p.read_text())
        model, lr = s["model_name"][5:].rsplit("_lr", 1)
        rows.append({"model": model, "lr": float(lr), "val_auroc": s["best_val_auroc"],
                     "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
                     "minutes": round(s["train_sec"] / 60, 1)})
    if not rows:
        print("[skip lr] no tuning runs")
        return {}
    df = pd.DataFrame(rows)
    # Rule: a learning rate that diverged (non-finite logits) in any attempt is not eligible.
    # concat_fusion @1.35e-3 diverged in the first attempt (job tune_lr2) and trained in the rerun (tune_lr3).
    diverged = {("concat_fusion", 1.35e-3)}
    df["eligible"] = [(m, l) not in diverged for m, l in zip(df.model, df.lr)]
    df["note"] = ["diverged in 1 of 2 attempts" if not e else "" for e in df.eligible]
    best = df[df.eligible].loc[lambda d: d.groupby("model").val_auroc.idxmax()].set_index("model").lr.to_dict()
    df["selected"] = [("yes" if best[m] == l else "") for m, l in zip(df.model, df.lr)]
    df["Model"] = df.model.map(C.LABEL)
    out = df[["Model", "lr", "val_auroc", "best_epoch", "epochs_run", "minutes", "selected", "note"]].copy()
    out["val_auroc"] = out.val_auroc.map(lambda v: f"{v:.4f}")
    save("lr_selection", out.rename(columns={"lr": "Learning rate", "val_auroc": "Val. AUROC",
                                             "best_epoch": "Best epoch", "epochs_run": "Epochs run",
                                             "minutes": "Train min", "selected": "Selected"}))
    return best


def sec_training():
    """Training cost and early-stopping behaviour per model (from runs_v2/*/seed*/summary.json)."""
    rows = []
    for m in available(C.MAIN_MODELS + C.ABLATION_MODELS[:3]):
        r = pd.DataFrame([json.loads((C.RUNS / m / f"seed{s}" / "summary.json").read_text()) for s in C.seeds(m)])
        rows.append({"Model": C.LABEL[m], "Parameters (M)": f"{r.n_params.iloc[0] / 1e6:.2f}",
                     "LR": f"{r.config.iloc[0]['lr']:.2e}",
                     "Best epoch": fmt(r.best_epoch.mean(), r.best_epoch.std(ddof=1), 1),
                     "Epochs run": fmt(r.epochs_run.mean(), r.epochs_run.std(ddof=1), 1),
                     "Train time (min)": fmt(r.train_sec.mean() / 60, r.train_sec.std(ddof=1) / 60, 0),
                     "Best val AUROC": fmt(r.best_val_auroc.mean(), r.best_val_auroc.std(ddof=1))})
    save("training", pd.DataFrame(rows))


DX = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def sec_errors(models=("baseline", "film", "field_token")):
    """Where the seed ensembles err on ISIC 2019 test at the calib-fitted 90%-sensitivity threshold:
    flag rate per true diagnosis (MEL row = sensitivity) and miss rate of melanomas per subgroup."""
    man = C.manifest("test")
    y, yc = man["target"].values, C.manifest("calib")["target"].values
    dx = man[DX].values.argmax(1)
    age = pd.cut(man["age_approx"], [-1, 30, 50, 70, 200], labels=["<=30", "31-50", "51-70", ">70"]).astype(str).values
    groups = {"Site": man["anatom_site_general_challenge"].values, "Age": age, "Sex": man["sex"].values}
    flags = {}
    for m in available(list(models)):
        thr = S.threshold_for_sens(yc, C.logit_matrix(m, "calib").mean(0), 0.90)
        flags[C.LABEL[m]] = C.logit_matrix(m, "test").mean(0) >= thr
    rows = [{"Diagnosis": d, "n": int((dx == k).sum()),
             **{f"Flagged: {name}": f"{f[dx == k].mean():.3f}" for name, f in flags.items()}}
            for k, d in enumerate(DX)]
    save("errors_dx", pd.DataFrame(rows))
    rows = []
    for gname, g in groups.items():
        for val in sorted(set(g)):
            mel = (g == val) & (y == 1)
            if mel.sum() < 20:
                continue
            rows.append({"Group": gname, "Value": val, "n mel.": int(mel.sum()),
                         **{f"Missed: {name}": f"{1 - f[mel].mean():.3f}" for name, f in flags.items()}})
    save("errors_missed", pd.DataFrame(rows))


def sec_transfer_tables():
    """Compact conference tables from reports/v2/transfer.csv (no recomputation): zero-label transfer of all
    models, and the label-budget curve of the image-only model on the two external targets."""
    t = pd.read_csv(OUT / "transfer.csv")
    names = {"test": "ISIC 2019 test", "external": "ISIC 2020", "pad": "PAD-UFES-20"}
    z = t[t["n labelled"] == 0]
    rows = [{"Model": m, **{f"{names[k]} {c[:4]}.": g[g.Target == k][c].iloc[0] for k in names for c in ("Sensitivity", "Specificity")}}
            for m, g in z.groupby("Model", sort=False)]
    save("transfer_zero", pd.DataFrame(rows))
    b = t[(t.Model == "Image-only") & (t.Target != "test")].copy()
    b["Sensitivity"] = b["Sensitivity"].astype(str)
    rows = []
    for (target, g) in b.groupby("Target", sort=False):
        for _, r in g.iterrows():
            rows.append({"Target": names[target], "n labelled": r["n labelled"], "melanomas (mean)": r["n mel. (mean)"],
                         "Sensitivity": r["Sensitivity"], "Specificity": r["Specificity"], "Usable draws": r["usable reps"]})
    save("transfer_budget", pd.DataFrame(rows))


SECTIONS = {
    "lr": sec_lr,
    "pitfalls": sec_pitfalls,
    "transfer": sec_transfer,
    "data": sec_data,
    "main": lambda: (sec_main("test"), sec_main("external"), sec_main("pad")),
    "pairwise": lambda: (sec_pairwise("test"), sec_pairwise("external"), sec_pairwise("pad")),
    "operating": sec_operating,
    "calibration": sec_calibration,
    "shift": sec_shift,
    "sensitivity": sec_sensitivity,
    "faithfulness": sec_faithfulness,
    "ablation": sec_ablation,
    "unk": sec_unk,
    "subgroups": sec_subgroups,
    "training": sec_training,
    "errors": sec_errors,
    "transfer_tables": sec_transfer_tables,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SECTIONS))
    a = ap.parse_args()
    for name in a.only:
        try:
            SECTIONS[name]()
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"[skip {name}] {type(e).__name__}: {e}")
