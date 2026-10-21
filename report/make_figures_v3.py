"""Figures for the SAFE-Prior / DINOv2 additions to the report.

    .venv/Scripts/python report/make_figures_v3.py      # -> report/figures/{safe_prior,backbones,young_misses,bound_sweep,reanchor}.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import OUT, ROOT, _boxes

R = ROOT / "reports" / "v2"
EFF, DINO = "#4c72b0", "#dd8452"


def safe_prior():
    fig, ax = plt.subplots(figsize=(10, 6))
    B, G, O, P, Rd = "#dbe9f6", "#dff2de", "#fde9d4", "#ece2f4", "#f9dcdc"
    boxes = {
        "img": (0.2, 8.4, 2.6, 1.1, "Dermoscopic image\n224 x 224", B),
        "bb": (3.6, 8.4, 2.8, 1.1, "Backbone\n(EfficientNet-B2 / DINOv2)", O),
        "head": (7.2, 8.4, 2.6, 1.1, "Projection (128) +\nlinear head -> f(x)", O),
        "meta": (0.2, 6.0, 2.6, 1.1, "Age, sex, site,\nage-missing flag", G),
        "lr": (3.6, 6.0, 2.8, 1.1, "Logistic regression g(m)\nfitted on train, FROZEN", P),
        "sum": (7.2, 6.0, 2.6, 1.1, "Training: z = f(x) + g(m)\nloss on z (product of experts)", Rd),
        "bound": (3.6, 3.6, 2.8, 1.1, "Deployment: bounded prior\ng_B = c + B tanh((g-c)/B)", P),
        "sel": (0.2, 3.6, 2.6, 1.1, "B chosen on calibration split\n(worst age-band constraint)", G),
        "out": (7.2, 3.6, 2.6, 1.1, "Score f(x) + g_B(m)\n-> melanoma decision", B),
    }
    arrows = [("img", "bb"), ("bb", "head"), ("head", "sum"), ("meta", "lr"), ("lr", "sum"),
              ("lr", "bound"), ("sel", "bound"), ("bound", "out")]
    _boxes(ax, boxes, arrows)
    ax.set_ylim(3.0, 10)
    fig.savefig(OUT / "safe_prior.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _main():
    d = pd.read_csv(R / "safe_main.csv")
    return d[d.Backbone.isin(["EfficientNet-B2", "DINOv2 ViT-B/14"])]


def backbones():
    d = _main()
    pick = {"EfficientNet-B2": "Image-only", "DINOv2 ViT-B/14": "dino_baseline_hl"}
    sets = ["ISIC19 AUROC", "ISIC20 AUROC", "PAD AUROC"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(sets))
    for i, (bb, m) in enumerate(pick.items()):
        v = d[(d.Backbone == bb) & (d.Method == m)][sets].astype(float).iloc[0].values
        bars = ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=[EFF, DINO][i], label=f"{bb} (image-only)")
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, ["ISIC 2019 test", "ISIC 2020 (external)", "PAD-UFES-20 (external)"])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("AUROC (five-seed ensemble)")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / "backbones.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


YOUNG = [  # (backbone, method label in safe_main.csv, display name)
    ("EfficientNet-B2", "Image-only", "Image-only"), ("EfficientNet-B2", "Concat", "Concat"),
    ("EfficientNet-B2", "FiLM", "FiLM"), ("EfficientNet-B2", "Field-token", "Field-token"),
    ("EfficientNet-B2", "concat_ogm", "Gradient mod."), ("EfficientNet-B2", "concat_moddrop", "Modality dropout"),
    ("EfficientNet-B2", "SAFE-Prior (safe_prior), bounded B=0", "SAFE-Prior"),
    ("DINOv2 ViT-B/14", "dino_baseline_hl", "Image-only"), ("DINOv2 ViT-B/14", "dino_concat_fusion_hl", "Concat"),
    ("DINOv2 ViT-B/14", "dino_film_hl", "FiLM"), ("DINOv2 ViT-B/14", "dino_concat_moddrop_hl", "Modality dropout"),
    ("DINOv2 ViT-B/14", "SAFE-Prior (dino_safe_prior_hl), bounded B=0.5", "SAFE-Prior"),
]


def young_misses():
    d = _main()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, bb, col in zip(axes, ["EfficientNet-B2", "DINOv2 ViT-B/14"], [EFF, DINO]):
        rows = [(lab, round(float(d[(d.Backbone == b) & (d.Method == m)]["Young miss"].iloc[0]) * 54))
                for b, m, lab in YOUNG if b == bb]
        names, vals = zip(*rows)
        colors = ["#2ca02c" if n == "SAFE-Prior" else col for n in names]
        bars = ax.barh(names, vals, color=colors)
        ax.bar_label(bars, fontsize=8, padding=2)
        ax.invert_yaxis()
        ax.set_title(bb, fontsize=10)
        ax.set_xlabel("melanomas missed, patients aged <= 30 (of 54)")
    fig.tight_layout()
    fig.savefig(OUT / "young_misses.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def bound_sweep():
    d = pd.read_csv(R / "safe_bound_sweep.csv")
    d = d[d.Backbone.isin(["EfficientNet-B2", "DINOv2 ViT-B/14"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    xs = np.arange(6)
    for bb, col in [("EfficientNet-B2", EFF), ("DINOv2 ViT-B/14", DINO)]:
        g = d[d.Backbone == bb]
        axes[0].plot(xs, g["PAD AUROC"], "o-", color=col, label=bb)
        axes[1].plot(xs, g["Calib worst age sens"], "o-", color=col, label=bb)
        sel = np.where(g.Selected.fillna("").values == "yes")[0]
        for ax, c in zip(axes, ["PAD AUROC", "Calib worst age sens"]):
            ax.plot(xs[sel], g[c].values[sel], "s", ms=11, mfc="none", mec="k")
    for ax, t in zip(axes, ["PAD-UFES-20 AUROC", "worst age-band sensitivity (calibration split)"]):
        ax.set_xticks(xs, ["0", "0.25", "0.5", "1", "2", "inf"])
        ax.set_xlabel("bound B on the metadata prior (square = selected)")
        ax.set_ylabel(t)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "bound_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def reanchor():
    d = pd.read_csv(R / "threshold_transfer.csv")
    d = d[(d.Method == "Source threshold") & (d.k.astype(str) == "5")]
    groups = [("Image-only", "ISIC20", "EffNet\nISIC 2020"), ("Image-only", "PAD", "EffNet\nPAD"),
              ("dino_baseline_hl", "ISIC20", "DINOv2\nISIC 2020"), ("dino_baseline_hl", "PAD", "DINOv2\nPAD")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(groups))
    for i, (sp, lab, col) in enumerate([("raw", "threshold applied unchanged", "#999999"),
                                         ("quantile", "label-free re-anchoring (ours)", "#2ca02c")]):
        v = [d[(d.Model == m) & (d.Target == t) & (d.Space == sp)].Sens.iloc[0] for m, t, _ in groups]
        bars = ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=col, label=lab)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.axhline(0.90, color="k", ls=":", lw=1)
    ax.text(3.6, 0.905, "target 0.90", fontsize=8, ha="right")
    ax.set_xticks(x, [g[2] for g in groups])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("sensitivity at the new site")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "reanchor.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def paper_figures():
    """Single-column (3.4 in) vector figures for the IEEE paper."""
    out = ROOT / "paper" / "conference" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6.5})

    # Fig. 1: SAFE-Prior, vertical flow
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    box = dict(boxstyle="round,pad=0.25", ec="#333", lw=0.7)
    items = {"img": (0.22, 0.88, "Image x", "#dbe9f6"), "bb": (0.22, 0.66, "Backbone\n(EffNet-B2 / DINOv2)", "#fde9d4"),
             "f": (0.22, 0.44, "Image logit f(x)", "#fde9d4"), "m": (0.78, 0.88, "Age, sex, site m", "#dff2de"),
             "g": (0.78, 0.66, "Logistic regression g(m)\nfitted on train, frozen", "#ece2f4"),
             "gb": (0.78, 0.44, "Bounded prior $g_B=c+B\\tanh((g-c)/B)$\nB chosen on calibration data", "#ece2f4"),
             "z": (0.5, 0.14, "Training: loss on f(x)+g(m)\nDeployment: score f(x)+g$_B$(m)", "#f9dcdc")}
    for k, (xx, yy, t, c) in items.items():
        ax.text(xx, yy, t, ha="center", va="center", bbox=dict(box, fc=c))
    for a, b in [("img", "bb"), ("bb", "f"), ("m", "g"), ("g", "gb"), ("f", "z"), ("gb", "z")]:
        ax.annotate("", xy=items[b][:2], xytext=items[a][:2], arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333",
                    shrinkA=14, shrinkB=14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(out / "safe_prior.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig. 2: young-patient misses, grouped by model
    d = _main()
    look = {(b, lab): round(float(d[(d.Backbone == b) & (d.Method == m)]["Young miss"].iloc[0]) * 54)
            for b, m, lab in YOUNG}
    models = ["Image-only", "Concat", "FiLM", "Modality dropout", "SAFE-Prior"]
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    xs = np.arange(len(models))
    for i, (bb, col, lab) in enumerate([("EfficientNet-B2", EFF, "EfficientNet-B2"), ("DINOv2 ViT-B/14", DINO, "DINOv2")]):
        v = [look[(bb, m)] for m in models]
        bars = ax.bar(xs + (i - 0.5) * 0.38, v, 0.38, color=col, label=lab)
        ax.bar_label(bars, fontsize=6, padding=1)
    ax.set_xticks(xs, ["Image-\nonly", "Concat", "FiLM", "Modality\ndropout", "SAFE-Prior\n(ours)"])
    ax.set_ylabel("missed (of 54)")
    ax.set_ylim(0, 32)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out / "young_misses.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    paper_figures()
    for f in (safe_prior, backbones, young_misses, bound_sweep, reanchor):
        f()
        print("wrote", f.__name__)
