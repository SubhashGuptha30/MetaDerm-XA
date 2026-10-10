"""Result figures from saved predictions/probes (CPU).

    .venv/Scripts/python -m analysis.figures     # -> report/figures/*.png and paper/figures/*.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from analysis import collect as C
from analysis import stats as S

REPORT = Path("report/figures")
PAPER = Path("paper/figures")
COLORS = {"baseline": "#7f7f7f", "baseline_wide": "#bcbcbc", "concat_fusion": "#1f77b4",
          "cross_attention": "#d62728", "film": "#ff7f0e", "field_token": "#2ca02c"}
SPLIT_LABEL = {"test": "ISIC 2019 test", "external": "ISIC 2020", "pad": "PAD-UFES-20"}


def _save(fig, name):
    REPORT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(PAPER / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def roc():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, split in zip(axes, SPLIT_LABEL):
        y = C.manifest(split)["target"].values
        for m in [m for m in C.MAIN_MODELS if C.seeds(m)]:
            z = C.logit_matrix(m, split).mean(0)
            fpr, tpr, _ = roc_curve(y, z)
            ax.plot(fpr, tpr, color=COLORS[m], lw=1.8,
                    label=f"{C.LABEL[m]} ({S.discrimination(y, z)['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("sensitivity")
        ax.legend(fontsize=7, loc="lower right")
    _save(fig, "roc_curves")


def age_sweep():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = C.manifest("test")["target"].values
    for m in ["concat_fusion", "cross_attention", "film", "field_token"]:
        curves, grid = [], None
        for s in C.seeds(m):
            pr = C.probes(m, s, "test")
            if pr is not None and "age" in pr:
                curves.append(S.sigmoid(pr["age"][y == 1]).mean(0))
                grid = pr["age_grid"]
        if curves:
            c = np.array(curves)
            ax.plot(grid, c.mean(0), color=COLORS[m], lw=2, label=C.LABEL[m])
            ax.fill_between(grid, c.min(0), c.max(0), color=COLORS[m], alpha=0.15)
    ax.set_xlabel("age set to (years)")
    ax.set_ylabel("mean P(melanoma), melanoma cases")
    ax.set_title("Image-fixed age sweep (ISIC 2019 test; band = range over seeds)")
    ax.legend()
    _save(fig, "age_sweep")


def attention_maps(n_cases=3):
    pr = C.probes("field_token", 42, "test") if C.seeds("field_token") else None
    if pr is None or "attn" not in pr:
        print("[skip attention_maps]")
        return
    import pandas as pd
    from PIL import Image
    y = pr["target"]
    p = S.sigmoid(pr["orig"])
    idx = np.where(y == 1)[0][np.argsort(-p[y == 1])[:n_cases]]  # most confident melanomas
    man = pd.read_csv(C.SPLITS / "test.csv").set_index("image_name")
    root = Path("data/kaggle_upload/metaderm-images")
    fig, axes = plt.subplots(n_cases, 5, figsize=(13, 2.8 * n_cases))
    for r, i in enumerate(idx):
        name = pr["image_name"][i]
        axes[r, 0].imshow(Image.open(root / man.loc[name, "img_dir"] / f"{name}.jpg"))
        axes[r, 0].set_title(f"{name}\nP={p[i]:.2f}", fontsize=8)
        a = pr["attn"][i].astype(float).reshape(7, 7, -1)
        for k, field in enumerate(["age", "sex", "site", "missing"]):
            im = axes[r, k + 1].imshow(a[:, :, k], vmin=0, vmax=1, cmap="viridis")
            axes[r, k + 1].set_title(field, fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.6, label="head-averaged attention")
    _save(fig, "attention_maps")


def reliability():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, split in zip(axes, SPLIT_LABEL):
        y = C.manifest(split)["target"].values
        for m in [m for m in C.MAIN_MODELS if C.seeds(m)]:
            p = S.sigmoid(C.logit_matrix(m, split).mean(0))
            bins = np.minimum((p * 10).astype(int), 9)
            keep = [b for b in range(10) if (bins == b).sum() >= 20]
            ax.plot([p[bins == b].mean() for b in keep], [y[bins == b].mean() for b in keep],
                    "o-", color=COLORS[m], ms=3, lw=1.4, label=C.LABEL[m])
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"{SPLIT_LABEL[split]} (raw probabilities)")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed melanoma fraction")
        ax.legend(fontsize=7)
    _save(fig, "reliability")


def transfer():
    import pandas as pd
    p = Path("reports/v2/transfer.csv")
    if not p.exists():
        print("[skip transfer] run analysis.report --only transfer first")
        return
    df = pd.read_csv(p)
    df["sens"] = df["Sensitivity"].astype(str).str.split().str[0].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, split in zip(axes, ["external", "pad"]):
        for model, g in df[df.Target == split].groupby("Model"):
            ax.plot(g["n labelled"], g["sens"], "o-", label=model)
        ax.axhline(0.90, color="k", ls=":", lw=0.8)
        ax.set_xscale("symlog", linthresh=25)
        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlabel("labelled target images used to refit threshold (0 = transfer as is)")
    axes[0].set_ylabel("sensitivity achieved (target 0.90)")
    axes[0].legend(fontsize=8)
    _save(fig, "transfer")


if __name__ == "__main__":
    for f in (roc, age_sweep, attention_maps, reliability, transfer):
        try:
            f()
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"[skip {f.__name__}] {type(e).__name__}: {e}")
