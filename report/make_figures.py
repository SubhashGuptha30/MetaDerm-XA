"""Results-independent figures for the report (pipeline, architecture, datasets, sample images).

    .venv/Scripts/python report/make_figures.py      # -> report/figures/*.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
IMG = ROOT / "data" / "kaggle_upload" / "metaderm-images"
SPLITS = ROOT / "splits" / "v2"


def _boxes(ax, boxes, arrows):
    for key, (x, y, w, h, text, color) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for a, b in arrows:
        xa, ya, wa, ha = boxes[a][:4]
        xb, yb, wb, hb = boxes[b][:4]
        if abs(ya - yb) < 1e-6:
            start, end = ((xa + wa, ya + ha / 2), (xb, yb + hb / 2)) if xb > xa else ((xa, ya + ha / 2), (xb + wb, yb + hb / 2))
        else:
            start, end = (xa + wa / 2, ya), (xb + wb / 2, yb + hb)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333"))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.5, 10)
    ax.axis("off")


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 7))
    B, G, O, P, R = "#dbe9f6", "#dff2de", "#fde9d4", "#ece2f4", "#f9dcdc"
    boxes = {
        "raw": (0.2, 8.4, 2.6, 1.1, "ISIC 2019 / ISIC 2020 /\nPAD-UFES-20 raw data", B),
        "dup": (3.6, 8.4, 2.8, 1.1, "Duplicate screen\n(pHash + pixel correlation)", R),
        "split": (7.2, 8.4, 2.6, 1.1, "Lesion-grouped splits\ntrain / val / calib", B),
        "train": (7.2, 6.3, 2.6, 1.1, "Train fusion + SAFE-Prior\n2 backbones x 5 seeds", O),
        "img": (3.6, 6.3, 2.8, 1.1, "Image resizing +\naugmentation", G),
        "meta": (0.2, 6.3, 2.6, 1.1, "Metadata encoding\n(train-only statistics)", G),
        "preds": (7.2, 4.2, 2.6, 1.1, "Per-image logits for\nval/calib/test/ext./PAD", O),
        "probe": (3.6, 4.2, 2.8, 1.1, "Probes: withheld / shuffled\nmetadata, sweeps, attention", P),
        "stats": (0.2, 4.2, 2.6, 1.1, "Offline statistics\n(bootstrap, DeLong, ECE)", P),
        "cal": (0.2, 2.1, 2.6, 1.1, "Thresholds, SAFE-Prior bound;\nlabel-free re-anchoring", G),
        "eval": (3.6, 2.1, 2.8, 1.1, "Evaluation: ISIC 2019 test,\nISIC 2020, PAD-UFES-20", R),
        "out": (7.2, 2.1, 2.6, 1.1, "Generated tables and\nfigures (paper, report)", B),
    }
    arrows = [("raw", "dup"), ("dup", "split"), ("split", "train"), ("meta", "img"), ("img", "train"),
              ("train", "preds"), ("preds", "probe"), ("probe", "stats"), ("stats", "cal"), ("cal", "eval"),
              ("eval", "out"), ("raw", "meta")]
    _boxes(ax, boxes, arrows)
    fig.savefig(OUT / "pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    B, G, P, O = "#dbe9f6", "#dff2de", "#ece2f4", "#fde9d4"
    boxes = {
        "x": (0.2, 7.6, 2.4, 1.2, "Dermoscopic image\n224 x 224", B),
        "cnn": (3.3, 7.6, 2.6, 1.2, "EfficientNet-B2\nfeature map 1408x7x7", B),
        "tok": (6.6, 7.6, 3.2, 1.2, "Flatten + projection\n49 image tokens (256-d)", B),
        "m": (0.2, 4.6, 2.4, 1.6, "Age | Sex | Site |\nMissing-age flag", G),
        "enc": (3.3, 4.6, 2.6, 1.6, "Separate encoder\nper field\n-> 4 metadata tokens", G),
        "att": (6.6, 4.6, 3.2, 1.6, "Cross-attention\nQ = image tokens\nK, V = metadata tokens", P),
        "ffn": (6.6, 2.3, 3.2, 1.2, "LayerNorm + feed-forward\n+ LayerNorm", P),
        "head": (3.3, 2.3, 2.6, 1.2, "Mean over tokens\n+ MLP", O),
        "y": (0.2, 2.3, 2.4, 1.2, "P(melanoma)", O),
    }
    arrows = [("x", "cnn"), ("cnn", "tok"), ("tok", "att"), ("m", "enc"), ("enc", "att"), ("att", "ffn"),
              ("ffn", "head"), ("head", "y")]
    _boxes(ax, boxes, arrows)
    ax.text(4.6, 6.9, "attention weights 49 x 4\n(one weight per field per location)", ha="center", fontsize=8, style="italic")
    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def datasets():
    names = {"train": "Train", "val": "Validation", "calib": "Calibration", "test": "ISIC 2019 test",
             "external": "ISIC 2020", "pad": "PAD-UFES-20"}
    dfs = {k: pd.read_csv(SPLITS / f"{k}.csv") for k in names}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(names.values(), [len(d) for d in dfs.values()], color="#4F83CC")
    axes[0].set_yscale("log")
    axes[0].set_title("Images per split (log scale)")
    axes[1].bar(names.values(), [100 * d.target.mean() for d in dfs.values()], color="#E05C5C")
    axes[1].set_title("Melanoma prevalence (%)")
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=35)
    for k, lab in [("train", "ISIC 2019 train"), ("external", "ISIC 2020"), ("pad", "PAD-UFES-20")]:
        axes[2].hist(dfs[k].age_approx, bins=range(0, 95, 5), histtype="step", lw=2, density=True, label=lab)
    axes[2].set_title("Age distribution")
    axes[2].set_xlabel("age (years)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "datasets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    sites = pd.DataFrame({lab: dfs[k].anatom_site_general_challenge.value_counts(normalize=True) * 100
                          for k, lab in [("train", "ISIC 2019 train"), ("external", "ISIC 2020"), ("pad", "PAD-UFES-20")]}).fillna(0)
    ax = sites.plot.barh(figsize=(8, 4.5), color=["#4F83CC", "#E05C5C", "#5CAE73"])
    ax.set_xlabel("% of images")
    ax.set_title("Anatomical site distribution")
    plt.tight_layout()
    plt.savefig(OUT / "sites.png", dpi=200, bbox_inches="tight")
    plt.close()


def samples():
    rows = [("ISIC 2019", "train"), ("ISIC 2020", "external"), ("PAD-UFES-20", "pad")]
    fig, axes = plt.subplots(3, 6, figsize=(13, 6.8))
    for r, (lab, split) in enumerate(rows):
        df = pd.read_csv(SPLITS / f"{split}.csv")
        pick = pd.concat([df[df.target == 1].sample(3, random_state=1), df[df.target == 0].sample(3, random_state=1)])
        for c, (_, row) in enumerate(pick.iterrows()):
            ax = axes[r, c]
            ax.imshow(Image.open(IMG / row.img_dir / f"{row.image_name}.jpg"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("melanoma" if row.target else "other", fontsize=9)
            if c == 0:
                ax.set_ylabel(lab, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def workflow():
    """Where each step runs: laptop (all data and results stay here) vs Kaggle (GPU work only)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    L, K, G = "#dbe9f6", "#fde9d4", "#dff2de"
    ax.add_patch(FancyBboxPatch((0.05, 1.7), 4.6, 8.1, boxstyle="round,pad=0.02", fc="none", ec="#1f77b4", lw=1.5, ls="--"))
    ax.add_patch(FancyBboxPatch((5.35, 1.7), 4.6, 8.1, boxstyle="round,pad=0.02", fc="none", ec="#ff7f0e", lw=1.5, ls="--"))
    ax.text(2.35, 9.95, "Laptop (CPU)", ha="center", fontsize=11, weight="bold")
    ax.text(7.65, 9.95, "Kaggle session (2 x NVIDIA T4)", ha="center", fontsize=11, weight="bold")
    boxes = {
        "prep": (0.3, 8.3, 4.1, 1.1, "preprocess_v2.py, near_duplicates.py\n-> frozen splits (splits/v2)", L),
        "img": (0.3, 6.5, 4.1, 1.1, "prepare_kaggle_images.py\n-> resized images (private dataset)", L),
        "code": (0.3, 4.7, 4.1, 1.1, "launch.py code / push <job>\n-> code dataset + kernel", L),
        "ana": (0.3, 2.0, 4.1, 1.1, "launch.py pull -> runs_v2/\nanalysis.report / figures", G),
        "run": (5.6, 8.3, 4.1, 1.1, "runner: job list, one worker per GPU,\ntime-budget guard", K),
        "train": (5.6, 6.5, 4.1, 1.1, "src.train: model.pt, history,\nlogits for val/calib/test/ext./PAD", K),
        "probe": (5.6, 4.7, 4.1, 1.1, "src.probes: withheld / shuffled\nmetadata, sweeps, masking", K),
        "out": (5.6, 2.0, 4.1, 1.1, "run folders + job_status.txt\nin /kaggle/working", K),
    }
    arrows = [("prep", "img"), ("img", "code"), ("run", "train"), ("train", "probe")]
    _boxes(ax, boxes, arrows)
    for (xa, ya), (xb, yb) in [((4.4, 5.25), (5.6, 8.85)), ((5.6, 2.55), (4.4, 2.55))]:
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333"))
    ax.add_patch(FancyArrowPatch((7.65, 4.7), (7.65, 3.1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333"))
    ax.set_ylim(1.5, 10.3)
    fig.savefig(OUT / "workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    workflow()
    architecture()
    datasets()
    samples()
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))
