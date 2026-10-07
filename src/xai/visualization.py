"""
XAI visualization — sensitivity curves and attention overlays — Phase 7.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
})


def plot_age_sensitivity(
    age_sweep: Dict,
    base_prob: float,
    image_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots predicted risk vs. patient age."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ages = age_sweep["ages"]
    probs = age_sweep["probs"]

    ax.plot(ages, probs, "-o", color="#4F83CC", lw=2, ms=5, label="Risk vs age")
    ax.axhline(base_prob, color="#E05C5C", ls="--", lw=1.5, label=f"Actual ({base_prob:.2%})")
    ax.fill_between(ages, probs, alpha=0.15, color="#4F83CC")
    ax.set_xlabel("Patient Age (years)")
    ax.set_ylabel("Predicted Malignancy Probability")
    ax.set_title(f"Age Sensitivity — {image_name}")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_site_sensitivity(
    site_sweep: Dict,
    base_prob: float,
    image_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots predicted risk across anatomical sites (bar chart)."""
    sites = site_sweep["sites"]
    probs = site_sweep["probs"]

    colors = ["#4F83CC" if p < base_prob else "#E05C5C" for p in probs]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(sites, probs, color=colors, edgecolor="white", linewidth=0.7)
    ax.axhline(base_prob, color="#333", ls="--", lw=1.5, label=f"Actual site ({base_prob:.2%})")
    ax.set_xlabel("Anatomical Site")
    ax.set_ylabel("Predicted Malignancy Probability")
    ax.set_title(f"Site Sensitivity — {image_name}")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_sex_sensitivity(
    sex_sweep: Dict,
    base_prob: float,
    image_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots predicted risk across sex categories."""
    sexes = sex_sweep["sexes"]
    probs = sex_sweep["probs"]

    fig, ax = plt.subplots(figsize=(5, 4))
    palette = ["#4F83CC", "#E05C5C", "#7AC67A"]
    ax.bar(sexes, probs, color=palette, edgecolor="white")
    ax.axhline(base_prob, color="#333", ls="--", lw=1.5, label=f"Actual ({base_prob:.2%})")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Predicted Malignancy Probability")
    ax.set_title(f"Sex Sensitivity — {image_name}")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_cam_overlay(
    original_image: np.ndarray,   # (H, W, 3) uint8
    cam_overlay: np.ndarray,      # (H, W, 3) uint8 blended
    image_name: str,
    pred_prob: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side: original image vs Grad-CAM overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original — {image_name}")
    axes[0].axis("off")

    axes[1].imshow(cam_overlay)
    axes[1].set_title(f"Grad-CAM | P(malignant)={pred_prob:.1%}")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_full_explanation(
    original_image: np.ndarray,
    cam_overlay: np.ndarray,
    age_sweep: Dict,
    site_sweep: Dict,
    base_prob: float,
    image_name: str,
    nl_summary: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """4-panel figure: original | CAM | age sensitivity | site sensitivity."""
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.35)

    # Panel 1: original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title("Dermoscopy Image", fontsize=11)
    ax1.axis("off")

    # Panel 2: CAM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cam_overlay)
    ax2.set_title(f"Grad-CAM | P={base_prob:.1%}", fontsize=11)
    ax2.axis("off")

    # Panel 3: age sweep
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(age_sweep["ages"], age_sweep["probs"], "-o", color="#4F83CC", lw=2, ms=4)
    ax3.axhline(base_prob, color="#E05C5C", ls="--", lw=1.5)
    ax3.set_xlabel("Age")
    ax3.set_ylabel("P(malignant)")
    ax3.set_title("Age Sensitivity")
    ax3.set_ylim(0, 1)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Panel 4: site sweep
    ax4 = fig.add_subplot(gs[0, 3])
    colors = ["#4F83CC" if p < base_prob else "#E05C5C" for p in site_sweep["probs"]]
    ax4.bar(range(len(site_sweep["sites"])), site_sweep["probs"], color=colors)
    ax4.set_xticks(range(len(site_sweep["sites"])))
    ax4.set_xticklabels(
        [s.replace("/", "/\n") for s in site_sweep["sites"]],
        rotation=45, ha="right", fontsize=8,
    )
    ax4.set_title("Site Sensitivity")
    ax4.set_ylim(0, 1)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(f"Explanation — {image_name}", fontsize=13, y=1.02)
    fig.text(0.5, -0.04, nl_summary, ha="center", fontsize=9, style="italic", wrap=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
