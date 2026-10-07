"""
Visualization utilities — Phase 8.
Confusion matrix, ROC curve, reliability diagram (calibration), training curves.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

# Consistent style
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_confusion_matrix(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Benign", "Pred Malignant"],
        yticklabels=["True Benign", "True Malignant"],
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auroc: float,
    label: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUROC={auroc:.3f})", color="#4F83CC")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Reliability Diagram (Calibration)",
) -> plt.Figure:
    """
    A well-calibrated model's points lie on the diagonal.
    Points above → underconfident; below → overconfident.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color="#E05C5C", label="Model", lw=2)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_aurocs: List[float],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss", color="#4F83CC", lw=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="#E05C5C", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, val_aurocs, label="Val AUROC", color="#5CAE73", lw=2, marker="o", ms=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Validation AUROC")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_ablation_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Renders the ablation table as a heatmap-style figure.
    results: {model_name: {metric: value}}
    """
    import pandas as pd

    df = pd.DataFrame(results).T
    metric_cols = [c for c in ["auroc", "recall", "precision", "f1"] if c in df.columns]
    df = df[metric_cols]

    fig, ax = plt.subplots(figsize=(len(metric_cols) * 2, len(results) + 1))
    sns.heatmap(
        df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Ablation Results", pad=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig
