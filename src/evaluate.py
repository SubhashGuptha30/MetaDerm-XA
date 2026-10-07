"""
Full evaluation suite — Phase 8.
Computes metrics with bootstrap CIs, calibration, DeLong test vs baseline.

Usage:
    python src/evaluate.py --config configs/cross_attention.yaml \
                           --checkpoint checkpoints/cross_attention_best.pth \
                           [--baseline_prob reports/results/baseline_probs.npy]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data.dataloader import get_dataloaders
from src.train import build_model
from src.utils.metrics import (
    auroc_bootstrap_ci,
    bootstrap_ci,
    compute_metrics,
    delong_test,
    find_optimal_threshold,
)
from src.utils.seed import set_seed
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, loader, device, model_type):
    """Returns (y_true, y_prob, image_names)."""
    model.eval()
    all_labels, all_probs, all_names = [], [], []

    for batch in tqdm(loader, desc="Inference"):
        images, meta_cont, site_idx, labels, names = batch
        images = images.to(device)
        meta_cont = meta_cont.to(device)
        site_idx = site_idx.to(device)

        if model_type in ("baseline", "baseline_wide"):
            logits = model(images)
        else:
            logits = model(images, meta_cont, site_idx)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_names.extend(names)

    return np.array(all_labels), np.array(all_probs), all_names


def run(config_path: str, checkpoint_path: str, baseline_prob_path: str = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model_type = cfg.get("model_type", "baseline")
    model_name = cfg.get("model_name", model_type)

    # --- Load model ---
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, val_auroc={ckpt.get('val_auroc', '?'):.4f}")

    # --- Data ---
    loaders = get_dataloaders(cfg)
    test_loader = loaders["test"]

    # --- Inference ---
    y_true, y_prob, image_names = run_inference(model, test_loader, device, model_type)

    # --- Optimal threshold ---
    val_loader = loaders["val"]
    y_val_true, y_val_prob, _ = run_inference(model, val_loader, device, model_type)
    threshold = find_optimal_threshold(y_val_true, y_val_prob, metric="youden")
    log.info(f"Optimal threshold (Youden, from val): {threshold:.3f}")

    # --- Metrics ---
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    log.info(f"Test metrics: {metrics}")

    # --- Bootstrap CIs ---
    from sklearn.metrics import roc_auc_score, recall_score
    auroc_pt, auroc_lo, auroc_hi = auroc_bootstrap_ci(y_true, y_prob, n_bootstrap=1000)
    recall_pt, recall_lo, recall_hi = bootstrap_ci(
        y_true, y_prob,
        lambda yt, yp: recall_score(yt, (yp >= threshold).astype(int), zero_division=0),
        n_bootstrap=1000,
    )
    metrics["auroc_ci"] = f"{auroc_pt:.3f} [{auroc_lo:.3f}–{auroc_hi:.3f}]"
    metrics["recall_ci"] = f"{recall_pt:.3f} [{recall_lo:.3f}–{recall_hi:.3f}]"
    log.info(f"AUROC: {metrics['auroc_ci']}")
    log.info(f"Recall: {metrics['recall_ci']}")

    # --- DeLong test vs baseline ---
    if baseline_prob_path and Path(baseline_prob_path).exists():
        baseline_probs = np.load(baseline_prob_path)
        z, p = delong_test(y_true, y_prob, baseline_probs)
        metrics["delong_z"] = z
        metrics["delong_p"] = p
        log.info(f"DeLong test vs baseline: z={z:.3f}, p={p:.4f}")

    # --- Save outputs ---
    reports_dir = Path(cfg.get("reports_dir", "reports/results"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(reports_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save raw probs for DeLong comparisons
    np.save(reports_dir / f"{model_name}_probs.npy", y_prob)
    np.save(reports_dir / f"{model_name}_labels.npy", y_true)

    # Plots
    plot_confusion_matrix(
        tp=metrics["tp"], fp=metrics["fp"], fn=metrics["fn"], tn=metrics["tn"],
        save_path=reports_dir / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} — Confusion Matrix",
    )
    plot_roc_curve(
        y_true, y_prob, auroc=metrics["auroc"],
        label=model_name,
        save_path=reports_dir / f"{model_name}_roc_curve.png",
    )
    plot_reliability_diagram(
        y_true, y_prob,
        save_path=reports_dir / f"{model_name}_calibration.png",
        title=f"{model_name} — Calibration",
    )

    # --- Error analysis: false negatives ---
    y_pred = (y_prob >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_names = [n for n, m in zip(image_names, fn_mask) if m]
    fn_probs = y_prob[fn_mask]

    fn_report = [
        {"image": n, "predicted_prob": float(p)}
        for n, p in zip(fn_names, fn_probs)
    ]
    with open(reports_dir / f"{model_name}_false_negatives.json", "w") as f:
        json.dump(fn_report, f, indent=2)

    log.info(f"False negatives: {len(fn_names)} cases saved to {reports_dir}")
    log.info(f"✓ Evaluation complete. Results in {reports_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline_prob", default=None, help="Path to baseline_probs.npy for DeLong test")
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.baseline_prob)
