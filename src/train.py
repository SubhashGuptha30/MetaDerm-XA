"""
Generic training loop — Phase 4 & 5.
Supports config-driven model selection, TensorBoard logging, early stopping.

Usage:
    python src/train.py --config configs/cross_attention.yaml
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataloader import get_dataloaders
from src.utils.seed import set_seed
from src.utils.visualization import plot_training_curves

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_model(cfg: dict) -> nn.Module:
    """Factory function — returns model based on cfg['model_type']."""
    model_type = cfg.get("model_type", "baseline")

    if model_type == "baseline":
        from src.models.baseline import BaselineModel
        return BaselineModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            dropout=cfg.get("dropout", 0.4),
        )

    elif model_type == "concat_fusion":
        from src.models.concat_fusion import ConcatFusionModel
        return ConcatFusionModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            feature_dim=cfg.get("feature_dim", 256),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "cross_attention":
        from src.models.multimodal_model import MultiModalModel
        return MultiModalModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            freeze_early=cfg.get("freeze_early", False),
            feature_dim=cfg.get("feature_dim", 256),
            num_attn_heads=cfg.get("num_attn_heads", 8),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "quantum_fusion":
        from src.models.quantum_fusion_model import QuantumFusionModel
        return QuantumFusionModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            feature_dim=cfg.get("feature_dim", 256),
            num_attn_heads=cfg.get("num_attn_heads", 8),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "field_token":
        from src.models.field_token_model import FieldTokenMultiModalModel
        return FieldTokenMultiModalModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            freeze_early=cfg.get("freeze_early", False),
            feature_dim=cfg.get("feature_dim", 256),
            num_attn_heads=cfg.get("num_attn_heads", 8),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "film":
        from src.models.film_model import FiLMModel
        return FiLMModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            feature_dim=cfg.get("feature_dim", 256),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "baseline_wide":
        from src.models.baseline_wide import BaselineWideModel
        return BaselineWideModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            dropout=cfg.get("dropout", 0.4),
        )

    elif model_type == "field_token_ablated":
        from src.models.ablated_field_token_model import AblatedFieldTokenModel
        return AblatedFieldTokenModel(
            backbone_name=cfg.get("backbone", "efficientnet_b2"),
            pretrained=cfg.get("pretrained", True),
            freeze_early=cfg.get("freeze_early", False),
            feature_dim=cfg.get("feature_dim", 256),
            num_attn_heads=cfg.get("num_attn_heads", 8),
            num_meta_tokens=cfg.get("num_meta_tokens", 2),
            dropout=cfg.get("dropout", 0.3),
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device,
    model_type: str,
    scaler,
) -> Tuple[float, float]:
    """Returns (avg_loss, auroc)."""
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        images, meta_cont, site_idx, labels, _ = batch
        images = images.to(device)
        meta_cont = meta_cont.to(device)
        site_idx = site_idx.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if model_type in ("baseline", "baseline_wide"):
                logits = model(images)
            else:
                logits = model(images, meta_cont, site_idx)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

    import numpy as np
    avg_loss = total_loss / len(loader.dataset)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = 0.0
    return avg_loss, auroc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion,
    device,
    model_type: str,
) -> Tuple[float, float]:
    """Returns (avg_loss, auroc)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in tqdm(loader, desc="Val  ", leave=False):
        images, meta_cont, site_idx, labels, _ = batch
        images = images.to(device)
        meta_cont = meta_cont.to(device)
        site_idx = site_idx.to(device)
        labels = labels.to(device)

        if model_type in ("baseline", "baseline_wide"):
            logits = model(images)
        else:
            logits = model(images, meta_cont, site_idx)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = 0.0
    return avg_loss, auroc


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Data ---
    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # --- Model ---
    model = build_model(cfg).to(device)
    model_type = cfg.get("model_type", "baseline")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {model_type} | Trainable params: {n_params:,}")

    # --- Loss with pos_weight for imbalance ---
    pos_weight = cfg.get("pos_weight", None)
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        log.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # --- Optimizer & Scheduler ---
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_epochs = cfg.get("epochs", 20)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # --- AMP ---
    use_amp = torch.cuda.is_available() and cfg.get("use_amp", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Checkpointing ---
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = cfg.get("model_name", model_type)
    best_ckpt = ckpt_dir / f"{ckpt_name}_best.pth"

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=f"runs/{ckpt_name}")

    # --- Training loop ---
    best_val_auroc = 0.0
    patience = cfg.get("patience", 7)
    patience_counter = 0
    train_losses, val_losses, val_aurocs = [], [], []

    log.info(f"Starting training for {n_epochs} epochs …")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, model_type, scaler
        )
        val_loss, val_auroc = evaluate_epoch(
            model, val_loader, criterion, device, model_type
        )
        scheduler.step()

        elapsed = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)

        log.info(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"train_loss={train_loss:.4f} train_auroc={train_auroc:.4f} | "
            f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} | "
            f"{elapsed:.1f}s"
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("AUROC/val", val_auroc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": val_auroc,
                    "config": cfg,
                },
                best_ckpt,
            )
            log.info(f"  ↑ New best val_auroc={val_auroc:.4f} — checkpoint saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    writer.close()

    # --- Save training curves ---
    reports_dir = Path(cfg.get("reports_dir", "reports/results"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(
        train_losses, val_losses, val_aurocs,
        save_path=reports_dir / f"{ckpt_name}_training_curves.png",
    )

    summary = {
        "model_name": ckpt_name,
        "best_val_auroc": best_val_auroc,
        "epochs_trained": len(train_losses),
        "n_params": n_params,
        "config": cfg,
    }
    with open(reports_dir / f"{ckpt_name}_train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Training complete. Best val AUROC: {best_val_auroc:.4f}")
    log.info(f"Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)
