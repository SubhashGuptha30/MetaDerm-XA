"""
Multi-Seed Training and Evaluation Script
==========================================
Trains all 5 model architectures across 5 random seeds, then aggregates
mean +/- std for all metrics to establish variance bounds.

Requirements: GPU (CUDA), ~50 training runs total (~12.5 hours on RTX 3060)
Usage: python scripts/multi_seed_train.py [--seeds 42 123 456 789 2024] [--models all]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, roc_curve
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_model, train_one_epoch
from src.data.dataloader import get_dataloaders
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 2024]
MODELS = ['baseline', 'baseline_wide', 'concat_fusion', 'cross_attention', 'field_token', 'field_token_k1', 'field_token_k2', 'field_token_k3', 'film']
MODEL_LABELS = {
    'baseline': 'Baseline',
    'baseline_wide': 'Baseline-Wide',
    'concat_fusion': 'Concat',
    'cross_attention': 'Cross-Attn',
    'field_token': 'Field-Token (K=4)',
    'field_token_k1': 'Field-Token (K=1)',
    'field_token_k2': 'Field-Token (K=2)',
    'field_token_k3': 'Field-Token (K=3)',
    'film': 'FiLM',
}

CONFIG_DIR = PROJECT_ROOT / 'configs' / 'isic2019'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'multi_seed'


def evaluate_with_preds(model, loader, criterion, device, model_type):
    """Evaluate and return (loss, auroc, labels_array, probs_array)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
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
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)
    try:
        auroc = roc_auc_score(labels_arr, probs_arr)
    except Exception:
        auroc = 0.0
    return avg_loss, auroc, labels_arr, probs_arr


def load_config(model_name, seed):
    config_path = CONFIG_DIR / f'{model_name}.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['seed'] = seed
    cfg['model_name'] = f'{model_name}_seed{seed}'
    cfg['checkpoint_dir'] = str(OUTPUT_DIR / 'checkpoints')
    cfg['reports_dir'] = str(OUTPUT_DIR / 'results')
    return cfg


def train_single_run(model_name, seed, device):
    log.info(f"Training {MODEL_LABELS[model_name]} with seed={seed}")
    
    cfg = load_config(model_name, seed)
    set_seed(seed)
    
    loaders = get_dataloaders(cfg)
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    model = build_model(cfg).to(device)
    
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    optimizer = AdamW(model.parameters(), lr=cfg.get('lr', 3e-5), weight_decay=cfg.get('weight_decay', 5e-4))
    pos_weight = torch.tensor([cfg.get('pos_weight', 5.0)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.get('epochs', 25))
    scaler = torch.amp.GradScaler() if cfg.get('use_amp', True) else None
    
    model_type = cfg.get('model_type', 'baseline')
    best_auroc = 0.0
    patience_counter = 0
    patience = cfg.get('patience', 8)
    
    checkpoint_dir = Path(cfg['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'{cfg["model_name"]}_best.pth'
    
    for epoch in range(cfg.get('epochs', 25)):
        train_loss, train_auroc = train_one_epoch(model, train_loader, optimizer, criterion, device, model_type, scaler)
        val_loss, val_auroc, val_labels, val_probs = evaluate_with_preds(model, val_loader, criterion, device, model_type)
        scheduler.step()
        
        log.info(f"  Epoch {epoch+1}/{cfg['epochs']} — Train: {train_auroc:.4f}, Val: {val_auroc:.4f}")
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best and evaluate
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    _, _, test_labels, test_probs = evaluate_with_preds(model, test_loader, criterion, device, model_type)
    _, _, val_labels2, val_probs2 = evaluate_with_preds(model, val_loader, criterion, device, model_type)
    
    # Youden threshold from validation
    fpr_v, tpr_v, thresholds_v = roc_curve(val_labels2, val_probs2)
    youden_idx = np.argmax(tpr_v - fpr_v)
    threshold = float(thresholds_v[youden_idx])
    
    test_preds = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    
    metrics = {
        'model': model_name,
        'seed': seed,
        'auroc': float(roc_auc_score(test_labels, test_probs)),
        'auprc': float(average_precision_score(test_labels, test_probs)),
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        'f1': float(2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 0,
        'brier': float(brier_score_loss(test_labels, test_probs)),
        'threshold': threshold,
        'best_val_auroc': best_auroc,
        'epochs_trained': epoch + 1,
    }
    
    results_dir = Path(cfg['reports_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / f'{cfg["model_name"]}_labels.npy', test_labels)
    np.save(results_dir / f'{cfg["model_name"]}_probs.npy', test_probs)
    with open(results_dir / f'{cfg["model_name"]}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    log.info(f"  => AUROC={metrics['auroc']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    return metrics


def aggregate_and_report(all_metrics):
    df = pd.DataFrame(all_metrics)
    
    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS (mean +/- std)")
    print("=" * 70)
    
    for model_name in MODELS:
        subset = df[df['model'] == model_name]
        if len(subset) == 0:
            continue
        label = MODEL_LABELS[model_name]
        print(f"\n{label} (n={len(subset)} seeds):")
        for metric in ['auroc', 'auprc', 'recall', 'specificity', 'f1', 'brier']:
            vals = subset[metric]
            print(f"  {metric}: {vals.mean():.4f} +/- {vals.std():.4f}  [{vals.min():.4f} - {vals.max():.4f}]")
    
    # LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-seed evaluation on ISIC 2019 test set ($n = 5$ seeds, mean $\pm$ std). Threshold-dependent metrics use the Youden-optimal threshold from each seed's validation set.}")
    lines.append(r"\label{tab:multi_seed}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & AUROC & AUPRC & Recall & Specificity & F1 & Brier \\")
    lines.append(r"\midrule")
    
    for model_name in MODELS:
        subset = df[df['model'] == model_name]
        if len(subset) == 0:
            continue
        label = MODEL_LABELS[model_name]
        suffix = r'$^\dagger$' if model_name == 'field_token' else ''
        
        cols = []
        for metric in ['auroc', 'auprc', 'recall', 'specificity', 'f1', 'brier']:
            m, s = subset[metric].mean(), subset[metric].std()
            cols.append(f"{m:.3f}$\\pm${s:.3f}")
        
        lines.append(f"{label}{suffix} & {' & '.join(cols)} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    print("\n".join(lines))
    return df


def main():
    parser = argparse.ArgumentParser(description='Multi-seed training')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    log.info(f"Device: {device}")
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        log.error("No GPU available! Use: python scripts/multi_seed_train.py --device cpu")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    total_runs = len(args.models) * len(args.seeds)
    run_idx = 0
    
    for model_name in args.models:
        for seed in args.seeds:
            run_idx += 1
            log.info(f"\n[Run {run_idx}/{total_runs}]")
            try:
                metrics = train_single_run(model_name, seed, device)
                all_metrics.append(metrics)
            except Exception as e:
                log.error(f"Failed: {model_name} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / 'all_metrics_partial.csv', index=False)
    
    if all_metrics:
        df = aggregate_and_report(all_metrics)
        df.to_csv(OUTPUT_DIR / 'all_metrics.csv', index=False)
        log.info(f"\nCompleted: {len(all_metrics)}/{total_runs} runs")
        log.info(f"Results: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
