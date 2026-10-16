"""
Training (v2) — one script for every model and seed.

After training, the best checkpoint (by validation AUROC) predicts every split and the
per-image logits are saved. ALL metrics, thresholds, calibration and statistics are then
computed offline from those logits by analysis/ scripts (no GPU needed).

Outputs in runs_v2/<model_name>/seed<seed>/:
    model.pt              best state_dict + config
    history.csv           per-epoch train/val loss and AUROC
    summary.json          run metadata (params, best epoch, timings, versions)
    preds/<split>.csv.gz  image_name, target, logit

Usage:
    python -m src.train --config configs/v2/field_token.yaml --set seed=123
"""

import argparse
import json
import logging
import math
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import load_config, run_dir
from src.data.dataloader import ALL_SPLITS, get_dataloaders, pos_weight_value
from src.models.factory import build_model
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _to(batch, device):
    images, meta, site, labels, names = batch
    return (images.to(device, non_blocking=True), meta.to(device, non_blocking=True),
            site.to(device, non_blocking=True), labels.to(device, non_blocking=True), names)


@torch.no_grad()
def predict(model, loader, device, amp: bool):
    model.eval()
    names, ys, logits = [], [], []
    for batch in loader:
        x, m, s, y, n = _to(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(x, m, s)
        logits.append(out.float().cpu().numpy())
        ys.append(y.cpu().numpy())
        names.extend(n)
    return pd.DataFrame({"image_name": names, "target": np.concatenate(ys).astype(int),
                         "logit": np.concatenate(logits)})


def make_scheduler(opt, steps_per_epoch: int, epochs: int, warmup_epochs: float):
    total, warm = steps_per_epoch * epochs, int(steps_per_epoch * warmup_epochs)

    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def run(cfg: dict) -> dict:
    out = run_dir(cfg)
    if (out / "summary.json").exists() and not cfg.get("overwrite", False):
        log.info(f"{out} already complete — skipping")
        return json.loads((out / "summary.json").read_text())
    (out / "preds").mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"], deterministic=cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and cfg.get("use_amp", True)
    log.info(f"run={out} device={device} amp={amp}")

    loaders = get_dataloaders(cfg, ("train", "val"))
    model = build_model(cfg).to(device)
    if hasattr(model, "set_prior"):  # SAFE-Prior: frozen metadata prior fitted on the training split
        from src.models.safe_prior import fit_metadata_prior
        model.set_prior(*fit_metadata_prior(cfg))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    imb = cfg.get("imbalance", "none")
    pw = torch.tensor([pos_weight_value(cfg)], device=device) if imb == "pos_weight" else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    log.info(f"model={cfg['model_type']} params={n_params:,} imbalance={imb} pos_weight={pw}")

    # optional larger lr for the freshly initialised layers (standard for fine-tuning pretrained ViTs)
    is_bb = lambda n: "backbone" in n.split(".")  # noqa: E731
    groups = [{"params": [p for n, p in model.named_parameters() if is_bb(n)], "lr": cfg["lr"]},
              {"params": [p for n, p in model.named_parameters() if not is_bb(n)], "lr": cfg.get("head_lr", cfg["lr"])}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=cfg["weight_decay"])
    epochs = cfg["epochs"]
    sched = make_scheduler(opt, len(loaders["train"]), epochs, cfg.get("warmup_epochs", 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    max_batches = cfg.get("debug_max_batches")  # smoke tests only

    history, best_auc, best_epoch, bad = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, batch in enumerate(loaders["train"]):
            if max_batches and i >= max_batches:
                break
            x, m, s, y, _ = _to(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(x, m, s).float(), y)
            if hasattr(model, "extra_loss"):  # SAFE-Prior HSIC penalty; OGM coefficients
                loss = loss + model.extra_loss(y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if hasattr(model, "after_backward"):  # OGM gradient modulation
                model.after_backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)

        val = predict(model, loaders["val"], device, amp)
        if not np.isfinite(val.logit).all():  # diverged (lr too high / fp16 overflow): keep best checkpoint
            log.warning(f"non-finite validation logits at epoch {epoch}; stopping (best epoch {best_epoch})")
            if best_epoch == 0:
                raise RuntimeError("diverged before any valid epoch")
            break
        val_auc = roc_auc_score(val.target, val.logit)
        val_loss = float(nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(val.logit.values), torch.tensor(val.target.values, dtype=torch.float32)))
        history.append({"epoch": epoch, "train_loss": tot / max(1, n), "val_loss": val_loss,
                        "val_auroc": val_auc, "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        log.info(f"ep {epoch:02d} train_loss={tot / max(1, n):.4f} val_loss={val_loss:.4f} "
                 f"val_auroc={val_auc:.4f} ({time.time() - t0:.0f}s)")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        if val_auc > best_auc:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, out / "model.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log.info(f"early stop at epoch {epoch} (best {best_epoch})")
                break
    train_sec = time.time() - t_start

    # --- predict every split with the best checkpoint ---
    model.load_state_dict(torch.load(out / "model.pt", map_location=device, weights_only=False)["model_state_dict"])
    del loaders
    for split in cfg.get("predict_splits", list(ALL_SPLITS)):
        loader = get_dataloaders(cfg, (split,))[split]
        predict(model, loader, device, amp).to_csv(out / "preds" / f"{split}.csv.gz", index=False)
        log.info(f"saved preds/{split}")

    summary = {
        "model_name": cfg["model_name"], "model_type": cfg["model_type"], "seed": cfg["seed"],
        "n_params": n_params, "best_val_auroc": best_auc, "best_epoch": best_epoch,
        "epochs_run": len(history), "train_sec": train_sec, "total_sec": time.time() - t_start,
        "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "config": cfg,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides, e.g. seed=123 lr=1e-4")
    a = ap.parse_args()
    run(load_config(a.config, a.set))
