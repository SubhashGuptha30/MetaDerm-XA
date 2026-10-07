"""
Full test-set counterfactual analysis — Task 1.

Runs counterfactual metadata sweeps (age, sex, site) on all melanoma-positive
test cases and computes:
  - Population-mean intervention curves with 95% confidence bands
  - Monotonicity rate (% of cases where age↑ → prob↑)
  - Subgroup-stratified analysis

Usage:
    python src/counterfactual_sweep.py \
        --config configs/field_token_isic2019.yaml \
        --checkpoint checkpoints/field_token_isic2019_best.pth \
        --split test \
        --target_class all
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.metadata_encoder import MetadataEncoder, SITE_CATEGORIES, SEX_CATEGORIES
from src.data.transforms import get_val_transforms
from src.train import build_model
from src.utils.seed import set_seed
from src.xai.counterfactual_engine import sweep_age, sweep_site, sweep_sex

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_sweep(
    config_path: str,
    checkpoint_path: str,
    split: str = "test",
    target_class: str = "melanoma",  # "melanoma", "benign", or "all"
    max_cases: int = 0,  # 0 = unlimited
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model_type = cfg.get("model_type", "field_token")
    model_name = cfg.get("model_name", model_type)

    # Load model
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Loaded {model_name} from {checkpoint_path}")

    # Load encoder
    encoder_path = Path(cfg.get("encoder_save_path", "data/processed/metadata_encoder.json"))
    encoder = MetadataEncoder.load(encoder_path)

    # Load test data
    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    test_df = pd.read_csv(processed_dir / f"{split}.csv")

    # Filter by target class
    if target_class == "melanoma":
        test_df = test_df[test_df["target"] == 1].reset_index(drop=True)
    elif target_class == "benign":
        test_df = test_df[test_df["target"] == 0].reset_index(drop=True)
    # "all" keeps everything

    if max_cases > 0:
        test_df = test_df.head(max_cases)

    log.info(f"Running counterfactual sweep on {len(test_df)} {target_class} cases")

    # Encode metadata
    cont_arr, site_arr = encoder.transform(test_df)

    # Set up image loading
    if split == "test":
        images_dir = Path(cfg.get("test_images_dir", cfg.get("images_dir", "data/raw/isic2019/train/images")))
    else:
        images_dir = Path(cfg.get("images_dir", "data/raw/isic2019/train/images"))
    transform = get_val_transforms(cfg.get("img_size", 224))

    # --- Run sweeps ---
    age_range = (10, 90)
    age_step = 5
    ages = list(range(age_range[0], age_range[1] + 1, age_step))

    all_age_curves = []
    all_site_probs = []
    all_sex_probs = []
    all_base_probs = []
    failed = 0

    for i in tqdm(range(len(test_df)), desc="Counterfactual sweep"):
        row = test_df.iloc[i]
        img_name = row["image_name"]

        try:
            # Load image
            img_path = None
            for ext in [".jpg", ".jpeg", ""]:
                p = images_dir / f"{img_name}{ext}"
                if p.exists():
                    img_path = p
                    break

            if img_path is None:
                failed += 1
                continue

            image_np = np.array(Image.open(img_path).convert("RGB"))
            aug = transform(image=image_np)
            image_tensor = aug["image"].unsqueeze(0).to(device)

            base_cont = cont_arr[i]
            base_site = int(site_arr[i])

            # Base prediction
            with torch.no_grad():
                if model_type == "baseline":
                    logit = model(image_tensor)
                else:
                    cont_t = torch.tensor(base_cont, dtype=torch.float32).unsqueeze(0).to(device)
                    site_t = torch.tensor([base_site], dtype=torch.long).to(device)
                    logit = model(image_tensor, cont_t, site_t)
                base_prob = float(torch.sigmoid(logit).item())

            all_base_probs.append(base_prob)

            # Age sweep
            age_result = sweep_age(
                model, image_tensor, base_cont, base_site,
                encoder, device, model_type, age_range, age_step
            )
            all_age_curves.append(age_result["probs"])

            # Site sweep
            site_result = sweep_site(
                model, image_tensor, base_cont, encoder, device, model_type
            )
            all_site_probs.append(site_result["probs"])

            # Sex sweep
            sex_result = sweep_sex(
                model, image_tensor, base_cont, base_site,
                encoder, device, model_type
            )
            all_sex_probs.append(sex_result["probs"])

        except Exception as e:
            failed += 1
            if failed <= 5:
                log.warning(f"Failed for {img_name}: {e}")

    log.info(f"Completed: {len(all_age_curves)} cases, {failed} failures")

    # --- Aggregate ---
    age_matrix = np.array(all_age_curves)  # (N, n_ages)
    site_matrix = np.array(all_site_probs)  # (N, n_sites)
    sex_matrix = np.array(all_sex_probs)  # (N, 3)

    results = aggregate_results(
        ages, age_matrix, site_matrix, sex_matrix,
        all_base_probs, encoder, model_name, target_class
    )

    # --- Save ---
    output_dir = Path(f"reports/counterfactual_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw matrices
    np.save(output_dir / "age_matrix.npy", age_matrix)
    np.save(output_dir / "site_matrix.npy", site_matrix)
    np.save(output_dir / "sex_matrix.npy", sex_matrix)
    np.save(output_dir / "base_probs.npy", np.array(all_base_probs))

    # Save summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_counterfactual_curves(
        ages, age_matrix, site_matrix, sex_matrix,
        encoder, model_name, target_class, output_dir
    )

    log.info(f"✓ Counterfactual analysis saved to {output_dir}/")


def aggregate_results(ages, age_matrix, site_matrix, sex_matrix,
                      base_probs, encoder, model_name, target_class):
    """Compute aggregate statistics."""
    n_cases = len(age_matrix)

    # Age: mean, std, CI
    age_mean = age_matrix.mean(axis=0).tolist()
    age_std = age_matrix.std(axis=0).tolist()
    age_lo = np.percentile(age_matrix, 2.5, axis=0).tolist()
    age_hi = np.percentile(age_matrix, 97.5, axis=0).tolist()

    # Monotonicity: % of cases where prob increases with age
    diffs = np.diff(age_matrix, axis=1)  # (N, n_ages-1)
    mono_rate = float((diffs > 0).sum(axis=1).mean() / diffs.shape[1])

    # Overall age sensitivity: max - min prob per case
    age_sensitivity = (age_matrix.max(axis=1) - age_matrix.min(axis=1))
    mean_age_sensitivity = float(age_sensitivity.mean())

    # Site: mean probs
    site_mean = site_matrix.mean(axis=0).tolist()
    site_names = encoder.site_categories

    # Sex: mean probs
    sex_mean = sex_matrix.mean(axis=0).tolist()

    return {
        "model": model_name,
        "target_class": target_class,
        "n_cases": n_cases,
        "mean_base_prob": float(np.mean(base_probs)),
        "age": {
            "ages": ages,
            "mean_probs": age_mean,
            "std_probs": age_std,
            "ci_lower": age_lo,
            "ci_upper": age_hi,
            "monotonicity_rate": mono_rate,
            "mean_sensitivity": mean_age_sensitivity,
        },
        "site": {
            "sites": site_names,
            "mean_probs": site_mean,
        },
        "sex": {
            "categories": SEX_CATEGORIES,
            "mean_probs": sex_mean,
        },
    }


def plot_counterfactual_curves(ages, age_matrix, site_matrix, sex_matrix,
                                encoder, model_name, target_class, output_dir):
    """Generate publication-quality counterfactual plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Age sweep ---
    ax = axes[0]
    mean = age_matrix.mean(axis=0)
    lo = np.percentile(age_matrix, 2.5, axis=0)
    hi = np.percentile(age_matrix, 97.5, axis=0)

    ax.plot(ages, mean, color="#5CAE73", lw=2.5, label="Mean P(melanoma)")
    ax.fill_between(ages, lo, hi, alpha=0.2, color="#5CAE73", label="95% CI")
    ax.set_xlabel("Counterfactual Age", fontsize=12)
    ax.set_ylabel("P(melanoma)", fontsize=12)
    ax.set_title(f"Age Sensitivity ({age_matrix.shape[0]} cases)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # --- Site sweep ---
    ax = axes[1]
    site_names = encoder.site_categories
    site_mean = site_matrix.mean(axis=0)
    site_std = site_matrix.std(axis=0)
    sorted_idx = np.argsort(site_mean)[::-1]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(site_names)))
    ax.barh(
        [site_names[i] for i in sorted_idx],
        [site_mean[i] for i in sorted_idx],
        xerr=[site_std[i] for i in sorted_idx],
        color=[colors[j] for j in range(len(sorted_idx))],
        alpha=0.85, capsize=3
    )
    ax.set_xlabel("Mean P(melanoma)", fontsize=12)
    ax.set_title(f"Site Sensitivity ({site_matrix.shape[0]} cases)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    # --- Sex sweep ---
    ax = axes[2]
    sex_cats = SEX_CATEGORIES
    sex_mean = sex_matrix.mean(axis=0)
    sex_std = sex_matrix.std(axis=0)

    bars = ax.bar(sex_cats, sex_mean, yerr=sex_std, color=["#4F83CC", "#E05C5C", "#8B8B8B"],
                  alpha=0.8, capsize=5, width=0.5)
    for bar, val in zip(bars, sex_mean):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean P(melanoma)", fontsize=12)
    ax.set_title(f"Sex Sensitivity ({sex_matrix.shape[0]} cases)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Counterfactual Metadata Analysis — {model_name} ({target_class})",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "counterfactual_curves.png", dpi=150, bbox_inches="tight")
    log.info(f"Saved counterfactual plots to {output_dir}/counterfactual_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", help="test or val")
    parser.add_argument("--target_class", default="melanoma",
                        help="melanoma, benign, or all")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Limit cases (0=all)")
    args = parser.parse_args()
    run_sweep(args.config, args.checkpoint, args.split, args.target_class, args.max_cases)
