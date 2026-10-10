"""
Unified XAI explain API — Phase 7.

`explain_case(model, image, metadata)` → {sensitivity_curves, attention_map, nl_summary}

Usage:
    python src/xai/explain.py --config configs/cross_attention.yaml \
                              --checkpoint checkpoints/cross_attention_best.pth \
                              --n_cases 10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.metadata_encoder import MetadataEncoder, SITE_CATEGORIES
from src.data.transforms import get_inference_transforms, denormalize
from src.models.factory import build_model
from src.utils.seed import set_seed
from src.xai.counterfactual_engine import run_full_counterfactual
from src.xai.gradcam import GradCAM, get_default_target_layer, overlay_cam_on_image
from src.xai.visualization import plot_full_explanation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _generate_nl_summary(
    image_name: str,
    base_prob: float,
    age_sweep: Dict,
    site_sweep: Dict,
) -> str:
    """Template-based natural language explanation."""
    prob_pct = f"{base_prob:.1%}"

    # Age sensitivity
    ages = age_sweep["ages"]
    probs = age_sweep["probs"]
    min_age, max_age = ages[np.argmin(probs)], ages[np.argmax(probs)]
    delta = max(probs) - min(probs)

    # Site sensitivity
    sites = site_sweep["sites"]
    site_probs = site_sweep["probs"]
    highest_site = sites[np.argmax(site_probs)]
    lowest_site = sites[np.argmin(site_probs)]

    age_sentence = (
        f"Age is a {'significant' if delta > 0.1 else 'moderate'} factor: "
        f"predicted risk ranges from {min(probs):.1%} (age {min_age}) "
        f"to {max(probs):.1%} (age {max_age})."
    )
    site_sentence = (
        f"Anatomical site matters: highest risk predicted for '{highest_site}' "
        f"({max(site_probs):.1%}), lowest for '{lowest_site}' ({min(site_probs):.1%})."
    )
    return (
        f"For {image_name}, the model predicted P(malignant) = {prob_pct}. "
        f"{age_sentence} {site_sentence}"
    )


def explain_case(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,      # (1, 3, H, W)
    image_np: np.ndarray,            # (H, W, 3) uint8 original
    base_cont: np.ndarray,           # (continuous_dim,)
    base_site_idx: int,
    encoder: MetadataEncoder,
    device: torch.device,
    model_type: str,
    backbone_name: str,
    image_name: str = "case",
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Unified explanation for a single test case.

    Returns:
        {
            'image_name': str,
            'base_prob': float,
            'age_sweep': dict,
            'site_sweep': dict,
            'sex_sweep': dict,
            'cam': np.ndarray (H, W),
            'nl_summary': str,
        }
    """
    image_tensor = image_tensor.to(device)
    model.eval()  # Ensure eval mode for BatchNorm1d compatibility with batch_size=1

    # --- Counterfactual sweeps ---
    cf_results = run_full_counterfactual(
        model=model,
        image=image_tensor,
        base_cont=base_cont,
        base_site_idx=base_site_idx,
        encoder=encoder,
        device=device,
        model_type=model_type,
    )

    # --- Grad-CAM ---
    try:
        target_layer = get_default_target_layer(model_type, backbone_name)
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(
            image=image_tensor,
            metadata_cont=torch.tensor(base_cont, dtype=torch.float32).unsqueeze(0).to(device) if model_type != "baseline" else None,
            site_idx=torch.tensor([base_site_idx], dtype=torch.long).to(device) if model_type != "baseline" else None,
            model_type=model_type,
        )
        gradcam.remove_hooks()
    except Exception as e:
        log.warning(f"Grad-CAM failed: {e} — using zero heatmap.")
        cam = np.zeros(image_np.shape[:2], dtype=np.float32)

    cam_overlay = overlay_cam_on_image(image_np, cam)

    # --- Natural language summary ---
    nl_summary = _generate_nl_summary(
        image_name=image_name,
        base_prob=cf_results["base_prob"],
        age_sweep=cf_results["age_sweep"],
        site_sweep=cf_results["site_sweep"],
    )

    result = {
        "image_name": image_name,
        "base_prob": cf_results["base_prob"],
        "age_sweep": cf_results["age_sweep"],
        "site_sweep": cf_results["site_sweep"],
        "sex_sweep": cf_results["sex_sweep"],
        "cam": cam,
        "cam_overlay": cam_overlay,
        "nl_summary": nl_summary,
    }

    # --- Save outputs ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full explanation figure
        plot_full_explanation(
            original_image=image_np,
            cam_overlay=cam_overlay,
            age_sweep=cf_results["age_sweep"],
            site_sweep=cf_results["site_sweep"],
            base_prob=cf_results["base_prob"],
            image_name=image_name,
            nl_summary=nl_summary,
            save_path=output_dir / f"{image_name}_explanation.png",
        )

        # JSON summary (excluding numpy arrays)
        json_result = {
            k: v for k, v in result.items() if k not in ("cam", "cam_overlay")
        }
        with open(output_dir / f"{image_name}_summary.json", "w") as f:
            json.dump(json_result, f, indent=2)

    return result


def run_batch(config_path: str, checkpoint_path: str, n_cases: int = 10,
              melanoma_only: bool = False) -> None:
    """Runs explain_case on n_cases randomly sampled test images."""
    import pandas as pd
    from src.data.transforms import get_val_transforms

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = cfg.get("model_type", "cross_attention")
    backbone_name = cfg.get("backbone", "efficientnet_b2")

    # Load model
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load encoder
    encoder_path = Path(cfg.get("encoder_save_path", "data/processed/metadata_encoder.json"))
    encoder = MetadataEncoder.load(encoder_path)

    # Load test CSV from config's processed_dir
    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    test_df = pd.read_csv(processed_dir / "test.csv")

    if melanoma_only:
        test_df = test_df[test_df["target"] == 1].reset_index(drop=True)
        log.info(f"Filtered to {len(test_df)} melanoma cases")

    sample = test_df.sample(min(n_cases, len(test_df)), random_state=42)

    cont_arr, site_arr = encoder.transform(sample)

    # Use test_images_dir if available
    images_dir = Path(cfg.get("test_images_dir", cfg.get("images_dir", "data/raw/isic2020/images")))
    transform = get_val_transforms(cfg.get("img_size", 224))

    model_name = cfg.get("model_name", model_type)
    output_dir = Path(f"reports/xai_{model_name}")

    log.info(f"Generating explanations for {len(sample)} cases …")

    for i, (idx, row) in enumerate(sample.iterrows()):
        img_name = row["image_name"]
        try:
            for ext in [".jpg", ".jpeg", ""]:
                p = images_dir / f"{img_name}{ext}"
                if p.exists():
                    break
            image_np = np.array(Image.open(p).convert("RGB"))
            aug = transform(image=image_np)
            image_tensor = aug["image"].unsqueeze(0)

            explain_case(
                model=model,
                image_tensor=image_tensor,
                image_np=image_np,
                base_cont=cont_arr[i],
                base_site_idx=int(site_arr[i]),
                encoder=encoder,
                device=device,
                model_type=model_type,
                backbone_name=backbone_name,
                image_name=img_name,
                output_dir=output_dir,
            )
            log.info(f"[{i+1}/{len(sample)}] {img_name} — done.")
        except Exception as e:
            log.error(f"Failed for {img_name}: {e}")

    log.info(f"✓ Explanations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_cases", type=int, default=10)
    parser.add_argument("--melanoma_only", action="store_true",
                        help="Only sample from melanoma-positive cases")
    args = parser.parse_args()
    run_batch(args.config, args.checkpoint, args.n_cases, args.melanoma_only)

