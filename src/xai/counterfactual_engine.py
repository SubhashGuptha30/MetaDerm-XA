"""
Counterfactual metadata sensitivity engine — Phase 7. Core novelty.

For a fixed test image, sweeps one metadata variable while holding others constant
and records how predicted malignancy probability changes.

This answers: "How does the model's risk estimate change if this patient were
older / younger / at a different anatomical site?"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.metadata_encoder import MetadataEncoder, SITE_CATEGORIES, SEX_CATEGORIES

log = logging.getLogger(__name__)


@torch.no_grad()
def _predict_prob(
    model,
    image: torch.Tensor,      # (1, 3, H, W)
    cont: np.ndarray,         # (continuous_dim,)
    site_idx: int,
    device: torch.device,
    model_type: str,
) -> float:
    """Single forward pass returning P(malignant)."""
    model.eval()  # Required: BatchNorm1d crashes with batch_size=1 in train mode
    cont_t = torch.tensor(cont, dtype=torch.float32).unsqueeze(0).to(device)
    site_t = torch.tensor([site_idx], dtype=torch.long).to(device)

    if model_type == "baseline":
        logit = model(image)
    else:
        logit = model(image, cont_t, site_t)

    return float(torch.sigmoid(logit).item())


def sweep_age(
    model,
    image: torch.Tensor,       # (1, 3, H, W)
    base_cont: np.ndarray,     # (continuous_dim,) — reference metadata
    site_idx: int,
    encoder: MetadataEncoder,
    device: torch.device,
    model_type: str = "cross_attention",
    age_range: Tuple[int, int] = (10, 90),
    step: int = 5,
) -> Dict[str, list]:
    """
    Sweeps age while holding sex, site, age_missing constant.

    Returns:
        {
            'ages': [...],
            'probs': [...],
        }
    """
    ages = list(range(age_range[0], age_range[1] + 1, step))
    probs = []

    for age in ages:
        cont = base_cont.copy()
        # Index 0 is age_norm; re-normalize using encoder stats
        cont[0] = (age - encoder.age_mean) / encoder.age_std
        prob = _predict_prob(model, image, cont, site_idx, device, model_type)
        probs.append(prob)

    log.info(f"Age sweep complete: {len(ages)} steps, prob range [{min(probs):.3f}, {max(probs):.3f}]")
    return {"ages": ages, "probs": probs}


def sweep_site(
    model,
    image: torch.Tensor,
    base_cont: np.ndarray,
    encoder: MetadataEncoder,
    device: torch.device,
    model_type: str = "cross_attention",
) -> Dict[str, list]:
    """
    Sweeps anatomical site across all categories while holding age and sex constant.

    Returns:
        {
            'sites': [...],  (str labels)
            'probs': [...],
        }
    """
    sites = encoder.site_categories
    probs = []

    for i, site_name in enumerate(sites):
        prob = _predict_prob(model, image, base_cont, i, device, model_type)
        probs.append(prob)

    log.info(f"Site sweep complete: {len(sites)} sites, prob range [{min(probs):.3f}, {max(probs):.3f}]")
    return {"sites": sites, "probs": probs}


def sweep_sex(
    model,
    image: torch.Tensor,
    base_cont: np.ndarray,
    site_idx: int,
    encoder: MetadataEncoder,
    device: torch.device,
    model_type: str = "cross_attention",
) -> Dict[str, list]:
    """
    Sweeps sex across male / female / unknown.

    Returns:
        {
            'sexes': [...],
            'probs': [...],
        }
    """
    sexes = SEX_CATEGORIES
    probs = []

    for sex_label in sexes:
        cont = base_cont.copy()
        # Indices 1,2,3 are sex one-hot: male=1, female=2, unknown=3
        cont[1:4] = 0.0
        sex_idx = SEX_CATEGORIES.index(sex_label)
        cont[1 + sex_idx] = 1.0
        prob = _predict_prob(model, image, cont, site_idx, device, model_type)
        probs.append(prob)

    log.info(f"Sex sweep complete: probs={dict(zip(sexes, [f'{p:.3f}' for p in probs]))}")
    return {"sexes": sexes, "probs": probs}


def run_full_counterfactual(
    model,
    image: torch.Tensor,       # (1, 3, H, W)
    base_cont: np.ndarray,     # baseline continuous metadata
    base_site_idx: int,        # baseline site index
    encoder: MetadataEncoder,
    device: torch.device,
    model_type: str = "cross_attention",
    age_range: Tuple[int, int] = (10, 90),
    age_step: int = 5,
) -> Dict:
    """
    Runs all three sweeps for one test case.

    Returns a dict with keys:
        - 'base_prob': baseline prediction
        - 'age_sweep': {'ages': [...], 'probs': [...]}
        - 'site_sweep': {'sites': [...], 'probs': [...]}
        - 'sex_sweep': {'sexes': [...], 'probs': [...]}
    """
    base_prob = _predict_prob(model, image, base_cont, base_site_idx, device, model_type)
    log.info(f"Base prediction: {base_prob:.3f}")

    age_sweep = sweep_age(model, image, base_cont, base_site_idx, encoder, device, model_type, age_range, age_step)
    site_sweep = sweep_site(model, image, base_cont, encoder, device, model_type)
    sex_sweep = sweep_sex(model, image, base_cont, base_site_idx, encoder, device, model_type)

    return {
        "base_prob": base_prob,
        "age_sweep": age_sweep,
        "site_sweep": site_sweep,
        "sex_sweep": sex_sweep,
    }
