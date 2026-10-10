"""Load per-image logits / probe outputs saved by Kaggle runs (runs_v2/<model>/seed<s>/...)."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

RUNS = Path("runs_v2")
SPLITS = Path("splits/v2")

MAIN_MODELS = ["baseline", "baseline_wide", "concat_fusion", "cross_attention", "film", "field_token"]
ABLATION_MODELS = ["field_token_k1", "field_token_k2", "field_token_k3", "field_token"]
LABEL = {
    "metadata_only": "Metadata-only (LR)",
    "baseline": "Image-only",
    "baseline_wide": "Image-only (wide)",
    "concat_fusion": "Concat",
    "cross_attention": "Cross-attn (1 token)",
    "film": "FiLM",
    "field_token": "Field-token",
    "field_token_k1": "Field-token K=1",
    "field_token_k2": "Field-token K=2",
    "field_token_k3": "Field-token K=3",
}


INCLUDE_DUPLICATES = False  # primary analysis drops test/external images that duplicate a dev image


def _keep(split: str) -> np.ndarray:
    m = pd.read_csv(SPLITS / f"{split}.csv")
    if INCLUDE_DUPLICATES or "dup_in_dev" not in m:
        return np.ones(len(m), bool)
    return (m["dup_in_dev"] == 0).values


def manifest(split: str) -> pd.DataFrame:
    m = pd.read_csv(SPLITS / f"{split}.csv")
    return m[_keep(split)].reset_index(drop=True)


def seeds(model: str):
    return sorted(int(p.name[4:]) for p in (RUNS / model).glob("seed*") if (p / "summary.json").exists())


def logits(model: str, split: str) -> Dict[int, np.ndarray]:
    """{seed: logits aligned to the split manifest order}."""
    ref = pd.read_csv(SPLITS / f"{split}.csv")["image_name"].values
    keep = _keep(split)
    out = {}
    for s in seeds(model):
        df = pd.read_csv(RUNS / model / f"seed{s}" / "preds" / f"{split}.csv.gz")
        assert (df["image_name"].values == ref).all(), f"{model} seed{s} {split}: order mismatch"
        out[s] = df["logit"].values.astype(np.float64)[keep]
    return out


def logit_matrix(model: str, split: str, seed_list=None) -> np.ndarray:
    d = logits(model, split)
    keys = seed_list or sorted(d)
    return np.vstack([d[k] for k in keys])


def probes(model: str, seed: int, split: str = "test"):
    p = RUNS / model / f"seed{seed}" / "probes" / f"{split}.npz"
    if not p.exists():
        return None
    z, keep = np.load(p, allow_pickle=False), _keep(split)
    n = len(keep)
    return {k: (z[k][keep] if z[k].shape[:1] == (n,) else z[k]) for k in z.files}
