"""
DataLoader factory (v2).

`get_dataloaders(cfg, splits)` reads split manifests from cfg['split_dir'] and images from
cfg['image_root'] (env var METADERM_IMAGE_ROOT overrides, so the same config runs locally
and on Kaggle).

v2 fixes:
  * AUDIT A7: imbalance handling is ONE explicit choice, cfg['imbalance'] ∈
    {'none', 'pos_weight', 'sampler'}; v1 silently combined a balanced sampler with
    pos_weight=5 (≈23x effective up-weighting).
  * AUDIT C3: workers and the shuffling generator are seeded.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import ISICDataset
from src.data.metadata_encoder import MetadataEncoder
from src.data.transforms import get_train_transforms, get_val_transforms

log = logging.getLogger(__name__)

ALL_SPLITS = ("train", "val", "calib", "test", "external", "pad")


def image_root(cfg: dict) -> Path:
    return Path(os.environ.get("METADERM_IMAGE_ROOT", cfg.get("image_root", "data/kaggle_upload/metaderm-images")))


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    df = pd.read_csv(Path(cfg.get("split_dir", "splits/v2")) / f"{split}.csv")
    frac = cfg.get("debug_fraction")  # smoke tests only
    if frac:
        idx = [g.sample(frac=frac, random_state=0).index for _, g in df.groupby("target")]
        df = df.loc[sorted(np.concatenate(idx))].reset_index(drop=True)
    return df


def fit_encoder(cfg: dict) -> MetadataEncoder:
    return MetadataEncoder().fit(load_split(cfg, "train"))


def get_dataloaders(cfg: dict, splits: Iterable[str] = ALL_SPLITS) -> Dict[str, DataLoader]:
    root = image_root(cfg)
    img_size = cfg.get("img_size", 224)
    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 4)
    seed = cfg.get("seed", 42)

    encoder = fit_encoder(cfg)
    loaders = {}
    for split in splits:
        df = load_split(cfg, split)
        cont, sites = encoder.transform(df)
        train = split == "train"
        ds = ISICDataset(df, root, cont, sites,
                         get_train_transforms(img_size) if train else get_val_transforms(img_size))
        g = torch.Generator().manual_seed(seed)
        sampler = None
        if train and cfg.get("imbalance", "none") == "sampler":
            y = df["target"].values.astype(int)
            w = (1.0 / np.bincount(y))[y]
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w),
                                            replacement=True, generator=g)
        loaders[split] = DataLoader(
            ds,
            batch_size=bs if train else bs * 2,
            shuffle=train and sampler is None,
            sampler=sampler,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=train,
            worker_init_fn=_seed_worker,
            generator=g,
            persistent_workers=nw > 0 and train,
        )
        log.info(f"{split}: {len(ds)} images, {int(df['target'].sum())} positive")
    return loaders


def pos_weight_value(cfg: dict) -> float:
    """'auto' → n_neg/n_pos on the training split; otherwise the configured number."""
    pw = cfg.get("pos_weight", "auto")
    if pw == "auto":
        y = load_split(cfg, "train")["target"].values
        return float((y == 0).sum() / max(1, (y == 1).sum()))
    return float(pw)
