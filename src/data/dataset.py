"""
ISIC dataset. Returns (image, metadata_continuous, site_index, label, image_name).

Image path = <image_root>/<img_dir>/<image_name>.jpg, where `img_dir` comes from the
split manifest (isic2019_train / isic2019_test / isic2020).
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Path,
        metadata_continuous: np.ndarray,
        site_indices: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.metadata_continuous = metadata_continuous.astype(np.float32)
        self.site_indices = site_indices.astype(np.int64)
        self.transform = transform
        assert len(self.metadata_continuous) == len(self.df) == len(self.site_indices)
        self._paths = [
            self.image_root / d / f"{n}.jpg" for d, n in zip(self.df["img_dir"], self.df["image_name"])
        ]
        self._labels = self.df["target"].astype(np.float32).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self._paths[idx]).convert("RGB")
        img = self.transform(img) if self.transform is not None else img
        return (
            img,
            torch.from_numpy(self.metadata_continuous[idx]),
            torch.tensor(self.site_indices[idx], dtype=torch.long),
            torch.tensor(self._labels[idx]),
            self.df.at[idx, "image_name"],
        )
