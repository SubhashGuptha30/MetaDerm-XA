"""
Metadata encoder — Phase 3.

Fits and applies transformations for:
  - age_approx: z-score normalization
  - sex: one-hot encoding (male / female / unknown)
  - anatom_site_general_challenge: integer label encoding (for embedding lookup)

Saves/loads encoder state so train-fit transforms are applied consistently to val/test.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SEX_CATEGORIES = ["male", "female", "unknown"]

SITE_CATEGORIES = [
    "head/neck",
    "upper extremity",
    "lower extremity",
    "torso",
    "palms/soles",
    "oral/genital",
    "unknown",
]

METADATA_DIM = 1 + 3 + 1  # age (1) + sex one-hot (3) + age_missing (1)
# Site is handled via an embedding layer in the model, so we only pass the index here.
METADATA_TOTAL_DIM = METADATA_DIM  # age(1) + sex_onehot(3) + age_missing(1) = 5


class MetadataEncoder:
    """
    Fits on training data; transforms train/val/test consistently.
    """

    def __init__(self) -> None:
        self.age_mean: float = 0.0
        self.age_std: float = 1.0
        self.sex_categories: List[str] = SEX_CATEGORIES
        self.site_categories: List[str] = SITE_CATEGORIES
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        self.age_mean = float(df["age_approx"].mean())
        self.age_std = float(df["age_approx"].std()) or 1.0
        log.info(
            f"MetadataEncoder fitted — age mean={self.age_mean:.1f}, std={self.age_std:.1f}"
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            continuous_features: np.ndarray shape (N, 5)
                [age_norm, sex_male, sex_female, sex_unknown, age_missing]
            site_indices: np.ndarray shape (N,) — integer index for embedding lookup
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        n = len(df)

        # Age z-score
        age_norm = (df["age_approx"].values - self.age_mean) / self.age_std

        # Age missing flag
        age_missing = df["age_missing"].values if "age_missing" in df.columns else np.zeros(n)

        # Sex one-hot
        sex = df["sex"].str.lower().fillna("unknown").values
        sex_onehot = np.zeros((n, 3), dtype=np.float32)
        for i, cat in enumerate(self.sex_categories):
            sex_onehot[:, i] = (sex == cat).astype(np.float32)

        # Site label index
        site = df["anatom_site_general_challenge"].str.lower().fillna("unknown").values
        site_indices = np.array(
            [
                self.site_categories.index(s)
                if s in self.site_categories
                else self.site_categories.index("unknown")
                for s in site
            ],
            dtype=np.int64,
        )

        continuous = np.column_stack([age_norm, sex_onehot, age_missing]).astype(np.float32)

        return continuous, site_indices

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "sex_categories": self.sex_categories,
            "site_categories": self.site_categories,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        log.info(f"MetadataEncoder saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "MetadataEncoder":
        with open(path) as f:
            state = json.load(f)
        enc = cls()
        enc.age_mean = state["age_mean"]
        enc.age_std = state["age_std"]
        enc.sex_categories = state["sex_categories"]
        enc.site_categories = state["site_categories"]
        enc._fitted = True
        log.info(f"MetadataEncoder loaded from {path}")
        return enc

    @property
    def num_sites(self) -> int:
        return len(self.site_categories)

    @property
    def continuous_dim(self) -> int:
        """Dimension of the continuous metadata feature vector (excl. site embedding)."""
        return 5  # age + sex(3) + age_missing
