"""
Phase 3 — Patient-level stratified split.

Produces:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/splits.csv   (full CSV with a 'split' column)

Run:
    python src/data/preprocess.py --config configs/baseline.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def impute_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing metadata values:
    - age_approx: filled with column median; adds 'age_missing' flag
    - sex: missing filled with 'unknown'
    - anatom_site_general_challenge: missing filled with 'unknown'
    """
    df = df.copy()

    # Age
    df["age_missing"] = df["age_approx"].isna().astype(int)
    median_age = df["age_approx"].median()
    df["age_approx"] = df["age_approx"].fillna(median_age)

    # Sex
    df["sex"] = df["sex"].fillna("unknown").str.lower()

    # Anatomical site
    df["anatom_site_general_challenge"] = (
        df["anatom_site_general_challenge"].fillna("unknown").str.lower()
    )

    log.info(
        f"Imputed metadata — age median: {median_age:.1f}, "
        f"age_missing: {df['age_missing'].sum()}"
    )
    return df


def patient_level_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified, patient-level train/val/test split.
    No patient appears in more than one split.

    Uses StratifiedGroupKFold to ensure:
    1. Class balance preserved across splits.
    2. All images of one patient stay in the same split.
    """
    df = df.copy()
    df["split"] = "train"

    patient_col = "patient_id"
    if patient_col not in df.columns:
        log.warning(
            "No 'patient_id' column found — falling back to 'image_name' as patient proxy."
        )
        df[patient_col] = df["image_name"]

    # Build patient-level label (majority class per patient)
    patient_labels = (
        df.groupby(patient_col)["target"].agg(lambda x: int(x.mean() >= 0.5)).reset_index()
    )
    patient_labels.columns = [patient_col, "patient_label"]
    patients = patient_labels[patient_col].values
    labels = patient_labels["patient_label"].values

    n_splits_val = round(1 / val_frac)  # e.g. 1/0.15 ~ 7 → use 1 fold as val
    n_splits_test = round(1 / test_frac)

    rng = np.random.default_rng(seed)

    # First: carve out test set
    splitter = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(patients, labels, patients))
    test_patients = set(patients[test_idx])

    # Second: from remaining patients, carve out val set
    remaining_patients = patients[train_val_idx]
    remaining_labels = labels[train_val_idx]
    val_splitter = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    train_idx2, val_idx2 = next(
        val_splitter.split(remaining_patients, remaining_labels, remaining_patients)
    )
    val_patients = set(remaining_patients[val_idx2])

    # Apply split labels to full dataframe
    df["split"] = df[patient_col].apply(
        lambda p: "test" if p in test_patients else ("val" if p in val_patients else "train")
    )

    counts = df["split"].value_counts()
    log.info(f"Split counts (images): {dict(counts)}")

    # Verify no leakage
    train_patients = set(df.loc[df["split"] == "train", patient_col])
    val_p = set(df.loc[df["split"] == "val", patient_col])
    test_p = set(df.loc[df["split"] == "test", patient_col])
    assert len(train_patients & val_p) == 0, "LEAKAGE: train ∩ val"
    assert len(train_patients & test_p) == 0, "LEAKAGE: train ∩ test"
    assert len(val_p & test_p) == 0, "LEAKAGE: val ∩ test"
    log.info("✓ No patient-level leakage confirmed.")

    return df


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    csv_path = Path(cfg.get("csv_path", "data/raw/isic2020/train.csv"))
    seed = cfg.get("seed", 42)

    log.info(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} rows.")

    df = impute_metadata(df)
    df = patient_level_split(
        df,
        val_frac=cfg.get("val_frac", 0.15),
        test_frac=cfg.get("test_frac", 0.15),
        seed=seed,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "splits.csv", index=False)

    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        out_path = PROCESSED_DIR / f"{split}.csv"
        subset.to_csv(out_path, index=False)
        mal = subset["target"].sum()
        log.info(
            f"{split}: {len(subset)} images, {int(mal)} malignant "
            f"({100*mal/len(subset):.1f}%)"
        )

    log.info("✓ Preprocessing complete. Splits saved to data/processed/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    run(args.config)
