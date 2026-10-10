"""
Preprocess ISIC 2019 for the skin-lesion pipeline.

Converts ISIC 2019 multi-class one-hot format into the same schema used by
the existing ISIC 2020 pipeline:
    image_name, target, age_approx, sex, anatom_site_general_challenge,
    patient_id, age_missing, split

Produces:
    data/processed_isic2019/train.csv
    data/processed_isic2019/val.csv
    data/processed_isic2019/test.csv          (ISIC 2019 official test set)
    data/processed_isic2019/external_test.csv (ISIC 2020 for cross-dataset eval)

Usage:
    python src/data/preprocess_isic2019.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ISIC 2019 uses finer-grained site names; map to ISIC 2020 categories
SITE_MAP = {
    "anterior torso": "torso",
    "lateral torso": "torso",
    "posterior torso": "torso",
    "head/neck": "head/neck",
    "upper extremity": "upper extremity",
    "lower extremity": "lower extremity",
    "palms/soles": "palms/soles",
    "oral/genital": "oral/genital",
}


def load_isic2019_split(
    ground_truth_path: str,
    metadata_path: str,
) -> pd.DataFrame:
    """Merge ground truth + metadata into a unified DataFrame."""
    gt = pd.read_csv(ground_truth_path)
    meta = pd.read_csv(metadata_path)

    # Merge on image ID
    df = meta.merge(gt, on="image", how="inner")

    # Binary target: MEL = 1, everything else = 0
    df["target"] = df["MEL"].astype(int)

    # Rename columns to match ISIC 2020 pipeline
    df = df.rename(columns={"image": "image_name"})

    # Map site names to ISIC 2020 categories
    site_col = "anatom_site_general"
    if site_col in df.columns:
        df["anatom_site_general_challenge"] = (
            df[site_col]
            .str.lower()
            .fillna("unknown")
            .map(lambda s: SITE_MAP.get(s, "unknown"))
        )
    else:
        df["anatom_site_general_challenge"] = "unknown"

    # Patient ID: use lesion_id if available, else image_name
    if "lesion_id" in df.columns:
        df["patient_id"] = df["lesion_id"].fillna(df["image_name"])
    else:
        df["patient_id"] = df["image_name"]

    # Sex: lowercase, fill missing
    df["sex"] = df["sex"].fillna("unknown").str.lower()

    # Age: add missing flag, impute with median
    df["age_missing"] = df["age_approx"].isna().astype(int)
    median_age = df["age_approx"].median()
    df["age_approx"] = df["age_approx"].fillna(median_age)

    # Keep only columns the pipeline expects
    keep_cols = [
        "image_name", "target", "age_approx", "sex",
        "anatom_site_general_challenge", "patient_id", "age_missing",
    ]
    # Also keep the multi-class columns for potential multi-class experiments
    class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    existing_class_cols = [c for c in class_cols if c in df.columns]
    df = df[keep_cols + existing_class_cols]

    log.info(
        f"Loaded {len(df)} images, "
        f"{df['target'].sum():.0f} melanoma ({100*df['target'].mean():.1f}%)"
    )
    return df


def patient_level_train_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Patient-level stratified train/val split.
    Uses lesion_id as patient proxy (no patient appears in both splits).
    """
    df = df.copy()
    patient_col = "patient_id"

    # Build patient-level label
    patient_labels = (
        df.groupby(patient_col)["target"]
        .agg(lambda x: int(x.mean() >= 0.5))
        .reset_index()
    )
    patient_labels.columns = [patient_col, "patient_label"]
    patients = patient_labels[patient_col].values
    labels = patient_labels["patient_label"].values

    n_splits = max(2, round(1 / val_frac))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(patients, labels, patients))
    val_patients = set(patients[val_idx])

    df["split"] = df[patient_col].apply(
        lambda p: "val" if p in val_patients else "train"
    )

    # Verify no leakage
    train_p = set(df.loc[df["split"] == "train", patient_col])
    val_p = set(df.loc[df["split"] == "val", patient_col])
    assert len(train_p & val_p) == 0, "LEAKAGE: train & val share patients!"

    for split in ["train", "val"]:
        subset = df[df["split"] == split]
        mel = subset["target"].sum()
        log.info(
            f"  {split}: {len(subset)} images, {int(mel)} melanoma "
            f"({100*mel/len(subset):.1f}%)"
        )

    return df


def prepare_isic2020_external(csv_path: str) -> pd.DataFrame:
    """
    Prepare full ISIC 2020 dataset as an external test set.
    Applies the same column formatting and imputation.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={})  # columns already match

    # Ensure required columns exist
    if "image_name" not in df.columns:
        raise ValueError("ISIC 2020 CSV missing 'image_name' column")

    df["sex"] = df["sex"].fillna("unknown").str.lower()
    df["anatom_site_general_challenge"] = (
        df["anatom_site_general_challenge"].fillna("unknown").str.lower()
    )
    df["age_missing"] = df["age_approx"].isna().astype(int)
    median_age = df["age_approx"].median()
    df["age_approx"] = df["age_approx"].fillna(median_age)

    if "patient_id" not in df.columns:
        df["patient_id"] = df["image_name"]

    keep_cols = [
        "image_name", "target", "age_approx", "sex",
        "anatom_site_general_challenge", "patient_id", "age_missing",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    log.info(
        f"ISIC 2020 external test: {len(df)} images, "
        f"{df['target'].sum():.0f} melanoma ({100*df['target'].mean():.1f}%)"
    )
    return df


def run(
    seed: int = 42,
    val_frac: float = 0.20,
    output_dir: str = "data/processed_isic2019",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- ISIC 2019 Training → train/val ---
    log.info("=== Processing ISIC 2019 Training Data ===")
    train_df = load_isic2019_split(
        ground_truth_path="data/raw/isic2019/train/ISIC_2019_Training_GroundTruth.csv",
        metadata_path="data/raw/isic2019/train/ISIC_2019_Training_Metadata.csv",
    )
    train_df = patient_level_train_val_split(train_df, val_frac=val_frac, seed=seed)

    train_split = train_df[train_df["split"] == "train"].drop(columns=["split"])
    val_split = train_df[train_df["split"] == "val"].drop(columns=["split"])
    train_split.to_csv(output_dir / "train.csv", index=False)
    val_split.to_csv(output_dir / "val.csv", index=False)

    # --- ISIC 2019 Test → test ---
    log.info("\n=== Processing ISIC 2019 Test Data ===")
    test_df = load_isic2019_split(
        ground_truth_path="data/raw/isic2019/test/ISIC_2019_Test_GroundTruth.csv",
        metadata_path="data/raw/isic2019/test/ISIC_2019_Test_Metadata.csv",
    )
    test_df.to_csv(output_dir / "test.csv", index=False)

    # --- ISIC 2020 → external test ---
    log.info("\n=== Processing ISIC 2020 External Validation ===")
    ext_df = prepare_isic2020_external("data/raw/isic2020/train.csv")
    ext_df.to_csv(output_dir / "external_test.csv", index=False)

    # --- Summary ---
    log.info("\n=== FINAL SUMMARY ===")
    log.info(f"Train:         {len(train_split):>6d} images, {int(train_split['target'].sum()):>5d} MEL")
    log.info(f"Val:           {len(val_split):>6d} images, {int(val_split['target'].sum()):>5d} MEL")
    log.info(f"Test (2019):   {len(test_df):>6d} images, {int(test_df['target'].sum()):>5d} MEL")
    log.info(f"External(2020):{len(ext_df):>6d} images, {int(ext_df['target'].sum()):>5d} MEL")
    log.info(f"\nAll files saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--output_dir", default="data/processed_isic2019")
    args = parser.parse_args()
    run(seed=args.seed, val_frac=args.val_frac, output_dir=args.output_dir)
