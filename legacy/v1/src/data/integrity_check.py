"""
Data integrity checker — Phase 1.
Verifies image count vs CSV rows, checks for corrupted images.
Run: python src/data/integrity_check.py
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


def check_integrity(
    csv_path: Path,
    images_dir: Path,
    output_log: Path = REPORTS_DIR / "data_integrity_log.txt",
) -> dict:
    """
    Checks:
    1. CSV exists and is readable.
    2. Every image referenced in CSV exists on disk.
    3. Every image on disk can be opened without error.

    Returns a summary dict.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)

    issues = []
    summary = {}

    # --- Load CSV ---
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
    summary["csv_rows"] = len(df)

    # --- Check required columns ---
    required_cols = {"image_name", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # --- Check images referenced in CSV ---
    log.info("Checking CSV → disk references …")
    missing_files = []
    for img_name in tqdm(df["image_name"], desc="CSV→disk"):
        img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = images_dir / img_name  # try without extension
        if not img_path.exists():
            missing_files.append(img_name)

    summary["missing_files"] = len(missing_files)
    if missing_files:
        issues.append(f"{len(missing_files)} images in CSV but missing on disk.")
        for f in missing_files[:20]:
            issues.append(f"  MISSING: {f}")

    # --- Verify image readability ---
    log.info("Verifying image readability …")
    corrupt = []
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    summary["disk_image_count"] = len(image_files)

    for img_path in tqdm(image_files, desc="Integrity"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            corrupt.append((img_path.name, str(e)))

    summary["corrupt_files"] = len(corrupt)
    if corrupt:
        issues.append(f"{len(corrupt)} corrupt/unreadable image files found.")
        for name, err in corrupt[:20]:
            issues.append(f"  CORRUPT: {name} — {err}")

    # --- Class distribution ---
    if "target" in df.columns:
        vc = df["target"].value_counts()
        summary["malignant_count"] = int(vc.get(1, 0))
        summary["benign_count"] = int(vc.get(0, 0))
        mal_pct = 100 * summary["malignant_count"] / len(df)
        log.info(
            f"Class balance — malignant: {summary['malignant_count']} ({mal_pct:.1f}%), "
            f"benign: {summary['benign_count']} ({100 - mal_pct:.1f}%)"
        )

    # --- Write log ---
    with open(output_log, "w") as f:
        f.write("=== ISIC 2020 Data Integrity Report ===\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\n--- Issues ---\n")
        if issues:
            for issue in issues:
                f.write(issue + "\n")
        else:
            f.write("No issues found. Data looks clean.\n")

    log.info(f"Integrity report saved to {output_log}")

    if issues:
        log.warning(f"{len(issues)} issues found — see {output_log}")
    else:
        log.info("✓ All integrity checks passed.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2020 data integrity checker")
    parser.add_argument(
        "--csv",
        default="data/raw/isic2020/train.csv",
        help="Path to train.csv",
    )
    parser.add_argument(
        "--images_dir",
        default="data/raw/isic2020/images",
        help="Path to images directory",
    )
    args = parser.parse_args()

    summary = check_integrity(
        csv_path=Path(args.csv),
        images_dir=Path(args.images_dir),
    )
    print("\nSummary:", summary)
