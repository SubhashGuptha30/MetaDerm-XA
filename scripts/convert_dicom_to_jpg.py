"""
Utility script to convert ISIC DICOM files to JPEG.
Reading DICOM files on-the-fly during PyTorch training creates a massive CPU bottleneck.
This script pre-converts them to JPEG to keep training fast.

Usage:
    python scripts/convert_dicom_to_jpg.py --input_dir /path/to/dicom --output_dir data/raw/isic2020/images
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
except ImportError:
    print("Error: pydicom is required. Run: pip install pydicom")
    exit(1)


def process_dicom(args_tuple):
    input_path, output_path = args_tuple
    try:
        if output_path.exists():
            return True  # Already processed

        dicom = pydicom.dcmread(str(input_path))
        
        # Extract pixel data
        img = dicom.pixel_array

        # Handle Photometric Interpretation
        pi = dicom.PhotometricInterpretation
        if pi == "MONOCHROME1":
            img = np.amax(img) - img
        
        # Normalize to 0-255 uint8
        img = img - np.min(img)
        img = img / np.max(img)
        img = (img * 255).astype(np.uint8)

        # Convert to PIL and save
        pil_img = Image.fromarray(img)
        
        # Convert YBR to RGB if necessary (pydicom sometimes loads color as YBR)
        if pi in ["YBR_FULL", "YBR_FULL_422"]:
            pil_img = pil_img.convert("RGB")
            
        pil_img.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"\nFailed to process {input_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .dcm files")
    parser.add_argument("--output_dir", type=str, default="data/raw/isic2020/images", help="Output directory for .jpg files")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU cores to use")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_files = list(input_dir.glob("*.dcm"))
    if not dicom_files:
        print(f"No .dcm files found in {input_dir}")
        return

    print(f"Found {len(dicom_files)} DICOM files. Converting to JPEG...")

    # Prepare arguments for multiprocessing
    tasks = []
    for dcm_path in dicom_files:
        jpg_path = output_dir / f"{dcm_path.stem}.jpg"
        tasks.append((dcm_path, jpg_path))

    # Process in parallel
    success_count = 0
    with Pool(args.workers) as pool:
        for success in tqdm(pool.imap_unordered(process_dicom, tasks), total=len(tasks)):
            if success:
                success_count += 1

    print(f"\nConversion complete: {success_count}/{len(dicom_files)} successful.")
    print(f"Images saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
