"""
Resize all ISIC images once, locally, for upload as a private Kaggle dataset.

Shorter side -> --short_side px (aspect ratio kept), JPEG quality 92.
Uses PIL draft mode so 6000x4000 ISIC 2020 JPEGs decode quickly.

Output layout (matches the `img_dir` column written by src/data/preprocess_v2.py):
    <out>/isic2019_train/<image>.jpg
    <out>/isic2019_test/<image>.jpg
    <out>/isic2020/<image>.jpg

Usage:
    python scripts/prepare_kaggle_images.py --out data/kaggle_upload/metaderm-images
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

SOURCES = {
    "isic2019_train": "data/raw/isic2019/train/images",
    "isic2019_test": "data/raw/isic2019/test/images",
    "isic2020": "data/raw/isic2020/images",
    "pad_ufes20": "data/raw/PAD-UFES-20/images/unz",  # PNGs in imgs_part_*/ subfolders (unzipped)
}


def _resize_one(args):
    src, dst, short_side = args
    if os.path.exists(dst):
        return 0
    try:
        img = Image.open(src)
        img.draft("RGB", (short_side * 2, short_side * 2))  # fast DCT downscale, still >= target
        img = img.convert("RGB")
        w, h = img.size
        s = short_side / min(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        img.save(dst, "JPEG", quality=92)
        return 1
    except Exception as e:  # report, don't crash the pool
        print(f"FAILED {src}: {e}")
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/kaggle_upload/metaderm-images")
    p.add_argument("--short_side", type=int, default=320)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--kaggle_user", default=None, help="writes dataset-metadata.json if given")
    a = p.parse_args()

    out = Path(a.out)
    jobs = []
    for sub, src_dir in SOURCES.items():
        (out / sub).mkdir(parents=True, exist_ok=True)
        for d, _, files in os.walk(src_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    stem = os.path.splitext(f)[0]
                    jobs.append((os.path.join(d, f), str(out / sub / f"{stem}.jpg"), a.short_side))
    print(f"{len(jobs)} images -> {out} (short side {a.short_side}px, {a.workers} workers)")

    done = failed = 0
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=64)):
            done += r == 1
            failed += r == -1
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"resized {done}, failed {failed}, skipped {len(jobs) - done - failed}")

    if a.kaggle_user:
        meta = {
            "title": "metaderm-images",
            "id": f"{a.kaggle_user}/metaderm-images",
            "licenses": [{"name": "CC-BY-NC-4.0"}],
        }
        (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
