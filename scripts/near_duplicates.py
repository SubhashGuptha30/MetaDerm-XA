"""
Near-duplicate screen between ISIC 2019 (train/val/calib/test) and ISIC 2020 (external), and between
ISIC 2019 dev and ISIC 2019 test, using perceptual hashes (pHash, 64 bit) on the resized images.

Pairs with Hamming distance <= --max_dist are written to reports/v2/near_duplicates.csv for visual review.

    .venv/Scripts/python scripts/near_duplicates.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/kaggle_upload/metaderm-images")


def phash(path):
    return int(str(imagehash.phash(Image.open(path).convert("L"))), 16)


def hashes(split):
    df = pd.read_csv(f"splits/v2/{split}.csv")
    paths = [ROOT / d / f"{n}.jpg" for d, n in zip(df.img_dir, df.image_name)]
    with ProcessPoolExecutor() as ex:
        h = list(ex.map(phash, paths, chunksize=256))
    return df.image_name.values, np.array(h, dtype=np.uint64)


def popcount(x):
    x = x.copy()
    c = np.zeros(x.shape, dtype=np.uint8)
    while x.any():
        c += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return c


def compare(na, ha, nb, hb, max_dist, tag):
    rows = []
    for i in range(0, len(ha), 256):
        d = popcount(ha[i:i + 256, None] ^ hb[None, :])
        for a, b in zip(*np.nonzero(d <= max_dist)):
            rows.append({"pair": tag, "a": na[i + a], "b": nb[b], "hamming": int(d[a, b])})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_dist", type=int, default=6)
    a = ap.parse_args()
    H = {s: hashes(s) for s in ["train", "val", "calib", "test", "external"]}
    dev_n = np.concatenate([H[s][0] for s in ["train", "val", "calib"]])
    dev_h = np.concatenate([H[s][1] for s in ["train", "val", "calib"]])
    rows = compare(dev_n, dev_h, *H["test"], a.max_dist, "dev2019-test2019")
    rows += compare(dev_n, dev_h, *H["external"], a.max_dist, "dev2019-isic2020")
    rows += compare(*H["test"], *H["external"], a.max_dist, "test2019-isic2020")
    rows += compare(*H["train"], *H["val"], a.max_dist, "train-val")
    rows += compare(*H["train"], *H["calib"], a.max_dist, "train-calib")
    out = pd.DataFrame(rows, columns=["pair", "a", "b", "hamming"])
    Path("reports/v2").mkdir(parents=True, exist_ok=True)
    out.to_csv("reports/v2/near_duplicates.csv", index=False)
    print(out.groupby(["pair", "hamming"]).size().unstack(fill_value=0) if len(out) else "no near-duplicates")

    # Verify candidates: pHash alone flags look-alikes (black vignettes dominate). Thresholds chosen by
    # visual review of contact sheets (reports/v2/near_dup_*.png); deliberately conservative, since a
    # false positive only merges two dev groups or drops one image from evaluation.
    loc = {}
    for s in H:
        m = pd.read_csv(f"splits/v2/{s}.csv")
        loc.update(dict(zip(m.image_name, m.img_dir)))

    def vec(n, centre):
        im = Image.open(ROOT / loc[n] / f"{n}.jpg").convert("RGB" if centre else "L")
        if centre:
            w, h = im.size
            im = im.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        x = np.asarray(im.resize((48, 48) if centre else (32, 32)), float).ravel()
        return (x - x.mean()) / (x.std() + 1e-6)

    c = out[out.hamming <= 4].copy()
    c["corr_gray"] = [float(np.mean(vec(x, False) * vec(y, False))) for x, y in zip(c.a, c.b)]
    c = c[c.corr_gray >= 0.95].copy()
    c["corr_centre"] = [float(np.mean(vec(x, True) * vec(y, True))) for x, y in zip(c.a, c.b)]
    dup = c[c.corr_centre >= 0.90]
    dup.to_csv("splits/v2/duplicate_pairs.csv", index=False)
    print(f"\nconfirmed duplicate / same-lesion pairs: {len(dup)}\n{dup.pair.value_counts()}")
