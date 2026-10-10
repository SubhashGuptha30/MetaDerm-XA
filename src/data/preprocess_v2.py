"""
Preprocessing v2 — builds the frozen split manifests used by every experiment.

Fixes vs. v1 (see docs/claude/AUDIT.md):
  * B1: missing age is imputed with the TRAINING-split median everywhere
        (v1 used each split's own median, leaking test statistics).
  * B4: the ISIC 2019 training set is split three ways, grouped by lesion_id:
        train (early stopping NOT done here) / val (early stopping + model selection)
        / calib (decision threshold + post-hoc calibration). No set is reused for two roles.

Outputs (small CSVs, tracked in git under splits/):
    splits/v2/{train,val,calib,test,external}.csv
    splits/v2/preprocess_info.json

Every CSV has: image_name, img_dir, target, age_approx, age_missing, sex,
anatom_site_general_challenge, group_id (+ diagnosis columns where available).

Usage:
    python -m src.data.preprocess_v2
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW = Path("data/raw")

SITE_MAP = {
    "anterior torso": "torso",
    "lateral torso": "torso",
    "posterior torso": "torso",
    "torso": "torso",
    "head/neck": "head/neck",
    "upper extremity": "upper extremity",
    "lower extremity": "lower extremity",
    "palms/soles": "palms/soles",
    "oral/genital": "oral/genital",
}
DX_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
KEEP = ["image_name", "img_dir", "target", "age_approx", "age_missing", "sex",
        "anatom_site_general_challenge", "group_id"]


def _harmonise(df: pd.DataFrame, site_col: str) -> pd.DataFrame:
    df["sex"] = df["sex"].fillna("unknown").str.lower()
    df["anatom_site_general_challenge"] = (
        df[site_col].str.lower().map(SITE_MAP).fillna("unknown")
    )
    df["age_missing"] = df["age_approx"].isna().astype(int)
    return df


def load_isic2019(split: str) -> pd.DataFrame:
    name = "Training" if split == "train" else "Test"
    gt = pd.read_csv(RAW / f"isic2019/{split}/ISIC_2019_{name}_GroundTruth.csv")
    meta = pd.read_csv(RAW / f"isic2019/{split}/ISIC_2019_{name}_Metadata.csv")
    df = meta.merge(gt, on="image", how="inner", validate="one_to_one")
    assert len(df) == len(gt) == len(meta), "ISIC 2019 GT/metadata mismatch"
    df = df.rename(columns={"image": "image_name"})
    df["target"] = df["MEL"].astype(int)
    df["img_dir"] = f"isic2019_{split}"
    if "lesion_id" in df.columns:
        df["group_id"] = df["lesion_id"].fillna(df["image_name"])
    else:
        df["group_id"] = df["image_name"]
    df = _harmonise(df, "anatom_site_general")
    extra = [c for c in DX_COLS + ["score_weight", "validation_weight"] if c in df.columns]
    return df[KEEP + extra]


def load_isic2020() -> pd.DataFrame:
    df = pd.read_csv(RAW / "isic2020/train.csv")
    df["img_dir"] = "isic2020"
    df["group_id"] = df["patient_id"]  # real patient IDs exist for 2020 → clustered bootstrap
    df = _harmonise(df, "anatom_site_general_challenge")
    return df[KEEP + ["diagnosis"]]


# PAD-UFES-20 body regions → the 7 ISIC site categories (FOOT/HAND are not necessarily acral → limbs)
PAD_REGION_MAP = {
    "FACE": "head/neck", "NOSE": "head/neck", "EAR": "head/neck", "LIP": "head/neck",
    "SCALP": "head/neck", "NECK": "head/neck",
    "FOREARM": "upper extremity", "ARM": "upper extremity", "HAND": "upper extremity",
    "CHEST": "torso", "BACK": "torso", "ABDOMEN": "torso",
    "THIGH": "lower extremity", "FOOT": "lower extremity",
}


def load_pad_ufes() -> pd.DataFrame:
    """PAD-UFES-20 (smartphone clinical photos): second external set, never used for training/tuning."""
    df = pd.read_csv(RAW / "PAD-UFES-20/metadata.csv")
    df["image_name"] = df["img_id"].str.replace(".png", "", regex=False)
    df["img_dir"] = "pad_ufes20"
    df["target"] = (df["diagnostic"] == "MEL").astype(int)
    df["group_id"] = df["patient_id"]
    df["age_approx"] = df["age"].astype(float)
    df["sex"] = df["gender"].fillna("unknown").str.lower()
    df["anatom_site_general_challenge"] = df["region"].map(PAD_REGION_MAP).fillna("unknown")
    df["age_missing"] = df["age_approx"].isna().astype(int)
    df["diagnosis"] = df["diagnostic"]
    df["fitzpatrick"] = df["fitspatrick"]
    return df[KEEP + ["diagnosis", "fitzpatrick"]]


def _merge_groups(dev: pd.DataFrame, dups: pd.DataFrame) -> pd.Series:
    """Union lesion groups linked by a duplicate / same-lesion image pair (union-find)."""
    parent = {g: g for g in dev["group_id"]}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    g_of = dict(zip(dev["image_name"], dev["group_id"]))
    for a, b in zip(dups["a"], dups["b"]):
        if a in g_of and b in g_of:
            parent[find(g_of[a])] = find(g_of[b])
    return dev["group_id"].map(find)


def three_way_split(df: pd.DataFrame, seed: int, n_folds: int = 7) -> pd.Series:
    """Grouped 7-fold split (≈71/14/14). Shuffled StratifiedGroupKFold stratifies poorly,
    so val/calib are the two folds whose melanoma prevalence is closest to the overall
    prevalence (label-only rule, fixed before any model is trained); the rest is train."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [idx for _, idx in sgkf.split(df, df["target"], df["group_id"])]
    overall = df["target"].mean()
    order = sorted(range(n_folds), key=lambda k: abs(df["target"].iloc[folds[k]].mean() - overall))
    role = np.array(["train"] * len(df), dtype=object)
    role[folds[order[0]]] = "val"
    role[folds[order[1]]] = "calib"
    return pd.Series(role, index=df.index)


def run(seed: int = 42, out_dir: str = "splits/v2") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = load_isic2019("train")
    dup_path = out / "duplicate_pairs.csv"  # from scripts/near_duplicates.py (pHash + pixel-correlation check)
    dups = pd.read_csv(dup_path) if dup_path.exists() else pd.DataFrame(columns=["a", "b"])
    dev["group_id"] = _merge_groups(dev, dups)
    dev["role"] = three_way_split(dev, seed)
    splits = {r: dev[dev["role"] == r].drop(columns="role").reset_index(drop=True)
              for r in ["train", "val", "calib"]}
    splits["test"] = load_isic2019("test")
    splits["external"] = load_isic2020()
    splits["pad"] = load_pad_ufes()
    dev_names = set(dev["image_name"])
    linked = {b for a, b in zip(dups.a, dups.b) if a in dev_names} | {a for a, b in zip(dups.a, dups.b) if b in dev_names}
    for name in ("test", "external"):  # images duplicating a dev image are excluded from primary evaluation
        splits[name]["dup_in_dev"] = splits[name]["image_name"].isin(linked).astype(int)

    # --- leakage checks -------------------------------------------------------
    roles = ["train", "val", "calib"]
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            assert not set(splits[a].group_id) & set(splits[b].group_id), f"group leak {a}/{b}"
    names = {k: set(v.image_name) for k, v in splits.items()}
    for i, a in enumerate(names):
        for b in list(names)[i + 1:]:
            assert not names[a] & names[b], f"image overlap {a}/{b}"

    # --- impute missing age with the TRAINING median only ---------------------
    train_median = float(splits["train"]["age_approx"].median())
    for df in splits.values():
        df["age_approx"] = df["age_approx"].fillna(train_median)

    info = {"seed": seed, "train_age_median": train_median, "splits": {}}
    for name, df in splits.items():
        df.to_csv(out / f"{name}.csv", index=False)
        s = {
            "n": int(len(df)),
            "n_pos": int(df["target"].sum()),
            "prevalence": round(float(df["target"].mean()), 4),
            "n_groups": int(df["group_id"].nunique()),
            "age_missing": int(df["age_missing"].sum()),
            "sex_unknown": int((df["sex"] == "unknown").sum()),
            "site_unknown": int((df["anatom_site_general_challenge"] == "unknown").sum()),
        }
        if "UNK" in df.columns:
            s["n_unk"] = int(df["UNK"].sum())
        info["splits"][name] = s
        log.info(f"{name:9s} {s}")
    (out / "preprocess_info.json").write_text(json.dumps(info, indent=2))
    log.info(f"Saved split manifests to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="splits/v2")
    a = ap.parse_args()
    run(a.seed, a.out_dir)
