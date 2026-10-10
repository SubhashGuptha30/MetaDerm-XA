"""
Post-training probes (GPU). For a trained run, saves raw logits under metadata
interventions; every statistic is computed offline by analysis/ scripts.

Per image (the backbone is run ONCE per batch; only the fusion part is re-run):
  * orig            — recorded metadata
  * age sweep       — age set to each value on AGE_GRID (age_missing flag kept as recorded)
  * sex sweep       — male / female / unknown
  * site sweep      — each of the 7 site categories
  * withheld        — all metadata set to the 'unknown' encoding (age=train median, flag=1)
  * shuffled        — metadata of a different, randomly permuted image from the same split
Token models additionally:
  * attn            — head-averaged attention (N_img_tokens × K) for the recorded metadata
  * faithfulness    — logits after zeroing the k most-attended / k least-attended /
                      k random image tokens (k ∈ FAITH_K, RAND_DRAWS random draws averaged)

Writes runs_v2/<model>/seed<s>/probes/{test,external}.npz

Usage:
    python -m src.probes --run_dir runs_v2/field_token/seed42
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.dataloader import fit_encoder, get_dataloaders, load_split
from src.models.factory import IMAGE_ONLY, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

AGE_GRID = np.arange(10, 90, 5)  # 10, 15, ..., 85 (16 points)
FAITH_K = (5, 10)
RAND_DRAWS = 20


def _with(meta, site, *, age=None, sex=None, site_idx=None, enc=None):
    m, s = meta.clone(), site.clone()
    if age is not None:
        m[:, 0] = (age - enc.age_mean) / enc.age_std
    if sex is not None:
        m[:, 1:4] = 0.0
        m[:, 1 + sex] = 1.0
    if site_idx is not None:
        s[:] = site_idx
    return m, s


@torch.no_grad()
def probe_split(model, cfg, split, device, full: bool):
    enc = fit_encoder(cfg)
    df = load_split(cfg, split)
    cont_all, site_all = enc.transform(df)
    perm = np.random.default_rng(12345).permutation(len(df))
    unk_site = enc.site_categories.index("unknown")
    median_norm = (_train_median(cfg) - enc.age_mean) / enc.age_std  # how a missing age is encoded

    loader = get_dataloaders(cfg, (split,))[split]
    image_only = cfg["model_type"] in IMAGE_ONLY
    token_model = hasattr(model, "cross_attn_weights")
    amp = device.type == "cuda"
    rng = torch.Generator(device="cpu").manual_seed(0)

    out = {k: [] for k in ["orig", "withheld", "shuffled", "age", "sex", "site", "attn"]}
    for k in FAITH_K:
        for kind in ("top", "bottom", "rand"):
            out[f"faith_{kind}{k}"] = []
    pos = 0
    for x, m, s, y, _ in loader:
        b = y.size(0)
        x, m, s = x.to(device), m.to(device), s.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            f = model.encode_image(x)
            fuse = lambda mm, ss: model.fuse(f, mm, ss).float().cpu().numpy()  # noqa: E731
            out["orig"].append(fuse(m, s))
            if token_model:
                a = model.cross_attn_weights.float().mean(1)  # (B, N, K)
                out["attn"].append(a.cpu().numpy().astype(np.float16))
            if not image_only:
                wm = torch.zeros_like(m)
                wm[:, 0] = median_norm  # imputed age
                wm[:, 3] = 1.0          # sex one-hot = unknown
                wm[:, 4] = 1.0          # age_missing flag
                out["withheld"].append(fuse(wm, torch.full_like(s, unk_site)))
                idx = perm[pos:pos + b]
                out["shuffled"].append(fuse(torch.as_tensor(cont_all[idx], device=device),
                                            torch.as_tensor(site_all[idx], device=device)))
                if full:
                    out["age"].append(np.stack([fuse(*_with(m, s, age=a_, enc=enc)) for a_ in AGE_GRID], 1))
                    out["sex"].append(np.stack([fuse(*_with(m, s, sex=k)) for k in range(3)], 1))
                    out["site"].append(np.stack([fuse(*_with(m, s, site_idx=k)) for k in range(len(enc.site_categories))], 1))
            if token_model and full:
                imp = model.cross_attn_weights.float().mean(1).sum(-1)  # (B, N) total attention
                order = imp.argsort(dim=1, descending=True)
                n_tok = f.shape[1]
                for k in FAITH_K:
                    for kind, idx_ in (("top", order[:, :k]), ("bottom", order[:, -k:])):
                        fm = f.clone()
                        fm.scatter_(1, idx_[..., None].expand(-1, -1, f.shape[2]), 0.0)
                        out[f"faith_{kind}{k}"].append(model.fuse(fm, m, s).float().cpu().numpy())
                    draws = []
                    for _ in range(RAND_DRAWS):
                        r = torch.rand(b, n_tok, generator=rng).argsort(1)[:, :k].to(device)
                        fm = f.clone()
                        fm.scatter_(1, r[..., None].expand(-1, -1, f.shape[2]), 0.0)
                        draws.append(model.fuse(fm, m, s).float().cpu().numpy())
                    out[f"faith_rand{k}"].append(np.mean(draws, 0))
        pos += b

    res = {k: np.concatenate(v) for k, v in out.items() if v}
    res["image_name"] = np.array(df["image_name"].tolist(), dtype="U")
    res["target"] = df["target"].values.astype(np.int8)
    res["age_grid"] = AGE_GRID
    res["site_categories"] = np.array(enc.site_categories)
    return res


def _train_median(cfg) -> float:
    import json
    info = Path(cfg.get("split_dir", "splits/v2")) / "preprocess_info.json"
    return float(json.loads(info.read_text())["train_age_median"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--splits", nargs="*", default=["test", "external", "pad"])
    a = ap.parse_args()
    run = Path(a.run_dir)
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["pretrained"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    (run / "probes").mkdir(exist_ok=True)
    for split in a.splits:
        res = probe_split(model, cfg, split, device, full=split == "test")
        np.savez_compressed(run / "probes" / f"{split}.npz", **res)
        log.info(f"saved probes/{split}.npz ({len(res['orig'])} images)")


if __name__ == "__main__":
    main()
