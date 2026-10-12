"""v2 pipeline tests: models, encode/fuse equivalence, config loading, and a CPU end-to-end
train → predict → probe run on fake images."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.config import load_config
from src.models.factory import MODEL_TYPES, build_model

B = 4


def _batch(img=64):
    torch.manual_seed(0)
    x = torch.randn(B, 3, img, img)
    meta = torch.tensor([[0.1, 1, 0, 0, 0], [-0.5, 0, 1, 0, 0], [1.2, 0, 0, 1, 1], [0.0, 1, 0, 0, 0]],
                        dtype=torch.float32)
    site = torch.tensor([0, 3, 6, 4])
    return x, meta, site


def _cfg(model_type, **kw):
    cfg = {"model_type": model_type, "pretrained": False, "backbone": "efficientnet_b0", "dropout": 0.4,
           "num_meta_tokens": 2}
    cfg.update(kw)
    return cfg


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_forward_and_encode_fuse_equivalence(model_type):
    model = build_model(_cfg(model_type)).eval()
    x, m, s = _batch()
    with torch.no_grad():
        out = model(x, m, s)
        out2 = model.fuse(model.encode_image(x), m, s)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, out2)


@pytest.mark.parametrize("model_type", ["concat_fusion", "film", "cross_attention", "field_token"])
def test_metadata_changes_output(model_type):
    model = build_model(_cfg(model_type)).eval()
    x, m, s = _batch()
    with torch.no_grad():
        a = model(x, m, s)
        m2 = m.clone()
        m2[:, 0] += 2.0
        b = model(x, m2, (s + 1) % 7)
    assert not torch.allclose(a, b), "metadata has no effect on output"


def test_image_only_ignores_metadata():
    model = build_model(_cfg("baseline")).eval()
    x, m, s = _batch()
    with torch.no_grad():
        torch.testing.assert_close(model(x, m, s), model(x, m * 0, s * 0))


def test_field_token_attention_shape():
    model = build_model(_cfg("field_token")).eval()
    x, m, s = _batch(img=64)
    with torch.no_grad():
        model(x, m, s)
    w = model.cross_attn_weights
    assert w.shape[:2] == (B, 8) and w.shape[-1] == 4
    torch.testing.assert_close(w.sum(-1), torch.ones_like(w.sum(-1)))


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/v2/field_token_k2.yaml", ["seed=7", "lr=1e-3"])
    assert cfg["model_type"] == "field_token_ablated" and cfg["num_meta_tokens"] == 2
    assert cfg["seed"] == 7 and cfg["lr"] == 1e-3 and cfg["imbalance"] == "none"
    base = load_config("configs/v2/base.yaml")
    dino = load_config("configs/v2/dino_base.yaml")
    drop = lambda d: {k: v for k, v in d.items() if k not in {"backbone", "lr"}}  # noqa: E731
    assert drop(dino) == drop(base), "dino_base may only change backbone/lr"
    # model identity and method switches; everything else (training recipe) must equal the shared base
    identity = {"model_name", "model_type", "num_meta_tokens", "meta_dropout", "ogm_alpha", "hsic_weight"}
    from pathlib import Path
    for path in Path("configs/v2").glob("*.yaml"):
        c = load_config(path)
        ref = dino if path.name.startswith("dino_") else base
        assert {k: v for k, v in c.items() if k not in identity} == ref, f"{path} deviates from its base config"


@pytest.fixture
def fake_project(tmp_path):
    """Tiny split manifests + random JPEGs mirroring the real layout."""
    rng = np.random.default_rng(0)
    split_dir, img_root = tmp_path / "splits", tmp_path / "images"
    split_dir.mkdir()
    sites = ["torso", "head/neck", "unknown", "palms/soles"]
    for split, n in [("train", 24), ("val", 12), ("calib", 8), ("test", 12), ("external", 8), ("pad", 8)]:
        d = "isic2020" if split == "external" else "isic2019_train"
        (img_root / d).mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            name = f"{split}_{i}"
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(img_root / d / f"{name}.jpg")
            rows.append({"image_name": name, "img_dir": d, "target": int(i % 3 == 0),
                         "age_approx": float(rng.integers(20, 80)), "age_missing": int(i == 1),
                         "sex": ["male", "female", "unknown"][i % 3],
                         "anatom_site_general_challenge": sites[i % 4], "group_id": name})
        pd.DataFrame(rows).to_csv(split_dir / f"{split}.csv", index=False)
    (split_dir / "preprocess_info.json").write_text(json.dumps({"train_age_median": 50.0}))
    return tmp_path, split_dir, img_root


@pytest.mark.parametrize("model_type", ["baseline", "field_token", "film"])
def test_end_to_end_train_and_probe(fake_project, model_type, monkeypatch):
    from src import probes, train
    tmp, split_dir, img_root = fake_project
    cfg = load_config("configs/v2/base.yaml", [
        f"split_dir={split_dir.as_posix()}", f"image_root={img_root.as_posix()}",
        f"output_root={(tmp / 'runs').as_posix()}", "backbone=efficientnet_b0", "pretrained=false",
        "img_size=64", "batch_size=4", "num_workers=0", "epochs=2", "patience=5"])
    cfg.update(model_name=model_type, model_type=model_type)
    summary = train.run(cfg)
    run = tmp / "runs" / model_type / "seed42"
    for split in ["val", "calib", "test", "external"]:
        p = pd.read_csv(run / "preds" / f"{split}.csv.gz")
        assert len(p) == len(pd.read_csv(split_dir / f"{split}.csv")) and p.logit.notna().all()
    assert summary["epochs_run"] == 2

    monkeypatch.setattr("sys.argv", ["probes", "--run_dir", str(run)])
    probes.main()
    r = np.load(run / "probes" / "test.npz")
    preds = pd.read_csv(run / "preds" / "test.csv.gz")
    np.testing.assert_allclose(r["orig"], preds.logit.values, rtol=1e-4, atol=1e-4)
    if model_type != "baseline":
        assert r["age"].shape == (12, 16) and r["sex"].shape == (12, 3) and r["site"].shape == (12, 7)
        assert r["withheld"].shape == (12,) and r["shuffled"].shape == (12,)
    if model_type == "field_token":
        assert r["attn"].shape[-1] == 4 and "faith_top5" in r
