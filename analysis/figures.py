"""Result figures from saved predictions/probes (CPU).

    .venv/Scripts/python -m analysis.figures     # -> report/figures/*.png and paper/figures/*.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from analysis import collect as C
from analysis import stats as S

REPORT = Path("report/figures")
PAPER = Path("paper/figures")
COLORS = {"baseline": "#7f7f7f", "baseline_wide": "#bcbcbc", "concat_fusion": "#1f77b4",
          "cross_attention": "#d62728", "film": "#ff7f0e", "field_token": "#2ca02c"}
SPLIT_LABEL = {"test": "ISIC 2019 test", "external": "ISIC 2020", "pad": "PAD-UFES-20"}


def _save(fig, name):
    REPORT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(PAPER / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def roc():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, split in zip(axes, SPLIT_LABEL):
        y = C.manifest(split)["target"].values
        for m in [m for m in C.MAIN_MODELS if C.seeds(m)]:
            z = C.logit_matrix(m, split).mean(0)
            fpr, tpr, _ = roc_curve(y, z)
            ax.plot(fpr, tpr, color=COLORS[m], lw=1.8,
                    label=f"{C.LABEL[m]} ({S.discrimination(y, z)['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("sensitivity")
        ax.legend(fontsize=7, loc="lower right")
    _save(fig, "roc_curves")


def age_sweep():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = C.manifest("test")["target"].values
    for m in ["concat_fusion", "cross_attention", "film", "field_token"]:
        curves, grid = [], None
        for s in C.seeds(m):
            pr = C.probes(m, s, "test")
            if pr is not None and "age" in pr:
                curves.append(S.sigmoid(pr["age"][y == 1]).mean(0))
                grid = pr["age_grid"]
        if curves:
            c = np.array(curves)
            ax.plot(grid, c.mean(0), color=COLORS[m], lw=2, label=C.LABEL[m])
            ax.fill_between(grid, c.min(0), c.max(0), color=COLORS[m], alpha=0.15)
    ax.set_xlabel("age set to (years)")
    ax.set_ylabel("mean P(melanoma), melanoma cases")
    ax.set_title("Image-fixed age sweep (ISIC 2019 test; band = range over seeds)")
    ax.legend()
    _save(fig, "age_sweep")


def attention_maps(n_cases=3):
    pr = C.probes("field_token", 42, "test") if C.seeds("field_token") else None
    if pr is None or "attn" not in pr:
        print("[skip attention_maps]")
        return
    import pandas as pd
    from PIL import Image
    y = pr["target"]
    p = S.sigmoid(pr["orig"])
    idx = np.where(y == 1)[0][np.argsort(-p[y == 1])[:n_cases]]  # most confident melanomas
    man = pd.read_csv(C.SPLITS / "test.csv").set_index("image_name")
    root = Path("data/kaggle_upload/metaderm-images")
    fig, axes = plt.subplots(n_cases, 5, figsize=(13, 2.8 * n_cases))
    for r, i in enumerate(idx):
        name = pr["image_name"][i]
        axes[r, 0].imshow(Image.open(root / man.loc[name, "img_dir"] / f"{name}.jpg"))
        axes[r, 0].set_title(f"{name}\nP={p[i]:.2f}", fontsize=8)
        a = pr["attn"][i].astype(float).reshape(7, 7, -1)
        for k, field in enumerate(["age", "sex", "site", "missing"]):
            im = axes[r, k + 1].imshow(a[:, :, k], vmin=0, vmax=1, cmap="viridis")
            axes[r, k + 1].set_title(field, fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.6, label="head-averaged attention")
    _save(fig, "attention_maps")


def reliability():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, split in zip(axes, SPLIT_LABEL):
        y = C.manifest(split)["target"].values
        for m in [m for m in C.MAIN_MODELS if C.seeds(m)]:
            p = S.sigmoid(C.logit_matrix(m, split).mean(0))
            bins = np.minimum((p * 10).astype(int), 9)
            keep = [b for b in range(10) if (bins == b).sum() >= 20]
            ax.plot([p[bins == b].mean() for b in keep], [y[bins == b].mean() for b in keep],
                    "o-", color=COLORS[m], ms=3, lw=1.4, label=C.LABEL[m])
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"{SPLIT_LABEL[split]} (raw probabilities)")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed melanoma fraction")
        ax.legend(fontsize=7)
    _save(fig, "reliability")


def transfer():
    import pandas as pd
    p = Path("reports/v2/transfer.csv")
    if not p.exists():
        print("[skip transfer] run analysis.report --only transfer first")
        return
    df = pd.read_csv(p)
    df["sens"] = df["Sensitivity"].astype(str).str.split().str[0].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, split in zip(axes, ["external", "pad"]):
        for model, g in df[df.Target == split].groupby("Model"):
            ax.plot(g["n labelled"], g["sens"], "o-", label=model)
        ax.axhline(0.90, color="k", ls=":", lw=0.8)
        ax.set_xscale("symlog", linthresh=25)
        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlabel("labelled target images used to refit threshold (0 = transfer as is)")
    axes[0].set_ylabel("sensitivity achieved (target 0.90)")
    axes[0].legend(fontsize=8)
    _save(fig, "transfer")


def training_curves():
    import pandas as pd
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for m in [m for m in C.MAIN_MODELS if C.seeds(m)]:
        h = pd.concat([pd.read_csv(C.RUNS / m / f"seed{s}" / "history.csv") for s in C.seeds(m)])
        g = h.groupby("epoch")  # epochs past early stopping exist for fewer seeds; mean over those that ran
        axes[0].plot(g.train_loss.mean(), color=COLORS[m], lw=1.6, label=C.LABEL[m])
        axes[0].plot(g.val_loss.mean(), color=COLORS[m], lw=1.2, ls="--")
        axes[1].plot(g.val_auroc.mean(), color=COLORS[m], lw=1.6, label=C.LABEL[m])
    axes[0].set_title("loss (solid: training, dashed: validation)")
    axes[1].set_title("validation AUROC")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    _save(fig, "training_curves")


def subgroups():
    import pandas as pd
    df = pd.read_csv("reports/v2/subgroups.csv")
    df["auc"] = df["AUROC [95% CI]"].str.split().str[0].astype(float)
    df = df[df.Value != "unknown"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True, gridspec_kw={"width_ratios": [2, 4, 5]})
    labels = {C.LABEL[m]: m for m in C.MAIN_MODELS}
    for ax, grp in zip(axes, ["Sex", "Age", "Site"]):
        d = df[df.Group == grp]
        order = ["<=30", "31-50", "51-70", ">70"] if grp == "Age" else sorted(d.Value.unique())
        w = 0.8 / d.Model.nunique()
        for i, (model, g) in enumerate(d.groupby("Model", sort=False)):
            g = g.set_index("Value").reindex(order)
            ax.bar(np.arange(len(order)) + i * w, g.auc, w, color=COLORS[labels[model]], label=model)
        ax.set_xticks(np.arange(len(order)) + 0.4 - w / 2, order, fontsize=8)
        ax.set_ylim(0.7, 1.0)
        ax.set_title(grp)
    axes[0].set_ylabel("AUROC (seed ensemble, ISIC 2019 test)")
    axes[2].legend(fontsize=7, ncol=2)
    _save(fig, "subgroups")


def error_examples(model="baseline", k=4):
    """The k most confident misses (melanomas) and false alarms of the seed ensemble on ISIC 2019 test."""
    from PIL import Image
    from analysis.report import DX
    from src.data.dataloader import image_root
    man = C.manifest("test")
    p = S.sigmoid(C.logit_matrix(model, "test").mean(0))
    y = man["target"].values
    dx = np.array(DX)[man[DX].values.argmax(1)]
    fn = np.where(y == 1)[0][np.argsort(p[y == 1])[:k]]
    fp = np.where(y == 0)[0][np.argsort(-p[y == 0])[:k]]
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 6.6))
    for row, idx, title in [(0, fn, "missed melanoma"), (1, fp, "false alarm")]:
        for ax, i in zip(axes[row], idx):
            r = man.iloc[i]
            ax.imshow(Image.open(image_root({}) / r.img_dir / f"{r.image_name}.jpg").convert("RGB"))
            ax.set_title(f"{title}: {dx[i]}, p = {p[i]:.3f}\n{r.sex}, {r.age_approx:.0f} y, "
                         f"{r.anatom_site_general_challenge}", fontsize=8)
            ax.axis("off")
    _save(fig, "error_examples")


def _gradcam(model_name, seed, img):
    import torch
    from src.data.transforms import get_val_transforms
    from src.models.factory import build_model
    from src.xai.gradcam import GradCAM, get_default_target_layer, overlay_cam_on_image
    from src.data.dataloader import fit_encoder
    ck = torch.load(C.RUNS / model_name / f"seed{seed}" / "model.pt", map_location="cpu", weights_only=False)
    cfg = {**ck["config"], "pretrained": False}
    model = build_model(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg, get_val_transforms(cfg.get("img_size", 224)), GradCAM, get_default_target_layer,         overlay_cam_on_image, fit_encoder


def case_studies(names=("ISIC_0064585", "ISIC_0057796"), seed=42):
    """Per case: image, Grad-CAM of image-only and field-token (seed 42), and the five-seed age sweep of every
    fusion model with the image-only ensemble probability as reference."""
    import torch
    from PIL import Image
    from src.data.dataloader import image_root
    man = C.manifest("test")
    fig, axes = plt.subplots(len(names), 4, figsize=(17, 4.2 * len(names)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    cams = {}
    for m in ("baseline", "field_token"):
        model, cfg, tf, GradCAM, layer, overlay, fit_encoder = _gradcam(m, seed, None)
        enc = fit_encoder(cfg)
        for name in names:
            r = man[man.image_name == name].iloc[[0]]
            pil = Image.open(image_root({}) / r.img_dir.iloc[0] / f"{name}.jpg").convert("RGB").resize((224, 224))
            cont, site = enc.transform(r)
            g = GradCAM(model, layer(cfg["model_type"], cfg.get("backbone", "efficientnet_b2")))
            cam = g.generate(image=tf(pil).unsqueeze(0), metadata_cont=torch.tensor(cont, dtype=torch.float32),
                             site_idx=torch.tensor(site, dtype=torch.long), model_type=cfg["model_type"])
            g.remove_hooks()
            cams[m, name] = (np.array(pil), overlay(np.array(pil), cam))
    p_io = S.sigmoid(C.logit_matrix("baseline", "test").mean(0))
    for row, name in zip(np.atleast_2d(axes), names):
        i = int(np.where(man.image_name == name)[0][0])
        r = man.iloc[i]
        row[0].imshow(cams["baseline", name][0])
        row[0].set_title(f"{name}\n{r.sex}, {r.age_approx:.0f} y, {r.anatom_site_general_challenge}", fontsize=9)
        row[1].imshow(cams["baseline", name][1])
        row[1].set_title("Grad-CAM, image-only", fontsize=9)
        row[2].imshow(cams["field_token", name][1])
        row[2].set_title("Grad-CAM, field-token", fontsize=9)
        for a in row[:3]:
            a.axis("off")
        ax = row[3]
        for m in ["concat_fusion", "cross_attention", "film", "field_token"]:
            pr = [C.probes(m, s, "test") for s in C.seeds(m)]
            curve = S.sigmoid(np.mean([q["age"][i] for q in pr], 0))
            ax.plot(pr[0]["age_grid"], curve, color=COLORS[m], lw=2, label=C.LABEL[m])
        ax.axhline(p_io[i], color=COLORS["baseline"], ls="--", lw=1.5, label=f"Image-only ({p_io[i]:.2f})")
        ax.axvline(r.age_approx, color="k", ls=":", lw=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("age set to (years); dotted line = recorded age")
        ax.set_ylabel("P(melanoma), five-seed ensemble")
        ax.legend(fontsize=7, loc="lower right")
    _save(fig, "case_studies")


def metadata_profile():
    import pandas as pd
    sets = {"ISIC 2019 (development)": pd.concat([pd.read_csv(f"splits/v2/{s}.csv") for s in ("train", "val", "calib")]),
            "ISIC 2020": pd.read_csv("splits/v2/external.csv"), "PAD-UFES-20": pd.read_csv("splits/v2/pad.csv")}
    sites = ["head/neck", "upper extremity", "torso", "lower extremity", "palms/soles"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for k, (name, d) in enumerate(sets.items()):
        d = d[d.age_missing == 0]
        for y, col, lab in [(0, "#7f7f7f", "other lesions"), (1, "#d62728", "melanoma")]:
            g = d[d.target == y]
            axes[0, k].hist(g.age_approx, bins=np.arange(0, 95, 5), density=True, alpha=0.55, color=col,
                            label=f"{lab} (mean {g.age_approx.mean():.1f} y)")
            share = g.anatom_site_general_challenge.value_counts(normalize=True).reindex(sites).fillna(0)
            axes[1, k].bar(np.arange(len(sites)) + (0.2 if y else -0.2), share.values, 0.4, color=col, label=lab)
        axes[0, k].set_title(name)
        axes[0, k].set_xlabel("age (years)")
        axes[0, k].legend(fontsize=8)
        axes[1, k].set_xticks(np.arange(len(sites)), sites, rotation=25, fontsize=8)
        axes[1, k].set_ylabel("share of images")
    axes[0, 0].set_ylabel("density")
    axes[1, 0].legend(fontsize=8)
    _save(fig, "metadata_profile")


if __name__ == "__main__":
    for f in (roc, age_sweep, attention_maps, reliability, transfer, training_curves, subgroups, error_examples,
              case_studies, metadata_profile):
        try:
            f()
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"[skip {f.__name__}] {type(e).__name__}: {e}")
