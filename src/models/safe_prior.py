"""
SAFE-Prior evidence network (Safe, Adaptable Fusion with an Explicit prior).

The fused log-odds are split into an image evidence term and an explicit metadata prior:

    logit(y | x, m) = f(x) + g_s(m)

g_s is a logistic regression on the encoded metadata [age, sex, missing flag, site one-hot], fitted once on the
training split and frozen (stored as buffers, so it travels with model.pt). The network is trained on the sum
(product-of-experts / conditional logit adjustment), so f only has to explain what the prior cannot. A
within-class HSIC penalty (normalised, RBF kernels) between f's embedding and the metadata discourages f from
re-learning age or site from the image (counterfactual invariance in the anticausal direction, f(x) ⊥ m | y).

At test time f and g are separable: the prior can be bounded, re-estimated for a new site, or dropped
(see analysis/safe_prior.py).
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.vision_backbone import create_backbone


def _rbf(x: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(x, x).pow(2)
    sigma2 = d[d > 0].median() if (d > 0).any() else d.new_tensor(1.0)
    return torch.exp(-d / sigma2.clamp_min(1e-6))


def nhsic(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Normalised HSIC (kernel alignment) in [0, 1]; 0 means no detectable dependence."""
    n = x.size(0)
    h = torch.eye(n, device=x.device) - 1.0 / n
    k, l = h @ _rbf(x) @ h, h @ _rbf(z) @ h
    return (k * l).sum() / ((k * k).sum().sqrt() * (l * l).sum().sqrt()).clamp_min(1e-8)


class SAFEPriorModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b2", pretrained: bool = True, dropout: float = 0.4,
                 continuous_dim: int = 5, num_sites: int = 7, hsic_weight: float = 0.0) -> None:
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained, num_classes=0, global_pool="avg")
        self.embed = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.num_features, 128),
                                   nn.BatchNorm1d(128), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout * 0.5), nn.Linear(128, 1))
        self.register_buffer("prior_w_cont", torch.zeros(continuous_dim))
        self.register_buffer("prior_w_site", torch.zeros(num_sites))
        self.register_buffer("prior_b", torch.zeros(1))
        self.num_sites, self.hsic_weight = num_sites, hsic_weight
        self._cache = None

    def set_prior(self, w_cont, w_site, b) -> None:
        self.prior_w_cont.copy_(torch.as_tensor(w_cont, dtype=torch.float32))
        self.prior_w_site.copy_(torch.as_tensor(w_site, dtype=torch.float32))
        self.prior_b.copy_(torch.as_tensor([b], dtype=torch.float32).reshape(1))

    def prior(self, metadata_cont, site_idx) -> torch.Tensor:
        return metadata_cont.float() @ self.prior_w_cont + self.prior_w_site[site_idx] + self.prior_b

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)  # (B, C)

    def evidence(self, feats) -> torch.Tensor:
        """Image evidence f(x) (log-likelihood-ratio term), without the metadata prior."""
        return self.head(self.embed(feats)).squeeze(-1)

    def fuse(self, feats, metadata_cont, site_idx) -> torch.Tensor:
        emb = self.embed(feats)
        if self.training and self.hsic_weight > 0:
            self._cache = (emb, metadata_cont, site_idx)
        return self.head(emb).squeeze(-1) + self.prior(metadata_cont, site_idx)

    def forward(self, image, metadata_cont, site_idx) -> torch.Tensor:
        return self.fuse(self.encode_image(image), metadata_cont, site_idx)

    def extra_loss(self, y: torch.Tensor) -> torch.Tensor:
        """hsic_weight * mean over classes of nHSIC(embedding, metadata) within the class."""
        if self.hsic_weight <= 0 or self._cache is None:
            return torch.zeros((), device=y.device)
        emb, cont, site = self._cache
        self._cache = None
        m = torch.cat([cont.float(), nn.functional.one_hot(site, self.num_sites).float()], dim=1)
        terms = [nhsic(emb[y == c].float(), m[y == c]) for c in (0, 1) if int((y == c).sum()) >= 4]
        if not terms:
            return torch.zeros((), device=y.device)
        return self.hsic_weight * torch.stack(terms).mean()


def fit_metadata_prior(cfg: dict):
    """Logistic regression of the label on the encoded training metadata -> (w_cont, w_site, b)."""
    from sklearn.linear_model import LogisticRegression

    from src.data.dataloader import fit_encoder, load_split
    enc = fit_encoder(cfg)
    tr = load_split(cfg, "train")
    cont, site = enc.transform(tr)
    n_sites = len(enc.site_categories)
    lr = LogisticRegression(max_iter=5000).fit(np.hstack([cont, np.eye(n_sites)[site]]), tr["target"].values)
    w = lr.coef_[0]
    return w[:cont.shape[1]], w[cont.shape[1]:], float(lr.intercept_[0])
