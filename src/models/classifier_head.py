"""
Classifier head — shared across all multi-modal model variants.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Pools spatial tokens → dense layers → output logit.

    Args:
        input_dim: dimension of input (feature_dim after fusion).
        hidden_dim: hidden layer size.
        dropout: dropout rate.
        pool: 'mean' or 'max' global pooling over token dimension.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        assert pool in ("mean", "max"), "pool must be 'mean' or 'max'"
        self.pool = pool

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # binary output — logit
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) or (B, D)
        Returns:
            logits: (B,) — raw logit (no sigmoid applied)
        """
        if tokens.dim() == 3:
            if self.pool == "mean":
                pooled = tokens.mean(dim=1)  # (B, D)
            else:
                pooled = tokens.max(dim=1).values
        else:
            pooled = tokens  # already (B, D)

        return self.head(pooled).squeeze(-1)  # (B,)
