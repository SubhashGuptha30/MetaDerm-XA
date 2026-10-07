"""
Field-token metadata encoder — Journal upgrade.

Instead of collapsing all metadata into a single vector (B, D),
this encoder produces one token per metadata field:

    age      → (B, 1, D)
    sex      → (B, 1, D)    →  stacked to (B, 4, D)
    site     → (B, 1, D)
    missing  → (B, 1, D)

This gives the cross-attention layer J=4 key/value positions, so
softmax(QK^T / sqrt(d_k)) is no longer degenerate (always 1.0).
Each image spatial location can now selectively attend to different
metadata fields.
"""

import torch
import torch.nn as nn


class FieldTokenMetadataEncoder(nn.Module):
    """
    Encodes clinical metadata into J separate field tokens.

    Inputs:
        continuous: (B, continuous_dim) — [age_norm, sex_m, sex_f, sex_u, age_missing]
        site_idx:   (B,) — integer index for site embedding

    Output:
        (B, 4, output_dim) — one token per metadata field
    """

    NUM_FIELDS = 4  # age, sex, site, missingness

    def __init__(
        self,
        num_sites: int = 7,
        site_embed_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # --- Age token: 1D scalar → D ---
        self.age_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # --- Sex token: 3D one-hot → D ---
        self.sex_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # --- Site token: embedding → D ---
        self.site_embedding = nn.Embedding(num_sites, site_embed_dim)
        self.site_proj = nn.Sequential(
            nn.Linear(site_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # --- Missingness token: 1D binary → D ---
        self.miss_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        continuous: torch.Tensor,  # (B, 5)
        site_idx: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """
        Returns:
            field_tokens: (B, 4, D) — [age_token, sex_token, site_token, miss_token]
        """
        # Split the continuous vector into semantic fields
        age = continuous[:, 0:1]       # (B, 1)  — normalized age
        sex = continuous[:, 1:4]       # (B, 3)  — one-hot sex
        miss = continuous[:, 4:5]      # (B, 1)  — age_missing flag

        # Project each field to D independently
        age_token = self.age_proj(age)                         # (B, D)
        sex_token = self.sex_proj(sex)                         # (B, D)
        site_emb = self.site_embedding(site_idx)               # (B, 16)
        site_token = self.site_proj(site_emb)                  # (B, D)
        miss_token = self.miss_proj(miss)                      # (B, D)

        # Stack into (B, 4, D)
        field_tokens = torch.stack(
            [age_token, sex_token, site_token, miss_token], dim=1
        )

        return field_tokens
