"""
Metadata encoder model — Phase 5.
Combines continuous metadata features with a site embedding → latent vector.
"""

import torch
import torch.nn as nn


class MetadataEncoderModel(nn.Module):
    """
    Encodes clinical metadata into a latent vector.

    Inputs:
        continuous: (B, continuous_dim) — [age_norm, sex_onehot(3), age_missing]
        site_idx:   (B,) — integer index for site embedding

    Output:
        (B, output_dim) — metadata latent vector
    """

    def __init__(
        self,
        continuous_dim: int = 5,
        num_sites: int = 7,
        site_embed_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.site_embedding = nn.Embedding(num_sites, site_embed_dim)

        input_dim = continuous_dim + site_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        continuous: torch.Tensor,  # (B, continuous_dim)
        site_idx: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        site_emb = self.site_embedding(site_idx)           # (B, site_embed_dim)
        x = torch.cat([continuous, site_emb], dim=-1)     # (B, input_dim)
        return self.mlp(x)                                 # (B, output_dim)
