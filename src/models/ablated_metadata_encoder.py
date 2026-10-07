"""
Ablated field-token metadata encoders for K=1,2,3 token configurations.

Used to verify that K=4 (one token per metadata field) is optimal and
that the separation of metadata fields into individual tokens is the
source of the interpretability advantage.

K=1: All metadata collapsed into one token (equivalent to CrossAttentionFusion)
K=2: [demographic (age+sex+miss), site] — tests whether site isolation matters
K=3: [age+miss, sex, site] — tests whether age/sex separation matters
K=4: [age, sex, site, miss] — full model (existing FieldTokenMetadataEncoder)
"""

import torch
import torch.nn as nn


class AblatedMetadataEncoder(nn.Module):
    """
    Configurable metadata encoder that produces K tokens from clinical metadata.

    Inputs:
        continuous: (B, 5) — [age_norm, sex_m, sex_f, sex_u, age_missing]
        site_idx:   (B,)   — integer index for site embedding

    Output:
        (B, K, output_dim) — K metadata tokens
    """

    def __init__(
        self,
        num_tokens: int = 4,
        num_sites: int = 7,
        site_embed_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        self.site_embedding = nn.Embedding(num_sites, site_embed_dim)

        if num_tokens == 1:
            # K=1: all metadata → single token
            # Input: age(1) + sex(3) + miss(1) + site_embed(16) = 21
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(5 + site_embed_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim),
                )
            ])
        elif num_tokens == 2:
            # K=2: [demographic (age+sex+miss), site]
            # Token 0: age(1) + sex(3) + miss(1) = 5
            # Token 1: site_embed(16)
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(5, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim),
                ),
                nn.Sequential(
                    nn.Linear(site_embed_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim),
                ),
            ])
        elif num_tokens == 3:
            # K=3: [age+miss, sex, site]
            # Token 0: age(1) + miss(1) = 2
            # Token 1: sex(3)
            # Token 2: site_embed(16)
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim),
                ),
                nn.Sequential(
                    nn.Linear(3, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim),
                ),
                nn.Sequential(
                    nn.Linear(site_embed_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                    nn.LayerNorm(output_dim),
                ),
            ])
        else:
            raise ValueError(f"num_tokens must be 1, 2, or 3, got {num_tokens}")

    def forward(
        self,
        continuous: torch.Tensor,  # (B, 5)
        site_idx: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """Returns (B, K, D) — K metadata tokens."""
        age = continuous[:, 0:1]       # (B, 1)
        sex = continuous[:, 1:4]       # (B, 3)
        miss = continuous[:, 4:5]      # (B, 1)
        site_emb = self.site_embedding(site_idx)  # (B, 16)

        if self.num_tokens == 1:
            combined = torch.cat([continuous, site_emb], dim=-1)  # (B, 21)
            tokens = [self.projectors[0](combined)]
        elif self.num_tokens == 2:
            demo = continuous  # (B, 5) — age + sex + miss
            tokens = [
                self.projectors[0](demo),
                self.projectors[1](site_emb),
            ]
        elif self.num_tokens == 3:
            age_miss = torch.cat([age, miss], dim=-1)  # (B, 2)
            tokens = [
                self.projectors[0](age_miss),
                self.projectors[1](sex),
                self.projectors[2](site_emb),
            ]

        return torch.stack(tokens, dim=1)  # (B, K, D)
