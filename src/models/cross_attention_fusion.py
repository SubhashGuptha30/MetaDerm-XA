"""
Cross-attention fusion — Phase 5. Core novelty.

Image spatial tokens → Query
Metadata latent vector → Key & Value (broadcast to match token count)

Implements standard scaled dot-product attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention where image tokens attend to the metadata vector.

    Args:
        feature_dim: dimension of image tokens and metadata vector.
        num_heads: number of attention heads.
        dropout: attention dropout.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q from image tokens, K/V from metadata
        self.W_Q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # Post-attention residual + norm
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        # Store last attention weights for XAI
        self.last_attn_weights: torch.Tensor = None

    def forward(
        self,
        image_tokens: torch.Tensor,   # (B, N, D)  — spatial image tokens
        metadata: torch.Tensor,       # (B, D) or (B, J, D) — metadata
    ) -> torch.Tensor:
        """
        Accepts metadata as either:
            - (B, D)    → single-token broadcast (original behaviour)
            - (B, J, D) → field-token cross-attention (J metadata tokens)

        Returns:
            fused: (B, N, D) — metadata-conditioned image representation
        """
        B, N, D = image_tokens.shape

        # Handle both single vector and multi-token metadata input
        if metadata.dim() == 2:
            meta = metadata.unsqueeze(1)  # (B, 1, D)
        else:
            meta = metadata               # (B, J, D)

        J = meta.shape[1]  # 1 for single-token, 4 for field-token

        # Project
        Q = self.W_Q(image_tokens)       # (B, N, D)
        K = self.W_K(meta)               # (B, J, D)
        V = self.W_V(meta)               # (B, J, D)

        # Reshape for multi-head: (B, heads, seq_len, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, J, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, J, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, heads, N, J)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)          # (B, heads, N, J)
        attn_weights = self.dropout(attn_weights)

        # Store for XAI (detach to avoid memory leak in eval)
        self.last_attn_weights = attn_weights.detach()

        # Weighted sum over V: (B, heads, N, head_dim)
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        attended = self.out_proj(attended)

        # Residual + LayerNorm 1
        x = self.norm1(image_tokens + attended)

        # FFN + residual + LayerNorm 2
        x = self.norm2(x + self.ffn(x))

        return x  # (B, N, D)
