"""
Quantum-Inspired Linear Layer — Phase 6.

Implements a layer with amplitude and phase parameters inspired by quantum
superposition mechanics:
    - amplitude (r): magnitude of each unit's "quantum state"
    - phase (θ): phase angle
    - output = Σ r_i * [cos(θ_i), sin(θ_i)] · x   → magnitude

This is a drop-in replacement for a standard nn.Linear that introduces
structured inductive bias in the frequency domain, at identical parameter count.

Reference framing: Quantum-Inspired Neural Networks (QNNs) have been shown
to exhibit different expressivity profiles for certain function classes.
We evaluate this empirically in the ablation — see reports/ablation_analysis.md.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumInspiredLinear(nn.Module):
    """
    Quantum-inspired linear transformation.

    For each output unit i:
        w_i = r_i * [cos(θ_i), sin(θ_i)]  (complex-valued weight represented as 2D real vector)
        output_i = |w_i · x + b_i|  (magnitude — non-negative by construction)

    The final activation is a learnable mixture of magnitude and phase outputs.

    Args:
        in_features: input feature dimension.
        out_features: output feature dimension.
        bias: whether to use bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Amplitude: r_i > 0 (stored in log-space for stability)
        self.log_amplitude = nn.Parameter(torch.zeros(out_features, in_features))
        # Phase: θ_i ∈ [0, 2π]
        self.phase = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable gate between quantum magnitude and linear paths
        self.gate = nn.Parameter(torch.ones(1) * 0.5)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Initialize similar to Kaiming uniform
        std = math.sqrt(2.0 / self.in_features)
        nn.init.normal_(self.log_amplitude, mean=0.0, std=std * 0.1)
        nn.init.uniform_(self.phase, 0, 2 * math.pi)

    def _build_weight(self) -> torch.Tensor:
        """
        Constructs effective weight matrix W of shape (out, in) from amplitude and phase.
        W[i, j] = exp(log_r[i,j]) * cos(θ[i,j])
        """
        r = torch.exp(self.log_amplitude)                         # (out, in), positive
        W = r * torch.cos(self.phase)                             # (out, in)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features)
        Returns:
            out: (..., out_features)
        """
        W = self._build_weight()  # (out, in)

        # Standard linear path
        linear_out = F.linear(x, W, self.bias)

        # Quantum magnitude path: uses both cos and sin components
        r = torch.exp(self.log_amplitude)
        W_sin = r * torch.sin(self.phase)
        cos_out = F.linear(x, W, None)   # real part
        sin_out = F.linear(x, W_sin, None)  # imaginary part
        magnitude_out = torch.sqrt(cos_out ** 2 + sin_out ** 2 + 1e-8)

        if self.bias is not None:
            magnitude_out = magnitude_out + self.bias

        # Gated mixture
        gate = torch.sigmoid(self.gate)
        out = gate * magnitude_out + (1 - gate) * linear_out

        return out


class QuantumInspiredBlock(nn.Module):
    """
    A block using QuantumInspiredLinear as drop-in for the final fusion/classification layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            QuantumInspiredLinear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            QuantumInspiredLinear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x).squeeze(-1)
