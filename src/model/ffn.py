"""Feed-Forward Network with SwiGLU activation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    Combines Swish (SiLU) activation with gated linear units for improved
    performance over standard activations like ReLU or GELU.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor
            gate: Gate tensor (same shape as x)

        Returns:
            Activated tensor: silu(gate) * x
        """
        return F.silu(gate) * x


class FeedForward(nn.Module):
    """Feed-Forward Network with SwiGLU activation.

    Uses the GLU variant where the hidden dimension is split into two parts:
    one for the gate and one for the value, requiring 3 linear projections
    (up, gate, down) instead of 2.

    The intermediate dimension is scaled by 2/3 to maintain parameter count
    comparable to a standard FFN: 4 * hidden_dim * (2/3) * 3 ≈ 8 * hidden_dim
    vs standard 8 * hidden_dim.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int | None = None,
        multiple_of: int = 256,
    ):
        super().__init__()

        if ffn_dim is None:
            ffn_dim = int(4 * hidden_dim * 2 / 3)
            ffn_dim = multiple_of * ((ffn_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.activation = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        return self.w2(self.activation(self.w1(x), self.w3(x)))
