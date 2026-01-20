"""Attention with Linear Biases (ALiBi) implementation.

Reference: Press et al., "Train Short, Test Long"
https://arxiv.org/abs/2108.12409

ALiBi adds linear biases to attention scores based on relative position distance.
No learnable position embeddings - uses fixed slopes per head.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.normalization import QKNorm


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each attention head.

    Uses geometric sequence: 2^(-8/n), 2^(-16/n), ... for n heads.
    For non-power-of-2 head counts, interpolates between closest powers.

    Args:
        num_heads: Number of attention heads

    Returns:
        Tensor of slopes with shape (num_heads,)
    """
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    else:
        # For non-power-of-2, interpolate
        closest_power = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power)
        extra = get_slopes_power_of_2(2 * closest_power)[0::2][:num_heads - closest_power]
        return torch.tensor(slopes + extra)


class ALiBiAttention(nn.Module):
    """Grouped Query Attention with ALiBi positional biases.

    Combines ALiBi for positional information with Grouped Query Attention
    for efficiency. Uses QKNorm for training stability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.num_groups = num_heads // num_kv_heads

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.wq = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        self.qk_norm = QKNorm(self.head_dim) if use_qk_norm else None

        # ALiBi slopes (fixed, not learned)
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer("alibi_slopes", slopes.view(1, num_heads, 1, 1))

        # Cache for ALiBi bias matrix
        self._cached_bias: torch.Tensor | None = None
        self._cached_seq_len: int = 0

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get ALiBi bias matrix of shape (1, num_heads, seq_len, seq_len).

        Args:
            seq_len: Sequence length
            device: Target device

        Returns:
            ALiBi bias tensor
        """
        if self._cached_bias is not None and self._cached_seq_len >= seq_len:
            return self._cached_bias[:, :, :seq_len, :seq_len]

        # Relative positions: positions[i,j] = j - i
        positions = torch.arange(seq_len, device=device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)

        # ALiBi bias = -slope * |relative_pos|
        # For causal attention, future positions are masked anyway
        bias = -self.alibi_slopes.to(device) * relative_pos.abs().float()

        self._cached_bias = bias
        self._cached_seq_len = seq_len
        return bias

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass through the attention layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask (unused - ALiBi uses causal)
            start_pos: Starting position (unused - ALiBi handles position internally)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_kv_heads)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Attention scores with ALiBi bias
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add ALiBi bias
        alibi_bias = self._get_alibi_bias(seq_len, x.device)
        scores = scores + alibi_bias

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = rearrange(output, "b h s d -> b s (h d)")
        return self.wo(output)
