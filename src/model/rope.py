"""Rotary Position Embeddings (RoPE) implementation.

This module provides a modular implementation of RoPE that can be extended
for various position encoding experiments (NTK-aware scaling, YaRN, etc.).
"""

import torch
import torch.nn as nn


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the frequency tensor for RoPE.

    Args:
        dim: Dimension of the head (must be even)
        max_seq_len: Maximum sequence length to precompute
        theta: Base for the frequency computation (default: 10000)
        device: Device to create tensor on

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2) containing frequencies
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim // 2)

    Returns:
        Tuple of (rotated_query, rotated_key) with same shapes as inputs
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module.

    Encapsulates RoPE computation with caching for efficiency.
    Supports dynamic sequence lengths up to max_seq_len.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings.

        Args:
            xq: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            xk: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
            start_pos: Starting position for the sequence (for KV-cache inference)

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        seq_len = xq.shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        return apply_rotary_emb(xq, xk, freqs_cis)

    def extend_seq_len(self, new_max_seq_len: int) -> None:
        """Extend the cached frequencies to support longer sequences.

        Args:
            new_max_seq_len: New maximum sequence length
        """
        if new_max_seq_len <= self.max_seq_len:
            return

        freqs_cis = precompute_freqs_cis(
            self.dim,
            new_max_seq_len,
            self.theta,
            device=self.freqs_cis.device,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.max_seq_len = new_max_seq_len


class NTKAwareRotaryEmbedding(RotaryEmbedding):
    """NTK-aware interpolation for RoPE.

    Scales the base theta to enable length extrapolation without fine-tuning.
    This allows models trained on shorter sequences to handle longer ones.

    Reference: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scale: float = 1.0,
    ):
        self.scale = scale
        scaled_theta = theta * (scale ** (dim / (dim - 2)))
        super().__init__(dim, max_seq_len, scaled_theta)


class YaRNRotaryEmbedding(RotaryEmbedding):
    """YaRN (Yet another RoPE extensioN) implementation.

    Combines NTK-aware interpolation with attention scaling for better
    length extrapolation performance.

    Reference: https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scale: float = 1.0,
        original_max_seq_len: int = 512,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__(dim, max_seq_len, theta)
        self.scale = scale
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self._compute_yarn_freqs()

    def _compute_yarn_freqs(self) -> None:
        """Compute YaRN-adjusted frequencies."""
        dim = self.dim
        max_seq_len = self.max_seq_len
        theta = self.theta
        scale = self.scale

        pos_freqs = theta ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low = max(
            int(torch.floor(dim * torch.log(torch.tensor(self.original_max_seq_len / (self.beta_fast * 2 * torch.pi))) / torch.log(torch.tensor(theta)))),
            0,
        )
        high = min(
            int(torch.ceil(dim * torch.log(torch.tensor(self.original_max_seq_len / (self.beta_slow * 2 * torch.pi))) / torch.log(torch.tensor(theta)))),
            dim - 1,
        )

        inv_freq = inv_freq_interpolation.clone()
        inv_freq[:low] = inv_freq_extrapolation[:low]

        if low < high:
            smooth = (torch.arange(low, high + 1).float() - low) / (high - low)
            inv_freq[low : high + 1] = (
                (1 - smooth) * inv_freq_extrapolation[low : high + 1]
                + smooth * inv_freq_interpolation[low : high + 1]
            )

        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        mscale = 0.1 * torch.log(torch.tensor(scale)) + 1.0
        self.register_buffer("mscale", mscale, persistent=False)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply YaRN rotary embeddings with attention scaling."""
        xq_out, xk_out = super().forward(xq, xk, start_pos)
        return xq_out * self.mscale, xk_out * self.mscale
