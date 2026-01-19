"""Grouped Query Attention (GQA) with Flash Attention and QKNorm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.normalization import QKNorm
from src.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with optional QKNorm.

    GQA reduces memory and compute by using fewer key-value heads than query heads.
    Each group of query heads shares the same key-value head.

    Uses PyTorch's scaled_dot_product_attention for Flash Attention when available.

    Reference: https://arxiv.org/abs/2305.13245
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        use_qk_norm: bool = True,
        rope: RotaryEmbedding | None = None,
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
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass through the attention layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch, 1, seq_len, seq_len)
                  or (1, 1, seq_len, seq_len) for causal mask
            start_pos: Starting position for RoPE (used in KV-cache inference)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_kv_heads)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos)

        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None),
            scale=1.0 / (self.head_dim ** 0.5),
        )

        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

        return self.wo(attn_output)
