"""FPoPE-aware Grouped Query Attention.

This module provides a GQA implementation that uses FoPE+PoPE positional encoding
instead of standard RoPE. The key difference is that position information is
encoded in the attention score computation via cosine phases, rather than
being applied to Q/K before the dot product.
"""

import torch
import torch.nn as nn
from einops import rearrange

from src.model.fpope import FoPEPoPEEmbedding
from src.model.normalization import QKNorm
from src.model.kernels.fpope_attention import fpope_attention_forward


class FPoPEGroupedQueryAttention(nn.Module):
    """Grouped Query Attention with FoPE+PoPE positional encoding.

    Unlike standard GQA which applies RoPE to Q/K before attention,
    FPoPE-GQA computes attention scores using:
        a_{t,s} = Σ_c softplus(q_{t,c}) × softplus(k_{s,c}) × cos((s-t)×ω_c + δ_c)

    This decouples content (magnitudes from softplus) from position (phases from cos).

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads (for GQA)
        head_dim: Dimension per attention head
        use_qk_norm: Whether to apply QKNorm (applied before softplus)
        fpope: FoPEPoPEEmbedding module for frequency/phase computation
        use_triton: Whether to use Triton kernel when available
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        use_qk_norm: bool = True,
        fpope: FoPEPoPEEmbedding | None = None,
        use_triton: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.num_groups = num_heads // num_kv_heads
        self.use_triton = use_triton

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # Linear projections
        self.wq = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # QKNorm is applied before softplus in FPoPE
        # This helps stabilize the magnitude computation
        self.qk_norm = QKNorm(self.head_dim) if use_qk_norm else None

        # FoPE+PoPE embedding
        self.fpope = fpope

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass through the FPoPE attention layer.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask (not used with causal=True)
            start_pos: Starting position for KV-cache inference

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to (batch, seq, heads, dim)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_kv_heads)

        # Reshape to (batch, heads, seq, dim) for attention
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        # Apply QKNorm before the softplus (if enabled)
        # This normalizes the raw projections before magnitude computation
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Expand KV heads for GQA
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Get frequencies and phase bias from FoPE-PoPE
        if self.fpope is not None:
            effective_freqs = self.fpope._compute_effective_freqs()
            phase_bias = self.fpope.phase_bias
        else:
            # Fallback: use default frequencies (like standard RoPE)
            effective_freqs = torch.ones(self.head_dim, device=x.device)
            phase_bias = torch.zeros(self.head_dim, device=x.device)

        # Compute FPoPE attention
        # Note: softplus is applied inside the attention kernel
        attn_output = fpope_attention_forward(
            q=q,
            k=k,
            v=v,
            freqs=effective_freqs,
            phase_bias=phase_bias,
            start_pos=start_pos,
            scale=1.0 / (self.head_dim ** 0.5),
            causal=(mask is None),
            use_triton=self.use_triton,
        )

        # Reshape back to (batch, seq, heads * dim)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

        # Output projection
        return self.wo(attn_output)


class FPoPEGroupedQueryAttentionWithCache(FPoPEGroupedQueryAttention):
    """FPoPE-GQA with KV-cache support for efficient inference.

    Extends FPoPEGroupedQueryAttention with caching of key/value tensors
    for autoregressive generation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        use_qk_norm: bool = True,
        fpope: FoPEPoPEEmbedding | None = None,
        use_triton: bool = True,
        max_cache_len: int = 2048,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            fpope=fpope,
            use_triton=use_triton,
        )

        self.max_cache_len = max_cache_len

        # KV cache buffers (will be initialized on first forward)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.cache_len = 0

    def reset_cache(self) -> None:
        """Reset the KV cache."""
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional KV-cache.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask
            start_pos: Starting position (used for cache indexing)
            use_cache: Whether to use KV-cache

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        if not use_cache:
            return super().forward(x, mask, start_pos)

        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_kv_heads)

        # Apply QKNorm
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Initialize cache if needed
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.max_cache_len,
                self.head_dim,
                dtype=k.dtype,
                device=k.device,
            )
            self.v_cache = torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.max_cache_len,
                self.head_dim,
                dtype=v.dtype,
                device=v.device,
            )

        # Update cache
        self.k_cache[:, :, start_pos : start_pos + seq_len] = k
        self.v_cache[:, :, start_pos : start_pos + seq_len] = v
        self.cache_len = start_pos + seq_len

        # Get cached K, V
        k = self.k_cache[:, :, : self.cache_len]
        v = self.v_cache[:, :, : self.cache_len]

        # Expand for GQA
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Get frequencies and phase bias
        if self.fpope is not None:
            effective_freqs = self.fpope._compute_effective_freqs()
            phase_bias = self.fpope.phase_bias
        else:
            effective_freqs = torch.ones(self.head_dim, device=x.device)
            phase_bias = torch.zeros(self.head_dim, device=x.device)

        # Compute attention
        attn_output = fpope_attention_forward(
            q=q,
            k=k,
            v=v,
            freqs=effective_freqs,
            phase_bias=phase_bias,
            start_pos=start_pos,
            scale=1.0 / (self.head_dim ** 0.5),
            causal=True,
            use_triton=self.use_triton,
        )

        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        return self.wo(attn_output)
