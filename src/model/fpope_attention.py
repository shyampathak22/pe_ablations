"""FPoPE-aware Grouped Query Attention with Content+Position Split.

This module implements the Cinnamon-style architecture where:
- Content path: Standard projections with NO positional encoding
- Position path: Projections with PoPE (Polar Position Embedding) applied
- Final attention: Concatenate content + position, use standard dot-product

Key insight: After the PoPE transform (softplus → cos/sin), we get:
  q = concat(q_c, q_r)  # Content + Position
  k = concat(k_c, k_r)
  score = q @ k.T = q_c @ k_c.T + q_r @ k_r.T

This allows us to use standard attention mechanisms (including Flash Attention)
instead of custom kernels, while still having position-aware attention.

References:
- Cinnamon PoPE: /home/hyperion/code/projects/cinnamon/src/attention.py
- PoPE paper: Polar Position Embedding for Length Generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.fpope import FoPEPoPEEmbedding
from src.model.normalization import RMSNorm


class FPoPEGroupedQueryAttention(nn.Module):
    """Grouped Query Attention with FoPE+PoPE using content+position split.

    Architecture (following Cinnamon):
    - Content projections (wq_c, wk_c): Standard Q/K with NO positional encoding
    - Position projections (wq_r, wk_r): Q/K with PoPE (softplus → cos/sin)
    - Normalization (qr_norm, kr_norm): Applied before PoPE for stability
    - Value projection (wv): Full head_dim
    - Output projection (wo): d_content + d_rope*2 → hidden_dim

    The final query/key are:
        q = concat(q_c, q_r)  where q_r = PoPE(wq_r(x)) → 2*d_rope dims
        k = concat(k_c, k_r)  where k_r = PoPE(wk_r(x)) → 2*d_rope dims

    Attention score = (q_c @ k_c.T + q_r @ k_r.T) / sqrt(d_content + 2*d_rope)

    This is just standard dot-product attention after concatenation!

    Args:
        hidden_dim: Model hidden dimension
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads (for GQA)
        head_dim: Dimension per attention head (for content path)
        d_rope: Dimension for position encoding (PoPE doubles this to 2*d_rope)
        use_qk_norm: Whether to apply RMSNorm to position projections
        fpope: FoPEPoPEEmbedding module for frequency/phase computation
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 64,
        d_rope: int = 32,
        use_qk_norm: bool = True,
        fpope: FoPEPoPEEmbedding | None = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.d_rope = d_rope
        self.d_content = head_dim  # Content uses full head_dim
        self.num_groups = num_heads // num_kv_heads

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # Content projections (NO positional encoding applied)
        self.wq_c = nn.Linear(hidden_dim, num_heads * self.d_content, bias=False)
        self.wk_c = nn.Linear(hidden_dim, num_kv_heads * self.d_content, bias=False)

        # Position projections (PoPE applied here)
        self.wq_r = nn.Linear(hidden_dim, num_heads * d_rope, bias=False)
        self.wk_r = nn.Linear(hidden_dim, num_kv_heads * d_rope, bias=False)

        # Normalization before PoPE (critical for stability, following Cinnamon)
        if use_qk_norm:
            self.qr_norm = RMSNorm(num_heads * d_rope)
            self.kr_norm = RMSNorm(num_kv_heads * d_rope)
        else:
            self.qr_norm = None
            self.kr_norm = None

        # Value uses full head_dim
        self.wv = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)

        # Output projection: based on value dimension (head_dim), not query/key dimension
        # The concatenation of content+position is only for query/key scoring
        # Value and output use standard head_dim
        self.wo = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # FoPE+PoPE embedding
        self.fpope = fpope

        # Attention scale: based on full query/key dimension
        self.scale = (self.d_content + d_rope * 2) ** -0.5

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
        B, S, _ = x.shape

        # Content projections (standard attention, no PE)
        q_c = self.wq_c(x).view(B, S, self.num_heads, self.d_content).transpose(1, 2)
        k_c = self.wk_c(x).view(B, S, self.num_kv_heads, self.d_content).transpose(1, 2)

        # Position projections with normalization
        q_r_proj = self.wq_r(x)
        k_r_proj = self.wk_r(x)

        if self.qr_norm is not None:
            q_r_proj = self.qr_norm(q_r_proj)
            k_r_proj = self.kr_norm(k_r_proj)

        q_r = q_r_proj.view(B, S, self.num_heads, self.d_rope)
        k_r = k_r_proj.view(B, S, self.num_kv_heads, self.d_rope)

        # Apply PoPE transform via forward_query/forward_key
        # Query: [μ·cos(t×θ), μ·sin(t×θ)] - no delta
        # Key:   [μ·cos(s×θ+δ), μ·sin(s×θ+δ)] - with delta
        if self.fpope is not None:
            q_r = self.fpope.forward_query(q_r, start_pos)  # (B, S, H, d_rope*2)
            k_r = self.fpope.forward_key(k_r, start_pos)    # (B, S, Hkv, d_rope*2)
        else:
            # Fallback: no PoPE, just duplicate dimensions (for testing)
            q_r = torch.cat([q_r, q_r], dim=-1)
            k_r = torch.cat([k_r, k_r], dim=-1)

        q_r = q_r.transpose(1, 2)  # (B, H, S, d_rope*2)
        k_r = k_r.transpose(1, 2)

        # Value
        v = self.wv(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expansion: repeat K/V for each query group
        if self.num_groups > 1:
            k_c = k_c.repeat_interleave(self.num_groups, dim=1)
            k_r = k_r.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Concatenate content + position
        q = torch.cat([q_c, q_r], dim=-1)  # (B, H, S, d_content + d_rope*2)
        k = torch.cat([k_c, k_r], dim=-1)

        # Standard scaled dot-product attention (no custom kernel!)
        # This can use Flash Attention via PyTorch's SDPA
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        output = torch.matmul(attn_probs, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(output)


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
        head_dim: int = 64,
        d_rope: int = 32,
        use_qk_norm: bool = True,
        fpope: FoPEPoPEEmbedding | None = None,
        max_cache_len: int = 2048,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            d_rope=d_rope,
            use_qk_norm=use_qk_norm,
            fpope=fpope,
        )

        self.max_cache_len = max_cache_len

        # KV cache buffers (will be initialized on first forward)
        # Note: We cache the concatenated (content + position) keys
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

        B, S, _ = x.shape

        # Content projections
        q_c = self.wq_c(x).view(B, S, self.num_heads, self.d_content).transpose(1, 2)
        k_c = self.wk_c(x).view(B, S, self.num_kv_heads, self.d_content).transpose(1, 2)

        # Position projections with normalization
        q_r_proj = self.wq_r(x)
        k_r_proj = self.wk_r(x)

        if self.qr_norm is not None:
            q_r_proj = self.qr_norm(q_r_proj)
            k_r_proj = self.kr_norm(k_r_proj)

        q_r = q_r_proj.view(B, S, self.num_heads, self.d_rope)
        k_r = k_r_proj.view(B, S, self.num_kv_heads, self.d_rope)

        # Apply PoPE transform
        if self.fpope is not None:
            q_r = self.fpope.forward_query(q_r, start_pos)
            k_r = self.fpope.forward_key(k_r, start_pos)
        else:
            q_r = torch.cat([q_r, q_r], dim=-1)
            k_r = torch.cat([k_r, k_r], dim=-1)

        q_r = q_r.transpose(1, 2)
        k_r = k_r.transpose(1, 2)

        # Value
        v = self.wv(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Concatenate content + position for keys
        k_new = torch.cat([k_c, k_r], dim=-1)  # (B, Hkv, S, d_content + d_rope*2)

        # Initialize cache if needed
        key_dim = self.d_content + self.d_rope * 2
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                B, self.num_kv_heads, self.max_cache_len, key_dim,
                dtype=k_new.dtype, device=k_new.device,
            )
            self.v_cache = torch.zeros(
                B, self.num_kv_heads, self.max_cache_len, self.head_dim,
                dtype=v.dtype, device=v.device,
            )

        # Update cache
        self.k_cache[:, :, start_pos:start_pos + S] = k_new
        self.v_cache[:, :, start_pos:start_pos + S] = v
        self.cache_len = start_pos + S

        # Get cached K, V
        k = self.k_cache[:, :, :self.cache_len]
        v = self.v_cache[:, :, :self.cache_len]

        # GQA expansion
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Concatenate query content + position
        q = torch.cat([q_c, q_r], dim=-1)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask for cached attention
        total_len = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(S, total_len, device=x.device, dtype=torch.bool),
            diagonal=total_len - S + 1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        output = torch.matmul(attn_probs, v)

        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(output)
