"""FPoPE (FoPE + PoPE) Attention Kernel.

This module provides optimized attention computation for FoPE+PoPE positional encoding.
The key difference from standard attention is that attention scores are computed as:

    a_{t,s} = Σ_c softplus(q_{t,c}) × softplus(k_{s,c}) × cos((s-t)×ω_c + δ_c)

This requires:
1. Softplus activation on Q and K (for magnitudes)
2. Phase computation from position differences
3. Cosine modulation of the magnitude product

Provides both Triton-optimized and pure PyTorch implementations.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd import Function

# Try to import Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
    @triton.jit
    def _fpope_attention_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,  # Input pointers
        Freqs_ptr, PhaseBias_ptr,  # Positional encoding pointers
        Out_ptr,  # Output pointer
        stride_qb, stride_qh, stride_qs, stride_qd,  # Q strides
        stride_kb, stride_kh, stride_ks, stride_kd,  # K strides
        stride_vb, stride_vh, stride_vs, stride_vd,  # V strides
        stride_ob, stride_oh, stride_os, stride_od,  # Output strides
        seq_len_q: tl.constexpr,
        seq_len_k: tl.constexpr,
        head_dim: tl.constexpr,
        start_pos,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D_TILE: tl.constexpr,  # Tile size for dimension loop (score computation)
        BLOCK_D_OUT: tl.constexpr,   # Output dimension size (must equal head_dim, power of 2)
    ):
        """FPoPE attention kernel with tiled dimension loop.

        Grid: (cdiv(seq_len_q, BLOCK_M), batch * num_heads)

        Uses BLOCK_D_TILE for tiling the dimension loop in score computation
        (keeps 3D tensor small), and BLOCK_D_OUT for V/output handling.
        BLOCK_D_OUT must equal head_dim.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d_out = tl.arange(0, BLOCK_D_OUT)

        # Online softmax accumulators - sized to full head_dim
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D_OUT], dtype=tl.float32)

        pos_q = (start_pos + offs_m).to(tl.float32)

        # Iterate over K/V blocks
        for start_n in range(0, seq_len_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            pos_k = (start_pos + offs_n).to(tl.float32)
            pos_diff = pos_k[None, :] - pos_q[:, None]  # (BLOCK_M, BLOCK_N)

            # Compute attention scores by tiling over dimensions (keeps 3D tensor small)
            scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for d_start in range(0, head_dim, BLOCK_D_TILE):
                offs_d_tile = d_start + tl.arange(0, BLOCK_D_TILE)
                d_mask = offs_d_tile < head_dim

                # Load frequencies and phase bias for this tile
                freq_d = tl.load(Freqs_ptr + offs_d_tile, mask=d_mask, other=0.0)
                bias_d = tl.load(PhaseBias_ptr + offs_d_tile, mask=d_mask, other=0.0)

                # Load Q slice (BLOCK_M, BLOCK_D_TILE)
                q_ptrs = Q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d_tile[None, :]
                q_mask = (offs_m[:, None] < seq_len_q) & d_mask[None, :]
                q_d = tl.load(q_ptrs, mask=q_mask, other=0.0)

                # Softplus on Q tile
                q_abs = tl.abs(q_d)
                mu_q_d = tl.maximum(q_d, 0.0) + tl.log(1.0 + tl.exp(-q_abs))

                # Load K slice (BLOCK_N, BLOCK_D_TILE)
                k_ptrs = K_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_d_tile[None, :]
                k_mask = (offs_n[:, None] < seq_len_k) & d_mask[None, :]
                k_d = tl.load(k_ptrs, mask=k_mask, other=0.0)

                # Softplus on K tile
                k_abs = tl.abs(k_d)
                mu_k_d = tl.maximum(k_d, 0.0) + tl.log(1.0 + tl.exp(-k_abs))

                # Phase: (BLOCK_M, BLOCK_N, BLOCK_D_TILE) - small 3D tensor
                phase = pos_diff[:, :, None] * freq_d[None, None, :] + bias_d[None, None, :]
                cos_phase = tl.cos(phase)

                # Accumulate scores: sum over this dimension tile
                contrib = mu_q_d[:, None, :] * mu_k_d[None, :, :] * cos_phase
                scores += tl.sum(contrib, axis=2)

            # Scale and causal mask
            scores = scores * scale
            causal_mask = offs_m[:, None] < offs_n[None, :]
            scores = tl.where(causal_mask, float("-inf"), scores)

            # Online softmax
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            p = tl.exp(scores - m_ij[:, None])

            # Load V with FULL head_dim (BLOCK_N, BLOCK_D_OUT)
            v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vs + offs_d_out[None, :]
            v_mask = (offs_n[:, None] < seq_len_k) & (offs_d_out[None, :] < head_dim)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + beta * tl.sum(p, axis=1)
            m_i = m_new

        # Normalize and store with FULL head_dim
        acc = acc / l_i[:, None]
        out_ptrs = Out_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_os + offs_d_out[None, :]
        out_mask = (offs_m[:, None] < seq_len_q) & (offs_d_out[None, :] < head_dim)
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


def triton_fpope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    start_pos: int = 0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Triton-accelerated FPoPE attention.

    Args:
        q: Query tensor (batch, heads, seq_q, dim)
        k: Key tensor (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        freqs: Effective frequencies (dim,)
        phase_bias: Phase bias (dim,)
        start_pos: Starting position for KV-cache
        scale: Attention scale factor

    Returns:
        Output tensor (batch, heads, seq_q, dim)
    """
    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Triton requires BLOCK_D_OUT to be a power of 2 and equal to head_dim
    # Fall back to PyTorch for non-power-of-2 or large dimensions
    is_power_of_2 = (dim & (dim - 1)) == 0 and dim > 0
    if not is_power_of_2 or dim > 128:
        return pytorch_fpope_attention(q, k, v, freqs, phase_bias, start_pos, scale, causal=True)

    output = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D_TILE = 32  # Small tile for dimension loop (keeps 3D tensor small: 64x64x32 = 128K)
    BLOCK_D_OUT = dim  # Full head_dim for V and output (must be power of 2)

    grid = (triton.cdiv(seq_q, BLOCK_M), batch * heads)

    _fpope_attention_fwd_kernel[grid](
        q, k, v,
        freqs, phase_bias,
        output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_q, seq_k, dim,
        start_pos,
        scale,
        BLOCK_M, BLOCK_N, BLOCK_D_TILE, BLOCK_D_OUT,
    )

    return output


def pytorch_fpope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    start_pos: int = 0,
    scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Pure PyTorch FPoPE attention (fallback implementation).

    This is a reference implementation that works on all devices but may be slower
    than the Triton kernel for large sequences.

    Args:
        q: Query tensor (batch, heads, seq_q, dim)
        k: Key tensor (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        freqs: Effective frequencies (dim,)
        phase_bias: Phase bias (dim,)
        start_pos: Starting position for KV-cache
        scale: Attention scale factor
        causal: Whether to apply causal masking

    Returns:
        Output tensor (batch, heads, seq_q, dim)
    """
    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Apply softplus for magnitudes
    mu_q = F.softplus(q)  # (batch, heads, seq_q, dim)
    mu_k = F.softplus(k)  # (batch, heads, seq_k, dim)

    # Compute position differences
    pos_q = torch.arange(start_pos, start_pos + seq_q, device=q.device, dtype=q.dtype)
    pos_k = torch.arange(start_pos, start_pos + seq_k, device=k.device, dtype=k.dtype)
    pos_diff = pos_k.unsqueeze(0) - pos_q.unsqueeze(1)  # (seq_q, seq_k)

    # Compute phases: (seq_q, seq_k, dim)
    phases = pos_diff.unsqueeze(-1) * freqs + phase_bias

    # Compute cosines
    cos_phases = torch.cos(phases)  # (seq_q, seq_k, dim)

    # Attention scores: sum over dim of (mu_q * mu_k * cos)
    # mu_q: (batch, heads, seq_q, 1, dim)
    # mu_k: (batch, heads, 1, seq_k, dim)
    # cos_phases: (1, 1, seq_q, seq_k, dim)
    mu_q_exp = mu_q.unsqueeze(3)
    mu_k_exp = mu_k.unsqueeze(2)
    cos_exp = cos_phases.unsqueeze(0).unsqueeze(0)

    attn_scores = (mu_q_exp * mu_k_exp * cos_exp).sum(dim=-1)  # (batch, heads, seq_q, seq_k)
    attn_scores = attn_scores * scale

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    # Softmax and apply to values
    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)

    return output


class FPoPEAttentionFunction(Function):
    """Autograd function for FPoPE attention with custom backward."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs: torch.Tensor,
        phase_bias: torch.Tensor,
        start_pos: int,
        scale: Optional[float],
        causal: bool,
        use_triton: bool,
    ) -> torch.Tensor:
        """Forward pass."""
        if use_triton and HAS_TRITON and q.is_cuda:
            output = triton_fpope_attention(
                q, k, v, freqs, phase_bias, start_pos, scale
            )
        else:
            output = pytorch_fpope_attention(
                q, k, v, freqs, phase_bias, start_pos, scale, causal
            )

        # Save for backward
        ctx.save_for_backward(q, k, v, freqs, phase_bias, output)
        ctx.start_pos = start_pos
        ctx.scale = scale
        ctx.causal = causal

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using PyTorch autograd on the reference implementation."""
        q, k, v, freqs, phase_bias, output = ctx.saved_tensors

        # Recompute with autograd for gradients
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            freqs = freqs.detach().requires_grad_(True)
            phase_bias = phase_bias.detach().requires_grad_(True)

            output_recomputed = pytorch_fpope_attention(
                q, k, v, freqs, phase_bias, ctx.start_pos, ctx.scale, ctx.causal
            )

            output_recomputed.backward(grad_output)

        return q.grad, k.grad, v.grad, freqs.grad, phase_bias.grad, None, None, None, None


def fpope_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    start_pos: int = 0,
    scale: Optional[float] = None,
    causal: bool = True,
    use_triton: bool = True,
) -> torch.Tensor:
    """Main entry point for FPoPE attention.

    Automatically selects between Triton and PyTorch implementations.

    Args:
        q: Query tensor (batch, heads, seq_q, dim) - raw projections
        k: Key tensor (batch, heads, seq_k, dim) - raw projections
        v: Value tensor (batch, heads, seq_k, dim)
        freqs: Effective frequencies (dim,)
        phase_bias: Learnable phase bias (dim,)
        start_pos: Starting position for KV-cache inference
        scale: Attention scale factor (default: 1/sqrt(dim))
        causal: Whether to apply causal masking
        use_triton: Whether to use Triton kernel when available

    Returns:
        Attention output (batch, heads, seq_q, dim)
    """
    return FPoPEAttentionFunction.apply(
        q, k, v, freqs, phase_bias, start_pos, scale, causal, use_triton
    )
