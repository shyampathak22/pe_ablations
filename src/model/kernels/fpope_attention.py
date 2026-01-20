"""FPoPE (FoPE + PoPE) Attention Kernel.

This module provides optimized attention computation for FoPE+PoPE positional encoding.
The key difference from standard attention is that attention scores are computed as:

    a_{t,s} = Σ_c softplus(q_{t,c}) × softplus(k_{s,c}) × cos((s-t)×ω_c + δ_c)

This requires:
1. Softplus activation on Q and K (for magnitudes)
2. Phase computation from position differences
3. Cosine modulation of the magnitude product

Provides implementations in order of preference:
1. Native CUDA kernels (fastest, 15-25% MFU)
2. Triton kernels (slower due to 3D tensor compilation overhead)
3. Pure PyTorch (reference implementation)
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd import Function

# Try to import native CUDA kernel (built via setup.py)
try:
    import fpope_cuda
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False
    fpope_cuda = None

# Try to import Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def _precompute_phase_tables(
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    max_pos_diff: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine lookup tables for position differences.

    Args:
        freqs: Effective frequencies (dim,)
        phase_bias: Phase bias (dim,)
        max_pos_diff: Maximum position difference (typically seq_len)

    Returns:
        Tuple of (cos_table, sin_table), each of shape (2 * max_pos_diff, dim)
        Index with [pos_diff + max_pos_diff] to get cos/sin values.
    """
    dim = freqs.shape[0]
    device = freqs.device
    dtype = freqs.dtype

    # Position differences range from -max_pos_diff+1 to max_pos_diff-1
    # We use range -max_pos_diff to max_pos_diff-1 for indexing simplicity
    pos_diffs = torch.arange(-max_pos_diff, max_pos_diff, device=device, dtype=dtype)

    # Compute phases: (2*max_pos_diff, dim)
    # phase[i, d] = pos_diffs[i] * freqs[d] + phase_bias[d]
    phases = pos_diffs[:, None] * freqs[None, :] + phase_bias[None, :]

    cos_table = torch.cos(phases)
    sin_table = torch.sin(phases)

    return cos_table, sin_table


if HAS_TRITON:
    @triton.jit
    def _fpope_attention_fwd_kernel(
        MuQ_ptr, MuK_ptr, V_ptr,  # Input pointers (precomputed softplus)
        Freqs_ptr, PhaseBias_ptr,  # Positional encoding pointers
        Out_ptr,  # Output pointer
        LSE_ptr,  # Logsumexp output for backward (batch*heads, seq_q)
        stride_qb, stride_qh, stride_qs, stride_qd,  # MuQ strides
        stride_kb, stride_kh, stride_ks, stride_kd,  # MuK strides
        stride_vb, stride_vh, stride_vs, stride_vd,  # V strides
        stride_ob, stride_oh, stride_os, stride_od,  # Output strides
        stride_lse_bh, stride_lse_m,  # LSE strides
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

        Uses small BLOCK_D_TILE for tiling the dimension loop in score computation
        to keep 3D tensor small. BLOCK_D_OUT must equal head_dim.

        NOTE: Expects precomputed softplus values (mu_q, mu_k) instead of raw Q/K.
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

                # Load precomputed mu_q (BLOCK_M, BLOCK_D_TILE)
                muq_ptrs = MuQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d_tile[None, :]
                q_mask = (offs_m[:, None] < seq_len_q) & d_mask[None, :]
                mu_q_d = tl.load(muq_ptrs, mask=q_mask, other=0.0)

                # Load precomputed mu_k (BLOCK_N, BLOCK_D_TILE)
                muk_ptrs = MuK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_d_tile[None, :]
                k_mask = (offs_n[:, None] < seq_len_k) & d_mask[None, :]
                mu_k_d = tl.load(muk_ptrs, mask=k_mask, other=0.0)

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

            # Online softmax with NaN guard
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            p = tl.where(m_ij[:, None] == float("-inf"), 0.0, tl.exp(scores - m_ij[:, None]))

            # Load V with FULL head_dim (BLOCK_N, BLOCK_D_OUT)
            v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vs + offs_d_out[None, :]
            v_mask = (offs_n[:, None] < seq_len_k) & (offs_d_out[None, :] < head_dim)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
            l_i = l_i * alpha + beta * tl.sum(p, axis=1)
            m_i = m_new

        # Normalize and store
        acc = acc / l_i[:, None]
        out_ptrs = Out_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_os + offs_d_out[None, :]
        out_mask = (offs_m[:, None] < seq_len_q) & (offs_d_out[None, :] < head_dim)
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)

        # Store logsumexp for backward pass
        lse = m_i + tl.log(l_i)
        lse_ptrs = LSE_ptr + pid_bh * stride_lse_bh + offs_m * stride_lse_m
        lse_mask = offs_m < seq_len_q
        tl.store(lse_ptrs, lse, mask=lse_mask)


def triton_fpope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    start_pos: int = 0,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _precomputed: Optional[tuple] = None,  # (mu_q, mu_k) if already computed
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Triton-accelerated FPoPE attention.

    Args:
        q: Query tensor (batch, heads, seq_q, dim)
        k: Key tensor (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        freqs: Effective frequencies (dim,)
        phase_bias: Phase bias (dim,)
        start_pos: Starting position for KV-cache
        scale: Attention scale factor
        return_lse: Whether to return logsumexp for backward pass
        _precomputed: Optional precomputed (mu_q, mu_k) for internal use

    Returns:
        Tuple of (output tensor, logsumexp tensor or None)
        - output: (batch, heads, seq_q, dim)
        - lse: (batch * heads, seq_q) if return_lse else None
    """
    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Triton requires BLOCK_D_OUT to be a power of 2 and equal to head_dim
    # Fall back to PyTorch for non-power-of-2 or large dimensions
    is_power_of_2 = (dim & (dim - 1)) == 0 and dim > 0
    if not is_power_of_2 or dim > 128:
        out = pytorch_fpope_attention(q, k, v, freqs, phase_bias, start_pos, scale, causal=True)
        return (out, None) if return_lse else out

    # Precompute softplus in PyTorch (avoids recomputation in kernel)
    if _precomputed is not None:
        mu_q, mu_k = _precomputed
    else:
        mu_q = F.softplus(q)
        mu_k = F.softplus(k)

    # Reshape for kernel: (batch, heads, seq, dim) -> (batch*heads, seq, dim)
    mu_q_flat = mu_q.reshape(batch * heads, seq_q, dim)
    mu_k_flat = mu_k.reshape(batch * heads, seq_k, dim)
    v_flat = v.reshape(batch * heads, seq_k, dim)

    output = torch.empty(batch, heads, seq_q, dim, device=q.device, dtype=q.dtype)
    output_flat = output.reshape(batch * heads, seq_q, dim)

    # Allocate LSE tensor: (batch * heads, seq_q)
    lse = torch.empty(batch * heads, seq_q, dtype=torch.float32, device=q.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D_TILE = 16  # Small tile for dimension loop (keeps 3D tensor small: 64x64x16 = 64K)
    BLOCK_D_OUT = dim  # Full head_dim for V and output (must be power of 2)

    grid = (triton.cdiv(seq_q, BLOCK_M), batch * heads)

    _fpope_attention_fwd_kernel[grid](
        mu_q_flat, mu_k_flat, v_flat,
        freqs, phase_bias,
        output_flat,
        lse,
        mu_q_flat.stride(0), mu_q_flat.stride(0), mu_q_flat.stride(1), mu_q_flat.stride(2),
        mu_k_flat.stride(0), mu_k_flat.stride(0), mu_k_flat.stride(1), mu_k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        output_flat.stride(0), output_flat.stride(0), output_flat.stride(1), output_flat.stride(2),
        lse.stride(0), lse.stride(1),
        seq_q, seq_k, dim,
        start_pos,
        scale,
        BLOCK_M, BLOCK_N, BLOCK_D_TILE, BLOCK_D_OUT,
    )

    if return_lse:
        return output, lse
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
    attn_probs = F.softmax(attn_scores, dim=-1).to(v.dtype)
    output = torch.matmul(attn_probs, v)

    return output


if HAS_TRITON:
    @triton.jit
    def _fpope_attention_bwd_kernel_dk(
        # Inputs (precomputed softplus and sigmoid)
        MuQ_ptr, MuK_ptr, SigmoidK_ptr, V_ptr,
        Freqs_ptr, PhaseBias_ptr,
        dO_ptr, LSE_ptr,
        # Outputs
        dK_ptr,
        dFreq_partial_ptr, dBias_partial_ptr,  # Per-block partial gradients
        # Strides
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_dkb, stride_dkh, stride_dks, stride_dkd,
        stride_lse_bh, stride_lse_m,
        stride_partial_block, stride_partial_d,  # For partial gradients buffer
        # Dimensions
        seq_len_q: tl.constexpr,
        seq_len_k: tl.constexpr,
        head_dim: tl.constexpr,
        start_pos,
        scale,
        num_blocks_q,  # For partial gradient indexing
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_D_TILE: tl.constexpr,
    ):
        """Backward kernel computing dK and partial dFreq/dBias (NO ATOMICS - local accumulation).

        Grid: (cdiv(seq_len_k, BLOCK_N), batch * heads)

        Each block owns its K positions - accumulates across Q blocks locally,
        then stores once at the end. Uses dimension tiling for score computation,
        full dimension for gradient computation.
        """
        pid_n = tl.program_id(0)  # K block index
        pid_bh = tl.program_id(1)  # batch * head index
        block_id = pid_bh * tl.cdiv(seq_len_k, BLOCK_N) + pid_n

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        pos_k = (start_pos + offs_n).to(tl.float32)

        # Load precomputed mu_k and sigmoid_k for this block (full dimension)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
        muk_ptrs = MuK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :]
        mu_k_full = tl.load(muk_ptrs, mask=k_mask, other=0.0)
        sigmoidk_ptrs = SigmoidK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :]
        sigmoid_k_full = tl.load(sigmoidk_ptrs, mask=k_mask, other=0.0)

        # Load V for dp computation
        v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :]
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load freqs and bias (full dimension)
        freqs_full = tl.load(Freqs_ptr + offs_d, mask=offs_d < head_dim, other=0.0)
        bias_full = tl.load(PhaseBias_ptr + offs_d, mask=offs_d < head_dim, other=0.0)

        # Initialize accumulators - NO ATOMICS
        dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dfreq_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        dbias_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Iterate over Q blocks
        for start_m in range(0, seq_len_q, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            pos_q = (start_pos + offs_m).to(tl.float32)
            pos_diff = pos_k[None, :] - pos_q[:, None]  # (BLOCK_M, BLOCK_N)

            # Load mu_q for this Q block (full dimension)
            q_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
            muq_ptrs = MuQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
            mu_q_full = tl.load(muq_ptrs, mask=q_mask, other=0.0)

            # Load dO and LSE
            do_ptrs = dO_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
            do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)

            lse_ptrs = LSE_ptr + pid_bh * stride_lse_bh + offs_m * stride_lse_m
            lse = tl.load(lse_ptrs, mask=offs_m < seq_len_q, other=0.0)

            # Recompute attention scores with dimension tiling (for memory efficiency)
            scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for d_start in range(0, head_dim, BLOCK_D_TILE):
                offs_chunk = d_start + tl.arange(0, BLOCK_D_TILE)
                chunk_mask = offs_chunk < head_dim

                freqs_chunk = tl.load(Freqs_ptr + offs_chunk, mask=chunk_mask, other=0.0)
                bias_chunk = tl.load(PhaseBias_ptr + offs_chunk, mask=chunk_mask, other=0.0)

                mu_q_chunk = tl.load(
                    MuQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_chunk[None, :],
                    mask=(offs_m[:, None] < seq_len_q) & chunk_mask[None, :], other=0.0
                )
                mu_k_chunk = tl.load(
                    MuK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_chunk[None, :],
                    mask=(offs_n[:, None] < seq_len_k) & chunk_mask[None, :], other=0.0
                )

                phase = pos_diff[:, :, None] * freqs_chunk[None, None, :] + bias_chunk[None, None, :]
                cos_phase = tl.cos(phase)
                contrib = mu_q_chunk[:, None, :] * mu_k_chunk[None, :, :] * cos_phase
                scores += tl.sum(contrib, axis=2)

            scores = scores * scale
            causal_mask = offs_m[:, None] < offs_n[None, :]
            scores = tl.where(causal_mask, float("-inf"), scores)

            p = tl.exp(scores - lse[:, None])
            p = tl.where(causal_mask, 0.0, p)

            dp = tl.dot(do, tl.trans(v))
            D_i = tl.sum(p * dp, axis=1)
            ds = p * (dp - D_i[:, None])
            ds_scaled = ds * scale

            # Compute dK/dfreq/dbias gradients using full dimension 3D tensors
            # phase: (BLOCK_M, BLOCK_N, BLOCK_D)
            phase_full = pos_diff[:, :, None] * freqs_full[None, None, :] + bias_full[None, None, :]
            cos_phase_full = tl.cos(phase_full)
            sin_phase_full = tl.sin(phase_full)

            # dK: d_mu_k = sum_m(ds[m,n] * mu_q[m,d] * cos(phase[m,n,d]))
            d_mu_k = tl.sum(ds_scaled[:, :, None] * mu_q_full[:, None, :] * cos_phase_full, axis=0)
            dk_acc += d_mu_k * sigmoid_k_full

            # dfreq/dbias: d_phase = -ds * mu_q * mu_k * sin(phase)
            d_phase = -ds_scaled[:, :, None] * mu_q_full[:, None, :] * mu_k_full[None, :, :] * sin_phase_full
            # Multi-axis reduction must be done in steps (Triton limitation)
            dfreq_temp = tl.sum(d_phase * pos_diff[:, :, None], axis=0)  # (BLOCK_N, BLOCK_D)
            dfreq_contrib = tl.sum(dfreq_temp, axis=0)  # (BLOCK_D,)
            dbias_temp = tl.sum(d_phase, axis=0)  # (BLOCK_N, BLOCK_D)
            dbias_contrib = tl.sum(dbias_temp, axis=0)  # (BLOCK_D,)
            dfreq_acc += dfreq_contrib
            dbias_acc += dbias_contrib

        # Store dK once after all Q blocks processed - NO ATOMICS
        dk_ptrs = dK_ptr + pid_bh * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :]
        tl.store(dk_ptrs, dk_acc.to(dK_ptr.dtype.element_ty), mask=k_mask)

        # Store partial dFreq/dBias once per block - NO ATOMICS
        partial_freq_ptrs = dFreq_partial_ptr + block_id * stride_partial_block + offs_d * stride_partial_d
        partial_bias_ptrs = dBias_partial_ptr + block_id * stride_partial_block + offs_d * stride_partial_d
        tl.store(partial_freq_ptrs, dfreq_acc.to(dFreq_partial_ptr.dtype.element_ty), mask=offs_d < head_dim)
        tl.store(partial_bias_ptrs, dbias_acc.to(dBias_partial_ptr.dtype.element_ty), mask=offs_d < head_dim)

    @triton.jit
    def _fpope_attention_bwd_kernel_dq(
        # Inputs (precomputed softplus and sigmoid)
        MuQ_ptr, MuK_ptr, SigmoidQ_ptr, V_ptr,
        Freqs_ptr, PhaseBias_ptr,
        dO_ptr, LSE_ptr,
        # Outputs
        dQ_ptr,
        # Strides
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_dqb, stride_dqh, stride_dqs, stride_dqd,
        stride_lse_bh, stride_lse_m,
        # Dimensions
        seq_len_q: tl.constexpr,
        seq_len_k: tl.constexpr,
        head_dim: tl.constexpr,
        start_pos,
        scale,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_D_TILE: tl.constexpr,
    ):
        """Backward kernel computing dQ (NO ATOMICS - local accumulation).

        Grid: (cdiv(seq_len_q, BLOCK_M), batch * heads)

        Each block owns its Q positions - accumulates across K blocks locally,
        then stores once at the end. Uses dimension tiling for score computation,
        full dimension for gradient computation.
        """
        pid_m = tl.program_id(0)  # Q block index
        pid_bh = tl.program_id(1)  # batch * head index

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        pos_q = (start_pos + offs_m).to(tl.float32)

        # Load precomputed mu_q and sigmoid_q for this block (full dimension)
        q_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
        muq_ptrs = MuQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
        mu_q_full = tl.load(muq_ptrs, mask=q_mask, other=0.0)
        sigmoidq_ptrs = SigmoidQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
        sigmoid_q_full = tl.load(sigmoidq_ptrs, mask=q_mask, other=0.0)

        # Load dO for this Q block
        do_ptrs = dO_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
        do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        # Load LSE for this Q block
        lse_ptrs = LSE_ptr + pid_bh * stride_lse_bh + offs_m * stride_lse_m
        lse = tl.load(lse_ptrs, mask=offs_m < seq_len_q, other=0.0)

        # Load freqs and bias (full dimension)
        freqs_full = tl.load(Freqs_ptr + offs_d, mask=offs_d < head_dim, other=0.0)
        bias_full = tl.load(PhaseBias_ptr + offs_d, mask=offs_d < head_dim, other=0.0)

        # Initialize dQ accumulator - NO ATOMICS
        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Iterate over K blocks
        for start_n in range(0, seq_len_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            pos_k = (start_pos + offs_n).to(tl.float32)
            pos_diff = pos_k[None, :] - pos_q[:, None]  # (BLOCK_M, BLOCK_N)

            # Load mu_k for this K block (full dimension)
            k_mask = (offs_n[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
            muk_ptrs = MuK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :]
            mu_k_full = tl.load(muk_ptrs, mask=k_mask, other=0.0)

            # Load V for dp computation
            v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :]
            v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Recompute attention scores with dimension tiling (for memory efficiency)
            scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for d_start in range(0, head_dim, BLOCK_D_TILE):
                offs_chunk = d_start + tl.arange(0, BLOCK_D_TILE)
                chunk_mask = offs_chunk < head_dim

                freqs_chunk = tl.load(Freqs_ptr + offs_chunk, mask=chunk_mask, other=0.0)
                bias_chunk = tl.load(PhaseBias_ptr + offs_chunk, mask=chunk_mask, other=0.0)

                mu_q_chunk = tl.load(
                    MuQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_chunk[None, :],
                    mask=(offs_m[:, None] < seq_len_q) & chunk_mask[None, :], other=0.0
                )
                mu_k_chunk = tl.load(
                    MuK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_chunk[None, :],
                    mask=(offs_n[:, None] < seq_len_k) & chunk_mask[None, :], other=0.0
                )

                phase = pos_diff[:, :, None] * freqs_chunk[None, None, :] + bias_chunk[None, None, :]
                cos_phase = tl.cos(phase)
                contrib = mu_q_chunk[:, None, :] * mu_k_chunk[None, :, :] * cos_phase
                scores += tl.sum(contrib, axis=2)

            scores = scores * scale
            causal_mask = offs_m[:, None] < offs_n[None, :]
            scores = tl.where(causal_mask, float("-inf"), scores)

            p = tl.exp(scores - lse[:, None])
            p = tl.where(causal_mask, 0.0, p)

            dp = tl.dot(do, tl.trans(v))
            D_i = tl.sum(p * dp, axis=1)
            ds = p * (dp - D_i[:, None])
            ds_scaled = ds * scale

            # Compute dQ gradients using full dimension 3D tensors
            # phase: (BLOCK_M, BLOCK_N, BLOCK_D)
            phase_full = pos_diff[:, :, None] * freqs_full[None, None, :] + bias_full[None, None, :]
            cos_phase_full = tl.cos(phase_full)

            # dQ: d_mu_q = sum_n(ds[m,n] * mu_k[n,d] * cos(phase[m,n,d]))
            d_mu_q = tl.sum(ds_scaled[:, :, None] * mu_k_full[None, :, :] * cos_phase_full, axis=1)
            dq_acc += d_mu_q * sigmoid_q_full

        # Store dQ once after all K blocks processed - NO ATOMICS
        dq_ptrs = dQ_ptr + pid_bh * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :]
        tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=q_mask)


    @triton.jit
    def _fpope_attention_bwd_kernel_dv(
        # Inputs (precomputed softplus)
        MuQ_ptr, MuK_ptr, dO_ptr, LSE_ptr,
        Freqs_ptr, PhaseBias_ptr,
        # Output
        dV_ptr,
        # Strides
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_dob, stride_doh, stride_dos, stride_dod,
        stride_lse_bh, stride_lse_m,
        # Dimensions
        seq_len_q: tl.constexpr,
        seq_len_k: tl.constexpr,
        head_dim: tl.constexpr,
        start_pos,
        scale,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_D_TILE: tl.constexpr,
    ):
        """Backward kernel computing dV with 3D tensor approach.

        Grid: (cdiv(seq_len_k, BLOCK_N), batch * heads)

        dV[n,d] = sum_m(p[m,n] * dO[m,d]) = p.T @ dO

        NOTE: Expects precomputed softplus values (mu_q, mu_k).
        Uses 3D tensors with small BLOCK_D_TILE for score recomputation.
        """
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        # Initialize dV accumulator
        dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        pos_k = (start_pos + offs_n).to(tl.float32)

        # Load mu_k for this K block (full head_dim)
        k_mask = (offs_n[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
        muk_ptrs = MuK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :]
        mu_k_full = tl.load(muk_ptrs, mask=k_mask, other=0.0)

        # Iterate over Q blocks
        for start_m in range(0, seq_len_q, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            pos_q = (start_pos + offs_m).to(tl.float32)
            pos_diff = pos_k[None, :] - pos_q[:, None]

            # Load mu_q for this Q block (full head_dim)
            q_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
            muq_ptrs = MuQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
            mu_q_block = tl.load(muq_ptrs, mask=q_mask, other=0.0)

            # Load LSE
            lse_ptrs = LSE_ptr + pid_bh * stride_lse_bh + offs_m * stride_lse_m
            lse = tl.load(lse_ptrs, mask=offs_m < seq_len_q, other=0.0)

            # Recompute scores using 3D tensor approach with small tiles
            scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for d_start in range(0, head_dim, BLOCK_D_TILE):
                d_offs = d_start + tl.arange(0, BLOCK_D_TILE)

                # Load freq and bias for this tile
                freq_d = tl.load(Freqs_ptr + d_offs, mask=d_offs < head_dim, other=0.0)
                bias_d = tl.load(PhaseBias_ptr + d_offs, mask=d_offs < head_dim, other=0.0)

                # Load mu_q tile for this Q block
                muq_tile_ptrs = MuQ_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qs + d_offs[None, :]
                muq_tile_mask = (offs_m[:, None] < seq_len_q) & (d_offs[None, :] < head_dim)
                mu_q_d = tl.load(muq_tile_ptrs, mask=muq_tile_mask, other=0.0)

                # Load mu_k tile for this K block
                muk_tile_ptrs = MuK_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_ks + d_offs[None, :]
                muk_tile_mask = (offs_n[:, None] < seq_len_k) & (d_offs[None, :] < head_dim)
                mu_k_d = tl.load(muk_tile_ptrs, mask=muk_tile_mask, other=0.0)

                # Compute phase: (M, N, BLOCK_D_TILE)
                phase = pos_diff[:, :, None] * freq_d[None, None, :] + bias_d[None, None, :]
                cos_phase = tl.cos(phase)

                # Compute contribution: mu_q * mu_k * cos(phase) summed over dimension tile
                contrib = mu_q_d[:, None, :] * mu_k_d[None, :, :] * cos_phase
                scores += tl.sum(contrib, axis=2)

            scores = scores * scale

            # Apply causal mask
            causal_mask = offs_m[:, None] < offs_n[None, :]
            scores = tl.where(causal_mask, float("-inf"), scores)

            # Compute p from LSE
            p = tl.exp(scores - lse[:, None])
            p = tl.where(causal_mask, 0.0, p)

            # Load dO
            do_ptrs = dO_ptr + pid_bh * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :]
            do_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
            do = tl.load(do_ptrs, mask=do_mask, other=0.0).to(tl.float32)

            # dV += p.T @ dO
            dv_acc += tl.dot(tl.trans(p.to(tl.float32)), do)

        # Store dV
        dv_ptrs = dV_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :]
        dv_mask = (offs_n[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
        tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=dv_mask)


def triton_fpope_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    start_pos: int = 0,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-accelerated FPoPE attention backward pass (NO ATOMICS for dQ/dK).

    Args:
        q: Query tensor (batch, heads, seq_q, dim)
        k: Key tensor (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        freqs: Effective frequencies (dim,)
        phase_bias: Phase bias (dim,)
        output: Forward output (batch, heads, seq_q, dim)
        grad_output: Gradient of loss w.r.t. output (batch, heads, seq_q, dim)
        lse: Logsumexp from forward (batch*heads, seq_q)
        start_pos: Starting position for KV-cache
        scale: Attention scale factor

    Returns:
        Tuple of (dq, dk, dv, dfreqs, dphase_bias)
    """
    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Check if we can use Triton
    is_power_of_2 = (dim & (dim - 1)) == 0 and dim > 0
    if not HAS_TRITON or not q.is_cuda or not is_power_of_2 or dim > 128 or lse is None:
        # Fall back to PyTorch backward
        return None  # Signal to use PyTorch

    # Precompute softplus and sigmoid in PyTorch (eliminates recomputation in kernels)
    mu_q = F.softplus(q)
    mu_k = F.softplus(k)
    sigmoid_q = torch.sigmoid(q)
    sigmoid_k = torch.sigmoid(k)

    # Reshape for kernel: (batch, heads, seq, dim) -> (batch*heads, seq, dim)
    mu_q_flat = mu_q.reshape(batch * heads, seq_q, dim)
    mu_k_flat = mu_k.reshape(batch * heads, seq_k, dim)
    sigmoid_q_flat = sigmoid_q.reshape(batch * heads, seq_q, dim)
    sigmoid_k_flat = sigmoid_k.reshape(batch * heads, seq_k, dim)
    v_flat = v.reshape(batch * heads, seq_k, dim)
    grad_output_flat = grad_output.reshape(batch * heads, seq_q, dim)

    # IMPORTANT: Use smaller blocks for backward to reduce 3D tensor size
    # 64x64x64 tensors take ~8 seconds in Triton, 32x32x64 takes ~1 second
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = dim  # Must equal head_dim (power of 2)
    BLOCK_D_TILE = 16  # Small tile for dimension loop in score computation

    num_blocks_q = triton.cdiv(seq_q, BLOCK_M)
    num_blocks_k = triton.cdiv(seq_k, BLOCK_N)
    total_blocks_dk = batch * heads * num_blocks_k

    # Allocate output tensors (no zeros needed - kernels write full blocks directly)
    dq = torch.empty(batch, heads, seq_q, dim, device=q.device, dtype=q.dtype)
    dk = torch.empty(batch, heads, seq_k, dim, device=k.device, dtype=k.dtype)
    dv = torch.empty(batch, heads, seq_k, dim, device=v.device, dtype=v.dtype)
    dq_flat = dq.reshape(batch * heads, seq_q, dim)
    dk_flat = dk.reshape(batch * heads, seq_k, dim)
    dv_flat = dv.reshape(batch * heads, seq_k, dim)

    # Allocate partial buffers for dFreq and dBias (each block writes once, no atomics)
    dfreq_partial = torch.empty(total_blocks_dk, dim, dtype=torch.float32, device=q.device)
    dbias_partial = torch.empty(total_blocks_dk, dim, dtype=torch.float32, device=q.device)

    # Launch dK kernel (also computes partial dFreq/dBias)
    grid_dk = (num_blocks_k, batch * heads)

    _fpope_attention_bwd_kernel_dk[grid_dk](
        mu_q_flat, mu_k_flat, sigmoid_k_flat, v_flat,
        freqs, phase_bias,
        grad_output_flat, lse,
        dk_flat,
        dfreq_partial, dbias_partial,
        mu_q_flat.stride(0), mu_q_flat.stride(0), mu_q_flat.stride(1), mu_q_flat.stride(2),
        mu_k_flat.stride(0), mu_k_flat.stride(0), mu_k_flat.stride(1), mu_k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        dk_flat.stride(0), dk_flat.stride(0), dk_flat.stride(1), dk_flat.stride(2),
        lse.stride(0), lse.stride(1),
        dfreq_partial.stride(0), dfreq_partial.stride(1),
        seq_q, seq_k, dim,
        start_pos,
        scale,
        num_blocks_q,
        BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE,
    )

    # Launch dQ kernel (separate kernel - no atomics)
    grid_dq = (num_blocks_q, batch * heads)

    _fpope_attention_bwd_kernel_dq[grid_dq](
        mu_q_flat, mu_k_flat, sigmoid_q_flat, v_flat,
        freqs, phase_bias,
        grad_output_flat, lse,
        dq_flat,
        mu_q_flat.stride(0), mu_q_flat.stride(0), mu_q_flat.stride(1), mu_q_flat.stride(2),
        mu_k_flat.stride(0), mu_k_flat.stride(0), mu_k_flat.stride(1), mu_k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        dq_flat.stride(0), dq_flat.stride(0), dq_flat.stride(1), dq_flat.stride(2),
        lse.stride(0), lse.stride(1),
        seq_q, seq_k, dim,
        start_pos,
        scale,
        BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE,
    )

    # Launch dV kernel
    grid_dv = (num_blocks_k, batch * heads)

    _fpope_attention_bwd_kernel_dv[grid_dv](
        mu_q_flat, mu_k_flat, grad_output_flat, lse,
        freqs, phase_bias,
        dv_flat,
        mu_q_flat.stride(0), mu_q_flat.stride(0), mu_q_flat.stride(1), mu_q_flat.stride(2),
        mu_k_flat.stride(0), mu_k_flat.stride(0), mu_k_flat.stride(1), mu_k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        grad_output_flat.stride(0), grad_output_flat.stride(0), grad_output_flat.stride(1), grad_output_flat.stride(2),
        lse.stride(0), lse.stride(1),
        seq_q, seq_k, dim,
        start_pos,
        scale,
        BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE,
    )

    # Reduce partial gradients: sum across all blocks
    dfreq = dfreq_partial.sum(dim=0)
    dbias = dbias_partial.sum(dim=0)

    return dq, dk, dv, dfreq, dbias


# =============================================================================
# Native CUDA Kernel Wrappers
# =============================================================================

def cuda_fpope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    start_pos: int = 0,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Native CUDA FPoPE attention forward pass.

    Args:
        q: Query tensor (batch, heads, seq_q, dim)
        k: Key tensor (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        freqs: Effective frequencies (dim,)
        phase_bias: Phase bias (dim,)
        start_pos: Starting position for KV-cache
        scale: Attention scale factor
        return_lse: Whether to return logsumexp for backward

    Returns:
        Tuple of (output, lse) where lse is None if not requested
    """
    if not HAS_CUDA_KERNEL:
        raise RuntimeError("CUDA kernel not available. Build with: uv run python setup.py build_ext --inplace")

    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Check dimension support
    if dim not in (32, 64, 128):
        # Fall back to Triton or PyTorch for unsupported dimensions
        if HAS_TRITON:
            return triton_fpope_attention(q, k, v, freqs, phase_bias, start_pos, scale, return_lse)
        else:
            out = pytorch_fpope_attention(q, k, v, freqs, phase_bias, start_pos, scale, causal=True)
            return (out, None) if return_lse else out

    # Precompute softplus in PyTorch
    mu_q = F.softplus(q)
    mu_k = F.softplus(k)

    # Reshape for kernel: (batch, heads, seq, dim) -> (batch*heads, seq, dim)
    mu_q_flat = mu_q.reshape(batch * heads, seq_q, dim).contiguous()
    mu_k_flat = mu_k.reshape(batch * heads, seq_k, dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_k, dim).contiguous()

    # Call CUDA kernel
    results = fpope_cuda.forward(
        mu_q_flat, mu_k_flat, v_flat,
        freqs.contiguous(), phase_bias.contiguous(),
        start_pos, scale, return_lse
    )

    output = results[0].reshape(batch, heads, seq_q, dim)

    if return_lse and len(results) > 1:
        lse = results[1]  # (batch*heads, seq_q)
        return output, lse
    return output, None


def cuda_fpope_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs: torch.Tensor,
    phase_bias: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    start_pos: int = 0,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native CUDA FPoPE attention backward pass.

    Returns:
        Tuple of (dq, dk, dv, dfreqs, dphase_bias)
    """
    if not HAS_CUDA_KERNEL:
        return None

    batch, heads, seq_q, dim = q.shape
    seq_k = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Check dimension support
    if dim not in (32, 64, 128):
        return None  # Signal to fall back

    # Precompute softplus and sigmoid
    mu_q = F.softplus(q)
    mu_k = F.softplus(k)
    sigmoid_q = torch.sigmoid(q)
    sigmoid_k = torch.sigmoid(k)

    # Reshape for kernel
    mu_q_flat = mu_q.reshape(batch * heads, seq_q, dim).contiguous()
    mu_k_flat = mu_k.reshape(batch * heads, seq_k, dim).contiguous()
    v_flat = v.reshape(batch * heads, seq_k, dim).contiguous()
    output_flat = output.reshape(batch * heads, seq_q, dim).contiguous()
    grad_output_flat = grad_output.reshape(batch * heads, seq_q, dim).contiguous()
    sigmoid_q_flat = sigmoid_q.reshape(batch * heads, seq_q, dim).contiguous()
    sigmoid_k_flat = sigmoid_k.reshape(batch * heads, seq_k, dim).contiguous()

    # Call CUDA kernel
    results = fpope_cuda.backward(
        mu_q_flat, mu_k_flat, v_flat,
        freqs.contiguous(), phase_bias.contiguous(),
        output_flat, grad_output_flat, lse,
        sigmoid_q_flat, sigmoid_k_flat,
        start_pos, scale
    )

    dq = results[0].reshape(batch, heads, seq_q, dim)
    dk = results[1].reshape(batch, heads, seq_k, dim)
    dv = results[2].reshape(batch, heads, seq_k, dim)
    dfreq = results[3]
    dbias = results[4]

    return dq, dk, dv, dfreq, dbias


class FPoPEAttentionFunction(Function):
    """Autograd function for FPoPE attention with custom backward.

    Dispatch priority:
    1. Native CUDA kernel (fastest)
    2. Triton kernel (slower due to 3D tensor JIT compilation)
    3. PyTorch reference (slowest but always works)
    """

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
        lse = None
        use_cuda = False

        # Priority 1: Native CUDA kernel
        if HAS_CUDA_KERNEL and q.is_cuda and q.shape[-1] in (32, 64, 128):
            output, lse = cuda_fpope_attention(
                q, k, v, freqs, phase_bias, start_pos, scale, return_lse=True
            )
            use_cuda = True
        # Priority 2: Triton kernel
        elif use_triton and HAS_TRITON and q.is_cuda:
            result = triton_fpope_attention(
                q, k, v, freqs, phase_bias, start_pos, scale, return_lse=True
            )
            if isinstance(result, tuple):
                output, lse = result
            else:
                output = result
        # Priority 3: PyTorch reference
        else:
            output = pytorch_fpope_attention(
                q, k, v, freqs, phase_bias, start_pos, scale, causal
            )

        # Save for backward
        ctx.save_for_backward(q, k, v, freqs, phase_bias, output)
        ctx.start_pos = start_pos
        ctx.scale = scale
        ctx.causal = causal
        ctx.use_triton = use_triton
        ctx.use_cuda = use_cuda
        ctx.lse = lse  # Save LSE for backward (may be None)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass - uses CUDA > Triton > PyTorch fallback."""
        q, k, v, freqs, phase_bias, output = ctx.saved_tensors

        # Priority 1: Native CUDA backward
        if ctx.use_cuda and HAS_CUDA_KERNEL and ctx.lse is not None:
            result = cuda_fpope_attention_backward(
                q, k, v, freqs, phase_bias,
                output, grad_output, ctx.lse,
                ctx.start_pos, ctx.scale
            )
            if result is not None:
                dq, dk, dv, dfreqs, dphase_bias = result
                return dq, dk, dv, dfreqs, dphase_bias, None, None, None, None

        # Priority 2: Triton backward
        if ctx.use_triton and HAS_TRITON and ctx.lse is not None and q.is_cuda:
            result = triton_fpope_attention_backward(
                q, k, v, freqs, phase_bias,
                output, grad_output, ctx.lse,
                ctx.start_pos, ctx.scale
            )
            if result is not None:
                dq, dk, dv, dfreqs, dphase_bias = result
                return dq, dk, dv, dfreqs, dphase_bias, None, None, None, None

        # Priority 3: PyTorch autograd fallback
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
