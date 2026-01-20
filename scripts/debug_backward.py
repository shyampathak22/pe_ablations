#!/usr/bin/env python3
"""Debug script for FPoPE backward kernels."""

import torch
import torch.nn.functional as F
import time
import sys

# Add project root to path
sys.path.insert(0, ".")

from src.model.kernels.fpope_attention import (
    HAS_TRITON,
    triton_fpope_attention,
    triton_fpope_attention_backward,
    pytorch_fpope_attention,
)

if HAS_TRITON:
    import triton

def debug_backward():
    """Test backward kernels with verbose output."""
    print("=" * 60)
    print("FPoPE Backward Kernel Debug")
    print("=" * 60)

    if not HAS_TRITON:
        print("ERROR: Triton not available")
        return

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = "cuda"

    # Small test case first
    batch, heads, seq, dim = 1, 1, 32, 64
    print(f"\nTest config: batch={batch}, heads={heads}, seq={seq}, dim={dim}")

    # Create inputs
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
    freqs = torch.randn(dim, device=device, dtype=torch.float32)
    phase_bias = torch.randn(dim, device=device, dtype=torch.float32)

    print("\n--- Forward Pass ---")
    start = time.time()
    result = triton_fpope_attention(q, k, v, freqs, phase_bias, start_pos=0, return_lse=True)
    torch.cuda.synchronize()
    fwd_time = time.time() - start

    if isinstance(result, tuple):
        output, lse = result
        print(f"Forward completed in {fwd_time*1000:.2f}ms")
        print(f"Output shape: {output.shape}")
        print(f"LSE shape: {lse.shape}")
    else:
        print("ERROR: Forward did not return LSE (fell back to PyTorch?)")
        return

    # Grad output
    grad_output = torch.randn_like(output)

    print("\n--- Backward Pass (Triton) ---")
    print("Starting backward kernels...")

    # Time each kernel separately by running them one at a time
    import math
    scale = 1.0 / math.sqrt(dim)

    # Precompute softplus and sigmoid
    mu_q = F.softplus(q)
    mu_k = F.softplus(k)
    sigmoid_q = torch.sigmoid(q)
    sigmoid_k = torch.sigmoid(k)

    # Reshape
    mu_q_flat = mu_q.view(batch * heads, seq, dim)
    mu_k_flat = mu_k.view(batch * heads, seq, dim)
    sigmoid_q_flat = sigmoid_q.view(batch * heads, seq, dim)
    sigmoid_k_flat = sigmoid_k.view(batch * heads, seq, dim)
    v_flat = v.view(batch * heads, seq, dim)
    grad_output_flat = grad_output.view(batch * heads, seq, dim)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = dim
    BLOCK_D_TILE = 16

    num_blocks_q = triton.cdiv(seq, BLOCK_M)
    num_blocks_k = triton.cdiv(seq, BLOCK_N)
    total_blocks_dk = batch * heads * num_blocks_k

    # Allocate outputs
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dq_flat = dq.view(batch * heads, seq, dim)
    dk_flat = dk.view(batch * heads, seq, dim)
    dv_flat = dv.view(batch * heads, seq, dim)

    dfreq_partial = torch.empty(total_blocks_dk, dim, dtype=torch.float32, device=device)
    dbias_partial = torch.empty(total_blocks_dk, dim, dtype=torch.float32, device=device)

    from src.model.kernels.fpope_attention import (
        _fpope_attention_bwd_kernel_dk,
        _fpope_attention_bwd_kernel_dq,
        _fpope_attention_bwd_kernel_dv,
    )

    # Test dK kernel
    print("\n  Testing dK kernel...")
    grid_dk = (num_blocks_k, batch * heads)
    print(f"    Grid: {grid_dk}")
    print(f"    BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_D={BLOCK_D}, BLOCK_D_TILE={BLOCK_D_TILE}")

    start = time.time()
    try:
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
            seq, seq, dim,
            0,  # start_pos
            scale,
            num_blocks_q,
            BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE,
        )
        torch.cuda.synchronize()
        dk_time = time.time() - start
        print(f"    dK kernel completed in {dk_time*1000:.2f}ms")
        print(f"    dK stats: min={dk_flat.min().item():.4f}, max={dk_flat.max().item():.4f}, mean={dk_flat.mean().item():.4f}")
    except Exception as e:
        print(f"    ERROR in dK kernel: {e}")
        return

    # Test dQ kernel
    print("\n  Testing dQ kernel...")
    grid_dq = (num_blocks_q, batch * heads)
    print(f"    Grid: {grid_dq}")

    start = time.time()
    try:
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
            seq, seq, dim,
            0,  # start_pos
            scale,
            BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE,
        )
        torch.cuda.synchronize()
        dq_time = time.time() - start
        print(f"    dQ kernel completed in {dq_time*1000:.2f}ms")
        print(f"    dQ stats: min={dq_flat.min().item():.4f}, max={dq_flat.max().item():.4f}, mean={dq_flat.mean().item():.4f}")
    except Exception as e:
        print(f"    ERROR in dQ kernel: {e}")
        return

    # Test dV kernel
    print("\n  Testing dV kernel...")
    grid_dv = (num_blocks_k, batch * heads)
    print(f"    Grid: {grid_dv}")

    start = time.time()
    try:
        _fpope_attention_bwd_kernel_dv[grid_dv](
            mu_q_flat, mu_k_flat, grad_output_flat, lse,
            freqs, phase_bias,
            dv_flat,
            mu_q_flat.stride(0), mu_q_flat.stride(0), mu_q_flat.stride(1), mu_q_flat.stride(2),
            mu_k_flat.stride(0), mu_k_flat.stride(0), mu_k_flat.stride(1), mu_k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            grad_output_flat.stride(0), grad_output_flat.stride(0), grad_output_flat.stride(1), grad_output_flat.stride(2),
            lse.stride(0), lse.stride(1),
            seq, seq, dim,
            0,  # start_pos
            scale,
            BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE,
        )
        torch.cuda.synchronize()
        dv_time = time.time() - start
        print(f"    dV kernel completed in {dv_time*1000:.2f}ms")
        print(f"    dV stats: min={dv_flat.min().item():.4f}, max={dv_flat.max().item():.4f}, mean={dv_flat.mean().item():.4f}")
    except Exception as e:
        print(f"    ERROR in dV kernel: {e}")
        return

    print("\n--- Backward Pass (PyTorch reference) ---")
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    freqs_ref = freqs.detach().clone().requires_grad_(True)
    bias_ref = phase_bias.detach().clone().requires_grad_(True)

    start = time.time()
    out_ref = pytorch_fpope_attention(q_ref, k_ref, v_ref, freqs_ref, bias_ref)
    out_ref.backward(grad_output)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    print(f"PyTorch backward completed in {pytorch_time*1000:.2f}ms")

    print("\n--- Gradient Comparison ---")
    print(f"dQ max diff: {(dq - q_ref.grad).abs().max().item():.6f}")
    print(f"dK max diff: {(dk - k_ref.grad).abs().max().item():.6f}")
    print(f"dV max diff: {(dv - v_ref.grad).abs().max().item():.6f}")

    dfreq = dfreq_partial.sum(dim=0)
    dbias = dbias_partial.sum(dim=0)
    print(f"dFreq max diff: {(dfreq - freqs_ref.grad).abs().max().item():.6f}")
    print(f"dBias max diff: {(dbias - bias_ref.grad).abs().max().item():.6f}")

    print("\n" + "=" * 60)
    print("Debug complete!")


if __name__ == "__main__":
    debug_backward()
