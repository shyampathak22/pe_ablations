#!/usr/bin/env python3
"""Benchmark FPoPE (Triton) vs RoPE performance."""

import torch
import torch.nn.functional as F
import time
import sys
import gc

sys.path.insert(0, ".")

from src.model.kernels.fpope_attention import (
    fpope_attention_forward,
    triton_fpope_attention,
    triton_fpope_attention_backward,
    HAS_TRITON,
    HAS_CUDA_KERNEL,
)


def rope_attention(q, k, v, freqs_cis):
    """Standard RoPE attention for comparison."""
    def apply_rotary(x, freqs):
        x_r = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_r)
        x_rot = x_complex * freqs
        return torch.view_as_real(x_rot).flatten(-2).type_as(x)

    q_rot = apply_rotary(q, freqs_cis)
    k_rot = apply_rotary(k, freqs_cis)

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale

    seq_len = q.shape[2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def precompute_freqs_cis(dim, seq_len, device):
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def benchmark_config(batch, heads, seq, dim, n_warmup=3, n_iters=10):
    """Benchmark a single configuration."""
    device = "cuda"

    # Allocate tensors
    q = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
    k = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
    v = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
    freqs = torch.randn(dim, device=device, requires_grad=True)
    bias = torch.randn(dim, device=device, requires_grad=True)
    freqs_cis = precompute_freqs_cis(dim, seq, device)
    grad_out = torch.randn(batch, heads, seq, dim, device=device)

    results = {}

    # ========== FPoPE Triton Forward ==========
    try:
        # Warmup (includes compilation if not cached)
        for _ in range(n_warmup):
            out = fpope_attention_forward(q, k, v, freqs, bias, use_triton=True)
        torch.cuda.synchronize()

        # Timed runs
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            out = fpope_attention_forward(q, k, v, freqs, bias, use_triton=True)
        torch.cuda.synchronize()
        results['fpope_triton_fwd'] = (time.time() - start) / n_iters * 1000
    except Exception as e:
        results['fpope_triton_fwd'] = f"ERROR: {e}"

    # ========== FPoPE Triton Forward+Backward ==========
    try:
        # Warmup
        for _ in range(n_warmup):
            q1 = q.detach().clone().requires_grad_(True)
            k1 = k.detach().clone().requires_grad_(True)
            v1 = v.detach().clone().requires_grad_(True)
            f1 = freqs.detach().clone().requires_grad_(True)
            b1 = bias.detach().clone().requires_grad_(True)
            out = fpope_attention_forward(q1, k1, v1, f1, b1, use_triton=True)
            out.backward(grad_out)
        torch.cuda.synchronize()

        # Timed runs
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            q1 = q.detach().clone().requires_grad_(True)
            k1 = k.detach().clone().requires_grad_(True)
            v1 = v.detach().clone().requires_grad_(True)
            f1 = freqs.detach().clone().requires_grad_(True)
            b1 = bias.detach().clone().requires_grad_(True)
            out = fpope_attention_forward(q1, k1, v1, f1, b1, use_triton=True)
            out.backward(grad_out)
        torch.cuda.synchronize()
        results['fpope_triton_fwd_bwd'] = (time.time() - start) / n_iters * 1000
    except Exception as e:
        results['fpope_triton_fwd_bwd'] = f"ERROR: {e}"

    # ========== FPoPE CUDA Forward+Backward ==========
    if HAS_CUDA_KERNEL:
        try:
            # Warmup
            for _ in range(n_warmup):
                q1 = q.detach().clone().requires_grad_(True)
                k1 = k.detach().clone().requires_grad_(True)
                v1 = v.detach().clone().requires_grad_(True)
                f1 = freqs.detach().clone().requires_grad_(True)
                b1 = bias.detach().clone().requires_grad_(True)
                out = fpope_attention_forward(q1, k1, v1, f1, b1, use_triton=False)
                out.backward(grad_out)
            torch.cuda.synchronize()

            # Timed runs
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(n_iters):
                q1 = q.detach().clone().requires_grad_(True)
                k1 = k.detach().clone().requires_grad_(True)
                v1 = v.detach().clone().requires_grad_(True)
                f1 = freqs.detach().clone().requires_grad_(True)
                b1 = bias.detach().clone().requires_grad_(True)
                out = fpope_attention_forward(q1, k1, v1, f1, b1, use_triton=False)
                out.backward(grad_out)
            torch.cuda.synchronize()
            results['fpope_cuda_fwd_bwd'] = (time.time() - start) / n_iters * 1000
        except Exception as e:
            results['fpope_cuda_fwd_bwd'] = f"ERROR: {e}"
    else:
        results['fpope_cuda_fwd_bwd'] = "N/A"

    # ========== RoPE Forward+Backward ==========
    try:
        # Warmup
        for _ in range(n_warmup):
            q2 = q.detach().clone().requires_grad_(True)
            k2 = k.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            out = rope_attention(q2, k2, v2, freqs_cis)
            out.backward(grad_out)
        torch.cuda.synchronize()

        # Timed runs
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            q2 = q.detach().clone().requires_grad_(True)
            k2 = k.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            out = rope_attention(q2, k2, v2, freqs_cis)
            out.backward(grad_out)
        torch.cuda.synchronize()
        results['rope_fwd_bwd'] = (time.time() - start) / n_iters * 1000
    except Exception as e:
        results['rope_fwd_bwd'] = f"ERROR: {e}"

    # Cleanup
    del q, k, v, freqs, bias, freqs_cis, grad_out
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    print("=" * 90)
    print("FPoPE (CUDA/Triton) vs RoPE Benchmark")
    print("=" * 90)
    print(f"Triton available: {HAS_TRITON}")
    print(f"CUDA kernel available: {HAS_CUDA_KERNEL}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Test configurations (smaller to avoid OOM)
    configs = [
        (1, 8, 64, 64),
        (2, 8, 64, 64),
        (2, 8, 128, 64),
        (4, 8, 128, 64),
        (2, 8, 256, 64),
        (4, 8, 256, 64),
    ]

    print(f"{'Config':<22} {'CUDA Fwd+Bwd':<14} {'Triton Fwd+Bwd':<16} {'RoPE Fwd+Bwd':<14} {'CUDA/RoPE':<10} {'Triton/RoPE':<12}")
    print("-" * 90)

    for batch, heads, seq, dim in configs:
        config_str = f"B={batch}, H={heads}, S={seq}, D={dim}"

        try:
            results = benchmark_config(batch, heads, seq, dim)

            cuda_fwd_bwd = results.get('fpope_cuda_fwd_bwd', 'N/A')
            triton_fwd_bwd = results.get('fpope_triton_fwd_bwd', 'N/A')
            rope_fwd_bwd = results.get('rope_fwd_bwd', 'N/A')

            if isinstance(cuda_fwd_bwd, float) and isinstance(rope_fwd_bwd, float):
                cuda_ratio = f"{cuda_fwd_bwd / rope_fwd_bwd:.1f}x"
            else:
                cuda_ratio = "N/A"

            if isinstance(triton_fwd_bwd, float) and isinstance(rope_fwd_bwd, float):
                triton_ratio = f"{triton_fwd_bwd / rope_fwd_bwd:.1f}x"
            else:
                triton_ratio = "N/A"

            cuda_str = f"{cuda_fwd_bwd:.2f}ms" if isinstance(cuda_fwd_bwd, float) else str(cuda_fwd_bwd)[:12]
            triton_str = f"{triton_fwd_bwd:.2f}ms" if isinstance(triton_fwd_bwd, float) else str(triton_fwd_bwd)[:14]
            rope_str = f"{rope_fwd_bwd:.2f}ms" if isinstance(rope_fwd_bwd, float) else str(rope_fwd_bwd)[:12]

            print(f"{config_str:<22} {cuda_str:<14} {triton_str:<16} {rope_str:<14} {cuda_ratio:<10} {triton_ratio:<12}")

        except torch.cuda.OutOfMemoryError:
            print(f"{config_str:<22} OOM")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"{config_str:<22} ERROR: {e}")

    print("=" * 90)


if __name__ == "__main__":
    main()
