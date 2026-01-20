#!/usr/bin/env python3
"""Diagnose what's happening with the backward kernels."""

import torch
import torch.nn.functional as F
import time
import subprocess
import threading
import sys

sys.path.insert(0, ".")

def gpu_monitor(stop_event, interval=0.5):
    """Monitor GPU utilization in background."""
    while not stop_event.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            print(f"  [GPU] util={parts[0]}%, mem={parts[1]}/{parts[2]} MB")
        time.sleep(interval)

def test_minimal_triton():
    """Test if basic Triton works at all."""
    print("\n=== Test 1: Basic Triton functionality ===")

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def simple_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        out = torch.empty_like(x)

        grid = (triton.cdiv(1024, 256),)
        simple_add_kernel[grid](x, y, out, 1024, 256)
        torch.cuda.synchronize()

        expected = x + y
        diff = (out - expected).abs().max().item()
        print(f"  Simple add kernel: PASS (diff={diff:.6f})")
        return True
    except Exception as e:
        print(f"  Simple add kernel: FAIL ({e})")
        return False

def test_3d_tensor_kernel():
    """Test if 3D tensor operations work in Triton."""
    print("\n=== Test 2: 3D tensor operations ===")

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def test_3d_kernel(
            out_ptr,
            M: tl.constexpr, N: tl.constexpr, D: tl.constexpr
        ):
            # Create 3D tensor and reduce
            offs_m = tl.arange(0, M)
            offs_n = tl.arange(0, N)
            offs_d = tl.arange(0, D)

            # 3D tensor: (M, N, D)
            tensor_3d = offs_m[:, None, None] + offs_n[None, :, None] + offs_d[None, None, :]
            tensor_3d = tensor_3d.to(tl.float32)

            # Reduce over axis 2
            result_2d = tl.sum(tensor_3d, axis=2)  # (M, N)

            # Reduce over axis 1
            result_1d = tl.sum(result_2d, axis=1)  # (M,)

            # Store
            tl.store(out_ptr + offs_m, result_1d)

        M, N, D = 32, 32, 64
        out = torch.empty(M, device="cuda", dtype=torch.float32)

        start = time.time()
        test_3d_kernel[(1,)](out, M, N, D)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Verify: sum over n and d of (m + n + d)
        # = sum_n sum_d (m + n + d) = N*D*m + D*sum_n(n) + N*sum_d(d)
        # = N*D*m + D*N*(N-1)/2 + N*D*(D-1)/2
        expected = torch.arange(M, device="cuda", dtype=torch.float32) * N * D
        expected += D * N * (N-1) / 2
        expected += N * D * (D-1) / 2

        diff = (out - expected).abs().max().item()
        print(f"  3D tensor kernel ({M}x{N}x{D}): PASS in {elapsed*1000:.2f}ms (diff={diff:.6f})")
        return True
    except Exception as e:
        print(f"  3D tensor kernel: FAIL ({e})")
        import traceback
        traceback.print_exc()
        return False

def test_3d_tensor_sizes():
    """Test various 3D tensor sizes to find the breaking point."""
    print("\n=== Test 3: 3D tensor size limits ===")

    import triton
    import triton.language as tl

    @triton.jit
    def size_test_kernel(
        a_ptr, b_ptr, out_ptr,
        M: tl.constexpr, N: tl.constexpr, D: tl.constexpr
    ):
        offs_m = tl.arange(0, M)
        offs_n = tl.arange(0, N)
        offs_d = tl.arange(0, D)

        # Load 2D tensors
        a = tl.load(a_ptr + offs_m[:, None] * D + offs_d[None, :])  # (M, D)
        b = tl.load(b_ptr + offs_n[:, None] * D + offs_d[None, :])  # (N, D)

        # Create 3D: a[:, None, :] * b[None, :, :] -> (M, N, D)
        prod_3d = a[:, None, :] * b[None, :, :]

        # Reduce over D
        result = tl.sum(prod_3d, axis=2)  # (M, N)

        # Store
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], result)

    configs = [
        (16, 16, 16),
        (32, 32, 32),
        (32, 32, 64),
        (64, 64, 32),
        (64, 64, 64),
    ]

    for M, N, D in configs:
        try:
            a = torch.randn(M, D, device="cuda")
            b = torch.randn(N, D, device="cuda")
            out = torch.empty(M, N, device="cuda")

            start = time.time()
            size_test_kernel[(1,)](a, b, out, M, N, D)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Verify with PyTorch
            expected = a @ b.T
            diff = (out - expected).abs().max().item()

            elements = M * N * D
            print(f"  {M}x{N}x{D} ({elements:,} elements): {elapsed*1000:.2f}ms, diff={diff:.6f}")
        except Exception as e:
            print(f"  {M}x{N}x{D}: FAIL ({e})")

def test_fpope_forward_kernel():
    """Test if the forward kernel works."""
    print("\n=== Test 4: FPoPE forward kernel ===")

    from src.model.kernels.fpope_attention import triton_fpope_attention, HAS_TRITON

    if not HAS_TRITON:
        print("  Triton not available")
        return False

    batch, heads, seq, dim = 1, 1, 32, 64
    q = torch.randn(batch, heads, seq, dim, device="cuda")
    k = torch.randn(batch, heads, seq, dim, device="cuda")
    v = torch.randn(batch, heads, seq, dim, device="cuda")
    freqs = torch.randn(dim, device="cuda")
    bias = torch.randn(dim, device="cuda")

    start = time.time()
    result = triton_fpope_attention(q, k, v, freqs, bias, start_pos=0, return_lse=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    if isinstance(result, tuple):
        out, lse = result
        print(f"  Forward kernel: PASS in {elapsed*1000:.2f}ms")
        print(f"    Output shape: {out.shape}, LSE shape: {lse.shape}")
        return True
    else:
        print(f"  Forward kernel: Fell back to PyTorch")
        return False

def test_backward_kernel_minimal():
    """Test backward kernel with minimal config and monitoring."""
    print("\n=== Test 5: Backward kernel (minimal) ===")

    import triton
    from src.model.kernels.fpope_attention import (
        _fpope_attention_bwd_kernel_dk,
        triton_fpope_attention,
    )
    import math

    batch, heads, seq, dim = 1, 1, 32, 64
    device = "cuda"

    q = torch.randn(batch, heads, seq, dim, device=device)
    k = torch.randn(batch, heads, seq, dim, device=device)
    v = torch.randn(batch, heads, seq, dim, device=device)
    freqs = torch.randn(dim, device=device)
    bias = torch.randn(dim, device=device)

    # Forward to get LSE
    result = triton_fpope_attention(q, k, v, freqs, bias, start_pos=0, return_lse=True)
    if not isinstance(result, tuple):
        print("  Cannot test - forward fell back to PyTorch")
        return False
    output, lse = result

    # Prepare backward inputs
    grad_output = torch.randn_like(output)
    scale = 1.0 / math.sqrt(dim)

    mu_q = F.softplus(q).view(batch * heads, seq, dim)
    mu_k = F.softplus(k).view(batch * heads, seq, dim)
    sigmoid_k = torch.sigmoid(k).view(batch * heads, seq, dim)
    v_flat = v.view(batch * heads, seq, dim)
    grad_flat = grad_output.view(batch * heads, seq, dim)

    BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE = 64, 64, dim, 16
    num_blocks_q = triton.cdiv(seq, BLOCK_M)
    num_blocks_k = triton.cdiv(seq, BLOCK_N)

    dk = torch.empty(batch * heads, seq, dim, device=device)
    dfreq_partial = torch.empty(batch * heads * num_blocks_k, dim, device=device)
    dbias_partial = torch.empty(batch * heads * num_blocks_k, dim, device=device)

    grid = (num_blocks_k, batch * heads)
    print(f"  Grid: {grid}, Blocks: M={BLOCK_M}, N={BLOCK_N}, D={BLOCK_D}, D_TILE={BLOCK_D_TILE}")
    print(f"  Launching dK kernel...")

    # Start GPU monitor
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=gpu_monitor, args=(stop_event, 1.0))
    monitor_thread.start()

    try:
        start = time.time()
        _fpope_attention_bwd_kernel_dk[grid](
            mu_q, mu_k, sigmoid_k, v_flat,
            freqs, bias,
            grad_flat, lse,
            dk,
            dfreq_partial, dbias_partial,
            mu_q.stride(0), mu_q.stride(0), mu_q.stride(1), mu_q.stride(2),
            mu_k.stride(0), mu_k.stride(0), mu_k.stride(1), mu_k.stride(2),
            v_flat.stride(0), v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            dk.stride(0), dk.stride(0), dk.stride(1), dk.stride(2),
            lse.stride(0), lse.stride(1),
            dfreq_partial.stride(0), dfreq_partial.stride(1),
            seq, seq, dim,
            0,  # start_pos
            scale,
            num_blocks_q,
            BLOCK_M, BLOCK_N, BLOCK_D, BLOCK_D_TILE,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  dK kernel: PASS in {elapsed*1000:.2f}ms")
        print(f"    dk stats: min={dk.min():.4f}, max={dk.max():.4f}")
        return True
    except Exception as e:
        print(f"  dK kernel: FAIL ({e})")
        import traceback
        traceback.print_exc()
        return False
    finally:
        stop_event.set()
        monitor_thread.join()


if __name__ == "__main__":
    print("=" * 60)
    print("FPoPE Kernel Diagnostics")
    print("=" * 60)

    test_minimal_triton()
    test_3d_tensor_kernel()
    test_3d_tensor_sizes()
    test_fpope_forward_kernel()
    test_backward_kernel_minimal()

    print("\n" + "=" * 60)
    print("Diagnostics complete")
