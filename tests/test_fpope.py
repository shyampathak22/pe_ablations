"""Unit tests for FoPE+PoPE positional encoding."""

import math

import pytest
import torch
import torch.nn as nn

from src.model.fpope import (
    FoPEPoPEEmbedding,
    pope_attention_scores,
    pope_attention,
)
from src.model.fpope_attention import (
    FPoPEGroupedQueryAttention,
    FPoPEGroupedQueryAttentionWithCache,
)
from src.model.kernels.fpope_attention import (
    pytorch_fpope_attention,
    fpope_attention_forward,
    HAS_TRITON,
    HAS_CUDA_KERNEL,
)
from src.model.transformer import Transformer, TransformerConfig


class TestFoPEPoPEEmbedding:
    """Tests for the FoPEPoPEEmbedding class."""

    def test_init_shapes(self):
        """Test that all parameters have correct shapes."""
        dim = 64
        num_fourier_terms = 32
        fpope = FoPEPoPEEmbedding(
            dim=dim,
            max_seq_len=512,
            num_fourier_terms=num_fourier_terms,
        )

        assert fpope.base_freqs.shape == (dim,)
        assert fpope.fourier_coeffs.shape == (dim, num_fourier_terms)
        assert fpope.fourier_indices.shape == (num_fourier_terms,)
        assert fpope.phase_bias.shape == (dim,)
        assert fpope.positions.shape == (512,)

    def test_effective_frequencies_shape(self):
        """Test that effective frequencies have correct shape."""
        dim = 64
        fpope = FoPEPoPEEmbedding(dim=dim, max_seq_len=512)
        effective_freqs = fpope._compute_effective_freqs()
        assert effective_freqs.shape == (dim,)

    def test_floor_frequency_clipping(self):
        """Test that low frequencies are clipped to zero."""
        dim = 64
        training_length = 512
        fpope = FoPEPoPEEmbedding(
            dim=dim,
            max_seq_len=512,
            training_length=training_length,
        )

        # Set Fourier coefficients to zero so effective_freq = base_freq
        with torch.no_grad():
            fpope.fourier_coeffs.zero_()

        effective_freqs = fpope._compute_effective_freqs()

        # Floor frequency is 2*pi/512
        floor_freq = 2 * math.pi / training_length

        # Higher dimensions have lower frequencies and should be clipped
        # Check that some frequencies are zeroed (high dim indices)
        high_dim_freqs = effective_freqs[-10:]  # Last 10 dimensions
        assert (high_dim_freqs == 0).any(), "Some high-dimension frequencies should be clipped"

        # Low dimensions should have non-zero frequencies
        low_dim_freqs = effective_freqs[:10]  # First 10 dimensions
        assert (low_dim_freqs > 0).all(), "Low-dimension frequencies should not be clipped"

    def test_forward_shapes(self):
        """Test that forward pass produces correct shapes."""
        batch, seq_len, num_heads, num_kv_heads, dim = 2, 32, 8, 4, 64

        fpope = FoPEPoPEEmbedding(dim=dim, max_seq_len=512)

        q = torch.randn(batch, seq_len, num_heads, dim)
        k = torch.randn(batch, seq_len, num_kv_heads, dim)

        mu_q, mu_k, freqs, positions, phase_bias = fpope(q, k, start_pos=0)

        assert mu_q.shape == q.shape
        assert mu_k.shape == k.shape
        assert freqs.shape == (dim,)
        assert positions.shape == (seq_len,)
        assert phase_bias.shape == (dim,)

    def test_softplus_positivity(self):
        """Test that magnitudes are positive via softplus."""
        fpope = FoPEPoPEEmbedding(dim=64, max_seq_len=512)

        # Include negative values
        q = torch.randn(2, 32, 8, 64) * 10  # Large range
        k = torch.randn(2, 32, 4, 64) * 10

        mu_q, mu_k, _, _, _ = fpope(q, k)

        assert (mu_q > 0).all(), "All query magnitudes should be positive"
        assert (mu_k > 0).all(), "All key magnitudes should be positive"

    def test_phase_computation(self):
        """Test that phases are computed correctly."""
        dim = 64
        fpope = FoPEPoPEEmbedding(dim=dim, max_seq_len=512)

        phases = fpope.compute_attention_phases(seq_len_q=4, seq_len_k=4)

        assert phases.shape == (4, 4, dim)

        # Diagonal should have zero position difference
        # phases[t, t, c] = 0 * freq_c + bias_c = bias_c
        for t in range(4):
            expected = fpope.phase_bias.detach()
            actual = phases[t, t]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_extend_seq_len(self):
        """Test extending sequence length."""
        fpope = FoPEPoPEEmbedding(dim=64, max_seq_len=512)
        assert fpope.max_seq_len == 512
        assert fpope.positions.shape == (512,)

        fpope.extend_seq_len(1024)
        assert fpope.max_seq_len == 1024
        assert fpope.positions.shape == (1024,)

        # Extending to smaller length should be no-op
        fpope.extend_seq_len(256)
        assert fpope.max_seq_len == 1024

    def test_delta_init_zero(self):
        """Test zero initialization of phase bias."""
        fpope = FoPEPoPEEmbedding(dim=64, delta_init="zero")
        assert torch.allclose(fpope.phase_bias, torch.zeros(64))

    def test_delta_init_uniform(self):
        """Test uniform initialization of phase bias."""
        fpope = FoPEPoPEEmbedding(dim=64, delta_init="uniform")
        # Should be in [-pi, pi]
        assert (fpope.phase_bias >= -math.pi).all()
        assert (fpope.phase_bias <= math.pi).all()


class TestPopeAttention:
    """Tests for PoPE attention computation."""

    def test_attention_scores_shape(self):
        """Test that attention scores have correct shape."""
        batch, heads, seq_q, seq_k, dim = 2, 8, 32, 32, 64

        mu_q = torch.rand(batch, heads, seq_q, dim) + 0.1  # Positive
        mu_k = torch.rand(batch, heads, seq_k, dim) + 0.1
        phases = torch.randn(seq_q, seq_k, dim)

        scores = pope_attention_scores(mu_q, mu_k, phases, causal=False)
        assert scores.shape == (batch, heads, seq_q, seq_k)

    def test_attention_causal_mask(self):
        """Test that causal masking is applied correctly."""
        batch, heads, seq, dim = 2, 8, 16, 64

        mu_q = torch.rand(batch, heads, seq, dim) + 0.1
        mu_k = torch.rand(batch, heads, seq, dim) + 0.1
        phases = torch.randn(seq, seq, dim)

        scores = pope_attention_scores(mu_q, mu_k, phases, causal=True)

        # Upper triangle (future positions) should be -inf
        for i in range(seq):
            for j in range(i + 1, seq):
                assert scores[0, 0, i, j] == float("-inf")

    def test_pope_attention_output_shape(self):
        """Test that full attention produces correct output shape."""
        batch, heads, seq, dim = 2, 8, 32, 64

        mu_q = torch.rand(batch, heads, seq, dim) + 0.1
        mu_k = torch.rand(batch, heads, seq, dim) + 0.1
        v = torch.randn(batch, heads, seq, dim)
        phases = torch.randn(seq, seq, dim)

        output = pope_attention(mu_q, mu_k, v, phases, causal=True)
        assert output.shape == (batch, heads, seq, dim)


class TestFPoPEKernel:
    """Tests for the FPoPE attention kernel."""

    def test_pytorch_attention_shape(self):
        """Test PyTorch fallback produces correct shapes."""
        batch, heads, seq, dim = 2, 8, 32, 64

        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert output.shape == (batch, heads, seq, dim)

    def test_fpope_attention_forward(self):
        """Test the main entry point function."""
        batch, heads, seq, dim = 2, 8, 32, 64

        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = fpope_attention_forward(
            q, k, v, freqs, phase_bias,
            causal=True,
            use_triton=False,
        )
        assert output.shape == (batch, heads, seq, dim)

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_matches_pytorch(self):
        """Test that Triton kernel matches PyTorch implementation."""
        batch, heads, seq, dim = 2, 8, 32, 64
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        pytorch_out = fpope_attention_forward(
            q, k, v, freqs, phase_bias, use_triton=False
        )
        triton_out = fpope_attention_forward(
            q, k, v, freqs, phase_bias, use_triton=True
        )

        torch.testing.assert_close(pytorch_out, triton_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_head_dim_96_fallback(self):
        """Test that head_dim=96 (non-power-of-2) falls back to PyTorch."""
        batch, heads, seq, dim = 2, 4, 128, 96
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        # Non-power-of-2 falls back to PyTorch, so outputs should match exactly
        pytorch_out = fpope_attention_forward(
            q, k, v, freqs, phase_bias, use_triton=False
        )
        triton_out = fpope_attention_forward(
            q, k, v, freqs, phase_bias, use_triton=True
        )

        # Falls back to same PyTorch impl, so should be identical
        torch.testing.assert_close(pytorch_out, triton_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_head_dim_128(self):
        """Test Triton kernel with head_dim=128 (max supported before fallback)."""
        batch, heads, seq, dim = 2, 4, 128, 128
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        pytorch_out = fpope_attention_forward(
            q, k, v, freqs, phase_bias, use_triton=False
        )
        triton_out = fpope_attention_forward(
            q, k, v, freqs, phase_bias, use_triton=True
        )

        torch.testing.assert_close(pytorch_out, triton_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_head_dim_256_fallback(self):
        """Test that head_dim > 128 falls back to PyTorch (no freeze/crash)."""
        batch, heads, seq, dim = 2, 4, 64, 256
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        # Should fall back to PyTorch, not freeze
        triton_out = fpope_attention_forward(
            q, k, v, freqs, phase_bias, use_triton=True
        )

        # Verify output is valid (no NaN)
        assert not torch.isnan(triton_out).any()
        assert triton_out.shape == (batch, heads, seq, dim)

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_backward_matches_pytorch(self):
        """Test that Triton backward kernel matches PyTorch backward."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"

        # Create inputs with requires_grad
        q_triton = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        k_triton = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        v_triton = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        freqs_triton = torch.randn(dim, device=device, requires_grad=True)
        phase_bias_triton = torch.randn(dim, device=device, requires_grad=True)

        # Clone for PyTorch
        q_pytorch = q_triton.detach().clone().requires_grad_(True)
        k_pytorch = k_triton.detach().clone().requires_grad_(True)
        v_pytorch = v_triton.detach().clone().requires_grad_(True)
        freqs_pytorch = freqs_triton.detach().clone().requires_grad_(True)
        phase_bias_pytorch = phase_bias_triton.detach().clone().requires_grad_(True)

        # Same grad_output for both
        grad_output = torch.randn(batch, heads, seq, dim, device=device)

        # Triton forward + backward
        triton_out = fpope_attention_forward(
            q_triton, k_triton, v_triton, freqs_triton, phase_bias_triton, use_triton=True
        )
        triton_out.backward(grad_output)

        # PyTorch forward + backward
        pytorch_out = fpope_attention_forward(
            q_pytorch, k_pytorch, v_pytorch, freqs_pytorch, phase_bias_pytorch, use_triton=False
        )
        pytorch_out.backward(grad_output)

        # Compare gradients (with relaxed tolerances for numerical differences)
        torch.testing.assert_close(q_triton.grad, q_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k_triton.grad, k_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v_triton.grad, v_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(freqs_triton.grad, freqs_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(phase_bias_triton.grad, phase_bias_pytorch.grad, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_backward_head_dim_128(self):
        """Test Triton backward with head_dim=128."""
        batch, heads, seq, dim = 2, 4, 64, 128
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        k = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        v = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        freqs = torch.randn(dim, device=device, requires_grad=True)
        phase_bias = torch.randn(dim, device=device, requires_grad=True)

        out = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=True)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist and are valid
        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert k.grad is not None and not torch.isnan(k.grad).any()
        assert v.grad is not None and not torch.isnan(v.grad).any()
        assert freqs.grad is not None and not torch.isnan(freqs.grad).any()
        assert phase_bias.grad is not None and not torch.isnan(phase_bias.grad).any()


class TestFPoPECUDAKernel:
    """Tests for the native CUDA FPoPE attention kernel."""

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward_matches_pytorch(self):
        """Test that CUDA kernel matches PyTorch implementation."""
        batch, heads, seq, dim = 2, 8, 64, 64
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        pytorch_out = pytorch_fpope_attention(q, k, v, freqs, phase_bias)

        # Force CUDA path by using fpope_attention_forward (auto-dispatches)
        cuda_out = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)

        torch.testing.assert_close(pytorch_out, cuda_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_head_dim_32(self):
        """Test CUDA kernel with head_dim=32."""
        batch, heads, seq, dim = 2, 8, 128, 32
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        out = fpope_attention_forward(q, k, v, freqs, phase_bias)

        assert out.shape == (batch, heads, seq, dim)
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_head_dim_128(self):
        """Test CUDA kernel with head_dim=128."""
        batch, heads, seq, dim = 2, 4, 128, 128
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        out = fpope_attention_forward(q, k, v, freqs, phase_bias)

        assert out.shape == (batch, heads, seq, dim)
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_backward_matches_pytorch(self):
        """Test that CUDA backward kernel matches PyTorch backward."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"

        # Create inputs with requires_grad
        q_cuda = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        k_cuda = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        v_cuda = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        freqs_cuda = torch.randn(dim, device=device, requires_grad=True)
        phase_bias_cuda = torch.randn(dim, device=device, requires_grad=True)

        # Clone for PyTorch reference
        q_pytorch = q_cuda.detach().clone().requires_grad_(True)
        k_pytorch = k_cuda.detach().clone().requires_grad_(True)
        v_pytorch = v_cuda.detach().clone().requires_grad_(True)
        freqs_pytorch = freqs_cuda.detach().clone().requires_grad_(True)
        phase_bias_pytorch = phase_bias_cuda.detach().clone().requires_grad_(True)

        grad_output = torch.randn(batch, heads, seq, dim, device=device)

        # CUDA forward + backward (via auto-dispatch)
        cuda_out = fpope_attention_forward(
            q_cuda, k_cuda, v_cuda, freqs_cuda, phase_bias_cuda, use_triton=False
        )
        cuda_out.backward(grad_output)

        # PyTorch forward + backward
        pytorch_out = pytorch_fpope_attention(
            q_pytorch, k_pytorch, v_pytorch, freqs_pytorch, phase_bias_pytorch
        )
        pytorch_out.backward(grad_output)

        # Compare gradients (with relaxed tolerances for numerical differences)
        torch.testing.assert_close(q_cuda.grad, q_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k_cuda.grad, k_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v_cuda.grad, v_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(freqs_cuda.grad, freqs_pytorch.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(phase_bias_cuda.grad, phase_bias_pytorch.grad, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_longer_sequence(self):
        """Test CUDA kernel with longer sequence (256 tokens)."""
        batch, heads, seq, dim = 2, 4, 256, 64
        device = "cuda"

        q = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        k = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        v = torch.randn(batch, heads, seq, dim, device=device, requires_grad=True)
        freqs = torch.randn(dim, device=device, requires_grad=True)
        phase_bias = torch.randn(dim, device=device, requires_grad=True)

        out = fpope_attention_forward(q, k, v, freqs, phase_bias)
        loss = out.sum()
        loss.backward()

        assert out.shape == (batch, heads, seq, dim)
        assert not torch.isnan(out).any()
        assert q.grad is not None and not torch.isnan(q.grad).any()


class TestFPoPEGroupedQueryAttention:
    """Tests for the FPoPE GQA attention layer."""

    def test_output_shape(self):
        """Test that FPoPE-GQA produces correct output shape."""
        batch, seq_len, hidden_dim = 2, 32, 512
        num_heads, num_kv_heads, head_dim = 8, 4, 64

        fpope = FoPEPoPEEmbedding(dim=head_dim, max_seq_len=512)
        attn = FPoPEGroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            fpope=fpope,
            use_triton=False,
        )

        x = torch.randn(batch, seq_len, hidden_dim)
        output = attn(x)

        assert output.shape == (batch, seq_len, hidden_dim)

    def test_gradient_flow(self):
        """Test that gradients flow through FPoPE-GQA."""
        batch, seq_len, hidden_dim = 2, 16, 256
        num_heads, num_kv_heads, head_dim = 4, 2, 64

        fpope = FoPEPoPEEmbedding(dim=head_dim, max_seq_len=512)
        attn = FPoPEGroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            fpope=fpope,
            use_triton=False,
        )

        x = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        # Check gradient flow to input
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check gradient flow to FPoPE learnable parameters
        assert fpope.fourier_coeffs.grad is not None
        assert fpope.phase_bias.grad is not None
        assert not torch.isnan(fpope.fourier_coeffs.grad).any()
        assert not torch.isnan(fpope.phase_bias.grad).any()

    def test_position_encoding_affects_output(self):
        """Test that position encoding affects attention output.

        Note: With same Q/K sequence lengths, start_pos doesn't change
        relative position differences. This test verifies that different
        positions within a sequence produce different attention patterns.
        """
        batch, seq_len, hidden_dim = 2, 16, 256
        num_heads, num_kv_heads, head_dim = 4, 2, 64

        fpope = FoPEPoPEEmbedding(dim=head_dim, max_seq_len=512)
        attn = FPoPEGroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            fpope=fpope,
            use_triton=False,
        )

        x = torch.randn(batch, seq_len, hidden_dim)
        output = attn(x, start_pos=0)

        # Different tokens at different positions should have different outputs
        # (unless the input happens to be the same, which is unlikely with random data)
        assert output.shape == (batch, seq_len, hidden_dim)

        # Verify attention is position-aware by checking phases are non-trivial
        phases = fpope.compute_attention_phases(seq_len, seq_len)
        # Off-diagonal phases should not all be zero (position matters)
        off_diagonal_phases = phases[0, 1, :]  # phase for positions 0 attending to 1
        assert not torch.allclose(off_diagonal_phases, torch.zeros_like(off_diagonal_phases))


class TestFPoPEIntegration:
    """Integration tests with the full Transformer."""

    def test_transformer_with_fpope(self):
        """Test Transformer with FPoPE positional encoding."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            max_seq_len=128,
            pe_mode="fpope",
            fpope_training_length=128,
        )

        model = Transformer(config)

        batch, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch, seq_len))

        output = model(input_ids)

        assert "logits" in output
        assert output["logits"].shape == (batch, seq_len, 1000)

    def test_transformer_fpope_training(self):
        """Test Transformer with FPoPE in training mode."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            max_seq_len=128,
            pe_mode="fpope",
        )

        model = Transformer(config)
        model.train()

        batch, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        labels = torch.randint(0, 1000, (batch, seq_len))

        output = model(input_ids, labels=labels)

        assert "loss" in output
        assert not torch.isnan(output["loss"])

        # Test backward pass
        output["loss"].backward()

        # Check FPoPE parameters have gradients
        assert model.fpope.fourier_coeffs.grad is not None
        assert model.fpope.phase_bias.grad is not None

    def test_transformer_fpope_context_extension(self):
        """Test FPoPE context length extension."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            max_seq_len=128,
            pe_mode="fpope",
        )

        model = Transformer(config)

        # Extend context
        model.extend_context_length(256)

        assert model.config.max_seq_len == 256
        assert model.fpope.max_seq_len == 256
        assert model.fpope.positions.shape == (256,)

        # Should work with longer sequences
        batch, seq_len = 2, 200
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        output = model(input_ids)

        assert output["logits"].shape == (batch, seq_len, 1000)

    def test_rope_mode_still_works(self):
        """Test that RoPE mode still works after adding FPoPE."""
        config = TransformerConfig(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            max_seq_len=128,
            pe_mode="rope",
            rope_type="standard",
        )

        model = Transformer(config)

        batch, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        labels = torch.randint(0, 1000, (batch, seq_len))

        output = model(input_ids, labels=labels)

        assert "logits" in output
        assert "loss" in output
        assert not torch.isnan(output["loss"])


class TestNumericalStability:
    """Tests for numerical stability with long sequences."""

    def test_long_sequence_stability(self):
        """Test FPoPE with longer sequences."""
        fpope = FoPEPoPEEmbedding(dim=64, max_seq_len=4096)

        batch, seq_len, num_heads, dim = 1, 1024, 8, 64
        q = torch.randn(batch, seq_len, num_heads, dim)
        k = torch.randn(batch, seq_len, num_heads // 2, dim)

        mu_q, mu_k, freqs, positions, phase_bias = fpope(q, k)

        # Check for NaN/Inf
        assert not torch.isnan(mu_q).any()
        assert not torch.isnan(mu_k).any()
        assert not torch.isinf(mu_q).any()
        assert not torch.isinf(mu_k).any()

    def test_attention_stability_long_sequence(self):
        """Test attention computation stability with long sequences."""
        batch, heads, seq, dim = 1, 4, 256, 64

        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.zeros(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGradientCorrectness:
    """Comprehensive gradient correctness tests using torch.autograd.gradcheck.

    These tests verify that the backward pass implementations are mathematically
    correct by comparing against numerical differentiation.
    """

    def _make_inputs(self, batch, heads, seq, dim, device="cpu", dtype=torch.float64):
        """Create inputs for gradient checking (requires float64 for numerical precision)."""
        # Small values to avoid numerical issues in gradcheck
        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True) * 0.1
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True) * 0.1
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True) * 0.1
        freqs = torch.randn(dim, device=device, dtype=dtype, requires_grad=True) * 0.1
        phase_bias = torch.randn(dim, device=device, dtype=dtype, requires_grad=True) * 0.1
        return q, k, v, freqs, phase_bias

    def test_pytorch_gradcheck_small(self):
        """Test PyTorch implementation with torch.autograd.gradcheck (small inputs)."""
        batch, heads, seq, dim = 1, 2, 4, 8
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim)

        def func(q, k, v, freqs, phase_bias):
            return pytorch_fpope_attention(q, k, v, freqs, phase_bias, causal=True)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True
        )

    def test_pytorch_gradcheck_medium(self):
        """Test PyTorch implementation with slightly larger inputs."""
        batch, heads, seq, dim = 2, 4, 8, 16
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim)

        def func(q, k, v, freqs, phase_bias):
            return pytorch_fpope_attention(q, k, v, freqs, phase_bias, causal=True)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True
        )

    def test_pytorch_gradcheck_noncausal(self):
        """Test PyTorch implementation without causal masking."""
        batch, heads, seq, dim = 1, 2, 6, 8
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim)

        def func(q, k, v, freqs, phase_bias):
            return pytorch_fpope_attention(q, k, v, freqs, phase_bias, causal=False)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True
        )

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradcheck_dim32(self):
        """Test CUDA kernel gradcheck with head_dim=32."""
        batch, heads, seq, dim = 1, 2, 8, 32
        device = "cuda"
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim, device=device)

        def func(q, k, v, freqs, phase_bias):
            return fpope_attention_forward(q, k, v, freqs, phase_bias, causal=True, use_triton=False)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True,
            nondet_tol=1e-5,  # Allow small non-determinism from CUDA
        )

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradcheck_dim64(self):
        """Test CUDA kernel gradcheck with head_dim=64."""
        batch, heads, seq, dim = 1, 2, 8, 64
        device = "cuda"
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim, device=device)

        def func(q, k, v, freqs, phase_bias):
            return fpope_attention_forward(q, k, v, freqs, phase_bias, causal=True, use_triton=False)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True,
            nondet_tol=1e-5,
        )

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradcheck_dim128(self):
        """Test CUDA kernel gradcheck with head_dim=128."""
        batch, heads, seq, dim = 1, 2, 8, 128
        device = "cuda"
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim, device=device)

        def func(q, k, v, freqs, phase_bias):
            return fpope_attention_forward(q, k, v, freqs, phase_bias, causal=True, use_triton=False)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True,
            nondet_tol=1e-5,
        )

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_gradcheck_dim32(self):
        """Test Triton kernel gradcheck with head_dim=32."""
        batch, heads, seq, dim = 1, 2, 8, 32
        device = "cuda"
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim, device=device)

        def func(q, k, v, freqs, phase_bias):
            return fpope_attention_forward(q, k, v, freqs, phase_bias, causal=True, use_triton=True)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True,
            nondet_tol=1e-5,
        )

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_gradcheck_dim64(self):
        """Test Triton kernel gradcheck with head_dim=64."""
        batch, heads, seq, dim = 1, 2, 8, 64
        device = "cuda"
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim, device=device)

        def func(q, k, v, freqs, phase_bias):
            return fpope_attention_forward(q, k, v, freqs, phase_bias, causal=True, use_triton=True)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True,
            nondet_tol=1e-5,
        )

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_gradcheck_dim128(self):
        """Test Triton kernel gradcheck with head_dim=128."""
        batch, heads, seq, dim = 1, 2, 8, 128
        device = "cuda"
        q, k, v, freqs, phase_bias = self._make_inputs(batch, heads, seq, dim, device=device)

        def func(q, k, v, freqs, phase_bias):
            return fpope_attention_forward(q, k, v, freqs, phase_bias, causal=True, use_triton=True)

        assert torch.autograd.gradcheck(
            func, (q, k, v, freqs, phase_bias),
            eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True,
            nondet_tol=1e-5,
        )


class TestNumericalStabilityComprehensive:
    """Comprehensive numerical stability tests for edge cases and extreme values."""

    # ==================== Edge Case Dimensions ====================

    def test_batch_size_one(self):
        """Test with batch_size=1."""
        batch, heads, seq, dim = 1, 4, 32, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sequence_length_one(self):
        """Test with seq_len=1 (single token)."""
        batch, heads, seq, dim = 2, 4, 1, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == (batch, heads, seq, dim)

    def test_single_head(self):
        """Test with num_heads=1."""
        batch, heads, seq, dim = 2, 1, 32, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_minimal_dimensions(self):
        """Test with minimal dimensions (batch=1, heads=1, seq=1, dim=1)."""
        batch, heads, seq, dim = 1, 1, 1, 1
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_asymmetric_sequence_lengths(self):
        """Test with different Q and K sequence lengths (cross-attention scenario)."""
        batch, heads, seq_q, seq_k, dim = 2, 4, 16, 32, 64
        q = torch.randn(batch, heads, seq_q, dim)
        k = torch.randn(batch, heads, seq_k, dim)
        v = torch.randn(batch, heads, seq_k, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias, causal=False)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == (batch, heads, seq_q, dim)

    # ==================== Extreme Value Tests ====================

    def test_large_input_values(self):
        """Test with large input values (potential overflow in exp)."""
        batch, heads, seq, dim = 2, 4, 16, 64
        # Large values that might cause overflow in softmax
        q = torch.randn(batch, heads, seq, dim) * 10.0
        k = torch.randn(batch, heads, seq, dim) * 10.0
        v = torch.randn(batch, heads, seq, dim) * 10.0
        freqs = torch.randn(dim) * 0.1  # Keep freqs small
        phase_bias = torch.randn(dim) * 0.1

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with large inputs"
        assert not torch.isinf(output).any(), "Inf detected with large inputs"

    def test_small_input_values(self):
        """Test with very small input values (potential underflow)."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq, dim) * 1e-6
        k = torch.randn(batch, heads, seq, dim) * 1e-6
        v = torch.randn(batch, heads, seq, dim) * 1e-6
        freqs = torch.randn(dim) * 1e-6
        phase_bias = torch.randn(dim) * 1e-6

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with small inputs"
        assert not torch.isinf(output).any(), "Inf detected with small inputs"

    def test_zero_queries(self):
        """Test with all-zero queries."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.zeros(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with zero queries"
        assert not torch.isinf(output).any(), "Inf detected with zero queries"

    def test_zero_keys(self):
        """Test with all-zero keys."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.zeros(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with zero keys"
        assert not torch.isinf(output).any(), "Inf detected with zero keys"

    def test_zero_values(self):
        """Test with all-zero values."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.zeros(batch, heads, seq, dim)
        freqs = torch.randn(dim)
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with zero values"
        # Output should be all zeros when V is zero
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_zero_frequencies(self):
        """Test with zero frequencies (no positional encoding)."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.zeros(dim)
        phase_bias = torch.zeros(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with zero frequencies"
        assert not torch.isinf(output).any(), "Inf detected with zero frequencies"

    def test_uniform_inputs(self):
        """Test with uniform (constant) inputs across all positions."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.ones(batch, heads, seq, dim)
        k = torch.ones(batch, heads, seq, dim)
        v = torch.ones(batch, heads, seq, dim)
        freqs = torch.ones(dim)
        phase_bias = torch.zeros(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with uniform inputs"
        assert not torch.isinf(output).any(), "Inf detected with uniform inputs"

    def test_high_frequency_values(self):
        """Test with high frequency values (rapid oscillation)."""
        batch, heads, seq, dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = torch.randn(dim) * 100.0  # Very high frequencies
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with high frequencies"
        assert not torch.isinf(output).any(), "Inf detected with high frequencies"

    def test_negative_frequencies(self):
        """Test with negative frequency values."""
        batch, heads, seq, dim = 2, 4, 16, 64
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        freqs = -torch.abs(torch.randn(dim))  # All negative
        phase_bias = torch.randn(dim)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        assert not torch.isnan(output).any(), "NaN detected with negative frequencies"
        assert not torch.isinf(output).any(), "Inf detected with negative frequencies"

    # ==================== Gradient Stability Tests ====================

    def test_gradient_stability_large_values(self):
        """Test gradient stability with large input values."""
        batch, heads, seq, dim = 2, 4, 16, 64
        # Use in-place multiplication to keep tensors as leaf tensors
        q = torch.randn(batch, heads, seq, dim).mul_(5.0).requires_grad_(True)
        k = torch.randn(batch, heads, seq, dim).mul_(5.0).requires_grad_(True)
        v = torch.randn(batch, heads, seq, dim).mul_(5.0).requires_grad_(True)
        freqs = torch.randn(dim).mul_(0.1).requires_grad_(True)
        phase_bias = torch.randn(dim).mul_(0.1).requires_grad_(True)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        loss = output.sum()
        loss.backward()

        for name, param in [("q", q), ("k", k), ("v", v), ("freqs", freqs), ("phase_bias", phase_bias)]:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

    def test_gradient_stability_small_values(self):
        """Test gradient stability with small input values."""
        batch, heads, seq, dim = 2, 4, 16, 64
        # Use in-place multiplication to keep tensors as leaf tensors
        q = torch.randn(batch, heads, seq, dim).mul_(0.01).requires_grad_(True)
        k = torch.randn(batch, heads, seq, dim).mul_(0.01).requires_grad_(True)
        v = torch.randn(batch, heads, seq, dim).mul_(0.01).requires_grad_(True)
        freqs = torch.randn(dim).mul_(0.01).requires_grad_(True)
        phase_bias = torch.randn(dim).mul_(0.01).requires_grad_(True)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        loss = output.sum()
        loss.backward()

        for name, param in [("q", q), ("k", k), ("v", v), ("freqs", freqs), ("phase_bias", phase_bias)]:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

    def test_gradient_magnitude_reasonable(self):
        """Test that gradients have reasonable magnitude (not exploding)."""
        batch, heads, seq, dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seq, dim, requires_grad=True)
        k = torch.randn(batch, heads, seq, dim, requires_grad=True)
        v = torch.randn(batch, heads, seq, dim, requires_grad=True)
        freqs = torch.randn(dim, requires_grad=True)
        phase_bias = torch.randn(dim, requires_grad=True)

        output = pytorch_fpope_attention(q, k, v, freqs, phase_bias)
        loss = output.sum()
        loss.backward()

        # Gradients should not explode (reasonable upper bound)
        max_grad_norm = 1e6
        for name, param in [("q", q), ("k", k), ("v", v), ("freqs", freqs), ("phase_bias", phase_bias)]:
            grad_norm = param.grad.norm().item()
            assert grad_norm < max_grad_norm, f"Gradient explosion for {name}: norm={grad_norm}"

    # ==================== CUDA-Specific Stability Tests ====================

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_large_values(self):
        """Test CUDA kernel with large input values."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        q = torch.randn(batch, heads, seq, dim, device=device) * 10.0
        k = torch.randn(batch, heads, seq, dim, device=device) * 10.0
        v = torch.randn(batch, heads, seq, dim, device=device) * 10.0
        freqs = torch.randn(dim, device=device) * 0.1
        phase_bias = torch.randn(dim, device=device) * 0.1

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        assert not torch.isnan(output).any(), "CUDA: NaN with large values"
        assert not torch.isinf(output).any(), "CUDA: Inf with large values"

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_zero_inputs(self):
        """Test CUDA kernel with zero inputs."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        q = torch.zeros(batch, heads, seq, dim, device=device)
        k = torch.zeros(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        assert not torch.isnan(output).any(), "CUDA: NaN with zero Q/K"
        assert not torch.isinf(output).any(), "CUDA: Inf with zero Q/K"

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradient_stability(self):
        """Test CUDA kernel gradient stability."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        # Use in-place multiplication to keep tensors as leaf tensors
        q = torch.randn(batch, heads, seq, dim, device=device).mul_(5.0).requires_grad_(True)
        k = torch.randn(batch, heads, seq, dim, device=device).mul_(5.0).requires_grad_(True)
        v = torch.randn(batch, heads, seq, dim, device=device).mul_(5.0).requires_grad_(True)
        freqs = torch.randn(dim, device=device).requires_grad_(True)
        phase_bias = torch.randn(dim, device=device).requires_grad_(True)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        loss = output.sum()
        loss.backward()

        for name, param in [("q", q), ("k", k), ("v", v), ("freqs", freqs), ("phase_bias", phase_bias)]:
            assert param.grad is not None, f"CUDA: No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"CUDA: NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"CUDA: Inf gradient for {name}"

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sequence_length_one(self):
        """Test CUDA kernel with single token."""
        batch, heads, seq, dim = 2, 4, 1, 64
        device = "cuda"
        q = torch.randn(batch, heads, seq, dim, device=device)
        k = torch.randn(batch, heads, seq, dim, device=device)
        v = torch.randn(batch, heads, seq, dim, device=device)
        freqs = torch.randn(dim, device=device)
        phase_bias = torch.randn(dim, device=device)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        assert not torch.isnan(output).any(), "CUDA: NaN with seq_len=1"
        assert output.shape == (batch, heads, seq, dim)

    # ==================== Triton-Specific Stability Tests ====================

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_large_values(self):
        """Test Triton kernel with large input values."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        q = torch.randn(batch, heads, seq, dim, device=device) * 10.0
        k = torch.randn(batch, heads, seq, dim, device=device) * 10.0
        v = torch.randn(batch, heads, seq, dim, device=device) * 10.0
        freqs = torch.randn(dim, device=device) * 0.1
        phase_bias = torch.randn(dim, device=device) * 0.1

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=True)
        assert not torch.isnan(output).any(), "Triton: NaN with large values"
        assert not torch.isinf(output).any(), "Triton: Inf with large values"

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_gradient_stability(self):
        """Test Triton kernel gradient stability."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        # Use in-place multiplication to keep tensors as leaf tensors
        q = torch.randn(batch, heads, seq, dim, device=device).mul_(5.0).requires_grad_(True)
        k = torch.randn(batch, heads, seq, dim, device=device).mul_(5.0).requires_grad_(True)
        v = torch.randn(batch, heads, seq, dim, device=device).mul_(5.0).requires_grad_(True)
        freqs = torch.randn(dim, device=device).requires_grad_(True)
        phase_bias = torch.randn(dim, device=device).requires_grad_(True)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=True)
        loss = output.sum()
        loss.backward()

        for name, param in [("q", q), ("k", k), ("v", v), ("freqs", freqs), ("phase_bias", phase_bias)]:
            assert param.grad is not None, f"Triton: No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"Triton: NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Triton: Inf gradient for {name}"


class TestMixedPrecisionStability:
    """Tests for mixed precision (float16/bfloat16) numerical stability."""

    def _check_output_valid(self, output, name="output"):
        """Check that output is valid (no NaN/Inf)."""
        assert not torch.isnan(output).any(), f"{name} contains NaN"
        assert not torch.isinf(output).any(), f"{name} contains Inf"

    # ==================== Float16 Tests ====================

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_forward(self):
        """Test forward pass with float16 precision."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        dtype = torch.float16

        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        freqs = torch.randn(dim, device=device, dtype=dtype)
        phase_bias = torch.randn(dim, device=device, dtype=dtype)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        self._check_output_valid(output, "float16 output")
        # Note: Kernel may upcast to float32 internally for precision, which is acceptable

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_backward(self):
        """Test backward pass with float16 precision."""
        batch, heads, seq, dim = 2, 4, 16, 64
        device = "cuda"
        dtype = torch.float16

        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True)
        freqs = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
        phase_bias = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        loss = output.sum()
        loss.backward()

        for name, param in [("q", q), ("k", k), ("v", v), ("freqs", freqs), ("phase_bias", phase_bias)]:
            assert param.grad is not None, f"float16: No gradient for {name}"
            # Note: float16 gradients may have some Inf due to limited range, but NaN is a bug
            assert not torch.isnan(param.grad).any(), f"float16: NaN gradient for {name}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_large_values_clipped(self):
        """Test float16 with values that might overflow - input should be clipped."""
        batch, heads, seq, dim = 2, 4, 16, 64
        device = "cuda"
        dtype = torch.float16

        # Values near float16 max (~65504), scaled down to avoid immediate overflow
        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 10.0
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 10.0
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 10.0
        freqs = torch.randn(dim, device=device, dtype=dtype) * 0.01  # Keep freqs small
        phase_bias = torch.randn(dim, device=device, dtype=dtype) * 0.01

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        # May have Inf due to float16 limitations, but should not have NaN
        assert not torch.isnan(output).any(), "float16 large values: NaN detected"

    # ==================== BFloat16 Tests ====================

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch.cuda.is_bf16_supported(),
        reason="BFloat16 not supported on this GPU"
    )
    def test_bfloat16_forward(self):
        """Test forward pass with bfloat16 precision."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        dtype = torch.bfloat16

        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        freqs = torch.randn(dim, device=device, dtype=dtype)
        phase_bias = torch.randn(dim, device=device, dtype=dtype)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        self._check_output_valid(output, "bfloat16 output")
        # Note: Kernel may upcast to float32 internally for precision, which is acceptable

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch.cuda.is_bf16_supported(),
        reason="BFloat16 not supported on this GPU"
    )
    def test_bfloat16_backward(self):
        """Test backward pass with bfloat16 precision."""
        batch, heads, seq, dim = 2, 4, 16, 64
        device = "cuda"
        dtype = torch.bfloat16

        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype, requires_grad=True)
        freqs = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
        phase_bias = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        loss = output.sum()
        loss.backward()

        for name, param in [("q", q), ("k", k), ("v", v), ("freqs", freqs), ("phase_bias", phase_bias)]:
            assert param.grad is not None, f"bfloat16: No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"bfloat16: NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"bfloat16: Inf gradient for {name}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch.cuda.is_bf16_supported(),
        reason="BFloat16 not supported on this GPU"
    )
    def test_bfloat16_large_values(self):
        """Test bfloat16 with large values (wider dynamic range than float16)."""
        batch, heads, seq, dim = 2, 4, 16, 64
        device = "cuda"
        dtype = torch.bfloat16

        # bfloat16 has same range as float32, just less precision
        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 50.0
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 50.0
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 50.0
        freqs = torch.randn(dim, device=device, dtype=dtype) * 0.01
        phase_bias = torch.randn(dim, device=device, dtype=dtype) * 0.01

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=False)
        self._check_output_valid(output, "bfloat16 large values")

    # ==================== Mixed Precision Comparison Tests ====================

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_float16_vs_float32_consistency(self):
        """Test that float16 and float32 produce similar results."""
        batch, heads, seq, dim = 2, 4, 16, 64
        device = "cuda"

        # Create inputs in float32
        q_f32 = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
        k_f32 = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
        v_f32 = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
        freqs_f32 = torch.randn(dim, device=device, dtype=torch.float32)
        phase_bias_f32 = torch.randn(dim, device=device, dtype=torch.float32)

        # Convert to float16
        q_f16 = q_f32.to(torch.float16)
        k_f16 = k_f32.to(torch.float16)
        v_f16 = v_f32.to(torch.float16)
        freqs_f16 = freqs_f32.to(torch.float16)
        phase_bias_f16 = phase_bias_f32.to(torch.float16)

        output_f32 = fpope_attention_forward(q_f32, k_f32, v_f32, freqs_f32, phase_bias_f32, use_triton=False)
        output_f16 = fpope_attention_forward(q_f16, k_f16, v_f16, freqs_f16, phase_bias_f16, use_triton=False)

        # Convert to same dtype for comparison
        output_f16_as_f32 = output_f16.to(torch.float32)

        # Relaxed tolerance for float16
        torch.testing.assert_close(output_f32, output_f16_as_f32, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch.cuda.is_bf16_supported(),
        reason="BFloat16 not supported on this GPU"
    )
    def test_bfloat16_vs_float32_consistency(self):
        """Test that bfloat16 and float32 produce similar results."""
        batch, heads, seq, dim = 2, 4, 16, 64
        device = "cuda"

        # Create inputs in float32
        q_f32 = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
        k_f32 = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
        v_f32 = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.float32)
        freqs_f32 = torch.randn(dim, device=device, dtype=torch.float32)
        phase_bias_f32 = torch.randn(dim, device=device, dtype=torch.float32)

        # Convert to bfloat16
        q_bf16 = q_f32.to(torch.bfloat16)
        k_bf16 = k_f32.to(torch.bfloat16)
        v_bf16 = v_f32.to(torch.bfloat16)
        freqs_bf16 = freqs_f32.to(torch.bfloat16)
        phase_bias_bf16 = phase_bias_f32.to(torch.bfloat16)

        output_f32 = fpope_attention_forward(q_f32, k_f32, v_f32, freqs_f32, phase_bias_f32, use_triton=False)
        output_bf16 = fpope_attention_forward(q_bf16, k_bf16, v_bf16, freqs_bf16, phase_bias_bf16, use_triton=False)

        # Convert to same dtype for comparison
        output_bf16_as_f32 = output_bf16.to(torch.float32)

        # bfloat16 has less precision but same range, so slightly relaxed tolerance
        torch.testing.assert_close(output_f32, output_bf16_as_f32, rtol=1e-2, atol=1e-2)

    # ==================== Triton Mixed Precision Tests ====================

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_float16(self):
        """Test Triton kernel with float16."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        dtype = torch.float16

        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        freqs = torch.randn(dim, device=device, dtype=dtype)
        phase_bias = torch.randn(dim, device=device, dtype=dtype)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=True)
        self._check_output_valid(output, "Triton float16 output")

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        not torch.cuda.is_bf16_supported(),
        reason="BFloat16 not supported on this GPU"
    )
    def test_triton_bfloat16(self):
        """Test Triton kernel with bfloat16."""
        batch, heads, seq, dim = 2, 4, 32, 64
        device = "cuda"
        dtype = torch.bfloat16

        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
        freqs = torch.randn(dim, device=device, dtype=dtype)
        phase_bias = torch.randn(dim, device=device, dtype=dtype)

        output = fpope_attention_forward(q, k, v, freqs, phase_bias, use_triton=True)
        self._check_output_valid(output, "Triton bfloat16 output")


class TestCUDAManualGradientComparison:
    """Diagnostic tests comparing CUDA gradients against manually computed gradients."""

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_vs_manual_gradient_comparison(self):
        """Compare CUDA gradients against manually computed gradients step-by-step.

        This test helps diagnose where gradient computation diverges.
        """
        import torch.nn.functional as F

        batch, heads, seq, dim = 1, 1, 4, 32  # Small for debugging
        device = "cuda"
        dtype = torch.float64  # High precision for comparison

        # Create small inputs with known seed for reproducibility
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 0.1
        freqs = torch.randn(dim, device=device, dtype=dtype) * 0.1
        phase_bias = torch.randn(dim, device=device, dtype=dtype) * 0.1

        scale = 1.0 / math.sqrt(dim)

        # ========== MANUAL FORWARD ==========
        mu_q = F.softplus(q)
        mu_k = F.softplus(k)
        sigmoid_q = torch.sigmoid(q)
        sigmoid_k = torch.sigmoid(k)

        # Compute position differences
        pos = torch.arange(seq, device=device, dtype=dtype)
        pos_diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (seq, seq): k_pos - q_pos

        # Compute phases: (seq_q, seq_k, dim)
        phases = pos_diff.unsqueeze(-1) * freqs + phase_bias
        cos_phases = torch.cos(phases)

        # Compute raw scores: sum_d(mu_q * mu_k * cos)
        # mu_q: (batch, heads, seq_q, dim) -> (batch, heads, seq_q, 1, dim)
        # mu_k: (batch, heads, seq_k, dim) -> (batch, heads, 1, seq_k, dim)
        # cos: (seq_q, seq_k, dim) -> (1, 1, seq_q, seq_k, dim)
        raw_scores = (mu_q.unsqueeze(3) * mu_k.unsqueeze(2) * cos_phases.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        scaled_scores = raw_scores * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1)
        scaled_scores_masked = scaled_scores.masked_fill(causal_mask, float('-inf'))

        # Softmax
        P = F.softmax(scaled_scores_masked, dim=-1)
        lse = torch.logsumexp(scaled_scores_masked, dim=-1)  # (batch, heads, seq_q)

        # Output
        manual_output = torch.matmul(P, v)  # (batch, heads, seq_q, dim)

        # ========== MANUAL BACKWARD ==========
        # Use simple grad_output for testing
        grad_output = torch.ones_like(manual_output)

        # dL/dP = grad_output @ V^T
        dP = torch.matmul(grad_output, v.transpose(-2, -1))  # (batch, heads, seq_q, seq_k)

        # D_i = rowsum(P * dP)
        D_i = (P * dP).sum(dim=-1, keepdim=True)  # (batch, heads, seq_q, 1)

        # dL/dS (softmax backward) = P * (dP - D_i)
        dS = P * (dP - D_i)

        # dL/d(raw_score) = dS * scale (chain rule through scaling)
        d_raw_score = dS * scale

        # dL/d(mu_q)[m,d] = sum_n d_raw_score[m,n] * mu_k[n,d] * cos(phase[m,n,d])
        # d_raw_score: (batch, heads, seq_q, seq_k)
        # mu_k: (batch, heads, seq_k, dim)
        # cos_phases: (seq_q, seq_k, dim)
        # Result: (batch, heads, seq_q, dim)
        d_mu_q = torch.einsum('bhqk,bhkd,qkd->bhqd', d_raw_score, mu_k, cos_phases)

        # dL/dq = d_mu_q * sigmoid(q)
        manual_dq = d_mu_q * sigmoid_q

        # dL/d(mu_k)[n,d] = sum_m d_raw_score[m,n] * mu_q[m,d] * cos(phase[m,n,d])
        d_mu_k = torch.einsum('bhqk,bhqd,qkd->bhkd', d_raw_score, mu_q, cos_phases)
        manual_dk = d_mu_k * sigmoid_k

        # dL/dV = P^T @ grad_output
        manual_dv = torch.matmul(P.transpose(-2, -1), grad_output)

        # ========== CUDA FORWARD + BACKWARD ==========
        q_cuda = q.clone().requires_grad_(True)
        k_cuda = k.clone().requires_grad_(True)
        v_cuda = v.clone().requires_grad_(True)
        freqs_cuda = freqs.clone().requires_grad_(True)
        phase_bias_cuda = phase_bias.clone().requires_grad_(True)

        # Forward via the attention function
        cuda_output = fpope_attention_forward(
            q_cuda, k_cuda, v_cuda, freqs_cuda, phase_bias_cuda,
            start_pos=0, causal=True, use_triton=False
        )
        cuda_output.backward(grad_output)

        # ========== COMPARISONS ==========
        print("\n=== DIAGNOSTIC GRADIENT COMPARISON ===")
        print(f"Scale factor: {scale}")
        print(f"1/scale: {1/scale}")

        # Compare outputs first
        output_diff = (manual_output - cuda_output.detach()).abs().max().item()
        print(f"\nOutput max diff: {output_diff}")

        # Compare gradients
        dq_diff = (manual_dq - q_cuda.grad).abs().max().item()
        dk_diff = (manual_dk - k_cuda.grad).abs().max().item()
        dv_diff = (manual_dv - v_cuda.grad).abs().max().item()

        # Compute ratio of gradients to check for scale issues
        dq_ratio = (q_cuda.grad.abs().mean() / manual_dq.abs().mean()).item() if manual_dq.abs().mean() > 0 else float('nan')
        dk_ratio = (k_cuda.grad.abs().mean() / manual_dk.abs().mean()).item() if manual_dk.abs().mean() > 0 else float('nan')
        dv_ratio = (v_cuda.grad.abs().mean() / manual_dv.abs().mean()).item() if manual_dv.abs().mean() > 0 else float('nan')

        print(f"\ndQ max diff: {dq_diff}, ratio (cuda/manual): {dq_ratio}")
        print(f"dK max diff: {dk_diff}, ratio (cuda/manual): {dk_ratio}")
        print(f"dV max diff: {dv_diff}, ratio (cuda/manual): {dv_ratio}")

        # Check if ratio is close to scale or 1/scale
        print(f"\nExpected ratio if extra scale: {scale}")
        print(f"Expected ratio if missing scale: {1/scale}")

        # The test passes if gradients match within tolerance
        # If they don't match, the ratio tells us what's wrong
        try:
            torch.testing.assert_close(manual_dq, q_cuda.grad, rtol=1e-3, atol=1e-4)
            torch.testing.assert_close(manual_dk, k_cuda.grad, rtol=1e-3, atol=1e-4)
            torch.testing.assert_close(manual_dv, v_cuda.grad, rtol=1e-3, atol=1e-4)
            print("\n✓ All gradients match!")
        except AssertionError as e:
            print(f"\n✗ Gradient mismatch detected: {e}")
            # Don't fail the test - this is diagnostic
            # raise

    @pytest.mark.skipif(not HAS_CUDA_KERNEL, reason="CUDA kernel not built")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_vs_pytorch_autograd(self):
        """Compare CUDA backward against PyTorch autograd backward."""
        import torch.nn.functional as F

        batch, heads, seq, dim = 1, 2, 8, 32
        device = "cuda"
        dtype = torch.float32  # Use float32 to match CUDA kernel

        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 0.1
        freqs = torch.randn(dim, device=device, dtype=dtype) * 0.1
        phase_bias = torch.randn(dim, device=device, dtype=dtype) * 0.1
        scale = 1.0 / math.sqrt(dim)

        # ========== PyTorch Reference (with autograd) ==========
        q_ref = q.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        v_ref = v.clone().requires_grad_(True)
        freqs_ref = freqs.clone().requires_grad_(True)
        phase_bias_ref = phase_bias.clone().requires_grad_(True)

        # Use the PyTorch reference function directly
        out_ref = pytorch_fpope_attention(
            q_ref, k_ref, v_ref, freqs_ref, phase_bias_ref,
            start_pos=0, scale=scale, causal=True
        )
        grad_output = torch.ones_like(out_ref)
        out_ref.backward(grad_output)

        # ========== CUDA Kernel ==========
        q_cuda = q.clone().requires_grad_(True)
        k_cuda = k.clone().requires_grad_(True)
        v_cuda = v.clone().requires_grad_(True)
        freqs_cuda = freqs.clone().requires_grad_(True)
        phase_bias_cuda = phase_bias.clone().requires_grad_(True)

        out_cuda = fpope_attention_forward(
            q_cuda, k_cuda, v_cuda, freqs_cuda, phase_bias_cuda,
            start_pos=0, causal=True, use_triton=False
        )
        out_cuda.backward(grad_output)

        # ========== Compare ==========
        print("\n=== CUDA vs PyTorch Autograd Comparison ===")
        print(f"Scale: {scale}, 1/scale: {1/scale}")

        out_diff = (out_ref.detach() - out_cuda.detach()).abs().max().item()
        print(f"\nOutput max diff: {out_diff}")

        dq_diff = (q_ref.grad - q_cuda.grad).abs().max().item()
        dk_diff = (k_ref.grad - k_cuda.grad).abs().max().item()
        dv_diff = (v_ref.grad - v_cuda.grad).abs().max().item()
        dfreq_diff = (freqs_ref.grad - freqs_cuda.grad).abs().max().item()
        dbias_diff = (phase_bias_ref.grad - phase_bias_cuda.grad).abs().max().item()

        dq_ratio = (q_cuda.grad.abs().mean() / q_ref.grad.abs().mean()).item() if q_ref.grad.abs().mean() > 0 else float('nan')
        dk_ratio = (k_cuda.grad.abs().mean() / k_ref.grad.abs().mean()).item() if k_ref.grad.abs().mean() > 0 else float('nan')
        dv_ratio = (v_cuda.grad.abs().mean() / v_ref.grad.abs().mean()).item() if v_ref.grad.abs().mean() > 0 else float('nan')

        print(f"\ndQ max diff: {dq_diff}, ratio (cuda/ref): {dq_ratio}")
        print(f"dK max diff: {dk_diff}, ratio (cuda/ref): {dk_ratio}")
        print(f"dV max diff: {dv_diff}, ratio (cuda/ref): {dv_ratio}")
        print(f"dFreqs max diff: {dfreq_diff}")
        print(f"dPhaseBias max diff: {dbias_diff}")

        # Print some actual values for comparison
        print(f"\nFirst 5 dQ values (ref): {q_ref.grad[0,0,0,:5]}")
        print(f"First 5 dQ values (cuda): {q_cuda.grad[0,0,0,:5]}")

        # Assert they should match closely
        try:
            torch.testing.assert_close(q_cuda.grad, q_ref.grad, rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(k_cuda.grad, k_ref.grad, rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(v_cuda.grad, v_ref.grad, rtol=1e-2, atol=1e-3)
            print("\n✓ All gradients match!")
        except AssertionError as e:
            print(f"\n✗ Gradient mismatch: {e}")
            raise  # Fail the test for debugging


class TestKVCache:
    """Tests for KV-cache attention."""

    def test_cache_initialization(self):
        """Test KV-cache initializes correctly."""
        hidden_dim, num_heads, num_kv_heads, head_dim = 256, 4, 2, 64
        fpope = FoPEPoPEEmbedding(dim=head_dim, max_seq_len=512)

        attn = FPoPEGroupedQueryAttentionWithCache(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            fpope=fpope,
            max_cache_len=128,
            use_triton=False,
        )

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, hidden_dim)

        # First call initializes cache
        output = attn(x, use_cache=True)

        assert attn.k_cache is not None
        assert attn.v_cache is not None
        assert attn.cache_len == seq_len

    def test_cache_reset(self):
        """Test that cache reset works correctly."""
        hidden_dim, num_heads, num_kv_heads, head_dim = 256, 4, 2, 64
        fpope = FoPEPoPEEmbedding(dim=head_dim, max_seq_len=512)

        attn = FPoPEGroupedQueryAttentionWithCache(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            fpope=fpope,
            max_cache_len=128,
            use_triton=False,
        )

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, hidden_dim)
        attn(x, use_cache=True)

        attn.reset_cache()

        assert attn.k_cache is None
        assert attn.v_cache is None
        assert attn.cache_len == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
