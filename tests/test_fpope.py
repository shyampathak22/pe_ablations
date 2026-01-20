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
