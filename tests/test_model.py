"""Unit tests for model components."""

import pytest
import torch

from src.model.normalization import RMSNorm, QKNorm
from src.model.rope import RotaryEmbedding, precompute_freqs_cis, apply_rotary_emb
from src.model.ffn import FeedForward, SwiGLU
from src.model.attention import GroupedQueryAttention
from src.model.embeddings import TokenEmbedding
from src.model.transformer import Transformer, TransformerConfig


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(512)
        x = torch.randn(2, 10, 512)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(512)
        x = torch.randn(2, 10, 512)
        out = norm(x)
        rms = (out ** 2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_learnable_weight(self):
        norm = RMSNorm(512)
        assert norm.weight.shape == (512,)
        assert norm.weight.requires_grad


class TestQKNorm:
    def test_output_shape(self):
        qk_norm = QKNorm(64)
        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 4, 10, 64)
        q_out, k_out = qk_norm(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_unit_norm(self):
        qk_norm = QKNorm(64)
        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 4, 10, 64)
        q_out, k_out = qk_norm(q, k)
        q_norms = q_out.norm(dim=-1)
        k_norms = k_out.norm(dim=-1)
        assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)
        assert torch.allclose(k_norms, torch.ones_like(k_norms), atol=1e-5)


class TestRoPE:
    def test_freqs_cis_shape(self):
        freqs = precompute_freqs_cis(64, 512)
        assert freqs.shape == (512, 32)
        assert freqs.dtype == torch.complex64

    def test_apply_rotary_emb_shape(self):
        freqs = precompute_freqs_cis(64, 10)
        q = torch.randn(2, 10, 8, 64)
        k = torch.randn(2, 10, 4, 64)
        q_out, k_out = apply_rotary_emb(q, k, freqs)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_rotary_embedding_module(self):
        rope = RotaryEmbedding(64, max_seq_len=512)
        q = torch.randn(2, 10, 8, 64)
        k = torch.randn(2, 10, 4, 64)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_extend_seq_len(self):
        rope = RotaryEmbedding(64, max_seq_len=512)
        assert rope.freqs_cis.shape[0] == 512
        rope.extend_seq_len(1024)
        assert rope.freqs_cis.shape[0] == 1024


class TestSwiGLU:
    def test_activation(self):
        swiglu = SwiGLU()
        x = torch.randn(2, 10, 512)
        gate = torch.randn(2, 10, 512)
        out = swiglu(x, gate)
        assert out.shape == x.shape


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(512)
        x = torch.randn(2, 10, 512)
        out = ffn(x)
        assert out.shape == x.shape

    def test_custom_ffn_dim(self):
        ffn = FeedForward(512, ffn_dim=2048)
        assert ffn.w1.out_features == 2048
        assert ffn.w2.in_features == 2048


class TestGroupedQueryAttention:
    def test_output_shape(self):
        rope = RotaryEmbedding(64, max_seq_len=512)
        attn = GroupedQueryAttention(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=4,
            head_dim=64,
            rope=rope,
        )
        x = torch.randn(2, 10, 512)
        out = attn(x)
        assert out.shape == x.shape

    def test_without_qk_norm(self):
        rope = RotaryEmbedding(64, max_seq_len=512)
        attn = GroupedQueryAttention(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=4,
            use_qk_norm=False,
            rope=rope,
        )
        x = torch.randn(2, 10, 512)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_ratio(self):
        rope = RotaryEmbedding(64, max_seq_len=512)
        attn = GroupedQueryAttention(
            hidden_dim=512,
            num_heads=8,
            num_kv_heads=2,
            rope=rope,
        )
        assert attn.num_groups == 4


class TestTokenEmbedding:
    def test_output_shape(self):
        embed = TokenEmbedding(50257, 512)
        x = torch.randint(0, 50257, (2, 10))
        out = embed(x)
        assert out.shape == (2, 10, 512)

    def test_weight_sharing(self):
        embed = TokenEmbedding(50257, 512)
        assert embed.weight.shape == (50257, 512)


class TestTransformerConfig:
    def test_default_config(self):
        config = TransformerConfig()
        assert config.vocab_size == 50257
        assert config.hidden_dim == 512
        assert config.num_layers == 24

    def test_param_count(self):
        config = TransformerConfig()
        assert config.num_params > 0
        assert config.num_params < 200_000_000


class TestTransformer:
    @pytest.fixture
    def small_config(self):
        return TransformerConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            max_seq_len=64,
        )

    def test_forward_logits(self, small_config):
        model = Transformer(small_config)
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 10, 1000)

    def test_forward_with_labels(self, small_config):
        model = Transformer(small_config)
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        output = model(input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].ndim == 0

    def test_gradient_flow(self, small_config):
        model = Transformer(small_config)
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        output = model(input_ids, labels=labels)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_tied_embeddings(self, small_config):
        small_config.tie_embeddings = True
        model = Transformer(small_config)
        assert model.output.weight is model.token_embedding.weight

    def test_untied_embeddings(self, small_config):
        small_config.tie_embeddings = False
        model = Transformer(small_config)
        assert model.output.weight is not model.token_embedding.weight

    def test_extend_context_length(self, small_config):
        model = Transformer(small_config)
        model.extend_context_length(128)
        assert model.config.max_seq_len == 128
        assert model.rope.max_seq_len == 128

    def test_generate(self, small_config):
        model = Transformer(small_config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 5))
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=10)
        assert output.shape == (1, 15)

    def test_gradient_checkpointing(self, small_config):
        small_config.gradient_checkpointing = True
        model = Transformer(small_config)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        output = model(input_ids, labels=labels)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestRoPEVariants:
    def test_ntk_aware_rope(self):
        from src.model.rope import NTKAwareRotaryEmbedding
        rope = NTKAwareRotaryEmbedding(64, max_seq_len=512, scale=2.0)
        q = torch.randn(2, 10, 8, 64)
        k = torch.randn(2, 10, 4, 64)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape

    def test_yarn_rope(self):
        from src.model.rope import YaRNRotaryEmbedding
        rope = YaRNRotaryEmbedding(64, max_seq_len=1024, scale=2.0, original_max_seq_len=512)
        q = torch.randn(2, 10, 8, 64)
        k = torch.randn(2, 10, 4, 64)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
