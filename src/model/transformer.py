"""Transformer model with modern components."""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.alibi import ALiBiAttention
from src.model.attention import GroupedQueryAttention
from src.model.embeddings import TokenEmbedding
from src.model.ffn import FeedForward
from src.model.normalization import RMSNorm
from src.model.rope import RotaryEmbedding, NTKAwareRotaryEmbedding, YaRNRotaryEmbedding
from src.model.fpope import FoPEPoPEEmbedding
from src.model.fpope_attention import FPoPEGroupedQueryAttention


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""

    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    hidden_dim: int = 512
    num_layers: int = 24
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 64
    ffn_dim: int | None = None  # Defaults to 4 * hidden_dim * 2/3 rounded
    ffn_multiple_of: int = 256

    # Positional encoding mode
    pe_mode: Literal["rope", "fpope", "alibi", "nope"] = "rope"

    # General positional encoding
    max_seq_len: int = 512

    # RoPE parameters (used when pe_mode="rope")
    rope_theta: float = 10000.0
    rope_type: Literal["standard", "ntk", "yarn"] = "standard"
    rope_scale: float = 1.0  # For NTK/YaRN scaling

    # FoPE+PoPE parameters (used when pe_mode="fpope")
    fpope_theta: float = 10000.0
    fpope_num_fourier_terms: int = 64
    fpope_sigma: float = 0.4
    fpope_training_length: int = 512  # For floor frequency clipping
    fpope_delta_init: str = "zero"  # "zero" for length gen, "uniform" for in-distribution

    # Normalization
    norm_eps: float = 1e-6
    use_qk_norm: bool = True

    # Regularization
    dropout: float = 0.0

    # Weight tying
    tie_embeddings: bool = True

    # Training
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads

    @property
    def num_params(self) -> int:
        """Estimate total number of parameters."""
        # Embedding
        embed_params = self.vocab_size * self.hidden_dim

        # Per layer
        # Attention: Wq, Wk, Wv, Wo
        attn_params = (
            self.hidden_dim * self.num_heads * self.head_dim +  # Wq
            self.hidden_dim * self.num_kv_heads * self.head_dim +  # Wk
            self.hidden_dim * self.num_kv_heads * self.head_dim +  # Wv
            self.num_heads * self.head_dim * self.hidden_dim  # Wo
        )

        # FFN: W1, W2, W3 (SwiGLU)
        ffn_dim = self.ffn_dim or int(4 * self.hidden_dim * 2 / 3)
        ffn_dim = self.ffn_multiple_of * ((ffn_dim + self.ffn_multiple_of - 1) // self.ffn_multiple_of)
        ffn_params = 3 * self.hidden_dim * ffn_dim

        # Norms: 2 per layer + 1 final
        norm_params = (2 * self.num_layers + 1) * self.hidden_dim

        layer_params = attn_params + ffn_params
        total = embed_params + self.num_layers * layer_params + norm_params

        # Output projection (tied or not)
        if not self.tie_embeddings:
            total += self.vocab_size * self.hidden_dim

        return total


class TransformerBlock(nn.Module):
    """Single transformer block with Pre-RMSNorm architecture."""

    def __init__(
        self,
        config: TransformerConfig,
        rope: RotaryEmbedding | None = None,
        fpope: FoPEPoPEEmbedding | None = None,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        if config.pe_mode == "fpope":
            # Use FPoPE-aware attention
            self.attention = FPoPEGroupedQueryAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                use_qk_norm=config.use_qk_norm,
                fpope=fpope,
            )
        elif config.pe_mode == "alibi":
            # Use ALiBi attention (no learnable position embeddings)
            self.attention = ALiBiAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                use_qk_norm=config.use_qk_norm,
            )
        elif config.pe_mode == "nope":
            # No positional encoding - standard attention without RoPE
            self.attention = GroupedQueryAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                use_qk_norm=config.use_qk_norm,
                rope=None,  # Explicitly no RoPE
            )
        else:
            # Use standard RoPE attention
            self.attention = GroupedQueryAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                use_qk_norm=config.use_qk_norm,
                rope=rope,
            )

        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            multiple_of=config.ffn_multiple_of,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Uses Pre-RMSNorm: norm -> attention/ffn -> residual

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask
            start_pos: Starting position for RoPE

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        x = x + self.attention(self.attention_norm(x), mask, start_pos)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Transformer language model with modern architecture.

    Features:
    - Pre-RMSNorm (norm before attention/FFN)
    - Grouped Query Attention (GQA)
    - QKNorm for stable training
    - SwiGLU activation in FFN
    - Rotary Position Embeddings (RoPE)
    - Optional gradient checkpointing
    - Optional tied embeddings
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_dim)

        # Initialize positional encoding based on pe_mode
        self.rope = None
        self.fpope = None

        if config.pe_mode == "rope":
            # Standard RoPE variants
            if config.rope_type == "standard":
                self.rope = RotaryEmbedding(
                    config.head_dim,
                    config.max_seq_len,
                    config.rope_theta,
                )
            elif config.rope_type == "ntk":
                self.rope = NTKAwareRotaryEmbedding(
                    config.head_dim,
                    config.max_seq_len,
                    config.rope_theta,
                    config.rope_scale,
                )
            elif config.rope_type == "yarn":
                self.rope = YaRNRotaryEmbedding(
                    config.head_dim,
                    config.max_seq_len,
                    config.rope_theta,
                    config.rope_scale,
                    original_max_seq_len=512,
                )
            else:
                raise ValueError(f"Unknown rope_type: {config.rope_type}")
        elif config.pe_mode == "fpope":
            # FoPE+PoPE combined encoding
            self.fpope = FoPEPoPEEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.fpope_theta,
                num_fourier_terms=config.fpope_num_fourier_terms,
                fourier_sigma=config.fpope_sigma,
                training_length=config.fpope_training_length,
                delta_init=config.fpope_delta_init,
            )
        elif config.pe_mode == "alibi":
            # ALiBi handles position internally via linear biases
            # No shared positional embedding needed
            pass
        elif config.pe_mode == "nope":
            # No positional encoding at all
            pass
        else:
            raise ValueError(f"Unknown pe_mode: {config.pe_mode}")

        self.layers = nn.ModuleList([
            TransformerBlock(config, rope=self.rope, fpope=self.fpope)
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        self.output = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.token_embedding.weight

        self.gradient_checkpointing = config.gradient_checkpointing

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the transformer.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            labels: Target token indices for loss computation (batch, seq_len)
            start_pos: Starting position for RoPE (for KV-cache inference)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    None,
                    start_pos,
                    use_reentrant=False,
                )
            else:
                x = layer(x, mask=None, start_pos=start_pos)

        x = self.norm(x)
        logits = self.output(x)

        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    def extend_context_length(self, new_max_seq_len: int) -> None:
        """Extend the model's context length for evaluation.

        Args:
            new_max_seq_len: New maximum sequence length
        """
        if self.rope is not None:
            self.rope.extend_seq_len(new_max_seq_len)
        if self.fpope is not None:
            self.fpope.extend_seq_len(new_max_seq_len)
        self.config.max_seq_len = new_max_seq_len

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting token indices of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if provided)
            top_p: Nucleus sampling threshold (if provided)

        Returns:
            Generated token indices of shape (batch, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                self.extend_context_length(input_ids.shape[1])

            output = self.forward(input_ids)
            logits = output["logits"][:, -1, :]

            # Temperature=0 means greedy decoding
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                continue

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
