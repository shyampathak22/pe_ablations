"""Token embedding layer."""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding with optional weight tying.

    Provides input embeddings that can be shared with the output projection
    layer (tied embeddings) for parameter efficiency.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed token indices.

        Args:
            x: Token indices of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        return self.embedding(x)

    @property
    def weight(self) -> torch.Tensor:
        """Return embedding weights for weight tying."""
        return self.embedding.weight
