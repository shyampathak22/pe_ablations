"""Normalization layers: RMSNorm and QKNorm."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simplification of LayerNorm that removes the mean-centering
    operation, computing only the RMS statistics for normalization.
    This is more computationally efficient while maintaining similar performance.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class QKNorm(nn.Module):
    """Query-Key L2 Normalization.

    Applies L2 normalization to query and key tensors before computing attention.
    This helps stabilize attention patterns, especially for long sequences.

    Reference: https://arxiv.org/abs/2302.05442 (Scaling Vision Transformers to 22B)
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Q and K tensors.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)

        Returns:
            Normalized (q, k) tuple
        """
        q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        k = k / (k.norm(dim=-1, keepdim=True) + self.eps)
        return q, k
