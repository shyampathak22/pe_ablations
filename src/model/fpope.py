"""FoPE + PoPE Combined Position Encoding.

This module implements a combination of Fourier Position Embedding (FoPE) and
Polar Coordinate Position Embedding (PoPE) to address orthogonal issues with RoPE:

- FoPE: Handles spectrum damage from linear layers/activations + zero-outs under-trained frequencies
- PoPE: Decouples "what" (content) from "where" (position) in attention

References:
- FoPE: "FoPE: Fourier Position Embedding" (hypothetical)
- PoPE: "PoPE: Polar Position Embedding for Length Generalization"
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FoPEPoPEEmbedding(nn.Module):
    """Combined Fourier + Polar Position Encoding.

    FoPE contributions:
    - Fourier series coefficients a_ω for multi-frequency per dimension
    - Floor frequency clipping (ω < 2π/training_len → zero-frequency)

    PoPE contributions:
    - Magnitudes via softplus(q), softplus(k) - pure content
    - Phases purely position-dependent + learnable bias δ_c
    - Doubled frequency count (dim instead of dim/2 like RoPE)

    Args:
        dim: Dimension per head (full, not halved like RoPE)
        max_seq_len: Maximum sequence length to support
        theta: Base frequency parameter (like RoPE's 10000)
        num_fourier_terms: Number of Fourier terms D for frequency mixing
        fourier_sigma: Standard deviation for Fourier coefficient initialization
        training_length: Training sequence length for floor frequency clipping
        delta_init: Initialization for phase bias ("zero" for length gen, "uniform" otherwise)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        num_fourier_terms: int = 64,
        fourier_sigma: float = 0.4,
        training_length: int = 512,
        delta_init: Literal["zero", "uniform"] = "zero",
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.num_fourier_terms = num_fourier_terms
        self.fourier_sigma = fourier_sigma
        self.training_length = training_length
        self.delta_init = delta_init

        # Floor frequency for clipping under-trained frequencies
        # Frequencies below 2π/training_length are too slow to learn
        self.floor_freq = 2 * math.pi / training_length

        # Base frequencies for each dimension (similar to RoPE but full dim)
        # freqs[c] = theta^(-c/dim) for c in [0, dim)
        base_freqs = theta ** (-torch.arange(0, dim, dtype=torch.float32) / dim)
        self.register_buffer("base_freqs", base_freqs)

        # FoPE: Learnable Fourier coefficients for frequency mixing
        # a_ω ~ N(0, fourier_sigma^2) for each dimension and Fourier term
        # Shape: (dim, num_fourier_terms)
        fourier_coeffs = torch.randn(dim, num_fourier_terms) * fourier_sigma
        self.fourier_coeffs = nn.Parameter(fourier_coeffs)

        # Fourier term indices for frequency computation
        # ω_j = j for j in [1, num_fourier_terms]
        fourier_indices = torch.arange(1, num_fourier_terms + 1, dtype=torch.float32)
        self.register_buffer("fourier_indices", fourier_indices)

        # PoPE: Learnable phase bias δ_c per dimension
        if delta_init == "zero":
            phase_bias = torch.zeros(dim)
        else:  # uniform
            phase_bias = torch.rand(dim) * 2 * math.pi - math.pi
        self.phase_bias = nn.Parameter(phase_bias)

        # Precompute position tensor for efficiency
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        self.register_buffer("positions", positions)

    def _compute_effective_freqs(self) -> torch.Tensor:
        """Compute effective frequencies with Fourier mixing and floor clipping.

        Returns:
            Tensor of shape (dim,) containing effective frequencies per dimension
        """
        # Fourier mixing: effective_freq[c] = sum_j(a_{c,j} * base_freq[c] * j)
        # This allows each dimension to have a weighted combination of harmonics
        # Shape: (dim, num_fourier_terms) * (num_fourier_terms,) -> (dim,)
        scaled_indices = self.base_freqs.unsqueeze(1) * self.fourier_indices.unsqueeze(0)
        effective_freqs = (self.fourier_coeffs * scaled_indices).sum(dim=1)

        # Add the base frequency as the fundamental
        effective_freqs = effective_freqs + self.base_freqs

        # Floor frequency clipping: zero out under-trained frequencies
        # Frequencies that can't complete a full cycle in training are unreliable
        effective_freqs = torch.where(
            effective_freqs.abs() < self.floor_freq,
            torch.zeros_like(effective_freqs),
            effective_freqs,
        )

        return effective_freqs

    def get_frequencies_and_phases(
        self,
        seq_len: int,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get frequencies, phases and phase bias for attention computation.

        Args:
            seq_len: Sequence length
            start_pos: Starting position for KV-cache inference

        Returns:
            Tuple of:
            - effective_freqs: (dim,) effective frequencies per dimension
            - positions: (seq_len,) position indices
            - phase_bias: (dim,) learnable phase bias
        """
        effective_freqs = self._compute_effective_freqs()
        positions = self.positions[start_pos : start_pos + seq_len]
        return effective_freqs, positions, self.phase_bias

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare Q/K tensors and position info for PoPE attention.

        Unlike standard RoPE which returns rotated Q/K, FoPE-PoPE returns:
        - Magnitude-transformed Q/K (via softplus for positivity)
        - Frequency and phase information for attention kernel

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
            start_pos: Starting position for KV-cache inference

        Returns:
            Tuple of:
            - mu_q: Query magnitudes (batch, seq_len, num_heads, head_dim)
            - mu_k: Key magnitudes (batch, seq_len, num_kv_heads, head_dim)
            - effective_freqs: (head_dim,) frequencies per dimension
            - positions: (seq_len,) position indices
            - phase_bias: (head_dim,) learnable phase bias
        """
        seq_len = q.shape[1]

        # PoPE: Magnitudes from content (position-independent)
        # softplus ensures positivity while maintaining gradients
        mu_q = F.softplus(q)
        mu_k = F.softplus(k)

        # Get frequency and phase information
        effective_freqs, positions, phase_bias = self.get_frequencies_and_phases(
            seq_len, start_pos
        )

        return mu_q, mu_k, effective_freqs, positions, phase_bias

    def compute_attention_phases(
        self,
        seq_len_q: int,
        seq_len_k: int,
        start_pos: int = 0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute phase matrix for attention scores.

        The phase for position pair (t, s) at dimension c is:
            phase[t, s, c] = (s - t) * ω_c + δ_c

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            start_pos: Starting position offset
            device: Device to create tensors on

        Returns:
            Phase tensor of shape (seq_len_q, seq_len_k, dim)
        """
        if device is None:
            device = self.base_freqs.device

        effective_freqs = self._compute_effective_freqs()

        # Position indices
        pos_q = torch.arange(start_pos, start_pos + seq_len_q, device=device)
        pos_k = torch.arange(start_pos, start_pos + seq_len_k, device=device)

        # Position differences: (seq_len_q, seq_len_k)
        pos_diff = pos_k.unsqueeze(0) - pos_q.unsqueeze(1)

        # Phases: (seq_len_q, seq_len_k, dim)
        phases = pos_diff.unsqueeze(-1) * effective_freqs + self.phase_bias

        return phases

    def extend_seq_len(self, new_max_seq_len: int) -> None:
        """Extend the maximum sequence length for evaluation.

        Unlike RoPE which needs to recompute frequency caches, FoPE-PoPE
        computes phases on-the-fly, so this just updates the position buffer.

        Args:
            new_max_seq_len: New maximum sequence length
        """
        if new_max_seq_len <= self.max_seq_len:
            return

        positions = torch.arange(
            new_max_seq_len,
            dtype=torch.float32,
            device=self.positions.device,
        )
        self.register_buffer("positions", positions, persistent=False)
        self.max_seq_len = new_max_seq_len


def pope_attention_scores(
    mu_q: torch.Tensor,
    mu_k: torch.Tensor,
    phases: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Compute PoPE attention scores (pure PyTorch fallback).

    Computes: a_{t,s} = Σ_c μ_q[t,c] × μ_k[s,c] × cos(phase[t,s,c])

    Args:
        mu_q: Query magnitudes (batch, heads, seq_q, dim)
        mu_k: Key magnitudes (batch, heads, seq_k, dim)
        phases: Phase matrix (seq_q, seq_k, dim)
        causal: Whether to apply causal masking

    Returns:
        Attention scores (batch, heads, seq_q, seq_k)
    """
    batch, heads, seq_q, dim = mu_q.shape
    seq_k = mu_k.shape[2]

    # Compute cosine of phases: (seq_q, seq_k, dim)
    cos_phases = torch.cos(phases)

    # Attention: sum over dimensions of (magnitude_product * cos_phase)
    # mu_q: (batch, heads, seq_q, 1, dim)
    # mu_k: (batch, heads, 1, seq_k, dim)
    # cos_phases: (1, 1, seq_q, seq_k, dim)
    mu_q_expanded = mu_q.unsqueeze(3)  # (batch, heads, seq_q, 1, dim)
    mu_k_expanded = mu_k.unsqueeze(2)  # (batch, heads, 1, seq_k, dim)
    cos_phases_expanded = cos_phases.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_q, seq_k, dim)

    # Element-wise product and sum over dim
    attn_scores = (mu_q_expanded * mu_k_expanded * cos_phases_expanded).sum(dim=-1)

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=mu_q.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    return attn_scores


def pope_attention(
    mu_q: torch.Tensor,
    mu_k: torch.Tensor,
    v: torch.Tensor,
    phases: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Full PoPE attention computation (pure PyTorch fallback).

    Args:
        mu_q: Query magnitudes (batch, heads, seq_q, dim)
        mu_k: Key magnitudes (batch, heads, seq_k, dim)
        v: Value tensor (batch, heads, seq_k, dim)
        phases: Phase matrix (seq_q, seq_k, dim)
        causal: Whether to apply causal masking
        scale: Attention scale factor (default: 1/sqrt(dim))

    Returns:
        Attention output (batch, heads, seq_q, dim)
    """
    dim = mu_q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(dim)

    attn_scores = pope_attention_scores(mu_q, mu_k, phases, causal=causal)
    attn_scores = attn_scores * scale

    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)

    return output
