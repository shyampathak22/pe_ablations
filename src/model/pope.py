"""Pure PoPE (Polar Position Embedding) implementation.

This module implements pure PoPE without FoPE's Fourier mixing, matching
Cinnamon's reference implementation. Optionally supports floor frequency
clipping for undertrained frequencies.

Key differences from FoPE+PoPE (fpope.py):
- NO Fourier mixing: frequencies are fixed, not learned
- Uses standard RoPE-style frequencies: theta^(-i/d)
- Optional floor clipping (pope_floor variant)

References:
- PoPE: "PoPE: Polar Position Embedding for Length Generalization"
- Cinnamon reference: cinnamon/src/attention.py:131-191
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoPEEmbedding(nn.Module):
    """Pure Polar Position Encoding (matching Cinnamon's reference).

    PoPE decouples "what" (content) from "where" (position):
    - Magnitudes via softplus(x) - pure content signal
    - Phases purely position-dependent + learnable bias δ
    - Full frequency count (dim instead of dim/2 like RoPE)

    Formulation:
    - Query at position t: [μ·cos(tθ), μ·sin(tθ)]
    - Key at position s:   [μ·cos(sθ+δ), μ·sin(sθ+δ)]
    - Attention: a_ts = Σ μ_q·μ_k·cos((s-t)θ + δ)

    Args:
        dim: Dimension per head (full, not halved like RoPE)
        max_seq_len: Maximum sequence length to support
        theta: Base frequency parameter (like RoPE's 10000)
        training_length: Training sequence length for floor frequency clipping
        use_floor: Whether to apply floor frequency clipping (pope_floor variant)
        delta_init: Initialization for phase bias ("zero" for length gen, "uniform" otherwise)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        training_length: int = 512,
        use_floor: bool = False,
        delta_init: Literal["zero", "uniform"] = "zero",
    ):
        super().__init__()
        self.dim = dim
        self.n_freq = dim
        self.max_seq_len = max_seq_len
        self.theta_base = theta
        self.training_length = training_length
        self.use_floor = use_floor
        self.delta_init = delta_init

        # Floor frequency for clipping under-trained frequencies
        # Frequencies below 2π/training_length can't complete a cycle in training
        self.floor_freq = 2 * math.pi / training_length

        # Fixed frequencies: θ_i = base^(-i/n_freq) for i in [0, n_freq)
        # Matches Cinnamon's PoPE exactly
        freqs = theta ** (-(torch.arange(0, dim, dtype=torch.float32) / dim))
        self.register_buffer("freqs", freqs)

        # Apply floor clipping if enabled (pope_floor variant)
        if use_floor:
            # Zero out frequencies that are too slow (undertrained)
            clipped_freqs = torch.where(
                freqs < self.floor_freq,
                torch.zeros_like(freqs),
                freqs,
            )
            self.register_buffer("effective_freqs", clipped_freqs)
        else:
            # Pure PoPE: use frequencies as-is
            self.register_buffer("effective_freqs", freqs)

        # Cache base angles for efficiency: (max_seq_len, n_freq)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        base_angles = torch.outer(pos, self.effective_freqs)
        self.register_buffer("base_angles", base_angles)

        # Learnable phase bias δ ∈ [-2π, 0] (applied to keys only)
        if delta_init == "zero":
            phase_bias = torch.zeros(dim)
        else:  # uniform
            phase_bias = torch.rand(dim) * 2 * math.pi - 2 * math.pi  # [-2π, 0]
        self.phase_bias = nn.Parameter(phase_bias)

    def _extend_base_angles(self, seq_len: int) -> torch.Tensor:
        """Extend angle cache for longer sequences."""
        if seq_len <= self.base_angles.size(0):
            return self.base_angles[:seq_len]
        pos = torch.arange(seq_len, device=self.freqs.device, dtype=self.freqs.dtype)
        return torch.outer(pos, self.effective_freqs)

    def extend_seq_len(self, new_max_seq_len: int) -> None:
        """Extend the maximum sequence length for evaluation.

        Args:
            new_max_seq_len: New maximum sequence length
        """
        if new_max_seq_len <= self.max_seq_len:
            return

        pos = torch.arange(
            new_max_seq_len,
            dtype=torch.float32,
            device=self.freqs.device,
        )
        base_angles = torch.outer(pos, self.effective_freqs)
        self.register_buffer("base_angles", base_angles, persistent=False)
        self.max_seq_len = new_max_seq_len

    def forward_query(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply PoPE to query (no delta bias).

        Query at position t: [μ·cos(tθ), μ·sin(tθ)]

        Args:
            x: (B, S, H, d_rope) - position portion of query after projection
            start_pos: Starting position for inference

        Returns:
            (B, S, H, d_rope*2) - [μ·cos(tθ), μ·sin(tθ)]
        """
        B, S, H, D = x.shape

        # Magnitude from softplus (content-dependent, always positive)
        mu = F.softplus(x)  # (B, S, H, D)

        # Get phases from cache or compute on-the-fly
        phases = self._extend_base_angles(start_pos + S)[start_pos:start_pos + S]
        phases = phases.view(1, S, 1, D)  # Broadcast over batch and heads

        # Polar to Cartesian
        cos_out = mu * phases.cos()
        sin_out = mu * phases.sin()

        return torch.cat([cos_out, sin_out], dim=-1)  # (B, S, H, D*2)

    def forward_key(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply PoPE to key (with delta bias).

        Key at position s: [μ·cos(sθ+δ), μ·sin(sθ+δ)]

        The delta bias allows keys to shift their phase relative to queries,
        which helps the model learn relative position patterns.

        Args:
            x: (B, S, H, d_rope) - position portion of key after projection
            start_pos: Starting position for inference

        Returns:
            (B, S, H, d_rope*2) - [μ·cos(sθ+δ), μ·sin(sθ+δ)]
        """
        B, S, H, D = x.shape

        # Magnitude from softplus
        mu = F.softplus(x)

        # Clamp delta to [-2π, 0] for stable training (as in Cinnamon)
        delta = self.phase_bias.clamp(-2 * math.pi, 0.0)

        # Get phases from cache and add delta
        phases = self._extend_base_angles(start_pos + S)[start_pos:start_pos + S]
        phases = phases + delta  # Add learnable phase bias
        phases = phases.view(1, S, 1, D)

        # Polar to Cartesian
        cos_out = mu * phases.cos()
        sin_out = mu * phases.sin()

        return torch.cat([cos_out, sin_out], dim=-1)  # (B, S, H, D*2)

    def forward(
        self,
        x: torch.Tensor,
        apply_delta: bool = True,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Unified forward matching Cinnamon's interface.

        Args:
            x: (B, S, H, d_rope) input tensor
            apply_delta: if True, add learnable phase offset (use for keys only)
            start_pos: Starting position for inference

        Returns:
            (B, S, H, d_rope*2) - [μ·cos(phases), μ·sin(phases)]
        """
        if apply_delta:
            return self.forward_key(x, start_pos)
        else:
            return self.forward_query(x, start_pos)
