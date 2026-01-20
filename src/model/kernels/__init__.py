"""Custom kernels for attention computation (LEGACY).

Note: The new FPoPE architecture uses standard dot-product attention
after the content+position split, so custom kernels are no longer required.

Legacy kernels are preserved in src/model/kernels/legacy/ for reference.
"""

# Legacy kernel imports - only import if explicitly needed
# The new FPoPE architecture uses standard attention, not custom kernels
try:
    from src.model.kernels.legacy.fpope_attention_legacy import (
        fpope_attention_forward,
        FPoPEAttentionFunction,
        HAS_TRITON,
    )
    HAS_LEGACY_KERNELS = True
except ImportError:
    HAS_LEGACY_KERNELS = False
    HAS_TRITON = False
    fpope_attention_forward = None
    FPoPEAttentionFunction = None

__all__ = [
    "fpope_attention_forward",
    "FPoPEAttentionFunction",
    "HAS_TRITON",
    "HAS_LEGACY_KERNELS",
]
