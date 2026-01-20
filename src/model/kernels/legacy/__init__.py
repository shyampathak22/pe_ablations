"""Legacy FPoPE attention kernels.

These kernels implemented the original FPoPE attention where ALL dimensions
go through softplus × cos(phase). This approach required custom CUDA/Triton
kernels for the non-standard attention score computation.

The new architecture (as of restructuring) uses the Cinnamon-style split:
- Content path: Standard dot-product (no positional encoding)
- Position path: PoPE transform (softplus → cos/sin)
- Final: Concatenate and use standard attention

The legacy kernels are preserved for:
1. Reference and comparison
2. Potential future experimentation
3. Backward compatibility with old checkpoints (with conversion)
"""
