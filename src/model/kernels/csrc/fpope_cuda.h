// FPoPE CUDA Kernel Header
// Native CUDA implementation for FoPE+PoPE attention
//
// Mathematical operation:
//   scores[m,n] = sum_d(mu_q[m,d] * mu_k[n,d] * cos((n-m) * freq[d] + bias[d]))
//   where mu = softplus(x) is precomputed in PyTorch

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block sizes for kernel tuning
constexpr int BLOCK_M = 64;  // Q positions per block (forward)
constexpr int BLOCK_N = 64;  // K positions per block (forward)
constexpr int WARP_SIZE = 32;

// Smaller block sizes for backward (fits in shared memory with dim=128)
constexpr int BWD_BLOCK_M = 32;  // Q positions per block (backward)
constexpr int BWD_BLOCK_N = 32;  // K positions per block (backward)

// Maximum head dimension supported
constexpr int MAX_HEAD_DIM = 128;

// Forward pass declarations
std::vector<torch::Tensor> fpope_attention_forward_cuda(
    torch::Tensor mu_q,        // (batch*heads, seq_q, dim) - precomputed softplus(q)
    torch::Tensor mu_k,        // (batch*heads, seq_k, dim) - precomputed softplus(k)
    torch::Tensor v,           // (batch*heads, seq_k, dim)
    torch::Tensor freqs,       // (dim,)
    torch::Tensor phase_bias,  // (dim,)
    int start_pos,
    float scale,
    bool return_lse
);

// Backward pass declarations
std::vector<torch::Tensor> fpope_attention_backward_cuda(
    torch::Tensor mu_q,        // (batch*heads, seq_q, dim)
    torch::Tensor mu_k,        // (batch*heads, seq_k, dim)
    torch::Tensor v,           // (batch*heads, seq_k, dim)
    torch::Tensor freqs,       // (dim,)
    torch::Tensor phase_bias,  // (dim,)
    torch::Tensor output,      // (batch*heads, seq_q, dim)
    torch::Tensor grad_output, // (batch*heads, seq_q, dim)
    torch::Tensor lse,         // (batch*heads, seq_q)
    torch::Tensor sigmoid_q,   // (batch*heads, seq_q, dim) - for chain rule
    torch::Tensor sigmoid_k,   // (batch*heads, seq_k, dim) - for chain rule
    int start_pos,
    float scale
);

// Utility: ceiling division
inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}
