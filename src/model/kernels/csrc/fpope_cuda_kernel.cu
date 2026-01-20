// FPoPE CUDA Kernels
// Native CUDA implementation for FoPE+PoPE attention
//
// Key optimizations over Triton:
// 1. Load freqs/phase_bias into shared memory once per block
// 2. Warp-level reductions with __shfl_down_sync for softmax
// 3. Fine-grained shared memory management to avoid bank conflicts
// 4. Pre-compiled kernels (no JIT overhead)

#include "fpope_cuda.h"
#include <cuda_fp16.h>
#include <cmath>

// Fast math intrinsics
#define CUDA_FAST_COSF __cosf
#define CUDA_FAST_SINF __sinf
#define CUDA_FAST_EXPF __expf

// =============================================================================
// Forward Kernel
// =============================================================================
// Grid: (cdiv(seq_q, BLOCK_M), batch_heads)
// Block: (BLOCK_M) - each thread handles one Q position

template <int HEAD_DIM>
__global__ void fpope_attention_fwd_kernel(
    const float* __restrict__ mu_q,
    const float* __restrict__ mu_k,
    const float* __restrict__ v,
    const float* __restrict__ freqs,
    const float* __restrict__ phase_bias,
    float* __restrict__ out,
    float* __restrict__ lse,
    int batch_heads,
    int seq_q,
    int seq_k,
    int start_pos,
    float scale
) {
    int pid_m = blockIdx.x;
    int pid_bh = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int m = pid_m * BLOCK_M + tid;
    bool m_valid = m < seq_q;

    extern __shared__ float smem[];
    float* s_freqs = smem;
    float* s_bias = smem + HEAD_DIM;
    float* s_mu_k = smem + 2 * HEAD_DIM;

    // Load freqs and phase_bias into shared memory
    for (int d = tid; d < HEAD_DIM; d += num_threads) {
        s_freqs[d] = freqs[d];
        s_bias[d] = phase_bias[d];
    }
    __syncthreads();

    // Load mu_q for this position into registers
    float r_mu_q[HEAD_DIM];
    if (m_valid) {
        int q_offset = pid_bh * seq_q * HEAD_DIM + m * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            r_mu_q[d] = __ldg(&mu_q[q_offset + d]);
        }
    }

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }

    float pos_q = (float)(start_pos + m);

    // Iterate over K blocks
    for (int start_n = 0; start_n < seq_k; start_n += BLOCK_N) {
        __syncthreads();
        // Load mu_k tile into shared memory
        for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += num_threads) {
            int n_local = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int n = start_n + n_local;
            if (n < seq_k) {
                s_mu_k[n_local * HEAD_DIM + d] = __ldg(&mu_k[pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM + d]);
            } else {
                s_mu_k[n_local * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        if (m_valid) {
            // Compute scores for this K block
            float scores[BLOCK_N];
            #pragma unroll
            for (int n_local = 0; n_local < BLOCK_N; n_local++) {
                int n = start_n + n_local;
                bool valid = (n < seq_k) && (m >= n);  // causal

                if (valid) {
                    float pos_k = (float)(start_pos + n);
                    float pos_diff = pos_k - pos_q;
                    float score = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float phase = pos_diff * s_freqs[d] + s_bias[d];
                        score += r_mu_q[d] * s_mu_k[n_local * HEAD_DIM + d] * CUDA_FAST_COSF(phase);
                    }
                    scores[n_local] = score * scale;
                } else {
                    scores[n_local] = -INFINITY;
                }
            }

            // Online softmax update
            float m_ij = -INFINITY;
            #pragma unroll
            for (int n_local = 0; n_local < BLOCK_N; n_local++) {
                m_ij = fmaxf(m_ij, scores[n_local]);
            }

            float m_new = fmaxf(m_i, m_ij);
            float alpha = CUDA_FAST_EXPF(m_i - m_new);

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] *= alpha;
            }
            l_i *= alpha;

            float block_sum = 0.0f;
            #pragma unroll
            for (int n_local = 0; n_local < BLOCK_N; n_local++) {
                int n = start_n + n_local;
                float p = (scores[n_local] > -INFINITY) ? CUDA_FAST_EXPF(scores[n_local] - m_ij) : 0.0f;
                block_sum += p;

                if (n < seq_k && p > 0.0f) {
                    int v_offset = pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[d] += p * __ldg(&v[v_offset + d]);
                    }
                }
            }

            float beta = CUDA_FAST_EXPF(m_ij - m_new);
            l_i += beta * block_sum;
            m_i = m_new;
        }
    }

    // Normalize and store
    if (m_valid && l_i > 0.0f) {
        int out_offset = pid_bh * seq_q * HEAD_DIM + m * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            out[out_offset + d] = acc[d] / l_i;
        }
        lse[pid_bh * seq_q + m] = m_i + logf(l_i);
    }
}

// =============================================================================
// Backward Kernel - Precompute D (rowsum of P * dP)
// D[m] = sum_n P[m,n] * (grad_out[m] · V[n])
// =============================================================================
template <int HEAD_DIM>
__global__ void fpope_compute_Di_kernel(
    const float* __restrict__ mu_q,
    const float* __restrict__ mu_k,
    const float* __restrict__ v,
    const float* __restrict__ freqs,
    const float* __restrict__ phase_bias,
    const float* __restrict__ grad_output,
    const float* __restrict__ lse,
    float* __restrict__ Di,
    int batch_heads,
    int seq_q,
    int seq_k,
    int start_pos,
    float scale
) {
    int pid_m = blockIdx.x;
    int pid_bh = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int m = pid_m * BWD_BLOCK_M + tid;
    bool m_valid = m < seq_q;

    extern __shared__ float smem[];
    float* s_freqs = smem;
    float* s_bias = smem + HEAD_DIM;
    float* s_mu_k = smem + 2 * HEAD_DIM;
    float* s_v = smem + 2 * HEAD_DIM + BWD_BLOCK_N * HEAD_DIM;

    for (int d = tid; d < HEAD_DIM; d += num_threads) {
        s_freqs[d] = freqs[d];
        s_bias[d] = phase_bias[d];
    }
    __syncthreads();

    float r_mu_q[HEAD_DIM], r_grad_out[HEAD_DIM];
    float r_lse = 0.0f;

    if (m_valid) {
        int offset = pid_bh * seq_q * HEAD_DIM + m * HEAD_DIM;
        r_lse = lse[pid_bh * seq_q + m];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            r_mu_q[d] = __ldg(&mu_q[offset + d]);
            r_grad_out[d] = __ldg(&grad_output[offset + d]);
        }
    }

    float pos_q = (float)(start_pos + m);
    float D_i = 0.0f;

    for (int start_n = 0; start_n < seq_k; start_n += BWD_BLOCK_N) {
        __syncthreads();
        for (int idx = tid; idx < BWD_BLOCK_N * HEAD_DIM; idx += num_threads) {
            int n_local = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int n = start_n + n_local;
            if (n < seq_k) {
                int k_offset = pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM + d;
                s_mu_k[n_local * HEAD_DIM + d] = __ldg(&mu_k[k_offset]);
                s_v[n_local * HEAD_DIM + d] = __ldg(&v[k_offset]);
            } else {
                s_mu_k[n_local * HEAD_DIM + d] = 0.0f;
                s_v[n_local * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        if (m_valid) {
            for (int n_local = 0; n_local < BWD_BLOCK_N; n_local++) {
                int n = start_n + n_local;
                if (n < seq_k && m >= n) {
                    float pos_k = (float)(start_pos + n);
                    float pos_diff = pos_k - pos_q;

                    float score = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float phase = pos_diff * s_freqs[d] + s_bias[d];
                        score += r_mu_q[d] * s_mu_k[n_local * HEAD_DIM + d] * CUDA_FAST_COSF(phase);
                    }
                    score *= scale;
                    float p = CUDA_FAST_EXPF(score - r_lse);

                    float dp = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dp += r_grad_out[d] * s_v[n_local * HEAD_DIM + d];
                    }
                    D_i += p * dp;
                }
            }
        }
    }

    if (m_valid) {
        Di[pid_bh * seq_q + m] = D_i;
    }
}

// =============================================================================
// Backward Kernel - dQ (uses precomputed Di)
// =============================================================================
template <int HEAD_DIM>
__global__ void fpope_attention_bwd_dq_kernel(
    const float* __restrict__ mu_q,
    const float* __restrict__ mu_k,
    const float* __restrict__ sigmoid_q,
    const float* __restrict__ v,
    const float* __restrict__ freqs,
    const float* __restrict__ phase_bias,
    const float* __restrict__ grad_output,
    const float* __restrict__ lse,
    const float* __restrict__ Di,
    float* __restrict__ dq,
    int batch_heads,
    int seq_q,
    int seq_k,
    int start_pos,
    float scale
) {
    int pid_m = blockIdx.x;
    int pid_bh = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int m = pid_m * BWD_BLOCK_M + tid;
    bool m_valid = m < seq_q;

    extern __shared__ float smem[];
    float* s_freqs = smem;
    float* s_bias = smem + HEAD_DIM;
    float* s_mu_k = smem + 2 * HEAD_DIM;
    float* s_v = smem + 2 * HEAD_DIM + BWD_BLOCK_N * HEAD_DIM;

    for (int d = tid; d < HEAD_DIM; d += num_threads) {
        s_freqs[d] = freqs[d];
        s_bias[d] = phase_bias[d];
    }
    __syncthreads();

    float r_mu_q[HEAD_DIM], r_sigmoid_q[HEAD_DIM], r_grad_out[HEAD_DIM];
    float r_lse = 0.0f;
    float D_i = 0.0f;

    if (m_valid) {
        int offset = pid_bh * seq_q * HEAD_DIM + m * HEAD_DIM;
        r_lse = lse[pid_bh * seq_q + m];
        D_i = Di[pid_bh * seq_q + m];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            r_mu_q[d] = __ldg(&mu_q[offset + d]);
            r_sigmoid_q[d] = __ldg(&sigmoid_q[offset + d]);
            r_grad_out[d] = __ldg(&grad_output[offset + d]);
        }
    }

    float r_dq[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        r_dq[d] = 0.0f;
    }

    float pos_q = (float)(start_pos + m);

    // Single pass: compute dQ using precomputed Di
    for (int start_n = 0; start_n < seq_k; start_n += BWD_BLOCK_N) {
        __syncthreads();
        for (int idx = tid; idx < BWD_BLOCK_N * HEAD_DIM; idx += num_threads) {
            int n_local = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int n = start_n + n_local;
            if (n < seq_k) {
                int k_offset = pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM + d;
                s_mu_k[n_local * HEAD_DIM + d] = __ldg(&mu_k[k_offset]);
                s_v[n_local * HEAD_DIM + d] = __ldg(&v[k_offset]);
            } else {
                s_mu_k[n_local * HEAD_DIM + d] = 0.0f;
                s_v[n_local * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        if (m_valid) {
            for (int n_local = 0; n_local < BWD_BLOCK_N; n_local++) {
                int n = start_n + n_local;
                if (n < seq_k && m >= n) {
                    float pos_k = (float)(start_pos + n);
                    float pos_diff = pos_k - pos_q;

                    float score = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float phase = pos_diff * s_freqs[d] + s_bias[d];
                        score += r_mu_q[d] * s_mu_k[n_local * HEAD_DIM + d] * CUDA_FAST_COSF(phase);
                    }
                    score *= scale;
                    float p = CUDA_FAST_EXPF(score - r_lse);

                    float dp = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dp += r_grad_out[d] * s_v[n_local * HEAD_DIM + d];
                    }

                    float ds = p * (dp - D_i) * scale;

                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float phase = pos_diff * s_freqs[d] + s_bias[d];
                        r_dq[d] += ds * s_mu_k[n_local * HEAD_DIM + d] * CUDA_FAST_COSF(phase);
                    }
                }
            }
        }
    }

    if (m_valid) {
        int out_offset = pid_bh * seq_q * HEAD_DIM + m * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            dq[out_offset + d] = r_dq[d] * r_sigmoid_q[d];
        }
    }
}

// =============================================================================
// Backward Kernel - dK (uses precomputed Di, also computes partial dFreq/dBias)
// =============================================================================
template <int HEAD_DIM>
__global__ void fpope_attention_bwd_dk_kernel(
    const float* __restrict__ mu_q,
    const float* __restrict__ mu_k,
    const float* __restrict__ sigmoid_k,
    const float* __restrict__ v,
    const float* __restrict__ freqs,
    const float* __restrict__ phase_bias,
    const float* __restrict__ grad_output,
    const float* __restrict__ lse,
    const float* __restrict__ Di,
    float* __restrict__ dk,
    float* __restrict__ dfreq_partial,
    float* __restrict__ dbias_partial,
    int batch_heads,
    int seq_q,
    int seq_k,
    int start_pos,
    float scale,
    int num_k_blocks
) {
    int pid_n = blockIdx.x;
    int pid_bh = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int block_id = pid_bh * num_k_blocks + pid_n;

    int n = pid_n * BWD_BLOCK_N + tid;
    bool n_valid = n < seq_k;

    extern __shared__ float smem[];
    float* s_freqs = smem;
    float* s_bias = smem + HEAD_DIM;
    float* s_mu_q = smem + 2 * HEAD_DIM;
    float* s_grad_out = smem + 2 * HEAD_DIM + BWD_BLOCK_M * HEAD_DIM;
    float* s_Di = smem + 2 * HEAD_DIM + 2 * BWD_BLOCK_M * HEAD_DIM;

    for (int d = tid; d < HEAD_DIM; d += num_threads) {
        s_freqs[d] = freqs[d];
        s_bias[d] = phase_bias[d];
    }
    __syncthreads();

    float r_mu_k[HEAD_DIM], r_sigmoid_k[HEAD_DIM], r_v[HEAD_DIM];
    if (n_valid) {
        int offset = pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            r_mu_k[d] = __ldg(&mu_k[offset + d]);
            r_sigmoid_k[d] = __ldg(&sigmoid_k[offset + d]);
            r_v[d] = __ldg(&v[offset + d]);
        }
    }

    float r_dk[HEAD_DIM], r_dfreq[HEAD_DIM], r_dbias[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        r_dk[d] = 0.0f;
        r_dfreq[d] = 0.0f;
        r_dbias[d] = 0.0f;
    }

    float pos_k = (float)(start_pos + n);

    for (int start_m = 0; start_m < seq_q; start_m += BWD_BLOCK_M) {
        __syncthreads();
        // Load mu_q, grad_out, and Di for this Q block
        for (int idx = tid; idx < BWD_BLOCK_M * HEAD_DIM; idx += num_threads) {
            int m_local = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int m = start_m + m_local;
            if (m < seq_q) {
                int q_offset = pid_bh * seq_q * HEAD_DIM + m * HEAD_DIM + d;
                s_mu_q[m_local * HEAD_DIM + d] = __ldg(&mu_q[q_offset]);
                s_grad_out[m_local * HEAD_DIM + d] = __ldg(&grad_output[q_offset]);
            } else {
                s_mu_q[m_local * HEAD_DIM + d] = 0.0f;
                s_grad_out[m_local * HEAD_DIM + d] = 0.0f;
            }
        }
        // Load Di for this Q block
        for (int m_local = tid; m_local < BWD_BLOCK_M; m_local += num_threads) {
            int m = start_m + m_local;
            if (m < seq_q) {
                s_Di[m_local] = Di[pid_bh * seq_q + m];
            } else {
                s_Di[m_local] = 0.0f;
            }
        }
        __syncthreads();

        if (n_valid) {
            for (int m_local = 0; m_local < BWD_BLOCK_M; m_local++) {
                int m = start_m + m_local;
                if (m < seq_q && m >= n) {
                    float pos_q = (float)(start_pos + m);
                    float pos_diff = pos_k - pos_q;
                    float r_lse = lse[pid_bh * seq_q + m];
                    float D_i = s_Di[m_local];

                    float score = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float phase = pos_diff * s_freqs[d] + s_bias[d];
                        score += s_mu_q[m_local * HEAD_DIM + d] * r_mu_k[d] * CUDA_FAST_COSF(phase);
                    }
                    score *= scale;
                    float p = CUDA_FAST_EXPF(score - r_lse);

                    float dp = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dp += s_grad_out[m_local * HEAD_DIM + d] * r_v[d];
                    }

                    // Correct ds with D_i term
                    float ds = p * (dp - D_i) * scale;

                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float phase = pos_diff * s_freqs[d] + s_bias[d];
                        float cos_phase = CUDA_FAST_COSF(phase);
                        float sin_phase = CUDA_FAST_SINF(phase);
                        float mu_q_d = s_mu_q[m_local * HEAD_DIM + d];

                        r_dk[d] += ds * mu_q_d * cos_phase;
                        r_dfreq[d] += -ds * mu_q_d * r_mu_k[d] * sin_phase * pos_diff;
                        r_dbias[d] += -ds * mu_q_d * r_mu_k[d] * sin_phase;
                    }
                }
            }
        }
    }

    if (n_valid) {
        int out_offset = pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            dk[out_offset + d] = r_dk[d] * r_sigmoid_k[d];
        }
    }

    // Reduce dfreq/dbias within block using shared memory
    __syncthreads();
    float* s_dfreq = smem;
    float* s_dbias = smem + HEAD_DIM;

    for (int d = tid; d < HEAD_DIM; d += num_threads) {
        s_dfreq[d] = 0.0f;
        s_dbias[d] = 0.0f;
    }
    __syncthreads();

    for (int d = 0; d < HEAD_DIM; d++) {
        atomicAdd(&s_dfreq[d], r_dfreq[d]);
        atomicAdd(&s_dbias[d], r_dbias[d]);
    }
    __syncthreads();

    for (int d = tid; d < HEAD_DIM; d += num_threads) {
        dfreq_partial[block_id * HEAD_DIM + d] = s_dfreq[d];
        dbias_partial[block_id * HEAD_DIM + d] = s_dbias[d];
    }
}

// =============================================================================
// Backward Kernel - dV
// =============================================================================
template <int HEAD_DIM>
__global__ void fpope_attention_bwd_dv_kernel(
    const float* __restrict__ mu_q,
    const float* __restrict__ mu_k,
    const float* __restrict__ freqs,
    const float* __restrict__ phase_bias,
    const float* __restrict__ grad_output,
    const float* __restrict__ lse,
    float* __restrict__ dv,
    int batch_heads,
    int seq_q,
    int seq_k,
    int start_pos,
    float scale
) {
    int pid_n = blockIdx.x;
    int pid_bh = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int n = pid_n * BWD_BLOCK_N + tid;
    bool n_valid = n < seq_k;

    extern __shared__ float smem[];
    float* s_freqs = smem;
    float* s_bias = smem + HEAD_DIM;
    float* s_mu_q = smem + 2 * HEAD_DIM;
    float* s_grad_out = smem + 2 * HEAD_DIM + BWD_BLOCK_M * HEAD_DIM;

    for (int d = tid; d < HEAD_DIM; d += num_threads) {
        s_freqs[d] = freqs[d];
        s_bias[d] = phase_bias[d];
    }
    __syncthreads();

    float r_mu_k[HEAD_DIM];
    if (n_valid) {
        int offset = pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            r_mu_k[d] = __ldg(&mu_k[offset + d]);
        }
    }

    float r_dv[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        r_dv[d] = 0.0f;
    }

    float pos_k = (float)(start_pos + n);

    for (int start_m = 0; start_m < seq_q; start_m += BWD_BLOCK_M) {
        __syncthreads();
        for (int idx = tid; idx < BWD_BLOCK_M * HEAD_DIM; idx += num_threads) {
            int m_local = idx / HEAD_DIM;
            int d = idx % HEAD_DIM;
            int m = start_m + m_local;
            if (m < seq_q) {
                int q_offset = pid_bh * seq_q * HEAD_DIM + m * HEAD_DIM + d;
                s_mu_q[m_local * HEAD_DIM + d] = __ldg(&mu_q[q_offset]);
                s_grad_out[m_local * HEAD_DIM + d] = __ldg(&grad_output[q_offset]);
            } else {
                s_mu_q[m_local * HEAD_DIM + d] = 0.0f;
                s_grad_out[m_local * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        if (n_valid) {
            for (int m_local = 0; m_local < BWD_BLOCK_M; m_local++) {
                int m = start_m + m_local;
                if (m < seq_q && m >= n) {
                    float pos_q = (float)(start_pos + m);
                    float pos_diff = pos_k - pos_q;
                    float r_lse = lse[pid_bh * seq_q + m];

                    float score = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float phase = pos_diff * s_freqs[d] + s_bias[d];
                        score += s_mu_q[m_local * HEAD_DIM + d] * r_mu_k[d] * CUDA_FAST_COSF(phase);
                    }
                    score *= scale;
                    float p = CUDA_FAST_EXPF(score - r_lse);

                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        r_dv[d] += p * s_grad_out[m_local * HEAD_DIM + d];
                    }
                }
            }
        }
    }

    if (n_valid) {
        int out_offset = pid_bh * seq_k * HEAD_DIM + n * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            dv[out_offset + d] = r_dv[d];
        }
    }
}

// =============================================================================
// Kernel Launcher Functions
// =============================================================================

#define LAUNCH_FWD_KERNEL(DIM) \
    fpope_attention_fwd_kernel<DIM><<<grid, block, smem_size>>>( \
        mu_q.data_ptr<float>(), mu_k.data_ptr<float>(), v.data_ptr<float>(), \
        freqs.data_ptr<float>(), phase_bias.data_ptr<float>(), \
        output.data_ptr<float>(), lse_tensor.data_ptr<float>(), \
        batch_heads, seq_q, seq_k, start_pos, scale);

std::vector<torch::Tensor> fpope_attention_forward_cuda(
    torch::Tensor mu_q,
    torch::Tensor mu_k,
    torch::Tensor v,
    torch::Tensor freqs,
    torch::Tensor phase_bias,
    int start_pos,
    float scale,
    bool return_lse
) {
    TORCH_CHECK(mu_q.is_cuda(), "mu_q must be a CUDA tensor");
    TORCH_CHECK(mu_q.is_contiguous(), "mu_q must be contiguous");

    int batch_heads = mu_q.size(0);
    int seq_q = mu_q.size(1);
    int dim = mu_q.size(2);
    int seq_k = mu_k.size(1);

    TORCH_CHECK(dim <= MAX_HEAD_DIM, "Head dimension exceeds maximum: ", dim, " > ", MAX_HEAD_DIM);

    auto output = torch::empty({batch_heads, seq_q, dim}, mu_q.options());
    auto lse_tensor = torch::empty({batch_heads, seq_q}, mu_q.options().dtype(torch::kFloat32));

    dim3 grid(cdiv(seq_q, BLOCK_M), batch_heads);
    dim3 block(BLOCK_M);
    int smem_size = (2 * dim + BLOCK_N * dim) * sizeof(float);

    switch (dim) {
        case 32:  LAUNCH_FWD_KERNEL(32);  break;
        case 64:  LAUNCH_FWD_KERNEL(64);  break;
        case 128: LAUNCH_FWD_KERNEL(128); break;
        default:
            TORCH_CHECK(false, "Unsupported head dimension: ", dim);
    }

    if (return_lse) {
        return {output, lse_tensor};
    }
    return {output};
}

#define LAUNCH_COMPUTE_DI_KERNEL(DIM) \
    fpope_compute_Di_kernel<DIM><<<grid_q, block, smem_size_di>>>( \
        mu_q.data_ptr<float>(), mu_k.data_ptr<float>(), v.data_ptr<float>(), \
        freqs.data_ptr<float>(), phase_bias.data_ptr<float>(), \
        grad_output.data_ptr<float>(), lse.data_ptr<float>(), \
        Di.data_ptr<float>(), batch_heads, seq_q, seq_k, start_pos, scale);

#define LAUNCH_BWD_DQ_KERNEL(DIM) \
    fpope_attention_bwd_dq_kernel<DIM><<<grid_q, block, smem_size_q>>>( \
        mu_q.data_ptr<float>(), mu_k.data_ptr<float>(), sigmoid_q.data_ptr<float>(), \
        v.data_ptr<float>(), freqs.data_ptr<float>(), phase_bias.data_ptr<float>(), \
        grad_output.data_ptr<float>(), lse.data_ptr<float>(), Di.data_ptr<float>(), \
        dq.data_ptr<float>(), batch_heads, seq_q, seq_k, start_pos, scale);

#define LAUNCH_BWD_DK_KERNEL(DIM) \
    fpope_attention_bwd_dk_kernel<DIM><<<grid_k, block, smem_size_k>>>( \
        mu_q.data_ptr<float>(), mu_k.data_ptr<float>(), sigmoid_k.data_ptr<float>(), \
        v.data_ptr<float>(), freqs.data_ptr<float>(), phase_bias.data_ptr<float>(), \
        grad_output.data_ptr<float>(), lse.data_ptr<float>(), Di.data_ptr<float>(), \
        dk.data_ptr<float>(), dfreq_partial.data_ptr<float>(), dbias_partial.data_ptr<float>(), \
        batch_heads, seq_q, seq_k, start_pos, scale, num_k_blocks);

#define LAUNCH_BWD_DV_KERNEL(DIM) \
    fpope_attention_bwd_dv_kernel<DIM><<<grid_k, block, smem_size_v>>>( \
        mu_q.data_ptr<float>(), mu_k.data_ptr<float>(), \
        freqs.data_ptr<float>(), phase_bias.data_ptr<float>(), \
        grad_output.data_ptr<float>(), lse.data_ptr<float>(), \
        dv.data_ptr<float>(), batch_heads, seq_q, seq_k, start_pos, scale);

std::vector<torch::Tensor> fpope_attention_backward_cuda(
    torch::Tensor mu_q,
    torch::Tensor mu_k,
    torch::Tensor v,
    torch::Tensor freqs,
    torch::Tensor phase_bias,
    torch::Tensor output,
    torch::Tensor grad_output,
    torch::Tensor lse,
    torch::Tensor sigmoid_q,
    torch::Tensor sigmoid_k,
    int start_pos,
    float scale
) {
    int batch_heads = mu_q.size(0);
    int seq_q = mu_q.size(1);
    int dim = mu_q.size(2);
    int seq_k = mu_k.size(1);

    auto dq = torch::empty_like(mu_q);
    auto dk = torch::empty_like(mu_k);
    auto dv = torch::empty_like(v);

    // Precompute Di (row sums for softmax backward)
    auto Di = torch::empty({batch_heads, seq_q}, lse.options());

    int num_q_blocks = cdiv(seq_q, BWD_BLOCK_M);
    int num_k_blocks = cdiv(seq_k, BWD_BLOCK_N);
    int total_blocks = batch_heads * num_k_blocks;
    auto dfreq_partial = torch::zeros({total_blocks, dim}, freqs.options());
    auto dbias_partial = torch::zeros({total_blocks, dim}, phase_bias.options());

    dim3 grid_q(num_q_blocks, batch_heads);
    dim3 grid_k(num_k_blocks, batch_heads);
    dim3 block(BWD_BLOCK_M);

    // Shared memory sizes for backward kernels
    // Di kernel: freqs + bias + mu_k_tile + v_tile
    int smem_size_di = (2 * dim + 2 * BWD_BLOCK_N * dim) * sizeof(float);
    // dQ kernel: freqs + bias + mu_k_tile + v_tile
    int smem_size_q = (2 * dim + 2 * BWD_BLOCK_N * dim) * sizeof(float);
    // dK kernel: freqs + bias + mu_q_tile + grad_out_tile + Di_tile
    int smem_size_k = (2 * dim + 2 * BWD_BLOCK_M * dim + BWD_BLOCK_M) * sizeof(float);
    // dV kernel: freqs + bias + mu_q_tile + grad_out_tile
    int smem_size_v = (2 * dim + 2 * BWD_BLOCK_M * dim) * sizeof(float);

    // First compute Di
    switch (dim) {
        case 32:  LAUNCH_COMPUTE_DI_KERNEL(32);  break;
        case 64:  LAUNCH_COMPUTE_DI_KERNEL(64);  break;
        case 128: LAUNCH_COMPUTE_DI_KERNEL(128); break;
        default:
            TORCH_CHECK(false, "Unsupported head dimension: ", dim);
    }

    // Then compute gradients
    switch (dim) {
        case 32:
            LAUNCH_BWD_DQ_KERNEL(32);
            LAUNCH_BWD_DK_KERNEL(32);
            LAUNCH_BWD_DV_KERNEL(32);
            break;
        case 64:
            LAUNCH_BWD_DQ_KERNEL(64);
            LAUNCH_BWD_DK_KERNEL(64);
            LAUNCH_BWD_DV_KERNEL(64);
            break;
        case 128:
            LAUNCH_BWD_DQ_KERNEL(128);
            LAUNCH_BWD_DK_KERNEL(128);
            LAUNCH_BWD_DV_KERNEL(128);
            break;
        default:
            TORCH_CHECK(false, "Unsupported head dimension: ", dim);
    }

    auto dfreq = dfreq_partial.sum(0);
    auto dbias = dbias_partial.sum(0);

    return {dq, dk, dv, dfreq, dbias};
}
