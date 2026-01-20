// FPoPE CUDA Extension - PyTorch C++ Bindings
// Exposes native CUDA kernels to Python

#include <torch/extension.h>
#include <vector>

// Forward declarations from fpope_cuda_kernel.cu
std::vector<torch::Tensor> fpope_attention_forward_cuda(
    torch::Tensor mu_q,
    torch::Tensor mu_k,
    torch::Tensor v,
    torch::Tensor freqs,
    torch::Tensor phase_bias,
    int start_pos,
    float scale,
    bool return_lse
);

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
);

// Python-facing wrapper for forward pass
// Handles input validation and tensor format conversion
std::vector<torch::Tensor> fpope_forward(
    torch::Tensor mu_q,
    torch::Tensor mu_k,
    torch::Tensor v,
    torch::Tensor freqs,
    torch::Tensor phase_bias,
    int64_t start_pos,
    double scale,
    bool return_lse
) {
    // Input validation
    TORCH_CHECK(mu_q.device().is_cuda(), "mu_q must be on CUDA");
    TORCH_CHECK(mu_k.device().is_cuda(), "mu_k must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "v must be on CUDA");
    TORCH_CHECK(freqs.device().is_cuda(), "freqs must be on CUDA");
    TORCH_CHECK(phase_bias.device().is_cuda(), "phase_bias must be on CUDA");

    TORCH_CHECK(mu_q.dim() == 3, "mu_q must be 3D (batch*heads, seq_q, dim)");
    TORCH_CHECK(mu_k.dim() == 3, "mu_k must be 3D (batch*heads, seq_k, dim)");
    TORCH_CHECK(v.dim() == 3, "v must be 3D (batch*heads, seq_k, dim)");
    TORCH_CHECK(freqs.dim() == 1, "freqs must be 1D (dim,)");
    TORCH_CHECK(phase_bias.dim() == 1, "phase_bias must be 1D (dim,)");

    // Ensure contiguous
    auto mu_q_c = mu_q.contiguous();
    auto mu_k_c = mu_k.contiguous();
    auto v_c = v.contiguous();
    auto freqs_c = freqs.contiguous();
    auto phase_bias_c = phase_bias.contiguous();

    // Convert to float32 for kernel (can extend to fp16 later)
    if (mu_q_c.scalar_type() != torch::kFloat32) {
        mu_q_c = mu_q_c.to(torch::kFloat32);
    }
    if (mu_k_c.scalar_type() != torch::kFloat32) {
        mu_k_c = mu_k_c.to(torch::kFloat32);
    }
    if (v_c.scalar_type() != torch::kFloat32) {
        v_c = v_c.to(torch::kFloat32);
    }
    if (freqs_c.scalar_type() != torch::kFloat32) {
        freqs_c = freqs_c.to(torch::kFloat32);
    }
    if (phase_bias_c.scalar_type() != torch::kFloat32) {
        phase_bias_c = phase_bias_c.to(torch::kFloat32);
    }

    return fpope_attention_forward_cuda(
        mu_q_c, mu_k_c, v_c, freqs_c, phase_bias_c,
        static_cast<int>(start_pos),
        static_cast<float>(scale),
        return_lse
    );
}

// Python-facing wrapper for backward pass
std::vector<torch::Tensor> fpope_backward(
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
    int64_t start_pos,
    double scale
) {
    // Input validation
    TORCH_CHECK(mu_q.device().is_cuda(), "mu_q must be on CUDA");
    TORCH_CHECK(lse.device().is_cuda(), "lse must be on CUDA");
    TORCH_CHECK(sigmoid_q.device().is_cuda(), "sigmoid_q must be on CUDA");
    TORCH_CHECK(sigmoid_k.device().is_cuda(), "sigmoid_k must be on CUDA");

    // Ensure contiguous and float32
    auto mu_q_c = mu_q.contiguous().to(torch::kFloat32);
    auto mu_k_c = mu_k.contiguous().to(torch::kFloat32);
    auto v_c = v.contiguous().to(torch::kFloat32);
    auto freqs_c = freqs.contiguous().to(torch::kFloat32);
    auto phase_bias_c = phase_bias.contiguous().to(torch::kFloat32);
    auto output_c = output.contiguous().to(torch::kFloat32);
    auto grad_output_c = grad_output.contiguous().to(torch::kFloat32);
    auto lse_c = lse.contiguous().to(torch::kFloat32);
    auto sigmoid_q_c = sigmoid_q.contiguous().to(torch::kFloat32);
    auto sigmoid_k_c = sigmoid_k.contiguous().to(torch::kFloat32);

    return fpope_attention_backward_cuda(
        mu_q_c, mu_k_c, v_c, freqs_c, phase_bias_c,
        output_c, grad_output_c, lse_c,
        sigmoid_q_c, sigmoid_k_c,
        static_cast<int>(start_pos),
        static_cast<float>(scale)
    );
}

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FPoPE (FoPE+PoPE) Native CUDA Attention Kernels";

    m.def("forward", &fpope_forward,
        "FPoPE attention forward pass (CUDA)",
        py::arg("mu_q"),
        py::arg("mu_k"),
        py::arg("v"),
        py::arg("freqs"),
        py::arg("phase_bias"),
        py::arg("start_pos") = 0,
        py::arg("scale") = 1.0,
        py::arg("return_lse") = true
    );

    m.def("backward", &fpope_backward,
        "FPoPE attention backward pass (CUDA)",
        py::arg("mu_q"),
        py::arg("mu_k"),
        py::arg("v"),
        py::arg("freqs"),
        py::arg("phase_bias"),
        py::arg("output"),
        py::arg("grad_output"),
        py::arg("lse"),
        py::arg("sigmoid_q"),
        py::arg("sigmoid_k"),
        py::arg("start_pos") = 0,
        py::arg("scale") = 1.0
    );
}
