#include <torch/extension.h>

// Forward declarations - GEMM
torch::Tensor naive_gemm(torch::Tensor a, torch::Tensor b);
torch::Tensor shared_gemm(torch::Tensor a, torch::Tensor b);
torch::Tensor cutlass_gemm(torch::Tensor a, torch::Tensor b);

// Forward declarations - Flash Attention 2
torch::Tensor flash_attn(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale
);

torch::Tensor flash_attn_rel(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor rel_pos_h,
    torch::Tensor rel_pos_w,
    float softmax_scale
);


// Dispatch (CPU vs CUDA)
torch::Tensor gemm(torch::Tensor a, torch::Tensor b, int version) {
    switch (version) {
        case 0:
            return naive_gemm(a, b);
        case 1:
            return shared_gemm(a, b);
        case 2:
            return cutlass_gemm(a, b);
        default:
            throw std::invalid_argument("Unsupported version");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "gemm");
    // Flash Attention 

    m.def("flash_attn", &flash_attn,
          "Flash Attention 2 forward pass (support only 64 head dim)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("softmax_scale") = 0.0f);

    
    m.def("flash_attn_rel", &flash_attn_rel,
          "Flash Attention forward pass (supports any head_dim)",
          py::arg("Q"), 
          py::arg("K"), 
          py::arg("V"), 
          py::arg("rel_pos_h"),  
          py::arg("rel_pos_w"),
          py::arg("softmax_scale") = 0.0f
    );
}
