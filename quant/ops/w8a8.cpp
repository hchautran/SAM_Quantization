// w8a8_pytorch.cpp
#include <torch/extension.h>

// Forward declarations for CUDA functions
torch::Tensor w8a8_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    float scale_A,
    float scale_B,
    float scale_C,
    bool use_tensor_cores
);

// Check tensor properties and call CUDA implementation
torch::Tensor w8a8_matmul(
    torch::Tensor A,        // [M, K] int8 tensor
    torch::Tensor B,        // [K, N] int8 tensor  
    float scale_A = 1.0f,
    float scale_B = 1.0f,
    float scale_C = 1.0f,
    bool use_tensor_cores = true
) {
    // Input validation
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    
    // Ensure tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();
    
    return w8a8_matmul_cuda(A, B, scale_A, scale_B, scale_C, use_tensor_cores);
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w8a8_matmul", &w8a8_matmul, "W8A8 Matrix Multiplication",
          py::arg("A"), py::arg("B"), 
          py::arg("scale_A") = 1.0f,
          py::arg("scale_B") = 1.0f,
          py::arg("scale_C") = 1.0f,
          py::arg("use_tensor_cores") = true);
}