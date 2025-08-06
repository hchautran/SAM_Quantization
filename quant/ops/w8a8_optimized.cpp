// Optimized W8A8 Matrix Multiplication C++ Wrapper
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Forward declarations for CUDA functions
torch::Tensor w8a8_matmul_cuda_optimized(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor weight_scales,
    float scale_A,
    float scale_B,
    float scale_C,
    bool use_tensor_cores,
    bool per_channel_quantization
);

torch::Tensor quantize_tensor_cuda(torch::Tensor input, float scale);

// Main optimized W8A8 matmul function
torch::Tensor w8a8_matmul_optimized(
    torch::Tensor A,        // [M, K] int8 tensor
    torch::Tensor B,        // [K, N] int8 tensor  
    torch::Tensor weight_scales = torch::Tensor(),  // Optional per-channel scales
    float scale_A = 1.0f,
    float scale_B = 1.0f,
    float scale_C = 1.0f,
    bool use_tensor_cores = true,
    bool per_channel_quantization = false
) {
    // Input validation
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for multiplication");
    
    // Validate weight scales if using per-channel quantization
    if (per_channel_quantization) {
        TORCH_CHECK(weight_scales.defined(), "Weight scales must be provided for per-channel quantization");
        TORCH_CHECK(weight_scales.device().is_cuda(), "Weight scales must be a CUDA tensor");
        TORCH_CHECK(weight_scales.dtype() == torch::kFloat32, "Weight scales must be float32");
        TORCH_CHECK(weight_scales.dim() == 1, "Weight scales must be 1D");
        TORCH_CHECK(weight_scales.size(0) == B.size(1), "Weight scales size must match output dimension");
    }
    
    // Ensure tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();
    if (weight_scales.defined()) {
        weight_scales = weight_scales.contiguous();
    }
    
    return w8a8_matmul_cuda_optimized(A, B, weight_scales, scale_A, scale_B, scale_C, 
                                     use_tensor_cores, per_channel_quantization);
}

// Quantization utility function
torch::Tensor quantize_tensor_optimized(torch::Tensor input, float scale) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(scale > 0, "Scale must be positive");
    
    input = input.contiguous();
    return quantize_tensor_cuda(input, scale);
}

// Batch matrix multiplication for multiple inputs
torch::Tensor w8a8_batch_matmul_optimized(
    torch::Tensor A,        // [batch_size, M, K] int8 tensor
    torch::Tensor B,        // [batch_size, K, N] int8 tensor
    torch::Tensor weight_scales = torch::Tensor(),
    float scale_A = 1.0f,
    float scale_B = 1.0f,
    float scale_C = 1.0f,
    bool use_tensor_cores = true,
    bool per_channel_quantization = false
) {
    // Input validation
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 3, "A must be 3D [batch_size, M, K]");
    TORCH_CHECK(B.dim() == 3, "B must be 3D [batch_size, K, N]");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Matrix dimensions must match for multiplication");
    
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(A.device());
    torch::Tensor C = torch::zeros({batch_size, M, N}, options);
    
    // Process each batch element
    for (int b = 0; b < batch_size; ++b) {
        auto A_batch = A[b];  // [M, K]
        auto B_batch = B[b];  // [K, N]
        auto C_batch = C[b];  // [M, N]
        
        C_batch = w8a8_matmul_optimized(A_batch, B_batch, weight_scales, 
                                       scale_A, scale_B, scale_C, 
                                       use_tensor_cores, per_channel_quantization);
    }
    
    return C;
}

// Utility function to compute optimal scales for quantization
std::pair<torch::Tensor, torch::Tensor> compute_quantization_scales(
    torch::Tensor input,
    std::string method = "absmax",
    int bits = 8
) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bits > 0 && bits <= 8, "Bits must be between 1 and 8");
    
    float max_val = (1 << (bits - 1)) - 1;  // e.g., 127 for 8-bit
    
    torch::Tensor scales;
    torch::Tensor zeros;
    
    if (method == "absmax") {
        // Per-tensor absmax quantization
        float abs_max = input.abs().max().item<float>();
        scales = torch::tensor({abs_max / max_val}, 
                              torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
        zeros = torch::zeros_like(scales);
    } else if (method == "per_channel") {
        // Per-channel absmax quantization
        scales = input.abs().amax(0) / max_val;  // [K]
        zeros = torch::zeros_like(scales);
    } else {
        TORCH_CHECK(false, "Unsupported quantization method: ", method);
    }
    
    return {scales, zeros};
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w8a8_matmul_optimized", &w8a8_matmul_optimized, "Optimized W8A8 Matrix Multiplication",
          py::arg("A"), py::arg("B"), 
          py::arg("weight_scales") = torch::Tensor(),
          py::arg("scale_A") = 1.0f,
          py::arg("scale_B") = 1.0f,
          py::arg("scale_C") = 1.0f,
          py::arg("use_tensor_cores") = true,
          py::arg("per_channel_quantization") = false);
    
    m.def("quantize_tensor_optimized", &quantize_tensor_optimized, "Quantize tensor to int8",
          py::arg("input"), py::arg("scale"));
    
    m.def("w8a8_batch_matmul_optimized", &w8a8_batch_matmul_optimized, "Optimized W8A8 Batch Matrix Multiplication",
          py::arg("A"), py::arg("B"),
          py::arg("weight_scales") = torch::Tensor(),
          py::arg("scale_A") = 1.0f,
          py::arg("scale_B") = 1.0f,
          py::arg("scale_C") = 1.0f,
          py::arg("use_tensor_cores") = true,
          py::arg("per_channel_quantization") = false);
    
    m.def("compute_quantization_scales", &compute_quantization_scales, "Compute quantization scales",
          py::arg("input"), py::arg("method") = "absmax", py::arg("bits") = 8);
} 