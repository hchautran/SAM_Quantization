// Optimized W8A8 Matrix Multiplication CUDA Kernel
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <mma.h>
#include <vector>
#include <cuda_fp16.h>

// Check if GPU supports Tensor Cores
#if __CUDA_ARCH__ >= 700
#define TENSOR_CORES_AVAILABLE
#endif

// Optimized tile sizes for different memory hierarchies
#define TILE_SIZE_M 32
#define TILE_SIZE_N 32
#define TILE_SIZE_K 32

// Error checking macro
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized W8A8 MatMul kernel with better memory access patterns
__global__ void w8a8_matmul_optimized_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    const int M, const int N, const int K,
    const float scale_A,
    const float scale_B,
    const float scale_C
) {
    __shared__ int8_t tile_A[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ int8_t tile_B[TILE_SIZE_K][TILE_SIZE_N];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE_M + ty;
    int col = bx * TILE_SIZE_N + tx;
    
    int32_t sum = 0;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE_K - 1) / TILE_SIZE_K; ++t) {
        // Load tile from A (coalesced access)
        int a_row = row;
        int a_col = t * TILE_SIZE_K + tx;
        if (a_row < M && a_col < K) {
            tile_A[ty][tx] = A[a_row * K + a_col];
        } else {
            tile_A[ty][tx] = 0;
        }
        
        // Load tile from B (coalesced access)
        int b_row = t * TILE_SIZE_K + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_B[ty][tx] = B[b_row * N + b_col];
        } else {
            tile_B[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE_K; ++k) {
            sum += (int32_t)tile_A[ty][k] * (int32_t)tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result with scaling
    if (row < M && col < N) {
        float scaled_result = (scale_A * scale_B / scale_C) * (float)sum;
        C[row * N + col] = (int32_t)roundf(scaled_result);
    }
}

// High-performance Tensor Core kernel for W8A8
__global__ void w8a8_matmul_tensor_core_optimized_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    const int M, const int N, const int K,
    const float scale_A,
    const float scale_B,
    const float scale_C
) {
    using namespace nvcuda::wmma;
    
    // Declare fragments
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0);
    
    // Calculate warp position
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 2;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 2;
    
    // Bounds check
    if (warpM * 16 >= M || warpN * 16 >= N) return;
    
    // Main computation loop
    for (int k = 0; k < K; k += 16) {
        if (k >= K) break;
        
        // Load matrices into fragments
        load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warpN * 16, N);
        
        // Perform matrix multiplication
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Apply scaling and store result
    for (int i = 0; i < c_frag.num_elements; i++) {
        float scaled_result = (scale_A * scale_B / scale_C) * (float)c_frag.x[i];
        c_frag.x[i] = (int32_t)roundf(scaled_result);
    }
    
    store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N, mem_row_major);
}

// Kernel for per-channel quantization support
__global__ void w8a8_matmul_per_channel_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    const float* __restrict__ weight_scales,
    const int M, const int N, const int K,
    const float scale_A,
    const float scale_C
) {
    __shared__ int8_t tile_A[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ int8_t tile_B[TILE_SIZE_K][TILE_SIZE_N];
    __shared__ float tile_scales[TILE_SIZE_N];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE_M + ty;
    int col = bx * TILE_SIZE_N + tx;
    
    int32_t sum = 0;
    
    for (int t = 0; t < (K + TILE_SIZE_K - 1) / TILE_SIZE_K; ++t) {
        // Load tiles
        int a_row = row;
        int a_col = t * TILE_SIZE_K + tx;
        if (a_row < M && a_col < K) {
            tile_A[ty][tx] = A[a_row * K + a_col];
        } else {
            tile_A[ty][tx] = 0;
        }
        
        int b_row = t * TILE_SIZE_K + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_B[ty][tx] = B[b_row * N + b_col];
        } else {
            tile_B[ty][tx] = 0;
        }
        
        // Load weight scales for this tile
        if (tx < TILE_SIZE_N && col < N) {
            tile_scales[tx] = weight_scales[col];
        }
        
        __syncthreads();
        
        // Compute with per-channel scaling
        for (int k = 0; k < TILE_SIZE_K; ++k) {
            sum += (int32_t)tile_A[ty][k] * (int32_t)tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        float weight_scale = weight_scales[col];
        float scaled_result = (scale_A * weight_scale / scale_C) * (float)sum;
        C[row * N + col] = (int32_t)roundf(scaled_result);
    }
}

// Main CUDA implementation with multiple optimization strategies
torch::Tensor w8a8_matmul_cuda_optimized(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor weight_scales,  // Optional per-channel scales
    float scale_A,
    float scale_B,
    float scale_C,
    bool use_tensor_cores,
    bool per_channel_quantization
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(A.device());
    torch::Tensor C = torch::zeros({M, N}, options);
    
    // Get raw pointers
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const int8_t* B_ptr = B.data_ptr<int8_t>();
    int32_t* C_ptr = C.data_ptr<int32_t>();
    
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool tensor_cores_used = false;
    
    // Check device compute capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    bool tensor_cores_supported = props.major >= 7;
    
    if (per_channel_quantization && weight_scales.defined()) {
        // Use per-channel quantization kernel
        const float* scales_ptr = weight_scales.data_ptr<float>();
        
        dim3 blockSize(TILE_SIZE_N, TILE_SIZE_M);
        dim3 gridSize((N + TILE_SIZE_N - 1) / TILE_SIZE_N, (M + TILE_SIZE_M - 1) / TILE_SIZE_M);
        
        w8a8_matmul_per_channel_kernel<<<gridSize, blockSize, 0, stream>>>(
            A_ptr, B_ptr, C_ptr, scales_ptr, M, N, K, scale_A, scale_C
        );
    }
#ifdef TENSOR_CORES_AVAILABLE
    else if (use_tensor_cores && tensor_cores_supported && 
             M % 16 == 0 && N % 16 == 0 && K % 16 == 0) {
        // Use Tensor Core kernel
        dim3 blockSize(32, 4);
        dim3 gridSize((N + 15) / 16, (M + 15) / 16);
        
        w8a8_matmul_tensor_core_optimized_kernel<<<gridSize, blockSize, 0, stream>>>(
            A_ptr, B_ptr, C_ptr, M, N, K, scale_A, scale_B, scale_C
        );
        tensor_cores_used = true;
    }
#endif
    else {
        // Use optimized regular kernel
        dim3 blockSize(TILE_SIZE_N, TILE_SIZE_M);
        dim3 gridSize((N + TILE_SIZE_N - 1) / TILE_SIZE_N, (M + TILE_SIZE_M - 1) / TILE_SIZE_M);
        
        w8a8_matmul_optimized_kernel<<<gridSize, blockSize, 0, stream>>>(
            A_ptr, B_ptr, C_ptr, M, N, K, scale_A, scale_B, scale_C
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

// Additional utility kernel for quantization
__global__ void quantize_tensor_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    const int size,
    const float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float scaled = input[idx] / scale;
        output[idx] = (int8_t)max(-128, min(127, (int)roundf(scaled)));
    }
}

// Quantization utility function
torch::Tensor quantize_tensor_cuda(torch::Tensor input, float scale) {
    CHECK_INPUT(input);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kInt8)
        .device(input.device());
    torch::Tensor output = torch::zeros(input.sizes(), options);
    
    const float* input_ptr = input.data_ptr<float>();
    int8_t* output_ptr = output.data_ptr<int8_t>();
    
    int size = input.numel();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    quantize_tensor_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_ptr, output_ptr, size, scale
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Quantization kernel failed: ", cudaGetErrorString(err));
    
    return output;
} 