// w8a8_pytorch.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <mma.h>
#include <vector>

// Check if GPU supports Tensor Cores
#if __CUDA_ARCH__ >= 700
#define TENSOR_CORES_AVAILABLE
#endif

#define TILE_SIZE 16

// Error checking macro
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Basic W8A8 MatMul kernel
__global__ void w8a8_matmul_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    const int M, const int N, const int K,
    const float scale_A,
    const float scale_B,
    const float scale_C
) {
    __shared__ int8_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    int32_t sum = 0;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        if (a_row < M && a_col < K) {
            tile_A[ty][tx] = A[a_row * K + a_col];
        } else {
            tile_A[ty][tx] = 0;
        }
        
        // Load tile from B
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_B[ty][tx] = B[b_row * N + b_col];
        } else {
            tile_B[ty][tx] = 0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += (int32_t)tile_A[ty][k] * (int32_t)tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        float scaled_result = (scale_A * scale_B / scale_C) * (float)sum;
        C[row * N + col] = (int32_t)roundf(scaled_result);
    }
}

// Tensor Core optimized kernel
__global__ void w8a8_matmul_tensor_core_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    const int M, const int N, const int K,
    const float scale_A,
    const float scale_B,
    const float scale_C
) {
    using namespace nvcuda::wmma;
    
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> c_frag;
    
    fill_fragment(c_frag, 0);
    
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 2;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 2;
    
    if (warpM * 16 >= M || warpN * 16 >= N) return;
    
    for (int k = 0; k < K; k += 16) {
        if (k >= K) break;
        
        load_matrix_sync(a_frag, A + warpM * 16 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warpN * 16, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Apply scaling
    for (int i = 0; i < c_frag.num_elements; i++) {
        float scaled_result = (scale_A * scale_B / scale_C) * (float)c_frag.x[i];
        c_frag.x[i] = (int32_t)roundf(scaled_result);
    }
    
    store_matrix_sync(C + warpM * 16 * N + warpN * 16, c_frag, N, mem_row_major);
}

// CUDA implementation called from C++
torch::Tensor w8a8_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    float scale_A,
    float scale_B,
    float scale_C,
    bool use_tensor_cores
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
    
#ifdef TENSOR_CORES_AVAILABLE
    // Check device compute capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    bool tensor_cores_supported = props.major >= 7;
    
    if (use_tensor_cores && tensor_cores_supported && 
        M % 16 == 0 && N % 16 == 0 && K % 16 == 0) {
        
        dim3 blockSize(32, 4);
        dim3 gridSize((N + 15) / 16, (M + 15) / 16);
        
        w8a8_matmul_tensor_core_kernel<<<gridSize, blockSize, 0, stream>>>(
            A_ptr, B_ptr, C_ptr, M, N, K, scale_A, scale_B, scale_C
        );
        tensor_cores_used = true;
    }
#endif
    
    if (!tensor_cores_used) {
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        w8a8_matmul_kernel<<<gridSize, blockSize, 0, stream>>>(
            A_ptr, B_ptr, C_ptr, M, N, K, scale_A, scale_B, scale_C
        );
    }
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}