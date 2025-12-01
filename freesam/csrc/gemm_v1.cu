#include <iostream>
#include "cuda_utils.cuh"
#include <torch/extension.h>


#define TILE_SIZE 16
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    float sum = 0.0f;
    for (int tile = 0; tile < CEIL_DIV(K, TILE_SIZE);  tile++) {
        if (row < M && tile*TILE_SIZE+ tx < K) {
            shared_A[ty][tx] = A[row*K + tile*TILE_SIZE + tx];
        }
        else {
            shared_A[ty][tx] = 0.0f; 
        }
        
        if (tile*TILE_SIZE + ty < K && col < N) {
            shared_B[ty][tx] = B[(tile*TILE_SIZE + ty)*N + col];
        }
        else {
            shared_B[ty][tx] = 0.0f; 
        }
        __syncthreads();
        
        for (int k=0; k < TILE_SIZE; k++) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        __syncthreads();
    }
    if (row<M && col<N) {
        C[row*N + col] = sum;
    }
}


torch::Tensor shared_gemm(torch::Tensor a, torch::Tensor b) {
    int64_t M = a.size(0);
    int64_t N = b.size(1);
    int64_t K = a.size(1);
    
    auto out = torch::empty({M, N}, a.options());
    dim3 n_blocks(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
    dim3 n_threads(TILE_SIZE, TILE_SIZE);
    
    gemm_kernel<<<n_blocks, n_threads>>> (
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    return out;
}