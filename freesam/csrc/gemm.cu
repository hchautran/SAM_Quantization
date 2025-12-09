#include <cstdint>
#include <torch/extension.h>
#include "cuda_utils.cuh"

#define TILE_SIZE 16

__global__ void gemm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t M,
    int64_t N,
    int64_t K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
                sum += a[row * K + i] * b[i * N + col];
        }
        out[row * N + col] = sum;
    }
}


torch::Tensor naive_gemm(torch::Tensor a, torch::Tensor b) {
    int64_t M= a.size(0);
    int64_t N= b.size(1);
    int64_t K= a.size(1);
    
    auto out = torch::empty({M, N}, a.options());

    dim3 n_blocks(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
    dim3 n_threads(TILE_SIZE, TILE_SIZE);

    gemm_kernel<<<n_blocks, n_threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        N,
        K
    );
    return out;
}
