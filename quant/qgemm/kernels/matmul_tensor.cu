#include <iostream>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// ---------------------- CUDA Core Kernel ----------------------
__global__ void matmul_cuda_cores(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// ---------------------- Tensor Core Kernel ----------------------
__global__ void matmul_tensor_cores(half *A, half *B, float *C, int N) {
    int lda = N, ldb = N, ldc = N;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    for (int k = 0; k < N; k += 16) {
        int aRow = warpM * 16;
        int bCol = warpN * 16;

        if (aRow < N && bCol < N) {
            wmma::load_matrix_sync(aFrag, A + aRow * lda + k, lda);
            wmma::load_matrix_sync(bFrag, B + k * ldb + bCol, ldb);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    int cRow = warpM * 16;
    int cCol = warpN * 16;
    if (cRow < N && cCol < N) {
        wmma::store_matrix_sync(C + cRow * ldc + cCol, cFrag, ldc, wmma::mem_row_major);
    }
}

// ---------------------- Main ----------------------
int main() {
    int N = 256; // must be multiple of 16 for WMMA
    size_t sizeF = N * N * sizeof(float);
    size_t sizeH = N * N * sizeof(half);

    // Host memory
    float *h_Af = new float[N * N];
    float *h_Bf = new float[N * N];
    float *h_Cf = new float[N * N];
    half  *h_Ah = new half[N * N];
    half  *h_Bh = new half[N * N];

    // Initialize
    for (int i = 0; i < N * N; i++) {
        h_Af[i] = 1.0f;
        h_Bf[i] = 1.0f;
        h_Ah[i] = __float2half(1.0f);
        h_Bh[i] = __float2half(1.0f);
    }

    // Device memory
    float *d_Af, *d_Bf, *d_Cf;
    half  *d_Ah, *d_Bh;
    float *d_Ch;
    cudaMalloc(&d_Af, sizeF);
    cudaMalloc(&d_Bf, sizeF);
    cudaMalloc(&d_Cf, sizeF);
    cudaMalloc(&d_Ah, sizeH);
    cudaMalloc(&d_Bh, sizeH);
    cudaMalloc(&d_Ch, sizeF);

    cudaMemcpy(d_Af, h_Af, sizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bf, h_Bf, sizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ah, h_Ah, sizeH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bh, h_Bh, sizeH, cudaMemcpyHostToDevice);

    // ---------------- CUDA Core version ----------------
    dim3 threads1(16, 16);
    dim3 blocks1((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_cuda_cores<<<blocks1, threads1>>>(d_Af, d_Bf, d_Cf, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "CUDA Core GEMM time: " << ms << " ms" << std::endl;

    // ---------------- Tensor Core version ----------------
    dim3 threads2(32, 4);
    dim3 blocks2(N / 16, N / 16);

    cudaEventRecord(start);
    matmul_tensor_cores<<<blocks2, threads2>>>(d_Ah, d_Bh, d_Ch, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Tensor Core GEMM time: " << ms << " ms" << std::endl;

    // Cleanup
    delete[] h_Af; delete[] h_Bf; delete[] h_Cf;
    delete[] h_Ah; delete[] h_Bh;
    cudaFree(d_Af); cudaFree(d_Bf); cudaFree(d_Cf);
    cudaFree(d_Ah); cudaFree(d_Bh); cudaFree(d_Ch);

    return 0;
}
