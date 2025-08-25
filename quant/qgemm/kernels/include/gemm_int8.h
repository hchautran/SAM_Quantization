#pragma once

#include <cstdint>

// INT8 Tensor Core GEMM using CUTLASS
// Computes: C[M, N] = A[M, K] @ (B[N, K])^T
// A: row-major int8, lda = K
// B: provided as row-major [N, K] but interpreted as column-major [K, N], ldb = K
// C: row-major int32, ldc = N

void matmul_i8_host(
        const int8_t *A,
        const int8_t *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);


