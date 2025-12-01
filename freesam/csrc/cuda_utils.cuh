
#pragma once

#include <stdlib.h>
#include <string.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "stdio.h"


#define CEIL_DIV(M,N) (((M) + (N) - 1) / (N))


inline void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

inline void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }
