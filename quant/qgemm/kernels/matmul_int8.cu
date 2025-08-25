#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(err)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t st=(x); if(st!=CUBLAS_STATUS_SUCCESS){ \
  std::cerr<<"cuBLAS error: "<<st<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} } while(0)

static float max_abs(const float* p, int n){
  float m=0.f; for(int i=0;i<n;++i) m = std::max(m, std::fabs(p[i])); return m;
}
static void quant_per_tensor_s8(const float* src, int8_t* dst, int n, float& scale){
  float m=max_abs(src,n);
  scale = (m>0.f) ? (m/127.f) : 1.f;
  float inv = 1.f/scale;
  for(int i=0;i<n;++i){
    int q = (int)lrintf(std::max(-127.f, std::min(127.f, src[i]*inv)));
    dst[i] = (int8_t)q;
  }
}

__global__ void dequant_int32_to_float(const int32_t* __restrict__ X, float* __restrict__ Y, int n, float scale){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) Y[i] = scale * static_cast<float>(X[i]);
}

__global__ void gemm_float_ref(const float* A, const float* B, float* C, int M, int N, int K){
  int m = blockIdx.y*blockDim.y + threadIdx.y;
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  if(m>=M || n>=N) return;
  float acc=0.f;
  for(int k=0;k<K;++k) acc += A[m*K+k]*B[k*N+n];
  C[m*N+n] = acc;
}

int main(){
  const int M = 1024;   // multiple of 16
  const int N = 1024;   // multiple of 16
  const int K = 1024;   // multiple of 32 for INT8 TC

  std::vector<float>  hA(M*K), hB(K*N), hC_ref(M*N), hC(M*N);
  std::vector<int8_t> hAq(M*K), hBq(K*N);
  float sA, sB;

  // Init some values
  for(int i=0;i<M*K;++i) hA[i] = std::sin(0.001f*i)*2.3f;
  for(int i=0;i<K*N;++i) hB[i] = std::cos(0.0012f*i)*1.8f;

  // Quantize per-tensor
  quant_per_tensor_s8(hA.data(), hAq.data(), M*K, sA);
  quant_per_tensor_s8(hB.data(), hBq.data(), K*N, sB);
  const float out_scale = sA * sB; // dequant scale

  // ---- Device buffers ----
  int8_t  *dAq=nullptr, *dBq=nullptr;
  int32_t *dCacc=nullptr;
  float   *dC=nullptr, *dA=nullptr, *dB=nullptr, *dCref=nullptr;

  CUDA_CHECK(cudaMalloc(&dAq, sizeof(int8_t)*M*K));
  CUDA_CHECK(cudaMalloc(&dBq, sizeof(int8_t)*K*N));
  CUDA_CHECK(cudaMalloc(&dCacc, sizeof(int32_t)*M*N));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(float)*M*N));
  // optional ref
  CUDA_CHECK(cudaMalloc(&dA, sizeof(float)*M*K));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(float)*K*N));
  CUDA_CHECK(cudaMalloc(&dCref, sizeof(float)*M*N));

  CUDA_CHECK(cudaMemcpy(dAq, hAq.data(), sizeof(int8_t)*M*K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBq, hBq.data(), sizeof(int8_t)*K*N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dCacc, 0, sizeof(int32_t)*M*N));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dCref, 0, sizeof(float)*M*N));

  // ---- cuBLASLt setup ----
  cublasLtHandle_t lt;
  CUBLAS_CHECK(cublasLtCreate(&lt));

  // Matmul descriptor: INT8 inputs, INT32 accumulate
  cublasLtMatmulDesc_t matmulDesc;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;  // int32 accumulation on TC
  cudaDataType_t scaleType = CUDA_R_32I;                 // alpha/beta type matched to compute
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));


  // Layouts (ROW-MAJOR for all)
  cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, M, K, K)); // ld = stride between rows in row-major
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, N, N));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, M, N, N));

  // Set row-major order explicitly
  cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));


  // ---- Heuristic algorithm selection ----
  cublasLtMatmulPreference_t pref;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t maxWorkspace = 1<<26; // 64MB workspace
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspace, sizeof(maxWorkspace)));

  cublasLtMatmulHeuristicResult_t heuristic;
  int returnedResults = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, matmulDesc, layoutA, layoutB, layoutC, layoutC, pref, 1, &heuristic, &returnedResults));
  if (returnedResults == 0) {
    std::cerr << "No cuBLASLt heuristic found for this problem.\n";
    return 1;
  }

  // ---- Run & time ----
  cudaEvent_t e0,e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  int32_t alpha = 1, beta = 0;

  CUDA_CHECK(cudaEventRecord(e0));
  CUBLAS_CHECK(cublasLtMatmul(
      lt, matmulDesc,
      &alpha,
      dAq, layoutA,
      dBq, layoutB,
      &beta,
      dCacc, layoutC,   // C (beta*C) input
      dCacc, layoutC,   // D output (int32 accumulators)
      &heuristic.algo,
      nullptr, 0, nullptr));
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));

  // ---- Dequantize to FP32 for inspection/metrics ----
  int total = M*N;
  int threads = 256;
  int blocks  = (total + threads - 1)/threads;
  dequant_int32_to_float<<<blocks,threads>>>(dCacc, dC, total, out_scale);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

  // ---- (Optional) FP32 reference for error stats ----
  dim3 t(16,16), b((N+15)/16,(M+15)/16);
  gemm_float_ref<<<b,t>>>(dA, dB, dCref, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC_ref.data(), dCref, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

  double mae=0.0, mre=0.0;
  for(int i=0;i<total;++i){
    double a=hC[i], r=hC_ref[i];
    mae += std::fabs(a-r);
    mre = std::max(mre, std::fabs(a-r)/(std::fabs(r)+1e-6));
  }
  mae /= total;

  // GEMM FLOPs = 2*M*N*K
  double gflops = (2.0*(double)M*N*K) / (ms/1000.0) / 1e9;

  std::cout<<"cuBLASLt INT8 GEMM time: "<<ms<<" ms  -> "<<gflops<<" GFLOP/s\n";
  std::cout<<"Mean Abs Error vs FP32: "<<mae<<"   Max Rel Error: "<<mre<<"\n";
  std::cout<<"Example C[0,0]: "<<hC[0]<<"\n";

  // ---- Cleanup ----
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatrixLayoutDestroy(layoutB);
  cublasLtMatrixLayoutDestroy(layoutC);
  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtDestroy(lt);
  cudaFree(dAq); cudaFree(dBq); cudaFree(dCacc); cudaFree(dC);
  cudaFree(dA); cudaFree(dB); cudaFree(dCref);
  return 0;
}
