#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"
#include "common.cuh"
#define MAX(a,b)((a)>(b)?(a):(b))

// Flash Attention 2 Implementation
// Based on the paper: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
// https://arxiv.org/abs/2307.08691

// Key improvements in Flash Attention 2:
// 1. Reduce non-matmul FLOPs by better online softmax computation
// 2. Better parallelism through better work partitioning between warps
// 3. Better work partitioning within a thread block to reduce communication and shared memory reads/writes

template<typename scalar_t, int TILE_SIZE,int HEAD_DIM>
__global__ void flash_attn_kernel(
    const scalar_t*__restrict__ Q,    
    const scalar_t*__restrict__ K,    
    const scalar_t*__restrict__ V,    
    scalar_t *__restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    extern __shared__ float mem_s[];
    int tid = threadIdx.x;
    float* q_s = mem_s;
    float* K_s = q_s + HEAD_DIM;
    float* V_s = K_s + HEAD_DIM * TILE_SIZE;
    float* qK_s = V_s + HEAD_DIM * TILE_SIZE;
    float* o_s = qK_s + TILE_SIZE;

    __shared__ float max_qk_s;
    __shared__ float sum_exp_s;
    max_qk_s = -INFINITY;
    sum_exp_s = 0.0f;

    // each block handle one row of q
    int batch_idx = blockIdx.z; // current batch of q
    int head_idx = blockIdx.y; // current head of q
    int q_idx = blockIdx.x; // current idx of q in the seq
    int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    
    // read current q into the share mem;
    for (int d = tid ; d < head_dim; d+=blockDim.x) {
        q_s[d] = static_cast<float>(Q[batch_head_offset + q_idx * head_dim + d]);
        o_s[d] = 0.0f;
    }

    for (int kv_idx = threadIdx.x; kv_idx < TILE_SIZE; kv_idx+=blockDim.x) {
        qK_s[kv_idx] = -INFINITY;
    }
    __syncthreads();

    
    for (int tile = 0; tile < CEIL_DIV(seq_len, TILE_SIZE); tile++) {
        // read current tile of kv into the share mem;
        for (int kv_idx = 0; kv_idx < TILE_SIZE; kv_idx++) {

            int kv_offset = tile * TILE_SIZE + kv_idx;
            if (kv_offset < seq_len) {
                for (int tid = threadIdx.x; tid < HEAD_DIM; tid+=blockDim.x) {
                    K_s[kv_idx * head_dim  + tid] = static_cast<float>(K[batch_head_offset + kv_offset * head_dim + tid]);
                    V_s[kv_idx * head_dim  + tid] = static_cast<float>(V[batch_head_offset + kv_offset * head_dim + tid]);
                }
            }
            else {
                for (int tid = threadIdx.x; tid < HEAD_DIM; tid+=blockDim.x) {
                    K_s[kv_idx * head_dim  + tid] = 0.0f;
                    V_s[kv_idx * head_dim  + tid] = 0.0f;
                }
            }
        }
        __syncthreads();

        //compute attention score qk^T

        for (int kv_idx = tid; kv_idx < TILE_SIZE; kv_idx+=blockDim.x) {
            int kv_offset = tile * TILE_SIZE + kv_idx;
            if (kv_offset < seq_len) {
                float qk = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    qk += q_s[d] * K_s[kv_idx*head_dim + d];
                }
                qK_s[kv_idx] = qk * softmax_scale;
            }
            else {
                qK_s[kv_idx] = -INFINITY;
            }
        }
   
        __syncthreads();


        // find max

        float prev_max = max_qk_s;
        if (tid == 0) {
            float block_max = -INFINITY;
            for (int kv_idx=0; kv_idx < TILE_SIZE; kv_idx++) {
                block_max = fmaxf(block_max, qK_s[kv_idx]);
            }
            max_qk_s = fmaxf(block_max, prev_max);
        }
        __syncthreads();
        float new_max = max_qk_s;

        for (int kv_idx=tid; kv_idx < TILE_SIZE; kv_idx+=blockDim.x) {
            qK_s[kv_idx] = expf(qK_s[kv_idx] - new_max);
        }
        float scale = expf(prev_max - new_max);

        __syncthreads();

        // for (int d=tid; d < TILE_SIZE; d+=blockDim.x) {
        //     o_s[d] = o_s[d] * scale;
        // }

        // accum current tile V and rescale previous O
        for (int d=tid; d < head_dim; d+=blockDim.x) {
            float acc = 0.0f;
            for (int kv_idx = 0; kv_idx < TILE_SIZE;  kv_idx++ ) {
                acc  +=  qK_s[kv_idx] * V_s[kv_idx * head_dim + d];
            }
            o_s[d] =  o_s[d] * scale +  acc;
            // o_s[d] =  acc;
        }
        
        // calculate sum exp

        if (tid == 0 ) {
            float block_sum_exp = 0.0f;
            for (int kv_idx=0; kv_idx < TILE_SIZE; kv_idx++) {
                block_sum_exp += qK_s[kv_idx];
            }
            sum_exp_s = sum_exp_s * scale + block_sum_exp;
        }
        __syncthreads();
    }

    float sum_exp_inv = 1.0f / sum_exp_s;
    if (q_idx < seq_len) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            O[batch_head_offset  + q_idx*head_dim + d] = static_cast<scalar_t>(o_s[d] * sum_exp_inv) ;
        }
    }
}


torch::Tensor flash_attn(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (batch, num_heads, seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D");
    TORCH_CHECK(V.dim() == 4, "V must be 4D");

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);

    TORCH_CHECK(K.size(0) == batch_size && K.size(1) == num_heads &&
                K.size(2) == seq_len && K.size(3) == head_dim,
                "K shape must match Q");
    TORCH_CHECK(V.size(0) == batch_size && V.size(1) == num_heads &&
                V.size(2) == seq_len && V.size(3) == head_dim,
                "V shape must match Q");

    // Allocate output
    auto O = torch::empty({batch_size, num_heads, seq_len, head_dim},
                          torch::dtype(torch::kFloat32).device(Q.device()));
    // auto O = torch::empty_like(Q);
    // auto L = torch::empty({batch_size, num_heads, seq_len},
                        //   torch::dtype(torch::kFloat32).device(Q.device()));

    // Kernel configuration
    constexpr int TILE_SIZE = 64;  // Tile size for K/V
    constexpr int HEAD_DIM = 64;    // Assuming head_dim = 64, can template for others
    constexpr int THREADS = 64;

    TORCH_CHECK(head_dim == HEAD_DIM,
                "Currently only head_dim=64 is supported. Got head_dim=", head_dim);

    // Shared memory size calculation
    const int smem_size = (
                        2 * HEAD_DIM +                    // smem_q, smem_o
                        2 * TILE_SIZE * HEAD_DIM +       // smem_k, smem_v
                        TILE_SIZE  // smem_qk
                        ) * sizeof(float);  

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(THREADS);

    if (softmax_scale == 0.0f) {
        softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "flash_attn", ([&] {
        flash_attn_kernel<scalar_t, TILE_SIZE, HEAD_DIM><<<grid, block, smem_size>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            O.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            softmax_scale
        );
    }));

    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return O;
}





