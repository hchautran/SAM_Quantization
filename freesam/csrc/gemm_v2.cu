#include <iostream>
#include "cuda_utils.cuh"
#include <torch/extension.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// CUTLASS GEMM using Tensor Cores (FP16)
// Performs: D = alpha * (A @ B) + beta * C
// A: [M x K] row-major FP16
// B: [K x N] row-major FP16
// C/D: [M x N] row-major FP16
torch::Tensor cutlass_gemm(torch::Tensor a, torch::Tensor b) {
    // Get problem dimensions
    int M = a.size(0);
    int N = b.size(1);
    int K = a.size(1);

    // Input validation
    TORCH_CHECK(a.is_cuda(), "Tensor A must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "Tensor B must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat16, "Tensor A must be FP16, got ", a.dtype());
    TORCH_CHECK(b.dtype() == torch::kFloat16, "Tensor B must be FP16, got ", b.dtype());
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match: A[", M, "x", K, "] vs B[", b.size(0), "x", N, "]");

    // Ensure contiguous memory
    a = a.contiguous();
    b = b.contiguous();

    // Allocate output tensor
    auto out = torch::empty({M, N}, a.options());

    // ============= CUTLASS Type Definitions =============

    // Element types
    using ElementA = cutlass::half_t;           // Input A element type
    using ElementB = cutlass::half_t;           // Input B element type
    using ElementC = cutlass::half_t;           // Output element type
    using ElementAccumulator = float;           // Accumulator type (FP32 for precision)

    // Memory layouts (PyTorch tensors are row-major by default)
    using LayoutA = cutlass::layout::RowMajor;     // A is row-major
    using LayoutB = cutlass::layout::RowMajor;     // B is row-major
    using LayoutC = cutlass::layout::RowMajor;     // C is row-major

    // Operation class and architecture
    using MMAOp = cutlass::arch::OpClassTensorOp;  // Use Tensor Cores
    using SmArch = cutlass::arch::Sm80;            // SM80 = Ampere (A100, RTX 3090)

    // Tile sizes for blocking
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Threadblock tile M, N, K
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;           // Warp tile M, N, K
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;     // HMMA instruction M, N, K

    // Alignment (number of elements per memory access)
    // 8 FP16 elements = 16 bytes (optimal for Tensor Cores)
    constexpr int AlignmentA = 8;
    constexpr int AlignmentB = 8;

    // Epilogue: D = alpha * (A @ B) + beta * C
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,                                        // Output element type
        128 / cutlass::sizeof_bits<ElementC>::value,    // Alignment (8 elements for FP16)
        ElementAccumulator,                              // Accumulator type
        ElementAccumulator                               // Scalar type (alpha, beta)
    >;

    // Thread block swizzle (how threadblocks are scheduled)
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // Define the GEMM kernel
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        ThreadblockSwizzle,
        3,           // Number of pipeline stages
        AlignmentA,  // Alignment for A
        AlignmentB   // Alignment for B
    >;

    // ============= Setup GEMM Arguments =============

    // Problem size
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Scalars: D = 1.0 * (A @ B) + 0.0 * C
    ElementAccumulator alpha = ElementAccumulator(1.0f);
    ElementAccumulator beta = ElementAccumulator(0.0f);

    // Leading dimensions (stride to next row for row-major matrices)
    int lda = K;  // A: [M × K] row-major, stride = K
    int ldb = N;  // B: [K × N] row-major, stride = N
    int ldc = N;  // C: [M × N] row-major, stride = N

    // Setup arguments
    typename Gemm::Arguments arguments{
        problem_size,
        {reinterpret_cast<ElementA*>(a.data_ptr<at::Half>()), lda},      // Tensor A
        {reinterpret_cast<ElementB*>(b.data_ptr<at::Half>()), ldb},      // Tensor B
        {reinterpret_cast<ElementC*>(out.data_ptr<at::Half>()), ldc},    // Tensor C (source)
        {reinterpret_cast<ElementC*>(out.data_ptr<at::Half>()), ldc},    // Tensor D (destination)
        {alpha, beta}                                                     // Scalars
    };

    // ============= Execute GEMM =============

    // Instantiate GEMM kernel
    Gemm gemm_op;

    // Check if configuration is valid
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::string error_msg = "CUTLASS GEMM configuration error!\n";
        error_msg += "  Problem size: M=" + std::to_string(M) + ", N=" + std::to_string(N) + ", K=" + std::to_string(K) + "\n";
        error_msg += "  Status code: " + std::to_string((int)status) + "\n";
        error_msg += "  Common causes:\n";
        error_msg += "    1. Alignment: Dimensions should be multiples of 8 for best performance\n";
        error_msg += "    2. GPU architecture mismatch (need SM80/Ampere or change SmArch)\n";
        error_msg += "    3. Input dtype mismatch (need FP16)\n";
        throw std::runtime_error(error_msg);
    }

    // Initialize GEMM workspace
    status = gemm_op.initialize(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS initialize failed");

    // Launch GEMM kernel
    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM execution failed");

    return out;
}

