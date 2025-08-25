#include <cutlass/gemm/device/gemm.h>
#include <gemm_int8.h>

void matmul_i8_host(
        const int8_t *A,
        const int8_t *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
)
{
    using Gemm = cutlass::gemm::device::Gemm<
            int8_t,                          // ElementA
            cutlass::layout::RowMajor,       // LayoutA
            int8_t,                          // ElementB
            cutlass::layout::ColumnMajor,    // LayoutB (B is given as [N,K])
            int32_t,                         // ElementOutput
            cutlass::layout::RowMajor,       // LayoutOutput
            int32_t,                         // ElementAccumulator
            cutlass::arch::OpClassTensorOp,  // Tensor Cores
            cutlass::arch::Sm80              // Target GPU arch
    >;

    Gemm gemmOp;
    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{
            {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
            {(int8_t *) A,                     K},
            {(int8_t *) B,                     K},
            {C,                                N},
            {C,                                N},
            {1,                                0}
    };

    auto status = gemmOp(arguments);
    (void)status; // If you want, add error checks similar to gemm.cu
}


