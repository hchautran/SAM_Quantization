"""
Batched FP16 GEMM: C[b] = A[b] @ B[b]^T using CuTe DSL on Ampere SM80.

Tensors (all fp16, row-major):
  A : (B, M, K)
  B : (B, N, K)
  C : (B, M, N)

Equivalent to: torch.bmm(A, B.transpose(-1, -2))

Tile 128×128×32, 3-stage SMEM pipeline, 2×2×1 atom layout, 128 threads.
Grid: (ceil(M/128) × raster, ceil(N/128) // raster, B) — bidz indexes the batch.

Usage:
    python batch_gemm.py --bmnk 4,256,256,128
"""

import argparse
import math
import functools

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack


def gpu_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        print(f"{func.__name__}: {start.elapsed_time(end):.4f} ms")
        return result
    return wrapper


class BatchFp16Gemm:
    """Batched FP16 GEMM: C[b,m,n] = sum_k A[b,m,k] * B[b,n,k].

    Tile 128×128×32, 3-stage SMEM pipeline, 128 threads.
    The batch dimension B is mapped to grid dim-z (bidz).
    """

    def __init__(self, tile_size=(128, 128, 32), num_threads=128):
        self.ab_dtype        = cutlass.Float16
        self.c_dtype         = cutlass.Float16
        self.acc_dtype       = cutlass.Float32
        self.cta_tiler       = tuple(tile_size)
        self.num_stages      = 3
        
        num_warps = num_threads // 32
        # Simple warp factorization for atom_layout_mnk (atom_m, atom_n, 1)
        if num_warps == 4:
            self.atom_layout_mnk = (2, 2, 1)
        elif num_warps == 8:
            self.atom_layout_mnk = (4, 2, 1) if tile_size[0] >= tile_size[1] else (2, 4, 1)
        elif num_warps == 2:
            self.atom_layout_mnk = (2, 1, 1) if tile_size[0] >= tile_size[1] else (1, 2, 1)
        elif num_warps == 16:
            self.atom_layout_mnk = (4, 4, 1)
        else:
            # Fallback
            w_m = 1
            w_n = 1
            while w_m * w_n < num_warps:
                if w_m * 16 < tile_size[0] and (w_m <= w_n or w_n * 8 >= tile_size[1]):
                    w_m *= 2
                else:
                    w_n *= 2
            self.atom_layout_mnk = (w_m, w_n, 1)

        self.num_threads     = num_threads
        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape  = (16, 8, 16)

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stride_factor: int = 1):
        """Configure SMEM layouts, copy atoms, tiled MMA, then launch the kernel.

        mA: (B, M, K) fp16 row-major
        mB: (B, N, K) fp16 row-major
        mC: (B, M, N) fp16 row-major  — written in place
        """
        # major_mode is inferred from the tensor's leading dim (K for row-major A/B, N for C)
        self.a_major_mode = utils.LayoutEnum.ROW_MAJOR if mA.leading_dim == 2 else utils.LayoutEnum.COL_MAJOR
        self.b_major_mode = utils.LayoutEnum.ROW_MAJOR if mB.leading_dim == 2 else utils.LayoutEnum.COL_MAJOR
        self.c_major_mode = utils.LayoutEnum.ROW_MAJOR if mC.leading_dim == 2 else utils.LayoutEnum.COL_MAJOR

        ab_copy_bits = 128

        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, self.b_major_mode, ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = self._make_smem_layout_C(
            mC.element_type, self.c_major_mode, ab_copy_bits,
            (self.bM, self.bN),
        )

        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout),
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        c_copy_bits  = 128
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        permutation_mnk = (
            atom_lay_M * self.mma_inst_shape[0],
            atom_lay_N * self.mma_inst_shape[1] * 2,
            atom_lay_K * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout(self.atom_layout_mnk), permutation_mnk=permutation_mnk
        )

        # Grid dim-z = batch; dim-x/y cover the M×N tile grid (with rasterization).
        # mC.shape = (B, M, N) → treat inner (M, N, 1) for ceil_div.
        mn_tiles = cute.ceil_div((mC.shape[1], mC.shape[2], 1), (self.bM, self.bN, 1))
        grid_dim_n = cute.size(mn_tiles[1])
        raster_factor = 1
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2

        grid = (
            cute.size(mn_tiles[0]) * raster_factor,
            (cute.size(mn_tiles[1]) + raster_factor - 1) // raster_factor,
            cute.size(mC.shape[0]),   # batch
        )

        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout, sC_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_C,
            tiled_mma, raster_factor, stride_factor,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,          # (B, M, K)
        mB: cute.Tensor,          # (B, N, K)
        mC: cute.Tensor,          # (B, M, N)
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
        stride_factor: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()   # bidz = batch index

        # Decode rasterized (bidx, bidy) → (tile_m, tile_n)
        mn_tiles      = cute.ceil_div((mC.shape[1], mC.shape[2], 1), (self.bM, self.bN, 1))
        offset_tile_m = bidx // rasterization_factor
        offset_tile_n = (bidx % rasterization_factor) + (bidy * rasterization_factor)

        if mn_tiles[0] <= offset_tile_m or mn_tiles[1] <= offset_tile_n:
            pass
        elif 0 < offset_tile_m < stride_factor:
            pass
        else:
            tiler_coord = (offset_tile_m, offset_tile_n, None)

            original_mC_b = mC[bidz, None, None]
            mA_b = mA[bidz, None, None]   # (M, K)
            mB_b = mB[bidz, None, None]   # (N, K)
            mC_b = original_mC_b          # (M, N)

            if offset_tile_m == 0:
                stride_m_A = mA_b.layout.stride[0]
                stride_k_A = mA_b.layout.stride[1]
                shape_m_A  = mA_b.layout.shape[0]
                shape_k_A  = mA_b.layout.shape[1]
                mA_b = cute.make_tensor(
                    mA_b.iterator,
                    cute.make_layout((shape_m_A // stride_factor, shape_k_A), stride=(stride_m_A * stride_factor, stride_k_A))
                )

                stride_m_C = mC_b.layout.stride[0]
                stride_n_C = mC_b.layout.stride[1]
                shape_m_C  = mC_b.layout.shape[0]
                shape_n_C  = mC_b.layout.shape[1]
                mC_b = cute.make_tensor(
                    mC_b.iterator,
                    cute.make_layout((shape_m_C // stride_factor, shape_n_C), stride=(stride_m_C * stride_factor, stride_n_C))
                )



            # CTA tiles from global memory
            gA = cute.local_tile(mA_b, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, None, 1))
            gB = cute.local_tile(mB_b, tiler=self.cta_tiler, coord=tiler_coord, proj=(None, 1, 1))
            gC = cute.local_tile(mC_b, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None))

            # Shift so the first tile covers the K-residue (not the last)
            residual_k = cute.size(mA_b, mode=[1]) - cutlass.Int32(self.bK) * cute.size(gA, mode=[2])
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            # Identity tensors for M/N/K boundary predication
            mcA = cute.make_identity_tensor(mA_b.layout.shape)
            mcB = cute.make_identity_tensor(mB_b.layout.shape)
            cA  = cute.local_tile(mcA, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, None, 1))
            cB  = cute.local_tile(mcB, tiler=self.cta_tiler, coord=tiler_coord, proj=(None, 1, 1))
            cA  = cute.domain_offset((0, residual_k, 0), cA)
            cB  = cute.domain_offset((0, residual_k, 0), cB)

            # Shared memory
            smem = cutlass.utils.SmemAllocator()
            sA   = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB   = smem.allocate_tensor(mB.element_type, sB_layout, 16)
            sC   = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout)

            # Thread-level copy partitions (G2S for A/B, epilogue for C)
            thr_copy_A       = tiled_copy_A.get_slice(tidx)
            thr_copy_B       = tiled_copy_B.get_slice(tidx)
            thr_copy_C       = tiled_copy_C.get_slice(tidx)
            tAgA             = thr_copy_A.partition_S(gA)
            tAsA             = thr_copy_A.partition_D(sA)
            tBgB             = thr_copy_B.partition_S(gB)
            tBsB             = thr_copy_B.partition_D(sB)
            tCsC_epilogue    = thr_copy_C.partition_S(sC)
            tCgC_epilogue    = thr_copy_C.partition_D(gC)
            tAcA             = thr_copy_A.partition_S(cA)
            tBcB             = thr_copy_B.partition_S(cB)

            # M/N boundary predicates (K predicate handled per-tile via domain_offset)
            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(
                        tAcA[(0, rest_v), m, 0, 0][0], mA_b.shape[0]
                    )
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(
                        tBcB[(0, rest_v), n, 0, 0][0], mB_b.shape[0]
                    )

            # ---- Prologue: pre-fill SMEM and prefetch first num_stages-1 tiles ----
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()

            num_smem_stages = cute.size(tAsA, mode=[3])
            k_tile_count    = cute.size(tAgA, mode=[3])
            k_tile_index    = cutlass.Int32(0)

            # First K-tile with residue predication on the K dimension
            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(tiled_copy_A, tAgA[None, None, k, k_tile_index], tAsA[None, None, k, 0], pred=tApA[None, None, k])
            for k in range(tBpB.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(tiled_copy_B, tBgB[None, None, k, k_tile_index], tBsB[None, None, k, 0], pred=tBpB[None, None, k])
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            # Remaining prologue stages (no K-residue)
            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, k_tile], pred=tApA)
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, k_tile], pred=tBpB)
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # ---- MMA partitions and register fragments ----
            thr_mma          = tiled_mma.get_slice(tidx)
            tCsA             = thr_mma.partition_A(sA)
            tCsB             = thr_mma.partition_B(sB)
            # cute.printf("tCsA: {}" , tCsA)
            print(tCsA)
            tCsC             = thr_mma.partition_C(sC)
            tCgC             = thr_mma.partition_C(gC)
            tCrA             = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB             = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC             = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            # S2R copy atoms (LdMatrix for efficient SMEM→register)
            atom_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mA.element_type,
            )
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mB.element_type,
            )
            tiled_copy_s2r_A     = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_copy_s2r_B     = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_ldmatrix_A       = tiled_copy_s2r_A.get_slice(tidx)
            thr_ldmatrix_B       = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view       = thr_ldmatrix_A.partition_S(sA)
            tCrA_copy_view       = thr_ldmatrix_A.retile(tCrA)
            tCsB_copy_view       = thr_ldmatrix_B.partition_S(sB)
            tCrB_copy_view       = thr_ldmatrix_B.retile(tCrB)

            smem_pipe_read  = 0
            smem_pipe_write = num_smem_stages - 1
            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            # Register pipeline prefetch
            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

            # ---- Mainloop: interleaved SMEM + register pipeline ----
            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_pipe_write], pred=tApA)

                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_pipe_write], pred=tBpB)
                        k_tile_index    = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read  = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ---- Epilogue: register → SMEM → global with M/N boundary predication ----
            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            tCrD[None] = tCrC.load().to(self.c_dtype)
            cute.autovec_copy(tCrD, tCsC)

            # Identity tensor for C boundary predication (uses padded tile dims)
            ceilM = cute.ceil_div(mC_b.shape[0], self.bM)
            ceilN = cute.ceil_div(mC_b.shape[1], self.bN)
            mcC = cute.make_identity_tensor((
                cute.size(ceilM) * self.cta_tiler[0],
                cute.size(ceilN) * self.cta_tiler[1]
            ))
            cC    = cute.local_tile(mcC, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None))
            tCcC  = thr_copy_C.partition_S(cC)

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.arch.sync_threads()
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            tCpC = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tCgC_epilogue.shape[0][1],
                        cute.size(tCgC_epilogue, mode=[1]),
                        cute.size(tCgC_epilogue, mode=[2]),
                    ),
                    stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tCpC.shape[0]):
                for m in range(tCpC.shape[1]):
                    tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC_b.shape[0])

            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC_b.shape[1]):
                        if offset_tile_m == 0:
                            for s in range(stride_factor):
                                shifted_mC = cute.domain_offset((s, 0), original_mC_b)
                                shape_m_C_loc = original_mC_b.layout.shape[0]
                                shape_n_C_loc = original_mC_b.layout.shape[1]
                                stride_m_C_loc = original_mC_b.layout.stride[0]
                                stride_n_C_loc = original_mC_b.layout.stride[1]
                                mC_b_s = cute.make_tensor(
                                    shifted_mC.iterator,
                                    cute.make_layout((shape_m_C_loc // stride_factor, shape_n_C_loc), stride=(stride_m_C_loc * stride_factor, stride_n_C_loc))
                                )
                                gC_s = cute.local_tile(mC_b_s, tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None))
                                tCgC_s = thr_copy_C.partition_D(gC_s)
                                cute.copy(
                                    tiled_copy_C,
                                    tCrC_epilogue[None, None, n],
                                    tCgC_s[None, None, n],
                                    pred=tCpC[None, None, n],
                                )
                        else:
                            cute.copy(
                                tiled_copy_C,
                                tCrC_epilogue[None, None, n],
                                tCgC_epilogue[None, None, n],
                                pred=tCpC[None, None, n],
                            )
        return

    # ---- SMEM / copy layout helpers (identical to Fp16Matmul) ----

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size
        swizzle_bits    = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits    = min(swizzle_bits, 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer)
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 4), 0, layout_atom_outer)
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(cute.make_swizzle(0, 3, 4), 0, layout_atom_outer)
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems   = copy_bits // dtype.width
        shape_dim_1  = cute.size(self.bK) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0   = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems)) if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems   = copy_bits // dtype.width
        shape_dim_1  = cute.size(self.bN) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0   = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems)) if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


def _wrap_tensor(t: torch.Tensor) -> "cute.Tensor":
    """Wrap a (B, M, K) or (B, N, K) or (B, M, N) row-major fp16 tensor as CuTe tensor.

    Dim 2 (K or N) is the innermost/contiguous dim (leading dim in CuTe terminology).
    """
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(
            mode=2,
            stride_order=t.dim_order(),   # (0,1,2): B outermost, K innermost
            divisibility=8,               # 128 bits / 16 bits per element
        )
    )


@gpu_timer
def torch_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b.transpose(-1, -2))


@gpu_timer
def torch_bmm_sparse(a: torch.Tensor, b: torch.Tensor, tile_m: int, stride_factor: int) -> torch.Tensor:
    # Emulate the sparse behavior: first block of M computes a strided gather.
    # The rest compute normal densely.
    c = torch.zeros((a.shape[0], a.shape[1], b.shape[1]), dtype=a.dtype, device=a.device)
    # Sparse block
    sparse_m_size = tile_m * stride_factor
    if sparse_m_size > a.shape[1]:
        sparse_m_size = a.shape[1]
    
    a_sparse = a[:, 0:sparse_m_size:stride_factor, :]
    
    c_sparse = torch.bmm(a_sparse, b.transpose(-1, -2))
    for s in range(stride_factor):
        c[:, s:sparse_m_size:stride_factor, :] = c_sparse
    
    # Dense blocks starting from tile_m * stride_factor
    dense_start = tile_m * stride_factor
    if dense_start < a.shape[1]:
        a_dense = a[:, dense_start:, :]
        c_dense = torch.bmm(a_dense, b.transpose(-1, -2))
        c[:, dense_start:, :] = c_dense

    return c


def run(B: int, M: int, N: int, K: int, tile_size=(128, 128, 32), num_threads=128, stride_factor=1):
    print(f"Batched FP16 GEMM: B={B} M={M} N={N} K={K}")
    print(f"Tile {tile_size[0]}×{tile_size[1]}×{tile_size[2]}, 3-stage pipeline, {num_threads} threads")
    print(f"Sparse Stride Factor: {stride_factor}")

    device = torch.device("cuda")

    a_torch = torch.empty(B, M, K, dtype=torch.int32).random_(-2, 2).to(torch.float16).cuda()
    b_torch = torch.empty(B, N, K, dtype=torch.int32).random_(-2, 2).to(torch.float16).cuda()
    c_torch = torch.zeros(B, M, N, dtype=torch.float16).cuda()
    print(a_torch.shape)
    print(b_torch.shape)
    print(c_torch.shape)

    mA = _wrap_tensor(a_torch)
    mB = _wrap_tensor(b_torch)
    mC = _wrap_tensor(c_torch)

    gemm = BatchFp16Gemm(tile_size, num_threads)

    print("Compiling kernel...")
    compiled_gemm = cute.compile(gemm, mA, mB, mC, stride_factor)

    # Reference
    if stride_factor > 1:
        ref = torch_bmm_sparse(a_torch, b_torch, tile_size[0], stride_factor)
    else:
        ref = torch_bmm(a_torch, b_torch)

    print("Executing kernel...")
    compiled_gemm(mA, mB, mC, stride_factor)
    torch.cuda.synchronize()

    print("Verifying results...")
    torch.testing.assert_close(c_torch.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
    print("PASS — results match reference!")

    # Benchmark
    warmup, iters = 10, 100
    flops = 2.0 * B * M * N * K

    with torch.no_grad():
        for _ in range(warmup):
            torch.bmm(a_torch, b_torch.transpose(-1, -2))
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            torch.bmm(a_torch, b_torch.transpose(-1, -2))
        e.record()
        torch.cuda.synchronize()
    torch_ms = s.elapsed_time(e) / iters

    for _ in range(warmup):
        compiled_gemm(mA, mB, mC, stride_factor)
    torch.cuda.synchronize()
    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(iters):
        compiled_gemm(mA, mB, mC, stride_factor)
    e2.record()
    torch.cuda.synchronize()
    cute_ms = s2.elapsed_time(e2) / iters

    print(f"\n{'='*50}")
    print(f"  torch.bmm : {torch_ms:.4f} ms  ({flops / torch_ms / 1e9:.2f} TFLOPS)")
    print(f"  CuTe BGEMM: {cute_ms:.4f} ms  ({flops / cute_ms / 1e9:.2f} TFLOPS)")
    print(f"  Speedup   : {torch_ms / cute_ms:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched FP16 GEMM — Ampere SM80")
    parser.add_argument("--bmnk", type=str, default="2,4096,1024,1024",
                        help="B,M,N,K dimensions (comma-separated)")
    parser.add_argument("--tile_size", type=str, default="64,128,32",
                        help="Tile size M,N,K (comma-separated)")
    parser.add_argument("--num_threads", type=int, default=128,
                        help="Number of threads per block")
    parser.add_argument("--stride_factor", type=int, default=1,
                        help="Sparse gather stride factor for the first block")
    args = parser.parse_args()
    B, M, N, K = (int(x.strip()) for x in args.bmnk.split(","))
    tile_size = tuple(int(x.strip()) for x in args.tile_size.split(","))
    run(B, M, N, K, tile_size, args.num_threads, args.stride_factor)
