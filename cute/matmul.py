"""
FP16 Matmul kernel optimized for A100 (SM80 Ampere) using CuTe DSL.

Computes C = A @ B^T where:
  - A is (M, K) row-major fp16
  - B is (N, K) row-major fp16
  - C is (M, N) row-major fp16

Equivalent to: torch.einsum("mk,nk->mn", a, b)

Tile: 128x128x32, 3-stage SMEM pipeline, 2x2x1 MMA atom layout, 128 threads.
Uses cp.async for G2S, LdMatrix for S2R, MmaF16BF16Op 16x8x16 tensor cores.

Usage:
    python matmul.py --mnk 256,256,256
    python matmul.py --mnk 8192,8192,8192
"""

import argparse
import math
import functools
from typing import Tuple, Type

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack


def gpu_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        print(f"{func.__name__}: {elapsed_ms:.4f} ms")
        return result
    return wrapper


class Fp16Matmul:
    """FP16 GEMM: C(M,N) = A(M,K) * B(N,K)^T using Ampere tensor cores.

    Tile shape 128x128x32, 3-stage pipeline, 2x2x1 atom layout = 128 threads.
    Supports arbitrary M/N/K with full boundary predication.
    """

    def __init__(self):
        self.ab_dtype = cutlass.Float16
        self.c_dtype = cutlass.Float16
        self.acc_dtype = cutlass.Float32
        self.cta_tiler = (128, 128, 32)
        self.num_stages = 3
        self.atom_layout_mnk = (2, 2, 1)
        self.num_threads = 2 * 2 * 1 * 32  # 128
        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8,16 )

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 block_mask: cute.Tensor):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        # ---- Swizzled SMEM layouts (avoid bank conflicts) ----
        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.bM, self.bK, self.num_stages), # row major
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, self.b_major_mode, ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = self._make_smem_layout_C(
            mC.element_type, self.c_major_mode, ab_copy_bits,
            (self.bM, self.bN),
        )

        # A/B SMEM buffers reused for C epilogue
        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout),
        )

        # ---- cp.async copy atoms for G2S (128-bit vectorized) ----
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
        print('tiled copy_A.layout_tv_tiled:')
        print(tiled_copy_A)
        
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        # ---- Sync copy for epilogue (R2S2G) ----
        c_copy_bits = 128
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )

        # ---- Tiled MMA: 16x8x16 fp16 tensor core, 2x2x1 atom layout ----
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        permutation_mnk = (
            atom_lay_M * self.mma_inst_shape[0],
            atom_lay_N * self.mma_inst_shape[1] * 2,
            atom_lay_K * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        # ---- Grid with threadblock rasterization ----
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        raster_grid = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            cute.size(grid_dim[2]),
        )

        self.kernel(
            mA, mB, mC, block_mask,
            sA_layout, sB_layout, sC_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_C,
            tiled_mma, raster_factor,
        ).launch(
            grid=raster_grid,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        block_mask: cute.Tensor,  # (L, num_m_tiles, num_n_tiles) int8; 0 = sparse (first 2 K-tiles only)
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))

        # Threadblock rasterization for better L2 reuse
        offset_tile_x = bidx // rasterization_factor
        offset_tile_y = (bidx % rasterization_factor) + (bidy * rasterization_factor)

        # Early exit for out-of-range CTAs
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            # ---- Per-CTA tiles from global memory ----
            gA = cute.local_tile(
                mA[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(1, None, 1),
            )
            # if tidx==0 and bidx==0:
            #     cute.printf(gA)
            #     cute.printf(mA)
            # if tidx==0:
                # cute.printf(gA )
            gB = cute.local_tile(
                mB[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(None, 1, 1),
            )
            gC = cute.local_tile(
                mC[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(1, 1, None),
            )
            # if tidx==0 and bidx==0:
                # cute.printf(gC)
                # cute.printf(mC)

            # Handle K residue: shift so first tile is irregular (not last)
            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(
                gA, mode=[2]
            )
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            # Identity tensors for boundary predication
            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB.layout.shape)
            cA = cute.local_tile(
                mcA[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(1, None, 1),
            )
            cB = cute.local_tile(
                mcB[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(None, 1, 1),
            )
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            # ---- Allocate shared memory ----
            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

            sC = cute.make_tensor(
                cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout
            )

            # ---- Thread-level copy partitions (G2S) ----
            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)

            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            # ---- M/N boundary predicate tensors ----
            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]),
                     cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBsB.shape[0][1], cute.size(tBsB, mode=[1]),
                     cute.size(tBsB, mode=[2])),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(
                        tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                    )
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(
                        tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                    )

            # ---- Prologue: prefetch first SMEM stages ----
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()

            num_smem_stages = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            k_tile_index = cutlass.Int32(0)

            # First k-tile with K-residue predication
            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, k, k_tile_index],
                        tAsA[None, None, k, 0],
                        pred=tApA[None, None, k],
                    )
            for k in range(tBpB.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, k, k_tile_index],
                        tBsB[None, None, k, 0],
                        pred=tBpB[None, None, k],
                    )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            # Remaining prologue stages
            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile_index],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # ---- MMA partitions and accumulator ----
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCsC = thr_mma.partition_C(sC)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            # ---- S2R copy atoms (LdMatrix for efficient SMEM->register) ----
            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mA.element_type,
            )
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mB.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            # SMEM pipeline read/write cursors
            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1
            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            # ---- Register pipeline prefetch ----
            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

            # ---- Mainloop: interleaved SMEM + register pipeline ----
            # Sparse blocks compute only the first 2 K-tiles; dense blocks compute all.
            is_sparse = block_mask[bidz, offset_tile_x, offset_tile_y] == cutlass.Int8(0)
            for k_tile in range(k_tile_count):
                compute_tile = (not is_sparse) or (k_tile < 2)
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    # Prefetch next k-block from SMEM to registers
                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_p[None, None, k_block_next],
                        tCrA_copy_view[None, None, k_block_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_p[None, None, k_block_next],
                        tCrB_copy_view[None, None, k_block_next],
                    )

                    # Fetch next A tile (interleaved with compute for latency hiding)
                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_A,
                                tAgA[None, None, None, k_tile_index],
                                tAsA[None, None, None, smem_pipe_write],
                                pred=tApA,
                            )

                    # Tensor core MMA
                    if compute_tile:
                        cute.gemm(
                            tiled_mma,
                            tCrC,
                            tCrA[None, None, k_block],
                            tCrB[None, None, k_block],
                            tCrC,
                        )

                    # Fetch next B tile and advance SMEM pipeline
                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, k_tile_index],
                                tBsB[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ---- Epilogue: register -> SMEM -> global with predication ----
            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            tCrD[None] = tCrC.load().to(self.c_dtype)
            cute.autovec_copy(tCrD, tCsC)

            # Coord tensor for C boundary predication
            ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
            mcC = cute.make_identity_tensor((
                cute.size(ceilM) * self.cta_tiler[0],
                cute.size(ceilN) * self.cta_tiler[1],
                1,
            ))
            cC = cute.local_tile(
                mcC[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(1, 1, None),
            )
            tCcC = thr_copy_C.partition_S(cC)

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.arch.sync_threads()
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            # M/N boundary predication for epilogue stores
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
                    tCpC[rest_v, m, 0] = cute.elem_less(
                        tCcC[(0, rest_v), m, 0][0], mC.shape[0]
                    )

            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(
                            tiled_copy_C,
                            tCrC_epilogue[None, None, n],
                            tCgC_epilogue[None, None, n],
                            pred=tCpC[None, None, n],
                        )
        return

    # ---- SMEM / copy layout helpers ----

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR
            else smem_tiler[0]
        )
        

        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size
        ### prevent bank conflicts for 128-bit copy with large tiles: swizzle the layout based on tile size, element size, and copy vector size
        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = (
            cute.make_layout(
                (8, major_mode_size), stride=(major_mode_size, 1)
            )
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout(
                (major_mode_size, 8), stride=(1, major_mode_size)
            )
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer,
        )
        # print(layout_atom_outer)
        # print(smem_tiler)
        smem_layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))
        return smem_layout

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR
            else smem_tiler[0]
        )
        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout(
                (major_mode_size, 8), stride=(1, major_mode_size)
            )
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4), 0, layout_atom_outer,
        )
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer,
            )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        # breakpoint()
        shape_dim_1 = cute.size(self.bK) // copy_elems # 8 for fp16
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), # [32,4]
            stride=(shape_dim_1, 1), 
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0),
                stride=(1, shape_dim_0),
            )
        value_layout = (
            cute.make_layout((1, copy_elems)) # [1, 8]
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout) 
        return  tiled_copy

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bN) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1),
            stride=(shape_dim_1, 1),
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0),
                stride=(1, shape_dim_0),
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


def _wrap_mask_tensor(t: torch.Tensor) -> "cute.Tensor":
    """Wrap a (L, num_m_tiles, num_n_tiles) int8 block-mask as a CuTe tensor."""
    return (
        from_dlpack(t, assumed_align=1)
        .mark_layout_dynamic(leading_dim=2)
        .mark_compact_shape_dynamic(mode=2, stride_order=t.dim_order(), divisibility=1)
    )


def _make_cute_tensor(torch_tensor, is_mode0_major):
    """Convert a 3D PyTorch tensor to CuTe tensor with layout annotations."""
    return (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
        .mark_compact_shape_dynamic(
            mode=(1 if not is_mode0_major else 0),
            stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
            divisibility=8,  # 128 bits / 16 bits per element
        )
    )


@gpu_timer
def torch_matmul(a: torch.Tensor, b: torch.Tensor):
    """Reference: einsum mk,nk->mn in fp32, cast back to fp16."""
    return a.half() @ b.half().T


def run(M: int, N: int, K: int, sparse_ratio: float = 0.0):
    print(f"FP16 Matmul: M={M}, N={N}, K={K}")
    print(f"Tile: 128x128x32, 3-stage pipeline, 128 threads")
    print(f"Sparse block ratio: {sparse_ratio:.2f} (sparse blocks compute only first 2 K-tiles)")

    device = torch.device("cuda")
    L = 1  # no batch

    # Row-major tensors: A(M,K) K-contiguous, B(N,K) K-contiguous, C(M,N) N-contiguous
    # 3D layout with batch dim L=1: A(M,K,L), B(N,K,L), C(M,N,L)
    a_torch = (
        torch.empty(L, M, K, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=torch.float16)
        .permute(1, 2, 0)  # (M, K, L) K-contiguous
        .cuda()
    )
    print(a_torch.shape)
    b_torch = (
        torch.empty(L, N, K, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=torch.float16)
        .permute(1, 2, 0)  # (N, K, L) K-contiguous
        .cuda()
    )
    print(b_torch.shape)
    c_torch = torch.zeros(L, M, N, dtype=torch.float16).permute(1, 2, 0).cuda()
    print(c_torch.shape)

    bM, bN, bK = 128, 128, 32
    num_m_tiles = (M + bM - 1) // bM
    num_n_tiles = (N + bN - 1) // bN
    num_k_tiles = (K + bK - 1) // bK
    mask_torch = (torch.rand(L, num_m_tiles, num_n_tiles) >= sparse_ratio).to(torch.int8).cuda()
    print(f"Block mask: {mask_torch.float().mean():.2%} dense, {1 - mask_torch.float().mean():.2%} sparse")

    # CuTe tensors with layout annotations
    # is_mode0_major=False means mode1 (K or N) is contiguous = row-major
    mA = _make_cute_tensor(a_torch, is_mode0_major=False)
    mB = _make_cute_tensor(b_torch, is_mode0_major=False)
    mC = _make_cute_tensor(c_torch, is_mode0_major=False)
    mMask = _wrap_mask_tensor(mask_torch)

    gemm = Fp16Matmul()

    print("Compiling kernel...")
    compiled_gemm = cute.compile(gemm, mA, mB, mC, mMask)

    a_2d = a_torch.squeeze(-1)  # (M, K)
    b_2d = b_torch.squeeze(-1)  # (N, K)

    # Reference: sparse blocks accumulate only the first 2 K-tiles
    def _sparse_matmul_ref(a, b, mask):
        c = torch.zeros(M, N, dtype=a.dtype, device=a.device)
        mask_2d = mask.squeeze(0)  # (num_m_tiles, num_n_tiles)
        for m_t in range(num_m_tiles):
            m0, m1 = m_t * bM, min((m_t + 1) * bM, M)
            for n_t in range(num_n_tiles):
                n0, n1 = n_t * bN, min((n_t + 1) * bN, N)
                k_limit = num_k_tiles if mask_2d[m_t, n_t].item() else 2
                for k_t in range(k_limit):
                    k0, k1 = k_t * bK, min((k_t + 1) * bK, K)
                    c[m0:m1, n0:n1] += a[m0:m1, k0:k1] @ b[n0:n1, k0:k1].T
        return c

    if sparse_ratio > 0.0:
        ref = _sparse_matmul_ref(a_2d, b_2d, mask_torch)
    else:
        ref = torch_matmul(a_2d, b_2d)

    # Execute kernel
    print("Executing kernel...")
    compiled_gemm(mA, mB, mC, mMask)
    torch.cuda.synchronize()

    # Verify correctness
    c_result = c_torch.squeeze(-1)  # (M, N)
    print("Verifying results...")
    torch.testing.assert_close(c_result.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
    print("PASS - Results match reference!")

    # Benchmark settings
    warmup = 10
    iters = 100
    flops = 2.0 * M * N * K

    # Benchmark torch matmul
    print("\nBenchmarking torch matmul...")
    with torch.no_grad():
        for _ in range(warmup):
            a_2d.half() @ b_2d.half().T
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = a_2d.half() @ b_2d.half().T
        end.record()
        torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / iters
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12

    # Benchmark cute matmul
    print("Benchmarking cute matmul...")
    for _ in range(warmup):
        compiled_gemm(mA, mB, mC, mMask)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        compiled_gemm(mA, mB, mC, mMask)
    end.record()
    torch.cuda.synchronize()
    cute_ms = start.elapsed_time(end) / iters
    cute_tflops = flops / (cute_ms * 1e-3) / 1e12

    # Comparison
    speedup = torch_ms / cute_ms
    print(f"\n{'='*50}")
    print(f"  Torch:   {torch_ms:.4f} ms  ({torch_tflops:.2f} TFLOPS)")
    print(f"  CuTe:    {cute_ms:.4f} ms  ({cute_tflops:.2f} TFLOPS)")
    print(f"  Speedup: {speedup:.2f}x {'(CuTe faster)' if speedup > 1 else '(Torch faster)'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FP16 Matmul kernel optimized for A100 (SM80)"
    )
    parser.add_argument(
        "--mnk", type=str, default="4096, 1024, 64",
        help="M,N,K dimensions (comma-separated)",
    )
    parser.add_argument(
        "--sparse_ratio", type=float, default=0.0,
        help="Fraction of output blocks that compute only the first 2 K-tiles",
    )
    args = parser.parse_args()
    M, N, K = (int(x.strip()) for x in args.mnk.split(","))
    run(M, N, K, args.sparse_ratio)
