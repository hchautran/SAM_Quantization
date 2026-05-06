"""
Flash Attention v2 forward pass for SAM3 with **fused 2D axial RoPE**.

This is a stripped-down fork of `flash_attn.py`:

  REMOVED (compared with the SAM1 variant)
    • m_rel_H / m_rel_W learnable bias inputs
    • sRelH / sRelW shared-memory buffers
    • rRelH / rRelW per-thread fragments
    • the cooperative GMEM→SMEM rel_pos load
    • the cooperative SMEM→RMEM rRelH/rRelW fill
    • the `add_rel_pos_bias` method and the `inv_softmax_scale` factor
    • m_perm_Q / m_perm_K (no token reordering)
    • the constructor's `win_shape` argument

  ADDED
    • m_cos / m_sin GMEM tensors of shape (Sq, head_dim), shared across
      batch and heads (RoPE depends only on token position).
    • sCosQ / sSinQ shared-memory buffers of shape
      (m_block_size, head_dim_padded), populated once per CTA.
    • sCosK / sSinK shared-memory buffers of shape
      (n_block_size, head_dim_padded), refreshed per K tile.
    • Two cooperative GMEM→SMEM cp.async loads (one for Q's cos/sin, one
      per K tile).
    • Two in-SMEM rotation passes — `_rotate_smem_inplace` — that update
      sQ and sK in place using the pairwise rotation
        x[2i  ] ← x[2i  ]·cos_i − x[2i+1]·sin_i
        x[2i+1] ← x[2i+1]·cos_i + x[2i  ]·sin_i

The rest of the kernel (online softmax, P·V MMA, normalize, output store)
is unchanged.

Tensors: Q/K/V/O are (B, S, N, D) — same as flash_attn.py.
cos/sin are (S, D) — broadcast over batch and heads.
"""

from types import SimpleNamespace
from typing import Type, Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.pipeline as pipeline
import cutlass.utils as utils


class FlashAttentionForwardAmpereRoPE:
    """FA2 + fused 2D-axial RoPE.  No rel-pos bias, no token permutations."""

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
    ):
        self._head_dim       = head_dim
        self._m_block_size   = m_block_size
        self._n_block_size   = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads    = num_threads

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads) -> bool:
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim % 4 != 0:
            return False  # axial RoPE requires D % 4 == 0
        if num_threads % 32 != 0:
            return False
        head_dim_padded = (head_dim + 31) // 32 * 32
        # SMEM: Q + 2·KV (Q overlaps with O on epilogue) + cos/sin for Q + cos/sin for K
        smem_q  = m_block_size * head_dim_padded * 2
        smem_kv = n_block_size * head_dim_padded * 2 * 2
        smem_cs_q = m_block_size * head_dim_padded * 2 * 2
        smem_cs_k = n_block_size * head_dim_padded * 2 * 2
        if smem_q + smem_kv + smem_cs_q + smem_cs_k > utils.get_smem_capacity_in_bytes("sm_80"):
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        return True

    # ────────────────────────────────────────────────────────────────────────
    # Host-side launcher
    # ────────────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_cos: cute.Tensor,        # (Sq, D), float32 broadcast over (B, H)
        m_sin: cute.Tensor,        # (Sq, D)
        m_block_mask: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (mQ.element_type == cutlass.Float16 or mQ.element_type == cutlass.BFloat16)
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1)
        )
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1)
        )
        sO_layout = sQ_layout

        # cos/sin SMEM uses simple row-major layout — read/written by element,
        # not via ldmatrix / cp.async-128 in the rotation pass.
        sCosQ_layout = cute.make_layout(
            (self._m_block_size, self._head_dim_padded),
            stride=(self._head_dim_padded, 1),
        )
        sCosK_layout = cute.make_layout(
            (self._n_block_size, self._head_dim_padded),
            stride=(self._head_dim_padded, 1),
        )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sCosQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._m_block_size * self._head_dim_padded], 128
            ]
            sSinQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._m_block_size * self._head_dim_padded], 128
            ]
            sCosK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._n_block_size * self._head_dim_padded], 128
            ]
            sSinK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._n_block_size * self._head_dim_padded], 128
            ]

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))

        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async_copy, tQKV_layout, vQKV_layout)
        gmem_tiled_copy_O   = cute.make_tiled_copy_tv(atom_universal_copy, tQKV_layout, vQKV_layout)

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        softmax_scale_log2 = softmax_scale * 1.4426950408889634074

        self.kernel(
            mQ, mK, mV, mO, m_cos, m_sin, m_block_mask,
            softmax_scale_log2,
            sQ_layout, sKV_layout, sO_layout,
            sCosQ_layout, sCosK_layout,
            gmem_tiled_copy_QKV, gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Device kernel
    # ────────────────────────────────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        m_cos: cute.Tensor,
        m_sin: cute.Tensor,
        m_block_mask: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sCosQ_layout: cute.Layout,
        sCosK_layout: cute.Layout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        n_block = n_block_max - 1

        gQ = cute.local_tile(
            mQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        gK = cute.local_tile(
            mK[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        # cos/sin live in GMEM as (Sq, D) — accessed directly by element from
        # the rotation pass (no per-CTA tile descriptors needed since
        # _load_cos_sin_K addresses m_cos / m_sin by global indices).

        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ      = storage.sQ.get_tensor(sQ_layout)
        sK      = storage.sK.get_tensor(sKV_layout)
        sV      = storage.sV.get_tensor(sKV_layout)
        sCosQ   = storage.sCosQ.get_tensor(sCosQ_layout)
        sSinQ   = storage.sSinQ.get_tensor(sCosQ_layout)
        sCosK   = storage.sCosK.get_tensor(sCosK_layout)
        sSinK   = storage.sSinK.get_tensor(sCosK_layout)

        # Transposed V view for O = P·V MMA
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        # Per-thread GMEM copy partitions
        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        # MMA register fragments
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ  = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK  = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        acc_O = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block_size, self._head_dim_padded)),
            cutlass.Float32,
        )
        acc_O.fill(0.0)

        # SMEM→RMEM ldmatrix atoms
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )

        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # head_dim predicate tensors (seqlen handled per tile below)
        mcQ  = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )
        tQcQ   = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)

        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (tKsK.shape[0][1], cute.size(tKsK, mode=[1]), cute.size(tKsK, mode=[2])),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )

        # Online softmax state
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)
        # ────────────────────────────────────────────────────────────────────
        # Prologue: load Q + Q's cos/sin, then rotate Q in SMEM in-place.
        # ────────────────────────────────────────────────────────────────────
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                tQsQ[None, m, None].fill(0)

        # Cooperative element-wise GMEM→SMEM load of cos/sin for Q's positions.
        # Range: m_block * m_block_size  ..  (m_block+1) * m_block_size − 1
        # We use a flat thread loop because cp.async-128 isn't worthwhile on
        # the rotation hot path (rotation is purely SMEM-resident).
        n_cs_q = self._m_block_size * self._head_dim_padded
        for j in cutlass.range_constexpr(cute.ceil_div(n_cs_q, self._num_threads)):
            flat = tidx + j * self._num_threads
            if cute.elem_less(flat, n_cs_q):
                m_local = flat // self._head_dim_padded
                d_idx   = flat %  self._head_dim_padded
                m_global = m_block * self._m_block_size + m_local
                if cute.elem_less(m_global, mQ.shape[1]) and cute.elem_less(d_idx, mQ.layout.shape[3]):
                    sCosQ[m_local, d_idx] = m_cos[m_global, d_idx].to(self._dtype)
                    sSinQ[m_local, d_idx] = m_sin[m_global, d_idx].to(self._dtype)
                else:
                    sCosQ[m_local, d_idx] = self._dtype(1.0)
                    sSinQ[m_local, d_idx] = self._dtype(0.0)

        # K load for the first (right-most) n_block
        last_block_enabled = m_block_mask[batch_size, num_head, m_block, n_block]
        if last_block_enabled:
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, n, None, n_block],
                        tKsK[None, n, None],
                        pred=tKVpKV[None, n, None],
                    )
                else:
                    tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Now sQ and sCosQ/sSinQ are valid — rotate sQ in SMEM in place.
        self._rotate_smem_inplace(sQ, sCosQ, sSinQ,
                                  rows=self._m_block_size,
                                  d_pad=self._head_dim_padded,
                                  d_real=mQ.layout.shape[3])

        # Load and rotate K for the first n-block.
        if last_block_enabled:
            self._load_cos_sin_K(mK, m_cos, m_sin, sCosK, sSinK, n_block)
            self.cta_sync_barrier.arrive_and_wait()
            self._rotate_smem_inplace(sK, sCosK, sSinK,
                                      rows=self._n_block_size,
                                      d_pad=self._head_dim_padded,
                                      d_real=mK.layout.shape[3])
        # Final sync so all threads see rotated Q/K before MMA
        self.cta_sync_barrier.arrive_and_wait()

        # First n-block (with seqlen-K residue mask)
        self.compute_one_n_block(
            mQ, mK, m_cos, m_sin, m_block_mask, sK, sCosK, sSinK, thr_mma, tiled_mma, tSrQ, tSrK, tOrVt, acc_O, gmem_tiled_copy_QKV, tKVcKV, tKgK, tKsK, tVgV, tVsV, tKVpKV, smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V, tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view, tOsVt, tOrVt_copy_view, row_max, row_sum, softmax_scale_log2,
            n_block_idx=n_block_max - 1,
            is_first_n_block=True, in_mask_steps=True,
        )

        # Remaining n-blocks
        for n_tile in range(1, n_block_max, 1):
            n_block_cur = n_block_max - n_tile - 1
            self.compute_one_n_block(
                mQ, mK, m_cos, m_sin, m_block_mask, sK, sCosK, sSinK, thr_mma, tiled_mma, tSrQ, tSrK, tOrVt, acc_O, gmem_tiled_copy_QKV, tKVcKV, tKgK, tKsK, tVgV, tVsV, tKVpKV, smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V, tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view, tOsVt, tOrVt_copy_view, row_max, row_sum, softmax_scale_log2,
                n_block_idx=n_block_cur,
                is_first_n_block=False, in_mask_steps=False,
            )

        # ────────────────────────────────────────────────────────────────────
        # Epilogue: normalize O, store
        # ────────────────────────────────────────────────────────────────────
        self.normalize_softmax(acc_O, row_sum)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))

        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O  = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O   = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        gO = cute.local_tile(
            mO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)

        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO  = cute.local_tile(
            mcO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    # ────────────────────────────────────────────────────────────────────────
    # Helpers: fused RoPE rotation in SMEM (in-place)
    # ────────────────────────────────────────────────────────────────────────

    @cute.jit
    def _rotate_smem_inplace(
        self,
        s_x:    cute.Tensor,      # (rows, d_pad), swizzled
        s_cos:  cute.Tensor,      # (rows, d_pad)
        s_sin:  cute.Tensor,      # (rows, d_pad)
        rows:   cutlass.Constexpr,
        d_pad:  cutlass.Constexpr,
        d_real: cutlass.Int32,
    ):
        """Apply pairwise RoPE rotation to a (rows, d) SMEM tile in-place.

            x[r, 2i  ] ← x[r, 2i  ]·cos − x[r, 2i+1]·sin
            x[r, 2i+1] ← x[r, 2i+1]·cos + x[r, 2i  ]·sin

        cos[r, 2i] == cos[r, 2i+1] (axial RoPE uses repeat_interleave(2)),
        same for sin — so we read one (cos, sin) per pair.
        """
        tidx, _, _ = cute.arch.thread_idx()
        n_pairs = rows * (d_pad // 2)
        for j in cutlass.range_constexpr(cute.ceil_div(n_pairs, self._num_threads)):
            flat = tidx + j * self._num_threads
            if cute.elem_less(flat, n_pairs):
                pairs_per_row = d_pad // 2
                r  = flat // pairs_per_row
                p  = flat %  pairs_per_row
                k0 = 2 * p
                k1 = k0 + 1
                if cute.elem_less(k1, d_real):
                    a = s_x[r, k0].to(cutlass.Float32)
                    b = s_x[r, k1].to(cutlass.Float32)
                    c = s_cos[r, k0].to(cutlass.Float32)
                    s = s_sin[r, k0].to(cutlass.Float32)
                    s_x[r, k0] = (a * c - b * s).to(self._dtype)
                    s_x[r, k1] = (b * c + a * s).to(self._dtype)
                # else: leave padded lanes untouched (they were zeroed on load)

    @cute.jit
    def _load_cos_sin_K(self, mK, mCos, mSin, sCosK, sSinK,
        n_block_idx:  cutlass.Int32,
    ):
        """Cooperative GMEM→SMEM load of cos/sin for K's positions in this n_block."""
        tidx, _, _ = cute.arch.thread_idx()
        rows  = self._n_block_size
        d_pad = self._head_dim_padded
        n_cs  = rows * d_pad
        for j in cutlass.range_constexpr(cute.ceil_div(n_cs, self._num_threads)):
            flat = tidx + j * self._num_threads
            if cute.elem_less(flat, n_cs):
                n_local = flat // d_pad
                d_idx   = flat %  d_pad
                n_global = n_block_idx * rows + n_local
                if cute.elem_less(n_global, mK.shape[1]) and cute.elem_less(
                    d_idx, mK.layout.shape[3]
                ):
                    sCosK[n_local, d_idx] = mCos[n_global, d_idx].to(self._dtype)
                    sSinK[n_local, d_idx] = mSin[n_global, d_idx].to(self._dtype)
                else:
                    sCosK[n_local, d_idx] = self._dtype(1.0)
                    sSinK[n_local, d_idx] = self._dtype(0.0)

    # ────────────────────────────────────────────────────────────────────────
    # Per-n-block compute (no rel_pos, no token perms)
    # ────────────────────────────────────────────────────────────────────────

    @cute.jit
    def compute_one_n_block(self, mQ, mK, mCos, mSin, m_block_mask, sK, sCosK, sSinK, thr_mma, tiled_mma, tSrQ, tSrK, tOrVt, acc_O, gmem_tiled_copy_QKV, tKVcKV, tKgK, tKsK, tVgV, tVsV, tKVpKV, smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V, tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view, tOsVt, tOrVt_copy_view, row_max, row_sum, softmax_scale_log2,
        n_block_idx: cutlass.Int32,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        acc_S = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)),
            cutlass.Float32,
        )

        # Wait for the K tile prefetched by the previous iteration (or by the
        # prologue, on the first n-block).
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        m_block, batch_size, num_head = cute.arch.block_idx()

        block_enabled = m_block_mask[
            batch_size, num_head,
            m_block, n_block_idx,
        ]

        # Rotate sK in place using cos/sin for THIS n-block's positions.
        # On the first n-block this rotation was already done in the prologue.
        if cutlass.const_expr(not is_first_n_block):
            if block_enabled:
                self._load_cos_sin_K(mK, mCos, mSin, sCosK, sSinK, n_block_idx)
                self.cta_sync_barrier.arrive_and_wait()
                self._rotate_smem_inplace(
                    sK, sCosK, sSinK,
                    rows=self._n_block_size,
                    d_pad=self._head_dim_padded,
                    d_real=mK.layout.shape[3],
                )
                self.cta_sync_barrier.arrive_and_wait()

        # V load for the current n-block
        if block_enabled:
            if is_first_n_block:
                for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                    if cute.elem_less(
                        tKVcKV[0, n, 0][1],
                        mK.layout.shape[1],
                    ):
                        cute.copy(
                            gmem_tiled_copy_QKV,
                            tVgV[None, n, None, n_block_idx],
                            tVsV[None, n, None],
                            pred=tKVpKV[None, n, None],
                        )
                    else:
                        tVsV[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tVgV[None, None, None, n_block_idx],
                    tVsV,
                    pred=tKVpKV,
                )

        cute.arch.cp_async_commit_group()

        # S = Q · K^T (Q,K already RoPE-rotated in SMEM)
        if block_enabled:
            acc_S.fill(0.0)
            cute.copy(
                smem_tiled_copy_Q,
                tSsQ[None, None, 0],
                tSrQ_copy_view[None, None, 0],
            )
            cute.copy(
                smem_tiled_copy_K,
                tSsK[None, None, 0],
                tSrK_copy_view[None, None, 0],
            )
            for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                k_next = (k + 1) % cute.size(tSsQ.shape[2])
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k_next],
                    tSrQ_copy_view[None, None, k_next],
                )
                cute.copy(
                    smem_tiled_copy_K,
                    tSsK[None, None, k_next],
                    tSrK_copy_view[None, None, k_next],
                )
                cute.gemm(
                    tiled_mma, acc_S,
                    tSrQ[None, None, k],
                    tSrK[None, None, k],
                    acc_S,
                )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # Prefetch K for the next n-block.  Rotation of this prefetched tile
        # happens at the TOP of the next compute_one_n_block call, so that
        # the cp_async load can overlap with the softmax+PV work below.
        if n_block_idx > 0:
            next_block_enabled = m_block_mask[
                batch_size, num_head,
                m_block, n_block_idx - 1,
            ]
            if next_block_enabled:
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, None, None, n_block_idx - 1],
                    tKsK,
                    pred=tKVpKV,
                )

            cute.arch.cp_async_commit_group()

        # Softmax + acc_O update
        if block_enabled:
            self.softmax_rescale_O(
                mQ, mK, m_block_mask, acc_O, thr_mma, row_max, row_sum, softmax_scale_log2,
                acc_S, is_first_n_block, in_mask_steps,
                n_tile_size=self._n_block_size,
                n_tile_coord=n_block_idx,
            )
            rP = cute.make_fragment_like(acc_S, self._dtype)
            rP.store(acc_S.load().to(self._dtype))
            rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                (
                    (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                    rP_layout_divided.shape[1],
                    rP_layout_divided.shape[2][1],
                ),
                stride=(
                    (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                    rP_layout_divided.stride[1],
                    rP_layout_divided.stride[2][1],
                ),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
            cute.copy(
                smem_tiled_copy_V,
                tOsVt[None, None, 0],
                tOrVt_copy_view[None, None, 0],
            )
            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                k_next = (k + 1) % cute.size(tOrS.shape[2])
                cute.copy(
                    smem_tiled_copy_V,
                    tOsVt[None, None, k_next],
                    tOrVt_copy_view[None, None, k_next],
                )
                cute.gemm(
                    tiled_mma, acc_O,
                    tOrS[None, None, k],
                    tOrVt[None, None, k],
                    acc_O,
                )

    # ────────────────────────────────────────────────────────────────────────
    # Softmax / normalize (unchanged)
    # ────────────────────────────────────────────────────────────────────────

    @cute.jit
    def softmax_rescale_O(self, mQ, mK, m_block_mask, acc_O, thr_mma, row_max, row_sum, softmax_scale_log2,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
        n_tile_size: cutlass.Constexpr,
        n_tile_coord: cutlass.Int32,
    ):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(row_max, cutlass.Float32)
            cute.basic_copy(row_max, row_max_prev)

        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor((
                mQ.shape[0], mQ.shape[1],
                mQ.shape[2], mK.shape[1],
            ))
            m_block, batch_size, num_head = cute.arch.block_idx()
            cS = cute.local_tile(
                mcS[batch_size, None, num_head, None],
                (self._m_block_size, n_tile_size),
                (m_block, n_tile_coord),
            )
            tScS_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cS))

        for r in cutlass.range_constexpr(cute.size(row_max)):
            if cutlass.const_expr(in_mask_steps):
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(mK.shape[1], tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = self._threadquad_reduce_max(
                acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            )

            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row  = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            else:
                row_max_cur_row = (
                    0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                )

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_scale_log2
                - row_max_cur_row * softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)

            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_scale_log2
                    - row_max_cur_row * softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum     = acc_S_row_sum + row_sum[r] * prev_minus_cur_exp
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp

            row_max[r] = row_max_cur_row
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None]         = acc_S_row_exp

    @cute.jit
    def normalize_softmax(self, acc_O: cute.Tensor, row_sum: cute.Tensor):
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = 1.0 if is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        s = cute.make_layout(acc.layout.shape)
        mn_layout = cute.make_layout(
            ((s.shape[0][1], s.shape[1]), (s.shape[0][0], s.shape[2])),
            stride=((s.stride[0][1], s.stride[1]), (s.stride[0][0], s.stride[2])),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn_layout))

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)
