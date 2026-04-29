"""
Benchmark: classic SDPA vs. block-sparse FA2 (no rel-pos, no RoPE).

Targets `flash_attn_block_sparse.py::FlashAttentionForwardAmpereBlockSparse` —
the kernel used by `PiToMe.algo.sparsesam.patch.sam2_hiera`.

Usage:
    python cute/bench_flash_attn_block_sparse.py
    python cute/bench_flash_attn_block_sparse.py --seqlen_q 4096 --num_head 16 --head_dim 64
    python cute/bench_flash_attn_block_sparse.py --sparsity 0.0 0.25 0.5 0.75
"""

import argparse
import math

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from flash_attn_block_sparse import FlashAttentionForwardAmpereBlockSparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkvo(B, S, H, D, torch_dtype, dtype):
    t = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")
    ct = (from_dlpack(t, assumed_align=16)
          .mark_layout_dynamic(leading_dim=3)
          .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                      divisibility=128 // dtype.width))
    return ct, t


def make_o_cute(o_t, dtype):
    return (from_dlpack(o_t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=3, stride_order=o_t.dim_order(),
                                        divisibility=128 // dtype.width))


def make_A_mask(B, H, nm, nn, sparsity):
    """Banded-diagonal + keep-bar block mask matching the sparsesam pattern.

    `sparsity` is the keep-bar fraction stripped from the leading columns;
    sparsity=0 ⇒ fully dense; sparsity=0.9 ⇒ only the diagonal + 10% of n-cols
    are kept active.
    """
    t = torch.zeros(B, H, nm, nn, dtype=torch.int32, device="cuda")
    t = t + torch.eye(nm, nn, dtype=torch.int32, device="cuda")
    keep_cols = int((1 - sparsity) * nn)
    if keep_cols > 0:
        t[:, :, :, :keep_cols] = 1
    ct = from_dlpack(t, assumed_align=4)
    return ct, t


def timeit(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters    # µs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(dtype_str, B, Sq, H, D, scale, m_block, n_block, threads,
                  warmup, iters, verify, sparsities=(0.0,)):
    dtype       = cutlass.dtype(dtype_str)
    torch_dtype = cutlass_torch.dtype(dtype)

    if not FlashAttentionForwardAmpereBlockSparse.can_implement(
        dtype, D, m_block, n_block, threads
    ):
        print(f"[SKIP] {dtype_str} D={D} m={m_block} n={n_block} T={threads}")
        return

    num_m_blocks = math.ceil(Sq / m_block)
    num_n_blocks = math.ceil(Sq / n_block)
    flops = 4 * B * H * Sq * Sq * D                          # 2x(QK + PV)

    print(f"\n{'='*86}")
    print(f"  {dtype_str}  B={B}  Sq={Sq}  H={H}  D={D}  scale={scale:.4f}")
    print(f"  m={m_block}  n={n_block}  threads={threads}  "
          f"blocks=({num_m_blocks}x{num_n_blocks})")
    print(f"{'='*86}")

    q_c, q_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    k_c, k_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    v_c, v_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    o_c, o_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)

    # Identity perms — kernel API accepts them but doesn't index by them.
    perm_q_t = torch.arange(Sq, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    perm_k_t = perm_q_t.clone()
    perm_q_c = from_dlpack(perm_q_t, assumed_align=4)
    perm_k_c = from_dlpack(perm_k_t, assumed_align=4)

    # Compile once with a dense mask; the kernel is shape-specialised.
    dense_mask_c, _ = make_A_mask(B, H, num_m_blocks, num_n_blocks, sparsity=0.0)
    cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        FlashAttentionForwardAmpereBlockSparse(D, m_block, n_block, threads),
        q_c, k_c, v_c, o_c, perm_q_c, perm_k_c,
        dense_mask_c, scale, cu_stream,
    )

    # Dense reference paths (no rel-pos, no RoPE).
    BH = B * H
    q_perm = q_t.permute(0, 2, 1, 3).contiguous()              # (B, H, Sq, D)
    k_perm = k_t.permute(0, 2, 1, 3).contiguous()
    v_perm = v_t.permute(0, 2, 1, 3).contiguous()

    def torch_sdpa():
        F.scaled_dot_product_attention(
            q_perm, k_perm, v_perm,
            attn_mask=None, is_causal=False, scale=scale,
        )

    def naive_attn():
        scores = (q_perm @ k_perm.transpose(-2, -1)) * scale
        return F.softmax(scores, dim=-1) @ v_perm

    t_naive = timeit(naive_attn, warmup, iters)
    t_torch = timeit(torch_sdpa, warmup, iters)
    print(f"\n  Dense baselines:")
    print(f"    naive (Q@Kᵀ + softmax + @V): {t_naive:>8.1f} us  "
          f"({flops/(t_naive*1e-6)/1e12:>5.2f} TFLOP/s)")
    print(f"    F.scaled_dot_product_attn  : {t_torch:>8.1f} us  "
          f"({flops/(t_torch*1e-6)/1e12:>5.2f} TFLOP/s)\n")

    print(f"  {'Sparsity':>9} | {'fa2 us':>9} | {'TFLOP/s':>8} | "
          f"{'vs naive':>9} | {'vs torch':>9} | {'verify':>7}")
    print(f"  {'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")

    for sp in sparsities:
        bmask_c, bmask_t = make_A_mask(B, H, num_m_blocks, num_n_blocks, sp)

        # ── correctness ────────────────────────────────────────────────────
        status = "-"
        if verify:
            compiled(q_c, k_c, v_c, o_c,
                     perm_q_c, perm_k_c, bmask_c, scale, cu_stream)
            torch.cuda.synchronize()

            q_ref = q_perm.float()
            k_ref = k_perm.float()
            v_ref = v_perm.float()
            scores = q_ref @ k_ref.transpose(-2, -1) * scale

            if sp > 0.0:
                # Expand block mask to per-token mask and apply -inf to disallowed pairs.
                mask_full = (
                    bmask_t.float()
                    .repeat_interleave(m_block, dim=2)
                    .repeat_interleave(n_block, dim=3)
                    [:, :, :Sq, :Sq].bool()
                )
                scores = scores.masked_fill(~mask_full, float("-inf"))

            # Fully-masked rows produce NaN in F.softmax(-inf,...) but the kernel
            # writes 0 for them — both are valid sentinels.
            ref = (F.softmax(scores, dim=-1).nan_to_num(0.0)
                   .matmul(v_ref)
                   .permute(0, 2, 1, 3)
                   .to(torch_dtype))

            max_diff = (o_t.float().cpu() - ref.float().cpu()).abs().max().item()
            status = "PASS" if max_diff <= 1e-2 else f"FAIL(d{max_diff:.3f})"

        def fa2_kernel():
            compiled(q_c, k_c, v_c, make_o_cute(o_t, dtype),
                     perm_q_c, perm_k_c, bmask_c, scale, cu_stream)

        t_fa2  = timeit(fa2_kernel, warmup, iters)
        tflops = flops / (t_fa2 * 1e-6) / 1e12
        spd_naive = t_naive / t_fa2
        spd_torch = t_torch / t_fa2

        print(f"  {sp:>9.2f} | {t_fa2:>9.1f} | {tflops:>8.2f} | "
              f"{spd_naive:>8.2f}x | {spd_torch:>8.2f}x | {status:>7}")

    print()


# ---------------------------------------------------------------------------
# SAM2 preset (Hiera global-attention seq sizes, default head_dim=64 / nH=16)
# ---------------------------------------------------------------------------

SAM2_CONFIGS = {
    # (Sq, num_heads, head_dim) — SAM2 Hiera-L global block at stage 3 sees
    # H*W ≈ 64*64 = 4096 tokens with 16 heads × head_dim 72.
    "sam2_l": (4096, 8, 72),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default=None, choices=list(SAM2_CONFIGS))
    p.add_argument("--dtype",         default="BFloat16")
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--seqlen_q",      type=int,   default=None)
    p.add_argument("--num_head",      type=int,   default=8)
    p.add_argument("--head_dim",      type=int,   default=None)
    p.add_argument("--softmax_scale", type=float, default=None)
    p.add_argument("--m_block_size",  type=int,   default=64)
    p.add_argument("--n_block_size",  type=int,   default=64)
    p.add_argument("--num_threads",   type=int,   default=128)
    p.add_argument("--warmup",        type=int,   default=5)
    p.add_argument("--iters",         type=int,   default=20)
    p.add_argument("--no_verify",     action="store_true")
    p.add_argument("--sparsity",      type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 0.9],
                   metavar="S",
                   help="Sparsity levels to sweep (fraction of n-cols zeroed).")
    args = p.parse_args()

    if args.model:
        configs = [SAM2_CONFIGS[args.model]]
    elif args.seqlen_q:
        configs = [(args.seqlen_q, args.num_head or 16, args.head_dim or 64)]
    else:
        configs = [SAM2_CONFIGS["sam2_l"]]

    for sq, nh, hd in configs:
        run_benchmark(
            dtype_str  = args.dtype,
            B          = args.batch_size,
            Sq         = sq, H = nh, D = hd,
            scale      = args.softmax_scale or 1.0 / math.sqrt(hd),
            m_block    = args.m_block_size,
            n_block    = args.n_block_size,
            threads    = args.num_threads,
            warmup     = args.warmup,
            iters      = args.iters,
            verify     = not args.no_verify,
            sparsities = args.sparsity,
        )


if __name__ == "__main__":
    main()
