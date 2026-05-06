"""
Benchmark: SAM3 RoPE attention (naive) vs. FA2 with **fused** 2D-axial RoPE.

Drives `FlashAttentionForwardAmpereRoPE` from flash_attn_rope_fused.py:
the kernel takes raw (un-rotated) Q/K and applies RoPE in-SMEM in a fused
rotation pass.  No rel_pos bias, no token permutations.

Usage:
    python cute/bench_flash_attn_rope.py
    python cute/bench_flash_attn_rope.py --model sam3_window --batch_size 25
    python cute/bench_flash_attn_rope.py --model sam3_global --batch_size 1
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

from flash_attn_rope_fused import FlashAttentionForwardAmpereRoPE
from flash_attn_rope import build_rope_2d, apply_rope_bshd
from transformers.models.sam3.modeling_sam3 import  Sam3ViTRoPEAttention 


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


def make_A_mask(B, H, nm, nn, sparsity):
    """Block mask: identity + dense prefix, 1=compute, 0=skip."""
    t = torch.zeros(B, H, nm, nn, dtype=torch.int32, device="cuda")
    t = t + torch.eye(nm, nn, dtype=torch.int32, device="cuda")
    t[:, :, :, :int((1 - sparsity) * nn)] = 1
    ct = from_dlpack(t, assumed_align=4)
    return ct, t


def make_o_cute(o_t, dtype):
    return (from_dlpack(o_t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=3, stride_order=o_t.dim_order(),
                                        divisibility=128 // dtype.width))


def make_cos_sin_cute(t):
    """cos/sin are (Sq, D) float32 — pass as plain dlpack tensors."""
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def timeit(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters   # µs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(dtype_str, B, end_x, end_y, H, D, scale,
                  m_block, n_block, threads,
                  warmup, iters, verify, sparsities=(0.0,)):
    """
    end_x, end_y : 2D spatial extent (Sq = end_x * end_y).
                   SAM3 windowed: end_x = end_y = 24
                   SAM3 global  : end_x = end_y = 64 (1024px / 16px patch)
    """
    Sq = end_x * end_y
    dtype       = cutlass.dtype(dtype_str)
    torch_dtype = cutlass_torch.dtype(dtype)

    if not FlashAttentionForwardAmpereRoPE.can_implement(
        dtype, D, m_block, n_block, threads
    ):
        print(f"[SKIP] {dtype_str} D={D} m={m_block} n={n_block} T={threads}")
        return

    num_m_blocks = math.ceil(Sq / m_block)
    num_n_blocks = math.ceil(Sq / n_block)
    flops = 4 * B * H * Sq * Sq * D   # 2x(QK + PV)

    print(f"\n{'='*84}")
    print(f"  {dtype_str}  B={B}  Sq={Sq} ({end_x}x{end_y})  H={H}  D={D}  scale={scale:.4f}")
    print(f"  m={m_block}  n={n_block}  threads={threads}  "
          f"blocks=({num_m_blocks}x{num_n_blocks})  [fused RoPE]")
    print(f"{'='*84}")

    # ── RoPE tables (Sq, D) — fp32, shared across batch and heads ────────────
    cos_t, sin_t = build_rope_2d(end_x, end_y, D, theta=10000.0, scale=1.0,
                                 device="cuda", dtype=torch.float32)
    cos_c = make_cos_sin_cute(cos_t)
    sin_c = make_cos_sin_cute(sin_t)

    # ── Q/K/V/O: kernel takes RAW Q/K (rotation happens in-kernel) ───────────
    q_c, q_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    k_c, k_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    v_c, v_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    o_c, o_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)

    # ── Compile ──────────────────────────────────────────────────────────────
    dense_mask_c, _ = make_A_mask(B, H, num_m_blocks, num_n_blocks, sparsity=0.0)
    cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        FlashAttentionForwardAmpereRoPE(D, m_block, n_block, threads),
        q_c, k_c, v_c, o_c, cos_c, sin_c,
        dense_mask_c, scale, cu_stream,
        options="",
    )

    # ── Naive SAM3 baseline: rotate(Q,K) then SDPA ───────────────────────────
    BH = B * H
    cos_b = cos_t.view(1, Sq, 1, D)
    sin_b = sin_t.view(1, Sq, 1, D)

    def _rotate_pairwise(x):
        x_ = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x_.unbind(-1)
        return torch.stack((-x2, x1), -1).flatten(-2)

    def sam3_naive():
        q_f = q_t.float()
        k_f = k_t.float()
        qr = (q_f * cos_b + _rotate_pairwise(q_f) * sin_b).to(v_t.dtype)
        kr = (k_f * cos_b + _rotate_pairwise(k_f) * sin_b).to(v_t.dtype)
        qm = qr.permute(0, 2, 1, 3).reshape(BH, Sq, D)
        km = kr.permute(0, 2, 1, 3).reshape(BH, Sq, D)
        vm = v_t.permute(0, 2, 1, 3).reshape(BH, Sq, D)
        attn = (qm @ km.transpose(-2, -1)) * scale
        return F.softmax(attn, dim=-1) @ vm

    t_sam = timeit(sam3_naive, warmup, iters)
    print(f"\n  SAM3 naive (RoPE + dense): {t_sam:.1f} us  "
          f"({flops/(t_sam*1e-6)/1e12:.2f} TFLOP/s)\n")

    print(f"  {'Sparsity':>9} | {'fa2-fused us':>13} | {'TFLOP/s':>8} | "
          f"{'vs SAM3 (naive)':>16} | {'vs SAM3':>9} | {'verify':>7}")
    print(f"  {'-'*9}-+-{'-'*13}-+-{'-'*8}-+-{'-'*16}-+-{'-'*9}-+-{'-'*7}")

    # ── SAM3's attention implementation ───────────────────────────
    from transformers.models.sam3.configuration_sam3 import Sam3ViTConfig
    sam3_config = Sam3ViTConfig(hidden_size=H*D, num_attention_heads=H, attention_dropout=0.0)
    sam3_config._attn_implementation = "sdpa"
    sam3_attn = Sam3ViTRoPEAttention(sam3_config).to("cuda", dtype=torch_dtype)
    sam3_attn.eval() # for stable benchmarking
    hidden_states = torch.randn(B, end_y, end_x, H*D, dtype=torch_dtype, device="cuda")
    
    for sp in sparsities:
        bmask_c, bmask_t = make_A_mask(B, H, num_m_blocks, num_n_blocks, sp)

        # ── Verify ────────────────────────────────────────────────────────────
        status = "-"
        if verify:
            compiled(q_c, k_c, v_c, o_c, cos_c, sin_c,
                     bmask_c, scale, cu_stream)
            torch.cuda.synchronize()

            # Reference: rotate Q/K, then attention with block mask
            q_rot, k_rot = apply_rope_bshd(q_t, k_t, cos_t, sin_t)
            q_ref = q_rot.permute(0, 2, 1, 3).float()
            k_ref = k_rot.permute(0, 2, 1, 3).float()
            v_ref = v_t.permute(0, 2, 1, 3).float()
            scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale

            if sp > 0.0:
                mask_full = (
                    bmask_t.float()
                    .repeat_interleave(m_block, dim=2)
                    .repeat_interleave(n_block, dim=3)
                    [:, :, :Sq, :Sq].bool()
                )
                scores = scores.masked_fill(~mask_full, float("-inf"))

            ref = (F.softmax(scores, dim=-1).nan_to_num(0.0)
                   .matmul(v_ref)
                   .permute(0, 2, 1, 3)
                   .to(torch_dtype))

            max_diff = (o_t.float().cpu() - ref.float().cpu()).abs().max().item()
            status = "PASS" if max_diff <= 1e-2 else f"FAIL(d{max_diff:.3f})"

        # ── FA2 fused-RoPE kernel ────────────────────────────────────────────
        def fa2_fused():
            
            compiled(q_c, k_c, v_c, make_o_cute(o_t, dtype),
                     cos_c, sin_c, bmask_c, scale, cu_stream)

        def run_sam3_attn():
            sam3_attn(hidden_states, position_embeddings=(cos_t, sin_t))

        t_fused = timeit(fa2_fused,  warmup, iters)
        t_sam3 = timeit(run_sam3_attn, warmup, iters)
        tflops = flops / (t_fused * 1e-6) / 1e12
        speedup_naive = t_sam / t_fused
        speedup_sam3 = t_sam3 / t_fused

        print(f"  {sp:>9.2f} | {t_fused:>13.1f} | {tflops:>8.2f} | "
              f"{speedup_naive:>15.2f}x | {speedup_sam3:>8.2f}x | {status:>7}")

    print()


# ---------------------------------------------------------------------------
# SAM3 presets
# ---------------------------------------------------------------------------
SAM3_CONFIGS = {
    # name:           (end_x, end_y, num_heads, head_dim)
    "sam3_window":    (24, 24, 16, 64),    # windowed layer:  Sq = 576
    # "sam3_global":    (72, 72, 16, 64),    # global layer:    Sq = 4096
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default=None, choices=list(SAM3_CONFIGS))
    p.add_argument("--dtype",         default="BFloat16")
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--end_x",         type=int,   default=None)
    p.add_argument("--end_y",         type=int,   default=None)
    p.add_argument("--num_head",      type=int,   default=None)
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
                   help="sparsity levels to sweep")
    args = p.parse_args()

    if args.model:
        configs = [SAM3_CONFIGS[args.model]]
    elif args.end_x and args.end_y:
        configs = [(args.end_x, args.end_y, args.num_head or 16, args.head_dim or 72)]
    else:
        configs = [SAM3_CONFIGS["sam3_window"]]

    for ex, ey, nh, hd in configs:
        run_benchmark(
            dtype_str  = args.dtype,
            B          = args.batch_size,
            end_x      = ex, end_y = ey,
            H          = nh, D = hd,
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
