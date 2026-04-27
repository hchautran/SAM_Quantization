"""
Benchmark: SAM attention (naive) vs. FA2 with decomposed rel_pos bias.

Usage:
    python cute/bench_flash_attn.py
    python cute/bench_flash_attn.py --model sam_l --batch_size 25
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

from flash_attn_rel_pos import FlashAttentionForwardAmpere


# ---------------------------------------------------------------------------
# SAM rel_pos helpers
# ---------------------------------------------------------------------------

def get_rel_pos(q_size, k_size, rel_pos):
    """rel_pos param (2*win-1, D) → (q_size, k_size, D) matrix."""
    max_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_dist:
        rel_pos = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_dist, mode="linear",
        ).reshape(-1, max_dist).permute(1, 0)
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    return rel_pos[(q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)).long()]


def compute_rel_bias(q_bshd, Rh, Rw, win):
    """SAM einsum: (B,Sq,H,D) + Rh/Rw → rel_h, rel_w both (B,Sq,H,win) for FA2."""
    B, Sq, H, D = q_bshd.shape
    r_q   = q_bshd.permute(0, 2, 1, 3).reshape(B * H, win, win, D).float()
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B * H, Sq, win)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B * H, Sq, win)
    to_fa2 = lambda t: t.reshape(B, H, Sq, win).permute(0, 2, 1, 3).contiguous()
    return to_fa2(rel_h), to_fa2(rel_w)


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


def make_block_mask(B, H, nm, nn, sparsity):
    """(B, H, nm, nn) int32 block mask; 1=compute, 0=skip."""
    t = torch.ones(B, H, nm, nn, dtype=torch.int32, device="cuda")
    if sparsity > 0.0:
        n_zero = int(sparsity * nm * nn)
        idx = torch.randperm(nm * nn, device="cuda")[:n_zero]
        t.view(B, H, -1)[:, :, idx] = 0
    ct = from_dlpack(t, assumed_align=4)
    return ct, t


def make_A_mask(B, H, nm, nn, sparsity):
    t = torch.zeros(B, H, nm, nn, dtype=torch.int32, device="cuda")
    t = t + torch.eye(nm, nn, dtype=torch.int32, device="cuda")
    t[:, :, :, :int((1-sparsity) * nn)] = 1
    ct = from_dlpack(t, assumed_align=4)
    return ct, t

def to_cute_pos(t):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)


def make_o_cute(o_t, dtype):
    return (from_dlpack(o_t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=3, stride_order=o_t.dim_order(),
                                        divisibility=128 // dtype.width))


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

def run_benchmark(dtype_str, B, Sq, H, D, scale, m_block, n_block, threads,
                  warmup, iters, verify, sparsities=(0.0,)):
    dtype       = cutlass.dtype(dtype_str)
    torch_dtype = cutlass_torch.dtype(dtype)

    if not FlashAttentionForwardAmpere.can_implement(dtype, D, m_block, n_block, threads):
        print(f"[SKIP] {dtype_str} D={D} m={m_block} n={n_block} T={threads}")
        return

    win = int(math.sqrt(Sq))
    assert win * win == Sq, f"Sq={Sq} must be a perfect square"

    num_m_blocks = math.ceil(Sq / m_block)
    num_n_blocks = math.ceil(Sq / n_block)
    flops = 4 * B * H * Sq * Sq * D   # 2x(QK + PV)
    print('num m block', num_m_blocks)
    print('num n block', num_n_blocks)

    print(f"\n{'='*80}")
    print(f"  {dtype_str}  B={B}  Sq={Sq} (win={win}^2)  H={H}  D={D}  scale={scale:.4f}")
    print(f"  m={m_block}  n={n_block}  threads={threads}  "
          f"blocks=({num_m_blocks}x{num_n_blocks})")
    print(f"{'='*80}")

    q_c, q_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    k_c, k_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    v_c, v_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)
    o_c, o_t = make_qkvo(B, Sq, H, D, torch_dtype, dtype)

    # SAM-style learnable rel_pos params (2*win-1, D), float32
    Rh = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))
    Rw = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))

    # Pre-computed biases (B, Sq, H, win) - static for this benchmark
    rel_h_t, rel_w_t = compute_rel_bias(q_t, Rh, Rw, win)
    rel_h_c, rel_w_c = to_cute_pos(rel_h_t), to_cute_pos(rel_w_t)

    # Identity permutations: no token reordering in this benchmark.
    # Shape (B, Sq) — kernel slices per-batch via m_perm[batch, None].
    perm_q_t = torch.arange(Sq, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    perm_k_t = torch.arange(Sq, dtype=torch.int32, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    perm_q_c = from_dlpack(perm_q_t, assumed_align=4)
    perm_k_c = from_dlpack(perm_k_t, assumed_align=4)

    # Compile once with a dense mask; the kernel is shape-specialized, not value-specialized.
    dense_mask_c, _ = make_A_mask(B, H, num_m_blocks, num_n_blocks, sparsity=0.0)
    cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        FlashAttentionForwardAmpere(D, m_block, n_block, threads, win),
        q_c, k_c, v_c, o_c, rel_h_c, rel_w_c, perm_q_c, perm_k_c,
        dense_mask_c, scale, cu_stream,
        options="",
    )

    # SAM naive baseline (dense, run once - sparsity doesn't change this path)
    BH = B * H
    k_f = k_t.permute(0,2,1,3).reshape(BH, Sq, D).half()
    v_f = v_t.permute(0,2,1,3).reshape(BH, Sq, D).half()
    q_f = q_t.half()
    def sam_naive():
        Rh = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))
        Rw = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))
        r_q = q_f.permute(0,2,1,3).reshape(BH, win, win, D)
        rh = torch.einsum("bhwc,hkc->bhwk", r_q, Rh.half()).reshape(BH, Sq, win)
        rw = torch.einsum("bhwc,wkc->bhwk", r_q, Rw.half()).reshape(BH, Sq, win)
        rp = (rh[:,:,:,None] + rw[:,:,None,:]).reshape(BH, Sq, Sq)
        q  = q_f.permute(0,2,1,3).reshape(BH, Sq, D)
        return torch.matmul(F.softmax(q @ k_f.transpose(-2,-1) * scale + rp, dim=-1), v_f)

    t_sam = timeit(sam_naive, warmup, iters)
    print(f"\n  SAM naive (dense baseline): {t_sam:.1f} us  "
          f"({flops/(t_sam*1e-6)/1e12:.2f} TFLOP/s)\n")

    print(f"  {'Sparsity':>9} | {'fa2-total us':>13} | {'fa2-kernel us':>14} | "
          f"{'TFLOP/s':>8} | {'vs SAM (naive)':>14} | {'vs Torch'}| {'verify':>7}")
    print(f"  {'-'*9}-+-{'-'*13}-+-{'-'*14}-+-{'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*7}")

    for sp in sparsities:
        bmask_c, bmask_t = make_A_mask(B, H, num_m_blocks, num_n_blocks, sp)

        # Verify correctness at this sparsity level (if requested).
        status = "-"
        if verify:
            compiled(q_c, k_c, v_c, o_c, rel_h_c, rel_w_c,
                     perm_q_c, perm_k_c, bmask_c, scale, cu_stream)
            torch.cuda.synchronize()

            q_ref = q_t.permute(0, 2, 1, 3).float()
            k_ref = k_t.permute(0, 2, 1, 3).float()
            v_ref = v_t.permute(0, 2, 1, 3).float()

            k_idx = torch.arange(Sq, dtype=torch.long)
            k_row = k_idx // win
            k_col = k_idx % win

            rel_H = rel_h_t.permute(0, 2, 1, 3).float()
            rel_W = rel_w_t.permute(0, 2, 1, 3).float()
            rel_pos = rel_H[:, :, :, k_row] + rel_W[:, :, :, k_col]

            scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale + rel_pos

            if sp > 0.0:
                mask_full = (
                    bmask_t.float()
                    .repeat_interleave(m_block, dim=2)
                    .repeat_interleave(n_block, dim=3)
                    [:, :, :Sq, :Sq].bool()
                )
                scores = scores.masked_fill(~mask_full, float("-inf"))

            # nan_to_num: fully-masked rows give softmax([-inf,...]) = NaN in PyTorch
            # but the kernel outputs 0 for those rows -- both are correct sentinels.
            ref = (F.softmax(scores, dim=-1).nan_to_num(0.0)
                   .matmul(v_ref)
                   .permute(0, 2, 1, 3)
                   .to(torch_dtype))

            max_diff = (o_t.float().cpu() - ref.float().cpu()).abs().max().item()
            status = "PASS" if max_diff <= 1e-2 else f"FAIL(d{max_diff:.3f})"

        # FA2 total: includes einsum (real-world cost). Mask is pre-built -- in
        # production it comes from a fixed sparsity pattern, not random sampling,
        # so randperm does not belong in the timed path.
        def fa2_total():
            Rh = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))
            Rw = get_rel_pos(win, win, torch.randn(2*win-1, D, device="cuda"))
            rh_t, rw_t = compute_rel_bias(q_t, Rh, Rw, win)
            compiled(q_c, k_c, v_c, make_o_cute(o_t, dtype),
                     to_cute_pos(rh_t), to_cute_pos(rw_t),
                     perm_q_c, perm_k_c, bmask_c, scale, cu_stream)

        # FA2 kernel: bias + mask pre-computed -- measures pure kernel time
        def fa2_kernel():
            compiled(q_c, k_c, v_c, make_o_cute(o_t, dtype),
                     rel_h_c, rel_w_c, perm_q_c, perm_k_c, bmask_c, scale, cu_stream)
        
        def fa2_torch():
            F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, is_causal=False, scale=scale)

        t_fa2  = timeit(fa2_total,  warmup, iters)
        t_kern = timeit(fa2_kernel, warmup, iters)
        t_torch= timeit(fa2_torch, warmup, iters)
        tflops = flops / (t_kern * 1e-6) / 1e12
        speedup = t_sam / t_fa2
        speedup_torch = t_torch / t_kern

        print(f"  {sp:>9.2f} | {t_fa2:>13.1f} | {t_kern:>14.1f} | "
              f"{tflops:>8.2f} | {speedup:>13.2f}x | {speedup_torch:>14.2f}x | {status:>7}")

    print()


# ---------------------------------------------------------------------------
# SAM presets  (effective batch = B x num_windows, win=14 -> 25 windows)
# ---------------------------------------------------------------------------
SAM_CONFIGS = {
    "sam_l": (4096, 16, 64),
    # "sam_h": (14*14, 16, 80),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default=None, choices=list(SAM_CONFIGS))
    p.add_argument("--dtype",         default="BFloat16")
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--seqlen_q",      type=int,   default=None)
    p.add_argument("--num_head",      type=int,   default=None)
    p.add_argument("--head_dim",      type=int,   default=None)
    p.add_argument("--softmax_scale", type=float, default=None)
    p.add_argument("--m_block_size",  type=int,   default=64)
    p.add_argument("--n_block_size",  type=int,   default=64)
    p.add_argument("--num_threads",   type=int,   default=128)
    p.add_argument("--warmup",        type=int,   default=5)
    p.add_argument("--iters",         type=int,   default=20)
    p.add_argument("--no_verify",     action="store_true")
    p.add_argument("--sparsity",      type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75,  0.9],
                   metavar="S",
                   help="one or more sparsity levels to sweep, e.g. --sparsity 0.0 0.25 0.5 0.75")
    args = p.parse_args()

    if args.model:
        configs = [SAM_CONFIGS[args.model]]
    elif args.seqlen_q:
        configs = [(args.seqlen_q, args.num_head or 16, args.head_dim or 64)]
    else:
        configs = [SAM_CONFIGS["sam_l"]]

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
