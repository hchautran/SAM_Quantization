#!/usr/bin/env python3
"""
Profile each component in the SAM2 / SAM-HQ2 / SAM-2.1 image encoder
(Hiera trunk + FpnNeck), optionally comparing against the sparsesam patch
(`PiToMe.algo.sparsesam.patch.sam2_hiera`).

Per-block table reports total / attn / mlp / type (windowed vs global) and
flags q_pool stage transitions. The comparison table shows base-vs-patched
timings + speedups for attention, MLP, and the residual block overhead.

Usage:
  # Baseline only
  python profile_sam2.py \
      --sam2-config sam2.1_hq_hiera_l.yaml \
      --model-ckt ./ckts/sam2.1_hq_hiera_large.pt

  # Baseline + sparsesam (block-sparse attention + local-block MLP prune)
  python profile_sam2.py \
      --sam2-config sam2.1_hq_hiera_l.yaml \
      --model-ckt ./ckts/sam2.1_hq_hiera_large.pt \
      --algo sparsesam --ratio 0.5

  # Disable MLP prune to isolate the attention-only effect
  python profile_sam2.py --algo sparsesam --ratio 0.5 --no-mlp-prune

  # Force the SDPA fallback path (no cute kernel)
  python profile_sam2.py --algo sparsesam --ratio 0.5 --no-cute
"""

import argparse
import os
import sys

import numpy as np
import torch

# ── repo paths ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "sam-hq"),
           os.path.join(_HERE, "sam-hq", "sam-hq2"),
           os.path.join(_HERE, "PiToMe")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA timing
# ─────────────────────────────────────────────────────────────────────────────

class CUDATimer:
    def __init__(self):
        self._start: torch.cuda.Event = None
        self.times_ms: list = []

    def start(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def stop(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        torch.cuda.synchronize()
        self.times_ms.append(self._start.elapsed_time(e))

    def reset(self):
        self.times_ms.clear()

    def mean(self):  return float(np.mean(self.times_ms))  if self.times_ms else 0.0
    def std(self):   return float(np.std(self.times_ms))   if self.times_ms else 0.0
    def total(self): return float(np.sum(self.times_ms))   if self.times_ms else 0.0


def _attach(module, name, timers, handles):
    t = CUDATimer()
    timers[name] = t
    handles.append(module.register_forward_pre_hook(lambda m, i:     t.start()))
    handles.append(module.register_forward_hook   (lambda m, i, o:   t.stop()))


# ─────────────────────────────────────────────────────────────────────────────
# Hooks: SAM2 ImageEncoder (Hiera trunk + FpnNeck)
# ─────────────────────────────────────────────────────────────────────────────

def attach_timers_sam2(image_encoder):
    timers, handles = {}, []
    a = lambda m, n: _attach(m, n, timers, handles)

    trunk = image_encoder.trunk
    neck  = image_encoder.neck

    a(trunk,             "trunk")
    a(trunk.patch_embed, "trunk.patch_embed")

    for i, blk in enumerate(trunk.blocks):
        a(blk,           f"trunk.block[{i:02d}]")
        a(blk.norm1,     f"trunk.block[{i:02d}].norm1")
        a(blk.attn,      f"trunk.block[{i:02d}].attn")
        a(blk.attn.qkv,  f"trunk.block[{i:02d}].attn.qkv")
        a(blk.attn.proj, f"trunk.block[{i:02d}].attn.proj")
        a(blk.norm2,     f"trunk.block[{i:02d}].norm2")
        a(blk.mlp,       f"trunk.block[{i:02d}].mlp")
        if hasattr(blk.mlp, 'layers') and len(blk.mlp.layers) >= 2:
            a(blk.mlp.layers[0], f"trunk.block[{i:02d}].mlp.l0")
            a(blk.mlp.layers[1], f"trunk.block[{i:02d}].mlp.l1")
        if getattr(blk, 'pool', None) is not None:
            a(blk.pool, f"trunk.block[{i:02d}].pool")

    a(neck, "neck")
    for i, conv in enumerate(neck.convs):
        a(conv, f"neck.conv[{i}]")
    a(neck.position_encoding, "neck.pos_enc")

    return timers, handles


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _get(timers, key):
    return timers[key].mean() if key in timers else 0.0


def _block_kind(blk, first_global_idx, idx):
    """Short tag describing this block's role for the report."""
    ws = getattr(blk, 'window_size', 0)
    is_global = (ws == 0)
    is_qpool  = (blk.q_stride is not None) or (blk.dim != blk.dim_out)
    parts = ["global" if is_global else f"win{ws}"]
    if is_qpool:
        parts.append("qpool")
    if idx < first_global_idx:
        parts.append("pre-G")
    return "/".join(parts)


def _print_block_table(timers, trunk, total_ms, first_global_idx):
    n_blocks = len(trunk.blocks)
    print(f"\n  {'block':<10} {'total':>8}  {'attn':>8}  {'mlp':>8}  "
          f"{'attn%':>6}  {'mlp%':>6}  {'type'}")
    print(f"  {'-'*72}")

    sum_total = sum_attn = sum_mlp = 0.0
    for i in range(n_blocks):
        tag      = f"trunk.block[{i:02d}]"
        blk_ms   = _get(timers, tag)
        attn_ms  = _get(timers, tag + ".attn")
        mlp_ms   = _get(timers, tag + ".mlp")
        pool_ms  = _get(timers, tag + ".pool")
        attn_pct = attn_ms / blk_ms * 100 if blk_ms > 0 else 0
        mlp_pct  = mlp_ms  / blk_ms * 100 if blk_ms > 0 else 0
        kind     = _block_kind(trunk.blocks[i], first_global_idx, i)
        pool_str = f"  pool={pool_ms:.2f}" if pool_ms > 0 else ""

        sum_total += blk_ms; sum_attn += attn_ms; sum_mlp += mlp_ms
        print(f"  block[{i:02d}]  {blk_ms:>8.3f}  {attn_ms:>8.3f}  {mlp_ms:>8.3f}  "
              f"{attn_pct:>5.1f}%  {mlp_pct:>5.1f}%  {kind}{pool_str}")

    a_pct = sum_attn / sum_total * 100 if sum_total > 0 else 0
    m_pct = sum_mlp  / sum_total * 100 if sum_total > 0 else 0
    blk_pct = sum_total / total_ms * 100 if total_ms > 0 else 0
    print(f"  {'-'*72}")
    print(f"  {'Σ blocks':<10} {sum_total:>8.1f}  {sum_attn:>8.1f}  {sum_mlp:>8.1f}  "
          f"{a_pct:>5.1f}%  {m_pct:>5.1f}%  ({blk_pct:.1f}% of total)")


def print_report(timers, trunk, n_runs, config_name, batch_size, first_global_idx):
    trunk_ms = _get(timers, "trunk")
    neck_ms  = _get(timers, "neck")
    total_ms = trunk_ms + neck_ms

    print(f"\n{'='*80}")
    print(f"  SAM2 Encoder  |  cfg={config_name}  bs={batch_size}  runs={n_runs}  "
          f"total={total_ms:.1f}ms")
    print(f"  first_global_idx={first_global_idx}  "
          f"global_att_blocks={list(getattr(trunk, 'global_att_blocks', []))}")
    print(f"{'='*80}")
    _print_block_table(timers, trunk, total_ms, first_global_idx)
    pe = _get(timers, "trunk.patch_embed")
    print(f"\n  trunk.patch_embed={pe:.2f}ms  neck={neck_ms:.2f}ms  trunk={trunk_ms:.2f}ms")
    print(f"{'='*80}\n")


def print_comparison(t_base, t_patched, trunk, config_name, batch_size, n_runs,
                     first_global_idx, label="patched", header_extra=""):
    n_blocks = len(trunk.blocks)
    top_keys = ["trunk.patch_embed"] + [f"trunk.block[{i:02d}]" for i in range(n_blocks)] + ["neck"]
    total_base    = sum(_get(t_base,    k) for k in top_keys)
    total_patched = sum(_get(t_patched, k) for k in top_keys)

    def spd(b, t): return b / t if t > 1e-6 else float('inf')
    def fmt_spd(s): return f"{s:.2f}x" if s != float('inf') else "  inf"

    W = 100
    print(f"\n{'='*W}")
    print(f"  Baseline vs {label.upper()}  cfg={config_name}  bs={batch_size}  runs={n_runs}"
          + (f"  {header_extra}" if header_extra else ""))
    print(f"{'='*W}")
    print(f"  {'block':<10} {'total':>14}  {'spd':>5} |"
          f"  {'attn':>14}  {'spd':>5} |"
          f"  {'mlp':>14}  {'spd':>5}  type")
    print(f"  {'':10} {'base':>7} {label[:6]:>6}  {'':>5} |"
          f"  {'base':>7} {label[:6]:>6}  {'':>5} |"
          f"  {'base':>7} {label[:6]:>6}")
    print(f"  {'-'*W}")

    for i in range(n_blocks):
        tag   = f"trunk.block[{i:02d}]"
        b_blk = _get(t_base,    tag);            p_blk = _get(t_patched, tag)
        b_att = _get(t_base,    tag + ".attn");  p_att = _get(t_patched, tag + ".attn")
        b_mlp = _get(t_base,    tag + ".mlp");   p_mlp = _get(t_patched, tag + ".mlp")
        kind  = _block_kind(trunk.blocks[i], first_global_idx, i)

        print(f"  block[{i:02d}]  {b_blk:>7.2f} {p_blk:>6.2f}  {fmt_spd(spd(b_blk,p_blk)):>5} |"
              f"  {b_att:>7.2f} {p_att:>6.2f}  {fmt_spd(spd(b_att,p_att)):>5} |"
              f"  {b_mlp:>7.2f} {p_mlp:>6.2f}  {fmt_spd(spd(b_mlp,p_mlp)):>5}  {kind}")

    def _agg(t, sfx):
        return sum(_get(t, f"trunk.block[{i:02d}]{sfx}") for i in range(n_blocks))

    b_attn = _agg(t_base, ".attn"); p_attn = _agg(t_patched, ".attn")
    b_mlp  = _agg(t_base, ".mlp");  p_mlp  = _agg(t_patched, ".mlp")

    print(f"\n  {'':10} {'base':>7} {label[:6]:>6}  {'spd':>5}    delta")
    print(f"  {'-'*40}")
    for tag, b, p in [("attn (Σ)",  b_attn, p_attn),
                       ("mlp  (Σ)",  b_mlp,  p_mlp),
                       ("TOTAL",     total_base, total_patched)]:
        delta = p - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {tag:<10} {b:>7.1f} {p:>6.1f}  {fmt_spd(spd(b,p)):>5}    {sign}{delta:.1f}ms")
    print(f"{'='*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Loader + patch helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sam2(config_file, model_ckt, device):
    from sam2.build_sam import build_sam2
    config_name = os.path.basename(config_file)
    if not config_name.startswith("configs/"):
        # Hydra search path is pkg://sam2; the canonical name lives under configs/.
        config_name = f"configs/sam2.1/{config_name}" if "sam2.1" in config_name else f"configs/{config_name}"
    print(f"Loading SAM2 config={config_name} from {model_ckt} ...")
    model = build_sam2(
        config_file=config_name,
        ckpt_path=model_ckt,
        device=device,
        apply_postprocessing=False,
    )
    return model.image_encoder


def apply_sparsesam(image_encoder, ratio, group_size, prune_mlp, use_cute):
    from PiToMe.algo.sparsesam.patch.sam2_hiera import apply_patch
    apply_patch(
        image_encoder.trunk,
        ratio=ratio,
        group_size=group_size,
        prune_mlp=prune_mlp,
        use_cute=use_cute,
    )


def remove_sparsesam(image_encoder):
    from PiToMe.algo.sparsesam.patch.sam2_hiera import remove_patch
    remove_patch(image_encoder.trunk)


# ─────────────────────────────────────────────────────────────────────────────
# Profiling driver
# ─────────────────────────────────────────────────────────────────────────────

def _run_profile(encoder, dummy, n_warmup, n_runs, label, amp_dtype):
    timers, handles = attach_timers_sam2(encoder)
    encoder.eval()

    from contextlib import nullcontext
    amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None else nullcontext())

    print(f"  Warming up [{label}] ({n_warmup} passes) ...")
    with torch.no_grad(), amp_ctx:
        for _ in range(n_warmup):
            encoder(dummy)
    for t in timers.values():
        t.reset()

    print(f"  Profiling  [{label}] ({n_runs} passes) ...")
    with torch.no_grad(), amp_ctx:
        for _ in range(n_runs):
            encoder(dummy)

    for h in handles:
        h.remove()
    return timers


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile SAM2 image-encoder components (baseline + sparsesam).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--sam2-config', type=str,
                        default='configs/sam2.1/sam2.1_hq_hiera_l.yaml')
    parser.add_argument('--model-ckt',   type=str,
                        default='./ckts/sam2.1_hq_hiera_large.pt')
    parser.add_argument('--n-warmup',    type=int, default=5)
    parser.add_argument('--n-runs',      type=int, default=20)
    parser.add_argument('--img-size',    type=int, default=1024)
    parser.add_argument('--batch-size',  type=int, default=1)
    parser.add_argument('--dtype',       type=str, default='fp16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Inference precision via autocast.')

    parser.add_argument('--algo',  type=str, default='none',
                        choices=['none', 'sparsesam'],
                        help='Optional second pass with this patch + comparison table.')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--group-size',  type=int, default=4)
    parser.add_argument('--no-mlp-prune', action='store_true')
    parser.add_argument('--no-cute',      action='store_true',
                        help='Force the SDPA fallback inside sparsesam (no cute kernel).')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.", file=sys.stderr)
        sys.exit(1)
    device = 'cuda'

    # ── load ──────────────────────────────────────────────────────────────────
    encoder = load_sam2(args.sam2_config, args.model_ckt, device)
    trunk   = encoder.trunk
    first_global_idx = (
        min(trunk.global_att_blocks)
        if getattr(trunk, "global_att_blocks", None) else 0
    )

    dummy = torch.randn(
        args.batch_size, 3, args.img_size, args.img_size,
        device=device, dtype=torch.float32,
    )

    amp_dtype = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]

    # ── baseline pass ─────────────────────────────────────────────────────────
    print("\n── Baseline ──")
    t_base = _run_profile(encoder, dummy, args.n_warmup, args.n_runs, "baseline", amp_dtype)

    if args.algo == 'none':
        print_report(t_base, trunk, args.n_runs, args.sam2_config,
                     args.batch_size, first_global_idx)
        return

    # ── patched pass ──────────────────────────────────────────────────────────
    print(f"\n── {args.algo.upper()}  ratio={args.ratio}  "
          f"prune_mlp={not args.no_mlp_prune}  cute={not args.no_cute} ──")
    apply_sparsesam(
        encoder,
        ratio=args.ratio,
        group_size=args.group_size,
        prune_mlp=not args.no_mlp_prune,
        use_cute=not args.no_cute,
    )
    t_patched = _run_profile(encoder, dummy, args.n_warmup, args.n_runs,
                             f"{args.algo} r={args.ratio}", amp_dtype)
    remove_sparsesam(encoder)

    print_comparison(
        t_base, t_patched, trunk,
        config_name=args.sam2_config,
        batch_size=args.batch_size,
        n_runs=args.n_runs,
        first_global_idx=first_global_idx,
        label=args.algo,
        header_extra=(f"ratio={args.ratio}  prune_mlp={not args.no_mlp_prune}  "
                      f"cute={not args.no_cute}  dtype={args.dtype}"),
    )


if __name__ == '__main__':
    main()
