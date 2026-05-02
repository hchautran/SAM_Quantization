#!/usr/bin/env python3
"""Per-component profiler for the PE vision tower, baseline vs.
ToMe/GradToMe/SparseSAM/FlashRoPE patched."""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PM_ROOT = os.path.join(_REPO, "perception_models")
PITOME_PATH = os.path.join(_REPO, "PiToMe")
sys.path.insert(0, _REPO)                          # shared utils at repo root
sys.path.insert(0, os.path.join(_REPO, "sam-hq"))  # registry eagerly registers SAM
sys.path.insert(0, PM_ROOT)
sys.path.insert(0, PITOME_PATH)


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


def attach_timers(visual):
    """Register CUDA timing hooks on the PE VisionTransformer.

    Layout per ResidualAttentionBlock:
        ln_1 → attn → ls_1 → ln_2 → mlp(c_fc, gelu, c_proj) → ls_2
    """
    timers, handles = {}, []
    a = lambda m, n: _attach(m, n, timers, handles)

    a(visual.conv1,   "conv1")
    if not isinstance(visual.ln_pre, torch.nn.Identity):
        a(visual.ln_pre,  "ln_pre")
    a(visual.transformer, "transformer")
    if not isinstance(visual.ln_post, torch.nn.Identity):
        a(visual.ln_post, "ln_post")
    if visual.attn_pool is not None:
        a(visual.attn_pool, "attn_pool")

    for i, blk in enumerate(visual.transformer.resblocks):
        a(blk,           f"block[{i:02d}]")
        a(blk.ln_1,      f"block[{i:02d}].ln_1")
        a(blk.attn,      f"block[{i:02d}].attn")
        if not isinstance(blk.ls_1, torch.nn.Identity):
            a(blk.ls_1,  f"block[{i:02d}].ls_1")
        a(blk.ln_2,      f"block[{i:02d}].ln_2")
        a(blk.mlp,       f"block[{i:02d}].mlp")
        # mlp is an nn.Sequential of (c_fc, gelu, c_proj).
        if hasattr(blk.mlp, 'c_fc'):
            a(blk.mlp.c_fc,   f"block[{i:02d}].mlp.c_fc")
            a(blk.mlp.c_proj, f"block[{i:02d}].mlp.c_proj")
        if not isinstance(blk.ls_2, torch.nn.Identity):
            a(blk.ls_2,  f"block[{i:02d}].ls_2")

    return timers, handles


def _get(timers, key):
    return timers[key].mean() if key in timers else 0.0


def _stage_end_set(n_blocks, num_stages):
    if num_stages is None or num_stages <= 1 or n_blocks < num_stages:
        return set()
    chunk = n_blocks // num_stages
    return {chunk * (i + 1) - 1 for i in range(num_stages - 1)}


def _block_table(timers, total_ms, n_blocks, stage_ends=None):
    if n_blocks == 0:
        return
    stage_ends = stage_ends or set()

    print(f"\n  {'block':<12} {'total':>8}  {'attn':>8}  {'mlp':>8}  {'attn%':>6}  {'mlp%':>6}  {'tag'}")
    print(f"  {'-'*68}")

    sum_total = sum_attn = sum_mlp = 0.0
    for i in range(n_blocks):
        tag      = f"block[{i:02d}]"
        blk_ms   = _get(timers, tag)
        attn_ms  = _get(timers, tag + ".attn")
        mlp_ms   = _get(timers, tag + ".mlp")
        attn_pct = attn_ms / blk_ms * 100 if blk_ms > 0 else 0
        mlp_pct  = mlp_ms  / blk_ms * 100 if blk_ms > 0 else 0
        marker   = "stage-end" if i in stage_ends else ""

        sum_total += blk_ms; sum_attn += attn_ms; sum_mlp += mlp_ms
        print(f"  block[{i:02d}]    {blk_ms:>8.3f}  {attn_ms:>8.3f}  {mlp_ms:>8.3f}  "
              f"{attn_pct:>5.1f}%  {mlp_pct:>5.1f}%  {marker}")

    a_pct   = sum_attn  / sum_total * 100 if sum_total > 0 else 0
    m_pct   = sum_mlp   / sum_total * 100 if sum_total > 0 else 0
    blk_pct = sum_total / total_ms  * 100 if total_ms  > 0 else 0
    print(f"  {'-'*68}")
    print(f"  {'all blocks':<12} {sum_total:>8.1f}  {sum_attn:>8.1f}  {sum_mlp:>8.1f}  "
          f"{a_pct:>5.1f}%  {m_pct:>5.1f}%  ({blk_pct:.1f}% of total)")


def print_report(timers, n_runs, model_name, batch_size, n_blocks, stage_ends=None):
    top_keys = ["conv1", "ln_pre", "transformer", "ln_post", "attn_pool"]
    total_ms = sum(_get(timers, k) for k in top_keys)

    print(f"\n{'='*72}")
    print(f"  PE Encoder  |  model={model_name}  bs={batch_size}  runs={n_runs}  total={total_ms:.1f}ms")
    print(f"{'='*72}")
    _block_table(timers, total_ms, n_blocks, stage_ends=stage_ends)
    parts = []
    for k in top_keys:
        v = _get(timers, k)
        if v > 0:
            parts.append(f"{k}={v:.2f}ms")
    print("\n  " + "  ".join(parts))
    print(f"{'='*72}\n")


# PE patch dispatch — uses the central registry
# (`PiToMe/algo/registry.py`). To add a new algorithm, register it there;
# this module needs no changes.

from PiToMe.algo.registry import (
    PE_REGISTRY, algo_choices as _pe_choices,
    apply_pe as _registry_apply_pe, remove_all_pe as _registry_remove_all,
)


def _warmup_rope(model, dummy):
    """rope.freq is built lazily inside VisionTransformer.forward_features.
    Run one dummy forward so cute kernels can pull cos/sin during patching."""
    with torch.no_grad():
        model.encode_image(dummy)


def apply_pe_patch(model, dummy, algo, ratio, args):
    """Apply a PE patch via the registry. Cute-kernel patches need rope.freq
    populated upfront; we warm it up unconditionally before applying."""
    if algo == "none":
        return
    if algo not in PE_REGISTRY:
        raise ValueError(f"Unknown algo: {algo}. Choices: {_pe_choices()}")
    _warmup_rope(model, dummy)
    _registry_apply_pe(model, algo, args=args, ratio=ratio)


def remove_pe_patch(model):
    """Strip every supported PE patch. Safe to call even if nothing is patched."""
    _registry_remove_all(model)


def print_comparison(t_base, t_patched, model_name, batch_size, algo, ratio,
                     n_runs, n_blocks, stage_ends=None, patched_label=None):
    """Compact speedup table: baseline vs patched."""
    col = patched_label or algo[:4]
    stage_ends = stage_ends or set()

    top_keys = ["conv1", "ln_pre", "transformer", "ln_post", "attn_pool"]
    total_base    = sum(_get(t_base,    k) for k in top_keys)
    total_patched = sum(_get(t_patched, k) for k in top_keys)

    def spd(b, t): return b / t if t > 1e-6 else float('inf')
    def fmt_spd(s): return f"{s:.2f}x" if s != float('inf') else "  inf"

    W = 92
    print(f"\n{'='*W}")
    title = f"  Baseline vs {algo.upper()}"
    if ratio is not None:
        title += f"  ratio={ratio}"
    print(f"{title}  model={model_name}  bs={batch_size}  runs={n_runs}")
    print(f"{'='*W}")
    print(f"  {'block':<10} {'total':>12}  {'spd':>5} |"
          f"  {'attn':>12}  {'spd':>5} |"
          f"  {'mlp':>12}  {'ovhd':>6}  tag")
    print(f"  {'':10} {'base':>6} {col:>6}  {'':>5} |"
          f"  {'base':>6} {col:>6}  {'':>5} |"
          f"  {'base':>6} {col:>6}")
    print(f"  {'-'*W}")

    sum_ovhd = 0.0
    for i in range(n_blocks):
        tag   = f"block[{i:02d}]"
        b_blk = _get(t_base,    tag);            p_blk = _get(t_patched, tag)
        b_att = _get(t_base,    tag + ".attn");  p_att = _get(t_patched, tag + ".attn")
        b_mlp = _get(t_base,    tag + ".mlp");   p_mlp = _get(t_patched, tag + ".mlp")
        p_n1  = _get(t_patched, tag + ".ln_1");  p_n2  = _get(t_patched, tag + ".ln_2")
        p_l1  = _get(t_patched, tag + ".ls_1");  p_l2  = _get(t_patched, tag + ".ls_2")
        ovhd  = max(0.0, p_blk - p_att - p_mlp - p_n1 - p_n2 - p_l1 - p_l2)
        sum_ovhd += ovhd
        marker = "stage-end" if i in stage_ends else ""

        print(f"  block[{i:02d}]  {b_blk:>6.2f} {p_blk:>6.2f}  {fmt_spd(spd(b_blk,p_blk)):>5} |"
              f"  {b_att:>6.2f} {p_att:>6.2f}  {fmt_spd(spd(b_att,p_att)):>5} |"
              f"  {b_mlp:>6.2f} {p_mlp:>6.2f}  {ovhd:>6.2f}  {marker}")

    def _agg(t, sfx):
        return sum(_get(t, f"block[{i:02d}]{sfx}") for i in range(n_blocks))

    b_attn = _agg(t_base,    ".attn"); p_attn = _agg(t_patched, ".attn")
    b_mlp  = _agg(t_base,    ".mlp");  p_mlp  = _agg(t_patched, ".mlp")

    print(f"\n  {'':10} {'base':>6} {col:>6}  {'spd':>5}")
    print(f"  {'-'*36}")
    for label, b, p in [("attn (Σ)",  b_attn, p_attn),
                         ("mlp  (Σ)",  b_mlp,  p_mlp),
                         ("overhead",  0.0,    sum_ovhd),
                         ("TOTAL",     total_base, total_patched)]:
        delta   = p - b
        sign    = "+" if delta >= 0 else ""
        spd_str = fmt_spd(spd(b, p)) if b > 0 else "  n/a"
        print(f"  {label:<10} {b:>6.1f} {p:>6.1f}  {spd_str:>5}   {sign}{delta:.1f}ms")
    print(f"{'='*W}\n")


def load_pe(model_name, checkpoint_path, device, dtype, vision_only=False):
    """Load PE and return the VisionTransformer plus a callable that drives
    the full forward (so RoPE-warmup uses the same path the patch expects)."""
    import core.vision_encoder.pe as pe
    print(f"Loading PE model: {model_name} (ckpt={checkpoint_path or 'HF default'})")
    if vision_only:
        # Skip text tower for cheaper load; rope warmup still works via
        # visual.forward (encode_image is just normalize + visual()).
        model = pe.VisionTransformer.from_config(
            model_name, pretrained=True, checkpoint_path=checkpoint_path,
        )
        model = model.to(device=device, dtype=dtype).eval()

        class _VisualWrapper:
            def __init__(self, vt): self.visual = vt
            def encode_image(self, x): return self.visual(x)
            def modules(self):  return self.visual.modules()
            def __getattr__(self, name):
                return getattr(self.visual, name)
        return _VisualWrapper(model), model

    model = pe.CLIP.from_config(
        model_name, pretrained=True, checkpoint_path=checkpoint_path,
    )
    model = model.to(device=device, dtype=dtype).eval()
    return model, model.visual


def _run_profile(model, visual, dummy, n_warmup, n_runs, label):
    timers, handles = attach_timers(visual)
    visual.eval()

    print(f"  Warming up [{label}] ({n_warmup} passes) ...")
    with torch.no_grad():
        for _ in range(n_warmup):
            model.encode_image(dummy)
    for t in timers.values():
        t.reset()

    print(f"  Profiling  [{label}] ({n_runs} passes) ...")
    with torch.no_grad():
        for _ in range(n_runs):
            model.encode_image(dummy)

    for h in handles:
        h.remove()
    return timers


def main():
    parser = argparse.ArgumentParser(
        description="Profile PE vision encoder components (with optional patch comparison)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--model',       type=str, default='PE-Core-L14-336',
                        help="PE config name (see eval_pe_clip.py --list-models).")
    parser.add_argument('--checkpoint',  type=str, default=None,
                        help="Local checkpoint; if omitted, HF weights are downloaded.")
    parser.add_argument('--vision-only', action='store_true',
                        help="Load just the VisionTransformer (skip text tower).")

    parser.add_argument('--n-warmup',   type=int, default=5)
    parser.add_argument('--n-runs',     type=int, default=20)
    parser.add_argument('--img-size',   type=int, default=None,
                        help="Override input H=W. Defaults to model.image_size.")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dtype',      type=str, default='fp16',
                        choices=['fp32', 'fp16', 'bf16'])

    parser.add_argument('--tome-algo',   type=str, default='none',
                        choices=_pe_choices(),
                        help="Run a second pass with this algo and show comparison table. "
                             "Choices come from PiToMe/algo/registry.py — see "
                             "docs/ADDING_ALGORITHMS.md for adding new ones.")
    parser.add_argument('--tome-ratio',  type=float, default=0.7,
                        help="Per-stage-boundary token-keep fraction in (0, 1].")
    parser.add_argument('--sparse-ratio', type=float, default=None,
                        help="(sparsesam_partial) keep-bar width inside the cute mask as a "
                             "fraction of n-blocks; defaults to --tome-ratio.")
    parser.add_argument('--partial-start-block', type=int, default=0,
                        help="(sparsesam_partial only) first block index (0-indexed) at which "
                             "the patch kicks in. Earlier blocks run stock SDPA + full MLP. "
                             "e.g. '5' means apply partial-MLP starting from layer 6.")
    parser.add_argument('--mlp-merge', action=argparse.BooleanOptionalAction, default=True,
                        help="(*_partial algos) when True (default), MLP runs on the merged "
                             "token set with merge/MLP/unmerge. Use --no-mlp-merge to run MLP "
                             "on full S tokens — only K/V attention compression remains.")
    parser.add_argument('--num-stages',  type=int,   default=4,
                        help="Split L blocks into this many stages; merging happens at "
                             "each of the (num_stages-1) internal boundaries. Ignored "
                             "when --compress-at-blocks is set.")
    parser.add_argument('--compress-at-blocks', type=int, nargs='+', default=None,
                        help="Override --num-stages with explicit block indices (0-"
                             "indexed) after which compression fires. e.g. '4' compresses "
                             "once after layer 5; '4 11' compresses twice. Each "
                             "compression applies the full --tome-ratio.")
    parser.add_argument('--group-size',  type=int,   default=4,
                        help="(sparsesam) tokens per scoring group.")
    parser.add_argument('--use-flash-rope', action='store_true',
                        help="Route attention through the fused FA2 + 2D-axial RoPE cute "
                             "kernel for any of tome/gradtome/sparsesam.")

    parser.add_argument('--fa2', action='store_true',
                        help="Run a second pass with the standalone fused FA2+RoPE patch "
                             "(no token compression) and show comparison table.")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.", file=sys.stderr)
        sys.exit(1)

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, visual = load_pe(args.model, args.checkpoint, device, dtype,
                            vision_only=args.vision_only)
    img_size = args.img_size or visual.image_size
    dummy = torch.randn(args.batch_size, 3, img_size, img_size,
                        device=device, dtype=dtype)

    n_blocks = len(visual.transformer.resblocks)
    stage_ends = _stage_end_set(n_blocks, args.num_stages)

    print(f"  image_size={img_size}, dtype={args.dtype}, "
          f"L={n_blocks}, stage_ends={sorted(stage_ends)}")

    print("\n── Baseline (vanilla PE) ──")
    t_base = _run_profile(model, visual, dummy, args.n_warmup, args.n_runs, "baseline")

    want_tome = args.tome_algo != 'none'
    want_fa2  = args.fa2

    if not want_tome and not want_fa2:
        print_report(t_base, args.n_runs, args.model, args.batch_size,
                     n_blocks, stage_ends=set())
        return

    if want_fa2:
        print("\n── FA2 (fused FA2+RoPE, no token merging) ──")
        apply_pe_patch(model, dummy, algo='flash_rope', ratio=1.0, args=args)
        t_fa2 = _run_profile(model, visual, dummy, args.n_warmup, args.n_runs, "fa2")
        remove_pe_patch(model)
        print_comparison(
            t_base, t_fa2, model_name=args.model, batch_size=args.batch_size,
            algo="FA2", ratio=None, n_runs=args.n_runs, n_blocks=n_blocks,
            stage_ends=set(), patched_label="fa2",
        )

    if want_tome:
        print(f"\n── {args.tome_algo.upper()} ratio={args.tome_ratio}  num_stages={args.num_stages} ──")
        apply_pe_patch(model, dummy, algo=args.tome_algo,
                       ratio=args.tome_ratio, args=args)
        t_tome = _run_profile(model, visual, dummy, args.n_warmup, args.n_runs,
                              f"{args.tome_algo} r={args.tome_ratio}")
        remove_pe_patch(model)
        print_comparison(
            t_base, t_tome, model_name=args.model, batch_size=args.batch_size,
            algo=args.tome_algo, ratio=args.tome_ratio, n_runs=args.n_runs,
            n_blocks=n_blocks, stage_ends=stage_ends,
        )


if __name__ == '__main__':
    main()
