#!/usr/bin/env python3
"""
Fine-grained latency profiler for the sub-operations inside SAM's
Attention.forward:

  ① qkv          — self.qkv(x)  linear projection
  ② attn_compute — Q·Kᵀ scaling + add_decomposed_rel_pos + softmax
  ③ pv            — P·V  (attn @ v)
  ④ proj          — self.proj(x)  output linear projection
  ⑤ attn_total   — full Attention.forward (wraps all four above)

The profiler patches Attention.forward in-place via a thin wrapper so
no checkpoint re-load is needed. Results are reported per-block, with
mean ± std across runs, both in ms and as % of the block's total attn time.

Usage (standalone):
    python profile_attn_layers.py \
        --model-ckt ./ckts/sam_hq_vit_l.pth \
        --model-type vit_l \
        --n-warmup 5 \
        --n-runs 20

Usage (import into notebook):
    from profile_attn_layers import run_profile, print_report, plot_heatmap
    results = run_profile("vit_l", "./ckts/sam_hq_vit_l.pth")
    print_report(results)
    plot_heatmap(results)         # requires matplotlib
"""

import os
import sys
import argparse
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── SAM imports ────────────────────────────────────────────────────────────────
SAM1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam-hq")
if SAM1_PATH not in sys.path:
    sys.path.insert(0, SAM1_PATH)

from segment_anything import sam_model_registry
from segment_anything.modeling.image_encoder import (
    Attention,
    add_decomposed_rel_pos,
)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA timing helpers
# ─────────────────────────────────────────────────────────────────────────────

class CUDATimer:
    """Accumulates elapsed times (ms) using paired CUDA events."""

    def __init__(self):
        self.samples: List[float] = []
        self._start: torch.cuda.Event = None

    def start(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def stop(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        e.synchronize()
        self.samples.append(self._start.elapsed_time(e))

    def reset(self):
        self.samples.clear()

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if self.samples else 0.0

    @property
    def total(self) -> float:
        return float(np.sum(self.samples)) if self.samples else 0.0


@contextmanager
def record(timer: CUDATimer):
    timer.start()
    try:
        yield
    finally:
        timer.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Patched Attention.forward
# ─────────────────────────────────────────────────────────────────────────────

def _make_patched_forward(attn_module: Attention, timers: Dict[str, CUDATimer]):
    """Return a new forward() that records qkv / attn_compute / pv / proj."""

    def forward(x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        # ── ① QKV linear ──────────────────────────────────────────────────────
        with record(timers["qkv"]):
            qkv = attn_module.qkv(x).reshape(B, H * W, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * attn_module.num_heads, H * W, -1).unbind(0)

        # ── ② Attention scores: Q·Kᵀ + rel_pos + softmax ─────────────────────
        with record(timers["attn_compute"]):
            attn = (q * attn_module.scale) @ k.transpose(-2, -1)
            if attn_module.use_rel_pos:
                attn = add_decomposed_rel_pos(
                    attn, q,
                    attn_module.rel_pos_h, attn_module.rel_pos_w,
                    (H, W), (H, W),
                )
            attn = attn.softmax(dim=-1)

        # ── ③ P·V ─────────────────────────────────────────────────────────────
        with record(timers["pv"]):
            x = (attn @ v).view(B, attn_module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        # ── ④ Output projection ───────────────────────────────────────────────
        with record(timers["proj"]):
            x = attn_module.proj(x)

        return x

    return forward


# ─────────────────────────────────────────────────────────────────────────────
# Attach / detach hooks
# ─────────────────────────────────────────────────────────────────────────────

SUB_OPS = ("qkv", "attn_compute", "pv", "proj")


def attach_profiler(encoder) -> Tuple[Dict, List]:
    """
    Patch every Attention block in *encoder* with the timing wrapper.

    Returns
    -------
    all_timers : dict mapping block_index → {sub_op: CUDATimer}
    originals  : list of (attn_module, original_forward) for teardown
    """
    all_timers: Dict[int, Dict[str, CUDATimer]] = {}
    originals = []

    for i, blk in enumerate(encoder.blocks):
        attn = blk.attn
        block_timers = {op: CUDATimer() for op in SUB_OPS}
        # also time the entire attn module via a hook
        total_timer = CUDATimer()
        block_timers["attn_total"] = total_timer
        all_timers[i] = block_timers

        # patch forward
        originals.append((attn, attn.forward))
        attn.forward = _make_patched_forward(attn, block_timers)

        # hook for total attn time
        h_pre  = attn.register_forward_pre_hook(lambda m, inp, _t=total_timer: _t.start())
        h_post = attn.register_forward_hook   (lambda m, inp, out, _t=total_timer: _t.stop())
        originals.append(h_pre)
        originals.append(h_post)

    return all_timers, originals


def detach_profiler(originals):
    """Restore original forwards and remove hooks."""
    for item in originals:
        if isinstance(item, tuple):
            attn_mod, orig_fwd = item
            attn_mod.forward = orig_fwd
        else:
            item.remove()  # hook handle


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_sam1(model_type: str, model_ckt: str, device: str = "cuda"):
    print(f"Loading SAM1/{model_type} from {model_ckt} ...")
    sam = sam_model_registry[model_type](checkpoint=model_ckt).to(device)
    return sam.image_encoder


# ─────────────────────────────────────────────────────────────────────────────
# Main profiling loop
# ─────────────────────────────────────────────────────────────────────────────

def run_profile(
    model_type: str = "vit_l",
    model_ckt: str = "./ckts/sam_hq_vit_l.pth",
    img_size: int = 1024,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Dict:
    """
    Load SAM, attach sub-op timers, warm-up, measure, return results dict.

    Returns
    -------
    {
      'model_type': str,
      'n_runs': int,
      'batch_size': int,
      'blocks': {
          block_idx: {
              'window_size': int,            # 0 = global attention
              'qkv':          {'mean': float, 'std': float},
              'attn_compute': {'mean': float, 'std': float},
              'pv':           {'mean': float, 'std': float},
              'proj':         {'mean': float, 'std': float},
              'attn_total':   {'mean': float, 'std': float},
          },
          ...
      }
    }
    """
    assert torch.cuda.is_available(), "CUDA required"
    encoder = load_sam1(model_type, model_ckt, device)
    encoder.eval()
    if dtype == torch.float16:
        encoder = encoder.half()

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)

    all_timers, originals = attach_profiler(encoder)

    # ── warm-up ───────────────────────────────────────────────────────────────
    print(f"Warming up ({n_warmup} passes) ...")
    with torch.no_grad():
        for _ in range(n_warmup):
            encoder(dummy)

    for block_timers in all_timers.values():
        for t in block_timers.values():
            t.reset()

    # ── measure ───────────────────────────────────────────────────────────────
    print(f"Profiling ({n_runs} passes) ...")
    with torch.no_grad():
        for _ in range(n_runs):
            encoder(dummy)

    torch.cuda.synchronize()
    detach_profiler(originals)

    # ── collect results ───────────────────────────────────────────────────────
    results = {
        "model_type": model_type,
        "n_runs": n_runs,
        "batch_size": batch_size,
        "blocks": {},
    }
    for i, blk in enumerate(encoder.blocks):
        ws = getattr(blk, "window_size", 0)
        block_data = {"window_size": ws}
        for op in SUB_OPS + ("attn_total",):
            t = all_timers[i][op]
            block_data[op] = {"mean": t.mean, "std": t.std, "samples": list(t.samples)}
        results["blocks"][i] = block_data

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: Dict, show_std: bool = True):
    """Pretty-print a per-block table of sub-op latencies."""
    model_type = results["model_type"]
    n_runs     = results["n_runs"]
    bs         = results["batch_size"]
    blocks     = results["blocks"]

    col_w = 18
    hdr_ops = ["qkv", "attn_compute", "pv", "proj", "attn_total"]

    sep = "=" * (14 + col_w * len(hdr_ops) + 10)
    print(f"\n{sep}")
    print(f"  SAM-1 Attention Sub-Op Latency  |  model={model_type}  bs={bs}  runs={n_runs}")
    print(f"  All times in ms (mean ± std).  attn% = sub_op / attn_total")
    print(sep)

    # Header
    hdr = f"  {'block':<10} {'type':<8}"
    for op in hdr_ops:
        hdr += f"  {op:>{col_w}}"
    print(hdr)
    print(f"  {'-'*(12 + col_w * len(hdr_ops) + 4)}")

    # Accumulators for summary
    sums = {op: 0.0 for op in hdr_ops}
    n_blocks = len(blocks)

    for i, bdata in blocks.items():
        ws       = bdata["window_size"]
        btype    = "global" if ws == 0 else f"win{ws}"
        total_ms = bdata["attn_total"]["mean"]

        row = f"  block[{i:02d}]  {btype:<8}"
        for op in hdr_ops:
            m = bdata[op]["mean"]
            s = bdata[op]["std"]
            pct = m / total_ms * 100 if total_ms > 0 and op != "attn_total" else None
            if show_std:
                val_str = f"{m:.3f}±{s:.3f}"
            else:
                val_str = f"{m:.3f}"
            if pct is not None:
                val_str += f" ({pct:.1f}%)"
            row += f"  {val_str:>{col_w}}"
            sums[op] += m

        print(row)

    # Summary row
    print(f"  {'-'*(12 + col_w * len(hdr_ops) + 4)}")
    row = f"  {'Σ all blocks':<18}"
    for op in hdr_ops:
        val_str = f"{sums[op]:.2f} ms"
        row += f"  {val_str:>{col_w}}"
    print(row)

    # Ratios row (% of total attn)
    row = f"  {'% of attn_total':<18}"
    for op in hdr_ops:
        pct = sums[op] / sums["attn_total"] * 100 if sums["attn_total"] > 0 and op != "attn_total" else 100.0
        row += f"  {pct:>{col_w}.1f}%"
    print(row)
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Optional: matplotlib heatmap (used from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(results: Dict, figsize=(14, 6), save_path: str = None):
    """
    Heatmap: rows = blocks, cols = sub-ops, colour = mean ms.
    Requires matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if save_path else "TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available — skipping heatmap")
        return

    ops    = ["qkv", "attn_compute", "pv", "proj"]
    blocks = results["blocks"]
    n      = len(blocks)

    data       = np.zeros((n, len(ops)))
    row_labels = []
    for i, bdata in blocks.items():
        ws = bdata["window_size"]
        row_labels.append(f"blk{i:02d}\n{'G' if ws==0 else 'W'}")
        for j, op in enumerate(ops):
            data[i, j] = bdata[op]["mean"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Latency (ms)")

    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(["① QKV", "② Attn\n(Q·Kᵀ+rel+sfmx)", "③ P·V", "④ Proj"], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=7)

    # annotate cells
    for i in range(n):
        for j in range(len(ops)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if data[i,j] < data.max()*0.7 else "white")

    model = results["model_type"]
    ax.set_title(f"SAM-1 ({model}) – Attention Sub-Op Latency (ms)\n"
                 f"bs={results['batch_size']}  runs={results['n_runs']}  "
                 f"G=global  W=windowed", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Heatmap saved → {save_path}")
    else:
        plt.show()
    return fig


def plot_bar(results: Dict, block_idx: int = None, figsize=(12, 5), save_path: str = None):
    """
    Grouped bar chart: mean latency per sub-op.
    If block_idx is given, show only that block; else aggregate over all blocks.
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if save_path else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping bar chart")
        return

    ops    = ["qkv", "attn_compute", "pv", "proj"]
    blocks = results["blocks"]

    if block_idx is not None:
        blist = {block_idx: blocks[block_idx]}
        title_tag = f"block[{block_idx:02d}]"
    else:
        blist = blocks
        title_tag = "all blocks (mean across blocks)"

    # mean per op over selected blocks
    means = {op: np.mean([b[op]["mean"] for b in blist.values()]) for op in ops}
    stds  = {op: np.mean([b[op]["std"]  for b in blist.values()]) for op in ops}
    total = sum(means.values())

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(ops))
    bars = ax.bar(x, [means[op] for op in ops], yerr=[stds[op] for op in ops],
                  color=colors, capsize=5, width=0.5, edgecolor="white", linewidth=0.8)

    for bar, op in zip(bars, ops):
        pct = means[op] / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[op] + 0.002,
                f"{means[op]:.3f} ms\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["① QKV", "② Attn compute\n(Q·Kᵀ+rel+sfmx)", "③ P·V", "④ Proj"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"SAM-1 ({results['model_type']}) – {title_tag}\n"
                 f"bs={results['batch_size']}  runs={results['n_runs']}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Bar chart saved → {save_path}")
    else:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Profile SAM attention sub-ops")
    p.add_argument("--model-ckt",   default="./ckts/sam_hq_vit_l.pth")
    p.add_argument("--model-type",  default="vit_l", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--img-size",    type=int, default=1024)
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--n-warmup",    type=int, default=5)
    p.add_argument("--n-runs",      type=int, default=20)
    p.add_argument("--save-heatmap",default=None, help="Path to save heatmap PNG")
    p.add_argument("--save-bar",    default=None, help="Path to save bar chart PNG")
    args = p.parse_args()

    results = run_profile(
        model_type = args.model_type,
        model_ckt  = args.model_ckt,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        n_warmup   = args.n_warmup,
        n_runs     = args.n_runs,
    )
    print_report(results)

    if args.save_heatmap:
        plot_heatmap(results, save_path=args.save_heatmap)
    if args.save_bar:
        plot_bar(results, save_path=args.save_bar)


if __name__ == "__main__":
    main()
