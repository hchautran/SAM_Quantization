#!/usr/bin/env python3
"""
Profile each component in the SAM / SAM-HQ / SAM-2 / SAM-2.1 image encoder,
optionally comparing against a ToMe / PiToMe patched version.

Supports:
  sam1  — SAM / SAM-HQ   (ViT backbone: ImageEncoderViT)
  sam2  — SAM 2 / SAM-HQ2 / SAM 2.1  (Hiera backbone + FPN neck)

Usage:
  # SAM 1 baseline
  python profile_encoder.py --version sam1 \
      --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l

  # SAM 1 + ToMe comparison
  python profile_encoder.py --version sam1 \
      --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
      --tome-algo tome --tome-ratio 0.9 --sparsity 0.5

  # SAM 1 + PiToMe comparison
  python profile_encoder.py --version sam1 \
      --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
      --tome-algo pitome --tome-ratio 0.8 --tome-margin 0.5

  # SAM 2
  python profile_encoder.py --version sam2 \
      --model-ckt ./ckts/sam2.1_hiera_large.pt \
      --sam2-config ./sam2_configs/sam2.1/sam2.1_hiera_l.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch

# ── SAM 1 import ──────────────────────────────────────────────────────────────
SAM1_PATH = os.path.join(os.path.dirname(__file__), 'sam-hq')
sys.path.insert(0, SAM1_PATH)

# ── SAM 2 import ──────────────────────────────────────────────────────────────
SAM2_PATH = os.path.join(os.path.dirname(__file__), 'sam-hq', 'sam-hq2')
sys.path.insert(0, SAM2_PATH)

# ── ToMe import ───────────────────────────────────────────────────────────────
PITOME_PATH = os.path.join(os.path.dirname(__file__), 'PiToMe')
sys.path.insert(0, PITOME_PATH)


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
# SAM 1 / SAM-HQ  (ImageEncoderViT)
# ─────────────────────────────────────────────────────────────────────────────

def attach_timers_sam1(encoder):
    """Attach hooks to ImageEncoderViT."""
    timers, handles = {}, []
    a = lambda m, n: _attach(m, n, timers, handles)

    a(encoder.patch_embed, "patch_embed")
    a(encoder.neck,        "neck")

    for i, blk in enumerate(encoder.blocks):
        a(blk,           f"block[{i:02d}]")
        a(blk.norm1,     f"block[{i:02d}].norm1")
        a(blk.attn,      f"block[{i:02d}].attn")
        a(blk.attn.qkv,  f"block[{i:02d}].attn.qkv")
        a(blk.attn.proj, f"block[{i:02d}].attn.proj")
        a(blk.norm2,     f"block[{i:02d}].norm2")
        a(blk.mlp,       f"block[{i:02d}].mlp")
        a(blk.mlp.lin1,  f"block[{i:02d}].mlp.lin1")
        a(blk.mlp.lin2,  f"block[{i:02d}].mlp.lin2")

    return timers, handles


# ─────────────────────────────────────────────────────────────────────────────
# SAM 3  (HuggingFace Sam3VisionModel)
# ─────────────────────────────────────────────────────────────────────────────

def attach_timers_sam3(vision_enc):
    """Attach hooks to Sam3VisionModel (backbone layers + neck)."""
    timers, handles = {}, []
    a = lambda m, n: _attach(m, n, timers, handles)

    backbone = vision_enc.backbone

    a(backbone.embeddings, "embeddings")
    a(backbone.layer_norm, "backbone.final_ln")

    for i, layer in enumerate(backbone.layers):
        a(layer,             f"layer[{i:02d}]")
        a(layer.layer_norm1, f"layer[{i:02d}].norm1")
        a(layer.attention,   f"layer[{i:02d}].attn")
        a(layer.layer_norm2, f"layer[{i:02d}].norm2")
        a(layer.mlp,         f"layer[{i:02d}].mlp")

    a(vision_enc.neck, "neck")
    for i, fpn_layer in enumerate(vision_enc.neck.fpn_layers):
        a(fpn_layer, f"neck.fpn[{i}]")

    return timers, handles


# ─────────────────────────────────────────────────────────────────────────────
# SAM 2 / SAM-HQ2 / SAM 2.1  (Hiera trunk + FpnNeck)
# ─────────────────────────────────────────────────────────────────────────────

def attach_timers_sam2(image_encoder):
    """Attach hooks to SAM2 ImageEncoder (trunk=Hiera, neck=FpnNeck)."""
    timers, handles = {}, []
    a = lambda m, n: _attach(m, n, timers, handles)

    trunk = image_encoder.trunk
    neck  = image_encoder.neck

    # ── trunk ──
    a(trunk,            "trunk")
    a(trunk.patch_embed,"trunk.patch_embed")

    for i, blk in enumerate(trunk.blocks):
        a(blk,           f"trunk.block[{i:02d}]")
        a(blk.norm1,     f"trunk.block[{i:02d}].norm1")
        a(blk.attn,      f"trunk.block[{i:02d}].attn")
        a(blk.attn.qkv,  f"trunk.block[{i:02d}].attn.qkv")
        a(blk.attn.proj, f"trunk.block[{i:02d}].attn.proj")
        a(blk.norm2,     f"trunk.block[{i:02d}].norm2")
        a(blk.mlp,       f"trunk.block[{i:02d}].mlp")
        # MLP layers (MLP uses self.layers ModuleList)
        if hasattr(blk.mlp, 'layers') and len(blk.mlp.layers) >= 2:
            a(blk.mlp.layers[0], f"trunk.block[{i:02d}].mlp.l0")
            a(blk.mlp.layers[1], f"trunk.block[{i:02d}].mlp.l1")
        # Pool (only at stage-change blocks with q_stride)
        if getattr(blk, 'pool', None) is not None:
            a(blk.pool, f"trunk.block[{i:02d}].pool")

    # ── FPN neck ──
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


def _block_table(timers, prefix, total_ms, encoder=None):
    """Print per-block table: total, attn%, mlp%, type."""
    n_blocks = sum(
        1 for k in timers
        if k.startswith(f"{prefix}block[") and k.count('.') == (1 if prefix else 0)
    )
    if n_blocks == 0:
        return

    print(f"\n  {'block':<12} {'total':>8}  {'attn':>8}  {'mlp':>8}  {'attn%':>6}  {'mlp%':>6}  {'type'}")
    print(f"  {'-'*62}")

    sum_total = sum_attn = sum_mlp = 0.0
    for i in range(n_blocks):
        tag      = f"{prefix}block[{i:02d}]"
        blk_ms   = _get(timers, tag)
        attn_ms  = _get(timers, tag + ".attn")
        mlp_ms   = _get(timers, tag + ".mlp")
        pool_ms  = _get(timers, tag + ".pool")
        attn_pct = attn_ms / blk_ms * 100 if blk_ms > 0 else 0
        mlp_pct  = mlp_ms  / blk_ms * 100 if blk_ms > 0 else 0

        # Determine block type (windowed vs global) from encoder if available
        blk_type = ""
        if encoder is not None and not prefix:
            ws = getattr(encoder.blocks[i], 'window_size', None)
            if ws is not None:
                blk_type = "global" if ws == 0 else f"win{ws}"
        pool_str = f"  pool={pool_ms:.2f}" if pool_ms > 0 else ""

        sum_total += blk_ms; sum_attn += attn_ms; sum_mlp += mlp_ms
        print(f"  block[{i:02d}]    {blk_ms:>8.3f}  {attn_ms:>8.3f}  {mlp_ms:>8.3f}  "
              f"{attn_pct:>5.1f}%  {mlp_pct:>5.1f}%  {blk_type}{pool_str}")

    # Summary row
    a_pct = sum_attn / sum_total * 100 if sum_total > 0 else 0
    m_pct = sum_mlp  / sum_total * 100 if sum_total > 0 else 0
    blk_pct = sum_total / total_ms * 100 if total_ms > 0 else 0
    print(f"  {'-'*62}")
    print(f"  {'all blocks':<12} {sum_total:>8.1f}  {sum_attn:>8.1f}  {sum_mlp:>8.1f}  "
          f"{a_pct:>5.1f}%  {m_pct:>5.1f}%  ({blk_pct:.1f}% of total)")


def print_report_sam1(timers, n_runs, model_type, batch_size, encoder=None):
    n_blocks  = sum(1 for k in timers if k.startswith("block[") and k.endswith("]"))
    top_keys  = ["patch_embed"] + [f"block[{i:02d}]" for i in range(n_blocks)] + ["neck"]
    total_ms  = sum(_get(timers, k) for k in top_keys)

    print(f"\n{'='*70}")
    print(f"  SAM1 Encoder  |  model={model_type}  bs={batch_size}  runs={n_runs}  total={total_ms:.1f}ms")
    print(f"{'='*70}")
    _block_table(timers, "", total_ms, encoder=encoder)
    pe  = _get(timers, "patch_embed")
    nck = _get(timers, "neck")
    print(f"\n  patch_embed={pe:.2f}ms  neck={nck:.2f}ms")
    print(f"{'='*70}\n")


def print_report_sam2(timers, n_runs, config_name, batch_size, encoder=None):
    trunk_ms = _get(timers, "trunk")
    neck_ms  = _get(timers, "neck")
    total_ms = trunk_ms + neck_ms

    print(f"\n{'='*70}")
    print(f"  SAM2 Encoder  |  cfg={config_name}  bs={batch_size}  runs={n_runs}  total={total_ms:.1f}ms")
    print(f"{'='*70}")
    _block_table(timers, "trunk.", total_ms)
    pe  = _get(timers, "trunk.patch_embed")
    print(f"\n  trunk.patch_embed={pe:.2f}ms  neck={neck_ms:.2f}ms")
    print(f"{'='*70}\n")


def print_report_sam3(timers, n_runs, model_id, batch_size, vision_enc=None):
    backbone = vision_enc.backbone if vision_enc is not None else None
    n_layers = sum(1 for k in timers if k.startswith("layer[") and k.endswith("]"))

    top_keys = (
        ["embeddings"]
        + [f"layer[{i:02d}]" for i in range(n_layers)]
        + ["backbone.final_ln", "neck"]
    )
    total_ms = sum(_get(timers, k) for k in top_keys)

    global_idxs = set()
    if backbone is not None:
        global_idxs = set(backbone.config.global_attn_indexes)
        window_size = backbone.config.window_size
    else:
        window_size = "?"

    print(f"\n{'='*70}")
    print(f"  SAM3 Encoder  |  model={model_id}  bs={batch_size}  runs={n_runs}  total={total_ms:.1f}ms")
    print(f"{'='*70}")

    print(f"\n  {'layer':<12} {'total':>8}  {'attn':>8}  {'mlp':>8}  {'attn%':>6}  {'mlp%':>6}  {'type'}")
    print(f"  {'-'*62}")

    sum_total = sum_attn = sum_mlp = 0.0
    for i in range(n_layers):
        tag     = f"layer[{i:02d}]"
        blk_ms  = _get(timers, tag)
        attn_ms = _get(timers, tag + ".attn")
        mlp_ms  = _get(timers, tag + ".mlp")
        attn_pct = attn_ms / blk_ms * 100 if blk_ms > 0 else 0
        mlp_pct  = mlp_ms  / blk_ms * 100 if blk_ms > 0 else 0
        kind = "global" if i in global_idxs else f"win{window_size}"
        sum_total += blk_ms; sum_attn += attn_ms; sum_mlp += mlp_ms
        print(f"  layer[{i:02d}]    {blk_ms:>8.3f}  {attn_ms:>8.3f}  {mlp_ms:>8.3f}  "
              f"{attn_pct:>5.1f}%  {mlp_pct:>5.1f}%  {kind}")

    a_pct   = sum_attn  / sum_total * 100 if sum_total > 0 else 0
    m_pct   = sum_mlp   / sum_total * 100 if sum_total > 0 else 0
    blk_pct = sum_total / total_ms  * 100 if total_ms  > 0 else 0
    print(f"  {'-'*62}")
    print(f"  {'all layers':<12} {sum_total:>8.1f}  {sum_attn:>8.1f}  {sum_mlp:>8.1f}  "
          f"{a_pct:>5.1f}%  {m_pct:>5.1f}%  ({blk_pct:.1f}% of total)")

    emb_ms = _get(timers, "embeddings")
    fln_ms = _get(timers, "backbone.final_ln")
    nck_ms = _get(timers, "neck")
    print(f"\n  embeddings={emb_ms:.2f}ms  final_ln={fln_ms:.2f}ms  neck={nck_ms:.2f}ms")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ToMe helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_tome_patch(encoder, algo, ratio, margin, sparsity=0.0):
    from algo.sparsesam.patch.sam import apply_patch
    apply_patch(encoder, algo=algo, ratio=ratio, margin=margin,
                sparsity=sparsity)


def remove_tome_patch(encoder):
    from algo.sparsesam.patch.sam import ToMeSAMBlock, ToMeSAMAttention
    from segment_anything.modeling.image_encoder import Block, Attention
    for m in encoder.modules():
        if type(m) is ToMeSAMBlock:
            m.__class__ = Block
        elif type(m) is ToMeSAMAttention:
            m.__class__ = Attention
    encoder.__dict__.pop('forward',    None)
    encoder.__dict__.pop('tome_info',  None)


def apply_fa2_patch(encoder):
    """Patch attention with FlashAttentionForwardAmpere; ratio=1.0 disables token merging."""
    from algo.sparsesam.patch.sam import apply_patch
    apply_patch(encoder, algo='tome', ratio=1.0)


# remove_fa2_patch is identical to remove_tome_patch (same classes)
remove_fa2_patch = remove_tome_patch


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison report (SAM 1 only)
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_sam1(t_base, t_patched, model_type, batch_size, algo, ratio,
                          n_runs, encoder=None, patched_label=None):
    """Compact speedup table: baseline vs patched (ToMe/PiToMe/FA2)."""

    col = patched_label or algo[:4]          # short column header (≤6 chars)

    n_blocks = sum(1 for k in t_base if k.startswith("block[") and k.endswith("]"))
    top_keys = ["patch_embed"] + [f"block[{i:02d}]" for i in range(n_blocks)] + ["neck"]
    total_base    = sum(_get(t_base,    k) for k in top_keys)
    total_patched = sum(_get(t_patched, k) for k in top_keys)

    def spd(b, t): return b / t if t > 1e-6 else float('inf')
    def fmt_spd(s): return f"{s:.2f}x" if s != float('inf') else "  inf"

    W = 90
    print(f"\n{'='*W}")
    title = f"  Baseline vs {algo.upper()}"
    if ratio is not None:
        title += f"  ratio={ratio}"
    print(f"{title}  model={model_type}  bs={batch_size}  runs={n_runs}")
    print(f"{'='*W}")
    print(f"  {'block':<10} {'total':>12}  {'spd':>5} |"
          f"  {'attn':>12}  {'spd':>5} |"
          f"  {'mlp':>12}  {'ovhd':>6}  type")
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
        p_n1  = _get(t_patched, tag + ".norm1"); p_n2  = _get(t_patched, tag + ".norm2")
        ovhd  = max(0.0, p_blk - p_att - p_mlp - p_n1 - p_n2)
        sum_ovhd += ovhd

        blk_type = ""
        if encoder is not None:
            ws = getattr(encoder.blocks[i], 'window_size', None)
            if ws is not None:
                blk_type = "global" if ws == 0 else f"win{ws}"

        print(f"  block[{i:02d}]  {b_blk:>6.2f} {p_blk:>6.2f}  {fmt_spd(spd(b_blk,p_blk)):>5} |"
              f"  {b_att:>6.2f} {p_att:>6.2f}  {fmt_spd(spd(b_att,p_att)):>5} |"
              f"  {b_mlp:>6.2f} {p_mlp:>6.2f}  {ovhd:>6.2f}  {blk_type}")

    # ── summary ───────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_sam1(model_type, model_ckt, device):
    from segment_anything import sam_model_registry
    print(f"Loading SAM1/{model_type} from {model_ckt} ...")
    sam = sam_model_registry[model_type](checkpoint=model_ckt).to(device)
    return sam.image_encoder


def load_sam2(config_file, model_ckt, device):
    from sam2.build_sam import build_sam2
    # Hydra's search path is pkg://sam2, so only the bare filename is valid —
    # strip any directory prefix the user may have passed.
    config_name = os.path.basename(config_file)
    print(f"Loading SAM2 config={config_name} from {model_ckt} ...")
    model = build_sam2(
        config_file=config_name,
        ckpt_path=model_ckt,
        device=device,
        apply_postprocessing=False,
    )
    return model.image_encoder


def load_sam3(model_id, device, dtype=torch.float16, mock=False):
    from transformers import Sam3VisionModel, Sam3VisionConfig, Sam3ViTConfig
    if mock:
        print(f"Building mock SAM3 vision encoder (random weights, default config) ...")
        vit_cfg = Sam3ViTConfig(
            hidden_size=1024, intermediate_size=4096,
            num_hidden_layers=32, num_attention_heads=16,
            image_size=1008, patch_size=14, window_size=24,
            global_attn_indexes=[7, 15, 23, 31],
            layer_scale_init_value=1e-6,
        )
        vis_cfg = Sam3VisionConfig(
            backbone_config=vit_cfg, fpn_hidden_size=256,
            scale_factors=[4.0, 2.0, 1.0, 0.5],
        )
        enc = Sam3VisionModel(vis_cfg).to(dtype).to(device).eval()
    else:
        print(f"Loading SAM3 from HuggingFace: {model_id} ...")
        enc = Sam3VisionModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _run_profile(encoder, attach_fn, dummy, n_warmup, n_runs, label, pixel_values=False):
    """Attach hooks, warm up, measure, detach. Returns timers dict.

    pixel_values=True uses encoder(pixel_values=dummy) instead of encoder(dummy),
    which is required for HuggingFace SAM3.
    """
    timers, handles = attach_fn(encoder)
    encoder.eval()

    call = (lambda x: encoder(pixel_values=x)) if pixel_values else (lambda x: encoder(x))

    print(f"  Warming up [{label}] ({n_warmup} passes) ...")
    with torch.no_grad():
        for _ in range(n_warmup):
            call(dummy)
    for t in timers.values():
        t.reset()

    print(f"  Profiling  [{label}] ({n_runs} passes) ...")
    with torch.no_grad():
        for _ in range(n_runs):
            call(dummy)

    for h in handles:
        h.remove()
    return timers


def main():
    parser = argparse.ArgumentParser(
        description="Profile SAM / SAM-HQ / SAM-2 / SAM-2.1 encoder components",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--version', type=str, default='sam1', choices=['sam1', 'sam2', 'sam3'])

    # SAM 1
    parser.add_argument('--model-ckt',  type=str, default='./ckts/sam_hq_vit_l.pth')
    parser.add_argument('--model-type', type=str, default='vit_l',
                        choices=['vit_h', 'vit_l', 'vit_b'])

    # SAM 2
    parser.add_argument('--sam2-config', type=str, default='sam2.1_hq_hiera_l.yaml')

    # SAM 3
    parser.add_argument('--sam3-model', type=str, default='facebook/sam3',
                        help="HuggingFace model ID (e.g. facebook/sam3, facebook/sam3.1)")
    parser.add_argument('--sam3-mock',  action='store_true',
                        help="Use random-weight mock model (no HF download needed)")

    # Common
    parser.add_argument('--n-warmup',   type=int, default=5)
    parser.add_argument('--n-runs',     type=int, default=20)
    parser.add_argument('--img-size',   type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)

    # ToMe / PiToMe comparison (SAM 1 only)
    parser.add_argument('--tome-algo',   type=str, default='none',
                        choices=['none', 'tome', 'pitome', 'sparsesam', 'sparsesam_pitome',
                                 'sparsesam-dense'],
                        help="Run a second pass with this algo and show comparison table.")
    parser.add_argument('--tome-ratio',  type=float, default=0.9,
                        help="Token-keep ratio for ToMe/PiToMe (0 < ratio < 1).")
    parser.add_argument('--tome-margin', type=float, default=0.5,
                        help="PiToMe energy margin.")
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help="Block-sparse attention sparsity (0.0=dense, 0.9=90%% sparse). "
                             "Only used for sparsesam[-pitome].")
    parser.add_argument('--diagonal-width', type=int, default=1,
                        help="Diagonal band width for block-sparse attention "
                             "(1=exact diagonal, 3=±1 block). Only used for sparsesam[-pitome].")

    # FA2 comparison (SAM 1 only)
    parser.add_argument('--fa2', action='store_true',
                        help="Run a second pass with FlashAttentionForwardAmpere and show "
                             "comparison table (SAM1 only, no token merging).")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.", file=sys.stderr)
        sys.exit(1)

    device = 'cuda'

    # ── load ──────────────────────────────────────────────────────────────────
    if args.version == 'sam1':
        encoder      = load_sam1(args.model_type, args.model_ckt, device).half()
        attach_fn    = attach_timers_sam1
        report_fn    = lambda t, n: print_report_sam1(t, n, args.model_type, args.batch_size, encoder)
        img_size     = args.img_size
        pixel_values = False
    elif args.version == 'sam2':
        encoder      = load_sam2(args.sam2_config, args.model_ckt, device).half()
        attach_fn    = attach_timers_sam2
        report_fn    = lambda t, n: print_report_sam2(t, n, args.sam2_config, args.batch_size)
        img_size     = args.img_size
        pixel_values = False
    else:  # sam3
        encoder      = load_sam3(args.sam3_model, device, dtype=torch.float16, mock=args.sam3_mock)
        attach_fn    = attach_timers_sam3
        report_fn    = lambda t, n: print_report_sam3(t, n, args.sam3_model, args.batch_size, encoder)
        img_size     = 1008   # SAM3 native resolution
        pixel_values = True

    dummy = torch.randn(args.batch_size, 3, img_size, img_size, device=device, dtype=torch.float16)

    # ── baseline pass ─────────────────────────────────────────────────────────
    print("\n── Baseline (naive SAM) ──")
    t_base = _run_profile(encoder, attach_fn, dummy, args.n_warmup, args.n_runs, "baseline",
                          pixel_values=pixel_values)

    want_tome = args.tome_algo != 'none' and args.version == 'sam1'
    want_fa2  = args.fa2 and args.version == 'sam1'

    if (args.tome_algo != 'none' or args.fa2) and args.version != 'sam1':
        print("Note: --tome-algo and --fa2 are only supported for SAM1; ignoring.")


    if not want_tome and not want_fa2:
        report_fn(t_base, args.n_runs)
        return

    # ── FA2 pass (SAM 1 only) ─────────────────────────────────────────────────
    if want_fa2:
        print("\n── FA2 (FlashAttentionForwardAmpere, no token merging) ──")
        apply_fa2_patch(encoder)
        t_fa2 = _run_profile(encoder, attach_fn, dummy, args.n_warmup, args.n_runs, "fa2")
        remove_fa2_patch(encoder)

        print_comparison_sam1(
            t_base, t_fa2,
            model_type=args.model_type,
            batch_size=args.batch_size,
            algo="FA2",
            ratio=None,
            n_runs=args.n_runs,
            encoder=encoder,
            patched_label="fa2",
        )

    # ── ToMe / PiToMe pass (SAM 1 only) ──────────────────────────────────────
    if want_tome:
        sp_tag = (f" sparsity={args.sparsity} dw={args.diagonal_width}"
                  if args.tome_algo in ('sparsesam', 'sparsesam_pitome') else "")
        print(f"\n── {args.tome_algo.upper()} ratio={args.tome_ratio}{sp_tag} ──")
        apply_tome_patch(encoder, algo=args.tome_algo,
                         ratio=args.tome_ratio, margin=args.tome_margin,
                         sparsity=args.sparsity )
        t_tome = _run_profile(encoder, attach_fn, dummy, args.n_warmup, args.n_runs,
                              f"{args.tome_algo} r={args.tome_ratio}")
        remove_tome_patch(encoder)

        print_comparison_sam1(
            t_base, t_tome,
            model_type=args.model_type,
            batch_size=args.batch_size,
            algo=args.tome_algo,
            ratio=args.tome_ratio,
            n_runs=args.n_runs,
            encoder=encoder,
        )


if __name__ == '__main__':
    main()
