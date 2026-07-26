#!/usr/bin/env python3
"""Image-encoder latency benchmark on synthetic 1024x1024 noise.

Compares SAM-B against SparseSAM and the efficient-SAM family, and reports
speedup = (SAM-B ms/image) / (model ms/image) at each batch size.

No dataset, no COCO, no HQ44K — inputs are `torch.randn(B, 3, 1024, 1024)`,
so the only things needed on a fresh machine are the four upstream repos and
their checkpoints.

Usage
-----
    python speed_profile/bench_encoder_speed.py --batch-sizes 1 2 4 8

    # subset of models
    python speed_profile/bench_encoder_speed.py --models sam_b sparsesam_b

What is timed: one forward of the image encoder only (no prompt encoder, no
mask decoder), wrapped in CUDA events with a synchronize on each iteration.
FastSAM has no encoder/decoder split — it is a single-pass YOLOv8-seg network —
so its whole forward is timed and the CSV marks it `timed_component=full_forward`.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import math
import platform
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent


def add_path(p: Path) -> None:
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders. Each returns (forward_fn, meta) or raises.
# ─────────────────────────────────────────────────────────────────────────────

def load_sam_b(root: Path, ckpt: Optional[str], dtype, device, **_):
    add_path(root / "sam-hq")
    from segment_anything import sam_model_registry

    path = ckpt or str(root / "ckts/sam_hq_vit_b.pth")
    sam = sam_model_registry["vit_b"](checkpoint=path)
    enc = sam.image_encoder.to(device=device, dtype=dtype).eval()
    return (lambda x: enc(x)), {"checkpoint": path, "timed_component": "image_encoder",
                                "params_m": sum(p.numel() for p in enc.parameters()) / 1e6}


def block_mask_density(n_tokens: int, ratio: float, n_block: int, band_width: int,
                       exact_local: bool = False) -> float:
    """Fraction of column-blocks the cute kernel attends, replicating
    `sam.py::make_A_mask` (plus the `exact_local` variant below)."""
    num = math.ceil(n_tokens / n_block)
    t = np.zeros((num, num), dtype=np.float32)
    half = band_width // 2
    for k in range(-half, band_width - half):
        i = np.arange(max(0, -k), min(num, num - k))
        t[i, i + k] = 1.0
    n_keep = int(ratio * num)
    cols = n_keep if (exact_local and band_width == 0) else n_keep - band_width + 1
    if cols > 0:
        t[:, :cols] = 1.0
    return float(t.mean())


_ORIG_MAKE_A_MASK = None


def _set_local_exact_density(sam_mod, enabled: bool) -> None:
    """Install or remove the exact-local-density mask, clearing the mask cache.

    Both SparseSAM rows live in one process, so the patch must be toggled per
    model rather than installed once.
    """
    global _ORIG_MAKE_A_MASK
    if enabled:
        if _ORIG_MAKE_A_MASK is None:
            _ORIG_MAKE_A_MASK = _patch_local_exact_density(sam_mod)
    elif _ORIG_MAKE_A_MASK is not None:
        sam_mod.make_A_mask = _ORIG_MAKE_A_MASK
        _ORIG_MAKE_A_MASK = None
    sam_mod._SPARSE_MASK_CACHE.clear()


def _patch_local_exact_density(sam_mod):
    """Make windowed blocks hit the nominal density instead of one block above it.

    `make_A_mask` sets the keep bar to `n_keep_cols - band_width + 1`. The
    `- band_width` term compensates for the diagonal; local blocks are built
    with `band_width=0` (sam.py:178), so the `+1` is uncompensated and adds one
    extra column-block. On a 14x14 window that is only 4 blocks wide, that is
    +25 percentage points: keep-ratio 0.25 realises as 0.50.

    This replacement drops the `+1` for local blocks only, so ratio 0.25 gives
    exactly 1 of 4 column-blocks. Global blocks are untouched.

    ACCURACY WARNING: with band_width=0 there is no diagonal, so at ratio 0.25
    only row-block 0 lies inside the keep bar — 132 of 196 tokens per window
    (67%) can no longer attend their own block. Use this for latency
    measurement; any accuracy number must be re-measured, not carried over.
    """
    from cutlass.cute.runtime import from_dlpack

    original = sam_mod.make_A_mask

    def make_A_mask(B, H, T, ratio, m_block, n_block,
                    band_width: int = sam_mod._DIAG_BAND_WIDTH,
                    keep_bar_scale: float = sam_mod._KEEP_BAR_SCALE,
                    device="cuda"):
        if band_width != 0:                       # global blocks: unchanged
            return original(B, H, T, ratio, m_block, n_block,
                            band_width, keep_bar_scale, device)
        num_m = math.ceil(T / m_block)
        num_n = math.ceil(T / n_block)
        t = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)
        n_keep_cols = max(1, int(ratio * num_n * keep_bar_scale))
        t[:, :, :, :n_keep_cols] = 1
        return from_dlpack(t, assumed_align=4), t

    sam_mod.make_A_mask = make_A_mask
    return original


def load_sparsesam_b(root: Path, ckpt: Optional[str], dtype, device, ratio: float = 0.25,
                     mlp_merge: bool = True, local_exact_density: bool = False, **_):
    add_path(root / "sam-hq")
    add_path(root)
    add_path(root / "PiToMe")
    from segment_anything import sam_model_registry
    import PiToMe.algo.sparsesam.sam as sam_mod

    if dtype is not torch.float16:
        raise RuntimeError("SparseSAM's block-sparse cute kernel is fp16-only")

    _set_local_exact_density(sam_mod, local_exact_density)

    path = ckpt or str(root / "ckts/sam_hq_vit_b.pth")
    sam = sam_model_registry["vit_b"](checkpoint=path)
    enc = sam.image_encoder.to(device=device, dtype=dtype).eval()
    sam_mod.apply_patch(enc, algo="tome", ratio=ratio, mlp_merge=mlp_merge)

    # Realised density: what the mask actually attends, per block type.
    win = next((b.window_size for b in enc.blocks if b.window_size > 0), 14)
    grid = enc.img_size // enc.patch_embed.proj.stride[0]
    d_local = block_mask_density(win * win, ratio, sam_mod._FA2_N_BLOCK_LOCAL, 0,
                                 local_exact_density)
    d_global = block_mask_density(grid * grid, ratio, sam_mod._FA2_N_BLOCK_GLOBAL,
                                  sam_mod._DIAG_BAND_WIDTH)
    n_glob = sum(1 for b in enc.blocks if b.window_size == 0)
    n_loc = len(enc.blocks) - n_glob

    return (lambda x: enc(x)), {
        "checkpoint": path, "timed_component": "image_encoder",
        "ratio_keep": ratio, "sparsity": 1.0 - ratio, "mlp_merge": mlp_merge,
        "local_exact_density": local_exact_density,
        "density_local": d_local, "density_global": d_global,
        "density_mean": (n_loc * d_local + n_glob * d_global) / len(enc.blocks),
        "params_m": sum(p.numel() for p in enc.parameters()) / 1e6}


def load_sparsesam_dense_kernel(root: Path, ckpt: Optional[str], dtype, device, **_):
    """Control row: same cute FA2 kernel and same permutation plumbing, but a
    fully dense mask and no MLP routing (ratio = 1.0).

    Separates "our kernel is faster than SAM's eager attention" from "our
    sparsity removes work". Speedup of this row over sam_b is the kernel
    contribution; the remaining gap up to sparsesam_b is the sparsity."""
    add_path(root / "sam-hq")
    add_path(root)
    add_path(root / "PiToMe")
    from segment_anything import sam_model_registry
    import PiToMe.algo.sparsesam.sam as sam_mod

    if dtype is not torch.float16:
        raise RuntimeError("SparseSAM's block-sparse cute kernel is fp16-only")

    _set_local_exact_density(sam_mod, False)

    path = ckpt or str(root / "ckts/sam_hq_vit_b.pth")
    sam = sam_model_registry["vit_b"](checkpoint=path)
    enc = sam.image_encoder.to(device=device, dtype=dtype).eval()
    sam_mod.apply_patch(enc, algo="tome", ratio=1.0, mlp_merge=False)
    return (lambda x: enc(x)), {
        "checkpoint": path, "timed_component": "image_encoder",
        "ratio_keep": 1.0, "sparsity": 0.0, "mlp_merge": False,
        "density_local": 1.0, "density_global": 1.0, "density_mean": 1.0,
        "params_m": sum(p.numel() for p in enc.parameters()) / 1e6}


def load_mobilesam(root: Path, ckpt: Optional[str], dtype, device, **_):
    add_path(root / "MobileSAM")
    from mobile_sam import sam_model_registry

    path = ckpt or str(root / "MobileSAM/weights/mobile_sam.pt")
    sam = sam_model_registry["vit_t"](checkpoint=path)
    enc = sam.image_encoder.to(device=device, dtype=dtype).eval()
    return (lambda x: enc(x)), {"checkpoint": path, "timed_component": "image_encoder",
                                "params_m": sum(p.numel() for p in enc.parameters()) / 1e6}


def load_efficientsam(root: Path, ckpt: Optional[str], dtype, device,
                      variant: str = "vitt", **_):
    add_path(root / "EfficientSAM")
    from efficient_sam.efficient_sam import build_efficient_sam

    dims = {"vitt": (192, 3), "vits": (384, 6)}
    path = ckpt or str(root / f"EfficientSAM/weights/efficient_sam_{variant}.pt")
    model = build_efficient_sam(*dims[variant], checkpoint=path)
    enc = model.image_encoder.to(device=device, dtype=dtype).eval()
    return (lambda x: enc(x)), {"checkpoint": path, "timed_component": "image_encoder",
                                "variant": variant,
                                "params_m": sum(p.numel() for p in enc.parameters()) / 1e6}


def load_efficientvit_sam(root: Path, ckpt: Optional[str], dtype, device,
                          variant: str = "efficientvit-sam-xl1", **_):
    # efficientvit builds its prompt encoder / mask decoder from the upstream
    # `segment_anything` package; sam-hq's fork provides the same symbols.
    add_path(root / "sam-hq")
    add_path(root / "efficientvit")
    _install_efficientvit_stubs()
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model

    path = ckpt or str(root / "ckts/efficientvit_sam_xl1.pt")
    model = create_efficientvit_sam_model(variant, pretrained=True, weight_url=path)

    # `UpSampleLayer` (models/nn/ops.py:95-100) disables autocast and force-casts
    # fp16 inputs to fp32, so hard-.half() weights hit a dtype mismatch at the
    # next conv. Upstream runs this model in fp32 weights under autocast, so
    # that is its fp16 path — recorded as precision_mode in the CSV.
    if dtype is torch.float16:
        enc = model.image_encoder.to(device=device, dtype=torch.float32).eval()

        def fwd(x):
            with torch.autocast("cuda", dtype=torch.float16):
                return enc(x)

        mode = "autocast_fp16"
    else:
        enc = model.image_encoder.to(device=device, dtype=dtype).eval()
        fwd = lambda x: enc(x)          # noqa: E731
        mode = "weights_fp32"

    return fwd, {"checkpoint": path, "timed_component": "image_encoder",
                 "variant": variant, "precision_mode": mode,
                 "params_m": sum(p.numel() for p in enc.parameters()) / 1e6}


def load_fastsam(root: Path, ckpt: Optional[str], dtype, device, **_):
    """FastSAM is YOLOv8-seg: no encoder/decoder split, so the full forward is
    what gets timed."""
    add_path(root / "FastSAM")           # its vendored `ultralytics` must win
    from fastsam import FastSAM

    path = ckpt or str(root / "FastSAM/.weights/FastSAM-x.pt")
    with _torch_load_full_pickle():
        net = FastSAM(path).model
    net = net.to(device=device, dtype=dtype).eval()
    return (lambda x: net(x)), {"checkpoint": path, "timed_component": "full_forward",
                                "params_m": sum(p.numel() for p in net.parameters()) / 1e6}


@contextlib.contextmanager
def _torch_load_full_pickle():
    """Load a checkpoint with `weights_only=False`.

    SECURITY: this executes arbitrary code contained in the checkpoint file.
    Ultralytics ships FastSAM-x.pt as a pickled `SegmentationModel` object, not
    a state_dict, so PyTorch >= 2.6 refuses it under the `weights_only=True`
    default. Only use with checkpoints you obtained yourself from the official
    FastSAM release.
    """
    original = torch.load

    def patched(*a, **kw):
        kw["weights_only"] = False
        return original(*a, **kw)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def _install_efficientvit_stubs():
    """efficientvit imports onnx/onnxsim at module scope for its export path."""
    import importlib.machinery
    import importlib.util
    import types

    for name in ("onnx", "onnxsim"):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        stub = types.ModuleType(name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        def _missing(*a, _n=name, **k):
            raise ImportError(f"{_n} is only needed for ONNX export")
        for attr in ("load_model", "save", "simplify"):
            setattr(stub, attr, _missing)
        sys.modules[name] = stub


LOADERS: Dict[str, Callable] = {
    "sam_b": load_sam_b,
    "sparsesam_dense_kernel": load_sparsesam_dense_kernel,
    "sparsesam_b": load_sparsesam_b,
    "fastsam": load_fastsam,
    "mobilesam": load_mobilesam,
    "efficientsam": load_efficientsam,
    "efficientvit_sam": load_efficientvit_sam,
}

BASELINE = "sam_b"


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def time_forward(fn: Callable, x: torch.Tensor, iters: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    out = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        out[i] = start.elapsed_time(end)
    return out


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(
        description="Image-encoder speed benchmark on synthetic 1024x1024 noise.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="Directory holding sam-hq/ PiToMe/ FastSAM/ MobileSAM/\n"
                        "EfficientSAM/ efficientvit/ and ckts/.")
    p.add_argument("--models", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--num-samples", type=int, default=100,
                   help="Images per (model, batch size). Iterations = ceil(n / batch).")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--ratio", type=float, default=0.25,
                   help="SparseSAM keep ratio. 0.25 = 25%% density / 75%% sparsity.")
    p.add_argument("--no-mlp-merge", dest="mlp_merge", action="store_false",
                   help="SparseSAM: sparse attention only, full MLP.")
    p.add_argument("--local-exact-density", action="store_true",
                   help="Windowed blocks keep exactly int(ratio*n_blocks) column-blocks\n"
                        "instead of one more (sam.py's keep bar is n_keep - band + 1 and\n"
                        "local blocks use band=0, so ratio 0.25 realises as 0.50).\n"
                        "Latency-only: it also removes self-attention from 3 of 4 row-\n"
                        "blocks per window, so accuracy must be re-measured.")
    p.add_argument("--efficientsam-variant", default="vitt", choices=["vitt", "vits"])
    p.add_argument("--efficientvit-variant", default="efficientvit-sam-xl1")
    for name in LOADERS:
        p.add_argument(f"--{name.replace('_', '-')}-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for a latency benchmark.")

    root = Path(args.root).resolve()
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {gpu_name()}   torch {torch.__version__}   dtype {args.dtype}   "
          f"input {args.image_size}x{args.image_size}")
    print(f"root: {root}\n")

    rows: List[Dict] = []
    skipped: Dict[str, str] = {}

    for name in args.models:
        try:
            fn, meta = LOADERS[name](
                root,
                getattr(args, f"{name}_checkpoint"),
                dtype, device,
                ratio=args.ratio, mlp_merge=args.mlp_merge,
                local_exact_density=args.local_exact_density,
                variant=(args.efficientsam_variant if name == "efficientsam"
                         else args.efficientvit_variant),
            )
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
            print(f"[skip] {name}: {skipped[name]}")
            continue

        for bs in args.batch_sizes:
            iters = max(1, math.ceil(args.num_samples / bs))
            x = torch.randn(bs, 3, args.image_size, args.image_size,
                            device=device, dtype=dtype)
            try:
                t = time_forward(fn, x, iters, args.warmup)
            except Exception as exc:
                print(f"[fail] {name} bs={bs}: {type(exc).__name__}: {exc}")
                continue
            finally:
                del x
                torch.cuda.empty_cache()

            rows.append({
                "model": name,
                "batch_size": bs,
                "num_images": iters * bs,
                "iters": iters,
                "batch_mean_ms": float(t.mean()),
                "batch_std_ms": float(t.std()),
                "encoder_per_image_mean_ms": float(t.mean() / bs),
                "encoder_per_image_std_ms": float(t.std() / bs),
                "encoder_per_image_median_ms": float(np.median(t) / bs),
                "throughput_imgs_per_sec": 1000.0 * bs / float(t.mean()),
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "dtype": args.dtype,
                "image_size": args.image_size,
                "gpu": gpu_name(),
                "torch": torch.__version__,
                "host": platform.node(),
                "timestamp": datetime.datetime.now().isoformat(),
                **meta,
            })
            print(f"  {name:<18} bs={bs:<3} {t.mean() / bs:8.2f} ms/img  "
                  f"({t.mean():8.2f} ms/batch)")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No model produced a measurement.")

    base = {r["batch_size"]: r["encoder_per_image_mean_ms"]
            for r in rows if r["model"] == BASELINE}
    for r in rows:
        b = base.get(r["batch_size"])
        r["speedup_vs_sam_b"] = (b / r["encoder_per_image_mean_ms"]) if b else ""

    out = Path(args.output_csv) if args.output_csv else (
        HERE / f"encoder_speed_{gpu_name().replace(' ', '_')}_"
               f"{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        fields = list(dict.fromkeys(fields + list(r.keys())))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'=' * 92}")
    print(f"Encoder latency — {gpu_name()} — {args.dtype} — "
          f"{args.image_size}x{args.image_size} noise")
    print("=" * 92)
    print(f"{'model':<18}{'bs':>4}{'ms/img':>10}{'std':>8}{'img/s':>10}"
          f"{'speedup':>10}{'mem MB':>9}  component")
    print("-" * 92)
    for bs in args.batch_sizes:
        for r in [x for x in rows if x["batch_size"] == bs]:
            sp = r["speedup_vs_sam_b"]
            sp_s = f"{sp:.2f}x" if isinstance(sp, float) else "—"
            print(f"{r['model']:<18}{bs:>4}{r['encoder_per_image_mean_ms']:>10.2f}"
                  f"{r['encoder_per_image_std_ms']:>8.2f}"
                  f"{r['throughput_imgs_per_sec']:>10.1f}{sp_s:>10}"
                  f"{r['peak_memory_mb']:>9.0f}  {r['timed_component']}")
        print("-" * 92)
    if skipped:
        print("skipped: " + "; ".join(f"{k} ({v.split(':')[0]})" for k, v in skipped.items()))
    print(f"CSV → {out}")
    return rows


if __name__ == "__main__":
    main()
