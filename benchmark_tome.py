#!/usr/bin/env python3
"""
Benchmark SAM-1 encoder with token merging (ToMe / PiToMe / SparseSAM / GradToMe).

Sweeps over (algo × ratio × batch_size) and reports throughput, latency,
memory, and mIoU for each combination, plus a final comparison table against
the uncompressed baseline.

Available algorithms
--------------------
  none             — uncompressed baseline
  tome             — bipartite soft matching, post-attn merge  (algo.tome)
  pitome           — energy-score ranked matching, post-attn   (algo.tome)
  sparsesam        — bipartite matching, pre-attn / Hilbert    (algo.sparsesam)
  sparsesam_pitome — PiToMe variant of sparsesam               (algo.sparsesam)
  gradtome         — gradient-guided matching, Hilbert         (algo.gradtome)
  gradtome_pitome  — PiToMe variant of gradtome                (algo.gradtome)

WARNING: sparsesam and gradtome patch files contain debug breakpoint() calls.
Remove them from the respective patch/sam.py files before running a sweep.

Usage:
    # Baseline only
    python benchmark_tome.py --algos none --batch-sizes 1 2 4

    # Classic ToMe sweep
    python benchmark_tome.py --algos none tome --ratios 1.0 0.9 0.8 0.7

    # Full sweep across all methods
    python benchmark_tome.py \
        --algos none tome pitome sparsesam gradtome \
        --ratios 0.9 0.8 0.7 0.6 \
        --batch-sizes 1 2 4 8 \
        --num-samples 100 \
        --model-ckt ./ckts/sam_hq_vit_l.pth \
        --model-type vit_l
"""

import os
import gc
import sys
import time
import argparse
import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

# ── SAM imports ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam-hq'))
from segment_anything import SamPredictor, sam_model_registry

# ── Token merging patches ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PiToMe'))
from algo.tome.patch.sam              import apply_patch as _patch_tome
from algo.sparsesam.patch.sam         import apply_patch as _patch_sparsesam
from algo.sparsesam.patch.sam_random  import apply_patch as _patch_sparsesam_random
from algo.gradtome.patch.sam          import apply_patch as _patch_gradtome
from algo.gradtome.patch.sam_hilbert  import apply_patch as _patch_gradtome_hilbert

# Maps benchmark algo name → (apply_patch_fn, internal algo arg passed to fn)
ALGO_REGISTRY = {
    "tome":                   (_patch_tome,              "tome"),
    "pitome":                 (_patch_tome,              "pitome"),
    "sparsesam":              (_patch_sparsesam,         "tome"),
    "sparsesam_pitome":       (_patch_sparsesam,         "pitome"),
    "sparsesam_random":       (_patch_sparsesam_random,  "sparsesam_random"),
    "gradtome":               (_patch_gradtome,          "tome"),
    "gradtome_pitome":        (_patch_gradtome,          "pitome"),
    "gradtome_hilbert":       (_patch_gradtome_hilbert,  "tome"),
}
VALID_ALGOS = ["none"] + list(ALGO_REGISTRY.keys())

# ── Local imports ─────────────────────────────────────────────────────────────
from sam_engine import get_default_datasets
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou


# ─────────────────────────────────────────────────────────────────────────────
# Token merging helpers
# ─────────────────────────────────────────────────────────────────────────────

_SPARSESAM_ALGOS = {"sparsesam", "sparsesam_pitome", "sparsesam_random"}

def apply_tome(encoder, algo: str, ratio: float, margin: float = 0.5,
               sparsity: float = 0.0):
    """Patch encoder in-place using the algo specified in ALGO_REGISTRY.

    algo='none' or ratio>=1.0 are no-ops.
    sparsity/diagonal_width are forwarded to sparsesam-family algos only.
    """
    if algo == "none" or ratio >= 1.0:
        return
    if algo not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo {algo!r}. Valid choices: {VALID_ALGOS}")
    patch_fn, internal_algo = ALGO_REGISTRY[algo]
    if algo in _SPARSESAM_ALGOS:
        patch_fn(encoder, algo=internal_algo, ratio=ratio, margin=margin,
                 sparsity=sparsity)
    else:
        patch_fn(encoder, algo=internal_algo, ratio=ratio, margin=margin)

def update_ratio(encoder, ratio: float):
    """Update merge ratio without re-patching (works if already patched)."""
    if hasattr(encoder, 'tome_info'):
        encoder.tome_info['ratio'] = ratio


def remove_tome(encoder):
    """
    Undo any token-merging patch by restoring original Block and Attention.
    Handles all three patch variants (tome, sparsesam, gradtome).
    Needed when sweeping multiple algos/ratios on the same model instance.
    """
    from algo.tome.patch.sam              import ToMeSAMBlock as _B_t,  ToMeSAMAttention as _A_t
    from algo.sparsesam.patch.sam         import ToMeSAMBlock as _B_s,  ToMeSAMAttention as _A_s
    from algo.sparsesam.patch.sam_random  import ToMeSAMBlockRandom as _B_sr, ToMeSAMAttentionRandom as _A_sr
    from algo.gradtome.patch.sam          import ToMeSAMBlock as _B_g,  ToMeSAMAttention as _A_g
    from algo.gradtome.patch.sam_hilbert  import ToMeSAMBlock as _B_gh, ToMeSAMAttention as _A_gh
    from segment_anything.modeling.image_encoder import Block, Attention

    _patched_blocks = (_B_t, _B_s, _B_sr, _B_g, _B_gh)
    _patched_attns  = (_A_t, _A_s, _A_sr, _A_g, _A_gh)

    for module in encoder.modules():
        if type(module) in _patched_blocks:
            module.__class__ = Block
        elif type(module) in _patched_attns:
            module.__class__ = Attention

    if hasattr(encoder, 'tome_info'):
        del encoder.tome_info
    # The patched forward is stored in the instance __dict__, not in the class,
    # so nn.Module.__delattr__ can't find it. Bypass it directly.
    encoder.__dict__.pop('forward', None)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (identical to benchmark_batch_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def custom_collate_fn(batch):
    ori_ims = [item['ori_im'] for item in batch]
    collated = {}
    for key in batch[0].keys():
        if key == 'ori_im':
            collated[key] = ori_ims
        elif key in ('ori_im_path', 'ori_gt_path'):
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def build_dataloader(datasets_config, dataset_idx, batch_size):
    valid_im_gt_list = get_im_gt_name_dict([datasets_config[dataset_idx]], flag="valid")
    dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark
# ─────────────────────────────────────────────────────────────────────────────

def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def process_batch(predictor, images, labels_boxes, labels_ori):
    """One encoder forward + per-image decode. Returns timing and quality metrics."""
    batch_size = images.shape[0]
    device = predictor.device

    images        = images.to(device)
    labels_boxes  = labels_boxes.to(device, dtype=torch.float16)
    labels_ori    = labels_ori.to(device)

    transformed = predictor.model.preprocess(images).half()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        features, interm_features = predictor.model.image_encoder(transformed)
    torch.cuda.synchronize()
    encoder_ms = (time.perf_counter() - t0) * 1000

    ious, b_ious = [], []
    for i in range(batch_size):
        predictor.features         = features[i:i+1]
        predictor.interm_features  = [f[i:i+1] for f in interm_features]
        predictor.original_size    = (images.shape[2], images.shape[3])
        predictor.input_size       = tuple(transformed.shape[-2:])
        predictor.is_image_set     = True
        try:
            masks, _, _ = predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=labels_boxes[i:i+1], hq_token_only=True,
            )
            ious.append(compute_iou(masks, labels_ori[i:i+1]))
            b_ious.append(compute_boundary_iou(masks, labels_ori[i:i+1]))
        except Exception as e:
            print(f"  decode error image {i}: {e}")
            ious.append(torch.tensor(0.0, device=device))
            b_ious.append(torch.tensor(0.0, device=device))

    return {
        'encoder_ms':           encoder_ms,
        'encoder_per_image_ms': encoder_ms / batch_size,
        'iou':                  torch.mean(torch.stack(ious)).item(),
        'boundary_iou':         torch.mean(torch.stack(b_ious)).item(),
    }


def run_one(predictor, batch_size, dataloader, num_samples, label="", warmup_batches: int = 3) -> Dict:
    """Run benchmark for a single (algo, ratio, batch_size) configuration."""
    reset_memory()

    # Warmup: run a few batches outside the timer so CUDA kernels are compiled
    # and GPU caches are hot before measurement begins. Without this, the first
    # algo in the sweep (often a patched variant) pays kernel-compilation cost
    # and appears artificially slower than the baseline.
    device = predictor.device
    warmup_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            data_val = next(warmup_iter)
        except StopIteration:
            break
        images = data_val['image'].to(device)
        transformed = predictor.model.preprocess(images).half()
        with torch.no_grad():
            predictor.model.image_encoder(transformed)
    torch.cuda.synchronize()
    del warmup_iter

    enc_times, enc_per_img, ious, b_ious = [], [], [], []
    total_images = 0
    pbar = tqdm(
        total=min(len(dataloader), num_samples // batch_size),
        desc=label or f"bs={batch_size}",
        leave=False,
    )
    t_overall = time.perf_counter()

    for data_val in dataloader:
        if total_images >= num_samples:
            break
        images      = data_val['image']
        labels_val  = data_val['label']
        labels_ori  = data_val['ori_label']
        bs_actual   = images.shape[0]

        if labels_val.dim() == 4:
            boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])
        else:
            boxes = misc.masks_to_boxes(labels_val[0:1, :, :])

        try:
            m = process_batch(predictor, images, boxes, labels_ori)
            enc_times.append(m['encoder_ms'])
            enc_per_img.append(m['encoder_per_image_ms'])
            ious.append(m['iou'])
            b_ious.append(m['boundary_iou'])
        except Exception as e:
            print(f"  batch error: {e}")
            continue

        total_images += bs_actual
        pbar.update(1)

    pbar.close()
    overall_sec = time.perf_counter() - t_overall

    enc_times    = np.array(enc_times)
    enc_per_img  = np.array(enc_per_img)

    mem = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem = {
            'peak_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'peak_memory_reserved_mb':  torch.cuda.max_memory_reserved()  / 1024**2,
        }

    return {
        'batch_size':                 batch_size,
        'num_images':                 total_images,
        'throughput_imgs_per_sec':    total_images / overall_sec,
        'encoder_batch_mean_ms':      float(np.mean(enc_times)),
        'encoder_batch_std_ms':       float(np.std(enc_times)),
        'encoder_per_image_mean_ms':  float(np.mean(enc_per_img)),
        'encoder_per_image_std_ms':   float(np.std(enc_per_img)),
        'miou':                       float(np.mean(ious)),
        'miou_std':                   float(np.std(ious)),
        'boundary_iou':               float(np.mean(b_ious)),
        'boundary_iou_std':           float(np.std(b_ious)),
        **mem,
        'timestamp': datetime.datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    predictor: SamPredictor,
    algos: List[str],
    ratios: List[float],
    batch_sizes: List[int],
    num_samples: int,
    datasets_config: List[Dict],
    dataset_idx: int,
    margin: float,
    sparsity: float = 0.0,
    diagonal_width: int = 1,
) -> List[Dict]:

    encoder = predictor.model.image_encoder
    all_results = []

    # ── build run list: (algo, ratio) pairs ──────────────────────────────────
    runs = []
    for algo in algos:
        if algo == "none":
            runs.append(("none", 1.0))
        else:
            for ratio in ratios:
                if ratio < 1.0:
                    runs.append((algo, ratio))

    # ── deduplicate baseline ──────────────────────────────────────────────────
    seen = set()
    deduped = []
    for r in runs:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    runs = deduped

    print(f"\n{'='*80}")
    print(f"Sweep: {len(runs)} algo/ratio configs × {len(batch_sizes)} batch sizes "
          f"= {len(runs)*len(batch_sizes)} total runs")
    print(f"{'='*80}\n")

    for algo, ratio in runs:
        # ── patch / unpatch encoder ──────────────────────────────────────────
        remove_tome(encoder)
        if algo != "none":
            apply_tome(encoder, algo=algo, ratio=ratio, margin=margin,
                       sparsity=sparsity)

        n_tokens = int(64 * 64 * ratio) if algo != "none" else 64 * 64

        for batch_size in batch_sizes:
            sp_tag = f" sp={sparsity:.2f} dw={diagonal_width}" if algo in _SPARSESAM_ALGOS else ""
            label = f"{algo} r={ratio:.2f}{sp_tag} bs={batch_size}"
            print(f"\n── {label} ──")

            dataloader = build_dataloader(datasets_config, dataset_idx, batch_size)
            result = run_one(predictor, batch_size, dataloader, num_samples, label=label)
            del dataloader

            result.update({
                'algo':            algo,
                'ratio':           ratio,
                'n_tokens_kept':   n_tokens,
                'sparsity':        sparsity if algo in _SPARSESAM_ALGOS else 0.0,
                'diagonal_width':  diagonal_width if algo in _SPARSESAM_ALGOS else 1,
            })
            all_results.append(result)

            # ── wandb log ────────────────────────────────────────────────────
            prefix = f"{algo}_r{ratio:.2f}_bs{batch_size}"
            wandb.log({
                f'{prefix}/throughput':           result['throughput_imgs_per_sec'],
                f'{prefix}/encoder_mean_ms':      result['encoder_batch_mean_ms'],
                f'{prefix}/encoder_per_img_ms':   result['encoder_per_image_mean_ms'],
                f'{prefix}/miou':                 result['miou'],
                f'{prefix}/boundary_iou':         result['boundary_iou'],
                f'{prefix}/peak_memory_mb':       result.get('peak_memory_allocated_mb', 0),
            })

            print(f"  throughput={result['throughput_imgs_per_sec']:.2f} img/s  "
                  f"enc/img={result['encoder_per_image_mean_ms']:.1f}ms  "
                  f"mIoU={result['miou']:.4f}  "
                  f"mem={result.get('peak_memory_allocated_mb', 0):.0f}MB")

    # restore encoder to clean state
    remove_tome(encoder)
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]):
    # Find baseline (algo=none) for each batch_size
    baselines = {r['batch_size']: r for r in results if r['algo'] == 'none'}

    print(f"\n{'='*100}")
    print("SUMMARY — token merging vs baseline")
    print(f"{'='*100}")
    print(
        f"{'Algo':<8} {'Ratio':>6} {'BS':>4} | "
        f"{'Throughput':>12} {'Δspeedup':>10} | "
        f"{'Enc/img(ms)':>12} | "
        f"{'Mem(MB)':>9} | "
        f"{'mIoU':>8} {'ΔmIoU':>8} | "
        f"{'B-IoU':>8}"
    )
    print("-" * 100)

    for r in results:
        bs  = r['batch_size']
        b   = baselines.get(bs)
        spd = r['throughput_imgs_per_sec']
        d_spd = (spd / b['throughput_imgs_per_sec'] - 1) * 100 if b else 0
        d_iou = r['miou'] - b['miou'] if b else 0

        sign_spd = "+" if d_spd >= 0 else ""
        sign_iou = "+" if d_iou >= 0 else ""

        print(
            f"{r['algo']:<8} {r['ratio']:>6.2f} {bs:>4} | "
            f"{spd:>12.2f} {sign_spd}{d_spd:>9.1f}% | "
            f"{r['encoder_per_image_mean_ms']:>12.2f} | "
            f"{r.get('peak_memory_allocated_mb', 0):>9.0f} | "
            f"{r['miou']:>8.4f} {sign_iou}{d_iou:>7.4f} | "
            f"{r['boundary_iou']:>8.4f}"
        )

    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SAM-1 with ToMe / PiToMe token merging',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Model
    parser.add_argument('--model-ckt',  type=str, default='./ckts/sam_hq_vit_l.pth')
    parser.add_argument('--model-type', type=str, default='vit_l',
                        choices=['vit_h', 'vit_l', 'vit_b'])

    # Sweep axes
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['none', 'tome', 'pitome'],
                        choices=VALID_ALGOS,
                        help=(
                            "Algorithms to sweep. 'none' = uncompressed baseline.\n"
                            "Available: " + ", ".join(VALID_ALGOS) + "\n"
                            "  tome/pitome       — algo.tome patch (post-attn merge)\n"
                            "  sparsesam[_pitome]— algo.sparsesam patch (pre-attn / Hilbert)\n"
                            "  gradtome[_pitome] — algo.gradtome patch (gradient-guided / Hilbert)"
                        ))
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=[0.9, 0.8, 0.7],
                        help='Token-keep ratios to sweep (ignored for algo=none).')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[1, 2, 4, 8])
    parser.add_argument('--num-samples', type=int, default=100)

    # PiToMe
    parser.add_argument('--margin', type=float, default=0.5,
                        help='PiToMe energy margin (ignored for ToMe).')

    # SparseSAM block-sparse attention
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help='Fraction of key blocks to skip in block-sparse attention '
                             '(0.0 = dense, 0.9 = 90%% sparse). Only used for sparsesam[-pitome].')
    parser.add_argument('--diagonal-width', type=int, default=1,
                        help='Width of the diagonal band kept in block-sparse attention '
                             '(1 = exact diagonal, 3 = ±1 block). Only used for sparsesam[-pitome].')

    # Dataset
    parser.add_argument('--dataset-idx', type=int, default=0,
                        help='Index into get_default_datasets() list.')

    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='sam-tome-benchmark')
    parser.add_argument('--wandb-run-name', type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    else:
        wandb.init(mode='disabled')

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading SAM {args.model_type} from {args.model_ckt} ...")
    sam       = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to('cuda').half()
    predictor = SamPredictor(sam)

    datasets = get_default_datasets()

    # ── run sweep ─────────────────────────────────────────────────────────────
    results = run_sweep(
        predictor       = predictor,
        algos           = args.algos,
        ratios          = args.ratios,
        batch_sizes     = args.batch_sizes,
        num_samples     = args.num_samples,
        datasets_config = datasets,
        dataset_idx     = args.dataset_idx,
        margin          = args.margin,
        sparsity        = args.sparsity,
    )

    # ── save CSV ──────────────────────────────────────────────────────────────
    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv = os.path.join(args.output_dir, f'tome_benchmark_{ts}.csv')
    df  = pd.DataFrame(results)
    df.to_csv(csv, index=False)
    print(f"\nResults saved → {csv}")

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(results)

    wandb.log({'results_table': wandb.Table(dataframe=df)})
    wandb.finish()
    return results


if __name__ == '__main__':
    main()
