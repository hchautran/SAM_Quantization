#!/usr/bin/env python3
"""SAM-HQ HQ44K sweep over (algo × ratio × batch_size). Reports throughput,
encoder latency, peak memory, mIoU / Boundary IoU."""

import os
import gc
import sys
import time
import argparse
import datetime
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "sam-hq"))
sys.path.insert(0, os.path.join(_REPO, "PiToMe"))

from segment_anything import SamPredictor, sam_model_registry

from sam_eval_utils import (
    SAM_REGISTRY, VALID_ALGOS, ALGO_REGISTRY, SPARSESAM_ALGOS as _SPARSESAM_ALGOS,
    apply_tome, remove_tome, update_ratio, reset_memory,
    apply_sam, remove_all_sam, update_sam_ratio,
)

from sam_engine import get_default_datasets
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou


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


def process_batch(predictor, images, labels_boxes, labels_ori):
    """One encoder forward + per-image decode. Returns timing/IoU."""
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
    """Bench one (algo, ratio, batch_size). Warmup is critical: the first
    patched config otherwise pays JIT-compile cost on the timed pass.

    num_samples=None or <=0 means "run the full val set".
    """
    reset_memory()

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

    run_all = num_samples is None or num_samples <= 0
    enc_times, enc_per_img, ious, b_ious = [], [], [], []
    total_images = 0
    pbar_total = len(dataloader) if run_all else min(len(dataloader), num_samples // batch_size)
    pbar = tqdm(
        total=pbar_total,
        desc=label or f"bs={batch_size}",
        leave=False,
    )
    t_overall = time.perf_counter()

    for data_val in dataloader:
        if not run_all and total_images >= num_samples:
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


def run_sweep(
    predictor: SamPredictor,
    algos: List[str],
    ratios: List[float],
    batch_sizes: List[int],
    num_samples: int,
    datasets_config: List[Dict],
    dataset_indices: List[int],
    margin: float,
    diagonal_width: int = 1,
    mlp_merge: bool = True,
) -> List[Dict]:

    encoder = predictor.model.image_encoder
    mask_decoder = predictor.model.mask_decoder
    all_results = []

    runs = []
    for algo in algos:
        if algo == "none":
            runs.append(("none", 1.0))
        else:
            for ratio in ratios:
                if ratio < 1.0:
                    runs.append((algo, ratio))

    seen = set(); deduped = []
    for r in runs:
        if r not in seen:
            seen.add(r); deduped.append(r)
    runs = deduped

    print(f"\n{'='*80}")
    print(f"Sweep: {len(runs)} algo/ratio configs × {len(batch_sizes)} batch sizes "
          f"= {len(runs)*len(batch_sizes)} total runs")
    print(f"{'='*80}\n")

    for algo, ratio in runs:
        remove_tome(encoder, mask_decoder=mask_decoder)
        if algo != "none":
            apply_tome(encoder, algo=algo, ratio=ratio, margin=margin,
                       mlp_merge=mlp_merge)

        n_tokens = int(64 * 64 * ratio) if algo != "none" else 64 * 64

        for batch_size in batch_sizes:
            for dataset_idx in dataset_indices:
                ds_name = datasets_config[dataset_idx]["name"]
                label = f"{algo} r={ratio:.2f} bs={batch_size} ds={ds_name}"
                print(f"\n── {label} ──")

                dataloader = build_dataloader(datasets_config, dataset_idx, batch_size)
                result = run_one(predictor, batch_size, dataloader, num_samples, label=label)
                del dataloader

                result.update({
                    'algo':            algo,
                    'ratio':           ratio,
                    'dataset':         ds_name,
                    'n_tokens_kept':   n_tokens,
                    'diagonal_width':  diagonal_width if algo in _SPARSESAM_ALGOS else 1,
                })
                all_results.append(result)

                prefix = f"{ds_name}/{algo}_r{ratio:.2f}_bs{batch_size}"
                wandb.log({
                    f'{prefix}/throughput':           result['throughput_imgs_per_sec'],
                    f'{prefix}/encoder_mean_ms':      result['encoder_batch_mean_ms'],
                    f'{prefix}/encoder_per_img_ms':   result['encoder_per_image_mean_ms'],
                    f'{prefix}/miou':                 result['miou'],
                    f'{prefix}/boundary_iou':         result['boundary_iou'],
                    f'{prefix}/peak_memory_mb':       result.get('peak_memory_allocated_mb', 0),
                })

                print(f"  [{ds_name}] throughput={result['throughput_imgs_per_sec']:.2f} img/s  "
                      f"enc/img={result['encoder_per_image_mean_ms']:.1f}ms  "
                      f"mIoU={result['miou']:.4f}  "
                      f"mem={result.get('peak_memory_allocated_mb', 0):.0f}MB")

    remove_tome(encoder, mask_decoder=mask_decoder)
    return all_results


def _plot_tradeoff_throughput(df, datasets, metrics, output_dir, ts, bs_pick) -> None:
    """y=quality, x=throughput. Baseline as a star marker; each algo a line
    through its ratio points (annotated with the ratio)."""
    import matplotlib.pyplot as plt

    for metric_key, metric_label in metrics:
        n = len(datasets)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)

        for ax, ds in zip(axes.flat, datasets):
            d_ds = df[df['dataset'] == ds]

            for algo in sorted(d_ds[d_ds['algo'] != 'none']['algo'].unique()):
                d = d_ds[d_ds['algo'] == algo].sort_values('ratio')
                ax.plot(d['throughput_imgs_per_sec'], d[metric_key],
                        marker='o', label=algo)
                for _, row in d.iterrows():
                    ax.annotate(f"r={row['ratio']:.2f}",
                                (row['throughput_imgs_per_sec'], row[metric_key]),
                                textcoords='offset points', xytext=(4, 4),
                                fontsize=7, alpha=0.7)

            base = d_ds[d_ds['algo'] == 'none']
            if not base.empty:
                base_x = float(base['throughput_imgs_per_sec'].mean())
                base_y = float(base[metric_key].mean())
                ax.scatter([base_x], [base_y],
                           marker='*', s=200, color='black', zorder=5,
                           label=f'baseline ({base_y:.4f})')

            ax.set_title(ds)
            ax.set_xlabel(f'throughput (img/s) @ bs={bs_pick}')
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for ax in axes.flat[len(datasets):]:
            ax.axis('off')

        fig.suptitle(f'{metric_label} vs throughput tradeoff (bs={bs_pick})')
        fig.tight_layout()
        out = os.path.join(output_dir, f'tome_benchmark_{ts}_{metric_key}_tradeoff.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Plot saved → {out}")


def _plot_quality_vs_ratio(df, datasets, metrics, output_dir, ts) -> None:
    """y=quality, x=keep ratio. Baseline shown as a horizontal dashed line
    (ratio=1.0 by definition) so each algo's degradation curve is read
    directly against it."""
    import matplotlib.pyplot as plt

    for metric_key, metric_label in metrics:
        n = len(datasets)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)

        for ax, ds in zip(axes.flat, datasets):
            d_ds = df[df['dataset'] == ds]

            for algo in sorted(d_ds[d_ds['algo'] != 'none']['algo'].unique()):
                d = (d_ds[d_ds['algo'] == algo]
                     .groupby('ratio', as_index=False)[metric_key].mean()
                     .sort_values('ratio'))
                ax.plot(d['ratio'], d[metric_key], marker='o', label=algo)

            base = d_ds[d_ds['algo'] == 'none']
            if not base.empty:
                base_y = float(base[metric_key].mean())
                ax.axhline(base_y, ls='--', color='gray',
                           label=f'baseline ({base_y:.4f})')

            ax.set_title(ds)
            ax.set_xlabel('keep ratio')
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for ax in axes.flat[len(datasets):]:
            ax.axis('off')

        fig.suptitle(f'{metric_label} vs keep ratio')
        fig.tight_layout()
        out = os.path.join(output_dir, f'tome_benchmark_{ts}_{metric_key}_vs_ratio.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Plot saved → {out}")


def plot_results(results: List[Dict], output_dir: str, ts: str) -> None:
    """Two figure families per metric (mIoU, B-IoU):
      1. quality vs throughput tradeoff (filtered to largest batch size)
      2. quality vs keep ratio (averaged across batch sizes)
    Both put baseline + algos on the same axes."""
    df = pd.DataFrame(results)
    if df.empty:
        return

    datasets = sorted(df['dataset'].unique())
    metrics  = [('miou', 'mIoU'), ('boundary_iou', 'Boundary IoU')]

    bs_pick = int(df['batch_size'].max())
    df_bs = df[df['batch_size'] == bs_pick]
    _plot_tradeoff_throughput(df_bs, datasets, metrics, output_dir, ts, bs_pick)
    _plot_quality_vs_ratio(df, datasets, metrics, output_dir, ts)


def print_summary(results: List[Dict]):
    baselines = {(r.get('dataset', ''), r['batch_size']): r
                 for r in results if r['algo'] == 'none'}

    print(f"\n{'='*120}")
    print("SUMMARY — token merging vs baseline")
    print(f"{'='*120}")
    print(
        f"{'Dataset':<18} {'Algo':<10} {'Ratio':>6} {'BS':>4} | "
        f"{'Throughput':>12} {'Δspeedup':>10} | "
        f"{'Enc/img(ms)':>12} | "
        f"{'Mem(MB)':>9} | "
        f"{'mIoU':>8} {'ΔmIoU':>8} | "
        f"{'B-IoU':>8}"
    )
    print("-" * 120)

    for r in results:
        bs  = r['batch_size']
        b   = baselines.get((r.get('dataset', ''), bs))
        spd = r['throughput_imgs_per_sec']
        d_spd = (spd / b['throughput_imgs_per_sec'] - 1) * 100 if b else 0
        d_iou = r['miou'] - b['miou'] if b else 0

        sign_spd = "+" if d_spd >= 0 else ""
        sign_iou = "+" if d_iou >= 0 else ""

        print(
            f"{r.get('dataset', ''):<18} {r['algo']:<10} {r['ratio']:>6.2f} {bs:>4} | "
            f"{spd:>12.2f} {sign_spd}{d_spd:>9.1f}% | "
            f"{r['encoder_per_image_mean_ms']:>12.2f} | "
            f"{r.get('peak_memory_allocated_mb', 0):>9.0f} | "
            f"{r['miou']:>8.4f} {sign_iou}{d_iou:>7.4f} | "
            f"{r['boundary_iou']:>8.4f}"
        )

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SAM-1 with ToMe / PiToMe token merging',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--model-ckt',  type=str, default='./ckts/sam_hq_vit_l.pth')
    parser.add_argument('--model-type', type=str, default='vit_l',
                        choices=['vit_h', 'vit_l', 'vit_b'])

    parser.add_argument('--algos', type=str, nargs='+',
                        default=['none', 'tome', 'pitome'],
                        choices=VALID_ALGOS)
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=[0.9, 0.8, 0.7])
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[1, 2, 4, 8])
    parser.add_argument('--num-samples', type=int, default=0,
                        help='Per-dataset image cap. 0 or negative = run the full val set.')

    parser.add_argument('--margin', type=float, default=0.5,
                        help='PiToMe energy margin.')
    parser.add_argument('--diagonal-width', type=int, default=1,
                        help='Diagonal band width for sparsesam block-sparse attn.')
    parser.add_argument('--mlp-merge', action=argparse.BooleanOptionalAction, default=True,
                        help='(sparsesam) MLP on top-K keep tokens (default) vs full N.')

    parser.add_argument('--dataset-idx', type=int, nargs='+', default=None)
    parser.add_argument('--no-plot', action='store_true')

    parser.add_argument('--output-dir', type=str, default='./benchmark_results')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='sam-tome-benchmark')
    parser.add_argument('--wandb-run-name', type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    else:
        wandb.init(mode='disabled')

    print(f"Loading SAM {args.model_type} from {args.model_ckt} ...")
    sam       = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to('cuda').half()
    predictor = SamPredictor(sam)

    datasets = get_default_datasets()
    dataset_indices = args.dataset_idx if args.dataset_idx is not None else list(range(len(datasets)))

    results = run_sweep(
        predictor       = predictor,
        algos           = args.algos,
        ratios          = args.ratios,
        batch_sizes     = args.batch_sizes,
        num_samples     = args.num_samples,
        datasets_config = datasets,
        dataset_indices = dataset_indices,
        margin          = args.margin,
        mlp_merge       = bool(args.mlp_merge),
    )

    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv = os.path.join(args.output_dir, f'tome_benchmark_{ts}.csv')
    df  = pd.DataFrame(results)
    df.to_csv(csv, index=False)
    print(f"\nResults saved → {csv}")

    print_summary(results)

    if not args.no_plot:
        plot_results(results, args.output_dir, ts)

    wandb.log({'results_table': wandb.Table(dataframe=df)})
    wandb.finish()
    return results


if __name__ == '__main__':
    main()
