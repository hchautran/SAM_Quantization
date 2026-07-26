#!/usr/bin/env python3
"""SparseSAM token-ordering ablation on HQ44K (default sparsity 0.5).

Sweeps the two design choices a reviewer flagged as un-ablated:

    grid size   — tokens per space-filling-curve group (2, 4, 8, 16; paper: 4)
    saliency    — Sobel 3/5/7, Scharr, Laplacian, feature-norm, attention-derived,
                  the shipped variance+dissimilarity scorer, and a random
                  ranking baseline

Only the permutation is swapped (see `sparsesam_saliency.py`); the SparseSAM
patch, the block-sparse cute kernel and the SAM-HQ decoder are untouched.

Examples
--------
    # grid-size ablation on the Sobel ordering, sparsity 0.5
    python eval_hq44k_saliency_ablation.py --group-sizes 2 4 8 16

    # saliency-estimator ablation at the paper's grid size
    python eval_hq44k_saliency_ablation.py --group-sizes 4 \
        --saliency sobel3 sobel5 sobel7 scharr3 laplacian3 feature_norm \
                   attention variance_dissim random

    # full grid + dense baseline
    python eval_hq44k_saliency_ablation.py --group-sizes 2 4 8 16 \
        --saliency sobel3 sobel5 sobel7 scharr3 laplacian3 feature_norm random \
        --include-baseline
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "sam-hq", ROOT / "PiToMe"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sparsesam_saliency as sal  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402
from segment_anything.modeling.image_encoder import Attention, Block  # noqa: E402
from data_utils import OnlineDataset  # noqa: E402
from train.train import compute_boundary_iou, compute_iou  # noqa: E402
from train.utils.dataloader import Resize, get_im_gt_name_dict  # noqa: E402
import train.utils.misc as misc  # noqa: E402


# The four HQSeg-44K validation splits (same set as eval_hq44k_efficient_sams.py).
DATASETS = [
    {"name": "DIS5K-VD", "im_dir": "./data/DIS5K/DIS-VD/im", "gt_dir": "./data/DIS5K/DIS-VD/gt", "im_ext": ".jpg", "gt_ext": ".png"},
    {"name": "COIFT", "im_dir": "./data/thin_object_detection/COIFT/images", "gt_dir": "./data/thin_object_detection/COIFT/masks", "im_ext": ".jpg", "gt_ext": ".png"},
    {"name": "HRSOD", "im_dir": "./data/thin_object_detection/HRSOD/images", "gt_dir": "./data/thin_object_detection/HRSOD/masks", "im_ext": ".jpg", "gt_ext": ".png"},
    {"name": "ThinObject5k-TE", "im_dir": "./data/thin_object_detection/ThinObject5K/images_test", "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test", "im_ext": ".jpg", "gt_ext": ".png"},
]

CHECKPOINTS = {
    "vit_b": ROOT / "ckts/sam_hq_vit_b.pth",
    "vit_l": ROOT / "ckts/sam_hq_vit_l.pth",
    "vit_h": ROOT / "ckts/sam_hq_vit_h.pth",
}


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if key in ("ori_im", "ori_im_path", "ori_gt_path"):
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def build_dataloader(dataset_config: Dict, batch_size: int, num_workers: int):
    valid_im_gt_list = get_im_gt_name_dict([dataset_config], flag="valid")
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
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )


def encoder_dtype_of(predictor: SamPredictor) -> torch.dtype:
    return next(predictor.model.image_encoder.parameters()).dtype


def decoder_dtype_of(predictor: SamPredictor) -> torch.dtype:
    return next(predictor.model.mask_decoder.parameters()).dtype


@torch.no_grad()
def process_batch(predictor: SamPredictor, images, boxes, labels_ori):
    """One encoder forward + per-image HQ decode. Returns timing and IoUs.

    The image encoder runs in fp16 (the block-sparse cute kernel is fp16-only);
    the prompt encoder / mask decoder stay in their own dtype — sam-hq's
    `PositionEmbeddingRandom` hard-casts coords to fp32, so halving the whole
    model breaks box prompts.
    """
    device = predictor.device
    batch_size = images.shape[0]
    enc_dtype = encoder_dtype_of(predictor)
    dec_dtype = decoder_dtype_of(predictor)

    images = images.to(device)
    boxes = boxes.to(device, dtype=dec_dtype)
    labels_ori = labels_ori.to(device)

    transformed = predictor.model.preprocess(images).to(dtype=enc_dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    features, interm_features = predictor.model.image_encoder(transformed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encoder_ms = (time.perf_counter() - t0) * 1000

    features = features.to(dtype=dec_dtype)
    interm_features = [
        f.to(dtype=dec_dtype) if torch.is_floating_point(f) else f
        for f in interm_features
    ]

    ious, b_ious = [], []
    for i in range(batch_size):
        predictor.features = features[i:i + 1]
        predictor.interm_features = [f[i:i + 1] for f in interm_features]
        predictor.original_size = (images.shape[2], images.shape[3])
        predictor.input_size = tuple(transformed.shape[-2:])
        predictor.is_image_set = True
        masks, _, _ = predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=boxes[i:i + 1], hq_token_only=True,
        )
        ious.append(compute_iou(masks, labels_ori[i:i + 1]))
        b_ious.append(compute_boundary_iou(masks, labels_ori[i:i + 1]))
    predictor.reset_image()

    return {
        "encoder_ms": encoder_ms,
        "encoder_per_image_ms": encoder_ms / batch_size,
        "iou": torch.mean(torch.stack(ious)).item(),
        "boundary_iou": torch.mean(torch.stack(b_ious)).item(),
    }


def evaluate_dataset(predictor, dataloader, num_samples: int, label: str,
                     warmup_batches: int = 2) -> Dict:
    """Warmup matters: the first patched config otherwise pays kernel-compile
    cost inside the timed loop."""
    reset_memory()
    device = predictor.device

    enc_dtype = encoder_dtype_of(predictor)
    warmup_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            data = next(warmup_iter)
        except StopIteration:
            break
        with torch.no_grad():
            transformed = predictor.model.preprocess(
                data["image"].to(device)
            ).to(dtype=enc_dtype)
            predictor.model.image_encoder(transformed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del warmup_iter

    enc_ms, enc_img_ms, ious, b_ious = [], [], [], []
    total_images = 0
    t_overall = time.perf_counter()
    pbar = tqdm(total=min(len(dataloader), max(1, num_samples // dataloader.batch_size)),
                desc=label, leave=False)

    for data in dataloader:
        if total_images >= num_samples:
            break
        images = data["image"]
        labels_val = data["label"]
        labels_ori = data["ori_label"]
        if labels_val.dim() == 4:
            boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])
        else:
            boxes = misc.masks_to_boxes(labels_val[0:1, :, :])

        try:
            m = process_batch(predictor, images, boxes, labels_ori)
        except Exception as exc:  # keep the sweep alive on a bad sample
            print(f"  batch error ({label}): {exc}")
            continue

        enc_ms.append(m["encoder_ms"])
        enc_img_ms.append(m["encoder_per_image_ms"])
        ious.append(m["iou"])
        b_ious.append(m["boundary_iou"])
        total_images += images.shape[0]
        pbar.update(1)

    pbar.close()
    overall_sec = time.perf_counter() - t_overall

    mem = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem = {
            "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2,
        }

    return {
        "num_images": total_images,
        "throughput_imgs_per_sec": total_images / overall_sec if overall_sec else 0.0,
        "encoder_batch_mean_ms": float(np.mean(enc_ms)) if enc_ms else 0.0,
        "encoder_per_image_mean_ms": float(np.mean(enc_img_ms)) if enc_img_ms else 0.0,
        "encoder_per_image_std_ms": float(np.std(enc_img_ms)) if enc_img_ms else 0.0,
        "miou": float(np.mean(ious)) if ious else 0.0,
        "miou_std": float(np.std(ious)) if ious else 0.0,
        "boundary_iou": float(np.mean(b_ious)) if b_ious else 0.0,
        "boundary_iou_std": float(np.std(b_ious)) if b_ious else 0.0,
        **mem,
        "elapsed_sec": overall_sec,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def save_rows(rows: List[Dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        fieldnames = list(dict.fromkeys(fieldnames + list(row.keys())))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_sparsesam_patch(encoder):
    """Revert the class swaps done by `sparsesam.sam.apply_patch`."""
    from PiToMe.algo.sparsesam.sam import ToMeSAMAttention, ToMeSAMBlock
    for module in encoder.modules():
        if isinstance(module, ToMeSAMBlock):
            module.__class__ = Block
        elif isinstance(module, ToMeSAMAttention):
            module.__class__ = Attention
        module.__dict__.pop("_tome_info", None)
    encoder.__dict__.pop("forward", None)
    encoder.__dict__.pop("tome_info", None)


def print_summary(rows: List[Dict]):
    if not rows:
        return
    baselines = {r["dataset"]: r for r in rows if r["config"] == "dense_baseline"}
    print("\n" + "=" * 118)
    print("SparseSAM token-ordering ablation — HQ44K")
    print("=" * 118)
    print(f"{'Dataset':<17}{'Saliency':<17}{'Grid':>5}{'Layout':>13}"
          f"{'mIoU':>9}{'ΔmIoU':>9}{'B-IoU':>9}{'enc/img ms':>12}{'img/s':>9}")
    print("-" * 118)
    for r in rows:
        base = baselines.get(r["dataset"])
        d = f"{r['miou'] - base['miou']:+.4f}" if base else "     —"
        print(f"{r['dataset']:<17}{r['saliency']:<17}{str(r['group_size']):>5}"
              f"{r['layout']:>13}{r['miou']:>9.4f}{d:>9}{r['boundary_iou']:>9.4f}"
              f"{r['encoder_per_image_mean_ms']:>12.2f}"
              f"{r['throughput_imgs_per_sec']:>9.2f}")
    print("=" * 118)


def parse_args():
    p = argparse.ArgumentParser(
        description="SparseSAM saliency / grid-size ablation on HQ44K.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model-type", default="vit_l", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--model-ckt", default=None, help="Defaults to ./ckts/sam_hq_<type>.pth")
    p.add_argument("--ratio", type=float, default=0.5,
                   help="Keep ratio. 0.5 = 50%% sparsity (default).")
    p.add_argument("--group-sizes", type=int, nargs="+", default=[2, 4, 8, 16],
                   help="Grid sizes (tokens per curve group) to ablate.")
    p.add_argument("--saliency", nargs="+", default=["sobel3"],
                   choices=sal.SALIENCY_CHOICES,
                   help="Saliency estimators to ablate.")
    p.add_argument("--curve", nargs="+", default=["z"], choices=["z", "hilbert", "raster"],
                   help="Space-filling curve used to form groups.")
    p.add_argument("--layout", default="grouped", choices=["grouped", "interleaved"],
                   help="'grouped': keep-prefix = top-ranked groups (ranking drives\n"
                        "the ordering). 'interleaved': shipped stride layout, in which\n"
                        "the keep-prefix is identical for every saliency estimator.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dataset-idx", type=int, nargs="+", default=None,
                   help=f"Subset of {[d['name'] for d in DATASETS]} by index.")
    p.add_argument("--mlp-merge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-baseline", action="store_true",
                   help="Also evaluate the unpatched dense encoder.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=str(ROOT / "benchmark_results/saliency_ablation_hq44k"))
    p.add_argument("--output-csv", default=None)
    p.add_argument("--self-test-only", action="store_true",
                   help="Validate the permutations and exit (no model, no data).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.self_test_only:
        n = sal.self_test(device=args.device,
                          group_sizes=tuple(args.group_sizes),
                          saliencies=tuple(args.saliency),
                          layouts=(args.layout,),
                          curves=tuple(args.curve))
        print(f"OK — {n} configs produce valid permutations")
        return []

    ckpt = Path(args.model_ckt) if args.model_ckt else CHECKPOINTS[args.model_type]
    dataset_indices = args.dataset_idx if args.dataset_idx is not None else range(len(DATASETS))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.output_csv) if args.output_csv else (
        Path(args.output_dir) / f"sparsesam_saliency_ablation_hq44k_{args.model_type}_{ts}.csv"
    )

    # Fail fast on a bad sweep spec before loading the model / data.
    n_checks = sal.self_test(
        device=args.device, group_sizes=tuple(args.group_sizes),
        saliencies=tuple(args.saliency), layouts=(args.layout,),
        curves=tuple(args.curve),
    )
    print(f"Permutation self-test passed ({n_checks} configs).")

    print(f"Loading SAM-HQ {args.model_type} from {ckpt} ...")
    sam = sam_model_registry[args.model_type](checkpoint=str(ckpt)).to(args.device)
    sam.eval()
    predictor = SamPredictor(sam)
    encoder = predictor.model.image_encoder
    encoder.half()  # cute block-sparse attention is fp16-only

    configs = [
        sal.SaliencyConfig(saliency=s, group_size=g, curve=c, layout=args.layout)
        for g in args.group_sizes
        for c in args.curve
        for s in args.saliency
    ]
    print(f"Sweep: {len(configs)} ordering configs × {len(list(dataset_indices))} datasets "
          f"at keep-ratio {args.ratio} (sparsity {1 - args.ratio:.2f})")

    rows: List[Dict] = []

    if args.include_baseline:
        for idx in dataset_indices:
            ds = DATASETS[idx]
            loader = build_dataloader(ds, args.batch_size, args.num_workers)
            label = f"baseline ds={ds['name']}"
            print(f"\n── {label} ──")
            res = evaluate_dataset(predictor, loader, args.num_samples, label)
            res.update({
                "config": "dense_baseline", "saliency": "none", "group_size": 0,
                "curve": "none", "layout": "none", "ratio_keep": 1.0, "sparsity": 0.0,
                "dataset": ds["name"], "model_type": args.model_type,
                "checkpoint": str(ckpt), "batch_size": args.batch_size,
                "mlp_merge": args.mlp_merge,
            })
            rows.append(res)
            save_rows(rows, out_csv)
            print(f"  mIoU={res['miou']:.4f}  B-IoU={res['boundary_iou']:.4f}  "
                  f"enc/img={res['encoder_per_image_mean_ms']:.2f} ms")
            del loader
            reset_memory()

    # Patch once (FA2 kernels warm up here); the ordering config is then swapped
    # per run via `sal.set_config` + cache clear.
    from PiToMe.algo.sparsesam.sam import apply_patch as apply_sparsesam_patch
    sal.install(configs[0])
    apply_sparsesam_patch(encoder, algo="tome", ratio=args.ratio,
                          mlp_merge=bool(args.mlp_merge))

    try:
        for cfg in configs:
            sal.set_config(cfg)
            sal.clear_perm_caches(encoder)
            for idx in dataset_indices:
                ds = DATASETS[idx]
                loader = build_dataloader(ds, args.batch_size, args.num_workers)
                label = f"{cfg.tag()} ds={ds['name']}"
                print(f"\n── {label} ──")
                res = evaluate_dataset(predictor, loader, args.num_samples, label)
                res.update({
                    "config": cfg.tag(), "saliency": cfg.saliency,
                    "group_size": cfg.group_size, "curve": cfg.curve,
                    "layout": cfg.layout, "ratio_keep": args.ratio,
                    "sparsity": 1.0 - args.ratio, "dataset": ds["name"],
                    "model_type": args.model_type, "checkpoint": str(ckpt),
                    "batch_size": args.batch_size, "mlp_merge": args.mlp_merge,
                })
                rows.append(res)
                save_rows(rows, out_csv)
                print(f"  mIoU={res['miou']:.4f}  B-IoU={res['boundary_iou']:.4f}  "
                      f"enc/img={res['encoder_per_image_mean_ms']:.2f} ms  "
                      f"mem={res.get('peak_memory_allocated_mb', 0):.0f} MB")
                del loader
                reset_memory()
    finally:
        remove_sparsesam_patch(encoder)
        sal.restore()

    save_rows(rows, out_csv)
    print_summary(rows)
    print(f"\nResults saved → {out_csv}")
    return rows


if __name__ == "__main__":
    main()
