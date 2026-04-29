#!/usr/bin/env python3
"""
Evaluate SAM2 on HQ44k dataset.

Usage:
    python eval_sam2_hq44k.py \
        --checkpoint ./checkpoints/sam2_hiera_large.pt \
        --num-samples 100
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# Make `train.*` (under sam-hq/) and SAM2 (under sam-hq/sam-hq2/) importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "sam-hq"),
           os.path.join(_HERE, "sam-hq", "sam-hq2")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Local imports
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou
from sam_engine import get_default_datasets


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized ori_im fields."""
    ori_ims = [item['ori_im'] for item in batch]

    collated = {}
    for key in batch[0].keys():
        if key == 'ori_im':
            collated[key] = ori_ims
        elif key == 'ori_im_path' or key == 'ori_gt_path':
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except:
                collated[key] = [item[key] for item in batch]

    return collated


class SAM2Evaluator:
    """Evaluate SAM2 on HQ44k dataset"""

    def __init__(self):
        self.results = {}

    def eval_hq44k(
        self,
        predictor: SAM2ImagePredictor,
        dataloader: DataLoader,
        num_samples: int = None,
        use_batch: bool = False,
        amp_dtype: torch.dtype = None,
    ):
        """
        Evaluate SAM2 on HQ44k dataset.

        Args:
            predictor: SAM2ImagePredictor instance
            dataloader: DataLoader for HQ44k dataset
            num_samples: Number of samples to evaluate (None = all)
            use_batch: If True, use SAM2's native batch processing

        Returns:
            Dict with mIoU and boundary IoU metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating SAM2 on HQ44k {'(Batch Mode)' if use_batch else '(Single Image Mode)'}")
        print(f"{'='*80}\n")

        ious = []
        boundary_ious = []
        total_images = 0

        progress_bar = tqdm(total=len(dataloader), desc="Evaluating")

        # Mixed-precision context: when amp_dtype is set, run encoder + heads
        # under autocast so QKV/MLP run in fp16/bf16 (also satisfies the cute
        # block-sparse kernel's dtype requirement).
        from contextlib import nullcontext
        amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
                   if amp_dtype is not None else nullcontext())

        for idx, data_val in enumerate(dataloader):
            if num_samples and total_images >= num_samples:
                break

            images = data_val['image']
            labels_val = data_val['label']
            labels_ori = data_val['ori_label']

            batch_size = images.shape[0]

            if use_batch and batch_size > 1:
                try:
                    image_list = []
                    for i in range(batch_size):
                        img = images[i].permute(1, 2, 0).cpu().numpy()
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                        image_list.append(img)

                    with amp_ctx:
                        predictor.set_image_batch(image_list)

                    box_batch = []
                    for i in range(batch_size):
                        label = labels_val[i, 0, :, :]
                        bbox = misc.masks_to_boxes(label.unsqueeze(0))
                        box_batch.append(bbox.cpu().numpy()[0])

                    with amp_ctx:
                        masks_batch, _, _ = predictor.predict_batch(
                            point_coords_batch=None,
                            point_labels_batch=None,
                            box_batch=box_batch,
                            multimask_output=False,
                        )

                    for i in range(batch_size):
                        mask_np = masks_batch[i]
                        mask_tensor = torch.from_numpy(mask_np).to(labels_ori.device)
                        mask_tensor = mask_tensor.unsqueeze(0).float()

                        if mask_tensor.shape[1] > 1:
                            mask_tensor = mask_tensor[:, 0:1, :, :]

                        iou = compute_iou(mask_tensor, labels_ori[i:i+1])
                        boundary_iou = compute_boundary_iou(mask_tensor, labels_ori[i:i+1])

                        ious.append(iou.item() if torch.is_tensor(iou) else iou)
                        boundary_ious.append(boundary_iou.item() if torch.is_tensor(boundary_iou) else boundary_iou)

                except Exception as e:
                    print(f"\nError processing batch {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    for _ in range(batch_size):
                        ious.append(0.0)
                        boundary_ious.append(0.0)

            else:
                for i in range(batch_size):
                    img = images[i].permute(1, 2, 0).cpu().numpy()

                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    with amp_ctx:
                        predictor.set_image(img)

                    label = labels_val[i, 0, :, :]
                    bbox = misc.masks_to_boxes(label.unsqueeze(0))
                    bbox = bbox.cpu().numpy()

                    try:
                        with amp_ctx:
                            masks, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=bbox[0],
                                multimask_output=False,
                            )

                        mask_tensor = torch.from_numpy(masks).to(labels_ori.device)
                        mask_tensor = mask_tensor.unsqueeze(0).float()

                        if mask_tensor.shape[1] > 1:
                            mask_tensor = mask_tensor[:, 0:1, :, :]

                        iou = compute_iou(mask_tensor, labels_ori[i:i+1])
                        boundary_iou = compute_boundary_iou(mask_tensor, labels_ori[i:i+1])

                        ious.append(iou.item() if torch.is_tensor(iou) else iou)
                        boundary_ious.append(boundary_iou.item() if torch.is_tensor(boundary_iou) else boundary_iou)

                    except Exception as e:
                        print(f"\nError processing sample {idx}, image {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        ious.append(0.0)
                        boundary_ious.append(0.0)

            total_images += batch_size
            progress_bar.update(1)
            if ious:
                progress_bar.set_postfix(
                    miou=f"{float(np.mean(ious)):.4f}",
                    biou=f"{float(np.mean(boundary_ious)):.4f}",
                    n=len(ious),
                )

        progress_bar.close()

        results = {
            'miou': np.mean(ious),
            'miou_std': np.std(ious),
            'boundary_iou': np.mean(boundary_ious),
            'boundary_iou_std': np.std(boundary_ious),
            'num_samples': len(ious),
        }

        print(f"\n{'='*80}")
        print("Results:")
        print(f"{'='*80}")
        print(f"  mIoU: {results['miou']:.4f} +/- {results['miou_std']:.4f}")
        print(f"  Boundary IoU: {results['boundary_iou']:.4f} +/- {results['boundary_iou_std']:.4f}")
        print(f"  Samples evaluated: {results['num_samples']}")
        print(f"{'='*80}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 on HQ44k dataset')

    parser.add_argument('--model-cfg', type=str, default='sam2_hiera_l.yaml',
                       help='SAM2 model config (e.g., sam2_hiera_l.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to SAM2 checkpoint')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for dataloader')
    parser.add_argument('--use-batch', action='store_true',
                       help='Use SAM2 native batch processing')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of dataloader workers')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--dtype', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='Inference precision (autocast). fp16 is required '
                            'by the cute block-sparse kernel; fp32 disables autocast.')
    parser.add_argument('--algorithms', type=str, nargs='+', default=['none'],
                       choices=['none', 'sparsesam'],
                       help='One or more token-compression algorithms to sweep. '
                            "'none' = uncompressed baseline.")
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.5],
                       help='Keep-bar fractions in (0, 1] to sweep '
                            '(ignored for algorithm=none; lower = more compression).')
    parser.add_argument('--group-size', type=int, default=4,
                       help='(sparsesam) Z-group size for tile-stride permute')
    parser.add_argument('--no-mlp-prune', action='store_true',
                       help='(sparsesam) disable MLP token pruning on local blocks')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"\n{'='*80}")
    print("SAM2 HQ44k Evaluation")
    print(f"{'='*80}")
    print(f"Model config: {args.model_cfg}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    print("Loading SAM2 model...")
    sam2_model = build_sam2(
        config_file=args.model_cfg,
        ckpt_path=args.checkpoint,
        device=args.device
    )
    predictor = SAM2ImagePredictor(sam2_model)
    print("Model loaded\n")

    print("Loading dataset...")
    datasets = get_default_datasets()
    valid_im_gt_list = get_im_gt_name_dict([datasets[0]], flag="valid")

    gos_dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True
    )

    dataloader = DataLoader(
        gos_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=custom_collate_fn if args.batch_size > 1 else None,
    )

    print(f"Dataset loaded: {len(gos_dataset)} samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Use batch mode: {args.use_batch}")
    print(f"  Samples to evaluate: {args.num_samples if args.num_samples else 'all'}\n")

    # Build (algo, ratio) configs to sweep. 'none' ignores ratio (run once);
    # the compression algos run once per ratio value.
    RATIO_ALGOS = {'sparsesam'}
    configs = []
    for algo in args.algorithms:
        if algo in RATIO_ALGOS:
            for r in args.ratios:
                configs.append((algo, float(r)))
        else:
            configs.append((algo, None))

    evaluator = SAM2Evaluator()
    all_results = []
    for algo, ratio in configs:
        tag = algo + (f"@r={ratio}" if ratio is not None else "")
        print(f"\n{'#'*80}\n# config: {tag}\n{'#'*80}")

        # Reset to baseline forwards before applying the next patch.
        try:
            from PiToMe.algo.sparsesam.patch.sam2_hiera import remove_patch as _remove_sparsesam
            _remove_sparsesam(sam2_model.image_encoder.trunk)
        except Exception:
            pass

        if algo == 'sparsesam':
            from PiToMe.algo.sparsesam.patch.sam2_hiera import apply_patch as _apply_sparsesam
            _apply_sparsesam(
                sam2_model.image_encoder.trunk,
                ratio=ratio,
                group_size=args.group_size,
                prune_mlp=not args.no_mlp_prune,
            )

        amp_dtype = {
            'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16,
        }[args.dtype]

        results = evaluator.eval_hq44k(
            predictor=predictor,
            dataloader=dataloader,
            num_samples=args.num_samples,
            use_batch=args.use_batch,
            amp_dtype=amp_dtype,
        )
        all_results.append({'algorithm': algo, 'ratio': ratio, 'results': results})

    return all_results


if __name__ == '__main__':
    main()
