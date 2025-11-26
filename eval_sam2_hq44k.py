#!/usr/bin/env python3
"""
Evaluate SAM2 on HQ44k dataset - adapted from benchmark_batch_inference.py

This script evaluates SAM2 models on the HQ44k dataset and reports mIoU and boundary IoU metrics.
Supports entropy-based attention head pruning/quantization for SAM2.

Usage:
    # Basic evaluation
    python eval_sam2_hq44k.py \
        --checkpoint ./checkpoints/sam2_hiera_large.pt \
        --num-samples 100

    # With entropy processor
    python eval_sam2_hq44k.py \
        --checkpoint ./checkpoints/sam2_hiera_large.pt \
        --processor POSITIONAL_PRUNE_SAM2 \
        --num-calib-samples 32 \
        --num-samples 100
"""

import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.backbones.hieradet import MultiScaleAttention

# Local imports
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou
from sam_engine import get_default_datasets

# SAM2 entropy processors
from processors.encoder.entropy_sam2 import (
    PositionalPruneSAM2Processor,
    HeadPruneSAM2Processor,
    PositionalQuantSAM2Processor,
)
from processors.sam2_observer import sam2_image_encoder_monkey_patch


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized ori_im fields.
    Stack tensors that can be stacked, keep lists for variable-sized items.
    """
    # Separate ori_im which has variable sizes
    ori_ims = [item['ori_im'] for item in batch]

    # Create a new batch dict without ori_im
    collated = {}
    for key in batch[0].keys():
        if key == 'ori_im':
            collated[key] = ori_ims  # Keep as list
        elif key == 'ori_im_path' or key == 'ori_gt_path':
            # These are strings, keep as list
            collated[key] = [item[key] for item in batch]
        else:
            # Stack tensors
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except:
                # If stacking fails, keep as list
                collated[key] = [item[key] for item in batch]

    return collated


# Registry for SAM2 processors
SAM2_PROCESSOR_REGISTRY = {
    "POSITIONAL_PRUNE_SAM2": PositionalPruneSAM2Processor,
    "HEAD_PRUNE_SAM2": HeadPruneSAM2Processor,
    "POSITIONAL_QUANT_SAM2": PositionalQuantSAM2Processor,
}


def get_sam2_processor(name: str, **kwargs):
    """Get SAM2 processor by name."""
    if name not in SAM2_PROCESSOR_REGISTRY:
        available = list(SAM2_PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown SAM2 processor '{name}'. Available: {available}")
    return SAM2_PROCESSOR_REGISTRY[name](**kwargs)


class SAM2Evaluator:
    """Evaluate SAM2 on HQ44k dataset"""

    def __init__(self):
        self.results = {}

    def eval_hq44k(
        self,
        predictor: SAM2ImagePredictor,
        dataloader: DataLoader,
        num_samples: int = None,
        use_batch: bool = False
    ):
        """
        Evaluate SAM2 on HQ44k dataset.

        Args:
            predictor: SAM2ImagePredictor instance
            dataloader: DataLoader for HQ44k dataset
            num_samples: Number of samples to evaluate (None = all)
            use_batch: If True, use SAM2's native batch processing (faster for batch > 1)

        Returns:
            Dict with mIoU and boundary IoU metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating SAM2 on HQ44k {'(Batch Mode)' if use_batch else '(Single Image Mode)'}")
        print(f"{'='*80}\n")

        ious = []
        boundary_ious = []
        total_images = 0

        progress_bar = tqdm(
            total=len(dataloader),
            desc="Evaluating"
        )

        for idx, data_val in enumerate(dataloader):
            if num_samples and total_images >= num_samples:
                break

            # Get data
            images = data_val['image']  # [B, C, H, W]
            labels_val = data_val['label']  # [B, 1, H, W]
            labels_ori = data_val['ori_label']  # [B, 1, H, W]

            batch_size = images.shape[0]

            if use_batch and batch_size > 1:
                # Use SAM2's native batch processing
                try:
                    # Convert batch of images to list of numpy arrays (H, W, C)
                    image_list = []
                    for i in range(batch_size):
                        img = images[i].permute(1, 2, 0).cpu().numpy()
                        # Ensure correct format (0-255 range)
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                        image_list.append(img)

                    # Set batch of images
                    predictor.set_image_batch(image_list)

                    # Get bounding boxes for all images
                    box_batch = []
                    for i in range(batch_size):
                        label = labels_val[i, 0, :, :]
                        bbox = misc.masks_to_boxes(label.unsqueeze(0))  # [1, 4]
                        box_batch.append(bbox.cpu().numpy()[0])  # [4]

                    # Predict batch
                    masks_batch, _, _ = predictor.predict_batch(
                        point_coords_batch=None,
                        point_labels_batch=None,
                        box_batch=box_batch,
                        multimask_output=False,
                    )

                    # Process results for each image in batch
                    for i in range(batch_size):
                        # masks_batch[i] shape: [num_masks, H, W]
                        mask_np = masks_batch[i]
                        mask_tensor = torch.from_numpy(mask_np).to(labels_ori.device)
                        mask_tensor = mask_tensor.unsqueeze(0).float()  # [1, num_masks, H, W]

                        # Take first mask if multiple
                        if mask_tensor.shape[1] > 1:
                            mask_tensor = mask_tensor[:, 0:1, :, :]

                        # Compute metrics
                        iou = compute_iou(mask_tensor, labels_ori[i:i+1])
                        boundary_iou = compute_boundary_iou(mask_tensor, labels_ori[i:i+1])

                        ious.append(iou.item() if torch.is_tensor(iou) else iou)
                        boundary_ious.append(boundary_iou.item() if torch.is_tensor(boundary_iou) else boundary_iou)

                except Exception as e:
                    print(f"\nError processing batch {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add zero scores for failed samples
                    for _ in range(batch_size):
                        ious.append(0.0)
                        boundary_ious.append(0.0)

            else:
                # Process images one by one (original behavior)
                for i in range(batch_size):
                    # Get single image (H, W, C) format for SAM2
                    # Convert from [C, H, W] to [H, W, C]
                    img = images[i].permute(1, 2, 0).cpu().numpy()

                    # Ensure correct format (0-255 range)
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    # Set image for SAM2
                    predictor.set_image(img)

                    # Get bounding box
                    label = labels_val[i, 0, :, :]
                    bbox = misc.masks_to_boxes(label.unsqueeze(0))  # [1, 4]
                    bbox = bbox.cpu().numpy()

                    # Predict with SAM2
                    # SAM2 predict() expects boxes in xyxy format [x1, y1, x2, y2]
                    try:
                        masks, _, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=bbox[0],  # Single box [4]
                            multimask_output=False,
                        )

                        # Convert mask to tensor for IoU computation
                        # masks shape: [num_masks, H, W]
                        mask_tensor = torch.from_numpy(masks).to(labels_ori.device)
                        mask_tensor = mask_tensor.unsqueeze(0).float()  # [1, num_masks, H, W]

                        # Take first mask if multiple
                        if mask_tensor.shape[1] > 1:
                            mask_tensor = mask_tensor[:, 0:1, :, :]

                        # Compute metrics
                        iou = compute_iou(mask_tensor, labels_ori[i:i+1])
                        boundary_iou = compute_boundary_iou(mask_tensor, labels_ori[i:i+1])

                        ious.append(iou.item() if torch.is_tensor(iou) else iou)
                        boundary_ious.append(boundary_iou.item() if torch.is_tensor(boundary_iou) else boundary_iou)

                    except Exception as e:
                        print(f"\nError processing sample {idx}, image {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Add zero scores for failed samples
                        ious.append(0.0)
                        boundary_ious.append(0.0)

            total_images += batch_size
            progress_bar.update(1)

        progress_bar.close()

        # Calculate statistics
        results = {
            'miou': np.mean(ious),
            'miou_std': np.std(ious),
            'boundary_iou': np.mean(boundary_ious),
            'boundary_iou_std': np.std(boundary_ious),
            'num_samples': len(ious),
        }

        # Print results
        print(f"\n{'='*80}")
        print("Results:")
        print(f"{'='*80}")
        print(f"  mIoU: {results['miou']:.4f} ± {results['miou_std']:.4f}")
        print(f"  Boundary IoU: {results['boundary_iou']:.4f} ± {results['boundary_iou_std']:.4f}")
        print(f"  Samples evaluated: {results['num_samples']}")
        print(f"{'='*80}\n")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SAM2 on HQ44k dataset'
    )

    # Model parameters
    parser.add_argument('--model-cfg', type=str,
                       default='sam2_hiera_l.yaml',
                       help='SAM2 model config (e.g., sam2_hiera_l.yaml, sam2_hiera_b_plus.yaml)')
    parser.add_argument('--checkpoint', type=str,
                       required=True,
                       help='Path to SAM2 checkpoint')

    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for dataloader')
    parser.add_argument('--use-batch', action='store_true',
                       help='Use SAM2 native batch processing (faster for batch > 1)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of dataloader workers')

    # Entropy processor parameters
    parser.add_argument('--processor', type=str, default=None,
                       choices=[None, 'POSITIONAL_PRUNE_SAM2', 'HEAD_PRUNE_SAM2', 'POSITIONAL_QUANT_SAM2'],
                       help='SAM2 entropy processor to use (None = no processing)')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to config YAML file for processor parameters')
    parser.add_argument('--num-calib-samples', type=int, default=32,
                       help='Number of calibration samples for entropy processor')
    parser.add_argument('--percent-entropy', type=float, default=0.5,
                       help='Percentage of heads to prune/quantize')
    parser.add_argument('--percent-entropy-global', type=float, default=0.3,
                       help='Percentage of heads to prune/quantize')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Percentage of heads to prune/quantize')
    parser.add_argument('--high-entropy', action='store_true',
                       help='Prune high entropy heads (default: prune low entropy)')
    parser.add_argument('--prune-global', action='store_true',
                       help='Apply global pruning across all layers')

    # Dataset
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Root directory for datasets')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"\n{'='*80}")
    print("SAM2 HQ44k Evaluation")
    print(f"{'='*80}")
    print(f"Model config: {args.model_cfg}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    if args.processor:
        print(f"Processor: {args.processor}")
        print(f"  Calibration samples: {args.num_calib_samples}")
        print(f"  Percent entropy: {args.percent_entropy}")
        print(f"  High entropy: {args.high_entropy}")
        print(f"  Prune global: {args.prune_global}")
    print(f"{'='*80}\n")

    # Build SAM2 model
    print("Loading SAM2 model...")
    sam2_model = build_sam2(
        config_file=args.model_cfg,
        ckpt_path=args.checkpoint,
        device=args.device
    )
    predictor = SAM2ImagePredictor(sam2_model)
    print("✓ Model loaded\n")

    # Apply entropy processor if specified
    if args.processor:
        print(f"\n{'='*80}")
        print(f"Setting up {args.processor}")
        print(f"{'='*80}\n")

        # Get processor
        print(args.processor)
        processor = get_sam2_processor(args.processor)

        # Create mock args for set_params (if config file not provided)
        if args.config_file:
            config = OmegaConf.load(args.config_file)
        else:
            # Create minimal config
            config = OmegaConf.create({
                'quantization': {
                    'percent_entropy': args.percent_entropy,
                    'percent_entropy_global': args.percent_entropy_global,
                    'high_entropy': args.high_entropy,
                    'prune_global': args.prune_global,
                    'threshold': args.threshold,
                }
            })

        # Set processor parameters
        processor.set_params(config)
        print(f"✓ Processor parameters set")
        print(f"  Percent: {processor.percent}")
        print(f"  Global Percent: {processor.global_percent}")
        print(f"  High entropy: {processor.prunehighentropy}")
        print(f"  Global: {processor.prune_global}\n")

        # Calibrate processor
        print("Calibrating processor...")
        processor.calibrate(
            predictor=predictor,
            modules=MultiScaleAttention,
            num_samples=args.num_calib_samples
        )
        print("✓ Processor calibrated\n")

        # Apply monkey patch to integrate processor into model
        print("Applying monkey patch to SAM2 image encoder...")
        sam2_image_encoder_monkey_patch(
            model=sam2_model,
            processor=processor,
            verbose=True
        )
        print("✓ Monkey patch applied\n")

    # Setup dataset
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

    print(f"✓ Dataset loaded: {len(gos_dataset)} samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Use batch mode: {args.use_batch}")
    print(f"  Samples to evaluate: {args.num_samples if args.num_samples else 'all'}\n")

    # Run evaluation
    evaluator = SAM2Evaluator()
    results = evaluator.eval_hq44k(
        predictor=predictor,
        dataloader=dataloader,
        num_samples=args.num_samples,
        use_batch=args.use_batch
    )

    return results


if __name__ == '__main__':
    main()
