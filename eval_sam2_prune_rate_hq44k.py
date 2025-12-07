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
from prunning_rate.sam2prune import monkey_patch_train_sam2

# SAM2 entropy processors
from processors.encoder.entropy_sam2 import (
    PositionalPruneSAM2Processor,
    HeadPruneSAM2Processor,
    PositionalQuantSAM2Processor,
    PositionalTrainingPruneRateSAM2Processor,
)
from processors.sam2_observer import sam2_image_encoder_monkey_patch
from prunning_rate.sam2prune import DiffPruneRateMultiScaleAttention

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
    "TRAINING_PRUNE_RATE_SAM2": PositionalTrainingPruneRateSAM2Processor,
}


def get_sam2_processor(name: str, **kwargs):
    """Get SAM2 processor by name."""
    if name not in SAM2_PROCESSOR_REGISTRY:
        available = list(SAM2_PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown SAM2 processor '{name}'. Available: {available}")
    return SAM2_PROCESSOR_REGISTRY[name](**kwargs)

def analyze_model_head_pruning_and_flops(predictor_model, manual_local_heads=None, manual_global_heads=None):
    """
    Analyze head pruning ratios and calculate FLOPs for attention operations.
    
    Args:
        predictor_model: The SAM model with DiffPruneRateMultiScaleAttention modules
        manual_local_heads: Number of local heads to keep manually (optional)
        manual_global_heads: Number of global heads to keep manually (optional)
        
    Returns:
        dict: Dictionary containing head statistics and FLOPs information
    """
    # Initialize counters
    total_kept_heads_local = 0
    total_original_heads_local = 0
    total_kept_heads_global = 0
    total_original_heads_global = 0
    
    # For FLOPs calculation
    total_flops = 0
    total_baseline_flops = 0  # FLOPs without pruning
    
    nu_layer_global = 0
    nu_layer_local = 0
    
    # Collect per-layer information
    layer_info = []
    
    # Iterate through all modules to find DiffPruneRateMultiScaleAttention modules
    for name, module in predictor_model.named_modules():
        if isinstance(module, DiffPruneRateMultiScaleAttention):
            # Get head information
            kept_heads = int(module.prune_ddp.update_kept_head_number())
            original_heads = module.prune_ddp.head_number
            
            # Determine if this is a local or global attention
            if original_heads ==8:
                is_local = False
            else:
                is_local = True
            
            # For FLOPs calculation, we assume fixed dimensions
           
            if original_heads == 8:
                batch_factor =1 
            elif original_heads %100 ==0:
                batch_factor =25
            else:
                batch_factor =1024
            

            if is_local: 
                nu_layer_local += 1
            else: 
                nu_layer_global += 1
            # Using typical attention map dimensions for SAM
            H, W = (14, 14) if is_local else (64, 64)
            
            # Calculate FLOPs for this attention module
            qkv_flops = module._calculate_qkv_flops(batch_factor, H, W)
            proj_flops = module._calculate_projection_flops(batch_factor, H, W)
            attention_flops = module._calculate_attention_flops(H, W, kept_heads)
            baseline_attention_flops = module._calculate_attention_flops(H, W, original_heads)
            
            module_flops = qkv_flops + proj_flops + attention_flops
            module_baseline_flops = qkv_flops + proj_flops + baseline_attention_flops
            
            # Update totals
            total_flops += module_flops
            total_baseline_flops += module_baseline_flops
            
            # Store layer information
            layer_info.append({
                'name': name,
                'kept_heads': kept_heads,
                'original_heads': original_heads,
                'is_local': is_local,
                'flops': module_flops,
                'baseline_flops': module_baseline_flops
            })
            
            # Update head counters
            if is_local:
                total_kept_heads_local += kept_heads
                total_original_heads_local += original_heads
            else:
                total_kept_heads_global += kept_heads
                total_original_heads_global += original_heads
    
    # Calculate overall ratios
    overall_local_ratio = total_kept_heads_local / total_original_heads_local if total_original_heads_local > 0 else 0
    overall_global_ratio = total_kept_heads_global / total_original_heads_global if total_original_heads_global > 0 else 0
    flops_reduction = (1 - total_flops / total_baseline_flops) * 100 if total_baseline_flops > 0 else 0
    
    # Calculate manual heads if not provided
    avg_local_heads = total_kept_heads_local / nu_layer_local if nu_layer_local > 0 else 0
    avg_global_heads = total_kept_heads_global / nu_layer_global if nu_layer_global > 0 else 0
    
    # Round up to nearest integer (ceiling)
    if manual_local_heads is None:
        manual_local_heads = int(avg_local_heads + 0.999)  # Equivalent to math.ceil()
    if manual_global_heads is None:
        manual_global_heads = int(avg_global_heads + 0.999)  # Equivalent to math.ceil()
    
    # Recalculate manual FLOPs
    total_manual_flops = 0
    layer_idx_local = 0
    layer_idx_global = 0
    
    for name, module in predictor_model.named_modules():
        if isinstance(module, DiffPruneRateMultiScaleAttention):
            original_heads = module.prune_ddp.head_number
            if original_heads ==8:
                is_local = False
            else:
                is_local = True
            
            if original_heads == 8:
                batch_factor =1 
            elif original_heads %100 ==0:
                batch_factor =25
            else:
                batch_factor =1024
            H, W = (14, 14) if is_local else (64, 64)
            
            # Calculate FLOPs with manual head counts
            qkv_flops = module._calculate_qkv_flops(batch_factor, H, W)
            proj_flops = module._calculate_projection_flops(batch_factor, H, W)
            
            # Use manual head counts
            if is_local:
                manual_attention_flops = module._calculate_attention_flops(H, W, manual_local_heads)
                layer_idx_local += 1
            else:
                manual_attention_flops = module._calculate_attention_flops(H, W, manual_global_heads)
                layer_idx_global += 1
            
            manual_module_flops = qkv_flops + proj_flops + manual_attention_flops
            total_manual_flops += manual_module_flops
    
    # Calculate manual FLOPs reduction
    manual_flops_reduction = None
    if total_baseline_flops > 0:
        manual_flops_reduction = (1 - total_manual_flops / total_baseline_flops) * 100
    
    return {
        'layer_info': layer_info,
        'head_stats': {
            'local_kept': total_kept_heads_local,
            'local_total': total_original_heads_local,
            'local_ratio': overall_local_ratio,
            'global_kept': total_kept_heads_global,
            'global_total': total_original_heads_global,
            'global_ratio': overall_global_ratio
        },
        'flops_stats': {
            'total_flops': total_flops,
            'baseline_flops': total_baseline_flops,
            'reduction_percent': flops_reduction,
            'manual_flops': total_manual_flops,
            'manual_reduction_percent': manual_flops_reduction
        },
        'manual_settings': {
            'manual_local_heads': manual_local_heads,
            'manual_global_heads': manual_global_heads,
            'avg_local_heads': avg_local_heads,
            'avg_global_heads': avg_global_heads
        }
    }
def print_head_pruning_and_flops_info(predictor_model):
    """
    Print detailed head pruning ratios and FLOPs information for the model.
    
    Args:
        predictor_model: The SAM model with DiffPruneRateMultiScaleAttention modules
    """
    print("\n=== Head Pruning Ratios ===")
    
    # Analyze model
    analysis = analyze_model_head_pruning_and_flops(predictor_model)
    
    # Print per-layer information
    for layer in analysis['layer_info']:
        layer_type = "Local" if layer['is_local'] else "Global"
        print(f"Layer {layer['name']}: {layer['kept_heads']}/{layer['original_heads']} ({layer_type})")
    
    # Print overall head statistics
    head_stats = analysis['head_stats']
    if head_stats['local_total'] > 0:
        print(f"\nOverall Local: {head_stats['local_kept']}/{head_stats['local_total']} heads kept "
              f"({head_stats['local_ratio']:.2%}) pruning rate: {1-head_stats['local_ratio']:.2%}")
    
    if head_stats['global_total'] > 0:
        print(f"\nOverall Global: {head_stats['global_kept']}/{head_stats['global_total']} heads kept "
              f"({head_stats['global_ratio']:.2%}) pruning rate: {1-head_stats['global_ratio']:.2%}")
    
    # Print manual head information
    manual_settings = analysis['manual_settings']
    print(f"\nManual Settings:")
    print(f"Average Local Heads: {manual_settings['avg_local_heads']:.2f}")
    print(f"Average Global Heads: {manual_settings['avg_global_heads']:.2f}")
    print(f"Manual Local Heads (rounded): {manual_settings['manual_local_heads']}")
    print(f"Manual Global Heads (rounded): {manual_settings['manual_global_heads']}")
    
    # Print FLOPs information
    flops_stats = analysis['flops_stats']
    print("\n=== FLOPs Information (Attention Only) ===")
    print(f"Total Attention FLOPs (with parameter pruning): {flops_stats['total_flops']/1e9:.2f} GFLOPs")
    print(f"Total Attention FLOPs (baseline): {flops_stats['baseline_flops']/1e9:.2f} GFLOPs")
    if flops_stats['reduction_percent'] >= 0:
        print(f"Attention FLOPs Reduction (parameter pruning): {flops_stats['reduction_percent']:.2f}%")
    
    # Print manual pruning FLOPs information if provided
    if flops_stats['manual_flops'] is not None:
        print(f"\nTotal Attention FLOPs (with manual pruning): {flops_stats['manual_flops']/1e9:.2f} GFLOPs")
        if flops_stats['manual_reduction_percent'] >= 0:
            print(f"Attention FLOPs Reduction (manual pruning): {flops_stats['manual_reduction_percent']:.2f}%")
        
        # Compare the two pruning approaches
        if flops_stats['total_flops'] and flops_stats['manual_flops']:
            flops_diff = flops_stats['manual_flops'] - flops_stats['total_flops']
            if flops_diff > 0:
                print(f"Manual pruning uses {flops_diff/1e9:.2f} GFLOPs more than parameter pruning")
            else:
                print(f"Manual pruning uses {abs(flops_diff)/1e9:.2f} GFLOPs less than parameter pruning")
    
    print("===========================\n")
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
                       choices=[None, 'POSITIONAL_PRUNE_SAM2', 'HEAD_PRUNE_SAM2', 'POSITIONAL_QUANT_SAM2',"TRAINING_PRUNE_RATE_SAM2"],
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
        if args.processor == 'TRAINING_PRUNE_RATE_SAM2':
            monkey_patch_train_sam2(
                model=sam2_model,
                processor=processor,
                model_type ='hiera_b_plus',
            )
        else :
            sam2_image_encoder_monkey_patch(
                model=sam2_model,
                processor=processor,
                verbose=True
            )
        print("✓ Monkey patch applied\n")

    # Setup dataset
    print("Loading dataset...")
    from sam_engine import get_default_datasets
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
    if args.processor == 'TRAINING_PRUNE_RATE_SAM2':
        ckpt_prune_rate_path = "/home/22chi.nh/project/SAMquantization/SAM_Quantization/logssam2/checkpoints/checkpoint.pt"
        sam2_model.load_state_dict(torch.load(ckpt_prune_rate_path)['model'])
    print_head_pruning_and_flops_info(predictor.model)
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
