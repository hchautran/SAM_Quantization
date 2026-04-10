#!/usr/bin/env python3
"""
Benchmark SAM encoder with true batch processing - multiple images in a single forward pass.

This script measures:
- Throughput (images/sec) at different batch sizes
- Latency per image at different batch sizes
- GPU memory usage at different batch sizes
- mIoU quality metrics maintained across batch sizes

Usage:
    python benchmark_batch_inference.py \
        --batch-sizes 1 2 4 8 16 \
        --num-samples 100
"""

import os
import gc
import time
import argparse
import datetime
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

# SAM imports
from segment_anything import SamPredictor, sam_model_registry

# Local imports
from sam_engine import get_default_datasets
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized ori_im fields.
    Stack tensors that can be stacked, keep lists for variable-sized items.
    """
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


class BatchEvaluator:
    """Benchmark SAM encoder with batched image processing"""

    def __init__(self):
        self.results = []

    def reset_memory(self):
        """Reset CUDA memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def process_batch_images(
        self,
        predictor: SamPredictor,
        images: torch.Tensor,
        labels_boxes: torch.Tensor,
        labels_ori: torch.Tensor
    ):
        """
        Process a batch of images through the encoder in a single forward pass.

        Returns:
            Dict with metrics (iou, boundary_iou, encoder_time_ms)
        """
        batch_size = images.shape[0]
        device = predictor.device

        images = images.to(device)
        labels_boxes = labels_boxes.to(device)
        labels_ori = labels_ori.to(device)

        transformed_images = predictor.model.preprocess(images)

        torch.cuda.synchronize()
        encoder_start = time.time()

        with torch.no_grad():
            features, interm_features = predictor.model.image_encoder(transformed_images)

        torch.cuda.synchronize()
        encoder_end = time.time()
        encoder_time_ms = (encoder_end - encoder_start) * 1000

        all_ious = []
        all_boundary_ious = []

        for i in range(batch_size):
            predictor.features = features[i:i+1]
            predictor.interm_features = [f[i:i+1] for f in interm_features]
            predictor.original_size = (images.shape[2], images.shape[3])
            predictor.input_size = tuple(transformed_images.shape[-2:])
            predictor.is_image_set = True

            try:
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=labels_boxes[i:i+1],
                    hq_token_only=True
                )

                iou = compute_iou(masks, labels_ori[i:i+1])
                boundary_iou = compute_boundary_iou(masks, labels_ori[i:i+1])

                all_ious.append(iou)
                all_boundary_ious.append(boundary_iou)

            except Exception as e:
                print(f"Error processing image {i} in batch: {e}")
                import traceback
                traceback.print_exc()
                all_ious.append(torch.tensor(0.0, device=device))
                all_boundary_ious.append(torch.tensor(0.0, device=device))

        return {
            'encoder_time_ms': encoder_time_ms,
            'encoder_time_per_image_ms': encoder_time_ms / batch_size,
            'iou': torch.mean(torch.stack(all_ious)).item(),
            'boundary_iou': torch.mean(torch.stack(all_boundary_ious)).item(),
        }

    def benchmark_batch_size(
        self,
        predictor: SamPredictor,
        batch_size: int,
        dataloader: DataLoader,
        num_samples: int
    ) -> Dict:
        """Benchmark a specific batch size."""
        print(f"\n{'='*80}")
        print(f"Benchmarking batch_size = {batch_size}")
        print(f"{'='*80}\n")

        self.reset_memory()

        encoder_times = []
        encoder_times_per_image = []
        ious = []
        boundary_ious = []
        total_images = 0

        progress_bar = tqdm(
            total=min(len(dataloader), num_samples // batch_size),
            desc=f"Batch {batch_size}"
        )

        overall_start = time.time()

        for batch_idx, data_val in enumerate(dataloader):
            if total_images >= num_samples:
                break

            images = data_val['image']
            labels_val = data_val['label']
            labels_ori = data_val['ori_label']
            current_batch_size = images.shape[0]

            if labels_val.dim() == 4:
                labels_boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])
            else:
                labels_boxes = misc.masks_to_boxes(labels_val[0:1, :, :])

            try:
                metrics = self.process_batch_images(
                    predictor,
                    images,
                    labels_boxes,
                    labels_ori
                )

                encoder_times.append(metrics['encoder_time_ms'])
                encoder_times_per_image.append(metrics['encoder_time_per_image_ms'])
                ious.append(metrics['iou'])
                boundary_ious.append(metrics['boundary_iou'])

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

            total_images += current_batch_size
            progress_bar.update(1)

        overall_end = time.time()
        overall_time = overall_end - overall_start

        progress_bar.close()

        encoder_times = np.array(encoder_times)
        encoder_times_per_image = np.array(encoder_times_per_image)

        throughput = total_images / overall_time

        memory_stats = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_stats = {
                'peak_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'peak_memory_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
            }

        result = {
            'batch_size': batch_size,
            'num_images': total_images,
            'total_time_sec': overall_time,
            'throughput_imgs_per_sec': throughput,
            'encoder_batch_mean_ms': np.mean(encoder_times),
            'encoder_batch_std_ms': np.std(encoder_times),
            'encoder_batch_min_ms': np.min(encoder_times),
            'encoder_batch_max_ms': np.max(encoder_times),
            'encoder_per_image_mean_ms': np.mean(encoder_times_per_image),
            'encoder_per_image_std_ms': np.std(encoder_times_per_image),
            'miou': np.mean(ious),
            'miou_std': np.std(ious),
            'boundary_iou': np.mean(boundary_ious),
            'boundary_iou_std': np.std(boundary_ious),
            **memory_stats,
            'timestamp': datetime.datetime.now().isoformat(),
        }

        print(f"\nResults for batch_size={batch_size}:")
        print(f"  Throughput: {throughput:.2f} images/sec")
        print(f"  Encoder time (batch): {np.mean(encoder_times):.2f} +/- {np.std(encoder_times):.2f} ms")
        print(f"  Encoder time (per image): {np.mean(encoder_times_per_image):.2f} +/- {np.std(encoder_times_per_image):.2f} ms")
        print(f"  Peak memory: {memory_stats.get('peak_memory_allocated_mb', 0):.2f} MB")
        print(f"  mIoU: {np.mean(ious):.4f} +/- {np.std(ious):.4f}")

        wandb.log({
            f'batch_{batch_size}/throughput_imgs_per_sec': throughput,
            f'batch_{batch_size}/encoder_batch_mean_ms': np.mean(encoder_times),
            f'batch_{batch_size}/encoder_batch_std_ms': np.std(encoder_times),
            f'batch_{batch_size}/encoder_per_image_mean_ms': np.mean(encoder_times_per_image),
            f'batch_{batch_size}/encoder_per_image_std_ms': np.std(encoder_times_per_image),
            f'batch_{batch_size}/peak_memory_allocated_mb': memory_stats.get('peak_memory_allocated_mb', 0),
            f'batch_{batch_size}/peak_memory_reserved_mb': memory_stats.get('peak_memory_reserved_mb', 0),
            f'batch_{batch_size}/miou': np.mean(ious),
            f'batch_{batch_size}/miou_std': np.std(ious),
            f'batch_{batch_size}/boundary_iou': np.mean(boundary_ious),
            f'batch_{batch_size}/boundary_iou_std': np.std(boundary_ious),
        })

        return result

    def run_benchmark(
        self,
        predictor: SamPredictor,
        batch_sizes: List[int],
        num_samples: int,
        datasets_config: List[Dict]
    ) -> List[Dict]:

        all_results = []

        for batch_size in batch_sizes:
            valid_im_gt_list = get_im_gt_name_dict([datasets_config[1]], flag="valid")

            gos_dataset = OnlineDataset(
                [valid_im_gt_list[0]],
                transform=transforms.Compose([Resize([1024, 1024])]),
                eval_ori_resolution=True
            )

            dataloader = DataLoader(
                gos_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )

            result = self.benchmark_batch_size(
                predictor=predictor,
                batch_size=batch_size,
                dataloader=dataloader,
                num_samples=num_samples
            )

            all_results.append(result)

            del dataloader, gos_dataset
            self.reset_memory()

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SAM encoder with batch inference'
    )

    # Benchmark parameters
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples per batch size')

    # Model parameters
    parser.add_argument('--model-ckt', type=str, default='./ckts/sam_hq_vit_l.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--model-type', type=str, default='vit_l',
                       help='SAM model type')

    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='sam-batch-benchmark',
                       help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Wandb run name')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'batch_sizes': args.batch_sizes,
                'num_samples': args.num_samples,
                'model_ckt': args.model_ckt,
                'model_type': args.model_type,
            }
        )
    else:
        wandb.init(mode='disabled')

    print("Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to('cuda')
    predictor = SamPredictor(sam)

    print(f"\n{'='*80}")
    print("Starting Batch Inference Benchmark")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Samples per batch: {args.num_samples}")
    print(f"{'='*80}\n")

    datasets = get_default_datasets()
    batch_evaluator = BatchEvaluator()

    results = batch_evaluator.run_benchmark(
        predictor=predictor,
        batch_sizes=args.batch_sizes,
        num_samples=args.num_samples,
        datasets_config=datasets
    )

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(args.output_dir, f'batch_inference_results_{timestamp}.csv')

    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results saved to: {csv_filename}\n")
    print("Summary:")
    print("-" * 80)
    print(f"{'Batch':>6} | {'Throughput':>12} | {'Encoder/img':>12} | {'Memory':>10} | {'mIoU':>8}")
    print(f"{'Size':>6} | {'(imgs/sec)':>12} | {'(ms)':>12} | {'(MB)':>10} | {'':>8}")
    print("-" * 80)

    for result in results:
        print(f"{result['batch_size']:>6} | "
              f"{result['throughput_imgs_per_sec']:>12.2f} | "
              f"{result['encoder_per_image_mean_ms']:>12.2f} | "
              f"{result.get('peak_memory_allocated_mb', 0):>10.0f} | "
              f"{result['miou']:>8.4f}")

    print("-" * 80)

    best_throughput_result = max(results, key=lambda x: x['throughput_imgs_per_sec'])
    print(f"\nBest throughput: {best_throughput_result['throughput_imgs_per_sec']:.2f} imgs/sec "
          f"at batch_size={best_throughput_result['batch_size']}")

    wandb.log({
        'summary/best_throughput': best_throughput_result['throughput_imgs_per_sec'],
        'summary/best_throughput_batch_size': best_throughput_result['batch_size'],
        'summary/best_miou': max(r['miou'] for r in results),
        'summary/min_latency_per_image': min(r['encoder_per_image_mean_ms'] for r in results),
    })

    wandb.log({'results_table': wandb.Table(dataframe=df)})
    wandb.finish()

    return results


if __name__ == '__main__':
    main()
