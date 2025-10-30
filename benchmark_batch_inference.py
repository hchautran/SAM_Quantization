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
        --config-file quant/config/hq44k/rtn.yaml \
        --batch-sizes 1 2 4 8 16 \
        --num-samples 100 \
        --quantize-encoder \
        --n-bits 4
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
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# SAM imports
from segment_anything import SamPredictor, sam_model_registry

# Local imports
from small_engine import Engine, override_args, get_default_datasets
from processors import get_encoder_processor, DecoderDoNothingProcessor
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou


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


class BatchInferenceBenchmark:
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

        Args:
            predictor: SamPredictor instance
            images: Tensor of shape [B, 3, H, W]
            labels_boxes: Bounding boxes for each image [B, 4]
            labels_ori: Original labels for IoU computation

        Returns:
            Dict with metrics (iou, boundary_iou, encoder_time_ms)
        """
        batch_size = images.shape[0]
        device = predictor.device

        # Ensure all tensors are on the correct device
        images = images.to(device)
        labels_boxes = labels_boxes.to(device)
        labels_ori = labels_ori.to(device)

        # Transform images for encoder
        transformed_images = predictor.model.preprocess(images)

        # Measure encoder time for the entire batch
        torch.cuda.synchronize()
        encoder_start = time.time()

        with torch.no_grad():
            features, interm_features = predictor.model.image_encoder(transformed_images)

        torch.cuda.synchronize()
        encoder_end = time.time()
        encoder_time_ms = (encoder_end - encoder_start) * 1000

        # Now process each image's decoder separately (SAM decoder doesn't support batching well)
        all_ious = []
        all_boundary_ious = []

        for i in range(batch_size):
            # Set features for this specific image
            predictor.features = features[i:i+1]
            predictor.interm_features = [f[i:i+1] for f in interm_features]
            predictor.original_size = (images.shape[2], images.shape[3])
            predictor.input_size = tuple(transformed_images.shape[-2:])
            predictor.is_image_set = True

            # Predict mask
            try:
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=labels_boxes[i:i+1],
                    hq_token_only=True
                )

                # Calculate metrics
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
        """
        Benchmark a specific batch size.

        Args:
            predictor: SamPredictor instance
            batch_size: Batch size to test
            dataloader: DataLoader with matching batch size
            num_samples: Total number of samples to process

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*80}")
        print(f"Benchmarking batch_size = {batch_size}")
        print(f"{'='*80}\n")

        self.reset_memory()

        # Track metrics
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

            # Get batch data
            images = data_val['image']  # [B, C, H, W]
            labels_val = data_val['label']  # [B, 1, H, W]
            labels_ori = data_val['ori_label']  # [B, 1, H, W]
            current_batch_size = images.shape[0]

            # Get bounding boxes - handle both batched and unbatched cases
            if labels_val.dim() == 4:
                # Batched: [B, 1, H, W]
                labels_boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])
            else:
                # Single image: [1, H, W]
                labels_boxes = misc.masks_to_boxes(labels_val[0:1, :, :])

            # Process batch
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

        # Calculate statistics
        encoder_times = np.array(encoder_times)
        encoder_times_per_image = np.array(encoder_times_per_image)

        throughput = total_images / overall_time

        # Get memory stats
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

            # Encoder batch time (total time for batch)
            'encoder_batch_mean_ms': np.mean(encoder_times),
            'encoder_batch_std_ms': np.std(encoder_times),
            'encoder_batch_min_ms': np.min(encoder_times),
            'encoder_batch_max_ms': np.max(encoder_times),

            # Encoder per-image time (batch time / batch_size)
            'encoder_per_image_mean_ms': np.mean(encoder_times_per_image),
            'encoder_per_image_std_ms': np.std(encoder_times_per_image),

            # Quality metrics
            'miou': np.mean(ious),
            'miou_std': np.std(ious),
            'boundary_iou': np.mean(boundary_ious),
            'boundary_iou_std': np.std(boundary_ious),

            # Memory
            **memory_stats,

            'timestamp': datetime.datetime.now().isoformat(),
        }

        # Print summary
        print(f"\n✓ Results for batch_size={batch_size}:")
        print(f"  Throughput: {throughput:.2f} images/sec")
        print(f"  Encoder time (batch): {np.mean(encoder_times):.2f} ± {np.std(encoder_times):.2f} ms")
        print(f"  Encoder time (per image): {np.mean(encoder_times_per_image):.2f} ± {np.std(encoder_times_per_image):.2f} ms")
        print(f"  Peak memory: {memory_stats.get('peak_memory_allocated_mb', 0):.2f} MB")
        print(f"  mIoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")

        return result

    def run_benchmark(
        self,
        predictor: SamPredictor,
        batch_sizes: List[int],
        num_samples: int,
        datasets_config: List[Dict]
    ) -> List[Dict]:
        """
        Run benchmark across multiple batch sizes.

        Args:
            predictor: SamPredictor instance
            batch_sizes: List of batch sizes to test
            num_samples: Number of samples per batch size
            datasets_config: Dataset configuration

        Returns:
            List of result dictionaries
        """
        all_results = []

        for batch_size in batch_sizes:
            # Create dataloader with specific batch size
            valid_im_gt_list = get_im_gt_name_dict([datasets_config[0]], flag="valid")

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

            # Run benchmark
            result = self.benchmark_batch_size(
                predictor=predictor,
                batch_size=batch_size,
                dataloader=dataloader,
                num_samples=num_samples
            )

            all_results.append(result)

            # Cleanup
            del dataloader, gos_dataset
            self.reset_memory()

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SAM encoder with batch inference'
    )

    # Config
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to config YAML file')

    # Benchmark parameters
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples per batch size')
    parser.add_argument('--num-calib-samples', type=int, default=16,
                       help='Number of calibration samples')

    # Model parameters
    parser.add_argument('--processor', type=str, default='POSITIONAL_PRUNE',
                       choices=['base','POSITIONAL_PRUNE', 'POSITIONAL_QUANT', 'HEAD_PRUNE'],
                       help='Processor to use')
    parser.add_argument('--quantize-encoder', action='store_true',
                       help='Enable encoder quantization')
    parser.add_argument('--quantize-decoder', action='store_true',
                       help='Enable decoder quantization')

    # Quantization parameters
    parser.add_argument('--n-bits', type=int, default=16,
                       help='Number of quantization bits')
    parser.add_argument('--n-bits-mlp', type=int, default=4,
                       help='Number of quantization bits for MLP')
    parser.add_argument('--en-weight-quant', type=str, default='per_channel',
                       help='Encoder weight quantization method')
    parser.add_argument('--en-act-quant', type=str, default='per_token',
                       help='Encoder activation quantization method')
    parser.add_argument('--de-weight-quant', type=str, default='per_channel',
                       help='Decoder weight quantization method')
    parser.add_argument('--de-act-quant', type=str, default='per_token',
                       help='Decoder activation quantization method')
    parser.add_argument('--k-preserve', type=int, default=0,
                       help='Number of channels to preserve')

    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config_file)
    config = override_args(args, config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    print("Loading SAM model...")
    model_type = 'vit_l'
    checkpoint_path = './pretrained_checkpoint/sam_hq_vit_l.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Initialize engine
    engine = Engine(
        'batch_benchmark',
        quantize_encoder=args.quantize_encoder,
        quantize_decoder=args.quantize_decoder
    )

    # Get processor
    enc_processor = get_encoder_processor(args.processor)

    # Setup and calibrate
    print(f"Calibrating {args.processor}...")
    encoder_processor, decoder_processor = engine.setup_and_calibrate_processors(
        predictor,
        num_calib_samples=args.num_calib_samples,
        encoder_processor=enc_processor,
        decoder_processor=DecoderDoNothingProcessor("DO_NOTHING"),
        args_yaml=config,
    )

    # Apply quantization
    encoder_config = {
        'processor': encoder_processor,
        'n_bits': args.n_bits,
        'weight_quant': args.en_weight_quant,
        'act_quant': args.en_act_quant,
    } if args.quantize_encoder else None

    decoder_config = {
        'processor': decoder_processor,
        'n_bits': args.n_bits,
        'weight_quant': args.de_weight_quant,
        'act_quant': args.de_act_quant,
        'k_preserve': args.k_preserve
    } if args.quantize_decoder else None

    engine.apply_quantization(predictor, encoder_config, decoder_config, config)

    # Run benchmark
    print(f"\n{'='*80}")
    print("Starting Batch Inference Benchmark")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Samples per batch: {args.num_samples}")
    print(f"{'='*80}\n")

    datasets = get_default_datasets()
    benchmark = BatchInferenceBenchmark()

    results = benchmark.run_benchmark(
        predictor=predictor,
        batch_sizes=args.batch_sizes,
        num_samples=args.num_samples,
        datasets_config=datasets
    )

    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(
        args.output_dir,
        f'batch_inference_results_{timestamp}.csv'
    )

    df = pd.DataFrame(results)

    # Add configuration info
    df['processor'] = args.processor
    df['quantize_encoder'] = args.quantize_encoder
    df['n_bits'] = args.n_bits
    df['weight_quant'] = args.en_weight_quant
    df['act_quant'] = args.en_act_quant

    df.to_csv(csv_filename, index=False)

    # Print summary
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
    print(f"\n✓ Best throughput: {max(r['throughput_imgs_per_sec'] for r in results):.2f} imgs/sec "
          f"at batch_size={max(results, key=lambda x: x['throughput_imgs_per_sec'])['batch_size']}")


if __name__ == '__main__':
    main()
