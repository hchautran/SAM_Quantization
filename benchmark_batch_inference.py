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
import wandb

# SAM imports
from segment_anything import SamPredictor, sam_model_registry

# Local imports
from sam_engine import Engine, override_args, get_default_datasets,  setup_logger
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

        # Add logging
        logger.info(f"{'='*80}")
        logger.info(f"Benchmarking batch_size = {batch_size}")
        logger.info(f"{'='*80}")

        self.reset_memory()

        # Track metrics
        encoder_times = []
        encoder_times_per_image = []
        ious = []
        boundary_ious = []
        total_images = 0

        # Calculate actual number of samples
        actual_num_samples = min(len(dataloader.dataset), num_samples)
        logger.info(f"Processing {actual_num_samples} samples (requested: {num_samples})")

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
                import traceback
                traceback.print_exc()
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
        print(f"  Peak reserved memory: {memory_stats.get('peak_memory_reserved_mb', 0):.2f} MB")
        print(f"  mIoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")

        # Add logging for all the same information
        logger.info(f"\n✓ Results for batch_size={batch_size}:")
        logger.info(f"  Throughput: {throughput:.2f} images/sec")
        logger.info(f"  Encoder time (batch): {np.mean(encoder_times):.2f} ± {np.std(encoder_times):.2f} ms")
        logger.info(f"  Encoder time (per image): {np.mean(encoder_times_per_image):.2f} ± {np.std(encoder_times_per_image):.2f} ms")
        logger.info(f"  Peak memory: {memory_stats.get('peak_memory_allocated_mb', 0):.2f} MB")
        logger.info(f"  Peak reserved memory: {memory_stats.get('peak_memory_reserved_mb', 0):.2f} MB")
        logger.info(f"  mIoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")

        # Log additional details
        logger.info(f"  Number of images processed: {total_images}")
        logger.info(f"  Total time: {overall_time:.2f} seconds")
        logger.info(f"  Boundary IoU: {np.mean(boundary_ious):.4f} ± {np.std(boundary_ious):.4f}")

        # Log to wandb
        # wandb.log({
        #     f'batch_{batch_size}/throughput_imgs_per_sec': throughput,
        #     f'batch_{batch_size}/encoder_batch_mean_ms': np.mean(encoder_times),
        #     f'batch_{batch_size}/encoder_batch_std_ms': np.std(encoder_times),
        #     f'batch_{batch_size}/encoder_per_image_mean_ms': np.mean(encoder_times_per_image),
        #     f'batch_{batch_size}/encoder_per_image_std_ms': np.std(encoder_times_per_image),
        #     f'batch_{batch_size}/peak_memory_allocated_mb': memory_stats.get('peak_memory_allocated_mb', 0),
        #     f'batch_{batch_size}/peak_memory_reserved_mb': memory_stats.get('peak_memory_reserved_mb', 0),
        #     f'batch_{batch_size}/miou': np.mean(ious),
        #     f'batch_{batch_size}/miou_std': np.std(ious),
        #     f'batch_{batch_size}/boundary_iou': np.mean(boundary_ious),
        #     f'batch_{batch_size}/boundary_iou_std': np.std(boundary_ious),
        # })

        return result

    def run_benchmark(
        self,
        predictor: SamPredictor,
        batch_sizes: List[int],
        num_samples: int,
        datasets_config: List[Dict],
        logger,  # Make sure logger is passed here
    ) -> List[Dict]:

        all_results = dict()

        for i in range(len(datasets_config)):
            logger.info(f"{'='*250}")
            dataname= datasets_config[i]["name"]
            print(f"Running benchmark for dataset {dataname}...")
            logger.info(f"Running benchmark for dataset {dataname}...")

            all_results[datasets_config[i]["name"]] = []
            
            # Log dataset information
            logger.info(f"Dataset: {datasets_config[i]['name']}")
            logger.info(f"Dataset path: {datasets_config[i]['im_dir']}")
            
            for batch_size in batch_sizes:
                # Create dataloader with specific batch size
                valid_im_gt_list = get_im_gt_name_dict([datasets_config[i]], flag="valid")

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

                # Log batch size before processing
                logger.info(f"Processing batch_size={batch_size} for dataset {datasets_config[i]['name']}")
                
                # Run benchmark
                result = self.benchmark_batch_size(
                    predictor=predictor,
                    batch_size=batch_size,
                    dataloader=dataloader,
                    num_samples=num_samples
                )

                all_results[datasets_config[i]["name"]].append(result)

                # Log completion of batch size
                logger.info(f"Completed benchmark for batch_size={batch_size}, dataset={datasets_config[i]['name']}")
                logger.info(f"  - Images processed: {result['num_images']}")
                logger.info(f"  - Throughput: {result['throughput_imgs_per_sec']:.2f} images/sec")
                logger.info(f"  - mIoU: {result['miou']:.4f}")
                logger.info(f"  - Peak memory: {result.get('peak_memory_allocated_mb', 0):.2f} MB")

                # Cleanup
                del dataloader, gos_dataset
                self.reset_memory()

        return all_results


def run_single_benchmark(args):
    """Run a single benchmark with given arguments"""
    # Initialize wandb
    # if not args.no_wandb:
    #     wandb.init(
    #         project=args.wandb_project,
    #         name=args.wandb_run_name,
    #         config={
    #             'percent': args.percent,
    #             'percent_global': args.percent_global,
    #             'prune_global': args.prune_global,
    #             'n_bits': args.n_bits,
    #             'n_bits_mlp': args.n_bits_mlp,
    #             'high_entropy': args.high_entropy,
    #             'batch_sizes': args.batch_sizes,
    #             'num_samples': args.num_samples,
    #             'num_calib_samples': args.num_calib_samples,
    #             'processor': args.processor,
    #             'quantize_encoder': args.quantize_encoder,
    #             'quantize_decoder': args.quantize_decoder,
    #             'en_weight_quant': args.en_weight_quant,
    #             'en_act_quant': args.en_act_quant,
    #             'de_weight_quant': args.de_weight_quant,
    #             'de_act_quant': args.de_act_quant,
    #             'k_preserve': args.k_preserve,
    #             'model_ckt': args.model_ckt,
    #             'model_type': args.model_type,
    #         }
    #     )
    # else:
    #     wandb.init(mode='disabled')

    # Initialize model
    print("Loading SAM model...")
    model_type = args.model_type
    checkpoint_path = args.model_ckt
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Initialize engine
    engine = Engine(
        'batch_benchmark',
        quantize_encoder=args.quantize_encoder,
        quantize_decoder=args.quantize_decoder
    )


    if args.processor == "BASE":
        states = "Baseline"
    elif args.processor == "POSITIONAL_PRUNE":
        states = "Manual_positional_prune"
    elif args.processor == "POSITIONAL_QUANT":
        states = "Manual_positional_quant"
    
    logger = setup_logger("./logs", states)  # Use the states variable to create logger

    # Get processor
    enc_processor = get_encoder_processor(args.processor)

    # Setup and calibrate
    print(f"Calibrating {args.processor}...")
    encoder_processor, decoder_processor = engine.setup_and_calibrate_processors(
        predictor,
        num_calib_samples=args.num_calib_samples,
        encoder_processor=enc_processor,
        decoder_processor=DecoderDoNothingProcessor("DO_NOTHING"),
        args_yaml=args,
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

    engine.apply_quantization(predictor, encoder_config, decoder_config, args)

    # Run benchmark
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
        datasets_config=datasets,
        logger=logger
    )

    # In the main run_single_benchmark function, after printing the summary, add:
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {csv_filename}")

    logger.info("Summary:")
    for result in results:
        logger.info(f"Batch Size {result['batch_size']}: "
                    f"Throughput={result['throughput_imgs_per_sec']:.2f} imgs/sec, "
                    f"Encoder/img={result['encoder_per_image_mean_ms']:.2f}ms, "
                    f"Memory={result.get('peak_memory_allocated_mb', 0):.0f}MB, "
                    f"mIoU={result['miou']:.4f}")

    best_throughput_result = max(results, key=lambda x: x['throughput_imgs_per_sec'])
    logger.info(f"Best throughput: {best_throughput_result['throughput_imgs_per_sec']:.2f} imgs/sec "
                f"at batch_size={best_throughput_result['batch_size']}")

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
    df['percent'] = args.percent
    df['percent_global'] = args.percent_global

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

    best_throughput_result = max(results, key=lambda x: x['throughput_imgs_per_sec'])
    print(f"\n Best throughput: {best_throughput_result['throughput_imgs_per_sec']:.2f} imgs/sec "
          f"at batch_size={best_throughput_result['batch_size']}")

    # # Log summary to wandb
    # wandb.log({
    #     'summary/best_throughput': best_throughput_result['throughput_imgs_per_sec'],
    #     'summary/best_throughput_batch_size': best_throughput_result['batch_size'],
    #     'summary/best_miou': max(r['miou'] for r in results),
    #     'summary/min_latency_per_image': min(r['encoder_per_image_mean_ms'] for r in results),
    # })

    # # Log the results table to wandb
    # wandb.log({'results_table': wandb.Table(dataframe=df)})

    # # Finish wandb run
    # wandb.finish()

    return results


def run_parameter_sweep(args):
    """Run parameter sweep over percent and percent-global values"""
    import itertools

    print(f"\n{'='*80}")
    print("PARAMETER SWEEP MODE")
    print(f"{'='*80}")
    print(f"Percent values: {args.percent_values}")
    print(f"Percent-global values: {args.percent_global_values}")

    # Generate all combinations
    combinations = list(itertools.product(args.percent_values, args.percent_global_values))
    total_runs = len(combinations)

    print(f"Total runs: {total_runs}")
    print(f"{'='*80}\n")

    all_sweep_results = []
    successful_runs = 0
    failed_runs = 0

    # Run each combination
    for idx, (percent, percent_global) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"SWEEP RUN {idx}/{total_runs}")
        print(f"percent={percent}, percent_global={percent_global}")
        print(f"{'='*80}\n")

        # Update args for this run
        args.percent = percent
        args.percent_global = percent_global

        # Set wandb run name if not specified
        if not args.wandb_run_name:
            args.wandb_run_name = f"p{percent}_pg{percent_global}_{args.processor}"

        try:
            results = run_single_benchmark(args)
            successful_runs += 1

            # Collect sweep results
            for result in results:
                result['sweep_percent'] = percent
                result['sweep_percent_global'] = percent_global
                all_sweep_results.append(result)

        except Exception as e:
            print(f"ERROR: Run failed for percent={percent}, percent_global={percent_global}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed_runs += 1

        # Reset wandb run name for next iteration
        args.wandb_run_name = None

    # Save sweep summary
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"{'='*80}\n")

    if all_sweep_results:
        # Save comprehensive sweep results
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        sweep_csv = args.summary_csv or os.path.join(
            args.output_dir,
            f'sweep_results_{timestamp}.csv'
        )

        df_sweep = pd.DataFrame(all_sweep_results)
        df_sweep.to_csv(sweep_csv, index=False)

        print(f"Sweep results saved to: {sweep_csv}")

        # Print best configurations
        best_throughput = df_sweep.loc[df_sweep['throughput_imgs_per_sec'].idxmax()]
        best_miou = df_sweep.loc[df_sweep['miou'].idxmax()]

        print(f"\nBest throughput configuration:")
        print(f"  percent={best_throughput['sweep_percent']}, "
              f"percent_global={best_throughput['sweep_percent_global']}, "
              f"batch_size={best_throughput['batch_size']}")
        print(f"  Throughput: {best_throughput['throughput_imgs_per_sec']:.2f} imgs/sec")

        print(f"\nBest mIoU configuration:")
        print(f"  percent={best_miou['sweep_percent']}, "
              f"percent_global={best_miou['sweep_percent_global']}, "
              f"batch_size={best_miou['batch_size']}")
        print(f"  mIoU: {best_miou['miou']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SAM encoder with batch inference'
    )

    # Config
    parser.add_argument('--percent', type=float, default=0.0,
                       help='Number of samples per batch size')
    parser.add_argument('--percent-global', type=float, default=0.0,
                       help='percent-global')
    parser.add_argument('--prune-global', action='store_true',
                       help='Number of samples per batch size')
    parser.add_argument('--n-bits', type=int, default=8,
                       help='Number of samples per batch size')
    parser.add_argument('--high-entropy', action='store_true',
                       help='Number of samples per batch size')

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
                       choices=['BASE','POSITIONAL_PRUNE', 'POSITIONAL_QUANT', 'HEAD_PRUNE'],
                       help='Processor to use')
    parser.add_argument('--quantize-encoder', action='store_true',
                       help='Enable encoder quantization')
    parser.add_argument('--quantize-decoder', action='store_true',
                       help='Enable decoder quantization')

    # Quantization parameters
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
    parser.add_argument('--model-ckt', type=str, default='./ckts/sam_hq_vit_l.pth',
                       help='Number of channels to preserve')
    parser.add_argument('--model-type', type=str, default='vit_l',
                       help='Number of channels to preserve')
    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='sam-quantization-batch-benchmark',
                       help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Wandb run name')

    # Sweep parameters
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep over percent and percent-global values')
    parser.add_argument('--percent-values', type=float, nargs='+',
                       default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                       help='Percent values to sweep over')
    parser.add_argument('--percent-global-values', type=float, nargs='+',
                       default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                       help='Percent-global values to sweep over')
    parser.add_argument('--summary-csv', type=str, default=None,
                       help='CSV file to save sweep summary results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle sweep mode
    if args.sweep:
        run_parameter_sweep(args)
    else:
        run_single_benchmark(args)


if __name__ == '__main__':
    main()