#!/usr/bin/env python3
"""
Benchmark script to compare PositionalPruneProcessor, PositionalQuantProcessor,
and HeadPruneProcessor across different percent_entropy values.

Measures:
- mIoU (mean Intersection over Union)
- Boundary IoU
- Theoretical GFLOPs using fvcore

Usage:
    python benchmark_processors.py \
        --config-file quant/config/hq44k/rtn.yaml \
        --num-samples 400 \
        --num-calib-samples 32 \
        --n-bits 4
"""

import os
import csv
import gc
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from copy import deepcopy

# SAM imports
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.image_encoder import Attention as EncoderSamAttention
from segment_anything.modeling.transformer import Attention as DecoderAttention

# Local imports
from small_engine import Engine, override_args
from processors import get_encoder_processor, DecoderDoNothingProcessor

# GFLOPs measurement
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    FVCORE_AVAILABLE = True
except ImportError:
    print("Warning: fvcore not available. GFLOPs measurement will be skipped.")
    print("Install with: pip install fvcore")
    FVCORE_AVAILABLE = False


class GFLOPSMeasurer:
    """Measures theoretical GFLOPs using fvcore FlopCountAnalysis"""

    def __init__(self):
        self.sample_inputs = {}
        self.hooks = []

    def capture_sample_input(self, predictor, dataloader):
        """Capture a sample input tensor for each module"""
        print("Capturing sample inputs for GFLOPs measurement...")

        def hook_fn(module, input, output, name):
            """Hook to capture module inputs"""
            if name not in self.sample_inputs:
                # Store the first input we see
                if isinstance(input, tuple):
                    self.sample_inputs[name] = input[0].clone().detach()
                else:
                    self.sample_inputs[name] = input.clone().detach()

        # Register hooks on encoder attention modules
        from segment_anything.modeling.image_encoder import Attention, Block
        for name, module in predictor.model.image_encoder.named_modules():
            if isinstance(module, Attention):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)

        # Run one forward pass to capture inputs
        for data_val in dataloader:
            imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
            predictor.set_image(imgs.squeeze())
            break

        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        print(f"Captured {len(self.sample_inputs)} sample inputs")

    def measure_encoder_gflops(self, predictor):
        """Measure encoder GFLOPs using fvcore"""
        if not FVCORE_AVAILABLE:
            return 0.0

        try:
            # Get a sample input for the encoder
            # SAM encoder expects (B, C, H, W) = (1, 3, 1024, 1024)
            sample_input = torch.randn(1, 3, 1024, 1024).to(predictor.device)

            # Measure encoder FLOPs
            encoder = predictor.model.image_encoder
            encoder.eval()

            with torch.no_grad():
                flops = FlopCountAnalysis(encoder, sample_input)
                total_flops = flops.total()
                gflops = total_flops / 1e9

            return gflops
        except Exception as e:
            print(f"Error measuring encoder GFLOPs: {e}")
            return 0.0

    def measure_total_gflops(self, predictor):
        """Measure total model GFLOPs (encoder + decoder)"""
        if not FVCORE_AVAILABLE:
            return 0.0, 0.0

        encoder_gflops = self.measure_encoder_gflops(predictor)

        try:
            # For decoder, we need to do a full forward pass
            # This is approximate since decoder GFLOPs depend on prompts
            # We'll use a simple box prompt
            sample_image = torch.randn(1, 3, 1024, 1024).to(predictor.device)

            # Set the image first
            with torch.no_grad():
                predictor.model.image_encoder.eval()
                image_embedding = predictor.model.image_encoder(sample_image)

            # Now measure decoder with a box prompt
            box = torch.tensor([[100, 100, 900, 900]], dtype=torch.float32).to(predictor.device)

            # This is tricky - decoder FLOPs are hard to measure in isolation
            # For now, we'll just return encoder GFLOPs
            decoder_gflops = 0.0  # Placeholder
            total_gflops = encoder_gflops + decoder_gflops

            return encoder_gflops, total_gflops
        except Exception as e:
            print(f"Error measuring total GFLOPs: {e}")
            return encoder_gflops, encoder_gflops


def reset_model_and_memory():
    """Reset CUDA memory and garbage collect"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_single_benchmark(
    processor_name: str,
    percent_entropy: float,
    config_path: str,
    args,
    gflops_measurer: GFLOPSMeasurer
):
    """
    Run a single benchmark configuration.

    Args:
        processor_name: One of ["POSITIONAL_PRUNE", "POSITIONAL_QUANT", "HEAD_PRUNE"]
        percent_entropy: Float value for percent_entropy
        config_path: Path to config YAML file
        args: Command-line arguments
        gflops_measurer: GFLOPSMeasurer instance

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {processor_name}, percent_entropy={percent_entropy}")
    print(f"{'='*80}\n")

    # Load and update config
    config = OmegaConf.load(config_path)
    config.quantization.percent_entropy = percent_entropy
    config.quantization.high_entropy = True
    config = override_args(args, config)

    # Initialize model
    model_type = 'vit_l'
    checkpoint_path = './pretrained_checkpoint/sam_hq_vit_l.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Initialize engine
    engine = Engine(
        'hq44k_benchmark',
        quantize_encoder=True,
        quantize_decoder=args.quantize_decoder
    )

    # Get processor
    try:
        enc_processor = get_encoder_processor(processor_name)
    except ValueError as e:
        print(f"Error: {e}")
        return None

    # Setup and calibrate processors
    print(f"Calibrating {processor_name} with percent_entropy={percent_entropy}...")
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
    }

    decoder_config = {
        'processor': decoder_processor,
        'n_bits': args.n_bits,
        'weight_quant': args.de_weight_quant,
        'act_quant': args.de_act_quant,
        'k_preserve': args.k_preserve
    } if args.quantize_decoder else None

    engine.apply_quantization(predictor, encoder_config, decoder_config, config)

    # Measure GFLOPs
    print("Measuring GFLOPs...")
    encoder_gflops, total_gflops = gflops_measurer.measure_total_gflops(predictor)

    # Run evaluation
    print(f"Evaluating on {args.num_samples} samples...")
    test_stats = engine.eval_hq44k(
        predictor=predictor,
        num_samples=args.num_samples,
        plot_figures=False
    )

    # Compile results
    result = {
        'processor': processor_name,
        'percent_entropy': percent_entropy,
        'high_entropy': True,
        'n_bits': args.n_bits,
        'weight_quant': args.en_weight_quant,
        'act_quant': args.en_act_quant,
        'encoder_gflops': encoder_gflops,
        'total_gflops': total_gflops,
        'num_calib_samples': args.num_calib_samples,
        'num_eval_samples': args.num_samples,
        'timestamp': datetime.datetime.now().isoformat(),
        **test_stats
    }

    # Cleanup
    del predictor, sam, engine, encoder_processor, decoder_processor
    reset_model_and_memory()

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SAM processors across different percent_entropy values'
    )

    # Config
    parser.add_argument('--config-file', type=str, required=True,
                        help='Path to config YAML file')

    # Benchmark parameters
    parser.add_argument('--percent-entropy-values', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help='List of percent_entropy values to test')
    parser.add_argument('--processors', type=str, nargs='+',
                        default=['HEAD_PRUNE', 'POSITIONAL_QUANT', 'POSITIONAL_PRUNE'],
                        help='List of processors to benchmark')

    # Evaluation parameters
    parser.add_argument('--num-samples', type=int, default=400,
                        help='Number of evaluation samples')
    parser.add_argument('--num-calib-samples', type=int, default=16,
                        help='Number of calibration samples')

    # Quantization parameters
    parser.add_argument('--n-bits', type=int, default=16,
                        help='Number of quantization bits')
    parser.add_argument('--n-bits-mlp', type=int, default=4,
                        help='Number of quantization bits for MLP')
    parser.add_argument('--en-weight-quant', type=str, default='per_channel',
                        choices=['per_channel', 'selective_channel'],
                        help='Encoder weight quantization method')
    parser.add_argument('--en-act-quant', type=str, default='per_token',
                        choices=['per_token', 'low_high_density_activation'],
                        help='Encoder activation quantization method')
    parser.add_argument('--de-weight-quant', type=str, default='per_channel',
                        choices=['per_channel', 'selective_channel'],
                        help='Decoder weight quantization method')
    parser.add_argument('--de-act-quant', type=str, default='per_token',
                        choices=['per_token', 'low_high_density_activation'],
                        help='Decoder activation quantization method')
    parser.add_argument('--k-preserve', type=int, default=0,
                        help='Number of channels to preserve')

    # Flags
    parser.add_argument('--quantize-decoder', action='store_true',
                        help='Enable decoder quantization')
    parser.add_argument('--smooth', action='store_true', default=False,
                        help='Enable smoothing')
    parser.add_argument('--quarot', action='store_true', default=False,
                        help='Enable quarot')
    parser.add_argument('--centerQ', action='store_true', default=False,
                        help='Enable centerQ')

    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--output-prefix', type=str, default='benchmark',
                        help='Prefix for output files')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize GFLOPs measurer
    gflops_measurer = GFLOPSMeasurer()

    # Create CSV file with headers before starting benchmarks
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(
        args.output_dir,
        f'{args.output_prefix}_results_{timestamp}.csv'
    )

    # Storage for all results (for summary statistics)
    all_results = []

    # Run benchmarks
    total_runs = len(args.processors) * len(args.percent_entropy_values)
    current_run = 0

    print(f"\n{'='*80}")
    print(f"Starting benchmark with {total_runs} total configurations")
    print(f"Processors: {args.processors}")
    print(f"Percent entropy values: {args.percent_entropy_values}")
    print(f"Results will be saved to: {csv_filename}")
    print(f"{'='*80}\n")

    # Track if CSV headers have been written
    csv_headers_written = False

    for processor_name in args.processors:
        for percent_entropy in args.percent_entropy_values:
            current_run += 1
            print(f"\n[Run {current_run}/{total_runs}]")

            try:
                result = run_single_benchmark(
                    processor_name=processor_name,
                    percent_entropy=percent_entropy,
                    config_path=args.config_file,
                    args=args,
                    gflops_measurer=gflops_measurer
                )

                if result is not None:
                    all_results.append(result)

                    # Write to CSV immediately
                    file_mode = 'w' if not csv_headers_written else 'a'
                    write_header = not csv_headers_written

                    df_single = pd.DataFrame([result])
                    df_single.to_csv(
                        csv_filename,
                        mode=file_mode,
                        header=write_header,
                        index=False
                    )
                    csv_headers_written = True

                    print(f"✓ Success: mIoU={result.get('val_iou_0', 0):.4f}, "
                          f"GFLOPs={result.get('encoder_gflops', 0):.2f}")
                    print(f"  → Result written to {csv_filename}")
                else:
                    print(f"✗ Failed: skipping this configuration")

            except Exception as e:
                print(f"✗ Error in benchmark: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Print final summary
    print(f"\n{'='*80}")
    if all_results:
        print(f"✓ Benchmark complete!")
        print(f"Results saved to: {csv_filename}")
        print(f"Total successful runs: {len(all_results)}/{total_runs}")
        print(f"{'='*80}\n")

        # Print summary statistics
        df = pd.DataFrame(all_results)
        print("\nSummary Statistics:")
        print("-" * 80)
        for processor in args.processors:
            proc_results = df[df['processor'] == processor]
            if len(proc_results) > 0:
                print(f"\n{processor}:")
                print(f"  Mean mIoU: {proc_results['val_iou_0'].mean():.4f} "
                      f"(std: {proc_results['val_iou_0'].std():.4f})")
                print(f"  Mean GFLOPs: {proc_results['encoder_gflops'].mean():.2f} "
                      f"(std: {proc_results['encoder_gflops'].std():.2f})")
                print(f"  Best mIoU: {proc_results['val_iou_0'].max():.4f} "
                      f"at percent={proc_results.loc[proc_results['val_iou_0'].idxmax(), 'percent_entropy']:.1f}")
    else:
        print("\n✗ No successful benchmark runs. Check errors above.")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
