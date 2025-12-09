#!/usr/bin/env python3
"""
Python-based sweep script for testing different percent and percent-global combinations.
Provides more flexibility and better error handling than shell scripts.

Usage:
    python run_sweep.py
    python run_sweep.py --quick  # Run a smaller sweep
    python run_sweep.py --custom  # Customize parameters interactively
"""

import argparse
import subprocess
import sys
import itertools
from typing import List, Tuple
import os


def run_benchmark(
    percent: float,
    percent_global: float,
    batch_sizes: List[int],
    num_samples: int,
    num_calib_samples: int,
    processor: str,
    n_bits: int,
    en_weight_quant: str,
    en_act_quant: str,
    model_ckt: str,
    model_type: str,
    wandb_project: str,
    output_dir: str,
    prune_global: bool = False,
    high_entropy: bool = False,
) -> bool:
    """
    Run a single benchmark with given parameters.

    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        'python', 'benchmark_batch_inference.py',
        '--percent', str(percent),
        '--percent-global', str(percent_global),
        '--batch-sizes', *[str(bs) for bs in batch_sizes],
        '--num-samples', str(num_samples),
        '--num-calib-samples', str(num_calib_samples),
        '--processor', processor,
        '--quantize-encoder',
        '--n-bits', str(n_bits),
        '--en-weight-quant', en_weight_quant,
        '--en-act-quant', en_act_quant,
        '--model-ckt', model_ckt,
        '--model-type', model_type,
        '--wandb-project', wandb_project,
        '--output-dir', output_dir,
    ]

    if prune_global:
        cmd.append('--prune-global')

    if high_entropy:
        cmd.append('--high-entropy')

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Run failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run parameter sweep for SAM quantization benchmark'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run a quick sweep with fewer parameter combinations')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing them')
    parser.add_argument('--wandb-project', type=str,
                       default='sam-quantization-sweep',
                       help='Wandb project name')
    parser.add_argument('--output-dir', type=str,
                       default='sweep_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Define sweep parameters
    if args.quick:
        percent_values = [0.0, 0.2, 0.4]
        percent_global_values = [0.0, 0.2, 0.4]
        batch_sizes = [1, 4, 8]
        num_samples = 50
    else:
        percent_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        percent_global_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        batch_sizes = [1, 2, 4, 8]
        num_samples = 100

    # Fixed parameters
    num_calib_samples = 16
    processor = 'POSITIONAL_PRUNE'
    n_bits = 4
    en_weight_quant = 'per_channel'
    en_act_quant = 'per_token'
    model_ckt = './ckts/sam_hq_vit_l.pth'
    model_type = 'vit_l'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate all combinations
    combinations = list(itertools.product(percent_values, percent_global_values))
    total_runs = len(combinations)

    print("=" * 80)
    print("SAM Quantization Parameter Sweep")
    print("=" * 80)
    print(f"Percent values: {percent_values}")
    print(f"Percent-global values: {percent_global_values}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total runs: {total_runs}")
    print(f"Wandb project: {args.wandb_project}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    if args.dry_run:
        print("DRY RUN MODE - Commands will not be executed")
        print()

    # Track results
    successful_runs = 0
    failed_runs = 0
    failed_combinations = []

    # Run all combinations
    for idx, (percent, percent_global) in enumerate(combinations, 1):
        print("=" * 80)
        print(f"Run {idx}/{total_runs}")
        print(f"percent={percent}, percent_global={percent_global}")
        print("=" * 80)

        if args.dry_run:
            print(f"Would run: percent={percent}, percent_global={percent_global}")
            print()
            continue

        success = run_benchmark(
            percent=percent,
            percent_global=percent_global,
            batch_sizes=batch_sizes,
            num_samples=num_samples,
            num_calib_samples=num_calib_samples,
            processor=processor,
            n_bits=n_bits,
            en_weight_quant=en_weight_quant,
            en_act_quant=en_act_quant,
            model_ckt=model_ckt,
            model_type=model_type,
            wandb_project=args.wandb_project,
            output_dir=args.output_dir,
        )

        if success:
            successful_runs += 1
            print(f"✓ Run {idx}/{total_runs} completed successfully")
        else:
            failed_runs += 1
            failed_combinations.append((percent, percent_global))
            print(f"✗ Run {idx}/{total_runs} failed")

        print()

    # Print summary
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")

    if failed_combinations:
        print("\nFailed combinations:")
        for percent, percent_global in failed_combinations:
            print(f"  - percent={percent}, percent_global={percent_global}")

    print(f"\nResults saved in: {args.output_dir}/")
    print(f"Wandb project: {args.wandb_project}")
    print("=" * 80)

    # Exit with error code if any runs failed
    sys.exit(0 if failed_runs == 0 else 1)


if __name__ == '__main__':
    main()
