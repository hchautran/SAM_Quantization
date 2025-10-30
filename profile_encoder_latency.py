"""
Example script to profile SAM encoder attention and MLP layer latencies.

This script demonstrates how to use the encoder profiler to measure the latency
of individual components in the SAM Vision Transformer encoder.

Usage:
    python profile_encoder_latency.py --model_type vit_b --checkpoint path/to/checkpoint.pth
"""
import argparse
import numpy as np
import torch
from PIL import Image

from segment_anything import sam_model_registry, SamPredictor
from processors.encoder.profiler import (
    encoder_latency_monkey_patch,
    analyze_encoder_breakdown,
)
from profiler import ProfilerStats


def load_sample_image(image_path: str = None):
    """
    Load a sample image for profiling.

    Args:
        image_path: Path to image file, or None for synthetic image

    Returns:
        numpy array of shape (H, W, 3)
    """
    if image_path:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        # Create synthetic image for testing
        print("No image provided, using synthetic 1024x1024 image")
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    return image


def profile_encoder(
    model_type: str = "vit_b",
    checkpoint: str = None,
    image_path: str = None,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cuda",
):
    """
    Profile SAM encoder latency with detailed breakdown.

    Args:
        model_type: SAM model variant (vit_b, vit_l, vit_h)
        checkpoint: Path to model checkpoint
        image_path: Path to test image
        num_warmup: Number of warmup iterations
        num_runs: Number of profiling iterations
        device: Device to run on (cuda or cpu)

    Returns:
        ProfilerStats instance with collected timings
    """
    print("="*80)
    print(f"PROFILING SAM ENCODER - {model_type.upper()}")
    print("="*80)

    # Load model
    print(f"\nLoading model: {model_type}")
    if checkpoint:
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
    else:
        print("Warning: No checkpoint provided, using random initialization")
        sam = sam_model_registry[model_type]()

    sam = sam.to(device)
    sam.eval()

    predictor = SamPredictor(sam)

    # Apply latency profiling monkey-patch
    print("Applying latency profiling to encoder...")
    profiler = ProfilerStats()
    encoder_latency_monkey_patch(predictor.model, profiler, sync_cuda=(device == "cuda"))

    # Load test image
    image = load_sample_image(image_path)
    print(f"Test image shape: {image.shape}")

    # Warmup
    print(f"\nWarming up for {num_warmup} iterations...")
    for _ in range(num_warmup):
        with torch.no_grad():
            predictor.set_image(image)

    # Profile
    print(f"Profiling for {num_runs} iterations...")
    profiler.clear()

    for i in range(num_runs):
        with torch.no_grad():
            predictor.set_image(image)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{num_runs} iterations")

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)

    # Analyze results
    results = analyze_encoder_breakdown(profiler, print_details=True)

    # Print detailed profiling table
    print("\nDETAILED PROFILING TABLE")
    profiler.print_summary(sort_by='mean')

    return profiler, results


def compare_model_variants():
    """
    Compare latency across different SAM model variants.

    Note: This requires checkpoints for all model variants.
    """
    variants = ["vit_b", "vit_l", "vit_h"]
    checkpoints = {
        "vit_b": "./pretrained_checkpoint/sam_vit_b.pth",
        "vit_l": "./pretrained_checkpoint/sam_vit_l.pth",
        "vit_h": "./pretrained_checkpoint/sam_vit_h.pth",
    }

    results = {}

    for variant in variants:
        checkpoint = checkpoints.get(variant)
        if checkpoint:
            print(f"\n\n{'='*80}")
            print(f"PROFILING {variant.upper()}")
            print('='*80)

            try:
                profiler, result = profile_encoder(
                    model_type=variant,
                    checkpoint=checkpoint,
                    num_runs=10,
                )
                results[variant] = result
            except FileNotFoundError:
                print(f"Checkpoint not found for {variant}, skipping...")

    # Print comparison
    if len(results) > 1:
        print("\n\n" + "="*80)
        print("MODEL VARIANT COMPARISON")
        print("="*80)
        print(f"{'Variant':<12} {'Total (ms)':<15} {'Attention (ms)':<18} {'MLP (ms)':<15} {'Attn %':<10}")
        print("-"*80)

        for variant, result in results.items():
            total_time = result['total_encoder_time'] * 1000
            attn_time = result['total_attention_time'] * 1000
            mlp_time = result['total_mlp_time'] * 1000
            attn_pct = result['attention_percentage']

            print(f"{variant:<12} {total_time:<15.2f} {attn_time:<18.2f} {mlp_time:<15.2f} {attn_pct:<10.1f}")

        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Profile SAM encoder attention and MLP latencies"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model variant",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image (uses synthetic image if not provided)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of profiling iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all model variants (requires all checkpoints)",
    )

    args = parser.parse_args()

    if args.compare:
        compare_model_variants()
    else:
        profile_encoder(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            image_path=args.image,
            num_warmup=args.warmup,
            num_runs=args.runs,
            device=args.device,
        )


if __name__ == "__main__":
    main()
