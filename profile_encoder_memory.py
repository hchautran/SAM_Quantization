"""
Example script to profile SAM encoder memory footprint.

This script demonstrates how to use the memory profiler to measure the memory
consumption of attention and MLP layers in the SAM Vision Transformer encoder.

Usage:
    python profile_encoder_memory.py --model_type vit_b --checkpoint path/to/checkpoint.pth
"""
import argparse
import numpy as np
import torch
from PIL import Image
import gc

from segment_anything import sam_model_registry, SamPredictor
from processors.encoder.memory_profiler import (
    encoder_memory_monkey_patch,
    analyze_memory_breakdown,
    MemoryStats,
)


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


def profile_encoder_memory(
    model_type: str = "vit_b",
    checkpoint: str = None,
    image_path: str = None,
    num_runs: int = 5,
    device: str = "cuda",
):
    """
    Profile SAM encoder memory with detailed breakdown.

    Args:
        model_type: SAM model variant (vit_b, vit_l, vit_h)
        checkpoint: Path to model checkpoint
        image_path: Path to test image
        num_runs: Number of profiling iterations
        device: Device to run on (cuda or cpu)

    Returns:
        MemoryStats instance with collected data
    """
    print("="*80)
    print(f"PROFILING SAM ENCODER MEMORY - {model_type.upper()}")
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

    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    # Get baseline memory
    baseline_memory = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
    print(f"Baseline memory (model loaded): {baseline_memory:.2f} MB")

    # Apply memory profiling monkey-patch
    print("Applying memory profiling to encoder...")
    memory_stats = MemoryStats(device=device)
    encoder_memory_monkey_patch(predictor.model, memory_stats, device=device)

    # Load test image
    image = load_sample_image(image_path)
    print(f"Test image shape: {image.shape}")

    # Profile model parameters
    print("\nComputing parameter memory...")
    total_params = 0
    for name, param in sam.named_parameters():
        if 'image_encoder' in name:
            total_params += param.numel()

    param_memory_mb = total_params * 4 / 1024**2  # Assuming float32
    print(f"Total encoder parameters: {total_params:,} ({param_memory_mb:.2f} MB)")

    # Warmup
    print(f"\nWarming up...")
    for _ in range(2):
        with torch.no_grad():
            predictor.set_image(image)
        if device == "cuda":
            torch.cuda.empty_cache()

    # Profile
    print(f"Profiling for {num_runs} iterations...")
    memory_stats.clear()

    peak_memories = []
    allocated_memories = []

    for i in range(num_runs):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            predictor.set_image(image)

        if device == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            allocated_mem = torch.cuda.memory_allocated() / 1024**2
            peak_memories.append(peak_mem)
            allocated_memories.append(allocated_mem)

        if (i + 1) % 2 == 0:
            print(f"  Completed {i+1}/{num_runs} iterations")

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)

    # Print memory statistics
    if device == "cuda":
        print(f"\nPeak memory usage:")
        print(f"  - Mean: {np.mean(peak_memories):.2f} MB")
        print(f"  - Max:  {np.max(peak_memories):.2f} MB")
        print(f"  - Min:  {np.min(peak_memories):.2f} MB")
        print(f"  - Std:  {np.std(peak_memories):.2f} MB")

        print(f"\nAllocated memory after forward pass:")
        print(f"  - Mean: {np.mean(allocated_memories):.2f} MB")
        print(f"  - Max:  {np.max(allocated_memories):.2f} MB")
        print(f"  - Min:  {np.min(allocated_memories):.2f} MB")

    # Analyze results
    results = analyze_memory_breakdown(memory_stats, print_details=True)

    return memory_stats, results, {
        'peak_memories': peak_memories,
        'allocated_memories': allocated_memories,
        'baseline_memory': baseline_memory,
    }


def compare_model_variants():
    """
    Compare memory footprint across different SAM model variants.

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
                memory_stats, result, runtime_stats = profile_encoder_memory(
                    model_type=variant,
                    checkpoint=checkpoint,
                    num_runs=5,
                )
                results[variant] = {
                    'memory_stats': memory_stats,
                    'result': result,
                    'runtime_stats': runtime_stats,
                }
            except FileNotFoundError:
                print(f"Checkpoint not found for {variant}, skipping...")

    # Print comparison
    if len(results) > 1:
        print("\n\n" + "="*80)
        print("MODEL VARIANT COMPARISON")
        print("="*80)
        print(f"{'Variant':<12} {'Params (MB)':<15} {'Peak Mem (MB)':<18} {'Attn/MLP Ratio':<20}")
        print("-"*80)

        for variant, data in results.items():
            result = data['result']
            runtime_stats = data['runtime_stats']

            total_params = result['total_parameter_memory']
            peak_mem = np.mean(runtime_stats['peak_memories'])
            attn_params = result['attention_parameter_memory']
            mlp_params = result['mlp_parameter_memory']
            ratio = attn_params / mlp_params if mlp_params > 0 else 0

            print(f"{variant:<12} {total_params:<15.2f} {peak_mem:<18.2f} {ratio:<20.2f}")

        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Profile SAM encoder memory footprint"
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
        "--runs",
        type=int,
        default=5,
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
        profile_encoder_memory(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            image_path=args.image,
            num_runs=args.runs,
            device=args.device,
        )


if __name__ == "__main__":
    main()
