"""
Visualization script for SAM2 encoder memory profiling.

This script profiles the SAM2 Hiera encoder and creates comprehensive visualizations
of memory consumption for attention and MLP layers.

Usage:
    python visualize_encoder_memory_sam2.py --model_config sam2_hiera_l.yaml --checkpoint path/to/sam2_hiera_large.pt
    python visualize_encoder_memory_sam2.py --model_config sam2_hiera_l.yaml --checkpoint path/to/sam2_hiera_large.pt --batch_size 4
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import gc

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from processors.encoder.memory_profiler_sam2 import (
    sam2_encoder_memory_monkey_patch,
    analyze_sam2_memory_breakdown,
    MemoryStats,
)


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_sample_images(image_path: str = None, batch_size: int = 1):
    """
    Load sample images for profiling.

    Args:
        image_path: Path to image file or directory
        batch_size: Number of images to load/generate

    Returns:
        List of images (each as numpy array)
    """
    images = []

    if image_path:
        image_path = Path(image_path)

        # If it's a directory, load multiple images
        if image_path.is_dir():
            image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
            print(f"Found {len(image_files)} images in directory")

            for i in range(min(batch_size, len(image_files))):
                img = np.array(Image.open(image_files[i]).convert("RGB"))
                images.append(img)

            # If we need more images, duplicate
            while len(images) < batch_size:
                images.append(images[0].copy())

        # If it's a single file, load and duplicate
        elif image_path.is_file():
            img = np.array(Image.open(image_path).convert("RGB"))
            for _ in range(batch_size):
                images.append(img.copy())
        else:
            print(f"Warning: {image_path} not found, using synthetic images")
            images = None

    # Generate synthetic images if needed
    if images is None or len(images) == 0:
        print(f"Using {batch_size} synthetic 1024x1024 images")
        for _ in range(batch_size):
            img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            images.append(img)

    print(f"Loaded {len(images)} images for batch processing")
    return images


def profile_and_collect_memory_data(
    model_config: str = "sam2_hiera_l.yaml",
    checkpoint: str = None,
    image_path: str = None,
    batch_size: int = 1,
    num_runs: int = 5,
    device: str = "cuda",
):
    """
    Profile the SAM2 encoder and collect memory data.

    Args:
        model_config: SAM2 model configuration file
        checkpoint: Path to checkpoint
        image_path: Path to test image(s)
        batch_size: Number of images to process in batch
        num_runs: Number of profiling runs
        device: Device to use

    Returns:
        Tuple of (memory_stats, results_dict, runtime_stats)
    """
    print(f"Loading SAM2 model: {model_config}")
    sam2_model = build_sam2(model_config, checkpoint, device=device)
    sam2_model.eval()
    predictor = SAM2ImagePredictor(sam2_model)

    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    baseline_memory = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0

    # Apply profiling
    print("Applying memory profiling...")
    memory_stats = MemoryStats(device=device)
    sam2_encoder_memory_monkey_patch(predictor.model, memory_stats, device=device)

    # Load images
    images = load_sample_images(image_path, batch_size)

    # Warmup
    print(f"Warming up with batch size {batch_size}...")
    for _ in range(2):
        with torch.no_grad():
            for img in images:
                predictor.set_image(img)
        if device == "cuda":
            torch.cuda.empty_cache()

    # Profile
    print(f"Profiling for {num_runs} iterations (batch_size={batch_size})...")
    memory_stats.clear()

    peak_memories = []
    allocated_memories = []

    for run_idx in range(num_runs):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            for img in images:
                predictor.set_image(img)

        if device == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            allocated_mem = torch.cuda.memory_allocated() / 1024**2
            peak_memories.append(peak_mem)
            allocated_memories.append(allocated_mem)

        if (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  Completed {run_idx + 1}/{num_runs} runs")

    # Analyze
    results = analyze_sam2_memory_breakdown(memory_stats, print_details=False)

    # Add batch size info
    results['batch_size'] = batch_size
    results['images_per_run'] = batch_size
    results['total_images_processed'] = num_runs * batch_size

    runtime_stats = {
        'peak_memories': peak_memories,
        'allocated_memories': allocated_memories,
        'baseline_memory': baseline_memory,
        'batch_size': batch_size,
    }

    return memory_stats, results, runtime_stats


def plot_parameter_memory_breakdown(memory_stats: MemoryStats, results: dict, output_dir: Path):
    """Create stacked bar chart of parameter memory per block."""
    stats_summary = memory_stats.get_stats_summary()

    # Extract parameter memory per block
    attention_params = []
    mlp_params = []
    block_indices = []

    for name, mem in stats_summary['parameter_memory'].items():
        if 'attention_params' in name and 'block_' in name:
            block_idx = int(name.split('_')[1])
            block_indices.append(block_idx)
            attention_params.append(mem)

    for name, mem in stats_summary['parameter_memory'].items():
        if 'mlp_params' in name and 'block_' in name:
            mlp_params.append(mem)

    # Sort by block index
    sorted_indices = np.argsort(block_indices)
    block_indices = [block_indices[i] for i in sorted_indices]
    attention_params = [attention_params[i] for i in sorted_indices]
    mlp_params = [mlp_params[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(block_indices))
    width = 0.8

    # Create stacked bars
    p1 = ax.bar(x, attention_params, width, label='Attention', color='#3498db')
    p2 = ax.bar(x, mlp_params, width, bottom=attention_params, label='MLP', color='#e74c3c')

    ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter Memory (MB)', fontsize=12, fontweight='bold')
    title = 'SAM2 Parameter Memory per Block: Attention vs MLP'
    if results.get('batch_size', 1) > 1:
        title += f' (Batch Size: {results["batch_size"]})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in block_indices])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_memory_per_block_sam2.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'parameter_memory_per_block_sam2.png'}")
    plt.close()


def plot_memory_pie_chart(results: dict, output_dir: Path):
    """Create pie chart of parameter memory breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart 1: Overall component breakdown
    components = ['Attention', 'MLP', 'Other']
    attn_mem = results['attention_parameter_memory']
    mlp_mem = results['mlp_parameter_memory']
    other_mem = results['total_parameter_memory'] - attn_mem - mlp_mem

    sizes = [attn_mem, mlp_mem, other_mem]
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0)

    # Check if we have valid data
    if sum(sizes) > 0:
        ax1.pie(sizes, labels=components, autopct='%1.1f%%',
                colors=colors, explode=explode, startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        title1 = 'SAM2 Parameter Memory Breakdown'
        if results.get('batch_size', 1) > 1:
            title1 += f'\n(Batch Size: {results["batch_size"]})'
        ax1.set_title(title1, fontsize=14, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('SAM2 Parameter Memory Breakdown', fontsize=14, fontweight='bold')

    # Pie chart 2: Just Hiera blocks
    block_components = ['Attention', 'MLP']
    block_sizes = [attn_mem, mlp_mem]
    block_colors = ['#3498db', '#e74c3c']

    # Check if we have valid data
    if sum(block_sizes) > 0 and all(s >= 0 for s in block_sizes):
        ax2.pie(block_sizes, labels=block_components, autopct='%1.1f%%',
                colors=block_colors, explode=(0.05, 0.05), startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Hiera Blocks Memory\n(Attention vs MLP)', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Hiera Blocks Memory\n(Attention vs MLP)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'memory_pie_chart_sam2.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'memory_pie_chart_sam2.png'}")
    plt.close()


def plot_activation_memory(memory_stats: MemoryStats, output_dir: Path, batch_size: int = 1):
    """Plot activation memory consumption during forward pass."""
    stats_summary = memory_stats.get_stats_summary()

    attention_act = []
    mlp_act = []
    block_indices = []

    for name, mem_dict in stats_summary['activation_memory'].items():
        if 'block_' in name and 'attention' in name:
            block_idx = int(name.split('_')[1])
            block_indices.append(block_idx)
            attention_act.append(mem_dict['mean'])

        if 'block_' in name and 'mlp' in name:
            block_idx = int(name.split('_')[1])
            mlp_act.append(mem_dict['mean'])

    # Sort by block index
    if block_indices:
        sorted_indices = np.argsort(block_indices)
        block_indices = [block_indices[i] for i in sorted_indices]
        attention_act = [attention_act[i] for i in sorted_indices]
        mlp_act = [mlp_act[i] for i in sorted_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(block_indices))
        width = 0.35

        bars1 = ax.bar(x - width/2, attention_act, width, label='Attention',
                        color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, mlp_act, width, label='MLP',
                        color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Activation Memory (MB)', fontsize=12, fontweight='bold')
        title = 'SAM2 Activation Memory: Attention vs MLP'
        if batch_size > 1:
            title += f' (Batch Size: {batch_size})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i}' for i in block_indices])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'activation_memory_sam2.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'activation_memory_sam2.png'}")
        plt.close()


def plot_peak_memory_trend(runtime_stats: dict, output_dir: Path):
    """Plot peak memory usage across iterations."""
    peak_memories = runtime_stats['peak_memories']
    batch_size = runtime_stats.get('batch_size', 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(1, len(peak_memories) + 1)
    ax.plot(iterations, peak_memories, marker='o', linewidth=2.5,
            markersize=8, color='#e74c3c', label='Peak Memory')

    # Add mean line
    mean_mem = np.mean(peak_memories)
    ax.axhline(y=mean_mem, color='#3498db', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Mean: {mean_mem:.2f} MB')

    # Add std band
    std_mem = np.std(peak_memories)
    ax.fill_between(iterations, mean_mem - std_mem, mean_mem + std_mem,
                     alpha=0.2, color='#3498db', label=f'±1 Std: {std_mem:.2f} MB')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    title = 'SAM2 Peak Memory Usage Across Iterations'
    if batch_size > 1:
        title += f' (Batch Size: {batch_size})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'peak_memory_trend_sam2.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'peak_memory_trend_sam2.png'}")
    plt.close()


def plot_memory_comparison_bars(results: dict, runtime_stats: dict, output_dir: Path):
    """Create comparison bar chart of different memory types."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Parameters\n(Total)', 'Parameters\n(Attention)', 'Parameters\n(MLP)',
                  'Peak Memory\n(Runtime)', 'Allocated\n(After Forward)']

    values = [
        results['total_parameter_memory'],
        results['attention_parameter_memory'],
        results['mlp_parameter_memory'],
        np.mean(runtime_stats['peak_memories']),
        np.mean(runtime_stats['allocated_memories']),
    ]

    colors = ['#3498db', '#1abc9c', '#e74c3c', '#f39c12', '#9b59b6']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    title = 'SAM2 Memory Breakdown: Parameters vs Runtime'
    if results.get('batch_size', 1) > 1:
        title += f' (Batch Size: {results["batch_size"]})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'memory_comparison_sam2.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'memory_comparison_sam2.png'}")
    plt.close()


def plot_cumulative_parameter_memory(memory_stats: MemoryStats, output_dir: Path):
    """Create cumulative parameter memory plot."""
    stats_summary = memory_stats.get_stats_summary()

    block_params = []
    block_indices = []

    for name, mem in stats_summary['parameter_memory'].items():
        if 'block_' in name and '_params' in name and 'attention' not in name and 'mlp' not in name:
            block_idx = int(name.split('_')[1])
            block_indices.append(block_idx)
            block_params.append(mem)

    # Sort by block index
    sorted_indices = np.argsort(block_indices)
    block_indices = [block_indices[i] for i in sorted_indices]
    block_params = [block_params[i] for i in sorted_indices]

    # Calculate cumulative
    cumulative_params = np.cumsum(block_params)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(block_indices, cumulative_params, linewidth=2.5,
            color='#2c3e50', marker='o', markersize=6, label='Cumulative Parameters')
    ax.fill_between(block_indices, cumulative_params, alpha=0.3, color='#3498db')

    ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Parameter Memory (MB)', fontsize=12, fontweight='bold')
    ax.set_title('SAM2 Cumulative Parameter Memory Through Blocks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add total annotation
    total_mem = cumulative_params[-1]
    ax.axhline(y=total_mem, color='red', linestyle='--', alpha=0.5)
    ax.text(0, total_mem, f' Total: {total_mem:.1f} MB',
            verticalalignment='bottom', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_parameter_memory_sam2.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cumulative_parameter_memory_sam2.png'}")
    plt.close()


def plot_batch_memory_efficiency(runtime_stats: dict, results: dict, output_dir: Path):
    """Create memory efficiency visualization for batch processing."""
    batch_size = runtime_stats.get('batch_size', 1)
    if batch_size == 1:
        return  # Skip if not batching

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Memory per image
    peak_mem_total = np.mean(runtime_stats['peak_memories'])
    allocated_mem_total = np.mean(runtime_stats['allocated_memories'])
    param_mem = results['total_parameter_memory']

    peak_mem_per_image = peak_mem_total / batch_size
    allocated_mem_per_image = allocated_mem_total / batch_size

    metrics = ['Peak Memory\n(Total)', 'Peak Memory\n(Per Image)',
               'Allocated\n(Total)', 'Allocated\n(Per Image)']
    values = [peak_mem_total, peak_mem_per_image, allocated_mem_total, allocated_mem_per_image]
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Memory Usage (Batch Size: {batch_size})', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Memory efficiency
    single_image_mem = peak_mem_per_image  # Approximate
    batch_overhead = peak_mem_total - (param_mem + single_image_mem * batch_size)
    memory_efficiency = (single_image_mem * batch_size) / peak_mem_total * 100 if peak_mem_total > 0 else 0

    components = ['Parameters', f'Activations\n({batch_size} images)', 'Overhead']
    mem_values = [param_mem, single_image_mem * batch_size, max(0, batch_overhead)]
    colors2 = ['#3498db', '#2ecc71', '#e74c3c']

    bars2 = ax2.bar(components, mem_values, color=colors2, alpha=0.8)
    ax2.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Breakdown by Component', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars2, mem_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add efficiency annotation
    ax2.text(0.5, 0.95, f'Memory Efficiency: {memory_efficiency:.1f}%',
            transform=ax2.transAxes, fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / 'batch_memory_efficiency_sam2.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'batch_memory_efficiency_sam2.png'}")
    plt.close()


def create_all_visualizations(
    model_config: str = "sam2_hiera_l.yaml",
    checkpoint: str = None,
    image_path: str = None,
    output_dir: str = "./memory_plots_sam2",
    batch_size: int = 1,
    num_runs: int = 5,
    device: str = "cuda",
):
    """
    Run memory profiling and create all visualizations for SAM2.

    Args:
        model_config: SAM2 model configuration file
        checkpoint: Path to checkpoint
        image_path: Path to test image(s) or directory
        output_dir: Directory to save plots
        batch_size: Number of images to process in batch
        num_runs: Number of profiling runs
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print(f"PROFILING AND VISUALIZING SAM2 ENCODER MEMORY - {model_config}")
    if batch_size > 1:
        print(f"BATCH INFERENCE MODE - Batch Size: {batch_size}")
    print("="*80)

    # Profile the encoder
    memory_stats, results, runtime_stats = profile_and_collect_memory_data(
        model_config=model_config,
        checkpoint=checkpoint,
        image_path=image_path,
        batch_size=batch_size,
        num_runs=num_runs,
        device=device,
    )

    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Create all plots
    plot_parameter_memory_breakdown(memory_stats, results, output_path)
    plot_memory_pie_chart(results, output_path)
    plot_activation_memory(memory_stats, output_path, batch_size)
    plot_peak_memory_trend(runtime_stats, output_path)
    plot_memory_comparison_bars(results, runtime_stats, output_path)
    plot_cumulative_parameter_memory(memory_stats, output_path)

    # Add batch efficiency plot if batch_size > 1
    if batch_size > 1:
        plot_batch_memory_efficiency(runtime_stats, results, output_path)

    print("\n" + "="*80)
    print(f"ALL VISUALIZATIONS SAVED TO: {output_path}")
    print("="*80)

    # Print summary
    analyze_sam2_memory_breakdown(memory_stats, print_details=True)

    # Print runtime stats
    print(f"\nRuntime Memory Statistics:")
    print(f"  Peak memory (mean):      {np.mean(runtime_stats['peak_memories']):.2f} MB")
    print(f"  Allocated (mean):        {np.mean(runtime_stats['allocated_memories']):.2f} MB")
    print(f"  Baseline (model loaded): {runtime_stats['baseline_memory']:.2f} MB")

    # Print batch-specific metrics
    if batch_size > 1:
        print("\n" + "="*80)
        print("BATCH PROCESSING MEMORY METRICS")
        print("="*80)
        peak_mem = np.mean(runtime_stats['peak_memories'])
        allocated_mem = np.mean(runtime_stats['allocated_memories'])
        print(f"Batch Size: {batch_size}")
        print(f"Total Images Processed: {results.get('total_images_processed', 0)}")
        print(f"Peak Memory per Batch: {peak_mem:.2f} MB")
        print(f"Peak Memory per Image: {peak_mem/batch_size:.2f} MB")
        print(f"Allocated Memory per Batch: {allocated_mem:.2f} MB")
        print(f"Allocated Memory per Image: {allocated_mem/batch_size:.2f} MB")

    print(f"\nGenerated {len(list(output_path.glob('*.png')))} plots:")
    for plot in sorted(output_path.glob('*.png')):
        print(f"  - {plot.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAM2 encoder memory profiling results"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="sam2_hiera_l.yaml",
        help="SAM2 model configuration file",
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
        help="Path to test image or directory of images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./memory_plots_sam2",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to process in batch (default: 1)",
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

    args = parser.parse_args()

    create_all_visualizations(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        image_path=args.image,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_runs=args.runs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
