"""
Visualization script for SAM encoder latency profiling with batch inference support.

This script profiles the SAM encoder and creates comprehensive visualizations
of attention and MLP layer latencies, with support for batch inference.

Usage:
    python visualize_encoder_latency.py --model_type vit_l --checkpoint path/to/checkpoint.pth
    python visualize_encoder_latency.py --model_type vit_l --checkpoint path/to/checkpoint.pth --batch_size 4
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

from segment_anything import sam_model_registry, SamPredictor
from processors.encoder.profiler import (
    encoder_latency_monkey_patch,
    analyze_encoder_breakdown,
)
from profiler import ProfilerStats


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


def profile_and_collect_data(
    model_type: str = "vit_b",
    checkpoint: str = None,
    image_path: str = None,
    batch_size: int = 1,
    num_runs: int = 10,
    device: str = "cuda",
):
    """
    Profile the encoder and collect timing data with batch inference support.

    Args:
        model_type: SAM model variant
        checkpoint: Path to checkpoint
        image_path: Path to test image(s)
        batch_size: Number of images to process in batch
        num_runs: Number of profiling runs
        device: Device to use

    Returns:
        Tuple of (profiler_stats, results_dict)
    """
    print(f"Loading model: {model_type}")
    if checkpoint:
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
    else:
        print("Warning: No checkpoint provided, using random initialization")
        sam = sam_model_registry[model_type]()

    sam = sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    # Apply profiling
    print("Applying latency profiling...")
    profiler = ProfilerStats()
    encoder_latency_monkey_patch(predictor.model, profiler, sync_cuda=(device == "cuda"))

    # Load images
    images = load_sample_images(image_path, batch_size)

    # Warmup
    print(f"Warming up with batch size {batch_size}...")
    for _ in range(3):
        with torch.no_grad():
            for img in images:
                predictor.set_image(img)

    # Profile
    print(f"Profiling for {num_runs} iterations (batch_size={batch_size})...")
    profiler.clear()

    for run_idx in range(num_runs):
        with torch.no_grad():
            for img in images:
                predictor.set_image(img)

        if (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  Completed {run_idx + 1}/{num_runs} runs")

    # Analyze
    results = analyze_encoder_breakdown(profiler, print_details=False)

    # Add batch size info to results
    results['batch_size'] = batch_size
    results['images_per_run'] = batch_size
    results['total_images_processed'] = num_runs * batch_size

    return profiler, results


def plot_per_block_breakdown(profiler: ProfilerStats, output_dir: Path, batch_size: int = 1):
    """Create stacked bar chart of per-block attention vs MLP time."""
    df = profiler.get_all_stats()

    # Extract attention and MLP times per block
    attention_data = []
    mlp_data = []
    block_indices = []

    for name in df['name']:
        if 'attention' in name and 'block_' in name and 'total' not in name:
            block_idx = int(name.split('_')[1])
            block_indices.append(block_idx)

            attn_time = df[df['name'] == name]['mean'].values[0] * 1000  # ms
            mlp_time = df[df['name'] == f'block_{block_idx:02d}_mlp']['mean'].values[0] * 1000

            attention_data.append(attn_time)
            mlp_data.append(mlp_time)

    # Sort by block index
    sorted_indices = np.argsort(block_indices)
    block_indices = [block_indices[i] for i in sorted_indices]
    attention_data = [attention_data[i] for i in sorted_indices]
    mlp_data = [mlp_data[i] for i in sorted_indices]

    # Identify windowed attention blocks (they have much higher attention time)
    windowed_blocks = []
    median_attn = np.median(attention_data)
    for i, attn_time in enumerate(attention_data):
        if attn_time > median_attn * 2:  # Heuristic for windowed attention
            windowed_blocks.append(block_indices[i])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(block_indices))
    width = 0.8

    # Create stacked bars
    p1 = ax.bar(x, attention_data, width, label='Attention', color='#3498db')
    p2 = ax.bar(x, mlp_data, width, bottom=attention_data, label='MLP', color='#e74c3c')

    # Highlight windowed attention blocks
    for block_idx in windowed_blocks:
        idx = block_indices.index(block_idx)
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.2, color='yellow', zorder=-1)

    ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    title = f'Per-Block Latency Breakdown: Attention vs MLP'
    if batch_size > 1:
        title += f' (Batch Size: {batch_size})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in block_indices])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add annotation for windowed blocks
    if windowed_blocks:
        ax.text(0.02, 0.98, f'Yellow = Windowed Attention Blocks ({windowed_blocks})',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'per_block_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'per_block_breakdown.png'}")
    plt.close()


def plot_overall_breakdown(profiler: ProfilerStats, results: dict, output_dir: Path):
    """Create pie chart of overall encoder breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart 1: High-level breakdown
    components = ['Attention', 'MLP', 'Patch Embed', 'Neck']
    times = [
        results['total_attention_time'] * 1000,
        results['total_mlp_time'] * 1000,
        results['patch_embed_time'] * 1000,
        results['neck_time'] * 1000,
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    explode = (0.05, 0.05, 0, 0)

    wedges1, texts1, autotexts1 = ax1.pie(
        times, labels=components, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    title1 = 'Encoder Component Breakdown'
    if results.get('batch_size', 1) > 1:
        title1 += f'\n(Batch Size: {results["batch_size"]})'
    ax1.set_title(title1, fontsize=14, fontweight='bold')

    # Pie chart 2: Transformer blocks only
    block_components = ['Attention', 'MLP']
    block_times = [
        results['total_attention_time'] * 1000,
        results['total_mlp_time'] * 1000,
    ]
    block_colors = ['#3498db', '#e74c3c']

    wedges2, texts2, autotexts2 = ax2.pie(
        block_times, labels=block_components, autopct='%1.1f%%',
        colors=block_colors, explode=(0.05, 0.05), startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title('Transformer Blocks Breakdown\n(Attention vs MLP)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'overall_breakdown.png'}")
    plt.close()


def plot_attention_vs_mlp_comparison(profiler: ProfilerStats, output_dir: Path, batch_size: int = 1):
    """Create comparison plot of attention vs MLP latency per block."""
    df = profiler.get_all_stats()

    # Extract attention and MLP times
    attention_times = []
    mlp_times = []
    block_indices = []

    for name in df['name']:
        if 'attention' in name and 'block_' in name and 'total' not in name:
            block_idx = int(name.split('_')[1])
            block_indices.append(block_idx)

            attn_time = df[df['name'] == name]['mean'].values[0] * 1000
            mlp_time = df[df['name'] == f'block_{block_idx:02d}_mlp']['mean'].values[0] * 1000

            attention_times.append(attn_time)
            mlp_times.append(mlp_time)

    # Sort by block index
    sorted_indices = np.argsort(block_indices)
    block_indices = [block_indices[i] for i in sorted_indices]
    attention_times = [attention_times[i] for i in sorted_indices]
    mlp_times = [mlp_times[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(block_indices))
    width = 0.35

    bars1 = ax.bar(x - width/2, attention_times, width, label='Attention',
                    color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, mlp_times, width, label='MLP',
                    color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    title = 'Attention vs MLP Latency per Block'
    if batch_size > 1:
        title += f' (Batch Size: {batch_size})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in block_indices])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add average lines
    avg_attn = np.mean(attention_times)
    avg_mlp = np.mean(mlp_times)
    ax.axhline(y=avg_attn, color='#3498db', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Avg Attention: {avg_attn:.2f}ms')
    ax.axhline(y=avg_mlp, color='#e74c3c', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Avg MLP: {avg_mlp:.2f}ms')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'attention_vs_mlp.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'attention_vs_mlp.png'}")
    plt.close()


def plot_latency_distribution(profiler: ProfilerStats, output_dir: Path):
    """Create box plot showing latency distribution."""
    df = profiler.get_all_stats()

    # Separate regular and windowed attention blocks
    regular_attn = []
    windowed_attn = []
    mlp_times = []

    for name in df['name']:
        if 'attention' in name and 'block_' in name and 'total' not in name:
            attn_time = df[df['name'] == name]['mean'].values[0] * 1000

            # Heuristic: windowed blocks have much higher latency
            if attn_time > 10:  # ms
                windowed_attn.append(attn_time)
            else:
                regular_attn.append(attn_time)

        if 'mlp' in name and 'block_' in name:
            mlp_time = df[df['name'] == name]['mean'].values[0] * 1000
            mlp_times.append(mlp_time)

    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [regular_attn, windowed_attn, mlp_times]
    labels = ['Regular Attention', 'Windowed Attention', 'MLP']
    colors = ['#3498db', '#9b59b6', '#e74c3c']

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2, color='red'),
                     meanprops=dict(linewidth=2, color='green', linestyle='--'))

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distribution by Component Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add legend for median and mean
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'latency_distribution.png'}")
    plt.close()


def plot_cumulative_time(profiler: ProfilerStats, output_dir: Path):
    """Create cumulative time plot showing how latency accumulates through blocks."""
    df = profiler.get_all_stats()

    # Extract per-block total times
    block_times = []
    block_indices = []

    for name in df['name']:
        if 'block_' in name and '_total' in name:
            block_idx = int(name.split('_')[1])
            block_indices.append(block_idx)
            block_time = df[df['name'] == name]['mean'].values[0] * 1000
            block_times.append(block_time)

    # Sort by block index
    sorted_indices = np.argsort(block_indices)
    block_indices = [block_indices[i] for i in sorted_indices]
    block_times = [block_times[i] for i in sorted_indices]

    # Calculate cumulative time
    cumulative_time = np.cumsum(block_times)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(block_indices, cumulative_time, linewidth=2.5,
            color='#2c3e50', marker='o', markersize=6, label='Cumulative Time')
    ax.fill_between(block_indices, cumulative_time, alpha=0.3, color='#3498db')

    ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Latency Through Transformer Blocks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add annotations for key points
    total_time = cumulative_time[-1]
    ax.axhline(y=total_time, color='red', linestyle='--', alpha=0.5)
    ax.text(0, total_time, f' Total: {total_time:.1f}ms',
            verticalalignment='bottom', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cumulative_time.png'}")
    plt.close()


def plot_heatmap(profiler: ProfilerStats, output_dir: Path):
    """Create heatmap of attention and MLP times across blocks."""
    df = profiler.get_all_stats()

    # Extract data
    attention_times = []
    mlp_times = []
    block_indices = []

    for name in df['name']:
        if 'attention' in name and 'block_' in name and 'total' not in name:
            block_idx = int(name.split('_')[1])
            block_indices.append(block_idx)

            attn_time = df[df['name'] == name]['mean'].values[0] * 1000
            mlp_time = df[df['name'] == f'block_{block_idx:02d}_mlp']['mean'].values[0] * 1000

            attention_times.append(attn_time)
            mlp_times.append(mlp_time)

    # Sort by block index
    sorted_indices = np.argsort(block_indices)
    block_indices = [block_indices[i] for i in sorted_indices]
    attention_times = [attention_times[i] for i in sorted_indices]
    mlp_times = [mlp_times[i] for i in sorted_indices]

    # Create heatmap data
    data = np.array([attention_times, mlp_times])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Attention', 'MLP'], fontsize=11)
    ax.set_xticks(range(len(block_indices)))
    ax.set_xticklabels([f'{i}' for i in block_indices])
    ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_title('Latency Heatmap: Attention and MLP Across Blocks',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Latency (ms)', fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'latency_heatmap.png'}")
    plt.close()


def plot_batch_throughput(profiler: ProfilerStats, results: dict, output_dir: Path):
    """Create throughput visualization for batch processing."""
    batch_size = results.get('batch_size', 1)
    if batch_size == 1:
        return  # Skip if not batching

    df = profiler.get_all_stats()

    # Get total encoder time
    total_time_per_image = results.get('total_encoder_time', 0) * 1000  # ms

    # Calculate throughput metrics
    images_per_second = batch_size / (total_time_per_image / 1000)
    time_per_image = total_time_per_image / batch_size

    # Create figure with metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Throughput bar
    metrics = ['Time per Image\n(ms)', 'Images per Second', 'Batch Size']
    values = [time_per_image, images_per_second, batch_size]
    colors = ['#3498db', '#2ecc71', '#f39c12']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Batch Processing Throughput Metrics', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Comparison - single vs batch
    single_image_time = total_time_per_image / batch_size  # Approximate
    batch_total_time = total_time_per_image
    batch_time_per_image = batch_total_time / batch_size

    scenarios = ['Single Image\n(Sequential)', f'Batch of {batch_size}\n(Per Image)']
    times = [single_image_time, batch_time_per_image]
    colors2 = ['#e74c3c', '#2ecc71']

    bars2 = ax2.bar(scenarios, times, color=colors2, alpha=0.8)
    ax2.set_ylabel('Time per Image (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Batch vs Sequential Processing', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels and speedup
    for bar, val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Calculate and show speedup
    speedup = single_image_time / batch_time_per_image if batch_time_per_image > 0 else 1.0
    ax2.text(0.5, 0.95, f'Speedup: {speedup:.2f}x',
            transform=ax2.transAxes, fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / 'batch_throughput.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'batch_throughput.png'}")
    plt.close()


def create_all_visualizations(
    model_type: str = "vit_b",
    checkpoint: str = None,
    image_path: str = None,
    output_dir: str = "./profiling_plots",
    batch_size: int = 1,
    num_runs: int = 10,
    device: str = "cuda",
):
    """
    Run profiling and create all visualizations with batch inference support.

    Args:
        model_type: SAM model variant
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
    print(f"PROFILING AND VISUALIZING SAM ENCODER - {model_type.upper()}")
    if batch_size > 1:
        print(f"BATCH INFERENCE MODE - Batch Size: {batch_size}")
    print("="*80)

    # Profile the encoder
    profiler, results = profile_and_collect_data(
        model_type=model_type,
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
    plot_per_block_breakdown(profiler, output_path, batch_size)
    plot_overall_breakdown(profiler, results, output_path)
    plot_attention_vs_mlp_comparison(profiler, output_path, batch_size)
    plot_latency_distribution(profiler, output_path)
    plot_cumulative_time(profiler, output_path)
    plot_heatmap(profiler, output_path)

    # Add batch throughput plot if batch_size > 1
    if batch_size > 1:
        plot_batch_throughput(profiler, results, output_path)

    print("\n" + "="*80)
    print(f"ALL VISUALIZATIONS SAVED TO: {output_path}")
    print("="*80)

    # Print summary
    analyze_encoder_breakdown(profiler, print_details=True)

    # Print batch-specific metrics
    if batch_size > 1:
        print("\n" + "="*80)
        print("BATCH PROCESSING METRICS")
        print("="*80)
        total_time = results.get('total_encoder_time', 0)
        print(f"Batch Size: {batch_size}")
        print(f"Total Images Processed: {results.get('total_images_processed', 0)}")
        print(f"Total Time per Batch: {total_time*1000:.2f} ms")
        print(f"Time per Image: {(total_time*1000)/batch_size:.2f} ms")
        print(f"Throughput: {batch_size/total_time:.2f} images/sec")

    print(f"\nGenerated {len(list(output_path.glob('*.png')))} plots:")
    for plot in sorted(output_path.glob('*.png')):
        print(f"  - {plot.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAM encoder latency profiling results with batch inference support"
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
        help="Path to test image or directory of images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./profiling_plots",
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

    args = parser.parse_args()

    create_all_visualizations(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        image_path=args.image,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_runs=args.runs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
