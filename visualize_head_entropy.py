#!/usr/bin/env python3
"""
Visualize attention head entropy distribution as heatmaps.

This script:
1. Calibrates a processor to collect entropy statistics
2. Creates heatmaps showing entropy values across layers and heads
3. Highlights which heads are marked for pruning/quantization

Usage:
    python visualize_head_entropy.py \
        --config-file quant/config/hq44k/rtn.yaml \
        --processor POSITIONAL_PRUNE \
        --percent-entropy 0.3 \
        --num-calib-samples 32
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from omegaconf import OmegaConf

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.image_encoder import Attention as EncoderSamAttention
from segment_anything.modeling.transformer import Attention as DecoderAttention
from train.segment_anything_training.modeling.image_encoder import Attention as EncoderAttentionTraining
from seginw.segment_anything.modeling.image_encoder import Attention as EncoderAttention

from processors import get_encoder_processor
from small_engine import Engine, override_args


def extract_entropy_statistics(processor):
    """
    Extract entropy statistics from calibrated processor.

    Returns:
        dict: {layer_name: {head_idx: entropy_value}}
    """
    if not hasattr(processor, 'entropy_stats'):
        print("Warning: Processor has no entropy_stats attribute")
        return {}

    print(f"\nExtracting entropy statistics from {processor.strategy_name}...")
    print(f"Found {len(processor.entropy_stats)} head entries")

    # Group by layer and head
    layer_head_entropy = {}

    for head_key, entropy_values in processor.entropy_stats.items():
        if len(entropy_values) == 0:
            continue

        # Parse key format: 'blocks.22.attn.head_233'
        parts = head_key.split('.')
        if len(parts) < 4:
            continue

        block_idx = int(parts[1])
        head_idx = int(parts[3].split('_')[1])

        # Calculate mean entropy
        entropy_tensor = torch.tensor(entropy_values) if not isinstance(entropy_values[0], torch.Tensor) else torch.stack(entropy_values)
        mean_entropy = torch.mean(entropy_tensor).item()

        layer_name = f"Block {block_idx}"
        if layer_name not in layer_head_entropy:
            layer_head_entropy[layer_name] = {}

        layer_head_entropy[layer_name][head_idx] = mean_entropy

    print(f"Organized into {len(layer_head_entropy)} layers")
    return layer_head_entropy


def extract_pruning_masks(processor):
    """
    Extract pruning/quantization masks from calibrated processor.

    Returns:
        dict: {layer_name: mask_tensor or list of head_indices}
    """
    if not hasattr(processor, 'final_entropy_stats'):
        print("Warning: Processor has no final_entropy_stats attribute")
        return {}

    print(f"\nExtracting pruning masks...")
    print(f"Found {len(processor.final_entropy_stats)} layers with masks")

    masks = {}
    for layer_name, mask_data in processor.final_entropy_stats.items():
        # Convert layer name format
        # from 'image_encoder.blocks.0.attn' to 'Block 0'
        parts = layer_name.split('.')
        if 'blocks' in parts:
            block_idx = int(parts[parts.index('blocks') + 1])
            simple_name = f"Block {block_idx}"
            masks[simple_name] = mask_data

    return masks


def create_entropy_heatmap(layer_head_entropy, pruning_masks, output_path, processor_name, percent_entropy):
    """
    Create a comprehensive entropy heatmap.

    Args:
        layer_head_entropy: Dict of {layer_name: {head_idx: entropy}}
        pruning_masks: Dict of {layer_name: mask}
        output_path: Path to save figure
        processor_name: Name of processor
        percent_entropy: Percent entropy value used
    """
    if not layer_head_entropy:
        print("No entropy data to plot")
        return

    # Determine the type of processor and head granularity
    # HeadPruneProcessor: 16 heads per layer
    # PositionalPrune/Quant: 400 positions = 16 heads × 25 positions

    sample_layer = list(layer_head_entropy.values())[0]
    max_head_idx = max(sample_layer.keys()) if sample_layer else 0

    # Determine granularity
    if max_head_idx < 20:
        # Head-level (HeadPruneProcessor)
        num_heads = 16
        granularity = "head"
        print(f"Detected head-level granularity (max_head_idx={max_head_idx})")
    else:
        # Positional-level (PositionalPrune/QuantProcessor)
        num_positions = 400
        num_heads = 16
        positions_per_head = 25
        granularity = "positional"
        print(f"Detected positional-level granularity (max_head_idx={max_head_idx})")

    # Sort layers by block number
    sorted_layers = sorted(layer_head_entropy.keys(), key=lambda x: int(x.split()[1]))
    num_layers = len(sorted_layers)

    # Create data matrix
    if granularity == "head":
        data_matrix = np.zeros((num_layers, num_heads))
        mask_matrix = np.zeros((num_layers, num_heads), dtype=bool)

        for i, layer_name in enumerate(sorted_layers):
            layer_data = layer_head_entropy[layer_name]
            for head_idx, entropy in layer_data.items():
                if head_idx < num_heads:
                    data_matrix[i, head_idx] = entropy

            # Mark pruned heads
            if layer_name in pruning_masks:
                mask = pruning_masks[layer_name]
                if isinstance(mask, torch.Tensor):
                    mask_matrix[i, :] = mask.cpu().numpy()[:num_heads]
                elif isinstance(mask, list):
                    for head_idx in mask:
                        if head_idx < num_heads:
                            mask_matrix[i, head_idx] = True
    else:
        # Positional: Show as 16 heads × num_layers
        # Average entropy across positions within each head
        data_matrix = np.zeros((num_layers, num_heads))
        mask_matrix = np.zeros((num_layers, num_heads), dtype=bool)

        for i, layer_name in enumerate(sorted_layers):
            layer_data = layer_head_entropy[layer_name]

            # Group positions by head (positions 0-24 -> head 0, 25-49 -> head 1, etc.)
            head_entropies = [[] for _ in range(num_heads)]
            for pos_idx, entropy in layer_data.items():
                head_idx = pos_idx // positions_per_head
                if head_idx < num_heads:
                    head_entropies[head_idx].append(entropy)

            # Average entropy per head
            for head_idx in range(num_heads):
                if head_entropies[head_idx]:
                    data_matrix[i, head_idx] = np.mean(head_entropies[head_idx])

            # Mark pruned positions (aggregate to head level for visualization)
            if layer_name in pruning_masks:
                mask = pruning_masks[layer_name]
                if isinstance(mask, torch.Tensor):
                    mask_array = mask.cpu().numpy()
                    # Count pruned positions per head
                    for head_idx in range(num_heads):
                        start_pos = head_idx * positions_per_head
                        end_pos = (head_idx + 1) * positions_per_head
                        if end_pos <= len(mask_array):
                            pruned_count = mask_array[start_pos:end_pos].sum()
                            # Mark as pruned if more than 50% positions are pruned
                            mask_matrix[i, head_idx] = (pruned_count / positions_per_head) > 0.5

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(10, num_layers * 0.3)))

    # Subplot 1: Entropy heatmap
    sns.heatmap(
        data_matrix,
        ax=ax1,
        cmap='YlOrRd',
        cbar_kws={'label': 'Mean Entropy'},
        xticklabels=[f'H{i}' for i in range(num_heads)],
        yticklabels=sorted_layers,
        linewidths=0.5,
        linecolor='gray'
    )
    ax1.set_xlabel('Attention Head', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_title(f'Attention Head Entropy Distribution\n({processor_name})',
                  fontsize=14, fontweight='bold')

    # Subplot 2: Pruning mask overlay
    # Create a visualization showing which heads are selected for pruning/quantization
    overlay_data = data_matrix.copy()
    overlay_data[mask_matrix] = np.nan  # Mark pruned heads

    sns.heatmap(
        overlay_data,
        ax=ax2,
        cmap='YlOrRd',
        cbar_kws={'label': 'Mean Entropy (Active Heads Only)'},
        xticklabels=[f'H{i}' for i in range(num_heads)],
        yticklabels=sorted_layers,
        linewidths=0.5,
        linecolor='gray',
        mask=mask_matrix,
        cbar=True
    )

    # Overlay X marks on pruned heads
    for i in range(num_layers):
        for j in range(num_heads):
            if mask_matrix[i, j]:
                ax2.text(j + 0.5, i + 0.5, 'X',
                        ha='center', va='center',
                        color='blue', fontsize=14, fontweight='bold')

    ax2.set_xlabel('Attention Head', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_title(f'Pruned/Quantized Heads (X marked)\npercent_entropy={percent_entropy}',
                  fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved entropy heatmap to: {output_path}")
    plt.close()

    # Create a summary statistics plot
    create_entropy_statistics_plot(data_matrix, mask_matrix, sorted_layers,
                                   output_path.replace('.png', '_stats.png'),
                                   processor_name, percent_entropy)


def create_entropy_statistics_plot(data_matrix, mask_matrix, layer_names,
                                   output_path, processor_name, percent_entropy):
    """Create detailed statistics plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Entropy distribution (histogram)
    all_entropies = data_matrix[data_matrix > 0].flatten()
    ax1.hist(all_entropies, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(all_entropies), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(all_entropies):.3f}')
    ax1.axvline(np.median(all_entropies), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(all_entropies):.3f}')
    ax1.set_xlabel('Entropy Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Entropy Distribution Across All Heads', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Per-layer entropy statistics
    layer_means = data_matrix.mean(axis=1)
    layer_stds = data_matrix.std(axis=1)
    x_pos = np.arange(len(layer_names))

    ax2.errorbar(x_pos, layer_means, yerr=layer_stds,
                 fmt='o-', capsize=5, capthick=2, markersize=6, linewidth=2)
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Entropy', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Layer Entropy Statistics', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos[::2])  # Show every other layer
    ax2.set_xticklabels([layer_names[i] for i in range(0, len(layer_names), 2)],
                        rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # 3. Pruning ratio per layer
    pruning_ratios = mask_matrix.sum(axis=1) / mask_matrix.shape[1]

    colors = ['green' if r < 0.3 else 'orange' if r < 0.6 else 'red'
              for r in pruning_ratios]
    ax3.barh(x_pos, pruning_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Pruning Ratio', fontsize=12, fontweight='bold')
    ax3.set_title(f'Pruning Ratio per Layer (percent_entropy={percent_entropy})',
                  fontsize=13, fontweight='bold')
    ax3.set_yticks(x_pos[::2])
    ax3.set_yticklabels([layer_names[i] for i in range(0, len(layer_names), 2)])
    ax3.set_xlim([0, 1])
    ax3.grid(True, alpha=0.3, axis='x')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Low (<30%)'),
        Patch(facecolor='orange', alpha=0.7, label='Medium (30-60%)'),
        Patch(facecolor='red', alpha=0.7, label='High (>60%)')
    ]
    ax3.legend(handles=legend_elements, loc='lower right')

    # 4. Entropy vs Pruning correlation
    pruned_entropies = data_matrix[mask_matrix]
    active_entropies = data_matrix[~mask_matrix & (data_matrix > 0)]

    if len(pruned_entropies) > 0 and len(active_entropies) > 0:
        data_to_plot = [active_entropies, pruned_entropies]
        labels = ['Active Heads', 'Pruned/Quantized Heads']

        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True,
                         showmeans=True, meanline=True)

        # Color boxes
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.set_ylabel('Entropy Value', fontsize=12, fontweight='bold')
        ax4.set_title('Entropy Comparison: Active vs Pruned Heads',
                      fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add text with statistics
        active_mean = np.mean(active_entropies)
        pruned_mean = np.mean(pruned_entropies)
        ax4.text(0.05, 0.95,
                f'Active mean: {active_mean:.3f}\n'
                f'Pruned mean: {pruned_mean:.3f}\n'
                f'Difference: {abs(active_mean - pruned_mean):.3f}',
                transform=ax4.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No pruning data available',
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, fontweight='bold')

    fig.suptitle(f'Entropy Statistics Summary - {processor_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved statistics plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention head entropy distribution'
    )

    parser.add_argument('--config-file', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--processor', type=str, default='POSITIONAL_PRUNE',
                        choices=['POSITIONAL_PRUNE', 'POSITIONAL_QUANT', 'HEAD_PRUNE'],
                        help='Processor to visualize')
    parser.add_argument('--percent-entropy', type=float, default=0.3,
                        help='Percent entropy value')
    parser.add_argument('--num-calib-samples', type=int, default=32,
                        help='Number of calibration samples')
    parser.add_argument('--output-dir', type=str, default='./entropy_plots',
                        help='Output directory for plots')
    parser.add_argument('--n-bits', type=int, default=4,
                        help='Number of quantization bits')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = OmegaConf.load(args.config_file)
    config.quantization.percent_entropy = args.percent_entropy
    config.quantization.high_entropy = True

    print(f"\n{'='*80}")
    print(f"Visualizing Entropy for {args.processor}")
    print(f"percent_entropy = {args.percent_entropy}")
    print(f"{'='*80}\n")

    # Initialize model
    model_type = 'vit_l'
    checkpoint_path = './pretrained_checkpoint/sam_hq_vit_l.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Initialize engine
    engine = Engine('entropy_viz', quantize_encoder=True, quantize_decoder=False)

    # Get processor
    processor = get_encoder_processor(args.processor)

    # Calibrate to collect entropy statistics
    print(f"\nCalibrating {args.processor}...")
    processor.set_dataloaders(engine.dataloaders)
    processor.set_accelerator(engine.accelerator)
    processor.set_params(config)

    processor.calibrate(
        predictor=predictor,
        modules=(DecoderAttention, EncoderAttentionTraining, EncoderAttention, EncoderSamAttention),
        num_samples=args.num_calib_samples
    )

    print("\nCalibration complete!")

    # Extract statistics
    layer_head_entropy = extract_entropy_statistics(processor)
    pruning_masks = extract_pruning_masks(processor)

    if not layer_head_entropy:
        print("\nError: No entropy statistics found!")
        print("Make sure the processor implements entropy collection during calibration.")
        return

    # Print summary
    print("\n" + "="*80)
    print("ENTROPY SUMMARY")
    print("="*80)
    all_entropies = []
    for layer_data in layer_head_entropy.values():
        all_entropies.extend(layer_data.values())

    print(f"Total heads/positions tracked: {len(all_entropies)}")
    print(f"Mean entropy: {np.mean(all_entropies):.4f}")
    print(f"Std entropy: {np.std(all_entropies):.4f}")
    print(f"Min entropy: {np.min(all_entropies):.4f}")
    print(f"Max entropy: {np.max(all_entropies):.4f}")

    if pruning_masks:
        total_pruned = sum(
            mask.sum().item() if isinstance(mask, torch.Tensor) else len(mask)
            for mask in pruning_masks.values()
        )
        print(f"\nTotal pruned/quantized elements: {total_pruned}")

    # Create visualizations
    output_path = os.path.join(
        args.output_dir,
        f'{args.processor}_entropy_heatmap_p{args.percent_entropy:.1f}.png'
    )

    create_entropy_heatmap(
        layer_head_entropy,
        pruning_masks,
        output_path,
        args.processor,
        args.percent_entropy
    )

    print("\n" + "="*80)
    print(f"✓ Visualization complete!")
    print(f"Plots saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
