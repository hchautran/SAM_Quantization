#!/usr/bin/env python3
"""
Visualization script for benchmark results.

Creates comparison plots for processor benchmarking:
- mIoU vs percent_entropy
- GFLOPs vs percent_entropy
- mIoU vs GFLOPs (efficiency trade-off)

Usage:
    python visualize_benchmark_results.py \
        --input benchmark_results/benchmark_results_20250129_123456.csv \
        --output plots/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_results(csv_path):
    """Load benchmark results from CSV"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    print(f"Processors: {df['processor'].unique()}")
    print(f"Percent entropy range: {df['percent_entropy'].min():.1f} - {df['percent_entropy'].max():.1f}")
    return df


def plot_miou_vs_percent(df, output_dir, metric='val_iou_0'):
    """Plot mIoU vs percent_entropy for each processor"""
    fig, ax = plt.subplots(figsize=(12, 8))

    processors = df['processor'].unique()
    colors = sns.color_palette("husl", len(processors))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, processor in enumerate(processors):
        proc_data = df[df['processor'] == processor].sort_values('percent_entropy')

        # Plot line
        ax.plot(
            proc_data['percent_entropy'],
            proc_data[metric],
            marker=markers[i % len(markers)],
            linestyle='-',
            linewidth=2.5,
            markersize=10,
            color=colors[i],
            label=processor.replace('_', ' '),
            alpha=0.8
        )

        # Add error bars if std is available
        if f'{metric}_std' in proc_data.columns:
            ax.fill_between(
                proc_data['percent_entropy'],
                proc_data[metric] - proc_data[f'{metric}_std'],
                proc_data[metric] + proc_data[f'{metric}_std'],
                alpha=0.2,
                color=colors[i]
            )

    ax.set_xlabel('Percent Entropy', fontsize=14, fontweight='bold')
    ax.set_ylabel('mIoU', fontsize=14, fontweight='bold')
    ax.set_title('mIoU vs Percent Entropy Across Processors', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([df['percent_entropy'].min() - 0.05, df['percent_entropy'].max() + 0.05])

    # Add horizontal line at baseline (percent=0 if available)
    if 0.0 in df['percent_entropy'].values:
        baseline_miou = df[df['percent_entropy'] == 0.0][metric].mean()
        ax.axhline(y=baseline_miou, color='red', linestyle='--', alpha=0.5, label='Baseline')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'miou_vs_percent_entropy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gflops_vs_percent(df, output_dir):
    """Plot GFLOPs vs percent_entropy for each processor"""
    fig, ax = plt.subplots(figsize=(12, 8))

    processors = df['processor'].unique()
    colors = sns.color_palette("husl", len(processors))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, processor in enumerate(processors):
        proc_data = df[df['processor'] == processor].sort_values('percent_entropy')

        ax.plot(
            proc_data['percent_entropy'],
            proc_data['encoder_gflops'],
            marker=markers[i % len(markers)],
            linestyle='-',
            linewidth=2.5,
            markersize=10,
            color=colors[i],
            label=processor.replace('_', ' '),
            alpha=0.8
        )

    ax.set_xlabel('Percent Entropy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Encoder GFLOPs', fontsize=14, fontweight='bold')
    ax.set_title('Computational Cost vs Percent Entropy', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([df['percent_entropy'].min() - 0.05, df['percent_entropy'].max() + 0.05])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'gflops_vs_percent_entropy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_tradeoff(df, output_dir, metric='val_iou_0'):
    """Plot mIoU vs GFLOPs trade-off curve"""
    fig, ax = plt.subplots(figsize=(12, 8))

    processors = df['processor'].unique()
    colors = sns.color_palette("husl", len(processors))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, processor in enumerate(processors):
        proc_data = df[df['processor'] == processor].sort_values('encoder_gflops')

        # Create scatter plot
        scatter = ax.scatter(
            proc_data['encoder_gflops'],
            proc_data[metric],
            s=200,
            marker=markers[i % len(markers)],
            color=colors[i],
            label=processor.replace('_', ' '),
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )

        # Add line connecting points
        ax.plot(
            proc_data['encoder_gflops'],
            proc_data[metric],
            linestyle='-',
            linewidth=1.5,
            color=colors[i],
            alpha=0.4
        )

        # Annotate points with percent_entropy
        for _, row in proc_data.iterrows():
            ax.annotate(
                f"{row['percent_entropy']:.1f}",
                (row['encoder_gflops'], row[metric]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3)
            )

    ax.set_xlabel('Encoder GFLOPs', fontsize=14, fontweight='bold')
    ax.set_ylabel('mIoU', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy-Efficiency Trade-off (Lower-Right is Better)', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier annotation
    ax.text(
        0.02, 0.98,
        'Note: Numbers indicate percent_entropy values',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'miou_vs_gflops_tradeoff.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_boundary_iou_comparison(df, output_dir, metric='val_boundary_iou_0'):
    """Plot boundary IoU comparison"""
    if metric not in df.columns:
        print(f"Skipping boundary IoU plot: {metric} not in results")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    processors = df['processor'].unique()
    colors = sns.color_palette("husl", len(processors))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, processor in enumerate(processors):
        proc_data = df[df['processor'] == processor].sort_values('percent_entropy')

        ax.plot(
            proc_data['percent_entropy'],
            proc_data[metric],
            marker=markers[i % len(markers)],
            linestyle='-',
            linewidth=2.5,
            markersize=10,
            color=colors[i],
            label=processor.replace('_', ' '),
            alpha=0.8
        )

    ax.set_xlabel('Percent Entropy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Boundary IoU', fontsize=14, fontweight='bold')
    ax.set_title('Boundary IoU vs Percent Entropy', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([df['percent_entropy'].min() - 0.05, df['percent_entropy'].max() + 0.05])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'boundary_iou_vs_percent_entropy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison_table(df, output_dir):
    """Create a comparison table figure"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Group by processor and calculate statistics
    summary_data = []
    for processor in df['processor'].unique():
        proc_data = df[df['processor'] == processor]

        summary_data.append([
            processor.replace('_', '\n'),
            f"{proc_data['val_iou_0'].mean():.4f}\n± {proc_data['val_iou_0'].std():.4f}",
            f"{proc_data['val_iou_0'].max():.4f}",
            f"{proc_data['val_iou_0'].min():.4f}",
            f"{proc_data['encoder_gflops'].mean():.2f}\n± {proc_data['encoder_gflops'].std():.2f}",
            f"{proc_data['encoder_gflops'].min():.2f}",
        ])

    headers = ['Processor', 'Mean mIoU\n(± std)', 'Max mIoU', 'Min mIoU', 'Mean GFLOPs\n(± std)', 'Min GFLOPs']

    table = ax.table(
        cellText=summary_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Style rows
    colors = ['#D9E1F2', '#FFFFFF']
    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[(i - 1) % 2])

    plt.title('Processor Comparison Summary', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap(df, output_dir, metric='val_iou_0'):
    """Create heatmap showing mIoU across processors and percent_entropy values"""
    # Pivot data for heatmap
    pivot_data = df.pivot_table(
        values=metric,
        index='processor',
        columns='percent_entropy',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        center=pivot_data.mean().mean(),
        cbar_kws={'label': 'mIoU'},
        linewidths=1,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Percent Entropy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Processor', fontsize=14, fontweight='bold')
    ax.set_title('mIoU Heatmap: Processor vs Percent Entropy', fontsize=16, fontweight='bold')

    # Fix y-axis labels
    yticklabels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, rotation=0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'miou_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_report(df, output_dir):
    """Generate a text report with key findings"""
    report_path = os.path.join(output_dir, 'benchmark_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total configurations tested: {len(df)}\n")
        f.write(f"Processors: {', '.join(df['processor'].unique())}\n")
        f.write(f"Percent entropy range: {df['percent_entropy'].min():.1f} - {df['percent_entropy'].max():.1f}\n")
        f.write(f"Quantization bits: {df['n_bits'].iloc[0]}\n\n")

        f.write("-"*80 + "\n")
        f.write("SUMMARY BY PROCESSOR\n")
        f.write("-"*80 + "\n\n")

        for processor in df['processor'].unique():
            proc_data = df[df['processor'] == processor]

            f.write(f"{processor}:\n")
            f.write(f"  Number of configurations: {len(proc_data)}\n")
            f.write(f"  Mean mIoU: {proc_data['val_iou_0'].mean():.4f} ± {proc_data['val_iou_0'].std():.4f}\n")
            f.write(f"  Max mIoU: {proc_data['val_iou_0'].max():.4f} (at percent={proc_data.loc[proc_data['val_iou_0'].idxmax(), 'percent_entropy']:.1f})\n")
            f.write(f"  Min mIoU: {proc_data['val_iou_0'].min():.4f} (at percent={proc_data.loc[proc_data['val_iou_0'].idxmin(), 'percent_entropy']:.1f})\n")
            f.write(f"  Mean GFLOPs: {proc_data['encoder_gflops'].mean():.2f} ± {proc_data['encoder_gflops'].std():.2f}\n")
            f.write(f"  Min GFLOPs: {proc_data['encoder_gflops'].min():.2f} (at percent={proc_data.loc[proc_data['encoder_gflops'].idxmin(), 'percent_entropy']:.1f})\n")

            if 'val_boundary_iou_0' in proc_data.columns:
                f.write(f"  Mean Boundary IoU: {proc_data['val_boundary_iou_0'].mean():.4f} ± {proc_data['val_boundary_iou_0'].std():.4f}\n")

            f.write("\n")

        f.write("-"*80 + "\n")
        f.write("BEST CONFIGURATIONS\n")
        f.write("-"*80 + "\n\n")

        # Best mIoU
        best_miou_idx = df['val_iou_0'].idxmax()
        best_miou_row = df.loc[best_miou_idx]
        f.write(f"Best mIoU: {best_miou_row['val_iou_0']:.4f}\n")
        f.write(f"  Processor: {best_miou_row['processor']}\n")
        f.write(f"  Percent entropy: {best_miou_row['percent_entropy']:.1f}\n")
        f.write(f"  GFLOPs: {best_miou_row['encoder_gflops']:.2f}\n\n")

        # Best efficiency (lowest GFLOPs with acceptable accuracy)
        # Define acceptable accuracy as within 1% of best
        acceptable_threshold = df['val_iou_0'].max() * 0.99
        acceptable_df = df[df['val_iou_0'] >= acceptable_threshold]
        if len(acceptable_df) > 0:
            best_efficiency_idx = acceptable_df['encoder_gflops'].idxmin()
            best_efficiency_row = acceptable_df.loc[best_efficiency_idx]
            f.write(f"Best Efficiency (lowest GFLOPs with >99% of best mIoU):\n")
            f.write(f"  mIoU: {best_efficiency_row['val_iou_0']:.4f}\n")
            f.write(f"  Processor: {best_efficiency_row['processor']}\n")
            f.write(f"  Percent entropy: {best_efficiency_row['percent_entropy']:.1f}\n")
            f.write(f"  GFLOPs: {best_efficiency_row['encoder_gflops']:.2f}\n\n")

        f.write("="*80 + "\n")

    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize benchmark results from processor comparison'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with benchmark results')
    parser.add_argument('--output', type=str, default='./plots',
                        help='Output directory for plots')
    parser.add_argument('--metric', type=str, default='val_iou_0',
                        help='Primary metric to plot (default: val_iou_0)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load results
    df = load_results(args.input)

    # Generate all plots
    print("\nGenerating plots...")
    plot_miou_vs_percent(df, args.output, metric=args.metric)
    plot_gflops_vs_percent(df, args.output)
    plot_efficiency_tradeoff(df, args.output, metric=args.metric)
    plot_boundary_iou_comparison(df, args.output)
    plot_comparison_table(df, args.output)
    plot_heatmap(df, args.output, metric=args.metric)

    # Generate report
    print("\nGenerating report...")
    generate_report(df, args.output)

    print(f"\n✓ All visualizations saved to: {args.output}")


if __name__ == '__main__':
    main()
