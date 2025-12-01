"""
Latency profiler for SAM2 encoder attention and MLP layers.

This module provides classes and utilities to profile the latency of individual
attention and MLP components in the SAM2 Hiera backbone encoder.
"""
import torch
import time
from typing import Optional
import pandas as pd
from collections import defaultdict

from sam2.modeling.backbones.hieradet import MultiScaleBlock, Hiera


class ProfilerStats:
    """
    Simple profiler statistics collector for tracking latency measurements.
    """

    def __init__(self):
        self.timings = defaultdict(list)

    def record(self, name: str, duration: float):
        """Record a timing measurement."""
        self.timings[name].append(duration)

    def clear(self):
        """Clear all recorded timings."""
        self.timings.clear()

    def get_all_stats(self) -> pd.DataFrame:
        """
        Get all statistics as a DataFrame.

        Returns:
            DataFrame with columns: name, count, mean, std, min, max, total
        """
        stats = []
        for name, times in self.timings.items():
            times_array = torch.tensor(times) if isinstance(times[0], (int, float)) else torch.stack(times)
            stats.append({
                'name': name,
                'count': len(times),
                'mean': times_array.mean().item(),
                'std': times_array.std().item(),
                'min': times_array.min().item(),
                'max': times_array.max().item(),
                'total': times_array.sum().item(),
            })
        return pd.DataFrame(stats)

    def print_summary(self):
        """Print a summary of profiling statistics."""
        df = self.get_all_stats()
        if df.empty:
            print("No profiling data")
            return

        print("\n" + "="*100)
        print("PROFILING SUMMARY")
        print("="*100)
        print(f"{'Name':<40} {'Count':>8} {'Mean (ms)':>12} {'Std (ms)':>12} {'Total (ms)':>12}")
        print("-"*100)

        for _, row in df.iterrows():
            print(f"{row['name']:<40} {row['count']:>8} "
                  f"{row['mean']*1000:>12.3f} {row['std']*1000:>12.3f} {row['total']*1000:>12.3f}")
        print("="*100 + "\n")


class profile:
    """
    Context manager for profiling code blocks.

    Usage:
        with profile("my_operation", profiler_stats):
            # code to profile
    """

    def __init__(self, name: str, profiler_stats: ProfilerStats, sync_cuda: bool = True):
        self.name = name
        self.profiler_stats = profiler_stats
        self.sync_cuda = sync_cuda
        self.start_time = None

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.time() - self.start_time
        self.profiler_stats.record(self.name, duration)


# Global profiler instance
_global_profiler = None


def get_profiler() -> ProfilerStats:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = ProfilerStats()
    return _global_profiler


class HieraBlockProfiler(MultiScaleBlock):
    """
    Profiling wrapper for SAM2 HieraBlock that measures attention and MLP latency.

    This class extends HieraBlock to add precise timing measurements for:
    - Attention layer (including normalization)
    - MLP layer (including normalization)

    Timings are recorded to a ProfilerStats instance for analysis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_idx = 0
        self.profiler_stats = None
        self.sync_cuda = True

    def set_profiler(self, profiler_stats: ProfilerStats, block_idx: int):
        """
        Configure the profiler for this block.

        Args:
            profiler_stats: ProfilerStats instance to record timings
            block_idx: Index of this block in the encoder
        """
        self.profiler_stats = profiler_stats
        self.block_idx = block_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with latency profiling.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Output tensor of shape (B, H', W', C') (may be downsampled)
        """
        profiler = self.profiler_stats if self.profiler_stats else get_profiler()

        shortcut = x  # B, H, W, C
        x_norm = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            from sam2.modeling.backbones.hieradet import do_pool
            shortcut = do_pool(self.proj(x_norm), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            from sam2.modeling.backbones.hieradet import window_partition
            H, W = x_norm.shape[1], x_norm.shape[2]
            x_norm, pad_hw = window_partition(x_norm, window_size)

        # Window Attention + Q Pooling (if stage change)
        with profile(f"block_{self.block_idx:02d}_attention", profiler, self.sync_cuda):
            x_attn = self.attn(x_norm)

        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            from sam2.modeling.backbones.hieradet import window_unpartition
            x_attn = window_unpartition(x_attn, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x_attn)

        # MLP
        with profile(f"block_{self.block_idx:02d}_mlp", profiler, self.sync_cuda):
            x_mlp = self.mlp(self.norm2(x))

        x = x + self.drop_path(x_mlp)

        return x


class HieraProfiler(Hiera):
    """
    Profiling wrapper for SAM2 Hiera backbone that measures per-block latencies.

    This class extends Hiera to profile the entire encoder forward pass
    and collect per-block attention and MLP timings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler_stats = None
        self.sync_cuda = True

    def set_profiler(self, profiler_stats: ProfilerStats):
        """
        Configure the profiler for this encoder.

        Args:
            profiler_stats: ProfilerStats instance to record timings
        """
        self.profiler_stats = profiler_stats

    def forward(self, x: torch.Tensor):
        """
        Forward pass with latency profiling.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output features list (same format as original Hiera)
        """
        profiler = self.profiler_stats if self.profiler_stats else get_profiler()

        # Profile patch embedding
        with profile("patch_embed", profiler, self.sync_cuda):
            x = self.patch_embed(x)
            # x: (B, H, W, C)

            # Add pos embed
            x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []

        # Profile each block
        for idx, blk in enumerate(self.blocks):
            with profile(f"block_{idx:02d}_total", profiler, self.sync_cuda):
                x = blk(x)

            # Collect intermediate features at stage boundaries
            if (idx == self.stage_ends[-1]) or (
                idx in self.stage_ends and self.return_interm_layers
            ):
                with profile("neck", profiler, self.sync_cuda):
                    feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs


def sam2_encoder_latency_monkey_patch(
    model,
    profiler_stats: Optional[ProfilerStats] = None,
    sync_cuda: bool = True
):
    """
    Apply monkey-patching to SAM2 image encoder for latency profiling.

    This function replaces the encoder's HieraBlock and Hiera classes with
    profiling versions that measure attention and MLP latencies.

    Args:
        model: SAM2 model to patch
        profiler_stats: ProfilerStats instance to use (defaults to global profiler)
        sync_cuda: Whether to synchronize CUDA before timing (recommended for GPU)

    Returns:
        ProfilerStats instance being used

    Example:
        >>> from sam2.build_sam import build_sam2
        >>> from sam2.sam2_image_predictor import SAM2ImagePredictor
        >>> sam2_model = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt")
        >>> predictor = SAM2ImagePredictor(sam2_model)
        >>> profiler = sam2_encoder_latency_monkey_patch(predictor.model)
        >>> # Run inference
        >>> predictor.set_image(image)
        >>> profiler.print_summary()
    """
    if profiler_stats is None:
        profiler_stats = ProfilerStats()

    block_idx = 0

    # Replace HieraBlock classes with profiling versions
    for module in model.modules():
        if isinstance(module, MultiScaleBlock) and not isinstance(module, HieraBlockProfiler):
            # Monkey-patch to HieraBlockProfiler
            module.__class__ = HieraBlockProfiler
            module.profiler_stats = profiler_stats
            module.sync_cuda = sync_cuda
            module.block_idx = block_idx
            block_idx += 1

        elif isinstance(module, Hiera) and not isinstance(module, HieraProfiler):
            # Monkey-patch to HieraProfiler
            module.__class__ = HieraProfiler
            module.profiler_stats = profiler_stats
            module.sync_cuda = sync_cuda

    return profiler_stats


def analyze_sam2_encoder_breakdown(profiler_stats: ProfilerStats, print_details: bool = True):
    """
    Analyze SAM2 encoder profiling results and compute breakdown statistics.

    Args:
        profiler_stats: ProfilerStats instance with recorded timings
        print_details: Whether to print detailed analysis

    Returns:
        Dictionary with analysis results
    """
    df = profiler_stats.get_all_stats()
    if df.empty:
        print("No profiling data available")
        return {}

    # Separate attention and MLP timings
    attention_df = df[df['name'].str.contains('attention') & ~df['name'].str.contains('total')]
    mlp_df = df[df['name'].str.contains('mlp')]
    total_df = df[df['name'].str.contains('total')]

    # Compute aggregated statistics
    total_attention_time = attention_df['total'].sum() if not attention_df.empty else 0
    total_mlp_time = mlp_df['total'].sum() if not mlp_df.empty else 0
    total_blocks_time = total_df['total'].sum() if not total_df.empty else 0

    # Get patch_embed and neck times
    patch_embed_time = df[df['name'] == 'patch_embed']['total'].sum()
    neck_time = df[df['name'] == 'neck']['total'].sum()

    total_encoder_time = df['total'].sum()

    results = {
        'total_attention_time': total_attention_time,
        'total_mlp_time': total_mlp_time,
        'total_blocks_time': total_blocks_time,
        'patch_embed_time': patch_embed_time,
        'neck_time': neck_time,
        'total_encoder_time': total_encoder_time,
        'attention_percentage': (total_attention_time / total_encoder_time * 100) if total_encoder_time > 0 else 0,
        'mlp_percentage': (total_mlp_time / total_encoder_time * 100) if total_encoder_time > 0 else 0,
        'num_blocks': len(attention_df),
    }

    if print_details:
        print("\n" + "="*80)
        print("SAM2 ENCODER LATENCY BREAKDOWN")
        print("="*80)
        print(f"Total encoder time:      {total_encoder_time*1000:8.2f} ms (100.0%)")
        print(f"  - Patch embedding:     {patch_embed_time*1000:8.2f} ms ({patch_embed_time/total_encoder_time*100:5.1f}%)")
        print(f"  - Hiera blocks:        {total_blocks_time*1000:8.2f} ms ({total_blocks_time/total_encoder_time*100:5.1f}%)")
        print(f"      * Attention:       {total_attention_time*1000:8.2f} ms ({total_attention_time/total_encoder_time*100:5.1f}%)")
        print(f"      * MLP:             {total_mlp_time*1000:8.2f} ms ({total_mlp_time/total_encoder_time*100:5.1f}%)")
        print(f"  - Neck:                {neck_time*1000:8.2f} ms ({neck_time/total_encoder_time*100:5.1f}%)")
        print("="*80)

        # Per-block breakdown
        print("\nPER-BLOCK LATENCY BREAKDOWN")
        print("="*80)
        print(f"{'Block':<10} {'Attention (ms)':<18} {'MLP (ms)':<18} {'Total (ms)':<18} {'Attn %':<10}")
        print("-"*80)

        for i in range(results['num_blocks']):
            attn_row = attention_df[attention_df['name'] == f'block_{i:02d}_attention']
            mlp_row = mlp_df[mlp_df['name'] == f'block_{i:02d}_mlp']
            total_row = total_df[total_df['name'] == f'block_{i:02d}_total']

            attn_time = attn_row['mean'].values[0] * 1000 if not attn_row.empty else 0
            mlp_time = mlp_row['mean'].values[0] * 1000 if not mlp_row.empty else 0
            total_time = total_row['mean'].values[0] * 1000 if not total_row.empty else 0
            attn_pct = (attn_time / total_time * 100) if total_time > 0 else 0

            print(f"Block {i:02d}   {attn_time:8.3f}          {mlp_time:8.3f}          {total_time:8.3f}          {attn_pct:5.1f}%")

        print("="*80 + "\n")

        # Summary statistics
        if not attention_df.empty and not mlp_df.empty:
            avg_attn = attention_df['mean'].mean() * 1000
            avg_mlp = mlp_df['mean'].mean() * 1000

            print("AVERAGE PER-BLOCK LATENCY")
            print("="*80)
            print(f"Average attention:  {avg_attn:.3f} ms")
            print(f"Average MLP:        {avg_mlp:.3f} ms")
            print(f"Attention/MLP ratio: {avg_attn/avg_mlp:.2f}x" if avg_mlp > 0 else "N/A")
            print("="*80 + "\n")

    return results
