"""
Latency profiler for SAM encoder attention and MLP layers.

This module provides classes and utilities to profile the latency of individual
attention and MLP components in the SAM Vision Transformer encoder.
"""
import torch
from typing import Optional

from segment_anything.modeling.image_encoder import (
    Block as EncoderBlock,
    ImageEncoderViT,
    window_partition,
    window_unpartition,
)
from profiler import profile, ProfilerStats, get_profiler


class BlockProfiler(EncoderBlock):
    """
    Profiling wrapper for SAM encoder Block that measures attention and MLP latency.

    This class extends EncoderBlock to add precise timing measurements for:
    - Attention layer (including normalization and window partitioning)
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
            Output tensor of shape (B, H, W, C)
        """
        profiler = self.profiler_stats if self.profiler_stats else get_profiler()

        shortcut = x

        # Profile attention (including norm1 and window operations)
        with profile(f"block_{self.block_idx:02d}_attention", profiler, self.sync_cuda):
            x = self.norm1(x)

            # Window partition
            if self.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x)

            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        # Profile MLP (including norm2)
        with profile(f"block_{self.block_idx:02d}_mlp", profiler, self.sync_cuda):
            x = x + self.mlp(self.norm2(x))

        return x


class ImageEncoderViTProfiler(ImageEncoderViT):
    """
    Profiling wrapper for ImageEncoderViT that measures per-block latencies.

    This class extends ImageEncoderViT to profile the entire encoder forward pass
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
            Output tensor and intermediate embeddings
        """
        profiler = self.profiler_stats if self.profiler_stats else get_profiler()

        # Profile patch embedding
        with profile("patch_embed", profiler, self.sync_cuda):
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed

        interm_embeddings = []

        # Profile each transformer block
        for idx, blk in enumerate(self.blocks):
            with profile(f"block_{idx:02d}_total", profiler, self.sync_cuda):
                x = blk(x)

            if blk.window_size == 0:
                interm_embeddings.append(x)

        # Profile neck
        with profile("neck", profiler, self.sync_cuda):
            x = self.neck(x.permute(0, 3, 1, 2))

        return x, interm_embeddings


def encoder_latency_monkey_patch(model, profiler_stats: Optional[ProfilerStats] = None,
                                  sync_cuda: bool = True):
    """
    Apply monkey-patching to SAM image encoder for latency profiling.

    This function replaces the encoder's Block and ImageEncoderViT classes with
    profiling versions that measure attention and MLP latencies.

    Args:
        model: SAM model to patch
        profiler_stats: ProfilerStats instance to use (defaults to global profiler)
        sync_cuda: Whether to synchronize CUDA before timing (recommended for GPU)

    Returns:
        ProfilerStats instance being used

    Example:
        >>> from segment_anything import sam_model_registry, SamPredictor
        >>> sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
        >>> predictor = SamPredictor(sam)
        >>> profiler = encoder_latency_monkey_patch(predictor.model)
        >>> # Run inference
        >>> predictor.set_image(image)
        >>> profiler.print_summary()
    """
    if profiler_stats is None:
        profiler_stats = ProfilerStats()

    block_idx = 0

    # Replace Block classes with profiling versions
    for _, module in model.named_modules():
        if isinstance(module, EncoderBlock) and not isinstance(module, BlockProfiler):
            # Monkey-patch to BlockProfiler
            module.__class__ = BlockProfiler
            module.profiler_stats = profiler_stats
            module.sync_cuda = sync_cuda
            module.block_idx = block_idx
            block_idx += 1

        elif isinstance(module, ImageEncoderViT) and not isinstance(module, ImageEncoderViTProfiler):
            # Monkey-patch to ImageEncoderViTProfiler
            module.__class__ = ImageEncoderViTProfiler
            module.profiler_stats = profiler_stats
            module.sync_cuda = sync_cuda

    return profiler_stats


def analyze_encoder_breakdown(profiler_stats: ProfilerStats, print_details: bool = True):
    """
    Analyze encoder profiling results and compute breakdown statistics.

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
        print("ENCODER LATENCY BREAKDOWN")
        print("="*80)
        print(f"Total encoder time:      {total_encoder_time*1000:8.2f} ms (100.0%)")
        print(f"  - Patch embedding:     {patch_embed_time*1000:8.2f} ms ({patch_embed_time/total_encoder_time*100:5.1f}%)")
        print(f"  - Transformer blocks:  {total_blocks_time*1000:8.2f} ms ({total_blocks_time/total_encoder_time*100:5.1f}%)")
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
