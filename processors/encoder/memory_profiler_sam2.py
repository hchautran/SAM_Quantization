"""
Memory profiling utilities for SAM2 encoder attention and MLP layers.

This module provides classes and utilities to profile the memory footprint of
attention and MLP components in the SAM2 Hiera encoder.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from collections import defaultdict
import gc

from sam2.modeling.backbones.hieradet import MultiScaleBlock, Hiera


class MemoryStats:
    """Track memory statistics for model components."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.peak_memory = defaultdict(float)  # Peak memory per component
        self.activation_memory = defaultdict(list)  # Activation memory per forward pass
        self.parameter_memory = {}  # Parameter memory per module
        self.enabled = True

    def reset_peak_stats(self):
        """Reset peak memory tracking."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_current_memory(self) -> float:
        """Get current allocated memory in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            # For CPU, we'd need psutil or similar
            return 0.0

    def get_peak_memory(self) -> float:
        """Get peak allocated memory in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        else:
            return 0.0

    def record_activation(self, name: str, memory_mb: float):
        """Record activation memory for a component."""
        if self.enabled:
            self.activation_memory[name].append(memory_mb)

    def record_peak(self, name: str, memory_mb: float):
        """Record peak memory for a component."""
        if self.enabled:
            self.peak_memory[name] = max(self.peak_memory[name], memory_mb)

    def compute_parameter_memory(self, module: nn.Module, name: str):
        """Compute and store parameter memory for a module."""
        param_memory = 0
        for param in module.parameters():
            param_memory += param.nelement() * param.element_size()

        buffer_memory = 0
        for buffer in module.buffers():
            buffer_memory += buffer.nelement() * buffer.element_size()

        total_memory_mb = (param_memory + buffer_memory) / 1024**2
        self.parameter_memory[name] = total_memory_mb

    def get_stats_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {
            'parameter_memory': dict(self.parameter_memory),
            'peak_memory': dict(self.peak_memory),
            'activation_memory': {},
        }

        for name, mem_list in self.activation_memory.items():
            if mem_list:
                summary['activation_memory'][name] = {
                    'mean': sum(mem_list) / len(mem_list),
                    'max': max(mem_list),
                    'min': min(mem_list),
                    'count': len(mem_list),
                }

        return summary

    def clear(self):
        """Clear all recorded statistics."""
        self.peak_memory.clear()
        self.activation_memory.clear()

    def enable(self):
        """Enable memory tracking."""
        self.enabled = True

    def disable(self):
        """Disable memory tracking."""
        self.enabled = False


# Global memory profiler instance
_global_memory_profiler = None


def get_memory_profiler() -> MemoryStats:
    """Get or create the global memory profiler instance."""
    global _global_memory_profiler
    if _global_memory_profiler is None:
        _global_memory_profiler = MemoryStats()
    return _global_memory_profiler


class HieraBlockMemoryProfiler(MultiScaleBlock):
    """
    Memory profiling wrapper for SAM2 HieraBlock that measures attention and MLP memory.

    This class extends MultiScaleBlock to add memory usage measurements for:
    - Attention layer (including normalization)
    - MLP layer (including normalization)

    Memory stats are recorded to a MemoryStats instance for analysis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_idx = 0
        self.memory_stats = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_profiler(self, memory_stats: MemoryStats, block_idx: int):
        """
        Configure the profiler for this block.

        Args:
            memory_stats: MemoryStats instance to record memory usage
            block_idx: Index of this block in the encoder
        """
        self.memory_stats = memory_stats
        self.block_idx = block_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory profiling.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Output tensor of shape (B, H', W', C') (may be downsampled)
        """
        memory_stats = self.memory_stats if self.memory_stats else get_memory_profiler()

        if not memory_stats.enabled:
            return super().forward(x)

        # Record memory before block
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before_block = memory_stats.get_current_memory()

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

        # Record memory before attention
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before_attn = memory_stats.get_current_memory()

        # Window Attention + Q Pooling (if stage change)
        x_attn = self.attn(x_norm)

        # Record memory after attention
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after_attn = memory_stats.get_current_memory()
        attn_mem_delta = mem_after_attn - mem_before_attn

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

        # Record memory before MLP
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before_mlp = memory_stats.get_current_memory()

        # MLP
        x_mlp = self.mlp(self.norm2(x))

        # Record memory after MLP
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after_mlp = memory_stats.get_current_memory()
        mlp_mem_delta = mem_after_mlp - mem_before_mlp

        x = x + self.drop_path(x_mlp)

        # Record final memory
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after_block = memory_stats.get_current_memory()

        # Record statistics
        memory_stats.record_activation(f"block_{self.block_idx:02d}_attention", attn_mem_delta)
        memory_stats.record_activation(f"block_{self.block_idx:02d}_mlp", mlp_mem_delta)
        memory_stats.record_peak(f"block_{self.block_idx:02d}_attention", mem_after_attn)
        memory_stats.record_peak(f"block_{self.block_idx:02d}_mlp", mem_after_mlp)
        memory_stats.record_peak(f"block_{self.block_idx:02d}_total", mem_after_block)

        return x


class HieraMemoryProfiler(Hiera):
    """
    Memory profiling wrapper for SAM2 Hiera backbone that measures per-block memory usage.

    This class extends Hiera to profile the entire encoder forward pass
    and collect per-block attention and MLP memory stats.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_stats = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_profiler(self, memory_stats: MemoryStats):
        """
        Configure the profiler for this encoder.

        Args:
            memory_stats: MemoryStats instance to record memory usage
        """
        self.memory_stats = memory_stats

    def forward(self, x: torch.Tensor):
        """
        Forward pass with memory profiling.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output features list (same format as original Hiera)
        """
        memory_stats = self.memory_stats if self.memory_stats else get_memory_profiler()

        # Record memory before patch embed
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before = memory_stats.get_current_memory()

        # Patch embedding
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        # Record memory after patch embed
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after_patch = memory_stats.get_current_memory()
        memory_stats.record_activation("patch_embed", mem_after_patch - mem_before)
        memory_stats.record_peak("patch_embed", mem_after_patch)

        outputs = []

        # Profile each block
        for idx, blk in enumerate(self.blocks):
            x = blk(x)

            # Collect intermediate features at stage boundaries
            if (idx == self.stage_ends[-1]) or (
                idx in self.stage_ends and self.return_interm_layers
            ):
                # Record memory before neck
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                mem_before_neck = memory_stats.get_current_memory()

                feats = x.permute(0, 3, 1, 2)

                # Record memory after neck
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                mem_after_neck = memory_stats.get_current_memory()
                memory_stats.record_activation("neck", mem_after_neck - mem_before_neck)
                memory_stats.record_peak("neck", mem_after_neck)

                outputs.append(feats)

        return outputs


def sam2_encoder_memory_monkey_patch(
    model,
    memory_stats: Optional[MemoryStats] = None,
    device: str = "cuda"
) -> MemoryStats:
    """
    Apply monkey-patching to SAM2 image encoder for memory profiling.

    This function replaces the encoder's MultiScaleBlock and Hiera classes with
    profiling versions that measure attention and MLP memory usage.

    Args:
        model: SAM2 model to patch
        memory_stats: MemoryStats instance to use (defaults to global profiler)
        device: Device being used

    Returns:
        MemoryStats instance being used

    Example:
        >>> from sam2.build_sam import build_sam2
        >>> from sam2.sam2_image_predictor import SAM2ImagePredictor
        >>> sam2_model = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt")
        >>> predictor = SAM2ImagePredictor(sam2_model)
        >>> memory_stats = sam2_encoder_memory_monkey_patch(predictor.model)
        >>> # Run inference
        >>> predictor.set_image(image)
        >>> # Analyze results
        >>> analyze_sam2_memory_breakdown(memory_stats)
    """
    if memory_stats is None:
        memory_stats = MemoryStats(device=device)

    block_idx = 0

    # Replace HieraBlock classes with memory profiling versions
    for module in model.modules():
        if isinstance(module, MultiScaleBlock) and not isinstance(module, HieraBlockMemoryProfiler):
            # Compute parameter memory for this block's components
            memory_stats.compute_parameter_memory(module, f"block_{block_idx:02d}_params")

            # Compute attention and MLP memory separately
            memory_stats.compute_parameter_memory(module.attn, f"block_{block_idx:02d}_attention_params")
            memory_stats.compute_parameter_memory(module.mlp, f"block_{block_idx:02d}_mlp_params")

            # Monkey-patch to HieraBlockMemoryProfiler
            module.__class__ = HieraBlockMemoryProfiler
            module.memory_stats = memory_stats
            module.device = device
            module.block_idx = block_idx
            block_idx += 1

        elif isinstance(module, Hiera) and not isinstance(module, HieraMemoryProfiler):
            # Compute parameter memory for encoder components
            memory_stats.compute_parameter_memory(module.patch_embed, "patch_embed_params")

            # Monkey-patch to HieraMemoryProfiler
            module.__class__ = HieraMemoryProfiler
            module.memory_stats = memory_stats
            module.device = device

    return memory_stats


def analyze_sam2_memory_breakdown(memory_stats: MemoryStats, print_details: bool = True) -> Dict:
    """
    Analyze SAM2 encoder memory profiling results and compute breakdown statistics.

    Args:
        memory_stats: MemoryStats instance with recorded data
        print_details: Whether to print detailed analysis

    Returns:
        Dictionary with analysis results
    """
    stats_summary = memory_stats.get_stats_summary()

    # Separate attention and MLP data
    attention_params = []
    mlp_params = []
    attention_activation = []
    mlp_activation = []

    for name, mem in stats_summary['parameter_memory'].items():
        if 'attention_params' in name:
            attention_params.append(mem)
        elif 'mlp_params' in name:
            mlp_params.append(mem)

    for name, mem_dict in stats_summary['activation_memory'].items():
        if 'attention' in name and 'block_' in name:
            attention_activation.append(mem_dict['mean'])
        elif 'mlp' in name and 'block_' in name:
            mlp_activation.append(mem_dict['mean'])

    total_attention_params = sum(attention_params)
    total_mlp_params = sum(mlp_params)
    total_params = sum(stats_summary['parameter_memory'].values())

    avg_attention_activation = sum(attention_activation) / len(attention_activation) if attention_activation else 0
    avg_mlp_activation = sum(mlp_activation) / len(mlp_activation) if mlp_activation else 0

    num_blocks = len([k for k in stats_summary['parameter_memory'].keys() if 'block_' in k and '_params' in k and 'attention' not in k and 'mlp' not in k])

    results = {
        'total_parameter_memory': total_params,
        'attention_parameter_memory': total_attention_params,
        'mlp_parameter_memory': total_mlp_params,
        'avg_attention_activation': avg_attention_activation,
        'avg_mlp_activation': avg_mlp_activation,
        'num_blocks': num_blocks,
        'parameter_memory_dict': stats_summary['parameter_memory'],
        'activation_memory_dict': stats_summary['activation_memory'],
        'peak_memory_dict': stats_summary['peak_memory'],
    }

    if print_details:
        print("\n" + "="*80)
        print("SAM2 ENCODER MEMORY BREAKDOWN")
        print("="*80)
        print(f"Total parameter memory:  {total_params:8.2f} MB (100.0%)")
        print(f"  - Attention params:    {total_attention_params:8.2f} MB ({total_attention_params/total_params*100:5.1f}%)")
        print(f"  - MLP params:          {total_mlp_params:8.2f} MB ({total_mlp_params/total_params*100:5.1f}%)")
        print("="*80)
        print(f"\nAverage activation memory per forward pass:")
        print(f"  - Attention:           {avg_attention_activation:8.4f} MB")
        print(f"  - MLP:                 {avg_mlp_activation:8.4f} MB")
        print("="*80)

        # Per-block parameter breakdown
        print("\nPER-BLOCK PARAMETER MEMORY")
        print("="*80)
        print(f"{'Block':<15} {'Total (MB)':<15} {'Attention (MB)':<18} {'MLP (MB)':<15}")
        print("-"*80)

        for i in range(num_blocks):
            block_total = stats_summary['parameter_memory'].get(f"block_{i:02d}_params", 0)
            block_attn = stats_summary['parameter_memory'].get(f"block_{i:02d}_attention_params", 0)
            block_mlp = stats_summary['parameter_memory'].get(f"block_{i:02d}_mlp_params", 0)
            print(f"Block {i:02d}       {block_total:8.2f}        {block_attn:8.2f}            {block_mlp:8.2f}")

        print("="*80 + "\n")

    return results
