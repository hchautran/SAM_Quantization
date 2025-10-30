"""
Memory profiling utilities for SAM encoder attention and MLP layers.

This module provides classes and utilities to profile the memory footprint of
attention and MLP components in the SAM Vision Transformer encoder.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import gc

from segment_anything.modeling.image_encoder import (
    Block as EncoderBlock,
    ImageEncoderViT,
    Attention as EncoderAttention,
    window_partition,
    window_unpartition,
)


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
_global_memory_profiler = MemoryStats()


def get_memory_profiler() -> MemoryStats:
    """Get the global memory profiler instance."""
    return _global_memory_profiler


class AttentionMemoryProfiler(EncoderAttention):
    """
    Memory profiling wrapper for Attention layer.

    Tracks activation memory and peak memory during forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_idx = 0
        self.memory_stats = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_profiler(self, memory_stats: MemoryStats, block_idx: int):
        """Configure the memory profiler."""
        self.memory_stats = memory_stats
        self.block_idx = block_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory profiling."""
        memory_stats = self.memory_stats if self.memory_stats else get_memory_profiler()

        if not memory_stats.enabled:
            return super().forward(x)

        B, H, W, _ = x.shape

        # Record memory before attention
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before = memory_stats.get_current_memory()

        # Run attention
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        # Record memory after attention
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after = memory_stats.get_current_memory()
        mem_delta = mem_after - mem_before

        # Record activation memory
        memory_stats.record_activation(f"block_{self.block_idx:02d}_attention", mem_delta)
        memory_stats.record_peak(f"block_{self.block_idx:02d}_attention", mem_after)

        return x


class MLPMemoryProfiler(nn.Module):
    """
    Memory profiling wrapper for MLP layer.

    Tracks activation memory and peak memory during forward pass.
    """

    def __init__(self, mlp_module: nn.Module, block_idx: int, memory_stats: Optional[MemoryStats] = None):
        super().__init__()
        self.mlp = mlp_module
        self.block_idx = block_idx
        self.memory_stats = memory_stats
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory profiling."""
        memory_stats = self.memory_stats if self.memory_stats else get_memory_profiler()

        if not memory_stats.enabled:
            return self.mlp(x)

        # Record memory before MLP
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before = memory_stats.get_current_memory()

        # Run MLP
        output = self.mlp(x)

        # Record memory after MLP
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after = memory_stats.get_current_memory()
        mem_delta = mem_after - mem_before

        # Record activation memory
        memory_stats.record_activation(f"block_{self.block_idx:02d}_mlp", mem_delta)
        memory_stats.record_peak(f"block_{self.block_idx:02d}_mlp", mem_after)

        return output


class BlockMemoryProfiler(EncoderBlock):
    """
    Memory profiling wrapper for encoder Block.

    Tracks memory usage for attention and MLP components separately.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_idx = 0
        self.memory_stats = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_profiler(self, memory_stats: MemoryStats, block_idx: int):
        """Configure the memory profiler."""
        self.memory_stats = memory_stats
        self.block_idx = block_idx

        # Wrap attention and MLP with memory profilers
        if not isinstance(self.attn, AttentionMemoryProfiler):
            self.attn.__class__ = AttentionMemoryProfiler
            self.attn.memory_stats = memory_stats
            self.attn.block_idx = block_idx
            self.attn.device = self.device

        # Wrap MLP
        if not isinstance(self.mlp, MLPMemoryProfiler):
            original_mlp = self.mlp
            self.mlp = MLPMemoryProfiler(original_mlp, block_idx, memory_stats)
            self.mlp.device = self.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory profiling."""
        memory_stats = self.memory_stats if self.memory_stats else get_memory_profiler()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before_block = memory_stats.get_current_memory()

        shortcut = x
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
        x = x + self.mlp(self.norm2(x))

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after_block = memory_stats.get_current_memory()

        memory_stats.record_peak(f"block_{self.block_idx:02d}_total", mem_after_block)

        return x


class ImageEncoderViTMemoryProfiler(ImageEncoderViT):
    """
    Memory profiling wrapper for ImageEncoderViT.

    Tracks overall encoder memory usage and per-block breakdowns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_stats = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_profiler(self, memory_stats: MemoryStats):
        """Configure the memory profiler."""
        self.memory_stats = memory_stats

    def forward(self, x: torch.Tensor):
        """Forward pass with memory profiling."""
        memory_stats = self.memory_stats if self.memory_stats else get_memory_profiler()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before = memory_stats.get_current_memory()

        # Patch embedding
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after_patch = memory_stats.get_current_memory()
        memory_stats.record_activation("patch_embed", mem_after_patch - mem_before)

        interm_embeddings = []

        # Transformer blocks
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        # Neck
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_before_neck = memory_stats.get_current_memory()

        x = self.neck(x.permute(0, 3, 1, 2))

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        mem_after_neck = memory_stats.get_current_memory()
        memory_stats.record_activation("neck", mem_after_neck - mem_before_neck)

        return x, interm_embeddings


def encoder_memory_monkey_patch(model, memory_stats: Optional[MemoryStats] = None,
                                 device: str = "cuda") -> MemoryStats:
    """
    Apply monkey-patching to SAM image encoder for memory profiling.

    Args:
        model: SAM model to patch
        memory_stats: MemoryStats instance to use
        device: Device being used

    Returns:
        MemoryStats instance being used
    """
    if memory_stats is None:
        memory_stats = MemoryStats(device=device)

    block_idx = 0

    # Replace Block classes with memory profiling versions
    for name, module in model.named_modules():
        if isinstance(module, EncoderBlock) and not isinstance(module, BlockMemoryProfiler):
            # Compute parameter memory for this block's components
            memory_stats.compute_parameter_memory(module, f"block_{block_idx:02d}_params")

            # Compute attention and MLP memory separately
            memory_stats.compute_parameter_memory(module.attn, f"block_{block_idx:02d}_attention_params")
            memory_stats.compute_parameter_memory(module.mlp, f"block_{block_idx:02d}_mlp_params")

            # Monkey-patch to BlockMemoryProfiler
            module.__class__ = BlockMemoryProfiler
            module.memory_stats = memory_stats
            module.device = device
            module.block_idx = block_idx
            module.set_profiler(memory_stats, block_idx)
            block_idx += 1

        elif isinstance(module, ImageEncoderViT) and not isinstance(module, ImageEncoderViTMemoryProfiler):
            # Compute parameter memory for encoder components
            memory_stats.compute_parameter_memory(module.patch_embed, "patch_embed_params")
            memory_stats.compute_parameter_memory(module.neck, "neck_params")

            # Monkey-patch to ImageEncoderViTMemoryProfiler
            module.__class__ = ImageEncoderViTMemoryProfiler
            module.memory_stats = memory_stats
            module.device = device

    return memory_stats


def analyze_memory_breakdown(memory_stats: MemoryStats, print_details: bool = True) -> Dict:
    """
    Analyze memory profiling results and compute breakdown statistics.

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
        if 'attention' in name:
            attention_activation.append(mem_dict['mean'])
        elif 'mlp' in name:
            mlp_activation.append(mem_dict['mean'])

    total_attention_params = sum(attention_params)
    total_mlp_params = sum(mlp_params)
    total_params = sum(stats_summary['parameter_memory'].values())

    avg_attention_activation = sum(attention_activation) / len(attention_activation) if attention_activation else 0
    avg_mlp_activation = sum(mlp_activation) / len(mlp_activation) if mlp_activation else 0

    results = {
        'total_parameter_memory': total_params,
        'attention_parameter_memory': total_attention_params,
        'mlp_parameter_memory': total_mlp_params,
        'avg_attention_activation': avg_attention_activation,
        'avg_mlp_activation': avg_mlp_activation,
        'num_blocks': len([k for k in stats_summary['parameter_memory'].keys() if 'block_' in k]),
        'parameter_memory_dict': stats_summary['parameter_memory'],
        'activation_memory_dict': stats_summary['activation_memory'],
        'peak_memory_dict': stats_summary['peak_memory'],
    }

    if print_details:
        print("\n" + "="*80)
        print("ENCODER MEMORY BREAKDOWN")
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
        print(f"{'Block':<10} {'Params (MB)':<15}")
        print("-"*80)

        block_params = [(k, v) for k, v in stats_summary['parameter_memory'].items() if 'block_' in k and 'params' in k]
        block_params.sort(key=lambda x: x[0])

        for name, mem in block_params:
            print(f"{name:<10} {mem:8.2f}")

        print("="*80 + "\n")

    return results
