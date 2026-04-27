from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Set

import torch

from segment_anything.modeling.image_encoder import (
    Block as EncoderBlock,
    window_partition,
    window_unpartition,
    Attention as SAMAttention,
    add_decomposed_rel_pos,
)

try:
    from token_merging.merge import grad_bipartite_soft_matching
except ImportError:
    from merge import grad_bipartite_soft_matching


class BaseMergeProcessor(ABC):
    """Block-level processor interface for SAM encoder token merging."""

    def __init__(self, layers: Optional[Sequence[int]] = None) -> None:
        self.layers: Optional[Set[int]] = set(layers) if layers else None

    def _extract_block_index(self, module_name: Optional[str]) -> Optional[int]:
        if not module_name:
            return None
        parts = module_name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == "blocks" and parts[idx + 1].isdigit():
                return int(parts[idx + 1])
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        return None

    def should_process(self, module_name: Optional[str]) -> bool:
        if self.layers is None:
            return True
        block_idx = self._extract_block_index(module_name)
        return block_idx in self.layers

    @abstractmethod
    def process(self, x: torch.Tensor, module: EncoderBlock, module_name: Optional[str] = None) -> torch.Tensor:
        shortcut = x
        x = module.norm1(x)
        # Window partition
        if module.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, module.window_size)

        x = module.attn(x)
        # Reverse window partition
        if module.window_size > 0:
            x = window_unpartition(x, module.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + module.mlp(module.norm2(x))

        return x


class GradBipartiteMergeProcessor(BaseMergeProcessor):
    """Apply grad_bipartite_soft_matching inside encoder blocks."""

    def __init__(
        self,
        r: int,
        sx: int = 2,
        sy: int = 2,
        grad_method: str = "sobel",
        layers: Optional[Sequence[int]] = None,
        merge_mlp: bool = True,
    ) -> None:
        super().__init__(layers=layers)
        self.r = r
        self.sx = sx
        self.sy = sy
        self.grad_method = grad_method
        self.merge_mlp = merge_mlp

    def _run_original_block(self, x: torch.Tensor, module: EncoderBlock) -> torch.Tensor:
        shortcut = x
        x = module.norm1(x)
        if module.window_size > 0:
            height, width = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, module.window_size)

        x = module.attn(x)

        if module.window_size > 0:
            x = window_unpartition(x, module.window_size, pad_hw, (height, width))

        x = shortcut + x
        x = x + module.mlp(module.norm2(x))
        return x

    def _build_merge_ops(self, reference_x: torch.Tensor):
        batch, height, width, channels = reference_x.shape
        if self.r <= 0 or height * width <= 1:
            return None
        if height % self.sy != 0 or width % self.sx != 0:
            return None
        tokens = reference_x.reshape(batch, height * width, channels)
        return grad_bipartite_soft_matching(
            metric=tokens,
            H=height,
            W=width,
            sx=self.sx,
            sy=self.sy,
            grad_method=self.grad_method,
            r=self.r,
        )

    def _apply_merge_ops(self, x: torch.Tensor, op, merge, unmerge) -> torch.Tensor:
        batch, height, width, channels = x.shape
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # end2 = torch.cuda.Event(enable_timing=True)
        # start.record()
        # breakpoint()
        tokens = x.reshape(batch, height * width, channels)
        merged_tokens, _ = merge(tokens)
        # end.record()
        # end.synchronize()
        # elapsed_ms = start.elapsed_time(end)
        merged_grid = merged_tokens.view(batch, merged_tokens.shape[1], 1, channels)
        merged_out = op(merged_grid)
        restored = unmerge(merged_out.view(batch, -1, channels))
        return restored.view(batch, height, width, channels)

    def do_merge_op(self, x: torch.Tensor, merge) -> torch.Tensor:
        batch, height, width, channels = x.shape
        tokens = x.reshape(batch, height * width, channels)
        merged_tokens, _ = merge(tokens)
        return merged_tokens
    def do_unmerge_op(self, merged_grid: torch.Tensor, unmerge, batch: int, height: int, width: int, channels: int) -> torch.Tensor:
        
        restored = unmerge(merged_grid)
        return restored.view(batch, height, width, channels)
    def _merge_apply_unmerge(self, x: torch.Tensor, op) -> torch.Tensor:
        merge_ops = self._build_merge_ops(x)
        if merge_ops is None:
            return op(x)
        merge, unmerge = merge_ops
        return self._apply_merge_ops(x, op, merge, unmerge)

    def process(self, x: torch.Tensor, module: EncoderBlock, module_name: Optional[str] = None) -> torch.Tensor:
        if not self.should_process(module_name):
            return self._run_original_block(x, module)

        shortcut = x
        x = module.norm1(x)

        if module.window_size > 0:
            height, width = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, module.window_size)

        # x = self._merge_apply_unmerge(x, module.attn)
        x = module.attn(x)

        if module.window_size > 0:
            x = window_unpartition(x, module.window_size, pad_hw, (height, width))

        x = shortcut + x
        # x = self._merge_apply_unmerge(x, module.mlp)

        norm2_x = module.norm2(x)
        if self.merge_mlp:
            mlp_out = module.mlp(norm2_x)
            # mlp_out = self._merge_apply_unmerge(norm2_x, module.mlp)
        else:
            mlp_out = module.mlp(norm2_x)
        x = x + mlp_out
        return x


class IndAttnToMlp(GradBipartiteMergeProcessor):
    """Merge MLP inputs using merge indices computed from the attention input."""

    def process(self, x: torch.Tensor, module: EncoderBlock, module_name: Optional[str] = None) -> torch.Tensor:
        if not self.should_process(module_name):
            return self._run_original_block(x, module)

        shortcut = x
        attn_input = module.norm1(x)

        # Compute merge ops on the unpartitioned tensor so indices live in (B, H, W)
        # space and can be reused after the attention residual.
        attn_merge_ops = self._build_merge_ops(attn_input)

        # Attention runs as usual (with window partition if needed).
        if module.window_size > 0:
            height, width = attn_input.shape[1], attn_input.shape[2]
            attn_windows, pad_hw = window_partition(attn_input, module.window_size)
            attn_out = module.attn(attn_windows)
            attn_out = window_unpartition(attn_out, module.window_size, pad_hw, (height, width))
        else:
            attn_out = module.attn(attn_input)

        # First residual: the MLP input is spatially aligned with `attn_input`.
        x = shortcut + attn_out

        norm2_x = module.norm2(x)

        # Apply the attention-derived merge indices to the MLP input (post residual),
        # without window partitioning.
        if (not self.merge_mlp) or (attn_merge_ops is None):
            mlp_out = module.mlp(norm2_x)
        else:
            merge, unmerge = attn_merge_ops
            mlp_out = self._apply_merge_ops(norm2_x, module.mlp, merge, unmerge)

        x = x + mlp_out
        return x

class ReUseMergeOps(GradBipartiteMergeProcessor):
    """Merge MLP inputs using merge indices computed from the attention input."""
    def take_merge_ops(self, input):
        self.merge_ops, self.unmerge_ops= self._build_merge_ops(input)

    def process(self, x: torch.Tensor, module: EncoderBlock, module_name: Optional[str] = None) -> torch.Tensor:
        if not self.should_process(module_name):
            return self._run_original_block(x, module)
        B, H, W, C = x.shape
        shortcut = x
        attn_input = module.norm1(x)

        # Compute merge ops on the unpartitioned tensor so indices live in (B, H, W)
        # space and can be reused after the attention residual.

        # Attention runs as usual (with window partition if needed).
        if module.window_size > 0:
            height, width = attn_input.shape[1], attn_input.shape[2]
            attn_windows, pad_hw = window_partition(attn_input, module.window_size)
            attn_out = module.attn(attn_windows)
            attn_out = window_unpartition(attn_out, module.window_size, pad_hw, (height, width))
        else:
            attn_out = module.attn(attn_input)

        # First residual: the MLP input is spatially aligned with `attn_input`.
        x = shortcut + attn_out
        

        # norm2_x = module.norm2(x)

        # # Apply the attention-derived merge indices to the MLP input (post residual),
        # # without window partitioning.
        # if (not self.merge_mlp):
        #     mlp_out = module.mlp(norm2_x)
        # else:
        #     if "blocks.0" in module_name:
        #         self.take_merge_ops(norm2_x)
        #     mlp_out = self._apply_merge_ops(norm2_x, module.mlp, self.merge_ops, self.unmerge_ops)

        # x = x + mlp_out
        # return x
        if "blocks.0" in module_name:
            self.take_merge_ops(x)
        merged_x = self.do_merge_op(x, self.merge_ops) 
        norm2_x = module.norm2(merged_x)
        
        
        # Apply the attention-derived merge indices to the MLP input (post residual),
        # without window partitioning.
        if (not self.merge_mlp):
            mlp_out = module.mlp(norm2_x)
        else:
            mlp_out = module.mlp(norm2_x)
            mlp_out = self.do_unmerge_op(mlp_out, self.unmerge_ops, B, H, W, C) 

        x = x + mlp_out
        return x
class ReUseMergeOpsAttn(GradBipartiteMergeProcessor):
    """Merge MLP inputs using merge indices computed from the attention input."""
    def take_merge_ops(self, input):
        self.merge_ops, self.unmerge_ops= self._build_merge_ops(input)

    def process(self, x: torch.Tensor, module: EncoderBlock, module_name: Optional[str] = None) -> torch.Tensor:
        if not self.should_process(module_name):
            return self._run_original_block(x, module)
        B, H, W, C = x.shape
        shortcut = x
        attn_input = module.norm1(x)

        # Compute merge ops on the unpartitioned tensor so indices live in (B, H, W)
        # space and can be reused after the attention residual.

        # Attention runs as usual (with window partition if needed).
        if module.window_size > 0:
            height, width = attn_input.shape[1], attn_input.shape[2]
            attn_windows, pad_hw = window_partition(attn_input, module.window_size)
            print(attn_windows.shape)
            if "blocks.0" in module_name:
                self.take_merge_ops(attn_windows)
            attn_windows_merged = self.do_merge_op(attn_windows, self.merge_ops)
            print(attn_windows_merged.shape)
            attn_out = module.attn(attn_windows_merged)
            attn_out = self.do_unmerge_op(attn_out, self.unmerge_ops, B, H, W, C)
            attn_out = window_unpartition(attn_out, module.window_size, pad_hw, (height, width))
        else:
            attn_out = module.attn(attn_input)

        # First residual: the MLP input is spatially aligned with `attn_input`.
        x = shortcut + attn_out
        
        norm2_x = module.norm2(x)
       
        # Apply the attention-derived merge indices to the MLP input (post residual),
        # without window partitioning.
        if (not self.merge_mlp):
            mlp_out = module.mlp(norm2_x)
        else:
            mlp_out = module.mlp(norm2_x)
            # mlp_out = self.do_unmerge_op(mlp_out, self.unmerge_ops, B, H, W, C) 

        x = x + mlp_out
        return x
class Attn(SAMAttention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, merged_dim , _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
class MergeBlock(EncoderBlock):
    """Encoder block wrapper delegating execution to a block processor."""

    def set_processor(self, processor: BaseMergeProcessor, module_name: str) -> None:
        self.processor = processor
        self.module_name = module_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor.process(x, self, self.module_name)



def monkey_patch_merge_blocks(model: torch.nn.Module, processor: BaseMergeProcessor) -> torch.nn.Module:
    """Replace SAM encoder blocks with MergeBlock wrappers."""
    for name, module in model.named_modules():
        if isinstance(module, EncoderBlock):
            module.__class__ = MergeBlock
            module.set_processor(processor, name)
    return model
