import math
from typing import Optional

import torch
import torch.nn.functional as F
from piecewise_attn import piecewise_sparse_attention
from processors import get_encoder_processor as get_registered_encoder_processor
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
from timm.models.hiera import MaskUnitAttention


SUPPORTED_PROCESSORS = {"BASE", "POSITIONAL_PRUNE", "HEAD_PRUNE", "PIECEWISE", "SPARGEATTN"}


class HieraPieceWiseProcessor:
    def __init__(self):
        self.percent = 0.0

    def set_params(self, args):
        self.percent = getattr(args, "percent", 0.0)

    def process(self, x: torch.Tensor, module: MaskUnitAttention, module_name: str) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        num_windows = (num_tokens // (module.q_stride * module.window_size)) if module.use_mask_unit_attn else 1
        qkv = module.qkv(x).reshape(batch_size, -1, num_windows, 3, module.heads, module.head_dim).permute(3, 0, 4, 2, 1, 5)
        q, k, v = qkv.unbind(0)

        if module.q_stride > 1:
            q = q.view(batch_size, module.heads, num_windows, module.q_stride, -1, module.head_dim).amax(dim=3)

        q_tokens = q.shape[-2]
        kv_tokens = k.shape[-2]
        q = q.permute(0, 2, 1, 3, 4).reshape(batch_size * num_windows, module.heads, q_tokens, module.head_dim).contiguous()
        k = k.permute(0, 2, 1, 3, 4).reshape(batch_size * num_windows, module.heads, kv_tokens, module.head_dim).contiguous()
        v = v.permute(0, 2, 1, 3, 4).reshape(batch_size * num_windows, module.heads, kv_tokens, module.head_dim).contiguous()

    
        x = piecewise_sparse_attention(q, k, v, density=self.percent, scale=module.scale)
        x = x.reshape(batch_size, num_windows, module.heads, q_tokens, module.head_dim).permute(0, 2, 1, 3, 4)
        x = x.transpose(1, 3).reshape(batch_size, -1, module.dim_out)
        x = module.proj(x)
        return x


class HieraSpargeProcessor(HieraPieceWiseProcessor):
    def process(self, x: torch.Tensor, module: MaskUnitAttention, module_name: str) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        num_windows = (num_tokens // (module.q_stride * module.window_size)) if module.use_mask_unit_attn else 1
        qkv = module.qkv(x).reshape(batch_size, -1, num_windows, 3, module.heads, module.head_dim).permute(3, 0, 4, 2, 1, 5)
        q, k, v = qkv.unbind(0)

        if module.q_stride > 1:
            q = q.view(batch_size, module.heads, num_windows, module.q_stride, -1, module.head_dim).amax(dim=3)

        q_tokens = q.shape[-2]
        kv_tokens = k.shape[-2]
        if q_tokens <= 128:
            if module.fused_attn:
                x = F.scaled_dot_product_attention(q, k, v)
            else:
                attn = (q * module.scale) @ k.transpose(-1, -2)
                attn = attn.softmax(dim=-1)
                x = attn @ v
        else:
            q = q.permute(0, 2, 1, 3, 4).reshape(batch_size * num_windows, module.heads, q_tokens, module.head_dim).contiguous()
            k = k.permute(0, 2, 1, 3, 4).reshape(batch_size * num_windows, module.heads, kv_tokens, module.head_dim).contiguous()
            v = v.permute(0, 2, 1, 3, 4).reshape(batch_size * num_windows, module.heads, kv_tokens, module.head_dim).contiguous()
            x = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=self.percent, scale=module.scale)
            x = x.reshape(batch_size, num_windows, module.heads, q_tokens, module.head_dim).permute(0, 2, 1, 3, 4)
            x = x.to(dtype=module.proj.weight.dtype)

        x = x.transpose(1, 3).reshape(batch_size, -1, module.dim_out)
        x = module.proj(x)
        return x


class ProcessorMaskUnitAttention(MaskUnitAttention):
    def set_processor(self, processor, module_name: str):
        self.processor = processor
        self.module_name = module_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor.process(x, self, self.module_name)


def image_encoder_monkey_patch(model, processor):
    patched_count = 0
    skipped_count = 0
    for name, module in model.named_modules():
        if isinstance(module, MaskUnitAttention):
            if int(getattr(module, "q_stride", 1)) != 1:
                skipped_count += 1
                continue
            print(f"Patching module {name} ")
            module.__class__ = ProcessorMaskUnitAttention
            module.set_processor(processor, name)
            patched_count += 1
    print(f"Patched {patched_count} MaskUnitAttention modules and skipped {skipped_count} downsampling modules")


def get_encoder_processor(processor_name: Optional[str]):
    if processor_name is None or processor_name == "BASE":
        return None
    if processor_name == "PIECEWISE":
        return HieraPieceWiseProcessor()
    if processor_name == "SPARGEATTN":
        return HieraSpargeProcessor()
    return get_registered_encoder_processor(processor_name)
