from typing import Optional

import torch
import torch.nn.functional as F
from core.vision_encoder.pe import SelfAttention
from piecewise_attn import piecewise_sparse_attention
from processors import get_encoder_processor as get_registered_encoder_processor
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda


SUPPORTED_PROCESSORS = {
    "PIECEWISE",
    "SPARSEATTN",
    "SPARGEATTN",
}


def pe_attention_forward(x: torch.Tensor, module, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    embed_dim = x.shape[-1]
    proj = F.linear(x, module.in_proj_weight, module.in_proj_bias)
    proj = (
        proj.unflatten(-1, (3, embed_dim))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
        .contiguous()
    )
    q, k, v = proj[0], proj[1], proj[2]
    q = q.view(q.shape[0], q.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
    k = k.view(k.shape[0], k.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
    v = v.view(v.shape[0], v.shape[1], module.num_heads, module.head_dim).transpose(1, 2)

    if module.rope:
        q, k = module.rope(q, k)

    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=module.scale,
    )
    attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], embed_dim)
    return F.linear(attn, module.out_proj.weight, module.out_proj.bias)


class PEPiecewiseProcessor:
    def __init__(self):
        self.percent = 0.0

    def set_params(self, args):
        self.percent = getattr(args, "percent", 0.0)

    def process_pe(self, x: torch.Tensor, module, module_name: str, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask is not None:
            return pe_attention_forward(x, module, attn_mask=attn_mask)

        embed_dim = x.shape[-1]
        proj = F.linear(x, module.in_proj_weight, module.in_proj_bias)
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]
        q = q.view(q.shape[0], q.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], module.num_heads, module.head_dim).transpose(1, 2)

        if module.rope:
            q, k = module.rope(q, k)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        attn = piecewise_sparse_attention(q, k, v, density=self.percent, scale=module.scale)
        attn = attn.to(dtype=module.out_proj.weight.dtype)
        attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], embed_dim)
        return F.linear(attn, module.out_proj.weight, module.out_proj.bias)


class PESpargeProcessor(PEPiecewiseProcessor):
    def process_pe(self, x: torch.Tensor, module, module_name: str, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask is not None:
            return pe_attention_forward(x, module, attn_mask=attn_mask)

        embed_dim = x.shape[-1]
        proj = F.linear(x, module.in_proj_weight, module.in_proj_bias)
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]
        q = q.view(q.shape[0], q.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
        
        if module.rope:
            q, k = module.rope(q, k)

        if q.shape[-2] <= 128:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=module.scale,
            )
            attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], embed_dim)
            return F.linear(attn, module.out_proj.weight, module.out_proj.bias)
        else :
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

            attn = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=self.percent, scale=module.scale)
            attn = attn.to(dtype=module.out_proj.weight.dtype)
            attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], embed_dim)
            return F.linear(attn, module.out_proj.weight, module.out_proj.bias)


class QuantizedAttention(SelfAttention):
    def set_processor(self, processor, module_name):
        self.processor = processor
        self.module_name = module_name

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.processor is None:
            return pe_attention_forward(x, self, attn_mask=attn_mask)
        if hasattr(self.processor, "process_pe"):
            return self.processor.process_pe(x, self, self.module_name, attn_mask=attn_mask)
        if hasattr(self.processor, "process_sequence"):
            return self.processor.process_sequence(x, self, self.module_name, attn_mask=attn_mask)
        raise NotImplementedError(
            f"Processor {type(self.processor).__name__} does not implement `process_pe` or `process_sequence` for PE SelfAttention."
        )


def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=16,
    weight_quant="per_channel",
    act_quant="per_token",
    device="cuda",
):
    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            print(f"Monkey patching {name} with processor {type(processor).__name__} .")
            module.__class__ = QuantizedAttention
            module.set_processor(processor, name)
    model.to(device)


def get_encoder_processor(processor_name: Optional[str]):
    if processor_name is None or processor_name == "BASE":
        return None
    if processor_name == "PIECEWISE":
        return PEPiecewiseProcessor()
    if processor_name in {"SPARSEATTN", "SPARGEATTN"}:
        return PESpargeProcessor()
    return get_registered_encoder_processor(processor_name)
