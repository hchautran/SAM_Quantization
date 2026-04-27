from typing import Optional

import timm
import torch
from piecewise_attn import piecewise_sparse_attention
from processors import get_encoder_processor as get_registered_encoder_processor
from timm.models.mvitv2 import (
    MultiScaleAttention,
    cal_rel_pos_type,
    reshape_post_pool,
    reshape_pre_pool,
)


SUPPORTED_PROCESSORS = {
    "BASE",
    "POSITIONAL_PRUNE",
    "POSITIONAL_QUANT",
    "HEAD_PRUNE",
    "SUB_IMAGE_PRUNE",
    "POSITIONAL_SPARSE",
    "POSITIONAL_SPARGE",
    "PIECE_WISE_ATTN",
    "MVITV2_PIECEWISE_ATTN",
}


def cal_rel_pos(
    q: torch.Tensor,
    has_cls_token: bool,
    q_size: list[int],
    k_size: list[int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
):
    sp_idx = 1 if has_cls_token else 0
    q_h, q_w = q_size
    k_h, k_w = k_size

    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h, device=q.device).unsqueeze(-1) * q_h_ratio
        - torch.arange(k_h, device=q.device).unsqueeze(0) * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w, device=q.device).unsqueeze(-1) * q_w_ratio
        - torch.arange(k_w, device=q.device).unsqueeze(0) * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    rel_h = rel_pos_h[dist_h.long()]
    rel_w = rel_pos_w[dist_w.long()]

    batch_size, num_heads, _, dim = q.shape
    r_q = q[:, :, sp_idx:].reshape(batch_size, num_heads, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, rel_h)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, rel_w)
    return rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2)


class Mvitv2PiecewiseAttnProcessor:
    def __init__(self):
        self.percent = 0.0
        self.percent_global = 0.0

    def set_params(self, args):
        self.percent = getattr(args, "percent", 0.0)
        self.percent_global = getattr(args, "percent_global", self.percent)

    def process(self, x: torch.Tensor, feat_size, module, module_name: str = None):
        B, N, _ = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if module.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, module.has_cls_token)
            q = module.pool_q(q)
            q, q_size = reshape_post_pool(q, module.num_heads, q_tok)
        else:
            q_size = feat_size
        if module.norm_q is not None:
            q = module.norm_q(q)

        if module.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, module.has_cls_token)
            k = module.pool_k(k)
            k, k_size = reshape_post_pool(k, module.num_heads, k_tok)
        else:
            k_size = feat_size
        if module.norm_k is not None:
            k = module.norm_k(k)

        if module.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, module.has_cls_token)
            v = module.pool_v(v)
            v, _ = reshape_post_pool(v, module.num_heads, v_tok)
        if module.norm_v is not None:
            v = module.norm_v(v)
        print("q_shape, k_shape, v_shape",q.shape, k.shape, v.shape)
        pos = cal_rel_pos(
            q,
            module.has_cls_token,
            q_size,
            k_size,
            module.rel_pos_h,
            module.rel_pos_w,
        )
        print("pos shape",pos.shape)
        print("q shape, k shape , v shape", q.shape, k.shape, v.shape)
        attn = (q * module.scale) @ k.transpose(-2, -1)
        if module.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(
                attn,
                q,
                module.has_cls_token,
                q_size,
                k_size,
                module.rel_pos_h,
                module.rel_pos_w,
            )
        attn = attn.softmax(dim=-1)
        x = attn @ v

        # topk = self.percent #if is_local else self.global_percent
        # pos = cal_rel_pos_type(
        #     q,
        #     module.has_cls_token,
        #     q_size,
        #     k_size,
        #     module.rel_pos_h,
        #     module.rel_pos_w,
        # )
        # pos.reshape(B,module.num_heads, q_size[0]*q_size[1], k_size[0]*k_size[1])

        # print(pos.shape)
        # x = piecewise_sparse_attention_pos(q, k, v, pos, density = topk)

        if module.residual_pooling:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, module.dim_out)
        x = module.proj(x)

        return x, q_size


class QuantizedAttention(timm.models.mvitv2.MultiScaleAttention):
    def set_processor(self, processor, module_name):
        self.processor = processor
        self.module_name = module_name

    def forward(self, x: torch.Tensor, feat_size: list[int]) -> torch.Tensor:
        return self.processor.process(x, feat_size, self, self.module_name)


def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=16,
    weight_quant="per_channel",
    act_quant="per_token",
    device="cuda",
):
    for name, module in model.named_modules():
        if isinstance(module, MultiScaleAttention):
            print("Monkey-patching Attention module:", name)
            module.__class__ = QuantizedAttention
            module.set_processor(processor, name)

    if n_bits < 16:
        from utils.quant_utils import QuantizationConfig, replace_linear_with_quantized

        config = QuantizationConfig(
            n_bits_w=n_bits,
            n_bits_a=n_bits,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_output=False,
        )
        replace_linear_with_quantized(module=model, config=config, module_name_to_exclude=["head"])

    model.to(device)


def get_encoder_processor(processor_name: Optional[str]):
    if processor_name is None or processor_name == "BASE":
        return None
    if processor_name == "MVITV2_PIECEWISE_ATTN":
        return Mvitv2PiecewiseAttnProcessor()
    return get_registered_encoder_processor(processor_name)
