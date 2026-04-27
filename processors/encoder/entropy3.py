from .entropy import BaseEntropyProcessor, PositionalPruneProcessor
from collections import defaultdict
import torch
import math
from spas_sage_attn import (
    spas_sage2_attn_meansim_topk_cuda,
    spas_sage2_attn_meansim_topk_cuda_pos,
)
from piecewise_attn import piecewise_sparse_attention_pos, piecewise_sparse_attention
from  timm.models.mvitv2 import (reshape_pre_pool, reshape_post_pool,cal_rel_pos_type)


def cal_rel_pos(
        q: torch.Tensor,
        has_cls_token: bool,
        q_size: list[int],
        k_size: list[int],
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_token else 0
    q_h, q_w = q_size
    k_h, k_w = k_size

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
            torch.arange(q_h, device=q.device).unsqueeze(-1) * q_h_ratio -
            torch.arange(k_h, device=q.device).unsqueeze(0) * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
            torch.arange(q_w, device=q.device).unsqueeze(-1) * q_w_ratio -
            torch.arange(k_w, device=q.device).unsqueeze(0) * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    rel_h = rel_pos_h[dist_h.long()]
    rel_w = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, rel_h)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, rel_w)

    pos= (rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2)
    ) #.view(B, -1, q_h * q_w, k_h * k_w)

    return pos

class Mvitv2PiecewiseAttnProcessor(PositionalPruneProcessor):
    def __init__(self, strategy_name: str = "Mvitv2PiecewiseAttnProcessor"):
        super().__init__(strategy_name)
        self.global_percent = 0.5

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


