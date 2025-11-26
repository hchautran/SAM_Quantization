"""
Quantization utilities for SM80 SageAttention
"""

import torch
from typing import Optional, Tuple

from . import _fused


def per_block_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128,
    BLKK: int = 64,
    sm_scale: Optional[float] = None,
    tensor_layout: str = "HND"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize Q and K tensors with per-block INT8 quantization.

    Args:
        q: Query tensor [batch_size, num_qo_heads, qo_len, head_dim] or [batch_size, qo_len, num_qo_heads, head_dim]
        k: Key tensor [batch_size, num_kv_heads, kv_len, head_dim] or [batch_size, kv_len, num_kv_heads, head_dim]
        km: Optional mean of k along sequence dimension
        BLKQ: Block size for Q quantization (default: 128)
        BLKK: Block size for K quantization (default: 64)
        sm_scale: Softmax scale factor (default: head_dim**-0.5)
        tensor_layout: "HND" or "NHD" (default: "HND")

    Returns:
        Tuple of (q_int8, q_scale, k_int8, k_scale)
    """
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    sm_scale *= 1.44269504

    _fused.quant_per_block_int8_cuda(q, q_int8, q_scale, sm_scale, BLKQ, _tensor_layout)
    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale


def per_warp_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128,
    WARPQ: int = 32,
    BLKK: int = 64,
    tensor_layout: str = "HND"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize Q with per-warp INT8 and K with per-block INT8 quantization.

    Args:
        q: Query tensor
        k: Key tensor
        km: Optional mean of k
        BLKQ: Block size for Q (default: 128)
        WARPQ: Warp size for Q (default: 32)
        BLKK: Block size for K (default: 64)
        tensor_layout: "HND" or "NHD"

    Returns:
        Tuple of (q_int8, q_scale, k_int8, k_scale)
    """
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, _tensor_layout)

    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale


def sub_mean(
    v: torch.Tensor,
    tensor_layout: str = "HND"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subtract mean from V tensor along sequence dimension.

    Args:
        v: Value tensor
        tensor_layout: "HND" or "NHD"

    Returns:
        Tuple of (v_smoothed, v_mean)
    """
    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    vm = v.mean(dim=1 if _tensor_layout == 0 else 2)

    v_smoothed = torch.empty(v.shape, dtype=torch.float16, device=v.device)
    _fused.sub_mean_cuda(v, vm, v_smoothed, _tensor_layout)

    return v_smoothed, vm
