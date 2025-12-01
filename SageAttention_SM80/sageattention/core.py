"""
Core SageAttention SM80 implementation
"""

import torch
from typing import Optional, Tuple
import warnings

try:
    from . import sm80_compile
    SM80_ENABLED = True
except:
    SM80_ENABLED = False
    warnings.warn("SM80 kernels not available. Please build the package.")

from .quant import per_block_int8, per_warp_int8, sub_mean


def sageattn_sm80(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    smooth_v: bool = False,
    qk_quant_gran: str = "per_warp",
) -> torch.Tensor:
    """
    SageAttention for SM80 (Ampere) GPUs - Optimized for Vision Models (SAM).

    Args:
        q: Query tensor, shape [B, H, N, D] (HND) or [B, N, H, D] (NHD)
        k: Key tensor, shape [B, H, N, D] (HND) or [B, N, H, D] (NHD)
        v: Value tensor, shape [B, H, N, D] (HND) or [B, N, H, D] (NHD)
        tensor_layout: "HND" or "NHD" (default: "HND")
        sm_scale: Softmax scale, defaults to 1/sqrt(head_dim)
        smooth_k: Whether to subtract mean from K (default: True)
        smooth_v: Whether to subtract mean from V (default: False)
        qk_quant_gran: Quantization granularity, only "per_warp" is supported (default: "per_warp")

    Returns:
        Output tensor with same shape as input
    """
    if not SM80_ENABLED:
        raise RuntimeError("SM80 kernels not compiled. Please build the package with CUDA support.")

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on CUDA"
    assert dtype in [torch.float16, torch.bfloat16], "Input must be fp16 or bf16"
    assert q.device == k.device == v.device, "All tensors must be on same device"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have same dtype"

    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    # Pad head dimension to 64 or 128
    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim must be contiguous"

    seq_dim = 1 if tensor_layout == "NHD" else 2

    # Smooth K if requested (improves quantization accuracy)
    km = None
    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)

    # Convert to fp16 if needed
    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    # Smooth V if requested
    if smooth_v:
        v, vm = sub_mean(v, tensor_layout)
    else:
        vm = None

    # Quantize Q and K
    # QuantGranularity enum: 0=kPerTensor, 1=kPerBlock, 2=kPerWarp, 3=kPerThread
    # Note: SM80 CUDA kernels only support per_warp (2) and per_thread (3)
    # For now, only per_warp is fully supported
    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, km, tensor_layout=tensor_layout)
        _qk_quant_gran = 2  # kPerWarp
    else:
        raise ValueError(f"Unsupported qk_quant_gran: {qk_quant_gran}. SM80 currently only supports 'per_warp'.")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_causal = 0  # SAM doesn't use causal masking
    _return_lse = 0  # No need for LSE in vision models

    # Create output tensor
    o = torch.empty_like(q, dtype=torch.float16)

    # Run attention kernel
    if smooth_v:
        sm80_compile.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
            q_int8, k_int8, v, o, q_scale, k_scale, vm,
            _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse
        )
    else:
        sm80_compile.qk_int8_sv_f16_accum_f32_attn(
            q_int8, k_int8, v, o, q_scale, k_scale,
            _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse
        )

    # Unpad output
    if head_dim_og < 64 or (head_dim_og > 64 and head_dim_og < 128):
        o = o[..., :head_dim_og]

    # Convert back to original dtype
    if dtype == torch.bfloat16:
        o = o.to(torch.bfloat16)

    return o
