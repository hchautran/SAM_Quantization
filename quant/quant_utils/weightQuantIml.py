
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Protocol
from abc import ABC, abstractmethod


# ============================================================================
# Weight Quantization Functions (Core Implementation)
# ============================================================================

@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize weights per output channel using absolute maximum scaling."""
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w



@torch.no_grad()
def quantize_weight_per_channel_absmax_selective(
    w: torch.Tensor,
    n_bits: int = 8,
    order: Optional[torch.Tensor] = None,
    topk: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Reorder weight channels first, then apply selective quantization."""
    w_reordered = w.clone()
    d_out, d_in = w.shape

    if order is not None:
        if order.dim() == 1:
            w_reordered = torch.gather(
                w_reordered, dim=1,
                index=order.unsqueeze(0).expand(w_reordered.size(0), -1)
            )
        elif order.dim() == 2:
            print('Reordering channels with per-output-channel order')
            w_reordered = w_reordered.reshape(8, d_out // 8, d_in)
            w_reordered = torch.gather(
                w_reordered, dim=1,
                index=order[..., None].expand(w_reordered.shape)
            )

    w_backup = None
    if topk is not None:
        if isinstance(topk, list):
            topk = torch.tensor(topk, device=w.device)
        w_backup = w_reordered[:, topk, :].clone()
        w_reordered = w_reordered.reshape(d_out, d_in)

    scales = w_reordered.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w_reordered.div_(scales).round_().mul_(scales)

    if w_backup is not None:
        w_reordered = w_reordered.reshape(8, d_out // 8, d_in)
        w_reordered[:, topk, :] = w_backup
        w_reordered = w_reordered.reshape(d_out, d_in)

    return w_reordered

    
@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize weights per output channel using absolute maximum scaling."""
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w



@torch.no_grad()
def quantize_weight_per_tensor_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize weights per tensor using absolute maximum scaling."""
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w



@torch.no_grad()
def quantize_weight_per_channel_random_round_up_down_absmax(
    w: torch.Tensor,
    n_bits: int = 8,
    state: str = "RTN",
    percent: float = 0.5
) -> torch.Tensor:
    """Quantize a random subset of weight channels using various rounding strategies."""
    original_dtype = w.dtype
    out_features, _ = w.shape
    num_channels_to_quantize = int(out_features * percent)
    random_indices = torch.randperm(out_features)[:num_channels_to_quantize]

    w_output = w.clone()
    for idx in random_indices:
        channel = w[idx]
        scale = channel.abs().max().clamp(min=1e-5) / (2 ** (n_bits - 1) - 1)
        channel_normalized = channel / scale

        if state == "up":
            channel_quantized = channel_normalized.ceil()
        elif state == "down":
            channel_quantized = channel_normalized.floor()
        elif state == "RTN":
            channel_quantized = channel_normalized.round()
        elif state == "random":
            random_mask = torch.rand_like(channel_normalized) > 0.5
            channel_quantized = torch.where(
                random_mask,
                channel_normalized.ceil(),
                channel_normalized.floor()
            )
        else:
            raise ValueError(f"Invalid state: {state}")

        w_output[idx] = channel_quantized * scale

    return w_output.to(original_dtype)


@torch.no_grad()
def quantize_weight_per_group_absmax_input_features(
    w: torch.Tensor,
    group_size: int,
    n_bits: int = 8
) -> torch.Tensor:
    """Quantize weights in groups along the input features dimension."""
    out_features, in_features = w.shape
    assert in_features % group_size == 0 and w.dim() == 2

    w_grouped = w.view(out_features, -1, group_size)
    w_reshaped = w_grouped.view(-1, group_size)
    quantized_w = quantize_weight_per_channel_absmax(w_reshaped, n_bits=n_bits)
    return quantized_w.view(out_features, in_features)

