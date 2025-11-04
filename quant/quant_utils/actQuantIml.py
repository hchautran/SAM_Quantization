import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Protocol
from abc import ABC, abstractmethod

# ============================================================================
# Activation Quantization Functions (Core Implementation)
# ============================================================================

@torch.no_grad()
def quantize_activation_per_token_absmax(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize activations per token using absolute maximum scaling."""
    t_shape = t.shape
    t = t.contiguous().view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t.view(t_shape)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize activations per tensor using absolute maximum scaling."""
    t_shape = t.shape
    t = t.contiguous().view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t.view(t_shape)


@torch.no_grad()
def quantize_activation_per_group_absmax_token_dim(
    t: torch.Tensor,
    group_size: int,
    n_bits: int = 8
) -> torch.Tensor:
    """Per-group activation quantization grouping features in the last dimension."""
    t_shape = t.shape
    last_dim = t_shape[-1]
    assert last_dim % group_size == 0

    new_shape = t_shape[:-1] + (last_dim // group_size, group_size)
    t_grouped = t.view(new_shape)
    t_reshaped = t_grouped.view(-1, group_size)
    t_quantized = quantize_activation_per_token_absmax(t_reshaped, n_bits=n_bits)
    return t_quantized.view(t_shape)


def cal_density(X: torch.Tensor, margin: float = 0.9) -> torch.Tensor:
    """Calculate token density scores based on self-similarity."""
    B, H, W, C = X.shape
    X = X.view(B, 1, H * W, C)
    X = F.normalize(X, p=2, dim=-1)
    score_map = F.elu(X @ X.transpose(-1, -2) - margin, alpha=0)
    return score_map.mean(-1)


@torch.no_grad()
def quantize_activation_low_high_density_activation(
    t: torch.Tensor,
    n_bits: int = 8,
    quantizehigh: bool = True,
    percent: float = 50
) -> torch.Tensor:
    """Quantize activations based on token density (high or low)."""
    original_shape = t.shape
    original_dtype = t.dtype

    B, H, W, C = t.shape
    scores = cal_density(t, 0.5).squeeze(1).reshape(-1)
    t_2d = t.view(B * H * W, C)

    _, sorted_indices = torch.sort(scores, descending=True)
    num_to_quantize = int(scores.numel() * (percent / 100.0))

    token_mask = torch.zeros_like(scores, dtype=torch.bool)
    if quantizehigh:
        token_mask[sorted_indices[:num_to_quantize]] = True
    else:
        token_mask[sorted_indices[-num_to_quantize:]] = True

    output = t_2d.clone()
    tokens_to_quantize = t_2d[token_mask]

    if tokens_to_quantize.numel() > 0:
        scales = tokens_to_quantize.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        output[token_mask] = (tokens_to_quantize / scales).round() * scales

    return output.view(original_shape).to(original_dtype)

@torch.no_grad()
def quantize_activation_low_high_density_activation_index(
    t: torch.Tensor,
    n_bits: int = 8,
    quantizehigh: bool = True,
    percent: float = 50,
    indices: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations based on token density (high or low)."""
    original_shape = t.shape
    original_dtype = t.dtype


    if len(t.shape) == 3 and t.shape[1] == t.shape[2]: # Attention matrix case: (B*nHead, HW, HW)
        B_nHead, HW, _ = t.shape
        output = t.clone()

        attn_norm = F.normalize(t, p=2, dim=1)

        # Compute channel similarity for all batches: (B*nHead, HW, HW)
        channel_similarity = torch.bmm(attn_norm.transpose(1, 2), attn_norm)

        # Apply ELU and compute density scores
        density_scores = F.elu(channel_similarity - 0.9, alpha=0)
        channel_density = density_scores.mean(dim=2)  # (B*nHead, HW)
        _, sorted_indices = torch.sort(channel_density, dim=1, descending=True)

        # Calculate number of channels to quantize
        num_to_quantize = int(HW * (percent / 100.0))
        
        # Create batch indices for gathering
        batch_indices = torch.arange(B_nHead, device=t.device).unsqueeze(1)

        if quantizehigh:
            selected_indices = sorted_indices[:, :num_to_quantize]  # (B*nHead, num_to_quantize)
        else:
            selected_indices = sorted_indices[:, -num_to_quantize:]  # (B*nHead, num_to_quantize)
        # Create mask for selected channels
        mask = torch.zeros(B_nHead, HW, dtype=torch.bool, device=t.device)
        mask.scatter_(1, selected_indices, True)
 
        # Quantize selected channels for all batches at once
        # First, we need to gather the selected channels
        mask_expanded = mask.unsqueeze(1).expand(-1, HW, -1)  # (B*nHead, HW, HW)
        channels_to_quantize = output[mask_expanded].view(B_nHead, HW, -1)  # (B*nHead, HW, num_to_quantize)
        if channels_to_quantize.numel() > 0:
            # Calculate scales for each selected channel
            scales = channels_to_quantize.abs().max(dim=2, keepdim=True)[0]  # (B*nHead, 1, num_to_quantize)
            q_max = 2 ** (n_bits - 1) - 1
            scales.clamp_(min=1e-5).div_(q_max)

            # Quantize
            quantized_channels = (channels_to_quantize / scales).round() * scales
        
            # Put quantized values back
            output[mask_expanded] = quantized_channels.view(-1)
        
       
        return output.to(original_dtype), selected_indices # selected_indices: (B*nHead, num_to_quantize)

    elif len(t.shape) == 3 and indices is not None:
        # Value matrix case: (B*nHead, H*W, C)
        B_nHead, HW, C = t.shape
        # indices shape of (B*nHead, num_to_quantize)
        num_to_quantize = indices.shape[1]
        
        output = t.clone()
        
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(B_nHead, device=t.device).unsqueeze(1)  # (B*nHead, 1)
        # import ipdb; ipdb.set_trace()
        # Extract tokens to quantize using advanced indexing
        tokens_to_quantize = t[batch_indices, indices]  # (B*nHead, num_to_quantize, C)
        if tokens_to_quantize.numel() > 0:
            # Calculate scales across the num_to_quantize dimension (dim=1)
            scales = tokens_to_quantize.abs().max(dim=1, keepdim=True)[0]  # (B*nHead, 1, C)
            q_max = 2 ** (n_bits - 1) - 1
            scales.clamp_(min=1e-5).div_(q_max)
            
            # Quantize
            quantized_tokens = (tokens_to_quantize / scales).round() * scales
            
            # Put quantized values back
            output[batch_indices, indices] = quantized_tokens

        return output.view(original_shape).to(original_dtype), indices


