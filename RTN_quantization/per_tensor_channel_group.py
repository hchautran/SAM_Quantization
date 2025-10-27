"""
Quantization module for weight and activation quantization.

This module provides various quantization strategies including:
- Per-channel and per-tensor weight quantization
- Per-token and per-group activation quantization
- Selective channel quantization with reordering
- Density-based activation quantization (high/low density tokens)
- Random rounding strategies (RTN, up, down, random)

Architecture:
- Strategy pattern for activation and weight quantization
- WnAnLinear class uses composition with strategy objects
- Clean separation between quantization algorithms and linear layer logic
"""

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
    scores = cal_density(t).squeeze(1).reshape(-1)
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

# ============================================================================
# Strategy Interfaces
# ============================================================================

class ActivationQuantizationStrategy(Protocol):
    """Protocol defining the interface for activation quantization strategies."""

    def quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        """
        Quantize activation tensor.

        Args:
            x: Input activation tensor
            n_bits: Number of quantization bits

        Returns:
            Quantized activation tensor
        """
        ...

    @property
    def name(self) -> str:
        """Return the strategy name for identification."""
        ...


class WeightQuantizationStrategy(Protocol):
    """Protocol defining the interface for weight quantization strategies."""

    def quantize(self, w: torch.Tensor, n_bits: int) -> torch.Tensor:
        """
        Quantize weight tensor.

        Args:
            w: Input weight tensor
            n_bits: Number of quantization bits

        Returns:
            Quantized weight tensor
        """
        ...

    @property
    def name(self) -> str:
        """Return the strategy name for identification."""
        ...


# ============================================================================
# Concrete Activation Quantization Strategies
# ============================================================================

class PerTokenActivationQuantization:
    """Per-token activation quantization strategy."""

    @property
    def name(self) -> str:
        return "per_token"

    def quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        return quantize_activation_per_token_absmax(x, n_bits=n_bits)


class PerTensorActivationQuantization:
    """Per-tensor activation quantization strategy."""

    @property
    def name(self) -> str:
        return "per_tensor"

    def quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        return quantize_activation_per_tensor_absmax(x, n_bits=n_bits)


class PerGroupActivationQuantization:
    """Per-group activation quantization strategy."""

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    @property
    def name(self) -> str:
        return "per_group_token"

    def quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        return quantize_activation_per_group_absmax_token_dim(
            x, group_size=self.group_size, n_bits=n_bits
        )


class DensityBasedActivationQuantization:
    """Density-based selective activation quantization strategy."""

    def __init__(self, quantize_high: bool = True, percent: float = 50):
        self.quantize_high = quantize_high
        self.percent = percent

    @property
    def name(self) -> str:
        return "low_high_density_activation"

    def quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        return quantize_activation_low_high_density_activation(
            x, n_bits=n_bits, quantizehigh=self.quantize_high, percent=self.percent
        )


# ============================================================================
# Concrete Weight Quantization Strategies
# ============================================================================

class PerChannelWeightQuantization:
    """Per-channel weight quantization strategy."""

    def __init__(self, rounding: str = "RTN", percent: float = 0.75):
        self.rounding = rounding
        self.percent = percent

    @property
    def name(self) -> str:
        return "per_channel"

    def quantize(self, w: torch.Tensor, n_bits: int) -> torch.Tensor:
        if self.rounding in ["up", "down", "RTN", "random"]:
            return quantize_weight_per_channel_random_round_up_down_absmax(
                w, n_bits=n_bits, state=self.rounding, percent=self.percent
            )
        else:
            return quantize_weight_per_channel_absmax(w, n_bits=n_bits)


class PerTensorWeightQuantization:
    """Per-tensor weight quantization strategy."""

    @property
    def name(self) -> str:
        return "per_tensor"

    def quantize(self, w: torch.Tensor, n_bits: int) -> torch.Tensor:
        return quantize_weight_per_tensor_absmax(w, n_bits=n_bits)


class PerGroupWeightQuantization:
    """Per-group weight quantization strategy."""

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    @property
    def name(self) -> str:
        return "per_group"

    def quantize(self, w: torch.Tensor, n_bits: int) -> torch.Tensor:
        return quantize_weight_per_group_absmax_input_features(
            w, group_size=self.group_size, n_bits=n_bits
        )


class SelectiveChannelWeightQuantization:
    """Selective channel weight quantization with reordering strategy."""

    def __init__(self, order: Optional[torch.Tensor] = None, topk: Optional[torch.Tensor] = None):
        self.order = order
        self.topk = topk

    @property
    def name(self) -> str:
        return "selective_channel"

    def quantize(self, w: torch.Tensor, n_bits: int) -> torch.Tensor:
        return quantize_weight_per_channel_absmax_selective(
            w, n_bits=n_bits, order=self.order, topk=self.topk
        )


# ============================================================================
# Base WnAnLinear Class
# ============================================================================

class WnAnLinear(nn.Module):
    """
    W8A8 quantized linear layer using strategy pattern.

    Uses composition with strategy objects for flexible quantization:
    - Activation quantization strategy for input quantization
    - Optional output quantization (reuses activation strategy)
    - Clean separation between quantization algorithms and layer logic
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        activation_strategy: Optional[ActivationQuantizationStrategy] = None,
        quantize_output: bool = False,
    ):
        """
        Initialize WnAnLinear layer with strategies.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            n_bits_w: Number of weight quantization bits
            n_bits_a: Number of activation quantization bits
            activation_strategy: Strategy for activation quantization
            quantize_output: Whether to quantize output activations
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits_w = n_bits_w
        self.n_bits_a = n_bits_a
        self.quantize_output_flag = quantize_output

        # Strategy for activation quantization
        self.activation_strategy = activation_strategy or PerTokenActivationQuantization()

        # Initialize weight buffer
        self.register_buffer(
            "weight",
            torch.randn(out_features, in_features, dtype=torch.float16, requires_grad=False),
        )

        # Initialize bias buffer
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((1, out_features), dtype=torch.float16, requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        # Metadata for backward compatibility
        self.weight_quant_name = "None"
        self.output_quant_name = "None"

    @property
    def act_quant_name(self) -> str:
        """Get activation quantization strategy name."""
        return self.activation_strategy.name

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input activation using the configured strategy.

        Args:
            x: Input activation tensor

        Returns:
            Quantized activation tensor
        """
        return self.activation_strategy.quantize(x, self.n_bits_a)

    def quantize_output(self, y: torch.Tensor) -> torch.Tensor:
        """
        Quantize output activation.

        Reuses input quantization strategy if enabled.

        Args:
            y: Output activation tensor

        Returns:
            Quantized output tensor or original tensor if output quantization disabled
        """
        if self.quantize_output_flag:
            return self.quantize_activation(y)
        return y

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.

        Args:
            x: Input tensor

        Returns:
            Output tensor after quantized linear transformation
        """
        q_x = self.quantize_activation(x)
        y = F.linear(q_x, self.weight, self.bias)
        q_y = self.quantize_output(y)
        return q_y

    def to(self, *args, **kwargs):
        """Move module to specified device/dtype."""
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    def __repr__(self) -> str:
        """String representation of the module."""
        return (
            f"{self.__class__.__name__}({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"
        )

    @staticmethod
    def from_float(
        module: nn.Linear,
        n_bits_w: int,
        n_bits_a: int,
        weight_quant: str = "per_channel",
        act_quant: str = "per_token",
        quantize_output: bool = False,
        group_size: Optional[int] = None,
        quantize_weight: bool = True,
        order: Optional[torch.Tensor] = None,
        topk: Optional[torch.Tensor] = None,
        quantizehigh: bool = True,
        up_down_RTN: str = "RTN",
        percent: float = 100
    ) -> 'WnAnLinear':
        """
        Factory method to create WnAnLinear from float linear layer using strategies.

        This method creates the appropriate quantization strategies based on parameters
        and applies them to create a quantized linear layer.

        Args:
            module: Source floating-point linear layer
            n_bits_w: Number of bits for weight quantization
            n_bits_a: Number of bits for activation quantization
            weight_quant: Weight quantization strategy name
            act_quant: Activation quantization strategy name
            quantize_output: Whether to quantize output
            group_size: Group size for group quantization
            quantize_weight: Whether to quantize weights
            order: Channel reordering for selective quantization
            topk: Top-k channels to preserve
            quantizehigh: For density-based quantization
            up_down_RTN: Rounding strategy
            percent: Percentage for density-based quantization

        Returns:
            WnAnLinear instance with configured strategies
        """
        assert isinstance(module, torch.nn.Linear)

        # Create activation quantization strategy
        if act_quant == "per_token":
            activation_strategy = PerTokenActivationQuantization()
        elif act_quant == "per_tensor":
            activation_strategy = PerTensorActivationQuantization()
        elif act_quant == "per_group_token":
            activation_strategy = PerGroupActivationQuantization(group_size=group_size or 128)
        elif act_quant == "low_high_density_activation":
            activation_strategy = DensityBasedActivationQuantization(
                quantize_high=quantizehigh, percent=percent
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        # Create WnAnLinear module with activation strategy
        new_module = WnAnLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            activation_strategy=activation_strategy,
            quantize_output=quantize_output
        )

        # Apply weight quantization using appropriate strategy
        if quantize_weight:
            if weight_quant == "selective_channel":
                weight_strategy = SelectiveChannelWeightQuantization(order=order, topk=topk)
            elif weight_quant == "per_channel":
                weight_strategy = PerChannelWeightQuantization(
                    rounding=up_down_RTN if up_down_RTN in ["up", "down", "RTN", "random"] else "RTN",
                    percent=0.75
                )
            elif weight_quant == "per_tensor":
                weight_strategy = PerTensorWeightQuantization()
            elif weight_quant == "per_group":
                weight_strategy = PerGroupWeightQuantization(group_size=group_size or 128)
            else:
                raise ValueError(f"Invalid weight_quant: {weight_quant}")

            # Quantize weights using strategy
            new_module.weight = weight_strategy.quantize(module.weight, n_bits=n_bits_w)
            new_module.weight_quant_name = weight_strategy.name
        else:
            new_module.weight = module.weight

        # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module


# ============================================================================
# Backward Compatibility Aliases (Legacy Subclasses)
# ============================================================================
# These classes are kept for backward compatibility with existing code.
# New code should use WnAnLinear directly with appropriate strategies.

class WnAnLinearPerChannel(WnAnLinear):
    """Legacy class for backward compatibility. Use WnAnLinear with PerTokenActivationQuantization instead."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
        rounding: str = "RTN",
    ):
        super().__init__(
            in_features, out_features, bias, n_bits_w, n_bits_a,
            activation_strategy=PerTokenActivationQuantization(),
            quantize_output=quantize_output
        )
        self.rounding = rounding


class WnAnLinearPerTensor(WnAnLinear):
    """Legacy class for backward compatibility. Use WnAnLinear with PerTensorActivationQuantization instead."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
    ):
        super().__init__(
            in_features, out_features, bias, n_bits_w, n_bits_a,
            activation_strategy=PerTensorActivationQuantization(),
            quantize_output=quantize_output
        )


class WnAnLinearPerGroup(WnAnLinear):
    """Legacy class for backward compatibility. Use WnAnLinear with PerGroupActivationQuantization instead."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        group_size: int = 128,
        quantize_output: bool = False,
    ):
        super().__init__(
            in_features, out_features, bias, n_bits_w, n_bits_a,
            activation_strategy=PerGroupActivationQuantization(group_size=group_size),
            quantize_output=quantize_output
        )
        self.group_size = group_size


class WnAnLinearDensityBased(WnAnLinear):
    """Legacy class for backward compatibility. Use WnAnLinear with DensityBasedActivationQuantization instead."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_high: bool = True,
        percent: float = 50,
        quantize_output: bool = False,
    ):
        super().__init__(
            in_features, out_features, bias, n_bits_w, n_bits_a,
            activation_strategy=DensityBasedActivationQuantization(
                quantize_high=quantize_high, percent=percent
            ),
            quantize_output=quantize_output
        )
        self.quantize_high = quantize_high
        self.percent = percent
        self.quantizehigh = quantize_high  # Backward compatibility


class WnAnLinearSelectiveChannel(WnAnLinear):
    """Legacy class for backward compatibility. Use WnAnLinear with SelectiveChannelWeightQuantization instead."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        order: Optional[torch.Tensor] = None,
        topk: Optional[torch.Tensor] = None,
        quantize_output: bool = False,
    ):
        super().__init__(
            in_features, out_features, bias, n_bits_w, n_bits_a,
            activation_strategy=PerTokenActivationQuantization(),
            quantize_output=quantize_output
        )
        self.order = order
        self.topk = topk
