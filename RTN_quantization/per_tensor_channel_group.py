"""
Quantization module for weight and activation quantization.

This module provides various quantization strategies including:
- Per-channel and per-tensor weight quantization
- Per-token and per-group activation quantization
- Selective channel quantization with reordering
- Density-based activation quantization (high/low density tokens)
- Random rounding strategies (RTN, up, down, random)

Architecture:
- Base W8A8Linear class with common functionality
- Specialized subclasses for different quantization strategies
- Clean inheritance hierarchy without factory patterns
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
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
    indices : Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Quantize activations based on token density (high or low)."""
    original_shape = t.shape
    original_dtype = t.dtype
    #B * nHead, H * W, C  matrix v
    # B * nHead , H*W , H * W matrix qkT
    
    B, H, W, C = t.shape
    scores = cal_density(t).squeeze(1).reshape(-1)
    t_2d = t.view(B * H * W, C)

    _, sorted_indices = torch.sort(scores, descending=True)
    num_to_quantize = int(scores.numel() * (percent / 100.0))

    token_mask = torch.zeros_like(scores, dtype=torch.bool)
    if not indices:
        if quantizehigh:
            print("yoooooooooooooooooo")
            indices = sorted_indices[:num_to_quantize]
            token_mask[indices] = True     
        else:
            indices = sorted_indices[:num_to_quantize]
            token_mask[indices] = True
    else :
        if quantizehigh:
            token_mask[indices] = True     
        else:
            token_mask[indices] = True
        
    output = t_2d.clone()
    tokens_to_quantize = t_2d[token_mask]

    if tokens_to_quantize.numel() > 0:
        scales = tokens_to_quantize.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        output[token_mask] = (tokens_to_quantize / scales).round() * scales

    return output.view(original_shape).to(original_dtype),indices

# ============================================================================
# Base W8A8Linear Class
# ============================================================================

class W8A8Linear(nn.Module, ABC):
    """
    Base class for all W8A8 quantized linear layers.

    This abstract base class provides:
    - Common weight/bias buffer management
    - Forward pass structure
    - Abstract methods for quantization strategies
    - Backward compatibility interface

    All specialized quantization variants should inherit from this class.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
    ):
        """
        Initialize base W8A8Linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            n_bits_w: Number of weight quantization bits
            n_bits_a: Number of activation quantization bits
            quantize_output: Whether to quantize output activations
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits_w = n_bits_w
        self.n_bits_a = n_bits_a
        self.quantize_output_flag = quantize_output

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
        self.act_quant_name = "unknown"
        self.output_quant_name = "None"

    @abstractmethod
    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input activation.

        Must be implemented by subclass to define specific activation quantization strategy.

        Args:
            x: Input activation tensor

        Returns:
            Quantized activation tensor
        """
        pass

    def quantize_output(self, y: torch.Tensor) -> torch.Tensor:
        """
        Quantize output activation.

        Default implementation reuses input quantization strategy if enabled.
        Can be overridden by subclasses for different output quantization.

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
    ) -> 'W8A8Linear':
        """
        Factory method to create appropriate W8A8Linear subclass from float linear layer.

        This method analyzes the parameters and instantiates the appropriate subclass.

        Args:
            module: Source floating-point linear layer
            n_bits_w: Number of bits for weight quantization
            n_bits_ac: Number of bits for activation quantization
            weight_quant: Weight quantization strategy
            act_quant: Activation quantization strategy
            quantize_output: Whether to quantize output
            group_size: Group size for group quantization
            quantize_weight: Whether to quantize weights
            order: Channel reordering for selective quantization
            topk: Top-k channels to preserve
            quantizehigh: For density-based quantization
            up_down_RTN: Rounding strategy
            percent: Percentage for density-based quantization

        Returns:
            Appropriate W8A8Linear subclass instance
        """
        assert isinstance(module, torch.nn.Linear)

        # Create appropriate subclass based on activation quantization strategy
        if weight_quant == "selective_channel":
            new_module = W8A8LinearSelectiveChannel(
                module.in_features, module.out_features, module.bias is not None,
                n_bits_w=n_bits_w, n_bits_a=n_bits_a,
                order=order, topk=topk, quantize_output=quantize_output
            )
        elif act_quant == "per_token":
            new_module = W8A8LinearPerChannel(
                module.in_features, module.out_features, module.bias is not None,
                n_bits_w=n_bits_w, n_bits_a=n_bits_a,
                quantize_output=quantize_output, rounding=up_down_RTN
            )
        elif act_quant == "per_tensor":
            new_module = W8A8LinearPerTensor(
                module.in_features, module.out_features, module.bias is not None,
                n_bits_w=n_bits_w, n_bits_a=n_bits_a,
                quantize_output=quantize_output
            )
        elif act_quant == "per_group_token":
            new_module = W8A8LinearPerGroup(
                module.in_features, module.out_features, module.bias is not None,
                n_bits_w=n_bits_w, n_bits_a=n_bits_a,
                group_size=group_size, quantize_output=quantize_output
            )
        elif act_quant == "low_high_density_activation":
            new_module = W8A8LinearDensityBased(
                module.in_features, module.out_features, module.bias is not None,
                n_bits_w=n_bits_w, n_bits_a=n_bits_a,
                quantize_high=quantizehigh, percent=percent,
                quantize_output=quantize_output
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        # Apply weight quantization
        if quantize_weight:
            if weight_quant == "selective_channel":
                new_module.weight = quantize_weight_per_channel_absmax_selective(
                    module.weight, n_bits=n_bits_w, order=order, topk=topk
                )
            elif weight_quant == "per_channel":
                if up_down_RTN in ["up", "down", "RTN", "random"]:
                    new_module.weight = quantize_weight_per_channel_random_round_up_down_absmax(
                        module.weight, n_bits=n_bits_w, state=up_down_RTN, percent=0.75
                    )
                else:
                    new_module.weight = quantize_weight_per_channel_absmax(
                        module.weight, n_bits=n_bits_w
                    )
            elif weight_quant == "per_tensor":
                new_module.weight = quantize_weight_per_tensor_absmax(
                    module.weight, n_bits=n_bits_w
                )
            elif weight_quant == "per_group":
                new_module.weight = quantize_weight_per_group_absmax_input_features(
                    module.weight, group_size, n_bits=n_bits_w
                )
            else:
                raise ValueError(f"Invalid weight_quant: {weight_quant}")
            new_module.weight_quant_name = weight_quant
        else:
            new_module.weight = module.weight

        # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module


# ============================================================================
# Specialized W8A8Linear Subclasses
# ============================================================================

class W8A8LinearPerChannel(W8A8Linear):

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
        """
        Initialize per-channel W8A8Linear.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            n_bits_w: Number of weight quantization bits
            n_bits_a: Number of activation quantization bits
            quantize_output: Whether to quantize output
            rounding: Rounding strategy ("up", "down", "RTN", "random")
        """
        super().__init__(in_features, out_features, bias, n_bits_w, n_bits_a, quantize_output)
        self.rounding = rounding
        self.act_quant_name = "per_token"

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation using per-token strategy."""
        return quantize_activation_per_token_absmax(x, n_bits=self.n_bits_a)


class W8A8LinearPerTensor(W8A8Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
    ):
        super().__init__(in_features, out_features, bias, n_bits_w, n_bits_a, quantize_output)
        self.act_quant_name = "per_tensor"

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation using per-tensor strategy."""
        return quantize_activation_per_tensor_absmax(x, n_bits=self.n_bits_a)


class W8A8LinearPerGroup(W8A8Linear):

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
        super().__init__(in_features, out_features, bias, n_bits_w, n_bits_a, quantize_output)
        self.group_size = group_size
        self.act_quant_name = "per_group_token"

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation using per-group strategy."""
        return quantize_activation_per_group_absmax_token_dim(
            x, group_size=self.group_size, n_bits=self.n_bits_a
        )


class W8A8LinearDensityBased(W8A8Linear):
    """
    W8A8Linear with density-based selective activation quantization.

    Selectively quantizes activations based on token density scores,
    preserving important tokens in full precision.
    """

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
        """
        Initialize density-based W8A8Linear.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            n_bits_w: Number of weight quantization bits
            n_bits_a: Number of activation quantization bits
            quantize_high: If True, quantize high-density tokens; else low-density
            percent: Percentage of tokens to quantize
            quantize_output: Whether to quantize output
        """
        super().__init__(in_features, out_features, bias, n_bits_w, n_bits_a, quantize_output)
        self.quantize_high = quantize_high
        self.percent = percent
        self.quantizehigh = quantize_high  # Backward compatibility
        self.act_quant_name = "low_high_density_activation"

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation using density-based strategy."""
        return quantize_activation_low_high_density_activation(
            x, n_bits=self.n_bits_a, quantizehigh=self.quantize_high, percent=self.percent
        )


class W8A8LinearSelectiveChannel(W8A8Linear):
    """
    W8A8Linear with selective channel weight quantization.

    Reorders and selectively preserves important weight channels in full precision,
    achieving better accuracy retention.
    """

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
        """
        Initialize selective channel W8A8Linear.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            n_bits_w: Number of weight quantization bits
            n_bits_a: Number of activation quantization bits
            order: Channel reordering tensor
            topk: Top-k channels to preserve in full precision
            quantize_output: Whether to quantize output
        """
        super().__init__(in_features, out_features, bias, n_bits_w, n_bits_a, quantize_output)
        self.order = order
        self.topk = topk
        self.act_quant_name = "per_token"

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation using per-token strategy."""
        return quantize_activation_per_token_absmax(x, n_bits=self.n_bits_a)
