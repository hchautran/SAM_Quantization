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
from dataclasses import dataclass
from typing import Optional
from .actQuant import *
from .weightQuant import *






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
        activation_strategy: Optional[ActivationQuantizationStrategy] = None,
        weight_strategy: Optional['WeightQuantizationStrategy'] = None,
        quantize_output: bool = False,
    ) -> 'WnAnLinear':
        """
        Factory method to create WnAnLinear from float linear layer using strategies.

        This is the clean strategy-based API. Use the appropriate subclass's from_float
        method for parameter-based construction (e.g., WnAnLinearPerChannel.from_float).

        Args:
            module: Source floating-point linear layer
            n_bits_w: Number of bits for weight quantization
            n_bits_a: Number of bits for activation quantization
            activation_strategy: Strategy for activation quantization
            weight_strategy: Strategy for weight quantization (optional)
            quantize_output: Whether to quantize output

        Returns:
            WnAnLinear instance with configured strategies

        Example:
            >>> linear = nn.Linear(128, 256)
            >>> # Using strategies directly
            >>> quant = WnAnLinear.from_float(
            ...     linear,
            ...     n_bits_w=8,
            ...     n_bits_a=8,
            ...     activation_strategy=PerTokenActivationQuantization(),
            ...     weight_strategy=PerChannelWeightQuantization()
            ... )
            >>> # Or use subclass for parameter-based API
            >>> quant = WnAnLinearPerChannel.from_float(linear, n_bits_w=8, n_bits_a=8)
        """
        assert isinstance(module, torch.nn.Linear)

        # Use default strategy if none provided
        if activation_strategy is None:
            activation_strategy = PerTokenActivationQuantization()

        # Create WnAnLinear module
        new_module = WnAnLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            activation_strategy=activation_strategy,
            quantize_output=quantize_output
        )

        # Apply weight quantization if strategy provided
        if weight_strategy is not None:
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

    @staticmethod
    def from_float(
        module: nn.Linear,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
        quantize_weight: bool = True,
        up_down_RTN: str = "RTN",
        **kwargs  # Accept and ignore other parameters for compatibility
    ) -> 'WnAnLinearPerChannel':
        """
        Create WnAnLinearPerChannel from float linear layer.

        Uses per-channel weight quantization and per-token activation quantization.
        """
        assert isinstance(module, torch.nn.Linear)

        # Create new module
        new_module = WnAnLinearPerChannel(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            quantize_output=quantize_output,
            rounding=up_down_RTN
        )

        # Apply weight quantization
        if quantize_weight:
            weight_strategy = PerChannelWeightQuantization(
                rounding=up_down_RTN if up_down_RTN in ["up", "down", "RTN", "random"] else "RTN",
                percent=0.75
            )
            new_module.weight = weight_strategy.quantize(module.weight, n_bits=n_bits_w)
            new_module.weight_quant_name = weight_strategy.name
        else:
            new_module.weight = module.weight

        # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module


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

    @staticmethod
    def from_float(
        module: nn.Linear,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
        quantize_weight: bool = True,
        **kwargs  # Accept and ignore other parameters for compatibility
    ) -> 'WnAnLinearPerTensor':
        """
        Create WnAnLinearPerTensor from float linear layer.

        Uses per-tensor weight quantization and per-tensor activation quantization.
        """
        assert isinstance(module, torch.nn.Linear)

        # Create new module
        new_module = WnAnLinearPerTensor(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            quantize_output=quantize_output
        )

        # Apply weight quantization
        if quantize_weight:
            weight_strategy = PerTensorWeightQuantization()
            new_module.weight = weight_strategy.quantize(module.weight, n_bits=n_bits_w)
            new_module.weight_quant_name = weight_strategy.name
        else:
            new_module.weight = module.weight

        # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module


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

    @staticmethod
    def from_float(
        module: nn.Linear,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        group_size: int = 128,
        quantize_output: bool = False,
        quantize_weight: bool = True,
        weight_quant: str = "per_group",
        **kwargs  # Accept and ignore other parameters for compatibility
    ) -> 'WnAnLinearPerGroup':
        """
        Create WnAnLinearPerGroup from float linear layer.

        Uses per-group weight quantization and per-group activation quantization.
        """
        assert isinstance(module, torch.nn.Linear)

        # Create new module
        new_module = WnAnLinearPerGroup(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            group_size=group_size,
            quantize_output=quantize_output
        )

        # Apply weight quantization
        if quantize_weight:
            weight_strategy = PerGroupWeightQuantization(group_size=group_size)
            new_module.weight = weight_strategy.quantize(module.weight, n_bits=n_bits_w)
            new_module.weight_quant_name = weight_strategy.name
        else:
            new_module.weight = module.weight

        # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module


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

    @staticmethod
    def from_float(
        module: nn.Linear,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
        quantize_weight: bool = True,
        quantizehigh: bool = True,
        percent: float = 50,
        up_down_RTN: str = "RTN",
        **kwargs  # Accept and ignore other parameters for compatibility
    ) -> 'WnAnLinearDensityBased':
        """
        Create WnAnLinearDensityBased from float linear layer.

        Uses density-based activation quantization (quantize high or low density tokens).
        """
        assert isinstance(module, torch.nn.Linear)

        # Create new module
        new_module = WnAnLinearDensityBased(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            quantize_high=quantizehigh,
            percent=percent,
            quantize_output=quantize_output
        )

        # Apply weight quantization
        if quantize_weight:
            weight_strategy = PerChannelWeightQuantization(
                rounding=up_down_RTN if up_down_RTN in ["up", "down", "RTN", "random"] else "RTN",
                percent=0.75
            )
            new_module.weight = weight_strategy.quantize(module.weight, n_bits=n_bits_w)
            new_module.weight_quant_name = weight_strategy.name
        else:
            new_module.weight = module.weight

        # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module


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

    @staticmethod
    def from_float(
        module: nn.Linear,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
        quantize_weight: bool = True,
        order: Optional[torch.Tensor] = None,
        topk: Optional[torch.Tensor] = None,
        **kwargs  # Accept and ignore other parameters for compatibility
    ) -> 'WnAnLinearSelectiveChannel':
        """
        Create WnAnLinearSelectiveChannel from float linear layer.

        Uses selective channel weight quantization with channel reordering.
        """
        assert isinstance(module, torch.nn.Linear)

        # Create new module
        new_module = WnAnLinearSelectiveChannel(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            order=order,
            topk=topk,
            quantize_output=quantize_output
        )

        # Apply weight quantization
        if quantize_weight:
            weight_strategy = SelectiveChannelWeightQuantization(order=order, topk=topk)
            new_module.weight = weight_strategy.quantize(module.weight, n_bits=n_bits_w)
            new_module.weight_quant_name = weight_strategy.name
        else:
            new_module.weight = module.weight

        # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module



class WnAnLinearRecenterChannel(WnAnLinear):
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
        self.register_buffer(
            "fp_weight",
            torch.randn(out_features, in_features, dtype=torch.float16, requires_grad=False),
        )

    @staticmethod
    def from_float(
        module: nn.Linear,
        n_bits_w: int = 8,
        n_bits_a: int = 8,
        quantize_output: bool = False,
        quantize_weight: bool = True,
        order: Optional[torch.Tensor] = None,
        topk: Optional[torch.Tensor] = None,
        **kwargs  # Accept and ignore other parameters for compatibility
    ) -> 'WnAnLinearRecenterChannel':
        """
        Create WnAnLinearSelectiveChannel from float linear layer.

        Uses selective channel weight quantization with channel reordering.
        """
        assert isinstance(module, torch.nn.Linear)

        # Create new module
        new_module = WnAnLinearRecenterChannel(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits_w=n_bits_w,
            n_bits_a=n_bits_a,
            quantize_output=quantize_output
        )

        # Apply weight quantization
        weight_strategy = PerChannelWeightQuantization(
            rounding= "RTN",
        )
        new_module.fp_weight =  module.weight
        new_module.weight = weight_strategy.quantize(module.weight, n_bits=n_bits_w)
        new_module.weight_quant_name = weight_strategy.name

    # Copy bias
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.

        Args:
            x: Input tensor

        Returns:
            Output tensor after quantized linear transformation
        """
        breakpoint()
        q_x = self.quantize_activation(x)
        y = F.linear(q_x, self.weight, self.bias)
        q_y = self.quantize_output(y)
        return q_y

        



@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters.

    This class encapsulates all quantization-related parameters to avoid
    passing many individual arguments through function calls.

    Attributes:
        n_bits_w: Number of bits for weight quantization (default: 8)
        n_bits_a: Number of bits for activation quantization (default: 8)
        weight_quant: Weight quantization strategy ("per_channel", "per_tensor", "per_group")
        act_quant: Activation quantization strategy ("per_token", "per_tensor", "per_group", "low_high_density")
        quantize_output: Whether to quantize output activations (default: False)
        quantize_weight: Whether to quantize weights (default: True)
        group_size: Group size for group-based quantization (default: None)
        order: Channel reordering indices for selective quantization (default: None)
        topk: Top-k channels to preserve in selective quantization (default: None)
        quantizehigh: For density-based quantization, whether to quantize high-density tokens (default: True)
        up_down_RTN: Rounding strategy ("RTN", "up", "down", "random") (default: "RTN")
        percent: Percentage of channels/tokens to quantize (default: 100)
    """
    n_bits_w: int = 8
    n_bits_a: int = 8
    weight_quant: str = "per_channel"
    act_quant: str = "per_token"
    quantize_output: bool = False
    quantize_weight: bool = True
    group_size: Optional[int] = None
    order: Optional[torch.Tensor] = None
    topk: Optional[torch.Tensor] = None
    
    quantizehigh: bool = True
    up_down_RTN: str = "RTN"
    percent: float = 100

    def get_WnAnLinear_class(self):
        """Determine the appropriate WnAnLinear subclass based on configuration.

        Returns:
            The appropriate WnAnLinear subclass to use
        """
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from quant.quant_utils import (
            WnAnLinearPerChannel,
            WnAnLinearPerTensor,
            WnAnLinearPerGroup,
            WnAnLinearDensityBased,
            WnAnLinearSelectiveChannel
        )

        # Selective channel quantization (has order or topk)
        if self.order is not None or self.topk is not None:
            return WnAnLinearSelectiveChannel

        # Density-based quantization
        if self.act_quant == "low_high_density_activation":
            print('got here')
            return WnAnLinearDensityBased

        # Group-based quantization
        if self.weight_quant == "per_group" or self.act_quant == "per_group":
            return WnAnLinearPerGroup

        # Per-tensor quantization
        if self.weight_quant == "per_tensor" and self.act_quant == "per_tensor":
            return WnAnLinearPerTensor

        # Default: Per-channel weight, per-token activation
        return WnAnLinearPerChannel

    def to_kwargs(self) -> dict:

        return {
            'n_bits_w': self.n_bits_w,
            'n_bits_a': self.n_bits_a,
            'weight_quant': self.weight_quant,
            'act_quant': self.act_quant,
            'quantize_output': self.quantize_output,
            'group_size': self.group_size,
            'quantize_weight': self.quantize_weight,
            'order': self.order,
            'topk': self.topk,
            'quantizehigh': self.quantizehigh,
            'up_down_RTN': self.up_down_RTN,
            'percent': self.percent
        }


def replace_linear_with_quantized(
    module: nn.Module,
    config: QuantizationConfig,
    module_name_to_exclude: Optional[list] = None,
    parent_name: str = ""
):
    """Replace linear layers with quantized versions using configuration object.

    Args:
        module: The module to traverse
        config: QuantizationConfig object containing all quantization parameters
        module_name_to_exclude: List of module names to exclude from quantization
        parent_name: The hierarchical name of the parent module (used internally)

    Example:
        >>> config = QuantizationConfig(
        ...     n_bits_w=4,
        ...     n_bits_a=8,
        ...     weight_quant="per_channel",
        ...     act_quant="per_token"
        ... )
        >>> replace_linear_with_quantized(model, config, ["lm_head"])
    """
    if module_name_to_exclude is None:
        module_name_to_exclude = []

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
           any([x == name for x in module_name_to_exclude]) and not \
           any([x in parent_name for x in module_name_to_exclude]):

            quantized_class = config.get_WnAnLinear_class()
            print(f'fake quantizing module {name}')
            new_module = quantized_class.from_float(child, **config.to_kwargs())
            setattr(module, name, new_module)

        else:
            # Recursively process child modules
            name_layer = parent_name + "." + name if parent_name else name
            replace_linear_with_quantized(
                child, config, module_name_to_exclude, name_layer
            )




# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base WnAnLinear class
    'WnAnLinear',

    # Legacy subclasses for backward compatibility
    'WnAnLinearPerChannel',
    'WnAnLinearPerTensor',
    'WnAnLinearPerGroup',
    'WnAnLinearDensityBased',
    'WnAnLinearSelectiveChannel',
]
