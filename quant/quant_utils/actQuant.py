import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Protocol
from abc import ABC, abstractmethod
from .actQuantIml import *

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

