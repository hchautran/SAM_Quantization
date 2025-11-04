import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Protocol
from abc import ABC, abstractmethod
from .weightQuantIml import *
# ============================================================================
# Strategy Interfaces
# ============================================================================
  

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
# Concrete Weight Quantization Strategies
# ============================================================================

class PerChannelWeightQuantization:
    """Per-channel weight quantization strategy."""

    def __init__(self, rounding: str = "RTN", percent: float = 1.00):
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