import math
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import os
import logging
import time
from typing import Optional
from matplotlib import pyplot as plt


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    """
    Quantize activations per token (per row) using absmax scaling.

    Args:
        t: Tensor to quantize (any shape)
        n_bits: Number of bits for quantization

    Returns:
        Quantized tensor with same shape as input
    """
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_activation_per_channel_absmax(t, n_bits=8):
    """
    Quantize activations per channel (along dim 1) using absmax scaling.
    Args:
        t: Tensor to quantize, shape (..., N, C)
        n_bits: Number of bits for quantization

    Returns:
        Quantized tensor with same shape as input
    """
    scales = t.abs().max(dim=1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    """
    Quantize entire tensor using single absmax scale.
    Args:
        t: Tensor to quantize
        n_bits: Number of bits for quantization

    Returns:
        Quantized tensor with same shape as input
    """
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


# All processor classes have been moved to the processors/ module
# Import them from processors if needed
