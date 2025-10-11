#!/usr/bin/env python
"""
Test script to verify the strategy pattern refactoring maintains backward compatibility.
"""

import torch
import torch.nn as nn
from RTN_quantization.per_tensor_channel_group import (
    W8A8Linear,
    W8A8LinearPerChannel,
    W8A8LinearPerTensor,
    W8A8LinearPerGroup,
    W8A8LinearDensityBased,
    W8A8LinearSelectiveChannel,
    PerTokenActivationQuantization,
    PerTensorActivationQuantization,
    PerGroupActivationQuantization,
    DensityBasedActivationQuantization,
    PerChannelWeightQuantization,
    PerTensorWeightQuantization,
    PerGroupWeightQuantization,
    SelectiveChannelWeightQuantization,
)


def test_strategy_classes():
    """Test that strategy classes work correctly."""
    print("Testing strategy classes...")

    # Test activation strategies
    per_token_strategy = PerTokenActivationQuantization()
    assert per_token_strategy.name == "per_token"

    per_tensor_strategy = PerTensorActivationQuantization()
    assert per_tensor_strategy.name == "per_tensor"

    per_group_strategy = PerGroupActivationQuantization(group_size=128)
    assert per_group_strategy.name == "per_group_token"

    density_strategy = DensityBasedActivationQuantization(quantize_high=True, percent=50)
    assert density_strategy.name == "low_high_density_activation"

    # Test weight strategies
    per_channel_weight = PerChannelWeightQuantization()
    assert per_channel_weight.name == "per_channel"

    per_tensor_weight = PerTensorWeightQuantization()
    assert per_tensor_weight.name == "per_tensor"

    per_group_weight = PerGroupWeightQuantization(group_size=128)
    assert per_group_weight.name == "per_group"

    selective_weight = SelectiveChannelWeightQuantization()
    assert selective_weight.name == "selective_channel"

    print("✓ Strategy classes work correctly")


def test_w8a8linear_with_strategies():
    """Test W8A8Linear with different strategies."""
    print("\nTesting W8A8Linear with strategies...")

    # Test with per-token strategy
    linear = W8A8Linear(
        in_features=256,
        out_features=512,
        activation_strategy=PerTokenActivationQuantization()
    )
    assert linear.act_quant_name == "per_token"

    # Test forward pass (use float16 to match weight dtype)
    x = torch.randn(2, 16, 256, dtype=torch.float16)
    y = linear(x)
    assert y.shape == (2, 16, 512)

    print("✓ W8A8Linear with strategies works correctly")


def test_backward_compatibility_classes():
    """Test that legacy subclasses still work."""
    print("\nTesting backward compatibility classes...")

    # Use float16 to match weight dtype
    x = torch.randn(2, 16, 256, dtype=torch.float16)

    # Test W8A8LinearPerChannel
    linear_per_channel = W8A8LinearPerChannel(256, 512)
    assert linear_per_channel.act_quant_name == "per_token"
    y = linear_per_channel(x)
    assert y.shape == (2, 16, 512)

    # Test W8A8LinearPerTensor
    linear_per_tensor = W8A8LinearPerTensor(256, 512)
    assert linear_per_tensor.act_quant_name == "per_tensor"
    y = linear_per_tensor(x)
    assert y.shape == (2, 16, 512)

    # Test W8A8LinearPerGroup
    linear_per_group = W8A8LinearPerGroup(256, 512, group_size=128)
    assert linear_per_group.act_quant_name == "per_group_token"
    y = linear_per_group(x)
    assert y.shape == (2, 16, 512)

    # Test W8A8LinearDensityBased (need 4D input)
    linear_density = W8A8LinearDensityBased(256, 512, quantize_high=True, percent=50)
    assert linear_density.act_quant_name == "low_high_density_activation"
    # Note: density-based requires 4D input

    # Test W8A8LinearSelectiveChannel
    linear_selective = W8A8LinearSelectiveChannel(256, 512)
    assert linear_selective.act_quant_name == "per_token"
    y = linear_selective(x)
    assert y.shape == (2, 16, 512)

    print("✓ Backward compatibility classes work correctly")


def test_from_float_factory():
    """Test the from_float factory method."""
    print("\nTesting from_float factory method...")

    # Create a float linear layer
    float_linear = nn.Linear(256, 512)

    # Test per_token activation
    quant_linear = W8A8Linear.from_float(
        float_linear,
        n_bits_w=8,
        n_bits_a=8,
        weight_quant="per_channel",
        act_quant="per_token"
    )
    assert quant_linear.act_quant_name == "per_token"
    assert quant_linear.weight_quant_name == "per_channel"

    # Test per_tensor activation
    quant_linear = W8A8Linear.from_float(
        float_linear,
        n_bits_w=8,
        n_bits_a=8,
        weight_quant="per_tensor",
        act_quant="per_tensor"
    )
    assert quant_linear.act_quant_name == "per_tensor"
    assert quant_linear.weight_quant_name == "per_tensor"

    # Test per_group activation
    quant_linear = W8A8Linear.from_float(
        float_linear,
        n_bits_w=8,
        n_bits_a=8,
        weight_quant="per_group",
        act_quant="per_group_token",
        group_size=128
    )
    assert quant_linear.act_quant_name == "per_group_token"
    assert quant_linear.weight_quant_name == "per_group"

    # Test density-based activation
    quant_linear = W8A8Linear.from_float(
        float_linear,
        n_bits_w=8,
        n_bits_a=8,
        weight_quant="per_channel",
        act_quant="low_high_density_activation",
        quantizehigh=True,
        percent=50
    )
    assert quant_linear.act_quant_name == "low_high_density_activation"

    print("✓ from_float factory method works correctly")


def test_quantization_functions():
    """Test that quantization actually changes values."""
    print("\nTesting quantization functions...")

    float_linear = nn.Linear(256, 512)
    quant_linear = W8A8Linear.from_float(
        float_linear,
        n_bits_w=8,
        n_bits_a=8,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_weight=True
    )

    # Weights should be quantized
    # Check that weights are different (quantized)
    weight_diff = (float_linear.weight - quant_linear.weight).abs().max()
    assert weight_diff > 0, "Weights should be quantized"

    # Test activation quantization (use float16 to match weight dtype)
    x = torch.randn(2, 16, 256, dtype=torch.float16)
    x_quant = quant_linear.quantize_activation(x)
    assert x_quant.shape == x.shape

    print("✓ Quantization functions work correctly")


if __name__ == "__main__":
    print("Running strategy pattern refactoring tests...\n")
    print("=" * 60)

    try:
        test_strategy_classes()
        test_w8a8linear_with_strategies()
        test_backward_compatibility_classes()
        test_from_float_factory()
        test_quantization_functions()

        print("\n" + "=" * 60)
        print("✅ All tests passed! Refactoring is successful.")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
