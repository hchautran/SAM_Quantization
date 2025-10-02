#!/usr/bin/env python3
"""
Small test script to verify the refactoring works correctly.
Tests the QuantizationConfig and replace_linear_with_quantized functions.
"""

import os
import sys
import torch

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from RTN_quantization.utils import QuantizationConfig, replace_linear_with_quantized
from omegaconf import OmegaConf


def test_quantization_config():
    """Test QuantizationConfig creation and methods."""
    print("="*80)
    print("TEST 1: QuantizationConfig Creation")
    print("="*80)

    # Test basic config
    config = QuantizationConfig(
        n_bits_w=4,
        n_bits_a=8,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False
    )

    print(f"✓ Created config: {config}")
    print(f"✓ n_bits_w: {config.n_bits_w}")
    print(f"✓ n_bits_a: {config.n_bits_a}")
    print(f"✓ weight_quant: {config.weight_quant}")
    print(f"✓ act_quant: {config.act_quant}")

    # Test get_w8a8linear_class
    print("\n" + "="*80)
    print("TEST 2: Automatic Class Selection")
    print("="*80)

    # Per-channel
    config1 = QuantizationConfig(act_quant="per_token")
    cls1 = config1.get_w8a8linear_class()
    print(f"✓ act_quant='per_token' -> {cls1.__name__}")

    # Per-tensor
    config2 = QuantizationConfig(
        weight_quant="per_tensor",
        act_quant="per_tensor"
    )
    cls2 = config2.get_w8a8linear_class()
    print(f"✓ weight_quant='per_tensor', act_quant='per_tensor' -> {cls2.__name__}")

    # Density-based
    config3 = QuantizationConfig(
        act_quant="low_high_density",
        quantizehigh=True,
        percent=50
    )
    cls3 = config3.get_w8a8linear_class()
    print(f"✓ act_quant='low_high_density' -> {cls3.__name__}")

    # Per-group
    config4 = QuantizationConfig(
        act_quant="per_group_token",
        group_size=128
    )
    cls4 = config4.get_w8a8linear_class()
    print(f"✓ act_quant='per_group_token' -> {cls4.__name__}")

    # Test to_kwargs
    print("\n" + "="*80)
    print("TEST 3: Config to kwargs conversion")
    print("="*80)

    kwargs = config.to_kwargs()
    print(f"✓ Config converted to kwargs: {len(kwargs)} parameters")
    for k, v in kwargs.items():
        print(f"  - {k}: {v}")

    return True


def test_replace_linear_with_quantized():
    """Test replace_linear_with_quantized function."""
    print("\n" + "="*80)
    print("TEST 4: Replace Linear with Quantized")
    print("="*80)

    # Create a simple test module
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(256, 512)
            self.linear2 = torch.nn.Linear(512, 256)
            self.linear3 = torch.nn.Linear(256, 128)

    model = SimpleModel()
    print(f"✓ Created test model with {sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))} linear layers")

    # Test quantization
    config = QuantizationConfig(
        n_bits_w=4,
        n_bits_a=8,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False
    )

    print(f"✓ Created quantization config")

    # Apply quantization
    replace_linear_with_quantized(model, config, module_name_to_exclude=[])

    # Check if layers are quantized
    from RTN_quantization.per_tensor_channel_group import W8A8Linear
    quantized_count = sum(1 for _ in model.modules() if isinstance(_, W8A8Linear))
    print(f"✓ Quantized {quantized_count} layers")

    # Verify layer types
    print(f"  - linear1: {type(model.linear1).__name__}")
    print(f"  - linear2: {type(model.linear2).__name__}")
    print(f"  - linear3: {type(model.linear3).__name__}")

    return quantized_count == 3


def test_density_based_config():
    """Test density-based quantization config."""
    print("\n" + "="*80)
    print("TEST 5: Density-Based Quantization Config")
    print("="*80)

    # Test low density
    config_low = QuantizationConfig(
        n_bits_w=4,
        n_bits_a=4,
        weight_quant="per_channel",
        act_quant="low_high_density",
        quantize_output=False,
        quantize_weight=True,
        quantizehigh=False,  # Low density
        percent=60
    )

    print(f"✓ Low density config created")
    print(f"  - quantizehigh: {config_low.quantizehigh}")
    print(f"  - percent: {config_low.percent}")

    cls_low = config_low.get_w8a8linear_class()
    print(f"  - Selected class: {cls_low.__name__}")

    # Test high density
    config_high = QuantizationConfig(
        n_bits_w=4,
        n_bits_a=4,
        weight_quant="per_channel",
        act_quant="low_high_density",
        quantize_output=False,
        quantize_weight=True,
        quantizehigh=True,  # High density
        percent=60
    )

    print(f"✓ High density config created")
    print(f"  - quantizehigh: {config_high.quantizehigh}")
    print(f"  - percent: {config_high.percent}")

    cls_high = config_high.get_w8a8linear_class()
    print(f"  - Selected class: {cls_high.__name__}")

    return cls_low == cls_high  # Should be same class, different params


def test_yaml_config_compatibility():
    """Test compatibility with existing YAML configs."""
    print("\n" + "="*80)
    print("TEST 6: YAML Config Compatibility")
    print("="*80)

    config_path = './quant/config/hq44k/low_high.yaml'

    if not os.path.exists(config_path):
        print(f"⚠ Config file not found: {config_path}")
        return True

    # Load YAML config
    yaml_config = OmegaConf.load(config_path)
    print(f"✓ Loaded YAML config from {config_path}")

    # Create QuantizationConfig from YAML
    quant_config = QuantizationConfig(
        n_bits_w=yaml_config.quantization.n_bits,
        n_bits_a=8,  # Default
        weight_quant=yaml_config.quantization.weight_quant,
        act_quant=yaml_config.quantization.act_quant,
        quantize_output=yaml_config.quantization.quantize_output,
        percent=yaml_config.quantization.get('percent', 100)
    )

    print(f"✓ Created QuantizationConfig from YAML")
    print(f"  - n_bits_w: {quant_config.n_bits_w}")
    print(f"  - weight_quant: {quant_config.weight_quant}")
    print(f"  - act_quant: {quant_config.act_quant}")
    print(f"  - percent: {quant_config.percent}")

    # Get appropriate class
    cls = quant_config.get_w8a8linear_class()
    print(f"✓ Selected class: {cls.__name__}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("REFACTORING VERIFICATION TESTS")
    print("="*80)
    print()

    tests = [
        ("QuantizationConfig Creation", test_quantization_config),
        ("Replace Linear with Quantized", test_replace_linear_with_quantized),
        ("Density-Based Config", test_density_based_config),
        ("YAML Config Compatibility", test_yaml_config_compatibility),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Refactoring is working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
