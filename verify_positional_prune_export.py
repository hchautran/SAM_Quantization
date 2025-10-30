#!/usr/bin/env python3
"""
Verification script for PositionalPruneProcessor export status.

This script verifies that PositionalPruneProcessor is properly exported
and can be instantiated through various methods.
"""

import sys


def verify_import():
    """Verify the processor can be imported"""
    print("=" * 70)
    print("TEST 1: Direct Import")
    print("=" * 70)

    try:
        from processors import PositionalPruneProcessor
        print("✓ Successfully imported PositionalPruneProcessor")
        print(f"  Class: {PositionalPruneProcessor}")
        print(f"  Module: {PositionalPruneProcessor.__module__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def verify_registry():
    """Verify the processor is in the registry"""
    print("\n" + "=" * 70)
    print("TEST 2: Registry Lookup")
    print("=" * 70)

    try:
        from processors import ENCODER_PROCESSOR_REGISTRY

        if "POSITIONAL_PRUNE" in ENCODER_PROCESSOR_REGISTRY:
            print("✓ POSITIONAL_PRUNE found in registry")
            print(f"  Registry key: 'POSITIONAL_PRUNE'")
            print(f"  Class: {ENCODER_PROCESSOR_REGISTRY['POSITIONAL_PRUNE']}")
            return True
        else:
            print("✗ POSITIONAL_PRUNE not found in registry")
            print(f"  Available keys: {list(ENCODER_PROCESSOR_REGISTRY.keys())}")
            return False
    except Exception as e:
        print(f"✗ Error accessing registry: {e}")
        return False


def verify_get_processor():
    """Verify processor can be instantiated via get_encoder_processor"""
    print("\n" + "=" * 70)
    print("TEST 3: Get Encoder Processor")
    print("=" * 70)

    try:
        from processors import get_encoder_processor

        processor = get_encoder_processor("POSITIONAL_PRUNE")
        print("✓ Successfully created processor via get_encoder_processor")
        print(f"  Instance: {processor}")
        print(f"  Strategy name: {processor.strategy_name}")
        print(f"  Default percent: {processor.percent}")
        print(f"  Default threshold: {processor.threshold}")
        print(f"  Prune high entropy: {processor.prunehighentropy}")
        return True
    except Exception as e:
        print(f"✗ Failed to get processor: {e}")
        return False


def verify_direct_instantiation():
    """Verify processor can be instantiated directly"""
    print("\n" + "=" * 70)
    print("TEST 4: Direct Instantiation")
    print("=" * 70)

    try:
        from processors import PositionalPruneProcessor

        processor = PositionalPruneProcessor(strategy_name="test_prune")
        print("✓ Successfully created processor via direct instantiation")
        print(f"  Instance: {processor}")
        print(f"  Strategy name: {processor.strategy_name}")

        # Test attributes
        assert hasattr(processor, 'threshold'), "Missing attribute: threshold"
        assert hasattr(processor, 'percent'), "Missing attribute: percent"
        assert hasattr(processor, 'prunehighentropy'), "Missing attribute: prunehighentropy"
        assert hasattr(processor, 'entropy_stats'), "Missing attribute: entropy_stats"
        assert hasattr(processor, 'calibrate'), "Missing method: calibrate"
        assert hasattr(processor, 'process'), "Missing method: process"
        assert hasattr(processor, 'set_params'), "Missing method: set_params"

        print("✓ All required attributes and methods present")
        return True
    except Exception as e:
        print(f"✗ Failed direct instantiation: {e}")
        return False


def verify_head_prune_processor():
    """Verify HeadPruneProcessor is also exported"""
    print("\n" + "=" * 70)
    print("TEST 5: HeadPruneProcessor Export")
    print("=" * 70)

    try:
        from processors import HeadPruneProcessor, get_encoder_processor

        # Direct import
        print("✓ HeadPruneProcessor imported successfully")

        # Registry
        processor = get_encoder_processor("HEAD_PRUNE")
        print("✓ HEAD_PRUNE available in registry")
        print(f"  Strategy name: {processor.strategy_name}")

        return True
    except Exception as e:
        print(f"✗ Failed HeadPruneProcessor verification: {e}")
        return False


def list_all_processors():
    """List all available processors in the registry"""
    print("\n" + "=" * 70)
    print("Available Encoder Processors")
    print("=" * 70)

    try:
        from processors import ENCODER_PROCESSOR_REGISTRY

        for i, (name, cls) in enumerate(ENCODER_PROCESSOR_REGISTRY.items(), 1):
            marker = "★" if "PRUNE" in name else " "
            print(f"{marker} {i:2d}. {name:20s} -> {cls.__name__}")

        return True
    except Exception as e:
        print(f"✗ Error listing processors: {e}")
        return False


def main():
    """Run all verification tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PositionalPruneProcessor Export Verification" + " " * 9 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    tests = [
        ("Direct Import", verify_import),
        ("Registry Lookup", verify_registry),
        ("Get Encoder Processor", verify_get_processor),
        ("Direct Instantiation", verify_direct_instantiation),
        ("HeadPruneProcessor Export", verify_head_prune_processor),
        ("List All Processors", list_all_processors),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! PositionalPruneProcessor is fully exported.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
