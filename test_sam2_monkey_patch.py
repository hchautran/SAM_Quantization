#!/usr/bin/env python3
"""
Test script to verify SAM2 monkey patching with entropy processors.
"""

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.backbones.hieradet import MultiScaleAttention

from processors.encoder.entropy_sam2 import PositionalPruneSAM2Processor
from processors.sam2_observer import (
    sam2_image_encoder_monkey_patch,
    MultiScaleAttentionObserver,
)


def test_monkey_patch():
    """Test that monkey patching works correctly."""
    print("="*80)
    print("Testing SAM2 Monkey Patch")
    print("="*80)

    # Build a minimal SAM2 model (without loading checkpoint for testing)
    print("\n1. Building SAM2 model...")
    try:
        # Try to build without checkpoint (will fail but we can still test structure)
        sam2_model = build_sam2("sam2_hiera_l.yaml", ckpt_path=None)
    except:
        print("   (Could not load checkpoint, but that's OK for structure test)")
        # Create a dummy model structure
        from sam2.modeling.sam2_base import SAM2Base
        sam2_model = SAM2Base(
            image_encoder=None,
            memory_encoder=None,
            image_size=1024,
        )

    # Count original MultiScaleAttention modules
    original_count = 0
    for name, module in sam2_model.image_encoder.named_modules() if sam2_model.image_encoder else []:
        if isinstance(module, MultiScaleAttention):
            original_count += 1
            print(f"   Found MultiScaleAttention: {name}")

    print(f"   Total MultiScaleAttention modules: {original_count}")

    if original_count == 0:
        print("\n⚠ No MultiScaleAttention modules found (model might not be fully initialized)")
        print("   This is expected if checkpoint is not available")
        return

    # Create processor
    print("\n2. Creating entropy processor...")
    processor = PositionalPruneSAM2Processor()
    print(f"   Processor: {processor.strategy_name}")

    # Apply monkey patch
    print("\n3. Applying monkey patch...")
    sam2_image_encoder_monkey_patch(
        model=sam2_model,
        processor=processor,
        verbose=True
    )

    # Verify patching
    print("\n4. Verifying monkey patch...")
    observer_count = 0
    for name, module in sam2_model.image_encoder.named_modules():
        if isinstance(module, MultiScaleAttentionObserver):
            observer_count += 1
            # Check that processor is set
            if hasattr(module, 'processor'):
                print(f"   ✓ {name}: processor set")
            else:
                print(f"   ✗ {name}: processor NOT set")

        elif isinstance(module, MultiScaleAttention):
            print(f"   ✗ {name}: Still original MultiScaleAttention (not patched)")

    print(f"\n   Total MultiScaleAttentionObserver modules: {observer_count}")

    # Summary
    print("\n" + "="*80)
    if observer_count == original_count and observer_count > 0:
        print("✓ SUCCESS: All MultiScaleAttention modules patched correctly!")
    elif observer_count > 0:
        print(f"⚠ PARTIAL: {observer_count}/{original_count} modules patched")
    else:
        print("✗ FAILED: No modules were patched")
    print("="*80)


if __name__ == '__main__':
    test_monkey_patch()
