"""
SAM2 encoder observer for monkey patching MultiScaleAttention with entropy processors.

This module provides observer wrappers for SAM2's Hiera backbone to integrate
entropy-based attention head pruning/quantization during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sam2.modeling.backbones.hieradet import MultiScaleAttention


class MultiScaleAttentionObserver(MultiScaleAttention):
    """
    Observer wrapper for SAM2's MultiScaleAttention that integrates entropy processors.

    This class wraps the original MultiScaleAttention and allows entropy-based
    processors to modify the attention computation (e.g., pruning heads, quantization).
    """

    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent MultiScaleAttention class."""
        super().__init__(*args, **kwargs)
        self.processor = None
        self.module_name = ""

    def set_processor(self, processor, module_name: str):
        """
        Set the entropy processor for this attention module.

        Args:
            processor: Entropy processor instance (e.g., PositionalPruneSAM2Processor)
            module_name: Full name of this module (e.g., "image_encoder.trunk.blocks.0.attn")
        """
        self.processor = processor
        self.module_name = module_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional processor intervention.

        If a processor is set and has been calibrated (has final_entropy_stats),
        it will process the attention computation (e.g., prune heads).
        Otherwise, falls back to standard attention.

        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Output tensor (B, H', W', C_out)
        """
        # If processor is set and calibrated, use processor's forward
        if self.processor is not None and hasattr(self.processor, 'final_entropy_stats'):
            if len(self.processor.final_entropy_stats) > 0:
                return self.processor.process(x, self, self.module_name)

        # Otherwise, use standard MultiScaleAttention forward
        return super().forward(x)


def sam2_image_encoder_monkey_patch(
    model,
    processor=None,
    verbose: bool = True,
):
    """
    Apply monkey-patching to SAM2 image encoder to integrate entropy processors.

    This function replaces all MultiScaleAttention modules in the SAM2 image encoder
    with MultiScaleAttentionObserver wrappers that can apply entropy-based processing.

    Args:
        model: SAM2 model to patch
        processor: Entropy processor instance (e.g., PositionalPruneSAM2Processor)
        verbose: Whether to print patching information

    Example:
        >>> from sam2.build_sam import build_sam2
        >>> from sam2.sam2_image_predictor import SAM2ImagePredictor
        >>> from processors.encoder.entropy_sam2 import PositionalPruneSAM2Processor
        >>> from processors.sam2_observer import sam2_image_encoder_monkey_patch
        >>>
        >>> # Build model
        >>> model = build_sam2("sam2_hiera_l.yaml", "checkpoints/sam2.pt")
        >>> predictor = SAM2ImagePredictor(model)
        >>>
        >>> # Create and calibrate processor
        >>> processor = PositionalPruneSAM2Processor()
        >>> processor.calibrate(predictor, MultiScaleAttention, num_samples=32)
        >>>
        >>> # Apply monkey patch
        >>> sam2_image_encoder_monkey_patch(model, processor)
        >>>
        >>> # Now the model uses pruned attention
        >>> predictor.set_image(image)
        >>> masks, _, _ = predictor.predict(...)
    """
    if processor is None:
        if verbose:
            print("Warning: No processor provided to monkey patch. Modules will use standard attention.")
        return

    patched_count = 0

    # Iterate through all modules in the image encoder
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, MultiScaleAttention):
            # Replace class with observer version
            module.__class__ = MultiScaleAttentionObserver

            # Set processor and module name
            full_name = f"image_encoder.{name}"
            module.set_processor(processor, full_name)

            patched_count += 1

            if verbose:
                print(f"  Patched: {full_name}")

    if verbose:
        print(f"\n✓ Monkey patched {patched_count} MultiScaleAttention modules")
        print(f"  Processor: {processor.strategy_name}")
        if hasattr(processor, 'final_entropy_stats'):
            num_layers_with_stats = len(processor.final_entropy_stats)
            print(f"  Layers with entropy stats: {num_layers_with_stats}")


__all__ = [
    'MultiScaleAttentionObserver',
    'sam2_image_encoder_monkey_patch',
]
