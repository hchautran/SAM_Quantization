"""
Attention processors for SAM quantization.

This module provides various attention processing strategies for quantizing
Segment Anything Model (SAM), particularly focused on the image encoder.
"""

from .base import AttentionProcessor, create_calib_dataloaders, setup_logger
from .encoder import (
    EncoderAttentionProcessor,
    EncoderRecenterAttentionProcessor,
    PositionalPruneProcessor,
    PruneRateProcessor,
    HeadPruneProcessor,
    PositionalQuantProcessor,
    EncoderAttentionProcessorSmoothMeanQ,
    EncoderAttentionProcessorCompensate,
    EncoderAttentionProcessorSmooth,
    EncoderAttentionProcessorHighLow,
    EncoderAttentionProcessorSmoothLogQ,
    EncoderAttentionProcessorQuarot,
)

# Decoder processors
from .decoder import (
    DecoderSignProcessor,
    DecoderDoNothingProcessor,
    DecoderChannelScaleProcessor,
    DecoderRecenterProcessor,
)

# Observer classes
from .decoder_observer import (
    TwoWayTransformerObserverElementLow,
    TwoWayAttentionBlockObserverElementLow,
    AttentionObserverElementLow,
    mask_decoder_monkey_patch,
)
from .encoder_observer import (
    AttentionObserver,
    BlockObserver,
    ImageEncoderViTObserver,
    AttentionObserver_compensate,
    BlockObserver_compensate,
    ImageEncoderViTObserver_compensate,
    image_encoder_monkey_patch,
)

# SAM2 observers
# from .sam2_observer import (
#     MultiScaleAttentionObserver,
#     sam2_image_encoder_monkey_patch,
# )

# Registry for encoder processors
ENCODER_PROCESSOR_REGISTRY = {}


def register_encoder_processor(name: str):
    """Decorator to register encoder attention processors."""
    def decorator(cls):
        ENCODER_PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_encoder_processor(name: str, **kwargs):
    """Get encoder processor by name."""
    if name not in ENCODER_PROCESSOR_REGISTRY:
        available = list(ENCODER_PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown encoder processor '{name}'. Available: {available}")
    return ENCODER_PROCESSOR_REGISTRY[name](**kwargs)


# Register all encoder processors
register_encoder_processor("BASE")(EncoderAttentionProcessor)
register_encoder_processor("RECENTER")(EncoderRecenterAttentionProcessor)
register_encoder_processor("POSITIONAL_PRUNE")(PositionalPruneProcessor)
register_encoder_processor("PRUNE_RATE")(PruneRateProcessor)
register_encoder_processor("HEAD_PRUNE")(HeadPruneProcessor)
register_encoder_processor("POSITIONAL_QUANT")(PositionalQuantProcessor)
register_encoder_processor("SMOOTH_MEAN_Q")(EncoderAttentionProcessorSmoothMeanQ)
register_encoder_processor("COMPENSATE")(EncoderAttentionProcessorCompensate)
register_encoder_processor("SMOOTH")(EncoderAttentionProcessorSmooth)
register_encoder_processor("HIGH_LOW_ATTN_V")(EncoderAttentionProcessorHighLow)
register_encoder_processor("SMOOTH_LOG_Q")(EncoderAttentionProcessorSmoothLogQ)
register_encoder_processor("QUAROT")(EncoderAttentionProcessorQuarot)


__all__ = [
    # Base classes and utilities
    "AttentionProcessor",
    "create_calib_dataloaders",
    "setup_logger",
    # Encoder processors
    "EncoderAttentionProcessor",
    "EncoderRecenterAttentionProcessor",
    "HeadPruneProcessor",
    "PositionalPruneProcessor",
    "PruneRateProcessor",
    "PositionalQuantProcessor",
    "EncoderAttentionProcessorSmoothMeanQ",
    "EncoderAttentionProcessorCompensate",
    "EncoderAttentionProcessorSmooth",
    "EncoderAttentionProcessorHighLow",
    "EncoderAttentionProcessorSmoothLogQ",
    "EncoderAttentionProcessorQuarot",
    # Decoder processors
    "DecoderSignProcessor",
    "DecoderDoNothingProcessor",
    "DecoderChannelScaleProcessor",
    "DecoderRecenterProcessor",
    # Decoder observers
    "TwoWayTransformerObserverElementLow",
    "TwoWayAttentionBlockObserverElementLow",
    "AttentionObserverElementLow",
    "mask_decoder_monkey_patch",
    # Encoder observers (SAM1)
    "AttentionObserver",
    "BlockObserver",
    "ImageEncoderViTObserver",
    "AttentionObserver_compensate",
    "BlockObserver_compensate",
    "ImageEncoderViTObserver_compensate",
    "image_encoder_monkey_patch",
    # SAM2 observers
    "MultiScaleAttentionObserver",
    "sam2_image_encoder_monkey_patch",
    # Registry functions
    "ENCODER_PROCESSOR_REGISTRY",
    "register_encoder_processor",
    "get_encoder_processor",
]
