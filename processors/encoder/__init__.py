"""Encoder processors for SAM quantization."""

from .basic import EncoderAttentionProcessor, EncoderRecenterAttentionProcessor
from .entropy import (PositionalPruneProcessor,   PruneRateDuoProcessor ,PruneRateProcessor, EntropyValueCheck, 
                      AttentionMapCollector, PositionalSparseProcessor,   PositionalSpargeAttnProcessor,PiecewiseAttnProcessor,
                      PositionalSparseFusedPosProcessor)
from .entropy3 import (Mvitv2PiecewiseAttnProcessor)

try:
    from .entropy2 import (PositionalSparseProcessorDiffDuo, PositionalSparseProcessorDuo, HeadPruneProcessor,
                            WholeSubImageProcessor, PositionalQuantProcessor)
except ModuleNotFoundError:
    PositionalSparseProcessorDiffDuo = None
    PositionalSparseProcessorDuo = None
    HeadPruneProcessor = None
    WholeSubImageProcessor = None
    PositionalQuantProcessor = None
from .entropy_sam2 import (
    PositionalPruneSAM2Processor,
    HeadPruneSAM2Processor,
    PositionalQuantSAM2Processor,
)
from .smooth import (
    EncoderAttentionProcessorSmoothMeanQ,
    EncoderAttentionProcessorSmooth,
    EncoderAttentionProcessorSmoothLogQ,
)
from .advanced import (
    EncoderAttentionProcessorCompensate,
    EncoderAttentionProcessorHighLow,
    EncoderAttentionProcessorQuarot,
)

__all__ = [
    "EncoderAttentionProcessor",
    "EncoderRecenterAttentionProcessor",
    "PositionalPruneProcessor",
    "PositionalSparseProcessor",
    "PositionalSparseFusedPosProcessor",
    "PositionalSparseProcessorDiffDuo",
    "PositionalSparseProcessorDuo",
    "PositionalSpargeAttnProcessor",
    "PiecewiseAttnProcessor",
    "HeadPruneProcessor",
    "WholeSubImageProcessor",
    "PositionalQuantProcessor",
    "EntropyValueCheck",
    "AttentionMapCollector",
    "PositionalPruneSAM2Processor",
    "HeadPruneSAM2Processor",
    "PositionalQuantSAM2Processor",
    "EncoderAttentionProcessorSmoothMeanQ",
    "EncoderAttentionProcessorCompensate",
    "EncoderAttentionProcessorSmooth",
    "EncoderAttentionProcessorHighLow",
    "EncoderAttentionProcessorSmoothLogQ",
    "EncoderAttentionProcessorQuarot",
    "PruneRateProcessor",
    "PruneRateDuoProcessor",
    "Mvitv2PiecewiseAttnProcessor",
]
