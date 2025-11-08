"""
Minimal SM80-only SageAttention
Optimized for Ampere GPUs (SM80/SM86)
"""

from .core import sageattn_sm80
from .quant import per_block_int8, per_warp_int8, sub_mean

__version__ = "1.0.0-sm80"
__all__ = ["sageattn_sm80", "per_block_int8", "per_warp_int8", "sub_mean"]
