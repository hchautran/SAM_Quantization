# SageAttention SM80 - Vision Model Optimized

This is a streamlined version of [SageAttention](https://github.com/thu-ml/SageAttention) optimized specifically for **vision models like SAM** running on **SM80 (Ampere) architecture** GPUs:
- NVIDIA A100
- NVIDIA A6000
- RTX 3090, RTX 3080 Ti, RTX A5000
- And other Ampere-based GPUs

## Features

✅ **SM80-only kernels** - No unnecessary SM89/SM90/SM120 code
✅ **Vision-optimized** - Removed LLM-specific features (causal masking, LSE)
✅ **Minimal dependencies** - Only essential CUDA kernels
✅ **Fast compilation** - Significantly faster build times
✅ **INT8 quantization** - Per-warp quantization for Q and K

## What's Removed

Compared to full SageAttention, this version removes:
- ❌ SM89/SM90/SM120 kernels (Ada/Hopper/Blackwell)
- ❌ FP8 kernels (SM89+ only)
- ❌ Causal masking (LLM-specific)
- ❌ LSE return (LLM-specific)
- ❌ Benchmark scripts
- ❌ Example inference scripts
- ❌ Triton kernels (CUDA only)

## Installation

### Requirements
- CUDA 12.0 or higher
- PyTorch with CUDA support
- SM80 or SM86 GPU

### Build from source

```bash
cd SageAttention_SM80
pip install -e .
```

Or with specific architecture:

```bash
TORCH_CUDA_ARCH_LIST="8.0" pip install -e .  # For A100/A6000
TORCH_CUDA_ARCH_LIST="8.6" pip install -e .  # For RTX 3090
```

## Usage

### Basic Usage for SAM

```python
import torch
from sageattention import sageattn_sm80

# Replace standard attention in SAM with quantized attention
# Typical SAM attention shapes: [B, H, N, D] where N = num_patches
B, H, N, D = 1, 8, 256, 64  # Example: 16x16 patches, 8 heads, 64 dim
q = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')

# Run quantized attention
output = sageattn_sm80(q, k, v, tensor_layout="HND")
```

### Integration Example

```python
# In your SAM model, replace:
# output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

# With:
from sageattention import sageattn_sm80
output = sageattn_sm80(q, k, v)
```

### Parameters

- `tensor_layout`: `"HND"` (default) or `"NHD"` - tensor layout format
- `smooth_k`: Subtract mean from K for better quantization (default: `True`)
- `smooth_v`: Subtract mean from V for better quantization (default: `False`)
- `qk_quant_gran`: `"per_warp"` (only supported mode)
- `sm_scale`: Softmax scale (default: `1/sqrt(head_dim)`)

## Performance

This minimal version compiles **2-3x faster** than the full SageAttention package while maintaining the same performance on SM80 GPUs.

## Quantization Details

The SM80 CUDA kernels use:
- **Q/K**: INT8 per-warp quantization (32 threads per warp)
- **V/O**: FP16 (no quantization)
- **Accumulation**: FP32 for numerical stability
- **Smoothing**: Optional mean subtraction for better quantization

This provides an excellent balance of speed and accuracy for vision models.

## License

Apache 2.0 - Same as original SageAttention

## Use Case: SAM (Segment Anything Model)

This package is specifically optimized for integrating into SAM and similar vision models:
- **No causal masking** - Not needed for bidirectional vision attention
- **No LSE output** - Simplified for standard vision tasks
- **Optimized for 2D spatial attention** - Common in vision transformers
- **Fast compilation** - Quick iteration during model development

## Credits

Based on [SageAttention](https://github.com/thu-ml/SageAttention) by the SageAttention team.
This is a vision-optimized minimal distribution focused on SM80 support for SAM integration.
