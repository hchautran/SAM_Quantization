# SAM2 Entropy Processors

This document describes the entropy-based attention head processors for SAM2 quantization, implemented in `entropy_sam2.py`.

## Overview

The SAM2 entropy processors are adapted versions of the original SAM entropy processors, specifically designed to work with SAM2's new architecture:

- **Hiera hierarchical backbone** instead of Vision Transformer (ViT)
- **MultiScaleAttention** with `F.scaled_dot_product_attention` instead of manual attention
- Different tensor shapes: `(B, num_heads, H*W, dim)` vs `(B*num_heads, H*W, dim)`
- Rotary position encoding (RoPE) instead of additive sine/cosine

## Available Processors

### 1. PositionalPruneSAM2Processor

Prunes attention heads based on mean entropy of their attention distribution.

**Strategy**:
- Computes a single mean entropy value for each head by treating the entire attention matrix as a flattened probability distribution
- Tracks entropy across calibration samples for each head position (batch × num_heads index)
- Selects heads with highest/lowest entropy for pruning
- Replaces pruned head outputs with mean of value vectors

**Usage**:
```python
from processors.encoder.entropy_sam2 import PositionalPruneSAM2Processor
from sam2.modeling.backbones.hieradet import MultiScaleAttention

# Initialize processor
processor = PositionalPruneSAM2Processor()

# Set parameters
processor.set_params(args)  # args should have quantization config

# Calibrate on sample data
processor.calibrate(predictor, MultiScaleAttention, num_samples=32)

# Get entropy statistics
entropy_stats = processor.get_entropy_stats()
```

**Key Parameters**:
- `percent_entropy`: Percentage of heads to prune per layer (0.0-1.0)
- `percent_entropy_global`: Global percentage for layers with many heads (0.0-1.0)
- `high_entropy`: If True, prune high-entropy heads; if False, prune low-entropy heads
- `prune_global`: If True, apply to all layers; if False, skip specific layers

### 2. HeadPruneSAM2Processor

Identifies and prunes attention heads based on per-position entropy.

**Strategy**:
- Calculates entropy for each position in a single attention head
- Averages entropy across batch and positions
- Selects heads with highest/lowest average entropy for pruning
- Creates boolean masks for head selection

**Usage**:
```python
from processors.encoder.entropy_sam2 import HeadPruneSAM2Processor

# Initialize processor
processor = HeadPruneSAM2Processor()

# Calibrate and process
processor.calibrate(predictor, MultiScaleAttention, num_samples=32)
```

**Key Parameters**:
- `percent`: Percentage of heads to select (0.0-1.0)
- `percent_global`: Global percentage for all layers
- `threshold`: Entropy threshold for head selection (if percent is None)
- `prunehighentropy`: Direction of pruning (high vs low entropy)

### 3. PositionalQuantSAM2Processor

Applies aggressive quantization to attention heads based on global entropy.

**Strategy**:
- Similar to PositionalPruneSAM2Processor but quantizes instead of pruning
- Computes global entropy for each head
- Selects heads with highest/lowest entropy
- Applies 2-bit or 4-bit quantization to Q, K, V, and attention matrices

**Usage**:
```python
from processors.encoder.entropy_sam2 import PositionalQuantSAM2Processor

# Initialize processor
processor = PositionalQuantSAM2Processor()

# Calibrate
processor.calibrate(predictor, MultiScaleAttention, num_samples=32)

# Process with quantization
output = processor.process(x, module, module_name)
```

**Key Features**:
- Uses per-token quantization for Q, K, and attention matrices
- Uses per-channel quantization for V
- Switches between 4-bit (default) and 2-bit (aggressive) quantization
- Merges quantized and non-quantized outputs

## Architecture Differences: SAM vs SAM2

### SAM (Original)
```python
# Attention computation
qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)  # Flat heads
attn = (q * module.scale) @ k.transpose(-2, -1)  # Manual
attn = attn.softmax(dim=-1)  # Explicit softmax
```

### SAM2
```python
# Attention computation
qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)  # (B, num_heads, H*W, dim) - separate batch dimension
# In actual forward pass (black box):
# x = F.scaled_dot_product_attention(q, k, v)  # Optimized CUDA kernel

# In our hooks (manual computation for entropy):
scale = (q.size(-1)) ** -0.5
attn = (q * scale) @ k.transpose(-2, -1)
attn = attn.softmax(dim=-1)
```

## Implementation Details

### Computing QKV for SAM2

```python
def _compute_qkv_sam2(self, x, module):
    """
    Compute Q, K, V from input tensor for SAM2 MultiScaleAttention.

    SAM2 uses shape (B, H, W, C) and produces (B, num_heads, H*W, dim).
    """
    B, H, W, C = x.shape

    # Project to QKV: (B, H, W, C) -> (B, H*W, 3*C)
    qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, C // module.num_heads)
    # Permute to (3, B, num_heads, H*W, dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    # Unbind to get separate Q, K, V: each (B, num_heads, H*W, dim)
    q, k, v = qkv.unbind(0)

    return q, k, v, B, H, W
```

### Attention Hook Pattern

```python
def _create_attention_hook(self, name):
    def attention_hook(module, input, output):
        x = input[0] if isinstance(input, tuple) else input
        B, H, W, C = x.shape

        # Compute QKV
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Compute attention manually (SDPA is black box)
        scale = (q.size(-1)) ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        # Calculate entropy and store
        for b in range(B):
            for head_idx in range(module.num_heads):
                attn_head = attn[b, head_idx]
                entropy = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{b * module.num_heads + head_idx}"
                self.entropy_stats[head_key].append(entropy)

    return attention_hook
```

### Processing with Pruning Mask

```python
def process(self, x: torch.Tensor, module, module_name: str = None):
    # Get pruning mask
    prune_mask = module.processor.final_entropy_stats.get(module_name, None)

    # Compute QKV
    q, k, v, B, H, W = self._compute_qkv_sam2(x, module)

    if prune_mask is not None:
        # Reshape to flat head dimension: (B, num_heads, ...) -> (B*num_heads, ...)
        q_flat = q.reshape(B * module.num_heads, H * W, -1)
        k_flat = k.reshape(B * module.num_heads, H * W, -1)
        v_flat = v.reshape(B * module.num_heads, H * W, -1)

        # Apply mask
        prune_mask = prune_mask.repeat(B)
        q_attn = q_flat[~prune_mask, :, :]
        k_attn = k_flat[~prune_mask, :, :]
        v_attn = v_flat[~prune_mask, :, :]
        v_pruned = v_flat[prune_mask, :, :]

        # Compute attention on non-pruned heads
        scale = (q_attn.size(-1)) ** -0.5
        attn = (q_attn * scale) @ k_attn.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x_attn = attn @ v_attn

        # Merge outputs
        x_out = torch.zeros_like(v_flat)
        x_out[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], -1)
        x_out[~prune_mask] = x_attn

        # Reshape back
        x_out = x_out.reshape(B, module.num_heads, H * W, -1)
    else:
        # Standard attention
        attn = self._compute_attention_sam2(q, k, module)
        x_out = attn @ v

    # Reshape to spatial and project
    x_out = self._reshape_output_sam2(x_out, B, module.num_heads, H, W)
    x_out = module.proj(x_out)
    return x_out
```

## Configuration Example

```yaml
quantization:
  # Entropy-based head selection
  percent_entropy: 0.5  # Prune 50% of heads per layer
  percent_entropy_global: 0.3  # Global percentage for large layers
  high_entropy: true  # Prune high-entropy heads (vs low-entropy)
  prune_global: true  # Apply to all layers (vs selective layers)

  # Quantization bits
  n_bits: 4  # Default quantization bits
  n_bits_aggressive: 2  # Aggressive quantization for selected heads
```

## Module Classes for Registration

When registering hooks, use these module classes:

```python
# SAM2 Image Encoder
from sam2.modeling.backbones.hieradet import MultiScaleAttention

# SAM2 Decoder (if needed)
from sam2.modeling.sam.transformer import Attention, RoPEAttention

# SAM2 Video (if needed)
from sam2.modeling.memory_attention import MemoryAttention
```

## Example: Complete Workflow

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.backbones.hieradet import MultiScaleAttention
from processors.encoder.entropy_sam2 import PositionalPruneSAM2Processor

# 1. Load SAM2 model
sam2_checkpoint = "path/to/sam2_checkpoint.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2)

# 2. Initialize processor
processor = PositionalPruneSAM2Processor()

# 3. Configure parameters
class Args:
    class Quantization:
        percent_entropy = 0.5
        percent_entropy_global = 0.3
        high_entropy = True
        prune_global = True
    quantization = Quantization()

processor.set_params(Args())

# 4. Calibrate on sample data
processor.calibrate(predictor, MultiScaleAttention, num_samples=32)

# 5. Get statistics
entropy_stats = processor.get_entropy_stats()
print(f"Processed {len(entropy_stats)} layers")

# 6. The processor is now ready to use in inference
# It will automatically apply pruning during forward passes
```

## Entropy Calculation Methods

### Mean Entropy (PositionalPrune/Quant)
```python
def calculate_entropy(self, attn_head):
    """Calculate mean entropy of the entire attention matrix."""
    eps = 1e-12
    attn_head = torch.clamp(attn_head, min=eps).flatten()
    entropy = -torch.sum(attn_head * torch.log(attn_head))
    return entropy
```

### Per-Position Entropy (HeadPrune)
```python
def calculate_entropy(self, attn_head):
    """Calculate entropy for each position in a single attention head."""
    eps = 1e-12
    attn_normalized = torch.clamp(attn_head, min=eps)
    entropy_per_position = -torch.mean(attn_normalized * torch.log(attn_normalized), dim=-1)
    return entropy_per_position
```

## Performance Considerations

1. **Calibration Time**: Entropy calculation requires forward passes on calibration data
   - Typical: 32-100 samples
   - Time: ~5-10 minutes depending on GPU

2. **Memory Usage**:
   - Hooks store entropy values for all heads across all samples
   - Memory grows linearly with num_samples × num_layers × num_heads

3. **Inference Impact**:
   - Pruning: Reduces computation by removing heads
   - Quantization: Reduces precision but maintains all heads
   - Trade-off: Accuracy vs Speed/Memory

## Troubleshooting

### Issue: "No module named 'sam2'"
**Solution**: Install SAM2:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### Issue: "CUDA out of memory during calibration"
**Solution**: Reduce num_samples or batch size:
```python
processor.calibrate(predictor, MultiScaleAttention, num_samples=16)  # Reduce from 32
```

### Issue: "Entropy values are all the same"
**Solution**: Check that hooks are being registered correctly:
```python
# Verify module names
for name, module in predictor.model.image_encoder.named_modules():
    if isinstance(module, MultiScaleAttention):
        print(f"Found attention module: {name}")
```

### Issue: "Process method fails with shape mismatch"
**Solution**: Ensure you're using SAM2 predictor, not SAM predictor:
```python
# Correct
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor(sam2)

# Incorrect
from segment_anything import SamPredictor  # This is for SAM, not SAM2
```

## References

- **SAM2 Paper**: [Segment Anything 2](https://arxiv.org/abs/2401.12741)
- **SAM2 Code**: [facebook/sam2](https://github.com/facebookresearch/sam2)
- **Original SAM Processors**: `processors/encoder/entropy.py`
- **Hiera Backbone**: `sam2/modeling/backbones/hieradet.py`

## License

Same license as the parent SAM_Quantization project.
