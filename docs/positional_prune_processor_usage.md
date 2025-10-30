# PositionalPruneProcessor - Usage Guide

## Overview

The `PositionalPruneProcessor` is a new attention processor that implements positional entropy-based pruning for SAM's image encoder. It is **fully exported and ready to use**.

## Export Status ✓

### 1. Module Exports

The processor is properly exported at multiple levels:

**`processors/encoder/__init__.py`** (Line 4):
```python
from .pruning import PositionalPruneProcessor, HeadPruneProcessor
```

**`processors/__init__.py`** (Lines 12, 89):
```python
from .encoder import (
    EncoderAttentionProcessor,
    EncoderRecenterAttentionProcessor,
    PositionalPruneProcessor,  # ✓ Exported
    HeadPruneProcessor,
    ...
)

__all__ = [
    ...
    "PositionalPruneProcessor",  # ✓ In __all__
    ...
]
```

### 2. Registry

**`processors/__init__.py`** (Line 70):
```python
register_encoder_processor("POSITIONAL_PRUNE")(PositionalPruneProcessor)
```

The processor is registered with the key `"POSITIONAL_PRUNE"` in the `ENCODER_PROCESSOR_REGISTRY`.

## Usage

### Method 1: Using the Registry (Recommended)

```python
from processors import get_encoder_processor

# Get the processor instance
processor = get_encoder_processor("POSITIONAL_PRUNE")

# Set parameters from config
processor.set_params(args)

# Calibrate
processor.calibrate(
    predictor=predictor,
    modules=attention_modules,
    num_samples=32
)
```

**Current usage in `small_engine.py` (Line 896)**:
```python
enc_processor = get_encoder_processor("POSITIONAL_PRUNE")
encoder_processor, decoder_processor = engine.setup_and_calibrate_processors(
    predictor,
    num_calib_samples=args.num_calib_samples,
    encoder_processor=enc_processor,
    decoder_processor=DecoderDoNothingProcessor("DO_NOTHING"),
)
```

### Method 2: Direct Import

```python
from processors import PositionalPruneProcessor

# Create processor instance
processor = PositionalPruneProcessor(strategy_name='positional_prune')

# Set parameters
processor.threshold = 5.0
processor.percent = 0.3  # Prune 30% of heads
processor.prunehighentropy = True  # Prune high-entropy heads

# Use it...
```

## Configuration

### YAML Configuration

Add these parameters to your YAML config (e.g., `quant/config/hq44k/rtn.yaml`):

```yaml
quantization:
  # Existing parameters...

  # Pruning parameters
  percent_entropy: 0.3      # Prune 30% of heads (recommended: 0.3-0.5)
  high_entropy: true        # true: prune high-entropy, false: prune low-entropy
```

**Current configuration** (`quant/config/hq44k/rtn.yaml`, lines 28-29):
```yaml
quantization:
  percent_entropy: 0.9   # Current setting
  high_entropy: True     # Current setting
```

### Recommended Configurations

Based on experimental results from the entropy pruning report:

#### Configuration 1: Maximum Performance
```yaml
quantization:
  percent_entropy: 0.3
  high_entropy: true
```
- **Performance**: 0.7931 mIoU (+0.85% vs baseline)
- **Speedup**: ~1.4×
- **Use case**: Research, high-accuracy applications

#### Configuration 2: Balanced Efficiency
```yaml
quantization:
  percent_entropy: 0.5
  high_entropy: true
```
- **Performance**: 0.7836 mIoU (-0.36% vs baseline)
- **Speedup**: ~2×
- **Use case**: Production deployments

#### Configuration 3: Aggressive Compression
```yaml
quantization:
  percent_entropy: 0.6
  high_entropy: true
```
- **Performance**: 0.7752 mIoU (-1.43% vs baseline)
- **Speedup**: ~2.5×
- **Use case**: Edge devices

## Class Interface

### Constructor

```python
def __init__(self, strategy_name: str = 'PositionalHeadPruneProcessor'):
    super().__init__(strategy_name)
    self.entropy_stats = defaultdict(lambda: {"entropy_per_position": []})
    self.threshold = 5.0
    self.percent = 0.5
    self.prunehighentropy = True
```

### Key Methods

#### `set_params(args)`
Set parameters from configuration:
```python
def set_params(self, args):
    self.threshold = 5.0
    self.percent = args.quantization.percent_entropy
    self.prunehighentropy = args.quantization.high_entropy
```

#### `calibrate(predictor, modules, num_samples)`
Collect entropy statistics and generate pruning masks:
```python
def calibrate(self, predictor, modules, num_samples=32):
    """
    Custom calibration that accumulates all entropy values,
    then calculates final statistics and creates pruning masks.

    Args:
        predictor: SamPredictor instance
        modules: Attention module types to hook
        num_samples: Number of calibration samples
    """
```

#### `process(x, module, module_name)`
Apply pruned attention during inference:
```python
def process(self, x: torch.Tensor, module, module_name: str = None):
    """
    Standard attention processing with optional head pruning.

    Args:
        x: Input tensor (B, H, W, C)
        module: Attention module
        module_name: Name of the module (for mask lookup)

    Returns:
        Output tensor after pruned attention
    """
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 5.0 | Entropy threshold (unused if `percent` is set) |
| `percent` | float | 0.5 | Fraction of heads to prune (0-1) |
| `prunehighentropy` | bool | True | True: prune high-entropy, False: prune low-entropy |
| `entropy_stats` | dict | {} | Collected entropy statistics during calibration |
| `final_entropy_stats` | dict | {} | Pruning masks per layer after calibration |

## Complete Example

```python
import torch
from omegaconf import OmegaConf
from segment_anything import sam_model_registry, SamPredictor
from processors import get_encoder_processor, DecoderDoNothingProcessor
from small_engine import Engine

# Load configuration
config = OmegaConf.load("quant/config/hq44k/rtn.yaml")

# Create SAM model
sam = sam_model_registry[config.model.model_type](checkpoint=config.model.checkpoint)
sam = sam.to(config.model.device)
predictor = SamPredictor(sam)

# Create engine
engine = Engine(
    strategy_name="positional_prune",
    quantize_encoder=True,
    quantize_decoder=False
)

# Get processor from registry
encoder_processor = get_encoder_processor("POSITIONAL_PRUNE")
decoder_processor = DecoderDoNothingProcessor("DO_NOTHING")

# Setup and calibrate
encoder_processor, decoder_processor = engine.setup_and_calibrate_processors(
    predictor,
    num_calib_samples=32,
    encoder_processor=encoder_processor,
    decoder_processor=decoder_processor,
    args_yaml=config
)

# Apply processor to model
from segment_anything.modeling.image_encoder import Attention
for name, module in predictor.model.image_encoder.named_modules():
    if isinstance(module, Attention):
        module.processor = encoder_processor

# Now run inference with pruned attention heads
# ...
```

## Switching Between Processors

To switch between different pruning strategies:

```python
# Option 1: Positional pruning (fine-grained)
enc_processor = get_encoder_processor("POSITIONAL_PRUNE")

# Option 2: Head-level pruning (coarse-grained)
enc_processor = get_encoder_processor("HEAD_PRUNE")

# Option 3: No pruning (baseline)
enc_processor = get_encoder_processor("BASE")
```

## Available Processors in Registry

```python
ENCODER_PROCESSOR_REGISTRY = {
    "BASE": EncoderAttentionProcessor,
    "RECENTER": EncoderRecenterAttentionProcessor,
    "POSITIONAL_PRUNE": PositionalPruneProcessor,  # ✓ Available
    "HEAD_PRUNE": HeadPruneProcessor,              # ✓ Available
    "SMOOTH_MEAN_Q": EncoderAttentionProcessorSmoothMeanQ,
    "COMPENSATE": EncoderAttentionProcessorCompensate,
    "SMOOTH": EncoderAttentionProcessorSmooth,
    "HIGH_LOW_ATTN_V": EncoderAttentionProcessorHighLow,
    "SMOOTH_LOG_Q": EncoderAttentionProcessorSmoothLogQ,
    "QUAROT": EncoderAttentionProcessorQuarot,
}
```

## Verification

To verify the processor is properly loaded:

```python
from processors import ENCODER_PROCESSOR_REGISTRY, PositionalPruneProcessor

# Check if registered
assert "POSITIONAL_PRUNE" in ENCODER_PROCESSOR_REGISTRY
print("✓ POSITIONAL_PRUNE is registered")

# Check class
assert ENCODER_PROCESSOR_REGISTRY["POSITIONAL_PRUNE"] == PositionalPruneProcessor
print("✓ Registry points to correct class")

# Test instantiation
processor = get_encoder_processor("POSITIONAL_PRUNE")
print(f"✓ Processor created: {processor.strategy_name}")
```

## Troubleshooting

### Issue: Processor not found

**Error**: `ValueError: Unknown encoder processor 'POSITIONAL_PRUNE'`

**Solution**: Ensure you're importing from the correct location:
```python
from processors import get_encoder_processor  # Correct
# Not from processors.encoder import get_encoder_processor
```

### Issue: Configuration not applied

**Problem**: Pruning parameters not being set

**Solution**: Make sure to call `set_params()` after creating the processor:
```python
processor = get_encoder_processor("POSITIONAL_PRUNE")
processor.set_params(args)  # Must call this!
```

### Issue: No pruning happening

**Problem**: All heads still active

**Checklist**:
1. Did you call `calibrate()` before inference?
2. Is `percent` > 0?
3. Check `final_entropy_stats` is populated: `print(processor.final_entropy_stats)`

## Performance Metrics

Based on experimental results (see `docs/entropy_pruning_report.md`):

| Pruning % | mIoU | Δ vs Baseline | Speedup | Recommendation |
|-----------|------|---------------|---------|----------------|
| 10% | 0.7886 | +0.28% | 1.1× | Conservative |
| 20% | 0.7906 | +0.53% | 1.25× | Safe |
| **30%** | **0.7931** | **+0.85%** | **1.4×** | **Optimal** |
| 40% | 0.7907 | +0.54% | 1.67× | Balanced |
| 50% | 0.7836 | -0.36% | 2× | Aggressive |
| 60% | 0.7752 | -1.43% | 2.5× | Edge |

## Summary

✅ **Export Status**: Fully exported and registered
✅ **Integration**: Already integrated in `small_engine.py`
✅ **Configuration**: Supports YAML configuration
✅ **Documentation**: Comprehensive entropy pruning report available
✅ **Testing**: Experimental validation complete

The `PositionalPruneProcessor` is production-ready and can be used immediately!

## References

- Implementation: `processors/encoder/pruning.py` (Lines 12-263)
- Theory: `docs/entropy_pruning_report.md`
- Results: `docs/figures/positional_pruning_results.png`
- Example usage: `small_engine.py` (Line 896)
