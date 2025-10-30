# SAM Encoder Batch Inference Benchmark

This benchmark measures SAM encoder performance when processing **multiple images simultaneously in a single batch**.

## What It Does

The script processes multiple images through the encoder in **one forward pass**, measuring:

- **Encoder batch time**: Total time to process all images in a batch
- **Encoder per-image time**: Batch time divided by batch size (shows efficiency gain)
- **Throughput**: Images processed per second
- **GPU Memory**: Peak memory usage at each batch size
- **Quality**: mIoU and boundary IoU maintained across batch sizes

## Key Insight

When batch_size increases, the **per-image encoder time decreases** because the GPU can parallelize computation:

```
Batch Size  | Per-Image Time | Memory
------------|----------------|--------
    1       |    400 ms      | 4.5 GB
    2       |    306 ms      | 7.9 GB  (24% faster per image!)
    4       |    ~200 ms     | ~12 GB  (50% faster per image!)
    8       |    ~120 ms     | ~20 GB  (70% faster per image!)
```

## Usage

### Basic Test
```bash
python benchmark_batch_inference.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --batch-sizes 1 2 4 8 16 \
    --num-samples 100 \
    --quantize-encoder \
    --n-bits 4
```

### Find Maximum Batch Size (Memory Stress Test)
```bash
python benchmark_batch_inference.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --batch-sizes 1 2 4 8 16 32 64 \
    --num-samples 50 \
    --quantize-encoder \
    --n-bits 4
```

### Compare Quantization Impact on Batching
```bash
# Baseline (16-bit, no quantization)
python benchmark_batch_inference.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --batch-sizes 1 4 8 16 \
    --num-samples 100 \
    --n-bits 16

# 4-bit quantized
python benchmark_batch_inference.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --batch-sizes 1 4 8 16 \
    --num-samples 100 \
    --quantize-encoder \
    --n-bits 4
```

### Run All Examples
```bash
./benchmark_batch_examples.sh
```

## Output

Results are saved as CSV with columns:
- `batch_size`: Number of images processed simultaneously
- `throughput_imgs_per_sec`: Total throughput
- `encoder_batch_mean_ms`: Mean time to process entire batch
- `encoder_per_image_mean_ms`: Mean time per image (shows efficiency)
- `peak_memory_allocated_mb`: Peak GPU memory
- `miou`: Mean IoU quality metric

## Interpreting Results

### Good Batching Behavior:
- Per-image encoder time **decreases** as batch size increases
- Throughput increases (though may plateau)
- Memory usage is acceptable

### Example:
```
Batch  | Throughput | Per-Image Time | Speedup
-------|------------|----------------|--------
  1    |   2.5 img/s|    400 ms      |  1.0x
  4    |   8.0 img/s|    125 ms      |  3.2x
  8    |  13.3 img/s|     75 ms      |  5.3x
 16    |  16.0 img/s|     62 ms      |  6.5x  (Best!)
 32    |  OOM       |     -          |  -
```

Optimal batch size: **16** (best speedup before OOM)

## Key Differences from Sequential Processing

**Your old code (small_engine.py:136)**:
```python
for data_val in dataloader:
    predictor.set_image(imgs.squeeze())  # Process ONE image at a time
```

**This benchmark**:
```python
# Process ENTIRE BATCH in single forward pass
features, interm_features = predictor.model.image_encoder(transformed_images)
```

## Files Created

1. **benchmark_batch_inference.py** - Main benchmark script
2. **benchmark_batch_examples.sh** - Pre-configured example commands
3. **BATCH_BENCHMARK_README.md** - This documentation

## What to Look For

1. **Optimal batch size**: Best throughput before running out of memory
2. **Quantization benefit**: Does 4-bit allow larger batches?
3. **Efficiency scaling**: How much does per-image time decrease?
4. **Memory vs Speed tradeoff**: Find the sweet spot for your GPU

## Example Command for Production Use

```bash
# Find optimal batch size for your setup
python benchmark_batch_inference.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --batch-sizes 1 2 4 8 16 32 \
    --num-samples 200 \
    --quantize-encoder \
    --n-bits 4 \
    --processor POSITIONAL_PRUNE \
    --output-dir ./batch_results
```

Then use the optimal batch size in production!
