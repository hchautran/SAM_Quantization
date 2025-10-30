# Processor Benchmarking Guide

This guide explains how to use the benchmark scripts to compare different SAM processors (PositionalPruneProcessor, PositionalQuantProcessor, and HeadPruneProcessor) across various `percent_entropy` values.

## Overview

The benchmark suite consists of two main scripts:

1. **`benchmark_processors.py`** - Runs the benchmark experiments
2. **`visualize_benchmark_results.py`** - Creates visualizations from results

## Prerequisites

```bash
# Install required packages
pip install fvcore pandas matplotlib seaborn
```

## Quick Start

### 1. Run Benchmark

```bash
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --num-samples 400 \
    --num-calib-samples 32 \
    --n-bits 4 \
    --en-weight-quant per_channel \
    --en-act-quant per_token
```

This will:
- Test all 3 processors: PositionalPruneProcessor, PositionalQuantProcessor, HeadPruneProcessor
- Sweep `percent_entropy` values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- Measure mIoU, boundary IoU, and theoretical GFLOPs
- Save results to `benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.csv`

**Expected runtime:** ~2-4 hours for 24 configurations (depending on hardware)

### 2. Visualize Results

```bash
python visualize_benchmark_results.py \
    --input benchmark_results/benchmark_results_20250129_123456.csv \
    --output plots/
```

This generates:
- `miou_vs_percent_entropy.png` - Accuracy comparison
- `gflops_vs_percent_entropy.png` - Computational cost comparison
- `miou_vs_gflops_tradeoff.png` - Efficiency trade-off curve
- `boundary_iou_vs_percent_entropy.png` - Boundary IoU comparison
- `comparison_table.png` - Summary statistics table
- `miou_heatmap.png` - Heatmap visualization
- `benchmark_report.txt` - Text report with key findings

## Advanced Usage

### Custom Percent Entropy Values

Test a custom range:

```bash
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --percent-entropy-values 0.1 0.3 0.5 0.7 \
    --num-samples 200
```

### Test Specific Processors

Benchmark only selected processors:

```bash
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --processors POSITIONAL_PRUNE POSITIONAL_QUANT \
    --num-samples 400
```

### Different Quantization Settings

```bash
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --n-bits 6 \
    --en-weight-quant selective_channel \
    --en-act-quant low_high_density_activation \
    --num-samples 400
```

### With Decoder Quantization

```bash
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --quantize-decoder \
    --de-weight-quant per_channel \
    --k-preserve 32 \
    --num-samples 400
```

## Understanding the Results

### CSV Output Columns

The benchmark results CSV contains:

- `processor`: Processor name (POSITIONAL_PRUNE, POSITIONAL_QUANT, HEAD_PRUNE)
- `percent_entropy`: Percentage of attention heads modified (0.1-0.8)
- `high_entropy`: Always True (modifies high-entropy heads)
- `val_iou_0`: Mean IoU on evaluation dataset
- `val_boundary_iou_0`: Boundary IoU metric
- `encoder_gflops`: Theoretical encoder GFLOPs
- `total_gflops`: Total model GFLOPs
- `n_bits`: Quantization bit-width used
- `weight_quant`, `act_quant`: Quantization methods
- `num_calib_samples`, `num_eval_samples`: Sample counts
- `timestamp`: Execution timestamp

### Interpreting the Plots

#### 1. mIoU vs Percent Entropy
- Shows accuracy trends as more heads are pruned/quantized
- Look for:
  - **Optimal range**: Where accuracy plateaus or improves
  - **Drop-off point**: Where accuracy degrades sharply
  - **Best processor**: Which maintains highest accuracy

#### 2. GFLOPs vs Percent Entropy
- Shows computational savings
- Lower GFLOPs = more efficient
- Compare theoretical speedup across processors

#### 3. mIoU vs GFLOPs Trade-off
- **Lower-right quadrant is better**: High accuracy, low cost
- Points are labeled with percent_entropy values
- Use this to select optimal operating point for your use case

#### 4. Comparison Table
- Quick summary statistics
- Compare mean, max, min values across processors

## Processor Characteristics

### PositionalPruneProcessor
- **Granularity**: Fine-grained (400 positions per layer)
- **Strategy**: Masks high-entropy positions with mean pooling
- **Trade-off**: Maintains accuracy well, moderate GFLOPs reduction

### PositionalQuantProcessor
- **Granularity**: Fine-grained (400 positions per layer)
- **Strategy**: 2-bit quantization on high-entropy positions
- **Trade-off**: Better GFLOPs reduction, may have slight accuracy drop

### HeadPruneProcessor
- **Granularity**: Coarse-grained (16 heads per layer)
- **Strategy**: Quantizes entire attention heads (4-bit for kept, 2-bit for pruned)
- **Trade-off**: Largest GFLOPs reduction, accuracy depends on head importance

## Typical Results

Based on experiments in the codebase, you can expect:

- **percent_entropy = 0.1-0.3**: Often **improves** accuracy while reducing GFLOPs
  - Sweet spot for most applications

- **percent_entropy = 0.4-0.5**: Maintains accuracy with significant GFLOPs reduction
  - Good balance point

- **percent_entropy = 0.6+**: Aggressive compression
  - For edge devices with strict constraints
  - May see accuracy degradation

## Tips

1. **Start with fewer samples** for quick testing:
   ```bash
   --num-samples 50 --num-calib-samples 16
   ```

2. **Use multiple runs** for statistical significance:
   - Run the same config 3-5 times
   - Average the results

3. **Monitor GPU memory**:
   - The script automatically clears CUDA cache between runs
   - If OOM occurs, reduce `num_calib_samples`

4. **Save configs**:
   - The results CSV includes all config parameters
   - Easy to reproduce best configurations

## Troubleshooting

### "fvcore not available"
```bash
pip install fvcore
```

### Out of Memory (OOM)
```bash
# Reduce calibration samples
--num-calib-samples 16

# Or reduce evaluation samples
--num-samples 200
```

### ImportError for processors
Make sure you've exported PositionalQuantProcessor:
```bash
grep "PositionalQuantProcessor" processors/__init__.py
```

### Results CSV is empty
- Check for errors in the terminal output
- Verify config file path is correct
- Ensure checkpoint file exists at `./pretrained_checkpoint/sam_hq_vit_l.pth`

## Example Workflow

Complete workflow from benchmark to analysis:

```bash
# 1. Run quick test (small sample)
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --num-samples 50 \
    --num-calib-samples 16 \
    --output-prefix quick_test

# 2. Review results
python visualize_benchmark_results.py \
    --input benchmark_results/quick_test_results_*.csv \
    --output plots/quick_test/

# 3. Run full benchmark
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --num-samples 400 \
    --num-calib-samples 32 \
    --output-prefix full_benchmark

# 4. Generate final plots
python visualize_benchmark_results.py \
    --input benchmark_results/full_benchmark_results_*.csv \
    --output plots/final/

# 5. Review plots/final/benchmark_report.txt
cat plots/final/benchmark_report.txt
```

## Citation

If you use these benchmarking scripts in your research, please cite the original SAM-HQ paper and include a reference to this quantization work.

## Contact

For issues or questions, please open an issue on the GitHub repository.
