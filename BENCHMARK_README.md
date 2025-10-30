# Processor Benchmark Suite

A comprehensive benchmarking toolkit for comparing SAM quantization processors.

## Quick Start

### Option 1: Using the Shell Script (Easiest)

```bash
# Quick test (~30-60 minutes)
./run_benchmark.sh quick

# Full benchmark (~2-4 hours)
./run_benchmark.sh full

# Custom configuration
./run_benchmark.sh custom --num-samples 200 --n-bits 6
```

### Option 2: Using Python Directly

```bash
# Run benchmark
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --num-samples 400 \
    --num-calib-samples 32 \
    --n-bits 4

# Visualize results
python visualize_benchmark_results.py \
    --input benchmark_results/benchmark_results_*.csv \
    --output plots/
```

## What Gets Tested

The benchmark compares **3 processors**:

1. **PositionalPruneProcessor** - Fine-grained pruning at positional level
2. **PositionalQuantProcessor** - Fine-grained quantization (2-bit for high-entropy)
3. **HeadPruneProcessor** - Coarse-grained head-level pruning

Across **8 percent_entropy values**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

**Total: 24 configurations**

## What Gets Measured

- ✅ **mIoU** (mean Intersection over Union) - Accuracy metric
- ✅ **Boundary IoU** - Edge detection accuracy
- ✅ **Theoretical GFLOPs** - Computational cost
- ✅ **Per-processor statistics** - Mean, std, min, max

## Output Files

### Benchmark Results
- `benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.csv` - Raw data

### Visualizations (in `plots/` directory)
- `miou_vs_percent_entropy.png` - Accuracy trends
- `gflops_vs_percent_entropy.png` - Computational cost
- `miou_vs_gflops_tradeoff.png` - Efficiency curve
- `boundary_iou_vs_percent_entropy.png` - Boundary accuracy
- `comparison_table.png` - Summary statistics
- `miou_heatmap.png` - Heat map visualization
- `benchmark_report.txt` - Text summary with key findings

## Example Workflow

```bash
# 1. Quick test to verify setup
./run_benchmark.sh quick

# 2. Review quick results
cat plots/quick_benchmark_*/benchmark_report.txt

# 3. Run full benchmark
./run_benchmark.sh full

# 4. Analyze results
# - Open plots/full_benchmark_*/miou_vs_gflops_tradeoff.png
# - Read plots/full_benchmark_*/benchmark_report.txt
# - Choose optimal configuration based on your accuracy/efficiency requirements
```

## Interpreting Results

### Look for:

1. **Best Accuracy**: Highest mIoU value
   - Typically at lower percent_entropy (0.1-0.3)

2. **Best Efficiency**: Lowest GFLOPs with acceptable accuracy
   - Check the trade-off curve
   - Find the "knee" of the curve

3. **Processor Comparison**:
   - **PositionalPrune**: Good accuracy retention
   - **PositionalQuant**: Better GFLOPs reduction
   - **HeadPrune**: Most aggressive compression

### Typical Sweet Spots:

- **percent_entropy = 0.2-0.3**: Often improves accuracy while reducing GFLOPs
- **percent_entropy = 0.4-0.5**: Good balance of accuracy and efficiency
- **percent_entropy = 0.6+**: For aggressive compression (edge devices)

## Requirements

```bash
pip install fvcore pandas matplotlib seaborn
```

## Files Structure

```
.
├── benchmark_processors.py          # Main benchmark script
├── visualize_benchmark_results.py   # Visualization script
├── run_benchmark.sh                 # Convenience wrapper
├── docs/BENCHMARK_GUIDE.md          # Detailed documentation
├── benchmark_results/               # Results CSVs (created)
└── plots/                           # Visualization outputs (created)
```

## Advanced Usage

See [docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) for:
- Custom percent_entropy ranges
- Testing specific processors
- Different quantization settings
- Decoder quantization
- Troubleshooting

## Expected Runtime

| Mode   | Samples | Calib | Configs | Time Estimate |
|--------|---------|-------|---------|---------------|
| Quick  | 50      | 16    | 24      | 30-60 min     |
| Full   | 400     | 32    | 24      | 2-4 hours     |

*Times vary based on hardware (GPU model, memory)*

## Troubleshooting

### Out of Memory
```bash
# Reduce sample counts
./run_benchmark.sh custom --num-samples 200 --num-calib-samples 16
```

### ImportError
```bash
# Verify PositionalQuantProcessor is exported
python -c "from processors import get_encoder_processor; print(get_encoder_processor('POSITIONAL_QUANT'))"
```

### fvcore missing
```bash
pip install fvcore
```

## Notes

- Results are saved with timestamps - you can run multiple benchmarks
- The script automatically cleans up GPU memory between runs
- All configuration parameters are saved in the results CSV
- Reproducible: rerun with same config to verify results

## Next Steps After Benchmarking

1. **Identify best configuration** from the trade-off curve
2. **Update your config YAML** with optimal percent_entropy
3. **Run full evaluation** on test set
4. **Profile actual latency** if deployment-critical
5. **Export to ONNX/TorchScript** for production

## Citation

If you use this benchmark suite in your research, please cite the SAM-HQ paper and acknowledge the quantization work.

---

**Happy Benchmarking!** 🚀

For detailed documentation, see [docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md)
