# SAM Quantization Scripts

This directory contains scripts for evaluating and analyzing density-based quantization for SAM-HQ.

## Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_density_sweep.py` | Comprehensive sweep over multiple percentages and density modes | Full evaluation |
| `quick_density_test.py` | Quick test of specific configuration | Fast testing |
| `analyze_density_results.py` | Analyze and visualize existing results | Post-processing |
| `run_density_sweep.sh` | Bash wrapper for easy execution | Convenience |

## Quick Start

### 1. Full Sweep (Recommended for Research)

Test all percentages with both low and high density modes:

```bash
# Using bash script
./scripts/run_density_sweep.sh 4  # Use GPU 4

# Or directly with python
CUDA_VISIBLE_DEVICES=4 python scripts/test_density_sweep.py
```

This will:
- Test percentages: 10%, 20%, 30%, ..., 100%
- Test both low and high density modes
- Generate plots and save results
- Take approximately 2-3 hours depending on dataset size

### 2. Quick Test (Single Configuration)

Test a specific configuration quickly:

```bash
# Test low density at 50%
python scripts/quick_density_test.py --density low --percent 50

# Test both densities at 60%
python scripts/quick_density_test.py --density both --percent 60
```

### 3. Analyze Results

After running sweeps, analyze and create visualizations:

```bash
python scripts/analyze_density_results.py \
    --results ./output/density_sweep/results.json \
    --output-dir ./output/analysis
```

## Detailed Usage

### test_density_sweep.py

Full parameter sweep testing:

```bash
python scripts/test_density_sweep.py \
    --config ./quant/config/hq44k/low_high.yaml \
    --percentages 10 20 30 40 50 60 70 80 90 100 \
    --density-modes low high \
    --output-dir ./output/density_sweep
```

**Parameters:**
- `--config`: Base configuration file
- `--percentages`: Space-separated list of percentages to test
- `--density-modes`: Which density modes to test (`low`, `high`, or both)
- `--output-dir`: Where to save results
- `--skip-existing`: Skip already completed configurations (for resuming)

**Output:**
```
output/density_sweep/
├── density_sweep_*.log              # Execution log
├── results.json                     # Raw data
├── val_iou_0_comparison.png        # IoU plot
├── val_boundary_iou_0_comparison.png  # Boundary IoU plot
└── combined_comparison.png          # All metrics combined
```

### quick_density_test.py

Quick single-configuration test:

```bash
python scripts/quick_density_test.py \
    --density both \
    --percent 50 \
    --config ./quant/config/hq44k/low_high.yaml \
    --output-dir ./output/quick_test
```

**Parameters:**
- `--density`: `low`, `high`, or `both`
- `--percent`: Single percentage value to test
- `--config`: Configuration file
- `--output-dir`: Output directory

**Use cases:**
- Quick validation
- Testing specific hypothesis
- Debugging configuration

### analyze_density_results.py

Post-processing and visualization:

```bash
python scripts/analyze_density_results.py \
    --results ./output/density_sweep/results.json \
    --output-dir ./output/analysis
```

**Parameters:**
- `--results`: Path to results.json file
- `--output-dir`: Where to save analysis outputs

**Output:**
```
output/analysis/
├── analysis_iou.png           # IoU comparison
├── analysis_boundary_iou.png  # Boundary IoU comparison
├── analysis_degradation.png   # Relative degradation plot
└── summary.txt                # Text summary table
```

## Common Workflows

### Workflow 1: Complete Evaluation

```bash
# 1. Run full sweep
./scripts/run_density_sweep.sh 4

# 2. Analyze results
python scripts/analyze_density_results.py \
    --results ./output/density_sweep/results.json

# 3. Check summary
cat ./output/density_sweep/summary.txt
```

### Workflow 2: Iterative Testing

```bash
# 1. Quick test to validate setup
python scripts/quick_density_test.py --percent 50

# 2. Run coarse sweep
python scripts/test_density_sweep.py --percentages 25 50 75 100

# 3. Run fine-grained sweep on interesting range
python scripts/test_density_sweep.py --percentages 45 50 55 60 65
```

### Workflow 3: Comparison Study

```bash
# 1. Run baseline (no quantization)
# (Configure low_high_density: "none" in config)

# 2. Run low density sweep
python scripts/test_density_sweep.py \
    --density-modes low \
    --output-dir ./output/low_density

# 3. Run high density sweep
python scripts/test_density_sweep.py \
    --density-modes high \
    --output-dir ./output/high_density

# 4. Compare results
python scripts/analyze_density_results.py --results ./output/low_density/results.json
python scripts/analyze_density_results.py --results ./output/high_density/results.json
```

## Configuration Files

The scripts use YAML configuration files. Example:

```yaml
# quant/config/hq44k/low_high.yaml
model:
  model_type: vit_l
  checkpoint: ./pretrained_checkpoint/sam_vit_l_0b3195.pth

quantization:
  low_high_density: low  # Overridden by script
  percent: 50            # Overridden by script
  n_bits: 4
  weight_quant: per_channel
  act_quant: per_token
```

## Understanding Results

### Metrics

- **val_iou_X**: IoU on validation dataset X
- **val_boundary_iou_X**: Boundary IoU on validation dataset X
- Lower number = first dataset, higher = later datasets

### Interpreting Plots

1. **IoU vs Percentage**:
   - Higher is better
   - Look for plateau regions (diminishing returns)
   - Compare low vs high density curves

2. **Degradation Plot**:
   - Shows % loss compared to 100% baseline
   - Lower is better (less degradation)
   - Helps identify sweet spot

### Finding Optimal Configuration

Look for configuration that:
1. Maintains IoU > threshold (e.g., 0.80)
2. Minimizes quantization percentage (saves computation)
3. Shows small boundary IoU degradation

Example findings:
```
Low Density Mode:
- 40%: IoU=0.815, minimal degradation
- 50%: IoU=0.820, good balance
- 60%: IoU=0.825, diminishing returns

High Density Mode:
- 60%: IoU=0.810, acceptable degradation
- 70%: IoU=0.815, reasonable trade-off
```

## Parallelization

Run different configurations in parallel:

```bash
# Terminal 1: Low density sweep on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/test_density_sweep.py \
    --density-modes low \
    --output-dir ./output/low_sweep &

# Terminal 2: High density sweep on GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/test_density_sweep.py \
    --density-modes high \
    --output-dir ./output/high_sweep &

# Wait for both to complete
wait
```

## Troubleshooting

### Out of Memory

Reduce batch size in config or use smaller evaluation dataset:

```yaml
# In config file
batch_size_valid: 1  # Reduce from default
```

### Slow Evaluation

Test on fewer datasets by modifying `hq44k_engine_quan.py`:

```python
# Line ~576
self.valid_datasets = [dataset_dis_val]  # Use only one dataset
```

### Missing Dependencies

Install required packages:

```bash
pip install matplotlib numpy torch omegaconf
```

## Tips

1. **Start Small**: Begin with `quick_density_test.py` to validate setup
2. **Use Logs**: Check log files for detailed progress and errors
3. **Resume Runs**: Use `--skip-existing` to resume interrupted sweeps
4. **Save Configs**: Keep track of which config generated which results
5. **Version Control**: Commit results and plots for reproducibility

## Examples

See `DENSITY_SWEEP_README.md` for more detailed examples and use cases.

## Citation

If you use these scripts in your research, please cite:

```bibtex
@article{sam-hq,
  title={Segment Anything in High Quality},
  author={...},
  journal={...},
  year={2023}
}
```
