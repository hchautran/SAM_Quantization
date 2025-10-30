# Attention Head Entropy Visualization

Visualize which attention heads have high and low entropy, and which are selected for pruning/quantization.

## Quick Start

### Generate heatmaps for all processors

```bash
# Use default percent_entropy=0.3
./plot_all_entropy_heatmaps.sh

# Or specify a custom value
./plot_all_entropy_heatmaps.sh 0.5
```

### Generate heatmap for a specific processor

```bash
python visualize_head_entropy.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --processor POSITIONAL_PRUNE \
    --percent-entropy 0.3 \
    --num-calib-samples 32
```

## What Gets Generated

For each processor, you get **2 plots**:

### 1. Entropy Heatmap (`*_entropy_heatmap_*.png`)

**Two subplots:**

- **Left**: Full entropy distribution across all layers and heads
  - Warmer colors (red/orange) = higher entropy
  - Cooler colors (yellow) = lower entropy
  - Rows = layers, Columns = attention heads

- **Right**: Pruned/quantized heads marked with "X"
  - Shows which heads are selected for modification
  - Blue X marks = pruned or quantized heads
  - Based on `percent_entropy` threshold

### 2. Statistics Plot (`*_entropy_heatmap_*_stats.png`)

**Four subplots:**

1. **Entropy Distribution Histogram**
   - Shows overall entropy distribution
   - Red line = mean, Green line = median

2. **Per-Layer Entropy Statistics**
   - Mean entropy ± std per layer
   - Identifies which layers have high/low entropy

3. **Pruning Ratio per Layer**
   - Horizontal bar chart
   - Green (<30%), Orange (30-60%), Red (>60%)
   - Shows how many heads are pruned in each layer

4. **Active vs Pruned Heads Comparison**
   - Box plot comparing entropy distributions
   - Shows if high-entropy or low-entropy heads are pruned

## Understanding the Results

### Processor Characteristics

**PositionalPruneProcessor / PositionalQuantProcessor:**
- Shows 16 heads per layer (averaged across 25 positions each)
- Fine-grained: Different positions within same head can have different entropy
- Heatmap shows averaged entropy per head

**HeadPruneProcessor:**
- Shows 16 heads per layer
- Coarse-grained: Entire heads are pruned
- Heatmap shows head-level entropy

### Interpreting Entropy Values

**High Entropy (red/orange):**
- Attention is more diffuse/spread out
- Head attends to many positions
- Generally indicates important, diverse attention patterns

**Low Entropy (yellow):**
- Attention is focused/concentrated
- Head attends to few positions
- May be redundant or less critical

### Which Heads Get Pruned?

With `high_entropy=True`:
- **High-entropy heads are pruned/quantized**
- Rationale: High-entropy attention can be approximated
- Low-entropy (focused) heads are kept at full precision

With `high_entropy=False`:
- **Low-entropy heads are pruned/quantized**
- Rationale: Focused attention is less critical
- Keep diverse attention patterns at full precision

## Examples

### Compare different percent_entropy values

```bash
# Generate for multiple values
for p in 0.1 0.3 0.5 0.7; do
    ./plot_all_entropy_heatmaps.sh $p
done
```

This creates multiple heatmaps showing how pruning intensity changes:
- `percent_entropy=0.1`: Very conservative, few heads pruned
- `percent_entropy=0.5`: Balanced, moderate pruning
- `percent_entropy=0.7`: Aggressive, many heads pruned

### Analyze a specific layer

After generating the heatmaps, look for patterns:

1. **Which layers have high entropy?**
   - Early layers (0-8) often have different patterns than late layers (16-31)
   - Middle layers sometimes have highest entropy

2. **Are pruning ratios uniform?**
   - Check the "Pruning Ratio per Layer" subplot
   - Some layers may have more pruned heads than others

3. **Does pruning match intuition?**
   - Compare the "Active vs Pruned Heads" box plot
   - Verify that high_entropy setting matches actual pruning

## Typical Workflow

### 1. Generate initial heatmaps

```bash
./plot_all_entropy_heatmaps.sh 0.3
```

### 2. Examine the results

```bash
# Open the plots
open entropy_plots/*_entropy_heatmap_p0.3.png
open entropy_plots/*_entropy_heatmap_p0.3_stats.png
```

### 3. Identify patterns

Look for:
- Which layers are most affected?
- Are high or low entropy heads pruned?
- What's the overall pruning ratio?

### 4. Adjust percent_entropy if needed

```bash
# Try different values based on what you observed
./plot_all_entropy_heatmaps.sh 0.5
```

### 5. Use insights for benchmarking

Now run the full benchmark with informed choices:

```bash
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --percent-entropy-values 0.2 0.3 0.4 \
    --num-samples 400
```

## Advanced Usage

### Change calibration samples

More samples = more accurate entropy statistics:

```bash
python visualize_head_entropy.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --processor POSITIONAL_PRUNE \
    --percent-entropy 0.3 \
    --num-calib-samples 64  # Double the default
```

### Custom output directory

```bash
python visualize_head_entropy.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --processor HEAD_PRUNE \
    --percent-entropy 0.5 \
    --output-dir ./my_entropy_analysis
```

## Output Files

Files are saved in `entropy_plots/` directory:

```
entropy_plots/
├── POSITIONAL_PRUNE_entropy_heatmap_p0.3.png
├── POSITIONAL_PRUNE_entropy_heatmap_p0.3_stats.png
├── POSITIONAL_QUANT_entropy_heatmap_p0.3.png
├── POSITIONAL_QUANT_entropy_heatmap_p0.3_stats.png
├── HEAD_PRUNE_entropy_heatmap_p0.3.png
└── HEAD_PRUNE_entropy_heatmap_p0.3_stats.png
```

## Troubleshooting

### "No entropy statistics found"

The processor must implement entropy collection. Make sure you're using:
- `POSITIONAL_PRUNE`
- `POSITIONAL_QUANT`
- `HEAD_PRUNE`

These processors collect entropy during calibration.

### Heatmap looks empty

- Increase `--num-calib-samples` (default: 32)
- Check that calibration is completing successfully
- Verify the config file has correct dataset paths

### Out of Memory

```bash
# Reduce calibration samples
python visualize_head_entropy.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --processor HEAD_PRUNE \
    --percent-entropy 0.3 \
    --num-calib-samples 16  # Reduced
```

## Tips

1. **Start with HEAD_PRUNE** - Easiest to visualize (16 heads only)
2. **Use default percent_entropy=0.3** first - Good starting point
3. **Compare all three processors** - Each shows different pruning strategies
4. **Look for layer patterns** - Some layers may be more amenable to pruning
5. **Cross-reference with mIoU** - Use benchmark results to validate choices

## Integration with Benchmarking

This visualization complements the benchmark suite:

1. **Run entropy visualization** → Understand which heads are pruned
2. **Run benchmark** → Measure accuracy and efficiency
3. **Correlate results** → Find optimal configurations

Example workflow:

```bash
# Step 1: Visualize entropy for key percent_entropy values
for p in 0.3 0.5 0.7; do
    ./plot_all_entropy_heatmaps.sh $p
done

# Step 2: Run benchmark
python benchmark_processors.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --percent-entropy-values 0.3 0.5 0.7 \
    --num-samples 400

# Step 3: Compare entropy patterns with accuracy results
python visualize_benchmark_results.py \
    --input benchmark_results/benchmark_results_*.csv \
    --output plots/
```

## Understanding Specific Patterns

### Pattern 1: Early layers have low entropy
- Early layers extract low-level features
- Focused attention is common
- May be safe to prune high-entropy heads here

### Pattern 2: Middle layers have high entropy
- Middle layers integrate information
- Diverse attention patterns
- Be careful pruning too much here

### Pattern 3: Late layers have variable entropy
- Final layers perform high-level reasoning
- Some heads are very focused, others very diffuse
- Good candidates for mixed-precision

## References

For more details on attention entropy and pruning strategies, see:
- Original SAM paper for attention mechanisms
- `processors/encoder/pruning.py` for implementation details
- `docs/BENCHMARK_GUIDE.md` for performance evaluation

---

**Happy Visualizing!** 🎨
