#!/bin/bash
# Generate entropy heatmaps for all three processors
# Usage: ./plot_all_entropy_heatmaps.sh [percent_entropy]

set -e

# Get percent_entropy from argument or use default
PERCENT_ENTROPY=${1:-0.3}
CONFIG_FILE="quant/config/hq44k/rtn.yaml"
NUM_CALIB_SAMPLES=32
OUTPUT_DIR="entropy_plots"

echo "========================================"
echo "Generating Entropy Heatmaps"
echo "percent_entropy = $PERCENT_ENTROPY"
echo "========================================"

# Array of processors
PROCESSORS=("POSITIONAL_PRUNE" "POSITIONAL_QUANT" "HEAD_PRUNE")

for processor in "${PROCESSORS[@]}"; do
    echo ""
    echo "Processing: $processor"
    echo "----------------------------------------"

    python visualize_head_entropy.py \
        --config-file "$CONFIG_FILE" \
        --processor "$processor" \
        --percent-entropy "$PERCENT_ENTROPY" \
        --num-calib-samples "$NUM_CALIB_SAMPLES" \
        --output-dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ $processor completed"
    else
        echo "✗ $processor failed"
    fi
done

echo ""
echo "========================================"
echo "All heatmaps generated!"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*p${PERCENT_ENTROPY}*.png
