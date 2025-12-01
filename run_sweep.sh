#!/bin/bash

# Sweep script for testing different percent and percent-global combinations
# Usage: ./run_sweep.sh

# Define sweep ranges
PERCENT_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5)
PERCENT_GLOBAL_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5)

# Fixed parameters
BATCH_SIZES="1 2 4 8"
NUM_SAMPLES=100
NUM_CALIB_SAMPLES=16
PROCESSOR="POSITIONAL_PRUNE"
N_BITS=4
EN_WEIGHT_QUANT="per_channel"
EN_ACT_QUANT="per_token"
MODEL_CKT="./ckts/sam_hq_vit_l.pth"
MODEL_TYPE="vit_l"
WANDB_PROJECT="sam-quantization-sweep"

# Create results directory
mkdir -p sweep_results

echo "Starting parameter sweep..."
echo "Percent values: ${PERCENT_VALUES[@]}"
echo "Percent-global values: ${PERCENT_GLOBAL_VALUES[@]}"
echo "Total runs: $((${#PERCENT_VALUES[@]} * ${#PERCENT_GLOBAL_VALUES[@]}))"
echo ""

# Counter for progress
total_runs=$((${#PERCENT_VALUES[@]} * ${#PERCENT_GLOBAL_VALUES[@]}))
current_run=0

# Loop through all combinations
for percent in "${PERCENT_VALUES[@]}"; do
    for percent_global in "${PERCENT_GLOBAL_VALUES[@]}"; do
        current_run=$((current_run + 1))
        echo "=================================================="
        echo "Run $current_run/$total_runs"
        echo "percent=$percent, percent_global=$percent_global"
        echo "=================================================="

        python benchmark_batch_inference.py \
            --percent $percent \
            --percent-global $percent_global \
            --batch-sizes $BATCH_SIZES \
            --num-samples $NUM_SAMPLES \
            --num-calib-samples $NUM_CALIB_SAMPLES \
            --processor $PROCESSOR \
            --quantize-encoder \
            --n-bits $N_BITS \
            --en-weight-quant $EN_WEIGHT_QUANT \
            --en-act-quant $EN_ACT_QUANT \
            --model-ckt $MODEL_CKT \
            --model-type $MODEL_TYPE \
            --wandb-project $WANDB_PROJECT \
            --output-dir sweep_results

        if [ $? -ne 0 ]; then
            echo "ERROR: Run failed for percent=$percent, percent_global=$percent_global"
            # Continue to next iteration instead of exiting
        fi

        echo ""
    done
done

echo "=================================================="
echo "Sweep complete! Results saved in sweep_results/"
echo "=================================================="
