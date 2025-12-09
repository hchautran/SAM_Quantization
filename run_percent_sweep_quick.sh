#!/bin/bash
# Quick test sweep with fewer combinations for testing

export CUDA_VISIBLE_DEVICES=1

# Test with just 4 combinations (2x2 grid)
python benchmark_batch_inference.py \
    --sweep \
    --percent-values 0.6 0.8 \
    --percent-global-values 0.7 0.9 \
    --batch-sizes 1 4 8 \
    --num-samples 50 \
    --num-calib-samples 8 \
    --processor POSITIONAL_QUANT \
    --quantize-encoder \
    --n-bits 16 \
    --high-entropy \
    --prune-global \
    --wandb-project sam-quantization-sweep \
    --wandb-run-name "quick_test_$(date +%Y%m%d_%H%M%S)" \
    --output-dir ./benchmark_results \
    --summary-csv ./sweep_quick_test.csv
