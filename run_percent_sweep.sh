#!/bin/bash
# Example script to run parameter sweep with wandb logging

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run sweep with custom parameter ranges
python benchmark_batch_inference.py \
    --sweep \
    --percent-values 0.3 0.5 0.6 0.7 0.8 0.9 1.0 \
    --percent-global-values 0.3 0.5 0.6 0.7 0.8 0.9 1.0 \
    --batch-sizes 1 \
    --num-samples 3000 \
    --num-calib-samples 16 \
    --processor POSITIONAL_QUANT \
    --quantize-encoder \
    --n-bits 16 \
    --high-entropy \
    --prune-global \
    --wandb-project sam-quantization-TR \
    # --wandb-run-name "percent_sweep_$(date +%Y%m%d_%H%M%S)" \
    --output-dir ./benchmark_results \
    --summary-csv ./sweep_results_$(date +%Y%m%d_%H%M%S).csv

# To run without wandb logging, add --no-wandb flag:
# python benchmark_batch_inference.py \
#     --sweep \
#     --percent-values 0.6 0.7 0.8 \
#     --percent-global-values 0.7 0.8 0.9 \
#     --no-wandb \
#     --batch-sizes 1 2 4 8 \
#     --num-samples 50
