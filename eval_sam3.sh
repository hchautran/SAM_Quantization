#!/usr/bin/env bash
# Evaluate SAM-3 on HQ44k, optionally with the sparsesam patch.
#
# Env knobs:
#   MODEL=facebook/sam3        — HF model id
#   ALGOS="none sparsesam"     — sweep algorithms
#   RATIOS="0.75 0.5 0.25"     — sparsesam keep-bar fractions
#   N=470                      — num samples
#   BATCH=4
#   DTYPE=fp16
#   TEXT="object"              — generic text prompt for SAM-3

set -e

MODEL=${MODEL:-facebook/sam3}
ALGOS=${ALGOS:-"none sparsesam"}
RATIOS=${RATIOS:-"0.75 0.5 0.25"}
N=${N:-470}
BATCH=${BATCH:-4}
DTYPE=${DTYPE:-fp16}
TEXT=${TEXT:-object}



python train_sam3_hq44k.py --epochs 8 --batch-size 4 --lr 1e-5

# python eval_sam3_hq44k.py \
#     --model        "$MODEL" \
#     --algorithms   $ALGOS \
#     --ratios       $RATIOS \
#     --num-samples  "$N" \
#     --batch-size   "$BATCH" \
#     --dtype        "$DTYPE" \
#     --text-prompt  "$TEXT"

# Examples:
#   ALGOS=none sh eval_sam3.sh                # baseline only
#   RATIOS="0.5" N=100 sh eval_sam3.sh        # quick check at ratio=0.5
#   MODEL=facebook/sam3-base sh eval_sam3.sh  # smaller model
