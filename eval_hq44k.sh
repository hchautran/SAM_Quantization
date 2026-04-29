#!/bin/bash
# Sweep token-merging patches on HQ44k (DIS5K validation split).
#
# Pick the backbone with BACKBONE=sam2 (default) or BACKBONE=sam3.
#
# Usage:
#   sh eval_hq44k.sh                       # SAM-2, sparsesam sweep
#   BACKBONE=sam3 sh eval_hq44k.sh         # SAM-3, sparsesam sweep
#   N=100 RATIOS="0.5" sh eval_hq44k.sh    # quick check

set -e

BACKBONE=${BACKBONE:-sam2}
N=${N:-470}
DTYPE=${DTYPE:-fp16}
RATIOS=${RATIOS:-"0.75 0.5 0.25"}

if [ "$BACKBONE" = "sam2" ]; then
    ALGOS=${ALGOS:-"none sparsesam"}
    MODEL_CFG=${MODEL_CFG:-configs/sam2.1/sam2.1_hq_hiera_l.yaml}
    CKPT=${CKPT:-./ckts/sam2.1_hq_hiera_large.pt}
    BATCH=${BATCH:-8}
    python eval_hq44k.py \
        --backbone    "$BACKBONE" \
        --model-cfg   "$MODEL_CFG" \
        --checkpoint  "$CKPT" \
        --algorithms  $ALGOS \
        --ratios      $RATIOS \
        --num-samples "$N" \
        --batch-size  "$BATCH" \
        --dtype       "$DTYPE"
        # add --use-batch for SAM-2 native batch predict (requires batch>1)

elif [ "$BACKBONE" = "sam3" ]; then
    ALGOS=${ALGOS:-"none sparsesam"}
    SAM3_MODEL=${SAM3_MODEL:-facebook/sam3}
    BATCH=${BATCH:-1}
    TEXT=${TEXT:-object}
    CKPT_ARG=""
    if [ -n "$CKPT" ]; then
        CKPT_ARG="--checkpoint $CKPT"   # optional fine-tune from train_sam3_hq44k.py
    fi
    python eval_hq44k.py \
        --backbone    "$BACKBONE" \
        --sam3-model  "$SAM3_MODEL" \
        --algorithms  $ALGOS \
        --ratios      $RATIOS \
        --num-samples "$N" \
        --batch-size  "$BATCH" \
        --dtype       "$DTYPE" \
        --text-prompt "$TEXT" \
        $CKPT_ARG

else
    echo "Unknown BACKBONE=$BACKBONE (expected: sam2 | sam3)" >&2
    exit 1
fi
