#!/bin/bash
# Sweep token-merging patches on COCO val2017 with GT-box prompts.
#
# Pick the backbone with BACKBONE=sam-hq (default) or BACKBONE=sam3.
#
# Usage:
#   sh eval_coco.sh                        # SAM-HQ, full sweep
#   BACKBONE=sam3 sh eval_coco.sh          # SAM-3, sparsesam sweep
#   N=50 RATIOS="0.5" sh eval_coco.sh      # quick check
#
# Expected data layout:
#   ./data/coco/
#     ├── val2017/
#     └── annotations/instances_val2017.json
#
# Download:
#   mkdir -p data/coco && cd data/coco
#   wget http://images.cocodataset.org/zips/val2017.zip
#   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#   unzip val2017.zip && unzip annotations_trainval2017.zip

set -e

# BACKBONE=${BACKBONE:-sam-hq}
BACKBONE=${BACKBONE:-sam3}
N=${N:-200}
MAX_INST=${MAX_INST:-30}
RATIOS=${RATIOS:-"0.5 0.25"}

if [ "$BACKBONE" = "sam-hq" ]; then
    ALGOS=${ALGOS:-"none sparsesam sparsesam_random"}
    MODEL_CKT=${MODEL_CKT:-./ckts/sam_hq_vit_l.pth}
    MODEL_TYPE=${MODEL_TYPE:-vit_l}
    python eval_coco.py \
        --backbone      "$BACKBONE" \
        --coco-root     ./data/coco \
        --split         val2017 \
        --num-images    "$N" \
        --max-instances "$MAX_INST" \
        --ratios        $RATIOS \
        --algos         $ALGOS \
        --model-ckt     "$MODEL_CKT" \
        --model-type    "$MODEL_TYPE" \
        --ap \
        --no-wandb

elif [ "$BACKBONE" = "sam3" ]; then
    ALGOS=${ALGOS:-"none sparsesam"}
    SAM3_MODEL=${SAM3_MODEL:-facebook/sam3}
    DTYPE=${DTYPE:-fp16}
    TEXT=${TEXT:-object}
    # Add --use-category-text to feed each instance's COCO class as the prompt.
    python eval_coco.py \
        --backbone      "$BACKBONE" \
        --coco-root     ./data/coco \
        --split         val2017 \
        --num-images    "$N" \
        --max-instances "$MAX_INST" \
        --ratios        $RATIOS \
        --algos         $ALGOS \
        --sam3-model    "$SAM3_MODEL" \
        --dtype         "$DTYPE" \
        --text-prompt   "$TEXT" \
        --ap \
        --no-wandb

else
    echo "Unknown BACKBONE=$BACKBONE (expected: sam-hq | sam3)" >&2
    exit 1
fi
