#!/bin/bash
# Evaluate SAM3 on COCO val2017 — text-prompted open-vocabulary detection.
# Mirrors the SAM3 paper protocol (one text prompt per COCO category).

python eval_coco_sam3.py \
    --coco-root ./data/coco \
    --split val2017 \
    --resolution 1008 \
    --iou-types bbox  \
    --no-wandb
    # --num-images 10 \
    # --confidence-threshold 0.0 \
    # --top-k-per-cat 300 \
