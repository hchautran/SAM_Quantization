#!/bin/bash
# Sweep all token-merging patches on COCO val2017 with GT-box prompts.
#
# Expected data layout:
#   ./data/coco/
#     ├── val2017/                          (images)
#     └── annotations/
#         └── instances_val2017.json
#
# Download:
#   mkdir -p data/coco && cd data/coco
#   wget http://images.cocodataset.org/zips/val2017.zip
#   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#   unzip val2017.zip && unzip annotations_trainval2017.zip

python eval_coco.py \
    --coco-root ./data/coco \
    --split val2017 \
    --algos none sparsesam sparsesam_random\
    --ratios 0.5 0.25 \
    --num-images 200 \
    --max-instances 30 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --ap \
    --no-wandb
