torchrun \
    --nproc_per_node=2 sam-hq/train/train.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
    --model-type vit_l \
    --output outputs \
    --eval \
    # --restore-model ./hq_ckts/sam_hq_vit_l.pth