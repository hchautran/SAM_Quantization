torchrun --nproc_per_node=4 sam-hq/train/train.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
    --model-type vit_b \
    --output work_dirs/hq_sam_b