for model_size in vit_l vit_h; do
    torchrun --nproc_per_node=3 sam-hq/train/train.py \
        --checkpoint ./ckts/sam_${model_size}.pth \
        --model-type ${model_size} \
        --output work_dirs/hq_sam_${model_size}
done
