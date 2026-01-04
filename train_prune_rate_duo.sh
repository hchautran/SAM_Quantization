CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29515 small_engine_train_duo.py \
    --config-file quant/config/coco/rtn.yaml \
    --num-samples 400 \
    --checkpoint-evaluation "./pretrained_checkpoint/prune_rate/duo_sam_hq_epoch_torchnograd_distill10_vit_l_reg-weight_5_lr0.05_lr_drop2.pth" \
    # --train
