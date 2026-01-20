CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29510 small_engine_train_duo.py \
    --config-file quant/config/coco/rtn.yaml \
    --num-samples 501 \
    --checkpoint-evaluation "./pretrained_checkpoint/prune_rate/diffduo_sam_hq_epoch_torchnograd_distill10_vit_l_reg-weight_0.5_lr0.05_lr_drop2.pth" \
    --num-calib-samples 16\
    # --train
