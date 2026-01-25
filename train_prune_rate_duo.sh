CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29512 small_engine_train_duo.py \
    --config-file quant/config/coco/rtn.yaml \
    --num-samples 501 \
    --checkpoint-evaluation "/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/pretrained_checkpoint/prune_rate/diffduo_sam_hq_epoch_torchnograd_distill_balance10_vit_l_reg-weight_0.5_lr0.02_lr_drop2.pth" \
    --num-calib-samples 16\
    --train
