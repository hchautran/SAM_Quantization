CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29514 small_engine_train.py \
    --config-file quant/config/coco/rtn.yaml \
    --processor PRUNE_RATE \
    --num-calib-samples 1 \
    --train
