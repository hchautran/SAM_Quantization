CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29513 small_engine_train.py \
    --config-file quant/config/coco/rtn.yaml \
    --processor PRUNE_RATE \
    --num-calib-samples 16 \
    --train