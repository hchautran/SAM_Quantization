CUDA_VISIBLE_DEVICES=4 CUDA_LAUNCH_BLOCKING=1 python small_engine_train_sam2.py \
        -c sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 0 \
        --num-gpus 1