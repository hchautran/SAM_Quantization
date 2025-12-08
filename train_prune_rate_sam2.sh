CUDA_VISIBLE_DEVICES=4 python small_engine_train_sam2.py \
    -c sam2.1_training/sam2.1_hiera_b+_MOSE_finetune \
    --use-cluster 0 \
    --num-gpus 1