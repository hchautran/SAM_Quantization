CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29514 benchmark_batch_inference_coco.py \
    --config-file quant/config/coco/rtn.yaml \
    --quantize-encoder \
    --n-bits 16 \
    --num-calib-samples 16 \
    --processor POSITIONAL_QUANT \
    --detector yolox \
    # --num-samples 100 \