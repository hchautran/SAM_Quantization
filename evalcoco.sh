CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29512 benchmark_batch_inference_coco.py \
    --config-file quant/config/coco/rtn.yaml \
    --quantize-encoder \
    --n-bits 16 \
    --num-calib-samples 16 \
    --processor POSITIONAL_PRUNE \
    --detector hdetr
    # --num-samples 100 \