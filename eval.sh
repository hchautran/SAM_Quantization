# CUDA_VISIBLE_DEVICES=4 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 400 --num-calib-samples 2 --config-file ./quant/config/hq44k/rtn.yaml 

# CUDA_VISIBLE_DEVICES=1 python benchmark_batch_inference.py \
#     --config-file quant/config/hq44k/rtn.yaml \
#     --batch-sizes 1 2 4 8 16 32 64 \
#     --num-samples 100 \
#     --quantize-encoder \
#     --n-bits 16 

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29512 benchmark_batch_inference_coco.py \
    --config-file quant/config/coco/rtn.yaml \
    --quantize-encoder \
    --n-bits 16 \
    --num-calib-samples 1 \
    --processor BASE \
    # --num-samples 100 \