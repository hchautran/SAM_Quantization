CUDA_VISIBLE_DEVICES=2 python benchmark_inference_dota.py \
    --config-file quant/config/coco/rtn.yaml \
    --quantize-encoder \
    --n-bits 16 \
    --processor BASE \
    --num-calib-samples 16 