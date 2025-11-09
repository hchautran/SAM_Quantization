# CUDA_VISIBLE_DEVICES=4 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 400 --num-calib-samples 2 --config-file ./quant/config/hq44k/rtn.yaml 

CUDA_VISIBLE_DEVICES=1 python benchmark_batch_inference.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --batch-sizes 1 2 4 8 16 32 64 \
    --num-samples 100 \
    --quantize-encoder \
    --n-bits 16 
