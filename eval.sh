

CUDA_VISIBLE_DEVICES=$1 python small_engine.py \
    --mode eval  \
    --quantize-encoder \
    --quantize-decoder \
    --n-bits 4   \
    --num-samples 400 \
    --num-calib-samples 16 \
    --encoder_processor base 
    # --en-act-quant per_token \
