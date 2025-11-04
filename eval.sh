CUDA_VISIBLE_DEVICES=4 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 400 --num-calib-samples 2 --config-file ./quant/config/hq44k/rtn.yaml 
