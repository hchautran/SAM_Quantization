CUDA_VISIBLE_DEVICES=2 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 401 --num-calib-samples 8 --config-file ./quant/config/hq44k/rtn.yaml 
